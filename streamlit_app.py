import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import plotly.graph_objects as go
from sklearn.covariance import LedoitWolf
import tushare as ts
import warnings
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time

# 1. 数据获取与预处理

def fetch_and_preprocess_data(ETF, asset,start, end):
    """
    获取数据并预处理

    参数:
    - ETF: dict, ETF代码和名称的映射。
    - start: str, 开始日期（格式：'YYYYMMDD'）。
    - end: str, 结束日期（格式：'YYYYMMDD'）。

    返回:
    - returns: DataFrame, 各资产的收益率数据（行为时间，列为资产）。
    """   
    df = pd.DataFrame(columns=['trade_date', 'pct_chg', 'name'])
    for name, code in ETF.items():
        temp = ts.pro_bar(ts_code=code, asset=asset,start_date=start, end_date=end)[['trade_date', 'pct_chg']]
        temp['name'] = name
        df = pd.concat([df, temp])
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df = df.pivot(index='trade_date', columns='name', values='pct_chg')
    df = df.fillna(0)
    return df


# 2. 优化函数（支持滚动与不滚动）

def optimize_portfolio(returns, risk_free_rate, alpha=1, beta=0, rolling_mode=True, window_size=252, step_size=63):
    import numpy as np
    from scipy.optimize import minimize
    from sklearn.covariance import LedoitWolf
    import pandas as pd

    num_periods = len(returns)
    num_assets = returns.shape[1]
    asset_names = returns.columns
    
    weights_history = pd.DataFrame(index=returns.index, columns=asset_names)
    portfolio_returns = pd.Series(index=returns.index, dtype=float)
    
    # 新增：记录组合波动率和资产风险贡献
    volatility_history = pd.Series(index=returns.index, dtype=float)
    risk_contributions_history = pd.DataFrame(index=returns.index, columns=asset_names)
    
    def risk_contribution(weights, cov_matrix):
        portfolio_var = weights.T @ cov_matrix @ weights
        marginal_contribution = cov_matrix @ weights
        return weights * marginal_contribution / portfolio_var
    
    def combined_objective(weights, mean_returns, cov_matrix, risk_free_rate, alpha, beta):
        portfolio_return = np.sum(mean_returns * weights)
        portfolio_volatility = np.sqrt(weights.T @ cov_matrix @ weights)
        sharpe = (portfolio_return - risk_free_rate) / portfolio_volatility
        risk_contrib = risk_contribution(weights, cov_matrix)
        risk_diff = np.sum((risk_contrib - np.mean(risk_contrib)) ** 2)
        return - alpha * sharpe + beta * risk_diff
    
    def get_constraints_and_bounds(num_assets):
        constraints = [
            {'type': 'ineq', 'fun': lambda x: np.sum(x) - 0.9},  # 权重和至少为0.9
            {'type': 'ineq', 'fun': lambda x: 1 - np.sum(x)},  # 权重和最多为1
            {'type': 'ineq', 'fun': lambda x: x - 0.01},  # 每个权重最低为0.01
            {'type': 'ineq', 'fun': lambda x: 0.8 - x},  # 每个权重上限为0.8
        ]
        
        bounds = tuple((0.01, 0.8) for _ in range(num_assets))  # 每个权重在0.01到0.8之间
        return constraints, bounds
    
    def optimize(mean_returns, cov_matrix, risk_free_rate, alpha, beta):
        num_assets = len(mean_returns)
        initial_weights = np.ones(num_assets) / num_assets
        constraints, bounds = get_constraints_and_bounds(num_assets)
        result = minimize(
            combined_objective, initial_weights,
            args=(mean_returns, cov_matrix, risk_free_rate, alpha, beta),
            method='SLSQP', bounds=bounds, constraints=constraints
        )
        if result.success:
            return result.x
        else:
            print("Optimization failed. Falling back to equal weights.")
            return initial_weights
    
    if rolling_mode:
        for start_idx in range(0, num_periods - window_size, step_size):
            end_idx = start_idx + window_size
            window_returns = returns.iloc[start_idx:end_idx]
            mean_returns = window_returns.mean()
            cov_matrix = LedoitWolf().fit(window_returns).covariance_
            optimal_weights = optimize(mean_returns, cov_matrix, risk_free_rate, alpha, beta)
            apply_start = end_idx
            apply_end = min(end_idx + step_size, num_periods)
            weights_history.iloc[apply_start:apply_end] = np.tile(optimal_weights, (apply_end - apply_start, 1))
            portfolio_returns.iloc[apply_start:apply_end] = returns.iloc[apply_start:apply_end] @ optimal_weights
            
            # 计算组合波动率和风险贡献
            portfolio_volatility = np.sqrt(optimal_weights.T @ cov_matrix @ optimal_weights)
            volatility_history.iloc[apply_start:apply_end] = portfolio_volatility
            rc = risk_contribution(optimal_weights, cov_matrix)
            risk_contributions_history.iloc[apply_start:apply_end] = np.tile(rc, (apply_end - apply_start, 1))
    else:
        mean_returns = returns.mean()
        cov_matrix = LedoitWolf().fit(returns).covariance_
        optimal_weights = optimize(mean_returns, cov_matrix, risk_free_rate, alpha, beta)
        weights_history[:] = optimal_weights
        portfolio_returns[:] = returns @ optimal_weights
        
        # 计算组合波动率和风险贡献
        portfolio_volatility = np.sqrt(optimal_weights.T @ cov_matrix @ optimal_weights)
        volatility_history[:] = portfolio_volatility
        rc = risk_contribution(optimal_weights, cov_matrix)
        risk_contributions_history[:] = rc
    
    return weights_history, portfolio_returns, volatility_history, risk_contributions_history

# 3. 可视化

def visualize_results(title,benchmark,weights_history, portfolio_returns, risk_free_rate, returns, volatility_history, risk_contributions_history, rolling_mode=True, window_size=252, volatility_threshold=0.05, height=600, important_dates=None,y_offset=1):
    """
    可视化投资组合表现

    参数:
    - benchmark: str, 基准资产名称。
    - weights_history: DataFrame, 各资产权重历史。
    - portfolio_returns: Series, 组合收益率历史。
    - risk_free_rate: float, 无风险利率。
    - returns: DataFrame, 各资产的收益率数据。
    - volatility_history: Series, 组合波动率历史。
    - risk_contributions_history: DataFrame, 各资产风险贡献历史。
    - rolling_mode: bool, 是否使用滚动窗口优化。
    - window_size: int, 滚动窗口大小。
    - volatility_threshold: float, 波动率阈值。
    - height: int, 图表高度。
    - important_dates: list, 重要日期列表。
    - y_offset: float, 标注的垂直偏移量。
    """

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Define risk contribution function
    def risk_contribution(weights, cov_matrix):
        portfolio_var = weights.T @ cov_matrix @ weights
        marginal_contribution = cov_matrix @ weights
        
        return weights * marginal_contribution / portfolio_var
    
    if rolling_mode:
        rolling_returns = portfolio_returns.rolling(window=1).mean()
        cumulative_returns = (1 + portfolio_returns).cumprod() - 1

        def calculate_risk_contributions(weights_history, returns, window_size):
            risk_contributions = pd.DataFrame(index=weights_history.index, columns=weights_history.columns)
            for date, weights in weights_history.iterrows():
                window_returns = returns.loc[:date].iloc[-window_size:]
                cov_matrix = LedoitWolf().fit(window_returns).covariance_
                risk_contributions.loc[date] = risk_contribution(weights, cov_matrix)
            return risk_contributions

        risk_contributions = calculate_risk_contributions(weights_history, returns, window_size)

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=("<b>资产权重及组合收益</b>", "<b>资产风险贡献</b>", "<b>Portfolio Volatility</b>"),
            specs=[
                [{"secondary_y": True}],  # 第一行，第一列，启用第二个 y 轴
                [{"secondary_y": True}],  # 第二行，第一列
            ]
            )

        # Custom color palette
        colors = px.colors.qualitative.Dark24
        #colors = px.colors.qualitative.Set1

        # 添加资产权重变化
        for i, asset in enumerate(weights_history.columns):
            color = colors[i % len(colors)]
            rgba_color = f'rgba{tuple(int(colors[i % len(colors)].lstrip("#")[j:j+2], 16) for j in (0, 2, 4)) + (0.3,)}'
            fig.add_trace(go.Scatter(
                x=weights_history.index, y=weights_history[asset],
                mode='lines',
                name=asset,
                stackgroup='one',
                line=dict(color=rgba_color,width=1),
                fill='tonexty',  # 填充区域
                fillcolor=rgba_color,
                hovertemplate=(
                    "<b>Date</b>: %{x|%Y-%m-%d}<br>"
                    "<b>Asset</b>: " + asset + "<br>"
                    "<b>Weight</b>: %{y:.2%}<br>"
                    "<extra></extra>"
                )
            ), row=1, col=1)

        # 添加日收益曲线
        fig.add_trace(
            go.Scatter(
                x=rolling_returns.index,
                y=rolling_returns*10,
                mode='lines',
                name='Daily Returns',
                visible=True,
                line=dict(color='blue', width=2, dash='dash'),
                hovertemplate=(
                    "<b>Date</b>: %{x|%Y-%m-%d}<br>"
                    "<b>Rolling Return</b>: %{customdata:.2%}<br>"
                    "<extra></extra>"
                ),
                customdata=rolling_returns,
            ),
            row=1, col=1
        )

    
        # 添加基准收益曲线
        if benchmark == '等权':
            num_assets = returns.shape[1]
            equal_weights = np.ones(num_assets) / num_assets  # 等权组合权重
            equal_weight_returns = returns.iloc[window_size:] @ equal_weights  # 等权组合每日收益率
            equal_weight_cumulative = (1 + equal_weight_returns).cumprod() - 1  # 等权组合累计收益率
            fig.add_trace(
            go.Scatter(
                x=equal_weight_cumulative.index,
                y=equal_weight_cumulative,
                mode='lines',
                name='比较基准-'+benchmark,
                visible=False,
                line=dict(color='black', width=1.5, dash='dash'),
                hovertemplate=(
                    "<b>Date</b>: %{x|%Y-%m-%d}<br>"
                    "<b>Benchmark Return</b>: %{y:.2%}<br>"
                    "<extra></extra>"
                )
            ),
            row=1, col=1,
            secondary_y=True  # 绑定到第二个 y 轴
        )
        else:
            benchmark_returns=(1 + returns[benchmark]).cumprod() - 1
            fig.add_trace(
                go.Scatter(
                    x=benchmark_returns.index,
                    y=benchmark_returns,
                    mode='lines',
                    name='比较基准-'+benchmark,
                    visible=False,
                    line=dict(color='black', width=1.5, dash='dash'),
                    hovertemplate=(
                        "<b>Date</b>: %{x|%Y-%m-%d}<br>"
                        "<b>Benchmark Return</b>: %{y:.2%}<br>"
                        "<extra></extra>"
                    )
                ),
                row=1, col=1,
                secondary_y=True  # 绑定到第二个 y 轴
            )

        # 添加累计收益曲线
        fig.add_trace(
            go.Scatter(
                x=cumulative_returns.index,
                y=cumulative_returns,
                mode='lines',
                name='累计收益',
                visible=False,
                line=dict(color='green', width=2),
                hovertemplate=(
                    "<b>Date</b>: %{x|%Y-%m-%d}<br>"
                    "<b>Cumulative Return</b>: %{y:.2%}<br>"
                    "<extra></extra>"
                )
            ),
            row=1, col=1,
            secondary_y=True  # 绑定到第二个 y 轴
        )


        # 绘制组合波动率
        fig.add_trace(go.Scatter(
            x=volatility_history.index,
            y=volatility_history*np.sqrt(252),
            marker=dict(size=5, color='rgba(0,0,0,0.2)', symbol='circle', line=dict(color='white', width=0.5)),
            fill='tonexty',  # 填充到下一个y值，但不填到 x 轴
            fillcolor='rgba(229,245,249,0.2)',  # 半透明区域填充
            mode='lines',
            name='组合波动率',
            #line=dict(color='red', width=2),
            hovertemplate="<b>Date</b>: %{x|%Y-%m-%d}<br><b>Volatility</b>: %{y:.2%}<extra></extra>"), 
            row=2, col=1,
            secondary_y=True,
                      
            )
        

        # 添加风险贡献曲线
        for i, asset in enumerate(risk_contributions.columns):
            rgba_color = f'rgba{tuple(int(colors[i % len(colors)].lstrip("#")[j:j+2], 16) for j in (0, 2, 4)) + (0.9,)}'
            fig.add_trace(
                go.Scatter(
                    x=risk_contributions.index,
                    y=risk_contributions[asset],
                    mode='lines',
                    name=f'{asset} Risk Contribution',
                    line=dict(color=rgba_color, width=1),
                    hovertemplate=(
                        "<b>Date</b>: %{x|%Y-%m-%d}<br>"
                        "<b>Asset</b>: " + asset + "<br>"
                        "<b>Risk Contribution</b>: %{y:.2%}<br>"
                        "<extra></extra>"
                    ),
                    showlegend=False
                ),
                row=2, col=1
            )
        


        if important_dates is None:
            important_dates = []
        
        y_offset = y_offset
        for idx, date_info in enumerate(important_dates):
            if 'date' not in date_info or 'event' not in date_info:
                print(f"Skipping invalid date_info: {date_info}")
                continue
            date = date_info['date']
            if date not in weights_history.index:
                print(f"Date {date} not in data range. Skipping.")
                continue
            event = date_info['event']
            fig.add_shape(
                type="line",
                x0=date,
                x1=date,
                y0=0,
                y1=1,
                xref="x",
                yref="paper",
                line=dict(
                    color="Red",
                    width=1,
                    dash="dot"
                )
            )
            fig.add_annotation(
                x=date,
                y=y_offset + 0.01 * idx,
                xref="x",
                yref="paper",
                text=event,
                showarrow=False,
                font=dict(
                    color="red",
                    size=12
                )
            )

        # Update layout with dropdown and paper-style theme
        num_assets = len(weights_history.columns)
        fig.update_layout(
            updatemenus=[
                dict(
                    type="dropdown",
                    direction="down",
                    buttons=list([
                        dict(
                            label="日收益",
                            method="update",
                            args=[{"visible": [True]*num_assets + [True, False] + [True]*num_assets}]
                        ),
                        dict(
                            label="累计收益",
                            method="update",
                            args=[{"visible": [True]*num_assets + [False, True] + [True]*num_assets}]
                        )
                    ]),
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=1.1,
                    xanchor="left",
                    y=1.1,
                    yanchor="top"
                )
            ],
            title=title,
            title_font=dict(size=30, family='Arial Black'),
            title_x=0.25,
            height=height,
            template='plotly_white',
            hovermode='x unified',
            legend_title='Returns',
            legend_font=dict(size=11),
            legend=dict(
            x=1.1,  # 将图例向右移动
             ),
            font=dict(
                family="Arial, sans-serif",
                size=14,
                color="black"
            ),
            xaxis=dict(
                gridcolor='lightgray',
                linecolor='black',
                tickfont=dict(size=12)
            ),
            yaxis=dict(
                gridcolor='lightgray',
                linecolor='black',
                tickfont=dict(size=12)
            )

        )

        fig.update_yaxes(title_text="Risk Contribution",title_font=dict(size=12),tickfont=dict(size=12),tickformat=".2%", row=2, col=1)
        fig.update_yaxes(title_text="Portdolio Volatility",title_font=dict(size=12),tickfont=dict(size=12),tickformat=".2%",row=2, col=1, secondary_y=True)
        fig.update_yaxes(title_text="Asset Weights",title_font=dict(size=12),tickfont=dict(size=12),tickformat=".2%", row=1, col=1)
        fig.update_yaxes(title_text="Return",title_font=dict(size=12),tickfont=dict(size=12),tickformat=".2f",row=1, col=1, secondary_y=True)
        fig.update_xaxes(tickfont=dict(size=12))

        # Calculate summary metrics
        overall_return = cumulative_returns.iloc[-1]
        mean_return = portfolio_returns.mean()
        volatility = portfolio_returns.std()
        # Calculate running maximum of cumulative returns
        running_max = cumulative_returns.cummax()
        # Calculate drawdown
        drawdown = (running_max - cumulative_returns) / (1+running_max)
        # Calculate maximum drawdown
        max_drawdown = drawdown.max()
        sharpe_ratio = (mean_return - risk_free_rate) / volatility * np.sqrt(252)

        # Format summary metrics
        summary_metrics = f'总收益: {overall_return:.2%}<br>波动率: {volatility* np.sqrt(252):.2%}<br>夏普比率: {sharpe_ratio:.2}<br>最大回撤: {max_drawdown:.2%}'

        # Add summary table as annotation
        fig.add_annotation(
            x=1.34,
            y=0.3,
            xref="paper",
            yref="paper",
            align="left",
            showarrow=False,
            text=f"<b>业绩评价</b><br>{summary_metrics}",
            font=dict(size=12)
        )


        # 计算等权组合的收益
        num_assets = returns.shape[1]
        equal_weights = np.ones(num_assets) / num_assets  # 等权组合权重
        equal_weight_returns = returns.iloc[window_size:] @ equal_weights
        equal_weight_cumulative = (1 + equal_weight_returns).cumprod() - 1  

        # 绩效指标计算
        equal_weight_mean_return = equal_weight_returns.mean()
        equal_weight_volatility = equal_weight_returns.std()
        equal_weight_sharpe_ratio = (equal_weight_mean_return - risk_free_rate) / equal_weight_volatility* np.sqrt(252)
        running_max = equal_weight_cumulative.cummax()
        drawdown = (running_max - equal_weight_cumulative) / (1 + running_max)
        max_drawdown = drawdown.max()

        # 格式化汇总指标
        equal_weight_summary_metrics = f'总收益: {equal_weight_cumulative.iloc[-1]:.2%}<br>' \
                                        f'波动率: {equal_weight_volatility* np.sqrt(252):.2%}<br>' \
                                        f'夏普比率: {equal_weight_sharpe_ratio:.2f}<br>' \
                                        f'最大回撤: {max_drawdown:.2%}'

        # 在可视化函数中显示等权组合的业绩评价
        fig.add_annotation(
            x=1.36,
            y=0.11,  # 根据布局调整 y 位置
            xref="paper",
            yref="paper",
            align="left",
            showarrow=False,
            text=f"<b>等权组合业绩评价</b><br>{equal_weight_summary_metrics}",
            font=dict(size=12)
)


        # 在底部添加标注
        fig.add_annotation(
            text="Coded By Wenhang Gu",  # 标注文本
            xref="paper",  # 使用相对坐标（整个图形范围）
            yref="paper",  # 使用相对坐标（整个图形范围）
            x=0.5,  # 标注的水平位置（0 是左边缘，1 是右边缘，0.5 是中间）
            y=-0.1,  # 标注的垂直位置（0 是底部，1 是顶部，负值表示图形外部）
            showarrow=False,  # 不显示箭头
            font=dict(size=12, color="gray")  # 设置字体大小和颜色
        )

        return fig


    else:
        # 不滚动模式：静态分布可视化
        # 计算风险贡献
        cov_matrix = LedoitWolf().fit(returns).covariance_
        risk_contrib = risk_contribution(weights_history.iloc[0], cov_matrix)

        # 创建饼图（资产权重）
        fig_pie = go.Figure(data=[go.Pie(
            labels=weights_history.columns,
            values=weights_history.iloc[0],
            hole=0.3,
            textinfo='percent+label',
            marker=dict(colors=px.colors.qualitative.Pastel, line=dict(color='#000000', width=1)),
            pull=[0.02] * len(weights_history.columns),
            hoverinfo='label+percent+value',
            textfont=dict(size=12, color='black', family='Arial'),
            opacity=0.9
        )])

        fig_pie.update_layout(
            width=600, height=600,
            title='<b>资产权重分布</b>',
            title_font_size=30,
            title_font_family='Arial Black',
            legend_title='资产',
            legend_title_font_size=14,
            legend_font_size=12,
            legend_font_family='Arial',
            template='plotly_white',
            hoverlabel=dict(bgcolor='white', font_size=12, font_family='Arial', font_color='black'),
            paper_bgcolor='rgba(0, 0, 0, 0)',
            plot_bgcolor='rgba(0, 0, 0, 0)'
        )

        # 创建柱状图（风险贡献）
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=weights_history.columns,
            y=risk_contrib,
            name='风险贡献',
            marker=dict(color=px.colors.qualitative.Pastel, line=dict(color='#000000', width=1)),
            text=[f'{contrib:.2%}' for contrib in risk_contrib],
            textposition='auto',
            hoverinfo='x+y',
            textfont=dict(size=12, color='black', family='Arial'),
            opacity=0.9
        ))

        fig_bar.update_layout(
            title='<b>各资产风险贡献</b>',
            title_font=dict(size=30, family='Arial Black'),
            yaxis_title='<b>风险贡献</b>',
            xaxis_title='<b>资产</b>',
            template='plotly_white',
            showlegend=False,
            hoverlabel=dict(bgcolor='white', font_size=12, font_family='Arial', font_color='black'),
            paper_bgcolor='rgba(0, 0, 0, 0)',
            plot_bgcolor='rgba(0, 0, 0, 0)',
            xaxis=dict(tickfont=dict(size=12, family='Arial'), showgrid=True, gridcolor='lightgray'),
            yaxis=dict(tickfont=dict(size=12, family='Arial'), showgrid=True, gridcolor='lightgray')
        )

        fig_bar.update_traces(
            textfont_size=12,
            textangle=0,
            textposition="outside",
            cliponaxis=False
        )


        return fig_pie,fig_bar



# 主函数

# 自定义CSS，调整布局提高总宽度

st.title('ETF 投资组合优化应用')

# 参数说明
st.sidebar.header("参数说明")
st.sidebar.markdown("""
- **Alpha (α)**: 风险平价与最优夏普比率之间的权衡参数。α的值越高，模型越偏向于风险平价，即更加重视各资产的风险均衡分配。
- **Beta (β)**: 在优化过程中，β的值越高，模型越偏向于最大化夏普比率，即对组合收益的追求更为强调。
- **无风险利率**: 您认为可以获得的无风险资产的年化收益率，通常用作计算投资组合回报的基准。
- **使用滚动窗口优化**: 选择是否使用固定大小的时间窗口进行动态权重调整。
- **滚动窗口大小**: 每次优化时所使用的历史数据的时间长度（以天为单位）。
- **再平衡天数**: 再平衡的频率。
""")


# 用户输入
ETF = {
    '沪深300ETF': '510300.SH',
    '标普ETF': '513500.SH',
    '黄金ETF': '518880.SH',
    '印度LOF-FOF': '164824.SZ',
    '日经225ETF': '513880.SH',
    '德国（DAX）ETF': '513030.SH',
    '中国国债ETF': '511010.SH',
    '大宗商品QDII-LOF': '160216.SZ',
}

# 多选框选择ETF
selected_etfs = st.multiselect(
    '选择您感兴趣的ETF',
    options=list(ETF.keys()),
    default=list(ETF.keys())  # 默认选中所有ETF
)

# 配置ETF代码
selected_etf_codes = {name: ETF[name] for name in selected_etfs}

start_date = st.date_input('开始日期', value=pd.to_datetime('2023-10-01'))
end_date = st.date_input('结束日期', value=pd.to_datetime('2099-12-31'))
risk_free_rate = st.number_input('无风险利率（年化）', value=0.02) / 252
rolling_mode = st.checkbox('使用滚动窗口优化', value=True)
window_size = st.number_input('滚动窗口大小', value=66)
step_size = st.number_input('再平衡天数', value=22)
alpha = st.number_input('Alpha', value=0.0)
beta = st.number_input('Beta', value=1.0)


if st.button('运行优化'):
    returns = fetch_and_preprocess_data(selected_etf_codes, 'FD', start_date.strftime('%Y%m%d'), end_date.strftime('%Y%m%d'))
    returns = round(returns / 100, 4)
    
    weights_history, portfolio_returns, volatility_history, risk_contributions_history = optimize_portfolio(
        returns, risk_free_rate, alpha, beta, rolling_mode, window_size, step_size
    )
    
    # 重要事件
    important_dates = [
        {'date': '2024-09-23', 'event': "中国 '牛市'"},
        {'date': '2020-02-24', 'event': '新冠'},
    ]

    if rolling_mode ==True:
        fig=visualize_results('基于ETF的风险平价','等权',
        weights_history, portfolio_returns, risk_free_rate, returns,
        volatility_history, risk_contributions_history, rolling_mode, window_size,
        volatility_threshold=0.05, height=650, important_dates=important_dates,y_offset=1.08)
        
        st.plotly_chart(fig)
    else:
        fig_pie,fig_bar=visualize_results('基于ETF的风险平价','等权',
        weights_history, portfolio_returns, risk_free_rate, returns,
        volatility_history, risk_contributions_history, rolling_mode, window_size,
        volatility_threshold=0.05, height=650, important_dates=important_dates,y_offset=1.08)
        
        st.plotly_chart(fig_pie)
        st.plotly_chart(fig_bar)
