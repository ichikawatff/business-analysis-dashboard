import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict

# 1. Data Setup
def get_sample_data() -> pd.DataFrame:
    """
    サンプルデータを作成し、計算列を追加して返す。
    """
    data = {
        "month": ["2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06"],
        "revenue": [300, 320, 350, 400, 380, 420],
        "labor_cost": [100, 100, 110, 120, 120, 130],
        "outsourcing_cost": [50, 60, 80, 100, 90, 110],
        "rent": [30, 30, 30, 30, 30, 30],
        "ad_cost": [20, 30, 50, 80, 40, 50],
        "other_cost": [10, 10, 10, 10, 10, 10]
    }
    df = pd.DataFrame(data)
    
    # 計算列の作成
    cost_columns = ["labor_cost", "outsourcing_cost", "rent", "ad_cost", "other_cost"]
    df["total_expense"] = df[cost_columns].sum(axis=1)
    df["operating_profit"] = df["revenue"] - df["total_expense"]
    df["profit_margin"] = (df["operating_profit"] / df["revenue"] * 100).round(1)
    df["labor_cost_ratio"] = (df["labor_cost"] / df["revenue"] * 100).round(1)
    
    return df

# 2. UI Layout (Streamlit)
st.set_page_config(layout="wide", page_title="経営分析ダッシュボード")

st.title("2026上期 経営分析ダッシュボード")

# Sidebar
df = get_sample_data()
st.sidebar.header("フィルター")
selected_months = st.sidebar.multiselect(
    "表示する月を選択してください",
    options=df["month"].tolist(),
    default=df["month"].tolist()
)

# データのフィルタリング
df_filtered = df[df["month"].isin(selected_months)].reset_index(drop=True)

if df_filtered.empty:
    st.warning("月を選択してください。")
    st.stop()

# 3. Visualizations

# --- KPI Cards ---
st.subheader("主要KPI (最新月)")
kpi_cols = st.columns(5)
latest_idx = len(df_filtered) - 1
prev_idx = latest_idx - 1 if latest_idx > 0 else None

def display_kpi(col, label: str, current_val: float, prev_val: float, unit: str = "万円", is_percentage: bool = False):
    delta = None
    if prev_val is not None:
        delta = current_val - prev_val
    
    val_str = f"{current_val:,.1f}{unit}" if is_percentage else f"{int(current_val)}{unit}"
    delta_str = f"{delta:+.1f}" if delta is not None else None
    
    col.metric(label=label, value=val_str, delta=delta_str)

latest_row = df_filtered.iloc[latest_idx]
prev_row = df_filtered.iloc[prev_idx] if prev_idx is not None else None

display_kpi(kpi_cols[0], "売上高", latest_row["revenue"], prev_row["revenue"] if prev_row is not None else None)
display_kpi(kpi_cols[1], "総支出", latest_row["total_expense"], prev_row["total_expense"] if prev_row is not None else None)
display_kpi(kpi_cols[2], "営業利益", latest_row["operating_profit"], prev_row["operating_profit"] if prev_row is not None else None)
display_kpi(kpi_cols[3], "利益率", latest_row["profit_margin"], prev_row["profit_margin"] if prev_row is not None else None, unit="%", is_percentage=True)
display_kpi(kpi_cols[4], "人件費率", latest_row["labor_cost_ratio"], prev_row["labor_cost_ratio"] if prev_row is not None else None, unit="%", is_percentage=True)

# --- Main Trend ---
st.subheader("売上・利益・支出推移")
fig_trend = go.Figure()
fig_trend.add_trace(go.Bar(x=df_filtered["month"], y=df_filtered["revenue"], name="売上", marker_color='rgb(55, 83, 109)'))
fig_trend.add_trace(go.Scatter(x=df_filtered["month"], y=df_filtered["operating_profit"], name="利益", line=dict(color='firebrick', width=4)))
fig_trend.add_trace(go.Scatter(x=df_filtered["month"], y=df_filtered["total_expense"], name="支出", line=dict(color='orange', width=2, dash='dash')))

fig_trend.update_layout(
    xaxis_title="月",
    yaxis_title="金額 (万円)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=0, r=0, t=30, b=0),
    font=dict(family="Meiryo, sans-serif")
)
st.plotly_chart(fig_trend, use_container_width=True)

# --- Detail Analysis ---
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("支出内訳")
    cost_categories = ["labor_cost", "outsourcing_cost", "rent", "ad_cost", "other_cost"]
    cost_labels = {"labor_cost": "人件費", "outsourcing_cost": "外注費", "rent": "家賃", "ad_cost": "広告費", "other_cost": "その他"}
    
    df_cost_melted = df_filtered.melt(id_vars=["month"], value_vars=cost_categories, var_name="category", value_name="amount")
    df_cost_melted["category_jp"] = df_cost_melted["category"].map(cost_labels)
    
    fig_cost = px.bar(df_cost_melted, x="month", y="amount", color="category_jp", 
                      labels={"amount": "金額 (万円)", "month": "月", "category_jp": "費目"},
                      color_discrete_sequence=px.colors.qualitative.Pastel)
    fig_cost.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=30, b=0),
        font=dict(family="Meiryo, sans-serif")
    )
    st.plotly_chart(fig_cost, use_container_width=True)

with col_right:
    st.subheader("売上と利益率（2軸）")
    fig_dual = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig_dual.add_trace(
        go.Bar(x=df_filtered["month"], y=df_filtered["revenue"], name="売上", marker_color='rgba(0,0,255,0.3)'),
        secondary_y=False,
    )
    fig_dual.add_trace(
        go.Scatter(x=df_filtered["month"], y=df_filtered["profit_margin"], name="利益率 (%)", line=dict(color='red', width=2)),
        secondary_y=True,
    )
    
    fig_dual.update_layout(
        xaxis_title="月",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=30, b=0),
        font=dict(family="Meiryo, sans-serif")
    )
    fig_dual.update_yaxes(title_text="売上 (万円)", secondary_y=False)
    fig_dual.update_yaxes(title_text="利益率 (%)", secondary_y=True)
    
    st.plotly_chart(fig_dual, use_container_width=True)

# --- Waterfall Chart ---
st.subheader(f"最新月 ({latest_row['month']}) の利益構成")

costs_latest = latest_row[cost_categories]
waterfall_data = {
    "label": ["売上"] + [cost_labels[c] for c in cost_categories] + ["営業利益"],
    "measure": ["relative"] + ["relative"] * len(cost_categories) + ["total"],
    "value": [latest_row["revenue"]] + [-c for c in costs_latest] + [0] # 0 for total
}

fig_waterfall = go.Figure(go.Waterfall(
    name="利益構成", orientation="v",
    measure=waterfall_data["measure"],
    x=waterfall_data["label"],
    textposition="outside",
    text=[f"{v:,.0f}" for v in [latest_row["revenue"]] + [-c for c in costs_latest] + [latest_row["operating_profit"]]],
    y=waterfall_data["value"],
    connector={"line": {"color": "rgb(63, 63, 63)"}},
))

fig_waterfall.update_layout(
    yaxis_title="金額 (万円)",
    margin=dict(l=0, r=0, t=30, b=0),
    font=dict(family="Meiryo, sans-serif"),
    showlegend=False
)
st.plotly_chart(fig_waterfall, use_container_width=True)
