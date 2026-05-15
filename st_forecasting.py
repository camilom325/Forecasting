import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go


# Carga de modelos y esquema de columnas (debe coincidir con el entrenamiento)

model = joblib.load("./Models/xgboost_forecast.pkl")
model_columns = joblib.load("./Models/model_columns.pkl")


# Configuración de app

st.set_page_config(
    page_title="Demand Forecasting",
    layout="wide"
)

st.title("Retail Demand Forecasting - XGBoost")
st.write("Developed by J. Camilo Mercado")


# Extraer las posibles categorías únicas de las columnas categóricas para los selectboxes

df = pd.read_csv("./Data/retail_store_inventory.csv")

store_ids = df["Store ID"].unique().tolist()
product_ids = df["Product ID"].unique().tolist()
categories = df["Category"].unique().tolist()
regions = df["Region"].unique().tolist()
weather_conditions = df["Weather Condition"].unique().tolist()
seasonalities = df["Seasonality"].unique().tolist()

# Sidebar

st.sidebar.header("Forecast Configuration")

horizon = st.sidebar.slider(
    "Forecast horizon (days)",
    1,
    15,
    7
)

store_id = st.sidebar.selectbox(
    "Store ID",
    store_ids
)

product_id = st.sidebar.selectbox(
    "Product ID",
    product_ids
)

region = st.sidebar.selectbox(
    "Region",
    regions
)

category = st.sidebar.selectbox(
    "Category",
    categories
)

price = st.sidebar.slider(
    "Price",
    1.0,
    20.0,
    10.0
)

discount = st.sidebar.slider(
    "Discount",
    0.0,
    20.0,
    0.0
)

inventory = st.sidebar.slider(
    "Inventory Level",
    0,
    500,
    200
)

orders = st.sidebar.slider(
    "Units Ordered",
    0,
    200,
    250
)

demand_forecast = st.sidebar.slider(
    "Demand Forecast",
    0,
    100,
    250
)

competitor_price = st.sidebar.slider(
    "Competitor Pricing",
    1.0,
    20.0,
    10.0
)

weather = st.sidebar.selectbox(
    "Weather Condition",
    weather_conditions
)

holiday = st.sidebar.selectbox(
    "Holiday / Promotion",
    [0, 1]
)

seasonality = st.sidebar.selectbox(
    "Seasonality",
    seasonalities
)

# Construir dataframe futuro con las características seleccionadas (simplificado)

future = pd.DataFrame({
    "Store ID": [store_id] * horizon,
    "Product ID": [product_id] * horizon,
    "Region": [region] * horizon,
    "Inventory Level": [inventory] * horizon,
    "Units Ordered": [orders] * horizon,
    "Demand Forecast": [demand_forecast] * horizon,
    "Price": [price] * horizon,
    "Discount": [discount] * horizon,
    "Competitor Pricing": [competitor_price] * horizon,
    "Weather Condition": [weather] * horizon,
    "Holiday/Promotion": [holiday] * horizon,
    "Seasonality": [seasonality] * horizon,
    "Category": [category] * horizon
})

# Feautures de fecha (simplificado, sin considerar feriados específicos)

future["Date"] = pd.date_range(
    start="2024-01-01",
    periods=horizon
)

future["weekday"] = future["Date"].dt.weekday
future["month"] = future["Date"].dt.month

predictions = []

# en producción debe venir del dataset real histórico
history = list(np.random.normal(120, 15, 14))


for i in range(horizon):

    row = future.iloc[i].copy()

    lag_1 = history[-1]
    lag_7 = np.mean(history[-7:]) if len(history) >= 7 else lag_1
    rolling_mean_7 = np.mean(history[-7:]) if len(history) >= 7 else lag_1

    revenue = price * orders
    net_revenue = revenue * (1 - discount / 100)
    competitor_diff = competitor_price - price

    row_dict = row.to_dict()

    row_dict.update({
        "Revenue": revenue,
        "Net Revenue": net_revenue,
        "Competitor Price Difference": competitor_diff,
        "lag_1": lag_1,
        "lag_7": lag_7,
        "rolling_mean_7": rolling_mean_7
    })

    X = pd.DataFrame([row_dict])

    X = pd.get_dummies(X)

    X = X.reindex(columns=model_columns, fill_value=0)

    pred = model.predict(X)[0]

    predictions.append(pred)

    history.append(pred)

future["Forecast"] = predictions

# Output en tabla

st.subheader("Forecasted Units Sold")
st.dataframe(future[["Date", "Forecast"]])

# Output en gráfica

st.subheader("Forecast Visualization")

fig_trend = go.Figure()

df_filtered = df[(df["Store ID"] == store_id) &
                 (df["Product ID"] == product_id)]

fig_trend.add_trace(
    go.Scatter(
        x=df_filtered["Date"],
        y=df_filtered["Units Sold"],
        mode="lines",
        name="Actual Units sold"
    )
)

fig_trend.add_trace(
    go.Scatter(
        x=future["Date"],
        y=predictions,
        mode="lines",
        name="Forecast"
    )
)

st.plotly_chart(fig_trend, use_container_width=True)

# Heatmap Store X Product

st.subheader("Sales Heatmap - Store X Product")

heatmap_data = df.pivot_table(
    values="Units Sold",
    index="Store ID",
    columns="Product ID",
    aggfunc="sum"
)

fig_heatmap = go.Figure(data=go.Heatmap(
    z=heatmap_data.values,
    x=heatmap_data.columns,
    y=heatmap_data.index,
    colorscale="YlOrRd"
))

fig_heatmap.update_layout(
    title="Units Sold by Store and Product",
    xaxis_title="Product ID",
    yaxis_title="Store ID"
)

st.plotly_chart(fig_heatmap, use_container_width=True)

# Hallazgos opcionales

st.subheader("Summary Statistics")

st.write("Average forecast:", np.mean(predictions))
st.write("Max forecast:", np.max(predictions))
st.write("Min forecast:", np.min(predictions))

# COnlusión

st.subheader("Conclusion")
st.write("The analysis of demand patterns across stores and products reveals that sales dynamics are not purely linear with respect to pricing and promotional strategies. Instead, demand is driven by a set of non-linear and context-dependent relationships that must be carefully balanced to optimize revenue and inventory efficiency.")
st.write("Firstly, the results indicate that higher prices do not systematically lead to lower sales. In several product segments, increased pricing is associated with stronger or stable demand, suggesting the presence of perceived value effects, brand positioning dynamics, or inelastic demand zones. This highlights that pricing strategy should not be based solely on discount-driven assumptions, but rather optimized around product-specific price elasticity.")
st.write("Secondly, discount strategies show diminishing returns beyond certain thresholds. While promotions can stimulate demand, excessive discounting does not necessarily translate into proportional sales increases and may erode profitability. This suggests the existence of an optimal discount range that varies depending on product category, seasonality, and demand context.")
st.write("Thirdly, inventory management plays a critical role in aligning operational capacity with forecasted demand. Higher predicted demand directly translates into higher realized sales, reinforcing the importance of maintaining inventory levels that are responsive to short-term forecasts. Failure to adjust inventory dynamically may result in either stockouts, leading to lost sales opportunities, or overstock situations, increasing holding costs.")
st.write("Overall, the findings emphasize that demand is driven by a combination of historical behavior, pricing strategy, promotional intensity, and operational readiness. Effective decision-making therefore requires a coordinated approach where pricing, discounts, and inventory levels are continuously optimized against forecasted demand rather than managed in isolation.")
st.write("In conclusion, the business should shift from static rule-based decisions to a more dynamic, data-driven optimization framework that identifies optimal pricing and discount thresholds while aligning inventory levels with short-term demand forecasts. This will enable improved revenue capture, reduced inefficiencies, and stronger responsiveness to market variability.")
