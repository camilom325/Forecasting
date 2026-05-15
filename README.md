# Retail Demand Forecasting with XGBoost & Streamlit

## Project Overview

This project is an end-to-end **retail demand forecasting system** designed to predict product-level sales across multiple stores using machine learning. The solution combines feature engineering, time-series-aware modeling, and an interactive Streamlit dashboard to support data-driven decision-making in pricing, inventory management, and demand planning.

The model is built using **XGBoost regression**, enriched with lag-based features, rolling statistics, and business-driven variables such as pricing, discounts, competitor pricing, weather conditions, and seasonality.

---

## Business Objective

The goal of this project is to enable better operational and strategic decisions by:

- Forecasting short-term product demand at store level
- Supporting inventory optimization to reduce stockouts and overstock
- Analyzing the impact of price and discounts on demand elasticity
- Understanding external drivers such as seasonality and weather
- Providing an interactive tool for scenario simulation

---

## Key Insights

- Demand is not linearly correlated with price; in some cases, higher prices are associated with higher sales due to perceived value or inelastic demand segments.
- Discounting exhibits diminishing returns, meaning there is an optimal promotional range beyond which additional discounts do not significantly increase demand.
- Inventory levels must be dynamically aligned with short-term forecasts to avoid inefficiencies such as stockouts or overstock situations.
- Historical demand (lag features) is the strongest predictor of future sales behavior.
