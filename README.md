# Macroeconomic Forecasting with Machine Learning

This project leverages global macroeconomic indicators to predict the GDP Growth Rate of countries using advanced Machine Learning techniques. The dataset is independently aggregated from highly reputable sources such as the World Bank, IMF, and WEO.

Project Objective

The primary goal is to build a robust, data-driven predictive model that moves beyond heuristic economic forecasting. By analyzing panel data across multiple decades and countries, the model identifies critical factors influencing economic growth and filters out economic "noise".

Data Source & Features

The dataset was curated by extracting and merging global indicators. Key features include:

Target Variable: GDP_Growth (Gross Domestic Product Growth Rate)

Economic Indicators: GFCF (Investment), FDI (Foreign Direct Investment), CPI (Inflation), Unemployment, Current Account, Exchange Rate, Govt Expenditure, Population Growth, Export/GDP.

Global Context: Global oil prices and interest rates were integrated to capture global market trends.

Methodology & Machine Learning Pipeline

Data Preprocessing: Handled missing values using Hybrid Imputation (Linear Interpolation + MICE).

Anomaly Detection: Applied Isolation Forest to detect and cap severe outliers common in macroeconomic data.

Feature Engineering: Created temporal Lag Features and applied Log transformations to stabilize variances.

Model Selection: Evaluated Multiple Linear Regression, Decision Trees, Random Forest, and XGBoost.

Feature Elimination (RFE): Dropped highly noisy/uncorrelated features (e.g., FDI) to improve model generalization.

Results & The Reality of Financial Forecasting

Best Performing Model: XGBoost Regressor

Performance: Achieved an R-squared score of ~0.60 and RMSE of ~2.12.

A Note on R-squared Scores in Macroeconomics:
In physical sciences or engineering, R-squared scores of 0.90+ are generally expected. However, in finance and macroeconomics, systems are highly stochastic. They are constantly subjected to unmeasurable exogenous variables (geopolitical events, global pandemics, policy shifts, and market psychology).

According to quantitative finance literature, predicting macroscopic targets like global GDP Growth with an R-squared of 0.60 is considered an exceptionally strong signal. It proves that the XGBoost model successfully captures the underlying economic mechanics despite the inherent noise of global markets, significantly outperforming heuristic or naive baseline forecasts.

How to Run

Clone the repository and install dependencies:

pip install -r requirements.txt


Place the raw dataset in the data/raw/ directory and name it main_data.xlsx.

Run the main pipeline:

python main.py


The best-performing model will be automatically saved in the models/saved_models/ directory as a .pkl file for future inference.
