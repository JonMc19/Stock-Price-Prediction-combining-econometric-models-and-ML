# Final-Project-Ironhack
Building a machine learning model to estimate future stock prices at 1, 3, and 5-year horizons, based on historical patterns and fundamentals, and evaluating its performance on past unseen data.

# 1. Project Goal
Predict stock prices at 1, 3, and 5-year horizons using historical stock and company fundamentals data, industry trends, and macroeconomic indicators.

# 2. Data Collection and Data Cleaning

📊 Stock & Fundamentals:

Yahoo Finance (via yfinance)
Alpha Vantage
Financial Modeling Prep
Kaggle datasets

Key variables to collect:
| Company-Level     | Sector-Level                      | Macro-Level                         |
| ----------------- | --------------------------------- | ----------------------------------- |
| P/E ratio         | Sector avg. growth                | GDP growth                          |
| Revenue growth    | Sector volatility                 | Interest rates                      |
| Net income growth | Sector avg. margins               | Inflation                           |
| Profit margin     | Industry outlook (manual tagging) | Currency exchange                   |
| Debt ratios       | Sector PE avg                     | Market index growth (S\&P500, etc.) |

# 3. Target Construction

Define future prices at T+1, T+3 & T+5 year for each company:

df['price_t+1'] = df.groupby('ticker')['price'].shift(-252)
df['price_t+3'] = df.groupby('ticker')['price'].shift(-252*3)
df['price_t+5'] = df.groupby('ticker')['price'].shift(-252*5)

Or use log returns:

df['log_return_1y'] = np.log(df['price_t+1'] / df['price'])


# 4. Feature Engineering

Normalize or standardize continuous features
Convert categorical (e.g., sector, country) into one-hot or embeddings
Handle missing values properly (median imputation or domain-specific)

# 5. Modeling

Try different regression models:

Baseline: Linear regression or Ridge
Tree-based: Random Forest, XGBoost
Advanced: LightGBM, or even time-aware LSTM (optional)
For each horizon (1y, 3y, 5y), train separate models, or try multi-output regression.

# 6. Evaluation

You won’t know the real future prices yet, so you simulate:

Train on data up to 2015 → predict 2016
Train up to 2016 → predict 2017, etc.
Metrics to use:

MAE (Mean Absolute Error)
RMSE
MAPE
Compare vs a naïve model (e.g., +8% yearly growth)

# 7. Visualization


Price prediction vs actual over time
Feature importance
Sector or country-wise error heatmap
Correlation matrix

# 8. Presentation & Report

Clearly explain:

Why this model matters for long-term investors
What patterns the model picks up on
Which features are most predictive (e.g. “Revenue growth is more useful than debt”)
Realistic limitations (e.g., "Model is sensitive to recession years")