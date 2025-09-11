# 1. Project Goal
Building a machine learning model to estimate future stock prices for the following 4 quarters, based on historical patterns and fundamentals, and evaluating its performance on past unseen data.

### 2. Data Collection and Data Cleaning

📊 Stock & Fundamentals Sources:  
1. Fundamental data* — downloaded directly from the SEC.  
2. Prices — obtained via the `yfinance` API.  
3. Macro-level data — collected from publicly available sources; this project uses the U.S. Bureau of Economic Analysis.  

---

#### Key Variables to Collect

| Company-Level     | Sector-Level                      | Macro-Level                         |
| ----------------- | --------------------------------- | ----------------------------------- |
| Revenue           | Sector average growth             | GDP growth                          |
| Net income        | Sector volatility                 | Interest rates                      |
| Total assets      | Sector average margins            | Inflation                           |
| Total liabilities | Industry outlook (manual tagging) | Currency exchange                   |
| Equity            | Sector P/E average                | Market index growth (S&P500, etc.)  |

---

#### Variables Used in the Model

- **Endogenous variable** → Stock Price (`y`)  

- **Exogenous variables**:  

| Company-Level     | Macro-Level   |
| ----------------- | ------------- |
| Revenue           | GDP growth    |
| Net income        | Interest rates|
| Total assets      |               |
| Total liabilities |               |
| Equity            |               |


# 3. Target Construction

1. Define a DataFrame where `y` (stock price) is the endogenous variable.  
2. Include the key financial and macroeconomic variables as exogenous regressors, which will be used to predict future stock prices.  

# 4. Modeling

1. The Prophet model is used to generate forecasts of the target variable (y).
2. Since Prophet requires future values of the exogenous variables, these regressors must first be forecasted.
3. To obtain these future regressors, an ARIMA model is applied to each company’s fundamental data, producing projections for the next 4 quarters.

# 5. Evaluation

1. To assess model performance, we apply backtesting. This involves training the model using data up to 4 quarters before the most recent date, then comparing its predictions against the actual observed values.

2. The following metrics are used to evaluate the quality of the forecasts:

- MAE (Mean Absolute Error) — measures the average magnitude of errors.
- RMSE (Root Mean Squared Error) — penalizes larger errors more heavily.
- R² (Coefficient of Determination) — indicates how well the model explains the variance in the data.

# 6. Visualization

1. To better interpret the results, several plots are generated:
2. Full dataset plot — shows the complete historical time series along with the model’s forecast.
3. Backtesting plot — compares predictions against actual observations in the backtesting window.
4. Comparison plot — overlays both forecasts to visually assess the difference between the backtest performance and the full model forecast.

# 7. Weaknesses

1. Limited data availability: The dataset is quarterly, so even with 40 years of history we would only have ~160 data points. This small sample size limits the model’s ability to learn complex patterns and increases the risk of overfitting. 

2.  Short forecast horizon: The current setup only predicts 4 quarters ahead. Longer horizons (1–5 years) would require additional validation and may accumulate significant error.

3. Dependence on regressor forecasts: Prophet requires future values of exogenous variables. Since these are predicted with ARIMA, any errors in ARIMA forecasts propagate into Prophet’s final predictions.

4. Simplified feature set: Only a limited number of fundamentals and macroeconomic indicators are used. Excluding other financial ratios or market data (e.g., volatility, technical indicators) may limit accuracy.

5. No hyperparameter optimization: The models are used largely with default or basic parameters. A systematic search (grid/random/Bayesian) could improve performance.

6. Stationarity assumptions (ARIMA): ARIMA requires stationary input series, and while transformations are applied, fundamental data may still violate assumptions, impacting forecast stability.

# 8. How to Make a Prediction

1. Download the data — collect fundamentals, stock prices, and macroeconomic indicators.  
2. Merge datasets into a single DataFrame aligned by date.  
3. Split into two DataFrames:  
   - Full dataset (for the main forecast).  
   - Backtesting dataset (same data but truncated 4 quarters before the most recent date).  
4. Forecast regressors — use ARIMA models to predict the exogenous variables needed for Prophet.  
5. Run Prophet — fit the Prophet model with regressors and generate the forecast.  
6. Repeat steps 4–5 for backtesting — apply the same pipeline to the truncated dataset for evaluation.  
7. Visualize results — plot:  
   - The full forecast.  
   - The backtesting forecast.  
   - A comparison between the two.  

# 9. Issues to Keep in Mind

1. When merging fundamental data with macroeconomic data, keep in mind that companies do not always report earnings in calendar quarters. Since macro data is published strictly on calendar quarters, misalignment can occur and the resulting merged DataFrame may be empty if dates do not match.  
