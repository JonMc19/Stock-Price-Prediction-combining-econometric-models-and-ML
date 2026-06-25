# 1. Project Goal
Building a machine learning model to estimate future stock prices for the following 4 quarters, based on historical patterns and fundamentals, and evaluating its performance on past unseen data.

### 2. Data Collection and Data Cleaning

The whole pipeline — scraping, merging, ARIMA/ARDL + Prophet modeling, backtesting, and plotting — lives
in a single notebook: [`Time series analysis/pipeline/full_pipeline.ipynb`](Time%20series%20analysis/pipeline/full_pipeline.ipynb).
It replaces four earlier, separate notebooks (kept for reference under
[`Time series analysis/old/`](Time%20series%20analysis/old/)).

📊 Stock & Fundamentals Sources:
1. Fundamental data (revenue, net income, total assets, total liabilities, shareholder equity) — scraped quarterly from [Macrotrends](https://www.macrotrends.net/).
2. Prices — obtained via the `yfinance` API, resampled to month-end.
3. Macro-level data — quarterly US real GDP growth from [multpl.com](https://www.multpl.com/us-real-gdp-growth-rate/table/by-quarter), and the effective Fed Funds rate from [FRED](https://fred.stlouisfed.org/) via the `fredapi` package (requires a free `FRED_API_KEY`, see [How to Run](#8-how-to-run-a-prediction) below).

All three sources are cached to CSV (`fundamentals_cache/`, `macro_cache/`, `price_cache/`) on first
fetch, so re-running the notebook doesn't re-scrape/re-download unless the cache is cleared or
`use_cache=False` is passed.

---

#### Key Variables Collected

| Company-Level      | Macro-Level                |
| ------------------- | --------------------------- |
| Revenue             | GDP growth                  |
| Net income          | Interest rates (Fed Funds)  |
| Total assets        |                              |
| Total liabilities   |                              |
| Shareholder equity  |                              |

Sector-level variables (sector growth/volatility/margins, industry outlook, sector P/E) and other
macro indicators (inflation, currency exchange, market index growth) were considered but are not
currently collected — see [Possible improvements](#10-possible-improvements).

---

#### Variables Used in the Model

- **Endogenous variable** → Stock Price (`y`)  

- **Exogenous variables**:  

| Company-Level      | Macro-Level    |
| ------------------- | --------------- |
| Revenue             | GDP growth      |
| Net income          | Interest rates  |
| Total assets        |                 |
| Total liabilities   |                 |
| Shareholder equity  |                 |


# 3. Target Construction

1. Define a DataFrame where `y` (stock price) is the endogenous variable.  
2. Include the key financial and macroeconomic variables as exogenous regressors, which will be used to predict future stock prices.  

# 4. Modeling

1. The Prophet model is used to generate forecasts of the target variable (y).
2. Since Prophet requires future values of the exogenous variables, these regressors must first be forecasted.
3. To obtain these future regressors, each fundamental metric is forecast two ways and the better one is kept:
   - An `auto_arima` (pmdarima) model with quarterly seasonality (`m=4`).
   - An ARDL (AutoRegressive Distributed Lag) model (`statsmodels`), using the metric's own lags plus lags of the macro regressors (GDP growth, interest rate), with lag order chosen via `ardl_select_order` (AIC).

   For each metric, whichever model has the lower backtest MAE wins, and its full-history forecast feeds the rest of the pipeline. Macro regressors themselves (GDP growth, interest rate) are held at their last observed value rather than forecasted, both as Prophet's future regressors and as ARDL's future exogenous inputs.

# 5. Evaluation

1. To assess model performance, we apply backtesting. This involves training the model using data up to 4 quarters (`BACKTEST_PERIODS`) before the most recent date, then comparing its predictions against the actual observed values.

2. The following metrics are used to evaluate the quality of the forecasts:

- MAE (Mean Absolute Error) — measures the average magnitude of errors.
- RMSE (Root Mean Squared Error) — penalizes larger errors more heavily.
- R² (Coefficient of Determination) — indicates how well the model explains the variance in the data.

# 6. Visualization

A single combined plot is generated showing:
1. Observed historical price.
2. The full-history forecast (trained on all available data, including genuinely unseen future quarters).
3. The backtest forecast (trained with the last `BACKTEST_PERIODS` quarters held out), overlaid for direct comparison against the full forecast and the actual observed values in that window.

# 7. Weaknesses

1. Limited data availability: The dataset is quarterly, so even with 40 years of history we would only have ~160 data points. This small sample size limits the model’s ability to learn complex patterns and increases the risk of overfitting. 

2.  Short forecast horizon: The current setup only predicts 4 quarters ahead. Longer horizons (1–5 years) would require additional validation and may accumulate significant error.

3. Dependence on regressor forecasts: Prophet requires future values of exogenous variables. Since these are predicted with ARIMA or ARDL (whichever wins the backtest comparison per metric), any errors in those forecasts propagate into Prophet's final predictions. The winning model's own forecast intervals are also discarded — only the point forecast feeds Prophet — so `yhat_lower`/`yhat_upper` understate true uncertainty for future quarters.

4. Stationarity assumptions (ARIMA) / lag selection (ARDL): ARIMA requires stationary input series, and while transformations are applied, fundamental data may still violate assumptions, impacting forecast stability. ARDL's lag order is also chosen by AIC over a limited search space (`ARDL_MAXLAG`, `ARDL_MAXORDER`), so it may not find the globally optimal lag structure.

5. Single train/test split: both the Prophet backtest and the ARIMA-vs-ARDL model selection in Step 5c use one fixed holdout window rather than a rolling-origin (walk-forward) backtest, which would give a more reliable error estimate.

# 8. How to Run a Prediction

1. Install dependencies: `pip install -r requirements.txt`.
2. Get a free FRED API key at https://fred.stlouisfed.org/docs/api/api_key.html and either `export FRED_API_KEY=...` before launching Jupyter, or put it in a `.env` file (`FRED_API_KEY=...`) next to `full_pipeline.ipynb`.
3. Open [`Time series analysis/pipeline/full_pipeline.ipynb`](Time%20series%20analysis/pipeline/full_pipeline.ipynb).
4. In the **Config** cell, set `TICKER` to one of the companies already in `TICKER_SLUGS` (`AAPL`, `MSFT`, `TSLA`, `NKE`, `AMD`, `CRM`, `ADBE`, `HNST`), or add a new ticker → Macrotrends slug mapping.
5. Run the notebook top to bottom:
   - **Steps 1–3** scrape/fetch fundamentals, macro data, and price (cached to CSV after the first run).
   - **Step 4** merges them into one modeling dataset, joined on date via `pd.merge_asof` (nearest calendar quarter, within `MACRO_TOLERANCE_DAYS`).
   - **Step 5a** forecasts each fundamental forward with ARIMA (`auto_arima`).
   - **Step 5b** forecasts each fundamental forward with ARDL (own lags + macro lags), holding macro regressors flat for the forecast horizon.
   - **Step 5c** compares ARIMA vs. ARDL backtest MAE per metric and picks the winner's full-history forecast.
   - **Step 6** builds the future frame Prophet needs (winning ARIMA/ARDL forecasts + held-flat macro regressors).
   - **Step 7** fits Prophet (once on full history, once on the backtest-truncated history) and forecasts price.
   - **Step 8** prints MAE/RMSE/R² for both the full and backtest forecasts.
   - **Step 9** plots observed price, the full forecast, and the backtest forecast together.

# 9. Issues to Keep in Mind

1. Companies don't always report earnings on calendar quarter-end dates, while the macro data is. The pipeline handles this by joining fundamentals to the *nearest* macro quarter (`pd.merge_asof`, `direction="nearest"`) within a configurable tolerance (`MACRO_TOLERANCE_DAYS`, default 40 days) — if a fundamentals row falls outside that tolerance for every macro quarter, the merge step raises an explicit error rather than silently dropping or misaligning rows.

2. `pmdarima` and recent NumPy/SciPy releases can be version-sensitive — pin versions per `requirements.txt` rather than upgrading individual packages in isolation.

3. Prophet depends on a working CmdStan backend (via `cmdstanpy`); on some platforms the bundled CmdStan install can be incomplete, causing a misleading `AttributeError: 'Prophet' object has no attribute 'stan_backend'` instead of a clear install error.

# 10. Possible Improvements

1. Add more regressors, especially on the macro side (inflation, government deficits) and sector level (sector growth/volatility, industry outlook); on the fundamental side, cash flow could also be added.

2. Add a longer time series.

3. Complement the time series analysis with a cross-sectional analysis.

4. Hyperparameter optimization. The model uses default or basic parameters. A systematic search (grid/random/Bayesian) could improve performance.

5. Walk-forward (rolling-origin) backtesting instead of a single train/test split, for a more reliable error estimate (this would also make the ARIMA-vs-ARDL model selection more robust).

6. Propagate the winning model's (ARIMA or ARDL) forecast uncertainty into Prophet's regressors instead of using point forecasts only.

7. Multi-ticker run: loop the pipeline over every entry in `TICKER_SLUGS` to build a model-comparison table across companies.
