import yfinance as yf
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

def get_monthly_quarterly_prices(ticker: str) -> pd.DataFrame:
    """
    Fetches daily adjusted prices for the given ticker, resamples to monthly (last trading day),
    and adds a calendar quarter label.
    Returns a DataFrame with columns: Date, Price, calendar_quarter.
    """
    dat = yf.Ticker(ticker)
    d = dat.history(period="17y", interval="1d", auto_adjust=True)
    s = d["Close"].tz_localize(None).resample("M").last()
    s.index.name = "month_end"
    monthly = s.reset_index(name="adj_close")
    monthly = monthly.rename(columns={"month_end": "Date", "adj_close": "Price"})
    df = pd.DataFrame(monthly)
    p = df["Date"].dt.to_period("Q")
    df["cal_year"] = p.dt.year.astype("Int64")
    df["cal_q"] = p.dt.quarter
    df["calendar_quarter"] = "Q" + df["cal_q"].astype(str) + " " + df["cal_year"].astype(str)
    df = df.drop(columns=["cal_year", "cal_q"])
    return df

# Example usage:
# tesla_price = get_monthly_quarterly_prices("TSLA")
# tesla_price.to_csv("tesla_price.csv", index=False)
