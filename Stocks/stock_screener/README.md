# Stock Screener

A Python CLI tool that screens stocks against 7 personal investment criteria using real-time data from Yahoo Finance. Supports custom ticker lists, the full S&P 500 universe, CSV export, and color-coded terminal output.

---

## Criteria

Each stock is scored out of 7. A full pass earns 1 point, a partial earns 0.5, and a fail or unknown earns 0.

| # | Criterion | Pass condition |
|---|-----------|---------------|
| 1 | **Price near 5-year low** | Current price is within 30% above the 5-year low |
| 2 | **Insider buying** | More insider buy transactions than sells in the last 6 months |
| 3 | **Moderate volatility** | Beta between 0.5 and 2.5 |
| 4 | **Market cap ≥ $500M** | Market capitalization is at least $500 million |
| 5 | **Reasonable P/E ratio** | Trailing or forward P/E is below a sector-adjusted ceiling |
| 6 | **Assets > liabilities** | Total assets exceed total liabilities on the latest balance sheet |
| 7 | **Revenue trend** | Last 4 quarters show consistent or growing revenue with no >20% QoQ drop |

### Ratings

| Score | Rating |
|-------|--------|
| 6 – 7 | Strong Buy |
| 5 | Buy |
| 4 | Hold / Watch |
| 2.5 – 3.5 | Weak |
| < 2.5 | Avoid |

---

## Installation

**Python 3.10 or higher is required.**

```bash
git clone https://github.com/your-username/stock-screener.git
cd stock-screener
pip install yfinance pandas tabulate colorama
```

Or install dependencies from a requirements file:

```bash
pip install -r requirements.txt
```

`requirements.txt`:
```
yfinance
pandas
tabulate
colorama
```

---

## Usage

### Screen the built-in watchlist (~50 stocks across diverse sectors)
```bash
python stock_screener.py
```

### Screen specific tickers
```bash
python stock_screener.py --tickers AAPL MSFT GOOGL PFE ET
```

### Screen the full S&P 500
```bash
python stock_screener.py --watchlist sp500
```
> This fetches ~500 tickers from Wikipedia and screens them all. Expect around 10–15 minutes due to API throttling.

### Only show stocks scoring 5 or higher
```bash
python stock_screener.py --min-score 5
```

### Export results to CSV
```bash
python stock_screener.py --export results.csv
```

### Quiet mode — summary table only, no per-stock breakdown
```bash
python stock_screener.py --quiet
```

### Combine options
```bash
python stock_screener.py --watchlist sp500 --min-score 5 --quiet --export top_picks.csv
```

---

## Sample Output

```
Stock Screener — 7 Criteria
Screening 50 tickers  |  min score: 4/7

[ 1] MSFT      score 5.5/7  [█████░░]  Buy
         ✅ 1. Price near 5yr low        Price $422.79 | 5yr low $213.43 | 98% above low
         ✅ 2. Insider buying             3 buy txns vs 1 sell txns in last 6 months
         ✅ 3. Moderate volatility        Beta = 0.90 (target 0.5–2.5)
         ✅ 4. Market cap $500M+          Market cap = $3.14T
         ✅ 5. Reasonable P/E             P/E = 33.2 | sector 'Technology' ceiling ~45
         ✅ 6. Assets > liabilities       Assets $512.16B vs Liabilities $243.91B | ratio 2.10x
         ❌ 7. Revenue trend              Last 4Q: $56.2B → $57.4B → $61.9B → $62.0B

════════════════════════════════════════════════════
  SCREENER RESULTS  —  8 stocks scored >= 4/7
════════════════════════════════════════════════════
Ticker   Name                    Price     Sector     Score   Rating        Criteria (1–7)
VTRS     Viatris Inc             $14.68    Healthcare  6.0/7  Strong Buy    ✅✅✅✅✅✅✅
PFE      Pfizer Inc              $27.56    Healthcare  5.5/7  Buy           ✅✅✅✅✅✅⚠️
NOK      Nokia OYJ ADR           $10.31    Technology  5.0/7  Buy           ✅✅✅✅✅❓✅
...
```

---

## Configuration

All thresholds are defined at the top of `stock_screener.py` and can be edited directly:

```python
# How far above the 5-year low still counts as "near low" (0.30 = 30%)
NEAR_LOW_THRESHOLD = 0.30

# Minimum market cap in USD
MIN_MARKET_CAP = 500_000_000

# Beta range for "moderate volatility"
BETA_MIN = 0.5
BETA_MAX = 2.5

# P/E ratio ceiling (sector-agnostic fallback)
PE_MAX_ACCEPTABLE = 40

# Revenue quarter-over-quarter miss threshold
REVENUE_MISS_PCT = 0.10

# Default minimum score to appear in the summary table
DEFAULT_MIN_SCORE = 4

# Delay between Yahoo Finance API calls (seconds)
THROTTLE_SECONDS = 0.8
```

Sector-specific P/E ceilings are also configurable inside `check_pe_ratio()`.

---

## All CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--tickers TICK ...` | — | One or more tickers to screen |
| `--watchlist default\|sp500` | `default` | Use a pre-built list (ignored if `--tickers` is set) |
| `--min-score N` | `4` | Minimum score (0–7) to include in summary |
| `--verbose` | on | Show per-criterion breakdown for each stock |
| `--quiet` | off | Show summary table only |
| `--export FILE.csv` | — | Save full results to a CSV file |
| `--throttle N` | `0.8` | Seconds to wait between API calls |

---

## Data source

All financial data is fetched live from **Yahoo Finance** via the [`yfinance`](https://github.com/ranaroussi/yfinance) library. Data includes:

- 5-year daily price history
- Insider transaction history
- Stock info (beta, P/E, market cap, sector)
- Quarterly balance sheet
- Quarterly income statement

> **Note:** Yahoo Finance data is free but unofficial. Accuracy may vary for small-cap or foreign-listed stocks. Always verify before making investment decisions.

---

## Disclaimer

This tool is for **informational and educational purposes only**. It is not financial advice. Always do your own research and consult a qualified financial advisor before making any investment decisions. Past performance and screening results do not guarantee future returns.

---

## License

MIT License. Free to use, modify, and distribute.
