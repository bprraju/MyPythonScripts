#!/usr/bin/env python3
"""
Stock Screener — based on 7 personal investment criteria:
  1. Price near 5-year low
  2. Insider buying (not selling)
  3. Some volatility (moderate beta)
  4. Market cap >= $500M
  5. Reasonable P/E ratio
  6. Assets > liabilities
  7. Last 4 quarters revenue trending up (no huge misses)

Requirements:
    pip install yfinance requests pandas tabulate colorama

Usage:
    python stock_screener.py                          # screen default watchlist
    python stock_screener.py --tickers AAPL MSFT GOOG # screen specific tickers
    python stock_screener.py --watchlist sp500        # screen S&P 500 universe
    python stock_screener.py --min-score 5            # only show stocks scoring 5+/7
    python stock_screener.py --export results.csv     # also save to CSV
"""

import argparse
import sys
import time
import math
import csv
from datetime import datetime, timedelta
from typing import Optional

# ── optional colour support ───────────────────────────────────────────────────
try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    GREEN  = Fore.GREEN
    RED    = Fore.RED
    YELLOW = Fore.YELLOW
    CYAN   = Fore.CYAN
    BOLD   = Style.BRIGHT
    RESET  = Style.RESET_ALL
except ImportError:
    GREEN = RED = YELLOW = CYAN = BOLD = RESET = ""

# ── third-party ───────────────────────────────────────────────────────────────
try:
    import yfinance as yf
except ImportError:
    print("Missing dependency. Run:  pip install yfinance pandas tabulate colorama")
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    print("Missing dependency. Run:  pip install pandas")
    sys.exit(1)

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — edit these to tune the screener
# ═══════════════════════════════════════════════════════════════════════════════

# How close to the 5-year low counts as "near low"?
# 0.30 means within 30% above the 5-year low price.
NEAR_LOW_THRESHOLD = 0.30

# Minimum market cap in USD
MIN_MARKET_CAP = 500_000_000   # $500M

# Beta range for "moderate volatility"
BETA_MIN = 0.5
BETA_MAX = 2.5

# P/E ratio thresholds (sector-agnostic fallback)
PE_MAX_ACCEPTABLE  = 40   # above this → FAIL
PE_MAX_PARTIAL     = 60   # above this → definitely bad

# Revenue quarter-over-quarter miss tolerance
# If any quarter's actual is more than X% below estimates, flag it
REVENUE_MISS_PCT = 0.10   # 10% miss threshold

# Minimum score (out of 7) to include in output
DEFAULT_MIN_SCORE = 4

# Request throttle — yfinance can get rate-limited
THROTTLE_SECONDS = 0.8

# Default watchlist (diverse sectors, mid-to-large cap names worth screening)
DEFAULT_WATCHLIST = [
    # Tech / AI
    "MSFT", "GOOGL", "META", "NVDA", "AMD", "ORCL", "CRM", "NOW", "SNOW",
    "PLTR", "PATH", "AI", "SMAR",
    # Fintech / Finance
    "PYPL", "SQ", "SOFI", "NU", "V", "MA", "AFRM", "HOOD",
    # Healthcare / Biotech
    "PFE", "MRNA", "ABBV", "BMY", "LLY", "AMGN", "BIIB", "REGN",
    # Energy
    "XOM", "CVX", "ET", "OXY", "DVN", "FANG",
    # Industrials / Aerospace
    "GE", "RTX", "BA", "NOC", "LMT", "FTI",
    # Consumer / Retail
    "AMZN", "WMT", "TGT", "COST", "CMG", "MCD",
    # Travel / Airlines / Cruise
    "DAL", "UAL", "AAL", "LUV", "CCL", "NCLH", "RCL",
    # EV / Clean energy
    "TSLA", "RIVN", "LCID", "CHPT", "PLUG", "FCEL",
    # Telecom / Hardware
    "NOK", "ERIC", "T", "VZ",
    # REITs
    "MPT", "CHCT", "O", "AGNC",
    # Misc growth
    "RBLX", "U", "DKNG", "LYFT", "UBER",
]


# ═══════════════════════════════════════════════════════════════════════════════
#  SCORING HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

class CriterionResult:
    PASS    = "PASS"
    PARTIAL = "PARTIAL"
    FAIL    = "FAIL"
    UNKNOWN = "UNKNOWN"

    def __init__(self, status: str, note: str, points: float = 0):
        self.status = status
        self.note   = note
        self.points = points   # 1=pass, 0.5=partial, 0=fail/unknown

    def __repr__(self):
        return f"{self.status}: {self.note}"


def _c(status, note, points=None):
    """Shorthand constructor for CriterionResult."""
    if points is None:
        pts = {"PASS": 1, "PARTIAL": 0.5, "FAIL": 0, "UNKNOWN": 0}[status]
    else:
        pts = points
    return CriterionResult(status, note, pts)


# ── Criterion 1: Price near 5-year low ────────────────────────────────────────

def check_near_5yr_low(hist: pd.DataFrame, current_price: float) -> CriterionResult:
    if hist is None or hist.empty or current_price is None:
        return _c("UNKNOWN", "No price history available")
    try:
        five_yr_low  = hist["Low"].min()
        five_yr_high = hist["High"].max()
        if five_yr_low <= 0:
            return _c("UNKNOWN", "Invalid price data")
        pct_above_low = (current_price - five_yr_low) / five_yr_low
        pct_of_range  = (current_price - five_yr_low) / max(five_yr_high - five_yr_low, 0.01)

        note = (f"Price ${current_price:.2f} | 5yr low ${five_yr_low:.2f} "
                f"| {pct_above_low*100:.0f}% above low | in bottom {pct_of_range*100:.0f}% of 5yr range")

        if pct_above_low <= NEAR_LOW_THRESHOLD:
            return _c("PASS", note)
        elif pct_above_low <= NEAR_LOW_THRESHOLD * 2:
            return _c("PARTIAL", note, 0.5)
        else:
            return _c("FAIL", note)
    except Exception as e:
        return _c("UNKNOWN", f"Error: {e}")


# ── Criterion 2: Insider buying ────────────────────────────────────────────────

def check_insider_buying(ticker_obj) -> CriterionResult:
    try:
        insiders = ticker_obj.insider_transactions
        if insiders is None or insiders.empty:
            return _c("UNKNOWN", "No insider transaction data available")

        # Keep last 6 months
        cutoff = datetime.now() - timedelta(days=180)
        if "Start Date" in insiders.columns:
            date_col = "Start Date"
        elif "Date" in insiders.columns:
            date_col = "Date"
        else:
            date_col = insiders.columns[0]

        recent = insiders.copy()
        try:
            recent[date_col] = pd.to_datetime(recent[date_col], errors="coerce")
            recent = recent[recent[date_col] >= cutoff]
        except Exception:
            recent = insiders.head(20)

        if recent.empty:
            return _c("UNKNOWN", "No recent insider transactions (last 6 months)")

        # Determine buy/sell from "Shares" column sign or "Transaction" column
        buys = sells = 0
        buy_shares = sell_shares = 0

        tx_col = None
        for col in ["Transaction", "Text", "Type"]:
            if col in recent.columns:
                tx_col = col
                break

        shares_col = "Shares" if "Shares" in recent.columns else None

        for _, row in recent.iterrows():
            is_buy = False
            if tx_col:
                txt = str(row.get(tx_col, "")).lower()
                is_buy = any(w in txt for w in ["buy", "purchase", "acquisition"])
                is_sell = any(w in txt for w in ["sell", "sale", "disposition"])
            elif shares_col:
                shares_val = row.get(shares_col, 0)
                try:
                    shares_val = float(str(shares_val).replace(",", ""))
                    is_buy  = shares_val > 0
                    is_sell = shares_val < 0
                except Exception:
                    continue
            else:
                continue

            if is_buy:
                buys += 1
                if shares_col:
                    try:
                        buy_shares += abs(float(str(row.get(shares_col, 0)).replace(",", "")))
                    except Exception:
                        pass
            elif is_sell:
                sells += 1
                if shares_col:
                    try:
                        sell_shares += abs(float(str(row.get(shares_col, 0)).replace(",", "")))
                    except Exception:
                        pass

        total = buys + sells
        if total == 0:
            return _c("UNKNOWN", "Could not classify insider transactions")

        note = f"{buys} buy txns vs {sells} sell txns in last 6 months"
        if buy_shares and sell_shares:
            note += f" | {buy_shares:,.0f} shares bought vs {sell_shares:,.0f} sold"

        if buys > sells:
            return _c("PASS", note)
        elif buys == sells:
            return _c("PARTIAL", note, 0.5)
        else:
            return _c("FAIL", note)

    except Exception as e:
        return _c("UNKNOWN", f"Could not retrieve insider data: {e}")


# ── Criterion 3: Moderate volatility (beta) ────────────────────────────────────

def check_volatility(info: dict) -> CriterionResult:
    beta = info.get("beta")
    if beta is None:
        return _c("UNKNOWN", "Beta not available")
    note = f"Beta = {beta:.2f} (target {BETA_MIN}–{BETA_MAX})"
    if BETA_MIN <= beta <= BETA_MAX:
        return _c("PASS", note)
    elif beta < BETA_MIN:
        return _c("PARTIAL", note + " — too low volatility", 0.5)
    elif beta <= BETA_MAX * 1.3:
        return _c("PARTIAL", note + " — slightly high volatility", 0.5)
    else:
        return _c("FAIL", note + " — too volatile")


# ── Criterion 4: Market cap >= $500M ──────────────────────────────────────────

def check_market_cap(info: dict) -> CriterionResult:
    mc = info.get("marketCap")
    if mc is None:
        return _c("UNKNOWN", "Market cap not available")
    mc_str = f"${mc/1e9:.2f}B" if mc >= 1e9 else f"${mc/1e6:.0f}M"
    note = f"Market cap = {mc_str}"
    if mc >= MIN_MARKET_CAP:
        return _c("PASS", note)
    else:
        return _c("FAIL", note + f" (minimum ${MIN_MARKET_CAP/1e6:.0f}M)")


# ── Criterion 5: Reasonable P/E ───────────────────────────────────────────────

def check_pe_ratio(info: dict) -> CriterionResult:
    pe = info.get("trailingPE") or info.get("forwardPE")
    sector = info.get("sector", "")

    # Sector-adjusted PE ceiling
    sector_pe_max = {
        "Technology": 45,
        "Healthcare": 35,
        "Consumer Cyclical": 30,
        "Financial Services": 20,
        "Energy": 18,
        "Utilities": 22,
        "Basic Materials": 20,
        "Real Estate": 30,
        "Communication Services": 30,
        "Industrials": 25,
    }.get(sector, PE_MAX_ACCEPTABLE)

    if pe is None:
        return _c("UNKNOWN", "P/E not available (may be unprofitable or pre-revenue)")

    note = f"P/E = {pe:.1f} | sector '{sector}' ceiling ~{sector_pe_max}"

    if pe <= 0:
        return _c("FAIL", f"Negative P/E = {pe:.1f} — company is losing money")
    elif pe <= sector_pe_max:
        return _c("PASS", note)
    elif pe <= sector_pe_max * 1.4:
        return _c("PARTIAL", note + " — slightly elevated", 0.5)
    else:
        return _c("FAIL", note + " — too high")


# ── Criterion 6: Assets > liabilities ────────────────────────────────────────

def check_balance_sheet(ticker_obj) -> CriterionResult:
    try:
        bs = ticker_obj.balance_sheet
        if bs is None or bs.empty:
            return _c("UNKNOWN", "Balance sheet not available")

        # Get most recent column
        col = bs.columns[0]

        total_assets = None
        total_liabilities = None

        for asset_key in ["Total Assets", "TotalAssets"]:
            if asset_key in bs.index:
                total_assets = bs.loc[asset_key, col]
                break

        for liab_key in ["Total Liabilities Net Minority Interest",
                         "Total Liab", "TotalLiabilitiesNetMinorityInterest"]:
            if liab_key in bs.index:
                total_liabilities = bs.loc[liab_key, col]
                break

        if total_assets is None or total_liabilities is None:
            # Try computing liabilities from equity
            for eq_key in ["Stockholders Equity", "Total Stockholders Equity",
                            "Total Equity Gross Minority Interest"]:
                if eq_key in bs.index and total_assets is not None:
                    equity = bs.loc[eq_key, col]
                    total_liabilities = total_assets - equity
                    break

        if total_assets is None or total_liabilities is None:
            return _c("UNKNOWN", "Could not parse assets/liabilities from balance sheet")

        def fmt(v):
            v = float(v)
            if abs(v) >= 1e9: return f"${v/1e9:.2f}B"
            return f"${v/1e6:.0f}M"

        ratio = float(total_assets) / max(float(total_liabilities), 1)
        note = f"Assets {fmt(total_assets)} vs Liabilities {fmt(total_liabilities)} | ratio {ratio:.2f}x"

        if total_assets > total_liabilities:
            return _c("PASS", note)
        else:
            return _c("FAIL", note + " — liabilities exceed assets")

    except Exception as e:
        return _c("UNKNOWN", f"Error reading balance sheet: {e}")


# ── Criterion 7: Revenue trend (last 4 quarters) ──────────────────────────────

def check_revenue_trend(ticker_obj) -> CriterionResult:
    try:
        # Quarterly financials
        qf = ticker_obj.quarterly_financials
        if qf is None or qf.empty:
            return _c("UNKNOWN", "Quarterly financials not available")

        rev_row = None
        for key in ["Total Revenue", "Revenue", "Net Revenue"]:
            if key in qf.index:
                rev_row = qf.loc[key]
                break

        if rev_row is None:
            return _c("UNKNOWN", "Revenue line not found in financials")

        # Take last 4 quarters (most recent first)
        rev_vals = rev_row.dropna().iloc[:4]
        if len(rev_vals) < 2:
            return _c("UNKNOWN", "Not enough quarterly data")

        # Reverse so oldest → newest
        rev_list = list(reversed(rev_vals.values))
        quarters  = list(reversed([str(d)[:7] for d in rev_vals.index]))

        # Check for huge single-quarter drop
        declines = 0
        huge_miss = False
        for i in range(1, len(rev_list)):
            if rev_list[i-1] > 0:
                chg = (rev_list[i] - rev_list[i-1]) / rev_list[i-1]
                if chg < -REVENUE_MISS_PCT:
                    declines += 1
                if chg < -0.20:
                    huge_miss = True

        def fmt_rev(v):
            if abs(v) >= 1e9: return f"${v/1e9:.2f}B"
            return f"${v/1e6:.0f}M"

        trend_str = " → ".join(fmt_rev(v) for v in rev_list)
        note = f"Last 4Q revenue: {trend_str}"

        if huge_miss:
            return _c("FAIL", note + " — contains >20% QoQ drop")
        elif declines >= 2:
            return _c("FAIL", note + f" — {declines} declining quarters")
        elif declines == 1:
            return _c("PARTIAL", note + " — 1 declining quarter", 0.5)
        else:
            return _c("PASS", note + " — consistent or growing revenue")

    except Exception as e:
        return _c("UNKNOWN", f"Error reading quarterly financials: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN SCREENER
# ═══════════════════════════════════════════════════════════════════════════════

class StockResult:
    def __init__(self, ticker: str):
        self.ticker   = ticker
        self.name     = ""
        self.price    = None
        self.sector   = ""
        self.criteria = []          # list of CriterionResult
        self.score    = 0.0
        self.error    = None

    @property
    def rating(self) -> str:
        s = self.score
        if s >= 6:   return "Strong Buy"
        if s >= 5:   return "Buy"
        if s >= 4:   return "Hold / Watch"
        if s >= 2.5: return "Weak"
        return "Avoid"

    @property
    def rating_color(self) -> str:
        s = self.score
        if s >= 5:   return GREEN
        if s >= 4:   return YELLOW
        return RED


def screen_ticker(ticker: str) -> StockResult:
    result = StockResult(ticker)
    try:
        t    = yf.Ticker(ticker)
        info = t.info

        if not info or info.get("regularMarketPrice") is None and info.get("currentPrice") is None:
            result.error = "No data returned (delisted or invalid ticker?)"
            return result

        result.price  = info.get("currentPrice") or info.get("regularMarketPrice")
        result.name   = info.get("longName") or info.get("shortName") or ticker
        result.sector = info.get("sector", "Unknown")

        # Historical price data (5 years)
        hist = t.history(period="5y")

        checks = [
            check_near_5yr_low(hist, result.price),
            check_insider_buying(t),
            check_volatility(info),
            check_market_cap(info),
            check_pe_ratio(info),
            check_balance_sheet(t),
            check_revenue_trend(t),
        ]

        result.criteria = checks
        result.score    = sum(c.points for c in checks)

    except Exception as e:
        result.error = str(e)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════

CRITERION_NAMES = [
    "1. Price near 5yr low",
    "2. Insider buying",
    "3. Moderate volatility",
    "4. Market cap $500M+",
    "5. Reasonable P/E",
    "6. Assets > liabilities",
    "7. Revenue trend",
]

STATUS_ICON = {
    "PASS":    "✅",
    "PARTIAL": "⚠️ ",
    "FAIL":    "❌",
    "UNKNOWN": "❓",
}

STATUS_COLOR = {
    "PASS":    GREEN,
    "PARTIAL": YELLOW,
    "FAIL":    RED,
    "UNKNOWN": "",
}


def print_result(r: StockResult, verbose: bool = True):
    if r.error:
        print(f"{RED}{r.ticker:8s}{RESET} — Error: {r.error}")
        return

    color = r.rating_color
    bar   = "█" * int(r.score) + "░" * (7 - int(r.score))
    header = (f"\n{BOLD}{color}{r.ticker:8s}{RESET}  "
              f"{r.name[:40]:40s}  "
              f"${r.price or 0:>10.2f}  "
              f"Score {r.score:.1f}/7  [{bar}]  "
              f"{color}{r.rating}{RESET}")
    print(header)

    if verbose:
        for name, c in zip(CRITERION_NAMES, r.criteria):
            icon  = STATUS_ICON.get(c.status, "?")
            col   = STATUS_COLOR.get(c.status, "")
            print(f"   {icon} {col}{name:30s}{RESET}  {c.note}")


def print_summary_table(results: list[StockResult], min_score: float):
    passing = [r for r in results if not r.error and r.score >= min_score]
    if not passing:
        print(f"\n{YELLOW}No stocks passed with score >= {min_score}/7{RESET}")
        return

    passing.sort(key=lambda r: r.score, reverse=True)

    rows = []
    for r in passing:
        statuses = "".join(STATUS_ICON.get(c.status, "?") for c in r.criteria)
        rows.append([
            r.ticker,
            r.name[:35],
            f"${r.price:.2f}" if r.price else "N/A",
            r.sector[:20],
            f"{r.score:.1f}/7",
            r.rating,
            statuses,
        ])

    headers = ["Ticker", "Name", "Price", "Sector", "Score", "Rating", "Criteria (1-7)"]

    print(f"\n{BOLD}{'═'*100}{RESET}")
    print(f"{BOLD}  SCREENER RESULTS  —  {len(passing)} stocks scored >= {min_score}/7  "
          f"(screened {len(results)} total){RESET}")
    print(f"{'═'*100}{RESET}")

    if HAS_TABULATE:
        print(tabulate(rows, headers=headers, tablefmt="simple"))
    else:
        print("  ".join(f"{h:15s}" for h in headers))
        for row in rows:
            print("  ".join(f"{str(v):15s}" for v in row))


def export_csv(results: list[StockResult], path: str):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["Ticker", "Name", "Price", "Sector", "Score", "Rating",
                  "Error"] + [n for n in CRITERION_NAMES]
        writer.writerow(header)

        for r in results:
            if r.error:
                writer.writerow([r.ticker, "", "", "", "", "", r.error] + [""] * 7)
            else:
                status_notes = [f"{c.status}: {c.note}" for c in r.criteria]
                writer.writerow([
                    r.ticker,
                    r.name,
                    f"{r.price:.2f}" if r.price else "",
                    r.sector,
                    f"{r.score:.1f}",
                    r.rating,
                    "",
                ] + status_notes)

    print(f"\n{GREEN}Exported {len(results)} results → {path}{RESET}")


# ═══════════════════════════════════════════════════════════════════════════════
#  S&P 500 / NASDAQ 100 universe fetcher
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_sp500_tickers() -> list[str]:
    """Fetch S&P 500 tickers from Wikipedia."""
    try:
        import urllib.request
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        with urllib.request.urlopen(url, timeout=10) as resp:
            html = resp.read().decode("utf-8")
        # Quick parse — find tickers in the first table
        import re
        tickers = re.findall(r'<td><a[^>]*>([A-Z]{1,5})</a></td>', html)
        if tickers:
            return list(dict.fromkeys(tickers))[:505]
    except Exception as e:
        print(f"{YELLOW}Warning: Could not fetch S&P 500 list: {e}{RESET}")
    return []


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Screen stocks against 7 personal investment criteria.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--tickers", nargs="+", metavar="TICKER",
                   help="Space-separated list of tickers to screen")
    p.add_argument("--watchlist", choices=["default", "sp500"], default="default",
                   help="Use a pre-built watchlist (ignored if --tickers is set)")
    p.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE,
                   help=f"Minimum score to show in summary table (default: {DEFAULT_MIN_SCORE})")
    p.add_argument("--verbose", action="store_true", default=True,
                   help="Show per-criterion breakdown for every stock")
    p.add_argument("--quiet", action="store_true",
                   help="Only show the summary table, no per-stock breakdown")
    p.add_argument("--export", metavar="FILE.csv",
                   help="Save all results to a CSV file")
    p.add_argument("--throttle", type=float, default=THROTTLE_SECONDS,
                   help=f"Seconds to wait between API calls (default: {THROTTLE_SECONDS})")
    return p.parse_args()


def main():
    args = parse_args()
    verbose = args.verbose and not args.quiet

    # ── Build ticker list ─────────────────────────────────────────────────────
    if args.tickers:
        tickers = [t.upper().strip() for t in args.tickers]
    elif args.watchlist == "sp500":
        print(f"{CYAN}Fetching S&P 500 ticker list...{RESET}")
        tickers = fetch_sp500_tickers()
        if not tickers:
            print(f"{YELLOW}Falling back to default watchlist.{RESET}")
            tickers = DEFAULT_WATCHLIST
        else:
            print(f"{GREEN}Got {len(tickers)} tickers.{RESET}")
    else:
        tickers = DEFAULT_WATCHLIST

    print(f"\n{BOLD}Stock Screener — 7 Criteria{RESET}")
    print(f"Screening {len(tickers)} tickers  |  min score: {args.min_score}/7")
    print(f"Criteria threshold: price within {NEAR_LOW_THRESHOLD*100:.0f}% of 5yr low  |  "
          f"beta {BETA_MIN}–{BETA_MAX}  |  market cap ≥ ${MIN_MARKET_CAP/1e6:.0f}M\n")
    print("─" * 70)

    results: list[StockResult] = []
    total = len(tickers)

    for i, ticker in enumerate(tickers, 1):
        print(f"[{i:3d}/{total}] {ticker:8s}", end="", flush=True)
        r = screen_ticker(ticker)
        results.append(r)

        if r.error:
            print(f"  ⚠  {r.error[:60]}")
        else:
            bar = "█" * int(r.score) + "░" * (7 - int(r.score))
            col = r.rating_color
            print(f"  score {r.score:.1f}/7  [{bar}]  {col}{r.rating}{RESET}")
            if verbose:
                for name, c in zip(CRITERION_NAMES, r.criteria):
                    icon = STATUS_ICON.get(c.status, "?")
                    col2 = STATUS_COLOR.get(c.status, "")
                    print(f"         {icon} {col2}{name:30s}{RESET}  {c.note}")
                print()

        if i < total:
            time.sleep(args.throttle)

    # ── Summary table ─────────────────────────────────────────────────────────
    print_summary_table(results, args.min_score)

    # ── Legend ────────────────────────────────────────────────────────────────
    print(f"\n{BOLD}Legend:{RESET}")
    print(f"  {GREEN}Strong Buy{RESET} = 6–7/7   "
          f"{GREEN}Buy{RESET} = 5/7   "
          f"{YELLOW}Hold/Watch{RESET} = 4/7   "
          f"{RED}Weak{RESET} = 2.5–3.5   "
          f"{RED}Avoid{RESET} = <2.5")
    print(f"  Criteria icons: {STATUS_ICON['PASS']} Pass  "
          f"{STATUS_ICON['PARTIAL']} Partial (0.5 pts)  "
          f"{STATUS_ICON['FAIL']} Fail  "
          f"{STATUS_ICON['UNKNOWN']} No data")

    # ── Export ────────────────────────────────────────────────────────────────
    if args.export:
        export_csv(results, args.export)

    # ── Top picks summary ─────────────────────────────────────────────────────
    top = sorted([r for r in results if not r.error and r.score >= 5],
                 key=lambda r: r.score, reverse=True)
    if top:
        print(f"\n{BOLD}{GREEN}Top picks (score ≥ 5/7):{RESET}")
        for r in top[:10]:
            passes = [CRITERION_NAMES[i].split(".")[1].strip()
                      for i, c in enumerate(r.criteria) if c.status == "PASS"]
            print(f"  {GREEN}{r.ticker:8s}{RESET} {r.score:.1f}/7  — passes: {', '.join(passes)}")


if __name__ == "__main__":
    main()
