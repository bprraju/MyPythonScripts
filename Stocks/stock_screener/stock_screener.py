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
    python stock_screener.py --export results.csv      # also save to CSV
    python stock_screener.py --export-html report.html # save as browser report
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
#  HTML EXPORT
# ═══════════════════════════════════════════════════════════════════════════════

def export_html(results: list, path: str, min_score: float):
    """Generate a self-contained, browser-ready HTML report."""

    generated_at = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    total        = len(results)
    errors       = sum(1 for r in results if r.error)
    screened     = total - errors
    passing      = sorted(
        [r for r in results if not r.error and r.score >= min_score],
        key=lambda r: r.score, reverse=True
    )
    all_valid    = sorted(
        [r for r in results if not r.error],
        key=lambda r: r.score, reverse=True
    )

    # ── rating badge colours ──────────────────────────────────────────────────
    def rating_cls(r):
        if r.score >= 6:   return "strong-buy"
        if r.score >= 5:   return "buy"
        if r.score >= 4:   return "watch"
        if r.score >= 2.5: return "weak"
        return "avoid"

    def status_cls(s):
        return {"PASS": "pass", "PARTIAL": "partial",
                "FAIL": "fail", "UNKNOWN": "unknown"}.get(s, "unknown")

    def status_icon(s):
        return {"PASS": "✓", "PARTIAL": "~", "FAIL": "✗", "UNKNOWN": "?"}.get(s, "?")

    def score_bar(score):
        filled = int(score)
        half   = 1 if (score - filled) >= 0.5 else 0
        empty  = 7 - filled - half
        return (
            '<span class="bar-fill">' + '█' * filled + '</span>' +
            ('<span class="bar-half">▒</span>' if half else '') +
            '<span class="bar-empty">' + '░' * empty + '</span>'
        )

    # ── summary cards ─────────────────────────────────────────────────────────
    top_picks    = sum(1 for r in all_valid if r.score >= 5)
    avg_score    = (sum(r.score for r in all_valid) / len(all_valid)) if all_valid else 0

    # ── detail rows for all stocks ────────────────────────────────────────────
    def detail_rows(stock_list):
        rows = []
        for r in stock_list:
            cls  = rating_cls(r)
            rows.append(f"""
        <tr class="stock-row" data-score="{r.score}" data-ticker="{r.ticker}"
            onclick="toggleDetail(this)">
          <td class="td-ticker"><strong>{r.ticker}</strong></td>
          <td class="td-name">{r.name[:40]}</td>
          <td>${r.price:.2f}</td>
          <td>{r.sector}</td>
          <td><span class="score-bar">{score_bar(r.score)}</span> {r.score:.1f}/7</td>
          <td><span class="badge {cls}">{r.rating}</span></td>
          <td class="td-icons">{"".join(
              f'<span class="ci {status_cls(c.status)}" title="{CRITERION_NAMES[i]}: {c.note}">{status_icon(c.status)}</span>'
              for i, c in enumerate(r.criteria)
          )}</td>
        </tr>
        <tr class="detail-row hidden" id="detail-{r.ticker}">
          <td colspan="7">
            <div class="detail-grid">
              {"".join(f'''
              <div class="detail-item {status_cls(c.status)}">
                <div class="detail-header">
                  <span class="detail-icon">{status_icon(c.status)}</span>
                  <span class="detail-name">{CRITERION_NAMES[i]}</span>
                </div>
                <div class="detail-note">{c.note}</div>
              </div>''' for i, c in enumerate(r.criteria))}
            </div>
          </td>
        </tr>""")
        return "\n".join(rows)

    passing_rows = detail_rows(passing)
    all_rows     = detail_rows(all_valid)

    # ── full HTML ─────────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Stock Screener Report — {generated_at}</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  :root {{
    --bg:       #f8f9fb;
    --surface:  #ffffff;
    --border:   #e2e5ea;
    --text:     #1a1d23;
    --muted:    #6b7280;
    --green:    #16a34a;
    --green-bg: #dcfce7;
    --amber:    #b45309;
    --amber-bg: #fef3c7;
    --red:      #dc2626;
    --red-bg:   #fee2e2;
    --blue:     #1d4ed8;
    --blue-bg:  #dbeafe;
    --gray:     #374151;
    --gray-bg:  #f3f4f6;
    --radius:   8px;
  }}

  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    font-size: 14px;
    line-height: 1.5;
  }}

  /* ── layout ── */
  .page {{ max-width: 1200px; margin: 0 auto; padding: 2rem 1.5rem; }}

  /* ── header ── */
  .header {{ margin-bottom: 2rem; }}
  .header h1 {{ font-size: 24px; font-weight: 700; margin-bottom: 4px; }}
  .header .sub {{ color: var(--muted); font-size: 13px; }}

  /* ── summary cards ── */
  .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; margin-bottom: 2rem; }}
  .card {{ background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 1rem; }}
  .card .label {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); margin-bottom: 4px; }}
  .card .value {{ font-size: 26px; font-weight: 700; }}
  .card.green .value {{ color: var(--green); }}
  .card.amber .value {{ color: var(--amber); }}

  /* ── tabs ── */
  .tabs {{ display: flex; gap: 4px; margin-bottom: 1rem; border-bottom: 2px solid var(--border); padding-bottom: 0; }}
  .tab {{ padding: 8px 18px; cursor: pointer; border: none; background: none; font-size: 14px;
          color: var(--muted); border-bottom: 2px solid transparent; margin-bottom: -2px; border-radius: 4px 4px 0 0; }}
  .tab:hover {{ color: var(--text); background: var(--bg); }}
  .tab.active {{ color: var(--blue); border-bottom-color: var(--blue); font-weight: 600; }}

  /* ── filters ── */
  .filters {{ display: flex; gap: 10px; margin-bottom: 1rem; flex-wrap: wrap; align-items: center; }}
  .filters input, .filters select {{
    border: 1px solid var(--border); border-radius: var(--radius);
    padding: 6px 10px; font-size: 13px; background: var(--surface);
    color: var(--text); outline: none;
  }}
  .filters input:focus, .filters select:focus {{ border-color: var(--blue); }}
  .filter-label {{ font-size: 12px; color: var(--muted); }}

  /* ── table ── */
  .table-wrap {{ overflow-x: auto; background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); }}
  table {{ width: 100%; border-collapse: collapse; }}
  th {{
    background: var(--bg); padding: 10px 12px; text-align: left;
    font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em;
    color: var(--muted); border-bottom: 1px solid var(--border);
    cursor: pointer; user-select: none; white-space: nowrap;
  }}
  th:hover {{ color: var(--text); }}
  th .sort-arrow {{ margin-left: 4px; opacity: 0.4; }}
  th.sorted .sort-arrow {{ opacity: 1; color: var(--blue); }}
  td {{ padding: 10px 12px; border-bottom: 1px solid var(--border); vertical-align: middle; }}
  .stock-row {{ cursor: pointer; transition: background 0.1s; }}
  .stock-row:hover {{ background: #f0f4ff; }}
  .stock-row:last-of-type td {{ border-bottom: none; }}
  .td-ticker {{ font-weight: 600; font-size: 14px; white-space: nowrap; }}
  .td-name {{ color: var(--muted); max-width: 220px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}

  /* ── score bar ── */
  .score-bar {{ font-family: monospace; letter-spacing: 1px; }}
  .bar-fill    {{ color: var(--green); }}
  .bar-half    {{ color: var(--amber); }}
  .bar-empty   {{ color: #d1d5db; }}

  /* ── badges ── */
  .badge {{ display: inline-block; padding: 3px 10px; border-radius: 99px; font-size: 11px; font-weight: 600; white-space: nowrap; }}
  .strong-buy {{ background: var(--green-bg); color: var(--green); }}
  .buy        {{ background: #d1fae5;          color: #065f46; }}
  .watch      {{ background: var(--amber-bg);  color: var(--amber); }}
  .weak       {{ background: #ffe4e6;          color: #9f1239; }}
  .avoid      {{ background: var(--red-bg);    color: var(--red); }}

  /* ── criterion icons ── */
  .td-icons {{ white-space: nowrap; }}
  .ci {{
    display: inline-flex; align-items: center; justify-content: center;
    width: 20px; height: 20px; border-radius: 50%;
    font-size: 11px; font-weight: 700; margin-right: 2px;
    cursor: default;
  }}
  .ci.pass    {{ background: var(--green-bg); color: var(--green); }}
  .ci.partial {{ background: var(--amber-bg); color: var(--amber); }}
  .ci.fail    {{ background: var(--red-bg);   color: var(--red); }}
  .ci.unknown {{ background: var(--gray-bg);  color: var(--gray); }}

  /* ── detail row ── */
  .detail-row td {{ background: #f8faff; padding: 12px 16px; }}
  .detail-row.hidden {{ display: none; }}
  .detail-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 8px;
  }}
  .detail-item {{
    padding: 8px 12px; border-radius: 6px; border-left: 3px solid transparent;
    background: var(--surface);
  }}
  .detail-item.pass    {{ border-color: var(--green); }}
  .detail-item.partial {{ border-color: var(--amber); }}
  .detail-item.fail    {{ border-color: var(--red); }}
  .detail-item.unknown {{ border-color: #d1d5db; }}
  .detail-header {{ display: flex; align-items: center; gap: 6px; margin-bottom: 3px; }}
  .detail-icon {{ font-weight: 700; font-size: 13px; }}
  .detail-item.pass    .detail-icon {{ color: var(--green); }}
  .detail-item.partial .detail-icon {{ color: var(--amber); }}
  .detail-item.fail    .detail-icon {{ color: var(--red); }}
  .detail-item.unknown .detail-icon {{ color: var(--muted); }}
  .detail-name {{ font-size: 12px; font-weight: 600; }}
  .detail-note {{ font-size: 12px; color: var(--muted); line-height: 1.4; }}

  /* ── legend ── */
  .legend {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 1.5rem; font-size: 12px; color: var(--muted); }}
  .legend-item {{ display: flex; align-items: center; gap: 5px; }}

  /* ── footer ── */
  .footer {{ margin-top: 2rem; padding-top: 1rem; border-top: 1px solid var(--border);
             font-size: 12px; color: var(--muted); }}

  /* ── tab panels ── */
  .panel {{ display: none; }}
  .panel.active {{ display: block; }}

  @media (max-width: 600px) {{
    .td-name, .td-icons {{ display: none; }}
  }}
</style>
</head>
<body>
<div class="page">

  <!-- header -->
  <div class="header">
    <h1>📈 Stock Screener Report</h1>
    <div class="sub">Generated {generated_at} &nbsp;·&nbsp; Data via Yahoo Finance &nbsp;·&nbsp; Min score shown: {min_score}/7</div>
  </div>

  <!-- summary cards -->
  <div class="summary">
    <div class="card"><div class="label">Tickers screened</div><div class="value">{screened}</div></div>
    <div class="card green"><div class="label">Score ≥ 5/7 (Buy+)</div><div class="value">{top_picks}</div></div>
    <div class="card"><div class="label">Avg score</div><div class="value">{avg_score:.1f}</div></div>
    <div class="card amber"><div class="label">Score ≥ {min_score}/7 shown</div><div class="value">{len(passing)}</div></div>
    <div class="card"><div class="label">Errors / no data</div><div class="value">{errors}</div></div>
  </div>

  <!-- tabs -->
  <div class="tabs">
    <button class="tab active" onclick="showTab('passing', this)">Top picks (score ≥ {min_score})</button>
    <button class="tab" onclick="showTab('all', this)">All stocks</button>
  </div>

  <!-- filters -->
  <div class="filters">
    <span class="filter-label">Filter:</span>
    <input type="text" id="searchBox" placeholder="Search ticker or name…" oninput="applyFilters()" style="width:200px">
    <select id="ratingFilter" onchange="applyFilters()">
      <option value="">All ratings</option>
      <option value="strong-buy">Strong Buy</option>
      <option value="buy">Buy</option>
      <option value="watch">Hold / Watch</option>
      <option value="weak">Weak</option>
      <option value="avoid">Avoid</option>
    </select>
    <select id="minScoreFilter" onchange="applyFilters()">
      <option value="0">Min score: any</option>
      <option value="3">3+</option>
      <option value="4">4+</option>
      <option value="5">5+</option>
      <option value="6">6+</option>
    </select>
  </div>

  <!-- passing tab -->
  <div id="panel-passing" class="panel active">
    <div class="table-wrap">
      <table id="table-passing">
        <thead>
          <tr>
            <th onclick="sortTable('table-passing',0)">Ticker <span class="sort-arrow">↕</span></th>
            <th onclick="sortTable('table-passing',1)">Name <span class="sort-arrow">↕</span></th>
            <th onclick="sortTable('table-passing',2)">Price <span class="sort-arrow">↕</span></th>
            <th onclick="sortTable('table-passing',3)">Sector <span class="sort-arrow">↕</span></th>
            <th onclick="sortTable('table-passing',4)" class="sorted">Score ↓</th>
            <th onclick="sortTable('table-passing',5)">Rating <span class="sort-arrow">↕</span></th>
            <th title="1=Price near low, 2=Insider buy, 3=Volatility, 4=Mkt cap, 5=P/E, 6=Balance sheet, 7=Revenue">Criteria 1–7 ℹ</th>
          </tr>
        </thead>
        <tbody id="tbody-passing">
          {passing_rows}
        </tbody>
      </table>
    </div>
  </div>

  <!-- all stocks tab -->
  <div id="panel-all" class="panel">
    <div class="table-wrap">
      <table id="table-all">
        <thead>
          <tr>
            <th onclick="sortTable('table-all',0)">Ticker <span class="sort-arrow">↕</span></th>
            <th onclick="sortTable('table-all',1)">Name <span class="sort-arrow">↕</span></th>
            <th onclick="sortTable('table-all',2)">Price <span class="sort-arrow">↕</span></th>
            <th onclick="sortTable('table-all',3)">Sector <span class="sort-arrow">↕</span></th>
            <th onclick="sortTable('table-all',4)" class="sorted">Score ↓</th>
            <th onclick="sortTable('table-all',5)">Rating <span class="sort-arrow">↕</span></th>
            <th title="1=Price near low, 2=Insider buy, 3=Volatility, 4=Mkt cap, 5=P/E, 6=Balance sheet, 7=Revenue">Criteria 1–7 ℹ</th>
          </tr>
        </thead>
        <tbody id="tbody-all">
          {all_rows}
        </tbody>
      </table>
    </div>
  </div>

  <!-- legend -->
  <div class="legend">
    <strong>Criteria:</strong>
    <span class="legend-item"><span class="ci pass">✓</span> Pass (1pt)</span>
    <span class="legend-item"><span class="ci partial">~</span> Partial (0.5pt)</span>
    <span class="legend-item"><span class="ci fail">✗</span> Fail (0pt)</span>
    <span class="legend-item"><span class="ci unknown">?</span> No data</span>
    &nbsp;|&nbsp;
    <span class="legend-item"><span class="badge strong-buy">Strong Buy</span> 6–7</span>
    <span class="legend-item"><span class="badge buy">Buy</span> 5</span>
    <span class="legend-item"><span class="badge watch">Hold/Watch</span> 4</span>
    <span class="legend-item"><span class="badge weak">Weak</span> 2.5–3.5</span>
    <span class="legend-item"><span class="badge avoid">Avoid</span> &lt;2.5</span>
    &nbsp;|&nbsp; Click any row to expand criteria detail.
  </div>

  <div class="footer">
    ⚠️ For informational purposes only. Not financial advice. Always verify data before making investment decisions.
  </div>

</div>

<script>
  /* ── tab switching ── */
  function showTab(name, btn) {{
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.getElementById('panel-' + name).classList.add('active');
    btn.classList.add('active');
    applyFilters();
  }}

  /* ── expand / collapse detail row ── */
  function toggleDetail(row) {{
    const ticker = row.dataset.ticker;
    const detail = document.getElementById('detail-' + ticker);
    if (detail) detail.classList.toggle('hidden');
  }}

  /* ── column sort ── */
  const sortState = {{}};
  function sortTable(tableId, colIdx) {{
    const table = document.getElementById(tableId);
    const tbody = table.querySelector('tbody');
    const key   = tableId + ':' + colIdx;
    sortState[key] = !sortState[key];
    const asc = sortState[key];

    // collect stock rows only (skip detail rows)
    const pairs = [];
    const rows  = Array.from(tbody.querySelectorAll('tr'));
    for (let i = 0; i < rows.length; i++) {{
      if (rows[i].classList.contains('stock-row')) {{
        pairs.push({{ stock: rows[i], detail: rows[i+1] }});
      }}
    }}

    pairs.sort((a, b) => {{
      const av = a.stock.cells[colIdx]?.innerText.replace(/[$,]/g,'') || '';
      const bv = b.stock.cells[colIdx]?.innerText.replace(/[$,]/g,'') || '';
      const an = parseFloat(av), bn = parseFloat(bv);
      if (!isNaN(an) && !isNaN(bn)) return asc ? an - bn : bn - an;
      return asc ? av.localeCompare(bv) : bv.localeCompare(av);
    }});

    pairs.forEach(p => {{ tbody.appendChild(p.stock); tbody.appendChild(p.detail); }});

    table.querySelectorAll('th').forEach(th => th.classList.remove('sorted'));
    table.querySelectorAll('th')[colIdx].classList.add('sorted');
    table.querySelectorAll('th')[colIdx].querySelector('.sort-arrow') &&
      (table.querySelectorAll('th')[colIdx].querySelector('.sort-arrow').textContent = asc ? '↑' : '↓');
  }}

  /* ── filters ── */
  function applyFilters() {{
    const q     = document.getElementById('searchBox').value.toLowerCase();
    const rat   = document.getElementById('ratingFilter').value;
    const minSc = parseFloat(document.getElementById('minScoreFilter').value) || 0;

    document.querySelectorAll('.stock-row').forEach(row => {{
      const ticker = row.dataset.ticker?.toLowerCase() || '';
      const name   = row.cells[1]?.innerText.toLowerCase() || '';
      const score  = parseFloat(row.dataset.score) || 0;
      const badge  = row.querySelector('.badge');
      const badgeCls = badge ? badge.className.replace('badge','').trim() : '';

      const matchQ   = !q   || ticker.includes(q) || name.includes(q);
      const matchRat = !rat || badgeCls === rat;
      const matchSc  = score >= minSc;
      const visible  = matchQ && matchRat && matchSc;

      row.style.display = visible ? '' : 'none';
      const detail = document.getElementById('detail-' + row.dataset.ticker);
      if (detail) detail.style.display = visible ? detail.classList.contains('hidden') ? 'none' : '' : 'none';
    }});
  }}
</script>
</body>
</html>"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n{GREEN}HTML report saved → {path}{RESET}")
    print(f"  Open it in any browser: open {path}   (macOS)  or  start {path}   (Windows)")


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
    p.add_argument("--export-html", metavar="FILE.html",
                   help="Save a self-contained HTML report (open in any browser)")
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

    if args.export_html:
        export_html(results, args.export_html, args.min_score)

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
