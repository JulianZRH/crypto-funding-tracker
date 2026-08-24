# ── Configuration ─────────────────────────────────────────────────────────
# Edit the values below, then run:
#   python crypto_funding.py download
#   python crypto_funding.py simulate

# Which command to run when executing `python crypto_funding.py` with no arguments.
# Options: "download", "simulate", "both"
MODE = "both"

# Lookback period in days (download window; must cover the largest chart window)
DAYS = 180

# Chart windows in days: one chart per asset per entry.
# With ASSET = "BOTH" and [180, 30] you get 4 charts (BTC/ETH x 180d/30d).
CHART_DAYS = [30, 180]

# Initial investment amount (USD) for the simulation
INVESTMENT = 100_000

# Show a table of the annualized percentage yield (APY) per day.
# Pairs listed in APY_TABLE_PAIRS in crypto_funding.py (default: BTC Perp, Kraken Linear).
SHOW_APY_TABLE = True

# Lookback period in days for the APY table
APY_TABLE_DAYS = 30

# Which asset to simulate: "BTC", "ETH", or "BOTH"
ASSET = "BOTH"

# SQLite database file path
DB = "funding_rates.db"
