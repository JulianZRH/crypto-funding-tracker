# ── Configuration ─────────────────────────────────────────────────────────
# Edit the values below, then run:
#   python crypto_funding.py download
#   python crypto_funding.py simulate

# Which command to run when executing `python crypto_funding.py` with no arguments.
# Options: "download", "simulate", "both"
MODE = "both"

# Lookback period in days
DAYS = 100

# Initial investment amount (USD) for the simulation
INVESTMENT = 100_000

# SQLite database file path
DB = "funding_rates.db"
