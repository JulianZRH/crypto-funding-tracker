# Crypto Funding Rate Tracker

Download perpetual futures funding rates from multiple exchanges and simulate basis-trade (long spot, short perp) performance.

## Supported Exchanges

| Exchange | Contract Types |
|----------|---------------|
| **Kraken** | Linear (PF_*) and Inverse (PI_*) perpetuals |
| **Binance** | USDT-M and COIN-M perpetuals |
| **Deribit** | BTC, ETH, SOL perpetuals |
| **Bybit** | USDT-M linear perpetuals (incl. USDe) |

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Download funding rates

```bash
python crypto_funding.py download              # last 100 days (default)
python crypto_funding.py download --days 365   # last year
python crypto_funding.py download --db my.db   # custom database path
```

### Simulate basis trade

```bash
python crypto_funding.py simulate                          # defaults: 100 days, $100k
python crypto_funding.py simulate --days 90 --investment 50000
```

This computes compounded returns from funding rate collection and plots a comparison chart across exchanges.

## How It Works

1. **Download** fetches historical funding rates from each exchange's public API and stores them in a SQLite database with a unified schema (`exchange, symbol, timestamp, fundingRate, relativeFundingRate`).

2. **Simulate** reads the stored rates, converts 8-hour rates (Binance, Deribit, Bybit) to hourly equivalents via `r_hour = (1 + r_8h)^(1/8) - 1`, then compounds hourly to show portfolio growth over time.
