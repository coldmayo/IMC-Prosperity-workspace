# Premade strategies
These are meant to be building blocks to build upon once we reach the competition
See some more here: https://www.quantconnect.com/learning/articles/investment-strategy-library

## Trend Following
- Go long when a short-term MA (e.g. 5-period) crosses above a long-term MA (e.g. 20-period)
- Go short on the reverse crossover
- Works well on assets with persistent directional moves

## Mean Reversion
### Bollinger Band Reversion
- Calculate a rolling mean and ±2 standard deviation bands
- Buy when price dips below the lower band, sell when it exceeds the upper band
- Works best on range-bound, low-volatility assets

### Z-Score Trading
- Compute z = (price - rolling_mean) / rolling_std
- Enter long when z < -1.5, short when z > +1.5, close near z = 0
- A clean, parameterizable version of mean reversion

## Market Making
### Passive Quote Strategy
- Post bids slightly below and asks slightly above the fair value (buying at 100 and then selling at 100.5)
- Best for envirenments with high volume but low volatility
    - You want a constant stream of buyers and sellers hitting your quotes so you can turn over your inventory rapidly

### Skewed Quoting
- Adjust bid/ask prices based on your current inventory position
- If long, skew quotes lower to encourage selling off excess inventory

## Statistical Arbitrage
### Pairs Trading
- Find two correlated assets (check Prosperity's product list for related goods)
- Trade the spread when it diverges beyond a threshold, expecting reversion
- Fit a hedge ratio using OLS regression

### Basket Arbitrage
- If a product's price should relate to a basket of inputs, trade deviations from that relationship

## Order Book Microstructure
### Order Book Imbalance
- If bid volume >> ask volume, price is likely to move up — go long
- Signal: imbalance = (bid_qty - ask_qty) / (bid_qty + ask_qty)

### VWAP / Fair Value Estimation
- Estimate fair value from the order book mid, VWAP, or a weighted combination
- Trade aggressively when market price deviates from your fair value estimate