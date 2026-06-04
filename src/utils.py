# a file with useful functions

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq

def moving_average_single(data, period, start_idx):
    if start_idx + period > len(data):
        raise ValueError("Not enough data points left to satisfy the period.")
        
    return np.sum(data[start_idx:start_idx+period]) / period

# or rolling mean
def moving_average(data, period):
    return np.convolve(data, np.ones(period), 'valid') / period

def z_score(data, period, ddof = 1):
    some_ones = np.ones(period)
    
    rolling_mean = np.convolve(data, some_ones / period, mode='valid')
    rolling_sum_sq = np.convolve(data**2, filt, mode='valid')
    rolling_var = (rolling_sum_sq - period * (rolling_mean**2)) / (period - ddof)
    rolling_var = np.maximum(rolling_var, 0)
    rolling_std = np.sqrt(rolling_var)
    
    pad = np.full(period - 1, np.nan)
    
    full_mean = np.concatenate([pad, rolling_mean])
    full_std = np.concatenate([pad, rolling_std])
    
    return (data - full_mean) / (full_std + 1e-9)

# Exponential Moving Average
def ema(data, period):
    data = np.asarray(data, dtype=float)
    alpha = 2 / (period + 1)
    result = np.empty_like(data)
    result[0] = data[0]
    for i in range(1, len(data)):
        result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
    return result

def ema_single(data, period):
    alpha = 2 / (period + 1)
    val = data[0]
    for x in data[1:]:
        val = alpha * x + (1 - alpha) * val
    return val
    
# Returns (upper, mid, lower) arrays aligned to input length. First (period-1) values are NaN.
def bollinger_bands(data, period=20, num_std=2.0):
    data = np.asarray(data, dtype=float)
    mid  = np.concatenate([np.full(period - 1, np.nan), moving_average(data, period)])
    std  = rolling_std(data, period)
    return mid + num_std * std, mid, mid - num_std * std

def vwap(order_depth, levels=3):
    bids = sorted(order_depth.buy_orders.items(),  reverse=True)[:levels]
    asks = sorted(order_depth.sell_orders.items())[:levels]
    all_levels = [(p, abs(q)) for p, q in bids + asks]
    if not all_levels:
        return None
    total_vol = sum(q for _, q in all_levels)
    if total_vol == 0:
        return None
    return sum(p * q for p, q in all_levels) / total_vol

# Range from -1 to 1 
# Positive = buy pressure, negative = sell pressure.
def book_imbalance(order_depth, levels=3):
    bid_vol = sum(abs(q) for _, q in
                  sorted(order_depth.buy_orders.items(), reverse=True)[:levels])
    ask_vol = sum(abs(q) for _, q in
                  sorted(order_depth.sell_orders.items())[:levels])
    total = bid_vol + ask_vol
    return (bid_vol - ask_vol) / total if total > 0 else 0.0

def spread(order_depth):
    if not order_depth.buy_orders or not order_depth.sell_orders:
        return None
    return min(order_depth.sell_orders) - max(order_depth.buy_orders)

# Returns: +1 (bullish cross), -1 (bearish cross), 0 (no change)
# Compares the last two fast/slow MA values to detect a fresh cross
def ma_crossover_signal(data, fast=5, slow=20):
    if len(data) < slow + 1:
        return 0
    fast_ma = moving_average(np.asarray(data, dtype=float), fast)
    slow_ma = moving_average(np.asarray(data, dtype=float), slow)

    n = min(len(fast_ma), len(slow_ma))
    fast_ma, slow_ma = fast_ma[-n:], slow_ma[-n:]
    prev_diff = fast_ma[-2] - slow_ma[-2]
    curr_diff = fast_ma[-1] - slow_ma[-1]
    if prev_diff <= 0 and curr_diff > 0:
        return  1
    if prev_diff >= 0 and curr_diff < 0:
        return -1
    return 0

def momentum(data, period=10):
    data = np.asarray(data, dtype=float)
    if len(data) <= period:
        return 0.0
    return data[-1] - data[-1 - period]

# OLS hedge ratio
def hedge_ratio(y, x):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    beta  = np.cov(x, y)[0, 1] / np.var(x)
    alpha = y.mean() - beta * x.mean()
    return beta, alpha

# Z-score of the pairs spread y - beta*x over a rolling window
def spread_zscore(y, x, beta, period=20):
    spread_series = np.asarray(y, dtype=float) - beta * np.asarray(x, dtype=float)
    return z_score_latest(spread_series, period)

def adf_stat(data, max_lag=1):
    data = np.asarray(data, dtype=float)
    y    = np.diff(data)
    y_lag = data[:-1]
    X = y_lag.reshape(-1, 1)
    # OLS: y = rho * y_lag + e
    rho = np.linalg.lstsq(X, y, rcond=None)[0][0]
    resid = y - X.flatten() * rho
    se    = np.sqrt(resid.var() / (X.flatten() ** 2).sum())
    return (rho - 1) / (se + 1e-12)

# K: strike price
# S: current price of the underlying asset
# T: time of option expiration
# r: continuously compounded risk-free interest rate
# sigma: volitility
def black_scholes(S, K, T, r, sigma, option='call'):
    if T <= 0 or sigma <= 0:
        if option == 'call':
            return max(S - K, 0)
        return max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# Delta: sensitivity of option price to underlying price movement.
def bs_delta(S, K, T, r, sigma, option='call'):
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if option == 'call':
        return norm.cdf(d1)
    return norm.cdf(d1) - 1

# Vega: sensitivity of option price to volatility (per 1.0 sigma move)
def bs_vega(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)

def implied_vol(market_price, S, K, T, r, option='call', tol=1e-6):
    intrinsic = max(S - K, 0) if option == 'call' else max(K - S, 0)
    if market_price <= intrinsic or T <= 0:
        return None
    try:
        return brentq(
            lambda sigma: black_scholes(S, K, T, r, sigma, option) - market_price,
            1e-6, 10.0, xtol=tol
        )
    except ValueError:
        return None

# Returns a value in [0, 1]. Plug directly into black_scholes as T
def time_to_expiry(current_timestamp, total_ticks=1000000):
    return max((total_ticks - current_timestamp) / total_ticks, 0)