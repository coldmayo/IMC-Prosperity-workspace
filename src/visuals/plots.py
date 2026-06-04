import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from utils import moving_average, bollinger_bands, z_score, realized_vol, book_imbalance

OUTPUT_DIR = "plotted_metrics"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def savefig(name):
    path = os.path.join(OUTPUT_DIR, name + ".png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

class DataViz:
    def __init__(self, price_files: dict, trade_files: dict):
        self.price_files = price_files
        self.trade_files = trade_files

    def to_bigdf(self, symbol=None, offset=1_000_000, showhead=False):
        trades, prices = [], []

        for i, (day, filepath) in enumerate(self.trade_files.items()):
            df = pd.read_csv(filepath, sep=';')
            if symbol is not None:
                df = df[df['symbol'] == symbol].copy()
            df['timestamp'] += i * offset
            df['day'] = day
            trades.append(df)

        for i, (day, filepath) in enumerate(self.price_files.items()):
            df = pd.read_csv(filepath, sep=';')
            if symbol is not None:
                df = df[df['product'] == symbol].copy()
            df['timestamp'] += i * offset
            df['day'] = day
            prices.append(df)

        trades_df = pd.concat(trades, ignore_index=True).sort_values('timestamp')
        prices_df = pd.concat(prices, ignore_index=True).sort_values('timestamp')

        if showhead:
            print("TRADES"); print(trades_df.head())
            print("PRICES"); print(prices_df.head())

        return trades_df, prices_df

    def _iter_products(self, prices_df, symbol=None):
        products = [symbol] if symbol else prices_df['product'].unique()
        return products

    def pnl_curve(self, prices_df, symbol=None):
        fig, ax = plt.subplots(figsize=(12, 4))
        for product in self._iter_products(prices_df, symbol):
            df = prices_df[prices_df['product'] == product].copy()
            ax.plot(df['timestamp'], df['profit_and_loss'], label=product)
        ax.set_title('PnL Curve')
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('PnL')
        ax.legend()
        ax.grid(True, alpha=0.3)
        savefig('pnl_curve')

    def price_fair_value(self, prices_df, fair_value=None, ma_period=20, symbol=None):
        for product in self._iter_products(prices_df, symbol):
            df = prices_df[prices_df['product'] == product].copy()
            mid = df['mid_price'].values
            ts = df['timestamp'].values

            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(ts, mid, label='Mid price', lw=1, alpha=0.8)

            if fair_value is not None:
                ax.axhline(fair_value, color='red', ls='--', lw=1.2, label=f'Fair value {fair_value}')

            if len(mid) >= ma_period:
                upper, ma, lower = bollinger_bands(mid, period=ma_period)
                pad = ma_period - 1
                ax.plot(ts[pad:], ma[pad:],    color='orange', lw=1.2, label=f'MA({ma_period})')
                ax.plot(ts[pad:], upper[pad:], color='gray',   lw=0.8, ls='--', label='+2σ')
                ax.plot(ts[pad:], lower[pad:], color='gray',   lw=0.8, ls='--', label='−2σ')
                ax.fill_between(ts[pad:], lower[pad:], upper[pad:], alpha=0.08, color='gray')

            ax.set_title(f'Price + Fair Value — {product}')
            ax.set_xlabel('Timestamp')
            ax.set_ylabel('Price')
            ax.legend()
            ax.grid(True, alpha=0.3)
            savefig(f'price_fair_value_{product}')
            
    def bid_ask_spread(self, prices_df, symbol=None):
        for product in self._iter_products(prices_df, symbol):
            df = prices_df[prices_df['product'] == product].copy()
            df['spread'] = df['ask_price_1'] - df['bid_price_1']

            fig, ax = plt.subplots(figsize=(12, 3))
            ax.plot(df['timestamp'], df['spread'], lw=1, color='steelblue')
            ax.axhline(df['spread'].mean(), color='red', ls='--', lw=1, label='Mean spread')
            ax.set_title(f'Bid-Ask Spread — {product}')
            ax.set_xlabel('Timestamp')
            ax.set_ylabel('Spread (ticks)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            savefig(f'bid_ask_spread_{product}')

    def inventory_vs_time(self, trades_df, symbol=None):
        products = [symbol] if symbol else trades_df['symbol'].unique()
        for product in products:
            df = trades_df[trades_df['symbol'] == product].copy()
            df = df.sort_values('timestamp')
            df['inventory'] = df['quantity'].cumsum()

            fig, ax = plt.subplots(figsize=(12, 3))
            ax.plot(df['timestamp'], df['inventory'], lw=1, color='darkorange')
            ax.axhline(0, color='black', lw=0.8, ls='--')
            ax.set_title(f'Inventory vs Time — {product}')
            ax.set_xlabel('Timestamp')
            ax.set_ylabel('Net position')
            ax.grid(True, alpha=0.3)
            savefig(f'inventory_{product}')

    def rolling_volatility(self, prices_df, period=20, symbol=None):
        for product in self._iter_products(prices_df, symbol):
            df = prices_df[prices_df['product'] == product].copy()
            mid = df['mid_price'].values
            ts = df['timestamp'].values

            log_ret = np.diff(np.log(mid + 1e-9))
            # Rolling std of log returns
            vol = np.array([
                np.std(log_ret[max(0, i - period):i], ddof=1)
                if i >= period else np.nan
                for i in range(1, len(log_ret) + 1)
            ])

            fig, ax = plt.subplots(figsize=(12, 3))
            ax.plot(ts[1:], vol, lw=1, color='purple')
            ax.set_title(f'Rolling Volatility (period={period}) — {product}')
            ax.set_xlabel('Timestamp')
            ax.set_ylabel('σ (log returns)')
            ax.grid(True, alpha=0.3)
            savefig(f'rolling_vol_{product}')

    # values beyond $\pm 1.5$ are highlighted as potential trade signals
    def zscore_plot(self, prices_df, period=20, symbol=None):
        for product in self._iter_products(prices_df, symbol):
            df  = prices_df[prices_df['product'] == product].copy()
            mid = df['mid_price'].values
            ts  = df['timestamp'].values
            z = z_score(mid, period)

            fig, ax = plt.subplots(figsize=(12, 3))
            ax.plot(ts, z, lw=1, color='teal', label='Z-score')
            ax.axhline( 1.5, color='green', ls='--', lw=1, label='+1.5σ')
            ax.axhline(-1.5, color='red',   ls='--', lw=1, label='−1.5σ')
            ax.axhline( 0,   color='black', lw=0.6)
            ax.fill_between(ts, z, 1.5,  where=(z >  1.5), alpha=0.15, color='green')
            ax.fill_between(ts, z, -1.5, where=(z < -1.5), alpha=0.15, color='red')
            ax.set_title(f'Z-Score of Mid Price (period={period}) — {product}')
            ax.set_xlabel('Timestamp')
            ax.set_ylabel('Z-score')
            ax.legend()
            ax.grid(True, alpha=0.3)
            savefig(f'zscore_{product}')

    def rolling_mean(self, prices_df, period_start, period_end, step=10, symbol=None):
        periods = range(period_start, period_end + 1, step)
        for product in self._iter_products(prices_df, symbol):
            df  = prices_df[prices_df['product'] == product].copy()
            mid = df['mid_price'].values
            ts  = df['timestamp'].values

            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(ts, mid, color='lightgray', lw=1, label='Mid price', zorder=0)
            for p in periods:
                if len(mid) >= p:
                    ma  = moving_average(mid, p)
                    pad = p - 1
                    ax.plot(ts[pad:], ma, lw=1.2, label=f'MA({p})', alpha=0.85)
            ax.set_title(f'Rolling Means — {product}')
            ax.set_xlabel('Timestamp')
            ax.set_ylabel('Price')
            ax.legend(fontsize=7, ncol=3)
            ax.grid(True, alpha=0.3)
            savefig(f'rolling_mean_{product}')

    def product_corr_matrix(self, prices_df):
        pivoted = prices_df.pivot_table(
            index='timestamp', columns='product', values='mid_price'
        )
        corr = pivoted.corr()
        fig, ax = plt.subplots(figsize=(max(6, len(corr)), max(5, len(corr) - 1)))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f',
                    linewidths=0.5, ax=ax, vmin=-1, vmax=1)
        ax.set_title('Product Correlation Matrix (mid price)')
        savefig('correlation_matrix')


    def dashboard(self, prices_df, symbol, fair_value=None, ma_period=20, z_period=20):
        df  = prices_df[prices_df['product'] == symbol].copy()
        mid = df['mid_price'].values
        ts  = df['timestamp'].values

        fig = plt.figure(figsize=(14, 12))
        gs  = gridspec.GridSpec(4, 1, hspace=0.45)

        ax0 = fig.add_subplot(gs[0])
        ax0.plot(ts, mid, lw=1, label='Mid price')
        if fair_value:
            ax0.axhline(fair_value, color='red', ls='--', lw=1, label=f'FV {fair_value}')
        if len(mid) >= ma_period:
            upper, ma, lower = bollinger_bands(mid, period=ma_period)
            pad = ma_period - 1
            ax0.plot(ts[pad:], ma[pad:],    color='orange', lw=1, label=f'MA({ma_period})')
            ax0.plot(ts[pad:], upper[pad:], color='gray',   lw=0.7, ls='--')
            ax0.plot(ts[pad:], lower[pad:], color='gray',   lw=0.7, ls='--')
        ax0.set_title(f'{symbol} — Price & Bollinger Bands')
        ax0.legend(fontsize=8); ax0.grid(True, alpha=0.3)

        ax1 = fig.add_subplot(gs[1])
        df['spread'] = df['ask_price_1'] - df['bid_price_1']
        ax1.plot(ts, df['spread'].values, lw=1, color='steelblue')
        ax1.axhline(df['spread'].mean(), color='red', ls='--', lw=1)
        ax1.set_title('Bid-Ask Spread'); ax1.grid(True, alpha=0.3)

        ax2 = fig.add_subplot(gs[2])
        log_ret = np.diff(np.log(mid + 1e-9))
        vol = np.array([
            np.std(log_ret[max(0, i - ma_period):i], ddof=1)
            if i >= ma_period else np.nan
            for i in range(1, len(log_ret) + 1)
        ])
        ax2.plot(ts[1:], vol, lw=1, color='purple')
        ax2.set_title(f'Rolling Volatility (period={ma_period})'); ax2.grid(True, alpha=0.3)

        ax3 = fig.add_subplot(gs[3])
        z = z_score(mid, z_period)
        ax3.plot(ts, z, lw=1, color='teal')
        ax3.axhline( 1.5, color='green', ls='--', lw=1)
        ax3.axhline(-1.5, color='red',   ls='--', lw=1)
        ax3.axhline(0,    color='black', lw=0.6)
        ax3.fill_between(ts, z, 1.5,  where=(z >  1.5), alpha=0.12, color='green')
        ax3.fill_between(ts, z, -1.5, where=(z < -1.5), alpha=0.12, color='red')
        ax3.set_title(f'Z-Score (period={z_period})'); ax3.grid(True, alpha=0.3)

        fig.suptitle(f'Dashboard — {symbol}', fontsize=13, y=1.01)
        savefig(f'dashboard_{symbol}')

# Basic use
if __name__ == "__main__":
    price_files = {
        -1: "data/prices_round1_day-1.csv",
         0: "data/prices_round1_day0.csv",
         1: "data/prices_round1_day1.csv",
    }
    
    trade_files = {
        -1: "data/trades_round1_day-1.csv",
         0: "data/trades_round1_day0.csv",
         1: "data/trades_round1_day1.csv",
    }

    viz = DataViz(price_files, trade_files)
    trades_df, prices_df = viz.to_bigdf(symbol="RAINFOREST_RESIN")

    viz.dashboard(prices_df, symbol="RAINFOREST_RESIN", fair_value=10000)
    viz.pnl_curve(prices_df)
    viz.zscore_plot(prices_df, period=20)
    viz.rolling_mean(prices_df, period_start=5, period_end=50, step=5)
    viz.product_corr_matrix(prices_df)  # load without symbol filter for this one