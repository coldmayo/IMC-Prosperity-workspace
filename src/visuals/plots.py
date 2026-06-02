import numpy as np
import seaborn as sns

# Plots needed here: PnL Curve, Inventory vs Time, Price + Fair Value Overlay, bid-ask spread over time, Rolling Volatility, Z-Score of Spread, Rolling mean, 

class DataViz():
    def __init__(self, price_files, trade_files):   # data should be pandas dataframe
        self.price_files = price_files
        self.trade_files = trade_files

    # function that takes the trading csvs and price csvs and puts it into two big datasets
    def to_bigdf(self, heads = 10, symbol=None, offset = 0, showhead = False):
        trades = []
        prices = []
        
        for i, (day, data) in enumerate(self.trade_files.items()):
            df = pd.read_csv(pf, sep=';')
            if symbol != None:
                df = df[df['product'] == symbol].copy()
                
            df['timestamp'] += i * offset
            df['day'] = day
            trades.append(df)
            
        trades_df = pd.concat(trades).sort_values('timestamp')
        
        for i, (day, tf) in enumerate(self.price_files.items()):
            df = pd.read_csv(tf, sep=';')
            if symbol != None:
                df = df[df['symbol'] == symbol].copy()
                
            df['timestamp'] += i * offset
            df['day'] = day
            prices.append(df)
            
        prices_df = pd.concat(prices).sort_values('timestamp')

        return trades_df, prices_df

    def product_corr_m(self, prices_df):
        p_prices_df = prices_df.pivot(index="timestamp", values="ask_price_1", columns="product").drop(['bidPrice', 'askPrice'], axis=1)

        correlation_matrix = p_prices_df.corr()

        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title('Product Correlation Matrix')
        plt.show()

    def rolling_mean(self, prices_df, ):
        