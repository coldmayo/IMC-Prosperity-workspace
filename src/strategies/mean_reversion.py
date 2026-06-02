from datamodel import Order, TradingState
import numpy as np
import sys
import os

sys.path.append('../')
from utils import *

# Logger, I think we need it for the backtester?
class Logger:
    def __init__(self):
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects, sep=" ", end="\n"):
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state, orders, conversions, trader_data):
        print(json.dumps({
            "timestamp": state.timestamp,
            "orders": {p: [[o.symbol, o.price, o.quantity] for o in v] for p, v in orders.items()},
            "conversions": conversions,
            "trader_data": trader_data,
            "logs": self.logs,
        }))
        self.logs = ""

logger = Logger()

# Mean Reversion strategy implemented on a single equity

class Trader:
    def __init__(self):
        self.equity = "KELP"
        self.limit = 50
        
        self.orders = {}
        self.conversions = 0
        self.traderData = "SAMPLE"

        self.eq_buy_order = 0
        self.eq_sell_order = 0
        self.eq_position = 0

        self.price_history = []
        
    def get_position(self, state: TradingState, product: str) -> int:
        return state.position.get(product, 0)

    def max_buy_capacity(self, product: str, position: int, buy_order_volume: int) -> int:
        return self.position_limits[product] - position - buy_order_volume

    def max_sell_capacity(self, product: str, position: int, sell_order_volume: int) -> int:
        return self.position_limits[product] + position - sell_order_volume

    def best_ask(self, order_depth: OrderDepth) -> Optional[int]:
        return min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

    def best_bid(self, order_depth: OrderDepth) -> Optional[int]:
        return max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None

    def get_mid_price(self, order_depth: OrderDepth) -> Optional[float]:
        best_bid = self.best_bid(order_depth)
        best_ask = self.best_ask(order_depth)

        if best_bid is None or best_ask is None:
            return None

        return (best_bid + best_ask) / 2

    def send_buy_order(self, price, qty, msg=None):
        self.orders[self.equity].append(Order(self.equity, price, qty))
        self.eq_buy_orders += qty
        if msg: logger.print(msg)

    def send_sell_order(self, price, qty, msg=None):
        # qty should be negative
        self.orders[self.equity].append(Order(self.equity, price, qty))
        self.eq_sell_orders += abs(qty)
        if msg: logger.print(msg)
        
    def search_buys(self):
        order_depth = state.order_depths[self.equity]
        for ask, amount in sorted(order_depth.sell_orders.items()):
            if ask <= acceptable_price:
                capacity = self.limit - self.eq_position - self.eq_buy_orders
                size = min(capacity, -amount)
                if size > 0:
                    self.send_buy_order(ask, size, f"BUY {size} @ {ask}")
                    
    def search_sells(self):
        order_depth = state.order_depths[self.equity]
        for bid, amount in sorted(order_depth.buy_orders.items(), reverse=True):
            if bid >= acceptable_price:
                capacity = self.limit + self.eq_position - self.eq_sell_orders
                size = min(capacity, amount)
                if size > 0:
                    self.send_sell_order(bid, -size, f"SELL {size} @ {bid}")
                    
    def trade_eq(self):
        # make sure price history is big enough to even do anything
        
        mid = self.get_mid_price(state)
        if mid is None:
            return
            
        self.price_history.append(mid)

        if len(self.price_history) < 20:
            return

        # Strategy starts now:
        ma = moving_average(np.array(self.price_history), 20)[-1]
        std = np.std(self.price_history[-20:])
        z = (mid - ma) / std if std > 0 else 0

        if z < -1.5:
            self.search_buys(state, mid)
        elif z > 1.5:
            self.search_sells(state, mid)

        max_buy = self.limit - self.eq_position - self.eq_buy_orders
        max_sell = self.limit + self.eq_position - self.eq_sell_orders

        if max_buy > 0:
            self.send_buy_order(int(mid) - 2, max_buy, f"PASSIVE BUY {max_buy} @ {int(mid)-2}")
        if max_sell > 0:
            self.send_sell_order(int(mid) + 2, -max_sell, f"PASSIVE SELL {max_sell} @ {int(mid)+2}")
        
    def reset_state(self):
        self.orders = {self.equity: []}
        self.conversions = 0
        self.eq_position = self.get_position()
        self.eq_buy_orders = 0
        self.eq_sell_orders = 0

    def save_state(self):
        return json.dumps({"price_history": self.price_history[-100:]})  # cap at 100

    def load_state(self, state):
        if state.traderData:
            try:
                data = json.loads(state.traderData)
                self.price_history = data.get("price_history", [])
            except:
                self.price_history = []
        
    def run(self, state: TradingState):
        self.load_state(state)
        self.reset_state(state)
        self.trade_eq(state)
        self.traderData = self.save_state()
        
        logger.flush(state, self.orders, self.conversions, self.traderData)
        return self.orders, self.conversions, self.traderData