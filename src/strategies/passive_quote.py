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

# This is the Passive Quote Strategy implemented
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

        self.base = 2
        self.vol_period = 20
        self.max_skew = 4
        ema_period = 10

    def send_buy_order(self, price, qty, msg=None):
        self.orders[self.equity].append(Order(self.equity, price, qty))
        self.eq_buy_order += qty
        if msg: logger.print(msg)

    def send_sell_order(self, price, qty, msg=None):
        self.orders[self.equity].append(Order(self.equity, price, qty))
        self.eq_sell_order += abs(qty)
        if msg: logger.print(msg)

    def get_position(self, state: TradingState, product: str) -> int:
        return state.position.get(product, 0)

    def fair_value(self, state, type = "simple"):
        depth = state.order_depths[self.equity]
        if not depth.buy_orders or not depth.sell_orders:
            return None
        best_bid = max(depth.eq_buy_orders)
        best_ask = min(depth.eq_sell_orders)

        mid = 0

        if type == "simple":
            mid = (best_bid + best_ask) / 2
            return mid
            
        elif type == "VWAP":
            mid = vwap(depth, levels=3)
            self.price_history.append(mid)
            if len(self.price_history) < self.ema_period:
                return mid
            return ema_single(self.price_history, self.ema_period)

    def spead_selector(self, state):
        if len(self.price_history) < self.vol_period:
            return self.base_spread
        vol = realized_vol(self.price_history, self.vol_period)
        if vol is None or vol != vol:
            return self.base_spread
        return max(self.base_spread, int(vol * 15))

    def skew(self):
        return -self.max_skew * (self.eq_position / self.limit)

    def aggressive_fill(self, fair, depth):
        for ask in sorted(depth.sell_orders):
            if ask >= fair: break
            size = min(self.cap_buy(), -depth.sell_orders[ask])
            if size > 0: self.send_buy_order(ask, size)

        for bid in sorted(depth.buy_orders, reverse=True):
            if bid <= fair: break
            size = min(self.cap_sell(), depth.buy_orders[bid])
            if size > 0: self.send_sell_order(bid, -size)

    def cap_buy(self):
        return self.limit - self.eq_position - self.eq_buy_order

    def cap_sell(self):
        return self.limit + self.eq_position - self.eq_sell_order

    def trade_eq(self, state):
        if self.equity not in state.order_depths:
            return

        depth = state.order_depths[self.equity]
        fair  = self.fair_value(state)
        if fair is None:
            return

        # Don't trade uif jumps are too extreme
        imbalance = book_imbalance(depth, levels=3)
        if abs(imbalance) > 0.8:
            return

        self.aggressive_fill(fair, depth)

        half  = self.spread_selector()
        skew  = self.skew()
        buy_p = round(fair - half + skew)
        sel_p = round(fair + half + skew)

        # Undercut any existing competing quotes
        comp_bids = [b for b in depth.buy_orders  if b < fair]
        comp_asks = [a for a in depth.sell_orders if a > fair]
        if comp_bids: buy_p = max(buy_p, max(comp_bids) + 1)
        if comp_asks: sel_p = min(sel_p, min(comp_asks) - 1)

        # never cross the spread
        if buy_p >= sel_p:
            buy_p, sel_p = round(fair) - 1, round(fair) + 1

        cap_buy  = self.cap_buy()
        cap_sell = self.cap_sell()
        if cap_buy  > 0: self.send_buy_order(buy_p,  cap_buy)
        if cap_sell > 0: self.send_sell_order(sel_p, -cap_sell)

    def reset_state(self, state):
        self.orders = {self.equity: []}
        self.conversions = 0
        self.eq_position = self.get_position()
        self.eq_buy_order = 0
        self.eq_sell_order = 0

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