from typing import *
from datamodel import *

# refactor
MAX_QUANT = 20

class RollingAverage:
    def __init__(self, window_size):
        self.window_size = window_size
        self.values = [0] * window_size
        self.sum = 0
        self.idx = 0
    
    def add_value(self, value):
        self.sum -= self.values[self.idx]
        self.sum += value
        self.values[self.idx] = value
        self.idx = (self.idx + 1) % self.window_size
    
    def get_average(self):
        return self.sum / min(len(self.values), self.window_size)


class ExponentialMovingAverage:
    def __init__(self, window_size):
        self.window_size = window_size
        self.alpha = 2 / (window_size + 1)
        self.value = 0
    
    def add_value(self, value):
        self.value = self.alpha * value + (1 - self.alpha) * self.value
    
    def get_average(self):
        return self.value
class Trader:
    profits: int
    ENABLED = True
    STATE_COLUMN_LOGGED = False
    
    def __init__(self): 
        self.ENABLED=True
        self.profits=0
    
    def prefix_print(self, string: str, prefix: str="LOG"):
        print(f"\n{prefix}: {string}\n")

    def to_json(self, o: Any) -> str:
        return json.dumps(o, cls=ProsperityEncoder)

    def log_state(self, state: TradingState):
        if not self.ENABLED:
            return
        if not self.STATE_COLUMN_LOGGED:
            self.prefix_print("timestamp;product;buy_depth;sell_depth;market_trades;own_trades;position;observations,profits", "STATE")
            self.STATE_COLUMN_LOGGED = True
        timestamp = str(state.timestamp)
        products = list(state.listings.keys())

        for product in products:
            # Order depths
            buy_depth = "{}"
            sell_depth = "{}"
            if state.order_depths.get(product) is not None:
                buy_depth =  self.to_json(state.order_depths[product].buy_orders)
                sell_depth = self.to_json(state.order_depths[product].sell_orders)
            
            # Market trades
            market_trades = "[]"
            if state.market_trades.get(product) is not None:
                market_trades = self.to_json(state.market_trades[product])
            
            # Own trades 
            own_trades = "[]"
            if state.own_trades.get(product) is not None:
                own_trades = self.to_json(state.own_trades[product])

            # Positions
            position = "0"
            if state.position.get(product) is not None:
                position = self.to_json(state.position[product])
                
            # Observations
            observations = "0"
            if state.observations.get(product) is not None:
                observations = self.to_json(state.observations[product])

            row = [timestamp, product, buy_depth, sell_depth, market_trades, own_trades, position, observations, str(self.profits)]
            self.prefix_print(';'.join(row), "STATE")

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        # Log
        self.log_state(state)
        
        # Trade builder
        trades = {}
        for product, depth in state.order_depths.items():
            # Profits
            if state.own_trades.get(product) is not None:
                for trade in state.own_trades[product]:
                    self.profits -= trade.quantity * trade.price

            # try simple +1 buy price and -1 sell price
            # Max buy price
            max_buy = max(depth.buy_orders.keys())
            max_buy_vol = depth.buy_orders[max_buy]

            min_sell = min(depth.sell_orders.keys())
            min_sell_vol = depth.sell_orders[min_sell]

            # Our inventory
            position = 0 if state.position.get(product) is None else state.position[product]

            trades[product] = []

            if product == "STARFRUIT":
                if position <= 0:
                    # buy
                    buy_order = Trade(product, max_buy, -position)
                    trades[product].append(buy_order)
                else:
                    # sell
                    sell_order = Trade(product, min_sell, -position)
                    trades[product].append(sell_order)
            else:
                # Is product == symbol ? TODO: check
                if max_buy < 10000:
                    buy_order = Trade(product, min_sell, MAX_QUANT - position)
                    trades[product].append(buy_order)
                if min_sell >= 10000:
                    sell_order = Trade(product, max_buy, -MAX_QUANT - position)
                    trades[product].append(sell_order)
        return trades