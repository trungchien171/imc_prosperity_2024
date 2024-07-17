import numpy as np
import pandas as pd
from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List

def get_sma(prices, rate):
    return prices.rolling(rate).mean()

def get_std(prices, rate):
    return prices.rolling(rate).std()

price_history_amethysts = np.array([])

class Trader:
    def __init__(self):
        self.holdings = 0
        self.last_trade = 0

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """ 
        Takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        
        result = {}
        traderData = "SAMPLE"  # Hardcoded traderData
        
        global price_history_amethysts
        
        for product in state.order_depths.keys():
            if product == 'AMETHYSTS':
                    
                start_trading = 2100
                position_limit = 20
                current_position = state.position.get(product,0)
                history_length = 10
                spread = 3
                
                order_depth: OrderDepth = state.order_depths[product]

                price = 0
                count = 0.000001

                for Trade in state.market_trades.get(product, []):
                    price += Trade.price * Trade.quantity
                    count += Trade.quantity
                current_avg_market_price = price / count
                
                price_history_amethysts = np.append(price_history_amethysts, current_avg_market_price)
                if len(price_history_amethysts) >= history_length+1:
                    price_history_amethysts = price_history_amethysts[1:]
                
                orders: list[Order] = []
                
                rate = 20
                m = 2 # of std devs
                    
                if state.timestamp >= start_trading:

                    df_amethysts_prices = pd.DataFrame(price_history_amethysts, columns=['mid_price'])
                    
                    sma = get_sma(df_amethysts_prices['mid_price'], rate).to_numpy()
                    std = get_std(df_amethysts_prices['mid_price'], rate).to_numpy()

                    upper_curr = sma[-1] + m * std
                    upper_prev = sma[-2] + m * std
                    lower_curr = sma[-1] - m * std
                    lower_prev = sma[-2] - m * std
                    print(lower_prev)

                    if len(order_depth.sell_orders) > 0:

                        best_ask = min(order_depth.sell_orders.keys())
                        best_ask_volume = order_depth.sell_orders[best_ask]

                        if price_history_amethysts[-2] > lower_prev and best_ask <= lower_curr and np.abs(best_ask_volume) > 0:
                            print("BUY", product, str(-best_ask_volume) + "x", best_ask)
                            orders.append(Order(product, best_ask, -best_ask_volume))

                    if len(order_depth.buy_orders) != 0:
                        best_bid = max(order_depth.buy_orders.keys())
                        best_bid_volume = order_depth.buy_orders[best_bid]
                       
                        if price_history_amethysts[-2] < upper_prev and best_bid >= upper_curr and best_bid_volume > 0:
                            print("SELL", product, str(best_bid_volume) + "x", best_bid)
                            orders.append(Order(product, best_bid, -best_bid_volume))

                        
                        
                result[product] = orders
                
        conversions = 1  # Hardcoded conversions value
        return result, conversions, traderData