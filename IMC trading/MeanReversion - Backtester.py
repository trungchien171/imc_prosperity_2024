
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
plt.style.use("seaborn-v0_8")


class MeanReversionBacktester():
    ''' Class for the vectorized backtesting of Bollinger Bands-based trading strategies.
    '''
    
    def __init__(self, symbol, SMA, dev, start, end, costs):
        '''
        Parameters
        ----------
        symbol: str
            ticker symbol (instrument) to be backtested
        SMA: int
            moving window in bars (e.g. days) for SMA
        dev: int
            distance for Lower/Upper Bands in Standard Deviation units
        start: str
            start timestamp for data import
        end: str
            end timestamp for data import
        costs: float
            transaction/trading costs per trade
        '''
        self.symbol = symbol
        self.SMA = SMA
        self.dev = dev
        self.start = start
        self.end = end
        self.costs = costs
        self.results = None
        self.get_data()
        self.prepare_data()
        
    def __repr__(self):
        rep = "MeanReversionBacktester(symbol = {}, SMA = {}, dev = {}, start = {}, end = {})"
        return rep.format(self.symbol, self.SMA, self.dev, self.start, self.end)
        
    def get_data(self):
        df = pd.read_csv("raw tutorial.csv", delimiter = ';')
        columns_to_keep = [col for col in df.columns if col not in ['product', 'timestamp']]
        df_pivoted = df.pivot_table(index='timestamp', columns='product', values=columns_to_keep, aggfunc='mean')
        df_pivoted.columns = ['_'.join(col).strip() for col in df_pivoted.columns.values]
        df_pivoted.reset_index(inplace=True)
        df_pivoted = df_pivoted[["timestamp","mid_price_STARFRUIT"]]
        df_pivoted = df_pivoted.set_index("timestamp")
        df_pivoted['returns'] = np.log(df_pivoted.div(df_pivoted.shift(1)))
        self.data = df_pivoted
        
    def prepare_data(self):
        data = self.data.copy()
        data["SMA"] = data["mid_price_STARFRUIT"].rolling(self.SMA).mean()
        data["Lower"] = data["SMA"] - data["mid_price_STARFRUIT"].rolling(self.SMA).std() * self.dev
        data["Upper"] = data["SMA"] + data["mid_price_STARFRUIT"].rolling(self.SMA).std() * self.dev
        self.data = data
        
    def set_parameters(self, SMA = None, dev = None):
        if SMA is not None:
            self.SMA = SMA
            self.data["SMA"] = self.data["mid_price_STARFRUIT"].rolling(self.SMA).mean()
            self.data["Lower"] = self.data["SMA"] - self.data["mid_price_STARFRUIT"].rolling(self.SMA).std() * self.dev
            self.data["Upper"] = self.data["SMA"] + self.data["mid_price_STARFRUIT"].rolling(self.SMA).std() * self.dev
            
        if dev is not None:
            self.dev = dev
            self.data["Lower"] = self.data["SMA"] - self.data["mid_price_STARFRUIT"].rolling(self.SMA).std() * self.dev
            self.data["Upper"] = self.data["SMA"] + self.data["mid_price_STARFRUIT"].rolling(self.SMA).std() * self.dev
            
    def test_strategy(self):
        data = self.data.copy().dropna()
        data["distance"] = data.mid_price_STARFRUIT - data.SMA
        data["position"] = np.where(data.mid_price_STARFRUIT < data.Lower, 1, np.nan)
        data["position"] = np.where(data.mid_price_STARFRUIT > data.Upper, -1, data["position"])
        data["position"] = np.where(data.distance * data.distance.shift(1) < 0, 0, data["position"])
        data["position"] = data.position.ffill().fillna(0)
        data["strategy"] = data.position.shift(1) * data["returns"]
        data.dropna(inplace = True)
        
        # determine the number of trades in each bar
        data["trades"] = data.position.diff().fillna(0).abs()
        
        # subtract transaction/trading costs from pre-cost return
        data.strategy = data.strategy - data.trades * self.costs
        
        data["c_returns"] = data["returns"].cumsum().apply(np.exp)
        data["c_strategy"] = data["strategy"].cumsum().apply(np.exp)
        self.results = data
       
        perf = data["c_strategy"].iloc[-1] # absolute performance of the strategy
        outperf = perf - data["c_returns"].iloc[-1] # out-/underperformance of strategy
        
        return round(perf, 6), round(outperf, 6)
    
    def plot_results(self):
        if self.results is None:
            print("Run test_strategy() first.")
        else:
            title = "{} | SMA = {} | dev = {} | TC = {}".format(self.symbol, self.SMA, self.dev, self.costs)
            self.results[["c_returns", "c_strategy"]].plot(title=title, figsize=(12, 8))     
   
    def optimize_parameters(self, SMA_range, dev_range):
        combinations = list(product(range(*SMA_range), range(*dev_range)))
        
        # test all combinations
        results = []
        for comb in combinations:
            self.set_parameters(comb[0], comb[1])
            results.append(self.test_strategy()[0])
        
        best_perf = np.max(results) # best performance
        opt = combinations[np.argmax(results)] # optimal parameters
        
        # run/set the optimal strategy
        self.set_parameters(opt[0], opt[1])
        self.test_strategy()
                   
        # create a df with many results
        many_results =  pd.DataFrame(data = combinations, columns = ["SMA", "dev"])
        many_results["performance"] = results
        self.results_overview = many_results
                            
        return opt, best_perf
    