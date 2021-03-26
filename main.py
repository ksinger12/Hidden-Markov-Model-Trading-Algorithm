from HiddenMarkovModel import *
from collections import defaultdict
import pandas as pd
from Trade import *
import numpy as np
import Stationary

class TradingWithHMM(QCAlgorithm):
    '''
    lookback: history lookback period
    risk: max percentage of cash to trade on
    predicted_price: previous day's predicted price
    ratio: ratio between same day predicted and actual price
        - Default is 5%
    cash_amount: amount of cash allocated to each stock
        - Default is even division
    num_regimes: number of regimes to train for - (to be determined by algorithm)
    '''

    def Initialize(self):
        self.SetStartDate(2021, 1, 11)
        self.SetEndDate(2021, 3, 25)
        
        starting_cash = 1000000
        self.SetCash(starting_cash)
        
        #self.tickers = ["AC","CCL","TD"] #difficult test portfolio
        self.tickers = ["ARKK","AAPL", "CRM", "SHOP", "MRNA"]
        
        self.lookback = 30
        self.models = {}
        self.data = defaultdict(list)
        
        self.risk = {}
        self.predicted_price = {}
        self.ratio = {}
        self.cash_amount = {}
        
        # Initializing and Training HMM Models
        for ticker in self.tickers:
            self.AddEquity(ticker, Resolution.Daily)
            
            model_name = "HMM_" + str(ticker)
            num_regimes = 2 
            
            # Initalize & Train
            model = HMM(num_regimes)
            self.data[ticker] = self.historical_data(ticker)
            trend_ratio = [self.data[ticker][j]/self.data[ticker][j-1] for j in range(1, len(self.data[ticker]))] 
            model.train(trend_ratio)
            
            self.models[model_name] = model 
            
            self.risk[model_name] = 0.05
            self.cash_amount[model_name] = starting_cash//len(self.tickers)
            self.ratio[model_name] = 0
            self.predicted_price[model_name] = 0
            

        # Scheduling Trading and Training
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.At(10, 00), self.EveryMarketOpen)
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.At(17, 00), self.AfterClose)


    def EveryMarketOpen(self):
        for ticker in self.tickers:
            # Initialize
            model_name = "HMM_" + str(ticker)
            model = self.models[model_name]
            data_point = self.Securities[ticker].Price
            
            if data_point > 0:
                # HMM Prediction Results
                regime = model.predict_regime(data_point)
                mean = model.get_regime_mean(regime)
                transition_probabilities = model.get_transition_probabilities(regime)
               
                current_stock_price = data_point
                self.ratio[model_name] = self.predicted_price[model_name]/current_stock_price
                
                # Prediction
                trading = Trade(regime, mean, current_stock_price, self.cash_amount[model_name], transition_probabilities, self.risk[model_name], self.Portfolio[ticker].Invested) 
                self.predicted_price[model_name] = trading.predicted_price
                
                amount_to_trade = 0
                number_of_shares = 0
                
                # Buying/Selling
                if trading.should_buy():
                    amount_to_trade = trading.bullish()
                    number_of_shares = amount_to_trade//self.Securities[ticker].Price
                elif trading.should_sell():
                    amount_to_trade = trading.bearish()
                    number_of_shares = -amount_to_trade//self.Securities[ticker].Price
                else:
                    print("For 3 Regime System - Stationary Regime")
                    
                # Order
                self.MarketOrder(ticker, number_of_shares)
                
                
    def AfterClose(self):
        difference = defaultdict(list)
        total = 0
        
        for ticker in self.tickers:
            model_name = "HMM_" + str(ticker)
            
            # Prepare for Re-Training
            closing_price = self.Securities[ticker].Close
            
            if closing_price > 0:
                self.data[ticker].pop(0)
                self.data[ticker].append(closing_price)
                
                # Train
                trend_ratio = [self.data[ticker][j]/self.data[ticker][j-1] for j in range(1, len(self.data[ticker]))]
                self.models[model_name].train(trend_ratio)
                
                # Adjusting Risk
                if self.ratio[model_name] < 1.05 and self.ratio[model_name] > 0.95:
                    self.risk[model_name] = self.risk[model_name] + 0.005 #was originally 5%
                elif self.ratio[model_name] >= 1.10 and self.ratio[model_name] <= 0.90:
                    self.risk[model_name] = self.risk[model_name] - 0.003 #was originally 5%
                
                difference[model_name] = abs(self.predicted_price[model_name] - closing_price)
                total = total + difference[model_name]
            
                # Re-Assign Cash 
                for ticker in self.tickers:
                    model_name = "HMM_" + str(ticker)
                    self.cash_amount[model_name] = (difference[model_name]/total)*self.Portfolio.Cash;
            
    def historical_data(self, ticker):
        dataframe = self.History(self.Symbol(ticker), timedelta(self.lookback), Resolution.Daily)
        closing_prices = dataframe["close"].to_list()
        return closing_prices
    
    def make_stationary(self, input_data):
        in_data = np.asarray(input_data)
        stationary_data, method = Stationary.auto_stationary(in_data, method=self.stat_method)
        stationary_data = stationary_data + abs(min(stationary_data)) + in_data.mean()
        stationary_result = Stationary.test_stationarity(stationary_data)
        return stationary_data
