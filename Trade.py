class Trade():
    '''
    regime: the regime of the predicted data point
    mean: mean of the regime
    current_price: the current stock price
    cash: amount of cash associated with the stock
    risk: max percentage of cash amount willing to trade with
    number_of_regimes: number of regimes
    amount_invested: has cash been invested (do we own shares of this stock)
    predicted_price: tomorrow's predicted stock price
        - current stock price * the mean of the regime the stock is in
    regime_probability: probability of being in the current regime
    '''
    
    def __init__(self, regime, mean, current_price, cash, transition_probabilities, risk, amount_invested):
        self.predicted_regime = regime[0]
        self.mean = mean
        self.current_price = current_price
        self.cash = cash 
        self.risk = risk
        self.number_of_regimes = len(transition_probabilities[0])
        self.amount_invested = amount_invested 
        
        self.predicted_price = self.current_price*self.mean[0]
        self.regime_probability = transition_probabilities[0][self.predicted_regime]
    
    def should_buy(self):
        return self.predicted_price[0] > self.current_price and self.predicted_regime == 1
    
    def should_sell(self):
        return  self.predicted_price[0] <= self.current_price and self.predicted_regime == 0 
        
    def bullish(self):
        amount_to_invest = 0
        if self.cash > 0: 
            if (self.regime_probability >= 0.5 and self.number_of_regimes == 3) or (self.regime_probability >= 0.7 and self.number_of_regimes == 2):
                amount_to_invest = self.risk*self.cash
            elif (self.regime_probability >= 0.4 and self.number_of_regimes == 3) or (self.regime_probability >= 0.6 and self.number_of_regimes == 2):
                amount_to_invest = 0.5*self.risk*self.cash
            else:
                amount_to_invest = 0.1*self.risk*self.cash

            if amount_to_invest > self.cash:
                amount_to_invest = self.cash
        
        return amount_to_invest
        
             
    def bearish(self):
        amount_to_sell = 0
        if self.amount_invested == True:
            if (self.regime_probability >= 0.5 and self.number_of_regimes == 3) or (self.regime_probability >= 0.7 and self.number_of_regimes == 2):
                amount_to_sell = self.risk*self.cash
            elif (self.regime_probability >= 0.4 and self.number_of_regimes == 3) or (self.regime_probability >= 0.6 and self.number_of_regimes == 2):
                amount_to_sell = 0.5*self.risk*self.cash
            else:
                amount_to_sell = 0.1*self.risk*self.cash
                
            if amount_to_sell > self.cash:
                amount_to_sell = self.cash

        return amount_to_sell
