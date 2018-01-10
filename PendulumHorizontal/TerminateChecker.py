eps = 1e-2

class TerminateChecker():
    
    def __init__(self, start_value, alpha, window_size, no_profit_max_times):
        self.ema = start_value
        self.alpha = alpha
        
        self.last_n_values = [start_value]
        self.n = window_size
        self.ma = start_value / self.n
        
        self.no_profit_max_times = no_profit_max_times
        self.no_profit_times = 0
        self.best_ema = -float('Inf')
        self.best_ma = -float('Inf')
    
    def get_ema(self, current):
        return self.ema  
    
    def get_ma(self):
        return self.ma
        
    def change_ema(self, value):
        self.ema = self.ema * (1 - self.alpha) + self.alpha * value
        self.best_ema = max(self.best_ema, self.ema)
    
    def change_ma(self, value):
        if(len(self.last_n_values) == self.n):
            self.ma = self.ma - self.last_n_values[0] / self.n + value / self.n
            self.last_n_values.pop(0)
            self.last_n_values.append(value)
            self.best_ma = max(self.best_ma, self.ma)
        else:
            self.last_n_values.append(value)
            self.ma += value / self.n
            
    def terminate_check(self, is_ema=True):
        if(self.no_profit_times > self.no_profit_max_times):
            return True
        if(is_ema):
            if(self.ema < self.best_ema):
                self.no_profit_times += 1
            else:
                self.no_profit_times = 0
        else:
            if(self.ma < self.best_ma):
                self.no_profit_times += 1
            else:
                self.no_profit_times = 0