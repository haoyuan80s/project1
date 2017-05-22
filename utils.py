import xgboost as xgb

def RMSLE(P,A):
    return np.sqrt( sum((np.log(P + 1) - np.log(A+1))**2) /len(P) )

def split_data(x, gamma = 0.9):
    """
    split_data into to parts. |part1|/ |part1 + part2| == gamma
    """
    offset = int(len(x) * gamma)
    return x[:offset], x[offset:]

class DataFun:
    def __init__(self, X,y):
        """
        training pairs: X[:,i] -- y[i]
        """
        self.__X = X
        self.__y = y
        self.__N = len(y)
        self.__models = []

    def get_X(self):
        return self.__X
    def get_Y(self):
        return self.__y
    def get_len(self):
        return self.__N
    def get_models(self):
        return self.__models
    
    def train_gradient_boost(self, params, num_boost_round = 50):
        """
        training grad boost with data = X,y. para = para
        """
        print "training GB....."
        dtrain = xgb.DMatrix(self.__X,self.__y)
        model = xgb.train(params, dtrain, num_boost_round = num_boost_round)
        self.__models += [model]

    def turn_gradient_boost(self):
        """ 
        turn gradient boost return  good params 
        """
        return NotImplemented

    def predict(self,X, i = 0):
        model = self.__models[i]
        return model.predict(xgb.DMatrix(X))
