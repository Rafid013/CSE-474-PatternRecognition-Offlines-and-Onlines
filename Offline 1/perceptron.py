import pandas as pd


def scaler_mul(series, scaler):
    for i in range(0, series.size):
        series.iloc[i] *= scaler
    return series


class Perceptron:
    def __init__(self, dataframe, class1, class2):
        self.weight = []
        for i in range(0, dataframe.shape[1]):
            self.weight.append(0)
        self.p = 0.5
        self.dataframe = dataframe
        self.class1 = class1
        self.class2 = class2

    def train(self):
        df = self.dataframe
        num_of_feature = df.shape[1] - 1
        num_of_samples = df.shape[0]
        w = pd.Series(self.weight)
        t = 0
        while True:
            print t
            y = []
            corresponding_delta = []
            for i in range(0, num_of_samples):
                xi = df.iloc[i, :num_of_feature]
                xi = xi.append(pd.Series([1], index=[num_of_feature]))
                if df.iloc[i, num_of_feature] == self.class1:
                    delta_xi = -1
                else:
                    delta_xi = 1
                temp = delta_xi*(w.dot(xi))
                if temp >= 0:
                    y.append(xi)
                    corresponding_delta.append(delta_xi)
            for i in range(0, len(y)):
                x = y[i]
                delta_x = corresponding_delta[i]
                temp_vect = scaler_mul(x, delta_x*self.p)
                w = w.sub(temp_vect)
            t += 1
            if len(y) == 0 or t == 100:
                break
        self.weight = w.tolist()

    def decide(self, x):
        w = pd.Series(self.weight)
        x = x.append(pd.Series([1], index=[self.dataframe.shape[1] - 1]))
        temp = w.dot(x)
        if temp > 0:
            return self.class1
        else:
            return self.class2
