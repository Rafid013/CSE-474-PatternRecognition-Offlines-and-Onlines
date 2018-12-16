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
        self.p = 1
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
            y = 0
            print(t)
            for i in range(0, num_of_samples):
                xi = df.iloc[i, :num_of_feature]
                xi = xi.append(pd.Series([1], index=[num_of_feature]))
                temp = w.dot(xi)
                if df.iloc[i, num_of_feature] == self.class1 and temp <= 0:
                    temp_vect = scaler_mul(xi, self.p)
                    w = w.add(temp_vect)
                    y += 1
                elif df.iloc[i, num_of_feature] == self.class2 and temp >= 0:
                    temp_vect = scaler_mul(xi, self.p)
                    w = w.sub(temp_vect)
                    y += 1
            t += 1
            if y == 0:
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
