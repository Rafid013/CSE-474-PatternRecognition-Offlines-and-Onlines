import pandas as pd
import numpy as np


def scaler_mul(series, scaler):
    ret_series = pd.Series([])
    for i in range(0, series.size):
        temp = series.iloc[i]*scaler
        ret_series = ret_series.append(pd.Series([temp], index=[i]))
    return ret_series


class Perceptron:
    def __init__(self, dataframe):
        self.p = 0.5
        self.dataframe = dataframe
        self.num_of_feature = dataframe.shape[1] - 1
        self.num_of_row = dataframe.shape[0]
        self.class_values = dataframe.iloc[:, self.num_of_feature].unique()
        self.num_of_class = len(self.class_values)
        w = pd.Series([])
        for i in range(0, self.num_of_class):
            w_temp = []
            for j in range(0, self.num_of_feature + 1):
                w_temp.append(0)
            w = w.append(pd.Series(w_temp))
        self.w = w.reset_index(drop=True)

    def train(self):
        df = self.dataframe
        nf = self.num_of_feature
        nr = self.num_of_row
        class_values = self.class_values
        num_of_class = self.num_of_class
        w = self.w

        zero_series = []
        for i in range(0, nf + 1):
            zero_series.append(0)
        zero_series = pd.Series(zero_series)

        x = []

        for i in range(0, nr):
            xi = df.iloc[i, :nf]
            xi = xi.append(pd.Series([1], index=[nf]))
            xi_neg = scaler_mul(xi, -1)

            class_value = df.iloc[i, nf]
            for j in range(0, num_of_class):
                if j != np.where(class_values == class_value)[0][0]:
                    x_temp = pd.Series([])
                    for k in range(0, num_of_class):
                        if k == np.where(class_values == class_value)[0][0]:
                            if k == 0:
                                x_temp = xi
                            else:
                                x_temp = pd.concat([x_temp, xi], ignore_index=True)
                        elif k == j:
                            if k == 0:
                                x_temp = xi_neg
                            else:
                                x_temp = pd.concat([x_temp, xi_neg], ignore_index=True)
                        else:
                            if k == 0:
                                x_temp = zero_series
                            else:
                                x_temp = pd.concat([x_temp, zero_series], ignore_index=True)
                    x.append(x_temp)

        t = 0
        while True:
            print t
            y = 0
            for i in range(0, len(x)):
                temp = x[i].dot(w)
                if temp <= 0:
                    y += 1
                    w = w.add(scaler_mul(x[i], self.p))
            t += 1
            if y == 0 or t == 50:
                break
        self.w = w

    def decide(self, x):
        w = self.w
        weights = []
        vals = []
        x = x.append(pd.Series([1], index=[self.dataframe.shape[1] - 1]))

        j = 0
        n = self.num_of_feature + 1
        for i in range(0, self.num_of_class):
            temp = w.iloc[j:(j + n)]
            temp = temp.reset_index(drop=True)
            weights.append(temp)
            j = j + n

        for i in range(0, self.num_of_class):
            val = x.dot(weights[i])
            vals.append(val)

        class_idx = vals.index(max(vals))
        return self.class_values[class_idx]


df_ = pd.read_csv('Train.txt', delimiter='\s+', header=None, skiprows=1)

df_ = df_.iloc[1:, :]
df_ = df_.reset_index(drop=True)
for l in range(0, df_.shape[1]):
    df_.iloc[:, l] = pd.to_numeric(df_.iloc[:, l])
p = Perceptron(df_)
p.train()


df_test = pd.read_csv('Test.txt', delimiter='\s+', header=None)
df_test = df_test.iloc[1:, :]
df_test = df_test.reset_index(drop=True)
for l in range(0, df_test.shape[1]):
    df_test.iloc[:, l] = pd.to_numeric(df_test.iloc[:, l])

results = []
for s in range(0, df_test.shape[0]):
    results.append(p.decide(df_test.iloc[s, :df_test.shape[1] - 1]))

true_res = 0
false_res = 0
for l in range(0, df_test.shape[0]):
    if df_test.iloc[l, df_test.shape[1] - 1] == results[l]:
        true_res += 1
    else:
        false_res += 1

print true_res, false_res
acc = float(true_res)/df_test.shape[0]
print acc
