import pandas as pd
import math
import random
from sklearn.preprocessing import MinMaxScaler
import time


def logistic_function(x):
    try:
        ans = 1.0 / (1 + math.exp(-x))
    except OverflowError:
        if x > 0:
            ans = 1
        elif x == 0:
            ans = 0.5
        else:
            ans = 0
    return ans


def logistic_function_bar(x):
    return logistic_function(x)*(1 - logistic_function(x))


def relu_func(x):
    if x < 0:
        return 0.01*x
    else:
        return x


def relu_func_bar(x):
    if x < 0:
        return 0.01
    else:
        return 1


def func3(x):
    return math.tanh(x/2.0)


def func3_bar(x):
    return 0.5*(1 - func3(x)*func3(x))


class MultiLayerPerceptron:
    def __init__(self, num_of_layers, node_counts, activation_function, activation_function_der, num_of_features):
        # num_of_layers = hidden + output
        # node_counts = number of nodes in every layer including output layer (num of classes)
        self.nof = num_of_features
        self.w = []
        self.nol = num_of_layers
        self.node_counts = node_counts
        self.af = activation_function
        self.daf = activation_function_der
        self.u = 1

        self.class_vectors = [[]]

        self.k = [num_of_features]

        for nc in node_counts:
            self.k.append(nc)

        self.w.append([])  # dummy append to start the index from 1

        for r in range(1, num_of_layers + 1):
            self.w.append([])
            self.w[r].append([])  # dummy append to start the index from 1
            for j in range(1, self.k[r] + 1):
                self.w[r].append([])
                for k in range(self.k[r - 1] + 1):
                    self.w[r][j].append(random.uniform(0, 1))

        for i in range(1, self.k[num_of_layers] + 1):
            self.class_vectors.append([0]*(self.k[num_of_layers] + 1))
            self.class_vectors[i][i] = 1

    def train(self, features, labels):
        N = features.shape[0]

        ym = []

        for i in range(0, N):
            ym.append(self.class_vectors[labels.iloc[i]])

        temp = [1]*N

        x = pd.concat([pd.DataFrame(temp), features], axis=1, ignore_index=True)

        v = []
        y = []
        delta = []

        for i in range(N):
            v.append([])
            y.append([])
            delta.append([])
            y[i].append(x.iloc[i, :].tolist())
            for r in range(self.nol + 1):
                v[i].append([])
                delta[i].append([])
                if r > 0:
                    y[i].append([])
                for j in range(self.k[r] + 1):
                    v[i][r].append(0)
                    delta[i][r].append(0)
                    if r > 0:
                        if j == 0:
                            y[i][r].append(1)
                        else:
                            y[i][r].append(0)

        iter_constraint = 0
        while True:
            iter_constraint += 1
            if iter_constraint > 50:
                break
            for i in range(0, N):
                for r in range(1, self.nol + 1):
                    for j in range(1, self.k[r] + 1):
                        v[i][r][j] = pd.Series(self.w[r][j]).dot(pd.Series(y[i][r - 1]))
                        y[i][r][j] = self.af(v[i][r][j])

                for j in range(1, self.k[self.nol] + 1):
                    err = y[i][self.nol][j] - ym[i][j]
                    delta[i][self.nol][j] = err*self.daf(v[i][self.nol][j])

                for r in range(self.nol, 1, -1):
                    for j in range(1, self.k[r - 1] + 1):
                        err = 0
                        for k in range(1, self.k[r] + 1):
                            err += delta[i][r][k]*self.w[r][k][j]
                        delta[i][r - 1][j] = err*self.daf(v[i][r - 1][j])

                for r in range(1, self.nol + 1):
                    for j in range(1, self.k[r] + 1):
                        update = pd.Series(y[i][r - 1]).multiply(delta[i][r][j])
                        update = update.multiply(-self.u)
                        w_new = pd.Series(self.w[r][j]).add(update)
                        self.w[r][j] = w_new.tolist()
            J = 0
            for i in range(0, N):
                ei = 0
                for j in range(1, self.k[self.nol] + 1):
                    ei += (ym[i][j] - y[i][self.nol][j])*(ym[i][j] - y[i][self.nol][j])
                ei *= 0.5
                J += ei
            print("Iteration " + str(iter_constraint) + ", Cost " + str(J))
            if J < 10:
                break

    def decide(self, x):
        x = pd.concat([pd.Series([1]), x], axis=0, ignore_index=True)
        y = [x.tolist()]

        for r in range(1, self.nol + 1):
            y.append([])
            y[r].append(1.0)
            for j in range(1, self.k[r] + 1):
                v = pd.Series(self.w[r][j]).dot(pd.Series(y[r - 1]))
                y[r].append(self.af(v))

        max_value_idx = y[self.nol][1:].index(max(y[self.nol][1:]))
        return max_value_idx + 1

        # for i in range(1, self.k[self.nol] + 1):
        #    if self.class_vectors[i][1:] == y[self.nol][1:]:
        #       return i


# df = pd.read_csv('trainNN.txt', delimiter='\s+', header=None)
df = pd.read_csv('NNevaluation/trainNN.txt', delimiter='\s+', header=None)

X = df.iloc[:, :df.shape[1] - 1]
Y = df.iloc[:, df.shape[1] - 1]

min_max_scaler = MinMaxScaler()  # min max scaler
min_max_scaler.fit(X)
X = min_max_scaler.transform(X)

num_of_class = len(Y.unique())

start = time.time()
mlp = MultiLayerPerceptron(2, [8, num_of_class], func3, func3_bar, X.shape[1])
mlp.train(pd.DataFrame(X), Y)
end = time.time()
print("Time to train " + str(end - start))

true_res = false_res = 0

for rn in range(X.shape[0]):
    res = mlp.decide(pd.Series(X[rn, :]))
    if res == Y.iloc[rn]:
        true_res += 1.0
    else:
        false_res += 1.0

accuracy = true_res/(true_res + false_res)
print("Accuracy of training data " + str(accuracy*100))

# test_df = pd.read_csv('testNN.txt', delimiter='\s+', header=None)
test_df = pd.read_csv('NNevaluation/testNN.txt', delimiter='\s+', header=None)
X = test_df.iloc[:, :test_df.shape[1] - 1]
Y = test_df.iloc[:, test_df.shape[1] - 1]

X = min_max_scaler.transform(X)

for rn in range(X.shape[0]):
    res = mlp.decide(pd.Series(X[rn, :]))
    if res == Y.iloc[rn]:
        true_res += 1.0
    else:
        false_res += 1.0

accuracy = true_res/(true_res + false_res)
print("Accuracy of test data " + str(accuracy*100))
