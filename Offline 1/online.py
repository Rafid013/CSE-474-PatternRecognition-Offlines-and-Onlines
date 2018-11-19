import pandas as pd
import perceptron
import reward_punishment
import pocket


dfls = pd.read_csv('trainLinearlySeparable.txt', delimiter='\s+', header=None, skiprows=1)
for j in range(0, dfls.shape[1]):
    dfls.iloc[:, j] = pd.to_numeric(dfls.iloc[:, j])

basic = perceptron.Perceptron(dfls, 1, 2)

basic.train()

rp = reward_punishment.Perceptron(dfls, 1, 2)
rp.train()

dfls_test = pd.read_csv('testLinearlySeparable.txt', delimiter='\s+', header=None)
for j in range(0, dfls_test.shape[1]):
    dfls_test.iloc[:, j] = pd.to_numeric(dfls_test.iloc[:, j])


results = []
for s in range(0, dfls_test.shape[0]):
    results.append(basic.decide(dfls_test.iloc[s, :dfls_test.shape[1] - 1]))

tn = fp = tp = fn = 0
for i in range(0, dfls_test.shape[0]):
    if dfls_test.iloc[i, dfls_test.shape[1] - 1] == results[i]:
        if results[i] == 1:
            tn += 1
        else:
            tp += 1
    else:
        if results[i] == 1:
            fn += 1
        else:
            fp += 1

print tp, tn, fp, fn
acc = float(tp + tn)/dfls_test.shape[0]
print acc


results = []
for s in range(0, dfls_test.shape[0]):
    results.append(rp.decide(dfls_test.iloc[s, :dfls_test.shape[1] - 1]))

tn = fp = tp = fn = 0
for i in range(0, dfls_test.shape[0]):
    if dfls_test.iloc[i, dfls_test.shape[1] - 1] == results[i]:
        if results[i] == 1:
            tn += 1
        else:
            tp += 1
    else:
        if results[i] == 1:
            fn += 1
        else:
            fp += 1

print tp, tn, fp, fn
acc = float(tp + tn)/dfls_test.shape[0]
print acc


dfnls = pd.read_csv('trainLinearlyNonSeparable.txt', delimiter='\s+', header=None, skiprows=1)

for j in range(0, dfls.shape[1]):
    dfnls.iloc[:, j] = pd.to_numeric(dfnls.iloc[:, j])

pckt = pocket.Perceptron(dfnls, 1, 2)
pckt.train()

dfnls_test = pd.read_csv('testLinearlyNonSeparable.txt', delimiter='\s+', header=None)
for j in range(0, dfnls_test.shape[1]):
    dfnls_test.iloc[:, j] = pd.to_numeric(dfnls_test.iloc[:, j])


results = []
for s in range(0, dfnls_test.shape[0]):
    results.append(basic.decide(dfnls_test.iloc[s, :dfnls_test.shape[1] - 1]))

tn = fp = tp = fn = 0
for i in range(0, dfnls_test.shape[0]):
    if dfnls_test.iloc[i, dfnls_test.shape[1] - 1] == results[i]:
        if results[i] == 1:
            tn += 1
        else:
            tp += 1
    else:
        if results[i] == 1:
            fn += 1
        else:
            fp += 1

print tp, tn, fp, fn
acc = float(tp + tn)/dfnls_test.shape[0]
print acc
