import pandas as pd
import perceptron
# import reward_punishment
# import pocket

df = pd.read_csv('Train2.csv', delimiter=',', header=None)
# df = df.iloc[1:, :]
# df = df.reset_index(drop=True)
for j in range(0, df.shape[1]):
    df.iloc[:, j] = pd.to_numeric(df.iloc[:, j])


# num_row = df.shape[0]
# num_col = df.shape[1]
# for j in range(num_row - 1, -1, -1):
#    if df.iloc[j, num_col - 1] == 3:
#        df = df.drop([j])

pr = perceptron.Perceptron(df, 1, 2)
pr.train()

# pr = reward_punishment.Perceptron(df, 1, 2)
# pr.train()

# pr = pocket.Perceptron(df, 1, 2)
# pr.train()


results = []
for s in range(0, df.shape[0]):
    results.append(pr.decide(df.iloc[s, :df.shape[1] - 1]))

tn = fp = tp = fn = 0
for i in range(0, df.shape[0]):
    if df.iloc[i, df.shape[1] - 1] == results[i]:
        if results[i] == 1:
            tn += 1
        else:
            tp += 1
    else:
        if results[i] == 1:
            fn += 1
        else:
            fp += 1

print(tp, tn, fp, fn)
acc = float(tp + tn)/df.shape[0]
print(acc)

df_test = pd.read_csv('Test2.csv', delimiter=',', header=None)
# df_test = df_test.iloc[1:, :]
# df_test = df_test.reset_index(drop=True)
for j in range(0, df_test.shape[1]):
    df_test.iloc[:, j] = pd.to_numeric(df_test.iloc[:, j])


# num_row = df_test.shape[0]
# num_col = df_test.shape[1]
# for j in range(num_row - 1, -1, -1):
#    if df_test.iloc[j, num_col - 1] == 3:
#        df_test = df_test.drop([j])


results = []
for s in range(0, df_test.shape[0]):
    results.append(pr.decide(df_test.iloc[s, :df_test.shape[1] - 1]))

tn = fp = tp = fn = 0
for i in range(0, df_test.shape[0]):
    if df_test.iloc[i, df_test.shape[1] - 1] == results[i]:
        if results[i] == 1:
            tn += 1
        else:
            tp += 1
    else:
        if results[i] == 1:
            fn += 1
        else:
            fp += 1

print(tp, tn, fp, fn)
acc = float(tp + tn)/df_test.shape[0]
print(acc)
