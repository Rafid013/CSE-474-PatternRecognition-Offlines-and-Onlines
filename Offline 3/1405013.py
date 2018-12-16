from channel_equalizer import ChannelEqualizer

coefficients = []

for i in range(1, 3):
    coefficient = input("Coefficient " + str(i) + "\n")
    coefficients.append(float(coefficient))

noise_mean = float(input("Noise Mean?\n"))
noise_variance = float(input("Noise Variance?\n"))

file = open('train.txt', 'r')
I = [0]
while True:
    c = file.read(1)
    if c == '':
        break
    I.append(int(c))

ce = ChannelEqualizer(coefficients, noise_mean, noise_variance)
ce.train(I)

file = open('test.txt', 'r')
I = [0]
while True:
    c = file.read(1)
    if c == '':
        break
    I.append(int(c))

y = ce.predict(I)

print(len(I))
print(len(y))

total_accurate = 0.0
total = len(I) - 1
for k in range(1, len(I)):
    if y[k] == I[k]:
        total_accurate += 1.0

print(total_accurate/total)
