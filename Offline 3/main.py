from channel_equalizer import ChannelEqualizer

coefficients = []

for i in range(1, 3):
    coefficient = input("Coefficient " + str(i) + "\n")
    coefficients.append(float(coefficient))

noise_mean = float(input("Noise Mean?\n"))
noise_variance = float(input("Noise Variance?\n"))

file = open('Evaluation/train.txt', 'r')
I = [0]
while True:
    c = file.read(1)
    if c == '':
        break
    I.append(int(c))

ce = ChannelEqualizer(coefficients, noise_mean, noise_variance)
ce.train(I)
file.close()

file = open('Evaluation/test.txt', 'r')
I = [0]
while True:
    c = file.read(1)
    if c == '':
        break
    I.append(int(c))
file.close()

y = ce.predict(I)

total_accurate = 0.0
total = len(I) - 1
output_bit_string = ""
for k in range(1, len(I)):
    if y[k] == I[k]:
        total_accurate += 1.0
    output_bit_string = output_bit_string + str(y[k])
file = open('out.txt', 'w')
file.write(output_bit_string)
file.close()

print("Accuracy = " + str(total_accurate*100/total))
