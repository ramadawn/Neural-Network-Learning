import math

def sigmoid(z):
  return 1 / (1 + math.exp(-z))

x1 = 0.4
x2 = 0.6
w1 = raw_input("enter weight 1 \n :")
w2 = raw_input("enter weight 2 \n :")
bias = raw_input("Enter Bias \n :")
w1 = float(w1)
w2 = float(w2)
bias = float(bias)


z = w1 * x1 + w2 * x2 + bias
print(sigmoid(z))
