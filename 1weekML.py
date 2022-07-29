import matplotlib.pyplot as plt
import numpy as np


# Model function f=wx+b
def calculate_model_output(w, b, x):
    m = x.shape
    f_wb = np.zeros(m)

    for i in range(0, len(x)):
        f_wb[i] = w * x[i] + b

    return f_wb


# Cost function j*=((f-y)**2)/m
def cost_function(y, f_wb):
    m = y.shape
    j_wb = 0
    for i in range(0, len(y)):
        j_wb += (f_wb[i] - y[i]) ** 2

    return j_wb / (2 * m)


x_train = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
y_train = np.array([45, 55, 70, 120, 190, 225, 260, 300, 370, 400])

m = len(x_train)
print(f"Number of training examples {m}")

for i in range(0, len(x_train)):
    print(f"(x({i}),y({i})) = ({x_train[i]},{y_train[i]})")

w = 220
b = -19
print(f"w: {w}")
print(f"b: {b}")

pred_f_wb = calculate_model_output(w, b, x_train)

plt.plot(x_train, pred_f_wb, c='r', label='Prediction')

plt.scatter(x_train, y_train, marker='x', c='g', label='Actual')
plt.title("Housing Prices")
plt.ylabel("Price in 1000s Dollars")
plt.xlabel("Size in 100 sq/m")
plt.legend()
plt.show()

j_wb = cost_function(y_train, pred_f_wb)
print(f"Cost function value {j_wb}")

size = 1.5
cost_client_sqft = w * size + b

print(f"\nHouse size is {size} \n House price could be ${int(cost_client_sqft * 10 ** 3)} ")
