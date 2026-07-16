import copy
import numpy as np

x1 = 3
x2 = x1

x2 = 5
print(x1, x2)

""""""
x3 = np.array([1, 2, 3])
x4 = np.array([4, 5, 6])

x5 = np.hypot(x3, x4)

print(x5)

print("""""")

x6 = np.roll(x3, 1)
print(x3, x6)


x7 = [0, 6, 6, 5, 2, 1]
x8 = np.bincount(x7).argmax()
print("x8")
print(x8)

print("x9")
x9 = np.array([[0, 1], [0, 1], [1, 1], [1, 1], [0, 2], [0, 3], [2, 3]])
print(np.unique(x9, axis=0))
