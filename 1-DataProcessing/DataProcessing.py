import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('data/Data.csv')

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1]. values

lineSeperator = "!-----------------------------------------------------------------!\n"

print("Data Representation\n")
print(lineSeperator)
print("Feature Metrix\n")
print(x)
print(lineSeperator)
print("Dependent Vector\n")
print(y)