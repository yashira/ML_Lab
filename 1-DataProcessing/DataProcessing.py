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

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

print(lineSeperator)
print("After Data clensing\n")
print(x)