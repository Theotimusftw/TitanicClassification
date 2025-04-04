import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple

df = pd.read_csv("train.csv")
df = df.drop(columns='Cabin')

print(df.isnull().sum())
df["Age"] = df["Age"].fillna(df["Age"].mean(), inplace=True)
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode(), inplace=True)
df["Age"] = df.fillna({"Age": df["Age"].mean()}, inplace=True)

print(df)
print(df.isnull().sum())

def LinearRegression(x_values: list[int], y_values: list[int]) -> Tuple[float, float]:

    def sum_x(x_values):
        return sum(x_values)
    def sum_y(y_values):
        return sum(y_values)
    def sum_x_squared(x_values):
        return sum(list(map(lambda x: x*x, x_values)))
    def sum_xy(x_values, y_values):
        n = 0
        for i, k in zip(x_values, y_values):
            n += i * k
        return n

    slope = len(x_values) * sum_xy(x_values, y_values) - sum_x(x_values) * sum_y(y_values)
    _ = len(x_values) * sum_x_squared(x_values) - sum_x(x_values) ** 2
    slope /= _

    y_intercept = sum_y(y_values) * sum_x_squared(x_values) - sum_x(x_values) * sum_xy(x_values, y_values)
    _ = len(x_values) * sum_x_squared(x_values) - sum_x(x_values) ** 2
    y_intercept /= _

    return slope, y_intercept

