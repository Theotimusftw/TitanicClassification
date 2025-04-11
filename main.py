import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("train.csv")
df = df.drop(columns=['Cabin', 'Name', 'PassengerId', 'Ticket'])
df.fillna({
    "Age": df["Age"].mean(),
    "Embarked": df["Embarked"].mode()[0]
}, inplace=True)

class LogisticRegression:
    
    def __init__(self, learning_rate: float, epochs: int, feature_matrix: pd.DataFrame) -> None: 
        self.learning_rate: float = learning_rate
        self.epochs: int = epochs
        self.weight: np.ndarray = np.zeros((self.X.shape[1], 1))
        self.bias: float = 0
        self.feature_matrix: pd.DataFrame = feature_matrix.copy()
        self.X: np.ndarray = self.feature_matrix.drop(columns="survived").to_numpy()
        self.y: np.ndarray = self.feature_matrix["survived"].to_numpy().reshape(-1, 1)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def train(self):     
        m = self.X.shape[0]
        for _ in range(self.epochs):
            z: np.ndarray = self.X @ self.weight + self.bias
            predicted_percentage = self.sigmoid(z)

            average_loss = -np.mean(self.y * np.log(predicted_percentage + 1e-9) + (1 - self.y) * np.log(1 - predicted_percentage + 1e-9))
            descent_weight = (1 / m) * self.X.T @ (predicted_percentage - self.y)
            descent_bias = (1 / m) * np.sum(predicted_percentage - self.y)

            self.weight -= self.learning_rate * descent_weight
            self.bias -= self.learning_rate * descent_bias

            if _ % 100 == 0:
                print(f"Epoch {_}, Loss: {average_loss:.4f}")
            
    def predict(self):
        return self.sigmoid(self.X @ self.weight + self.bias) >= 0.5


le = LabelEncoder()
le.fit(["S", "C", "Q"])
df["Embarked"] = le.transform(df["Embarked"])
le.fit(["male", "female"])
df["Sex"] = le.transform(df["Sex"])
print(df)

correlation_matrix = df.corr()
print(correlation_matrix)
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.show()
