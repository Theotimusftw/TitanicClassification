import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pickle

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
        self.bias: float = 0
        self.feature_matrix: pd.DataFrame = feature_matrix.copy()
        self.X: np.ndarray = self.feature_matrix.drop(columns="Survived").to_numpy()
        self.y: np.ndarray = self.feature_matrix["Survived"].to_numpy().reshape(-1, 1)
        self.weight: np.ndarray = np.zeros((self.X.shape[1], 1))


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
            
    def predict(self, X=None):
        if X is None:
            X = self.X
        elif isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        return self.sigmoid(X @ self.weight + self.bias) >= 0.5

    def evaluate(self, X=None, y=None):    
        if X is None:
            X = self.X
        elif isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if y is None:
            y = self.y
        elif isinstance(y, pd.Series):
            y = y.to_numpy().reshape(-1, 1)
        predictions = self.predict(X)
        return np.mean(predictions == y)

    def save(self, filepath: str):
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath: str):
        with open(filepath, "rb") as f:
            return pickle.load(f)

le = LabelEncoder()
le.fit(["S", "C", "Q"])
df["Embarked"] = le.transform(df["Embarked"])
le.fit(["male", "female"])
df["Sex"] = le.transform(df["Sex"])

scaler = StandardScaler()
numerical_cols = ["Age", "Fare", "SibSp", "Parch"]
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
print(df)

test_df = pd.read_csv("test.csv")
test_df = test_df.drop(columns=['Cabin', 'Name', 'PassengerId', 'Ticket'])
test_df.fillna({
    "Age": df["Age"].mean(),
    "Embarked": df["Embarked"].mode()[0]  
}, inplace=True)
le.fit(["S", "C", "Q"])
test_df["Embarked"] = le.transform(test_df["Embarked"])
le.fit(["male", "female"])
test_df["Sex"] = le.transform(test_df["Sex"])
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
test_df["FamilySize"] = test_df["SibSp"] + test_df["Parch"] + 1
test_df["IsAlone"] = (test_df["FamilySize"] == 1).astype(int)
train_set, val_set = train_test_split(df, test_size=0.2, random_state=42)

model = LogisticRegression(learning_rate=0.01, epochs=1000, feature_matrix=train_set)
model.train()
print("Validation Accuracy:", model.evaluate(val_set.drop(columns="Survived"), val_set["Survived"]))
model.save("logistic_model_1000epochs.pkl")
