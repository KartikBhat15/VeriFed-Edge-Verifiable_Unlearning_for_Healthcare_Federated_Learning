# src/client.py
import warnings
warnings.filterwarnings("ignore")   # ← This hides the deprecation warning

import flwr as fl
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import sys

X, y = load_iris(return_X_y=True)

class IrisClient(fl.client.NumPyClient):
    def __init__(self, client_id):
        self.client_id = client_id
        np.random.seed(int(client_id) * 10)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=int(client_id) * 10
        )
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        self.model = LogisticRegression(max_iter=200)
        self.model.fit(self.X_train, self.y_train)

    def get_parameters(self, config):
        return [self.model.coef_, self.model.intercept_]

    def set_parameters(self, parameters):
        self.model.coef_ = parameters[0]
        self.model.intercept_ = parameters[1]

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.fit(self.X_train, self.y_train)
        print(f"[Client {self.client_id}] Training completed | Samples: {len(self.X_train)}")
        return self.get_parameters(config), len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        accuracy = self.model.score(self.X_test, self.y_test)
        print(f"[Client {self.client_id}] Accuracy: {accuracy:.4f}")
        return 0.0, len(self.X_test), {"accuracy": accuracy}

def main():
    client_id = sys.argv[1] if len(sys.argv) > 1 else "1"
    client = IrisClient(client_id).to_client()
    fl.client.start_client(server_address="127.0.0.1:8080", client=client)

if __name__ == "__main__":
    main()