# src/server.py
import flwr as fl

def main():
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=3,              # ← Changed to 3
        min_evaluate_clients=3,         # ← Changed to 3
        min_available_clients=3,        # ← Changed to 3
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=5),   # Increased rounds
        strategy=strategy,
    )

if __name__ == "__main__":
    main()