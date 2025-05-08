# config.py

CONFIG = {
    "model": {
        "input_size": 784,  # For MNIST: 28x28
        "hidden_size": 256,
        "output_size": 10,
        "sparsity": 0.8,  # 80% connections pruned initially
        "rewire_every": 5,  # Epochs after which rewiring occurs
    },
    "training": {
        "epochs": 20,
        "batch_size": 64,
        "learning_rate": 0.001,
        "rewiring_strategy": "lottery_ticket"  # or 'hebbian', 'gradient'
    },
    "misc": {
        "save_path": "saved_models/model.pt",
        "log_interval": 10
    }
}
