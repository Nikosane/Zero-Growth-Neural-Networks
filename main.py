# main.py

from config import CONFIG
from train.trainer import Trainer
from data.dataloader import get_data_loaders
from model.zero_growth_net import ZeroGrowthNet

def main():
    # Load config
    model_config = CONFIG["model"]
    training_config = CONFIG["training"]

    # Prepare data
    train_loader, test_loader = get_data_loaders(CONFIG["training"]["batch_size"])

    # Initialize model
    model = ZeroGrowthNet(
        input_size=model_config["input_size"],
        hidden_size=model_config["hidden_size"],
        output_size=model_config["output_size"],
        sparsity=model_config["sparsity"]
    )

    # Initialize trainer
    trainer = Trainer(model, train_loader, test_loader, CONFIG)

    # Train
    trainer.train()

if __name__ == "__main__":
    main()
