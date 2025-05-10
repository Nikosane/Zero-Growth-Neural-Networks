# train/trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
from train.rewiring import apply_rewiring
from utils.logger import log_metrics
from utils.metrics import accuracy

class Trainer:
    def __init__(self, model, train_loader, test_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config["training"]["learning_rate"])
        self.rewire_every = config["model"]["rewire_every"]
        self.rewiring_strategy = config["training"]["rewiring_strategy"]

    def train(self):
        for epoch in range(self.config["training"]["epochs"]):
            self.model.train()
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(self.train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                if i % self.config["misc"]["log_interval"] == 0:
                    acc = accuracy(outputs, labels)
                    log_metrics(epoch, i, loss.item(), acc)

            # Rewiring step
            if (epoch + 1) % self.rewire_every == 0:
                apply_rewiring(self.model, self.rewiring_strategy)
                print(f"Epoch {epoch+1}: Rewiring performed with strategy '{self.rewiring_strategy}'.")

            # Optional: evaluate after each epoch
            self.evaluate(epoch)

    def evaluate(self, epoch):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        print(f"Epoch {epoch+1} Evaluation Accuracy: {acc:.2f}%")
