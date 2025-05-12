# utils/logger.py

import os
from datetime import datetime

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

def log_metrics(epoch, batch, loss, accuracy):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] Epoch {epoch+1}, Batch {batch}, Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%"
    print(log_line)
    with open(os.path.join(LOG_DIR, "train_log.txt"), "a") as f:
        f.write(log_line + "\n")
