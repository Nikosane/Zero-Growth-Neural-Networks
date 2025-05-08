# Zero-Growth Neural Networks

A biologically-inspired machine learning project where the neural network **cannot increase its parameter size** as it learns. Instead of expanding its architecture, the model dynamically **rewires its internal connections** to adapt, mimicking the learning constraints of the human brain.

---

## 📉 Motivation

In biology, the brain cannot simply add new neurons as it learns. Instead, it rewires existing synapses, optimizing its structure within a fixed volume. This project seeks to simulate this principle in neural networks by prohibiting growth in size and instead enabling learning through dynamic **synaptic rewiring**.

---

## 🏐 Core Idea

**Zero-Growth Neural Networks (ZGNNs)** work by maintaining a fixed number of parameters. The network improves by:

* Identifying underperforming or low-importance connections.
* Reallocating their capacity to more promising pathways.
* Utilizing sparse representations to encourage efficient learning.

## 📊 Features

* ✅ Fixed architecture neural network
* ✅ Dynamic connection rewiring
* ✅ Support for different rewiring strategies
* ✅ Visual logs and sparsity analysis
* ✅ Plug-and-play datasets (e.g., MNIST, CIFAR-10)

---

## 🚀 How to Run

1. **Install dependencies**:

```bash
pip install -r requirements.txt
```

2. **Train the model**:

```bash
python main.py
```

3. **View logs or saved models** in the respective directories.

---

## 🔄 Rewiring Strategies

Some of the implemented or planned algorithms include:

* **Lottery Ticket Hypothesis Pruning**
* **Hebbian Rewiring** ("fire together, wire together")
* **Spike-Timing Dependent Plasticity (STDP)**
* **Gradient-Based Importance Scoring**

These mimic neuroplasticity and allow for biological plausibility in machine learning.

---

## 🎓 Use Cases

* Edge devices with memory constraints
* Neuromorphic computing simulations
* Research in sparsity and plasticity
* Educational tool to understand brain-inspired AI

---

## ⚙️ Future Plans

* Implement attention-based rewiring
* Integrate with reinforcement learning environments
* Add visualization dashboards (e.g., weight heatmaps)
* Multi-task learning support within fixed size

---

## 📚 References

* Frankle & Carbin, "The Lottery Ticket Hypothesis"
* Spiking Neural Networks & STDP
* Deep Rewiring: Training Very Sparse Deep Networks

---

## 👥 Contributors

* You
* Open Source Collaborators (Submit a PR!)

---

## 🚫 License

MIT License. Use responsibly, especially for educational and experimental purposes.

---

## ✨ Acknowledgements

Inspired by nature, built for the future of efficient AI.

