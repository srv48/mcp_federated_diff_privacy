# Federated Learning with Context-Aware Model Orchestration and Differential Privacy

This project implements a privacy-preserving, adaptive federated learning system that enables decentralized model training across heterogeneous clients. It integrates three modern AI techniques:

- **Federated Learning (FL)** for collaborative on-device training
- **Model Context Protocol (MCP)** for dynamic model selection based on system resources
- **Differential Privacy (DP)** for protecting sensitive local data during training

---

## 🚀 Motivation

Inspired by privacy-focused learning on Apple devices (like Apple TV and iPhones), this project aims to replicate the same architecture using open-source tools. The system is designed to scale to a variety of devices and simulate real-world non-IID data conditions.

---

## ⚙️ Technologies Used

- **Python 3.10+**
- **PyTorch** — model development
- **Flower** — federated learning framework
- **Opacus** — differential privacy integration
- **psutil** — context monitoring (CPU, memory)
- **Streamlit (optional)** — dashboard for client system status

---

## 🏗️ Architecture

1. Each client monitors its CPU and memory in real-time.
2. Based on context, it selects `SmallCNN` or `BigCNN`.
3. Local training is performed with DP (using Opacus).
4. Model updates are sent to the federated server via Flower.
5. The server aggregates updates using FedAvg.

---
