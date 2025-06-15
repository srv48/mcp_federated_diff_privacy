# dp_proof_demo.py (Final with Metrics and Non-Blackbox Proof)

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from opacus import PrivacyEngine
import torch.nn.functional as F
import numpy as np
import random

class CIFARCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# --- Load and Sample CIFAR-10 Dataset ---
def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_data = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    test_data = datasets.CIFAR10('./data', train=False, transform=transform)

    train_subset = Subset(train_data, random.sample(range(len(train_data)), 5000))
    test_subset = Subset(test_data, list(range(1000)))

    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)
    return train_loader, test_loader

# --- Train Function ---
def train_model(model, train_loader, use_dp=False, noise=1.0, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    epsilon = None
    privacy_engine = None

    if use_dp:
        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=noise,
            max_grad_norm=1.0
        )

    model.train()
    total_loss = 0
    for epoch in range(epochs):
        epoch_loss = 0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")
        total_loss += epoch_loss

    if use_dp:
        epsilon = privacy_engine.get_epsilon(delta=1e-5)
        print(f"[DP Enabled] Final ε = {epsilon:.4f}")
    else:
        print("[DP Disabled] Training without privacy.")

    return model, epsilon

# --- Evaluation ---
def evaluate(model, dataloader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in dataloader:
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
    acc = 100.0 * correct / len(dataloader.dataset)
    print(f"Test Accuracy: {acc:.2f}%")
    return acc

# --- Membership Inference ---
def membership_inference(model, dataloader, threshold=0.5):
    model.eval()
    confident = 0
    total = 0
    scores = []
    with torch.no_grad():
        for data, _ in dataloader:
            out = model(data)
            probs = F.softmax(out, dim=1)
            max_probs = probs.max(dim=1)[0]
            confident += (max_probs > threshold).sum().item()
            total += data.size(0)
            scores.extend(max_probs.tolist())

    if total == 0:
        return 0.0

    print(f"Avg confidence: {np.mean(scores):.4f}, Above threshold: {confident}/{total}")
    return confident / total