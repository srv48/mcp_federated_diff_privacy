# flower_server.py

import flwr as fl
import torch
import torch.nn as nn
from models import SmallCNN
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load CIFAR-10 test set
def get_test_loader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    return DataLoader(testset, batch_size=32, shuffle=False)

testloader = get_test_loader()

# Create a place to store final parameters
final_parameters = []

# Define evaluation function that also stores parameters
def evaluate_fn(server_round, parameters, config):
    global final_parameters
    final_parameters = parameters  # Save the latest aggregated parameters

    model = SmallCNN().to(DEVICE)

    # Load parameters into model
    for param, new_param in zip(model.parameters(), parameters):
        param.data = torch.tensor(new_param, dtype=param.data.dtype).to(DEVICE)

    model.eval()
    loss_total = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss_total += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    loss_avg = loss_total / total

    print(f"[Server] Evaluation round {server_round} | Loss: {loss_avg:.4f} | Accuracy: {accuracy:.4f}")
    return loss_avg, {"accuracy": accuracy}

# Main Flower server
def main():
    strategy = fl.server.strategy.FedAvg(
        evaluate_fn=evaluate_fn,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
    )

    # Save final model
    print("\n[Server] Training complete. Saving final model...")
    model = SmallCNN()
    for param, new_param in zip(model.parameters(), final_parameters):
        param.data = torch.tensor(new_param, dtype=param.data.dtype)
    torch.save(model.state_dict(), "final_federated_model.pth")
    print("[Server] ✅ Final global model saved to 'final_federated_model.pth'")

if __name__ == "__main__":
    main()