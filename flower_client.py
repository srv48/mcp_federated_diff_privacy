import flwr as fl
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from models import SmallCNN, BigCNN
from utils import get_system_context, select_model_based_on_context
from privacy_wrapper import wrap_with_dp
import argparse
import ssl
import copy
import atexit
import pandas as pd
import matplotlib.pyplot as plt


client_log = []

# ----------------- Parse Arguments -----------------
parser = argparse.ArgumentParser()
parser.add_argument("--client_id", type=int, required=True, help="Unique ID for the client")
parser.add_argument("--model_type", type=str, choices=["big", "small"], required=True, help="Model type: 'big' or 'small'")
args = parser.parse_args()
CLIENT_ID = args.client_id
MODEL_TYPE = args.model_type

ssl._create_default_https_context = ssl._create_unverified_context
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ----------------- Load CIFAR-10 -----------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

# ----------------- Models -----------------
# Local personal model (used for evaluation only)
local_model = BigCNN() if MODEL_TYPE == "big" else SmallCNN()
local_model = local_model.to(DEVICE)
initial_path = f"client_{CLIENT_ID}_{MODEL_TYPE}_initial.pth"

torch.save(local_model.state_dict(), initial_path)


# Federated model (used for training)
global_model = copy.deepcopy(local_model)
global_model, optimizer, privacy_engine = wrap_with_dp(global_model, trainloader)

# ----------------- Flower Client -----------------
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.detach().cpu().numpy() for val in self.model.parameters()]

    def set_parameters(self, parameters):
        for param, new_param in zip(self.model.parameters(), parameters):
            param.data = torch.tensor(new_param, dtype=param.data.dtype).to(DEVICE)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optimizer.step()

        # Save trained model
        # torch.save(self.model.state_dict(), initial_path)
        torch.save(self.model._module.state_dict(), initial_path)
        return self.get_parameters(config={}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        # Always evaluate local model (not federated model)
        local_model.load_state_dict(torch.load(initial_path))  # Reset if needed
        local_model.eval()

        loss_total, correct, total = 0.0, 0, 0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = local_model(inputs)
                loss = criterion(outputs, labels)
                loss_total += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

        loss_avg = loss_total / total
        accuracy = correct / total

        client_log.append({"test_loss": loss_avg, "test_acc": accuracy})
        print(f"[Client eval {CLIENT_ID} | {MODEL_TYPE}] (Local Model) Accuracy: {accuracy:.4f}, Loss: {loss_avg:.4f}")
        return loss_avg, total, {"accuracy": accuracy}

# ----------------- Run -----------------
client = FlowerClient()
client.model = global_model  # Only used for training
fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=client)

def save_client_logs():
    df = pd.DataFrame(client_log)
    df["round"] = range(1, len(df) + 1)
    df.to_csv(f"logs/client{CLIENT_ID}_log.csv", index=False)
    plt.plot(df["round"], df["test_acc"], label="Accuracy")
    plt.plot(df["round"], df["test_loss"], label="Loss")
    plt.xlabel("Round")
    plt.legend()
    plt.title(f"Client {CLIENT_ID} Metrics")
    plt.savefig(f"logs/client{CLIENT_ID}_metrics.png")

atexit.register(save_client_logs)