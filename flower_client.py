import flwr as fl
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import psutil
from models import SmallCNN, BigCNN
from utils import get_system_context, select_model_based_on_context
from privacy_wrapper import wrap_with_dp

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load CIFAR-10 subset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

# Choose model based on context
context = get_system_context()
model_choice = select_model_based_on_context(context)

model = SmallCNN() if model_choice == "small" else BigCNN()
model = model.to(DEVICE)

# Wrap model with differential privacy
model, optimizer, privacy_engine = wrap_with_dp(model, trainloader)

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
        return self.get_parameters(config={}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        return 0.0, len(trainloader.dataset), {}

client = FlowerClient()
client.model = model
fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=client)