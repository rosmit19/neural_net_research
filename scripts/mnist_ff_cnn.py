# mnist_ff.py

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import time
import csv
import psutil
import pynvml

# Initialize GPU monitoring
pynvml.nvmlInit()
gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

# Define FF Model
class FFBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.linear = nn.Linear(in_ch, out_ch)
        self.norm = nn.LayerNorm(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.norm(x)
        return x

class FFModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = FFBlock(28*28, 256)
        self.block2 = FFBlock(256, 128)
        self.final_layer = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = self.block1(x)
        x = self.block2(x)
        x = self.final_layer(x)
        return x

# Dataloaders
def get_dataloaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    trainset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    return (
        DataLoader(trainset, batch_size=64, shuffle=True),
        DataLoader(testset, batch_size=64, shuffle=False)
    )

# Logger
log_path = "../logs/mnist/mnist_ff_metrics.csv"
os.makedirs("../logs/mnist", exist_ok=True)
with open(log_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Epoch", "Train Acc (%)", "Test Acc (%)", "Loss", "Time (s)", "GPU Mem (MB)", "CPU Mem (MB)", "GPU Power (W)"])

def log_metrics(epoch, train_acc, test_acc, loss, runtime):
    gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle).used / (1024 * 1024)
    power = pynvml.nvmlDeviceGetPowerUsage(gpu_handle) / 1000
    cpu_mem = psutil.Process().memory_info().rss / (1024 * 1024)
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, train_acc, test_acc, loss, runtime, gpu_mem, cpu_mem, power])

# Training
def train(model, trainloader, testloader, device, epochs=50):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    model.to(device)

    for epoch in range(1, epochs+1):
        model.train()
        total_loss, correct, total = 0, 0, 0
        start = time.time()

        for x, y in trainloader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        train_acc = 100 * correct / total

        # Test
        model.eval()
        correct_test, total_test = 0, 0
        with torch.no_grad():
            for x, y in testloader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(1)
                correct_test += (preds == y).sum().item()
                total_test += y.size(0)
        test_acc = 100 * correct_test / total_test

        runtime = time.time() - start
        print(f"Epoch {epoch:02d} | Loss: {total_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | Time: {runtime:.1f}s")
        log_metrics(epoch, train_acc, test_acc, total_loss, runtime)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainloader, testloader = get_dataloaders()
    model = FFModel()
    print("Training FF on MNIST")
    train(model, trainloader, testloader, device)
