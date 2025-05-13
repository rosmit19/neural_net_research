import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import os
import csv
import psutil
import pynvml

# Initialize GPU monitoring
pynvml.nvmlInit()
gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

# -------------------------
# Model Definition
# -------------------------
class FFA_Block(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear(x))

class FFA_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = FFA_Block(28*28, 256)
        self.block2 = FFA_Block(256, 128)
        self.final = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.block1(x)
        x = self.block2(x)
        return self.final(x)

    def goodness(self, x):
        return (x**2).mean(dim=1)  # Goodness is mean square of activations

# -------------------------
# Data Loader
# -------------------------
def get_mnist_loaders(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    trainset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    return trainloader, testloader

# -------------------------
# CSV Logger
# -------------------------
log_path = "../logs/mnist/mnist_ff_true.csv"
os.makedirs("../logs/mnist", exist_ok=True)
with open(log_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Epoch", "Train Acc (%)", "Test Acc (%)", "Loss", "Time (s)", "GPU Mem (MB)", "CPU Mem (MB)", "GPU Power (W)"])

def log_metrics(epoch, train_acc, test_acc, loss, runtime):
    gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle).used / (1024 ** 2)
    power = pynvml.nvmlDeviceGetPowerUsage(gpu_handle) / 1000
    cpu_mem = psutil.Process().memory_info().rss / (1024 ** 2)
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, train_acc, test_acc, loss, runtime, gpu_mem, cpu_mem, power])

# -------------------------
# Positive + Negative Sample Creator
# -------------------------
def create_pos_neg_batch(inputs, labels, num_classes=10):
    batch_size = labels.size(0)
    wrong_labels = (labels + torch.randint(1, num_classes, labels.size(), device=labels.device)) % num_classes

    pos_labels = nn.functional.one_hot(labels, num_classes=num_classes).float()
    neg_labels = nn.functional.one_hot(wrong_labels, num_classes=num_classes).float()

    pos_inputs = torch.cat([inputs.view(batch_size, -1), pos_labels], dim=1)
    neg_inputs = torch.cat([inputs.view(batch_size, -1), neg_labels], dim=1)

    return pos_inputs, neg_inputs, labels

# -------------------------
# FFA Loss (ReLU Margin)
# -------------------------
def ffa_loss(pos_goodness, neg_goodness, margin=1.0):
    return torch.mean(torch.relu(neg_goodness - pos_goodness + margin))

# -------------------------
# Training Function
# -------------------------
def train_ffa(model, trainloader, testloader, device, epochs=50):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct_train, total_train = 0, 0, 0
        start = time.time()

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Embed labels
            pos_inputs, neg_inputs, true_labels = create_pos_neg_batch(inputs, labels)

            optimizer.zero_grad()

            pos_out = model(pos_inputs)
            neg_out = model(neg_inputs)

            pos_goodness = model.goodness(pos_out)
            neg_goodness = model.goodness(neg_out)

            loss = ffa_loss(pos_goodness, neg_goodness)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            preds = pos_out.argmax(dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        train_acc = 100 * correct_train / total_train

        # --- Evaluation ---
        model.eval()
        correct_test, total_test = 0, 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs.view(inputs.size(0), -1))
                preds = outputs.argmax(dim=1)
                correct_test += (preds == labels).sum().item()
                total_test += labels.size(0)
        test_acc = 100 * correct_test / total_test

        runtime = time.time() - start
        print(f"Epoch {epoch:02d} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | Loss: {total_loss:.4f} | Time: {runtime:.1f}s")
        log_metrics(epoch, train_acc, test_acc, total_loss, runtime)

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainloader, testloader = get_mnist_loaders()
    model = FFA_Net().to(device)
    print("Training True FFA on MNIST...")
    train_ffa(model, trainloader, testloader, device)

