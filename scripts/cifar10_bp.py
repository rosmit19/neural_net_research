import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from utils import apply_fourier_label
import time
import pynvml
import psutil
import os
import csv

# Initialize GPU power monitor
pynvml.nvmlInit()
gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

# -------------------------
# FF Model with Aux Layers
# -------------------------
class FFBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool=True):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2) if pool else nn.Identity()
        )
        self.norm = nn.GroupNorm(1, out_ch)
        self.aux = nn.Conv2d(out_ch, 10, kernel_size=1)

    def forward(self, x):
        x = self.norm(self.seq(x))
        g = self.aux(x).mean(dim=(2, 3))
        return x, g

class FFModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([
            FFBlock(1, 32),
            FFBlock(32, 64),
            FFBlock(64, 128, pool=False)
        ])

    def forward(self, x):
        scores = []
        for block in self.blocks:
            x, g = block(x)
            scores.append(g)
        return torch.stack(scores, dim=1)

# -------------------------
# Label Embedder
# -------------------------
def embed_labels(images, labels):
    return torch.stack([
        apply_fourier_label(img.squeeze(0), label) for img, label in zip(images, labels)
    ]).unsqueeze(1)

# -------------------------
# ReLU Margin Loss
# -------------------------
def relu_margin_loss(pos_scores, neg_scores, margin=1.0):
    return torch.mean(torch.relu(neg_scores.mean(1) - pos_scores.mean(1) + margin))

# -------------------------
# Data Loaders
# -------------------------
def get_dataloaders():
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    return (
        DataLoader(trainset, batch_size=64, shuffle=True),
        DataLoader(testset, batch_size=64, shuffle=False)
    )

# -------------------------
# CSV Logger
# -------------------------
log_path = "../logs/cifar10/cifar10_ff_metrics.csv"
os.makedirs("../logs/cifar10", exist_ok=True)
with open(log_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Epoch", "Train Acc (%)", "Test Acc (%)", "Loss", "Time (s)",
                     "GPU Mem (MB)", "CPU Mem (MB)", "GPU Power (W)"])

def log_metrics(epoch, train_acc, test_acc, loss, runtime):
    gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle).used / 1024 / 1024
    power = pynvml.nvmlDeviceGetPowerUsage(gpu_handle) / 1000
    cpu_mem = psutil.Process().memory_info().rss / 1024 / 1024
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, train_acc, test_acc, loss, runtime, gpu_mem, cpu_mem, power])

# -------------------------
# Training Loop
# -------------------------
def train(model, trainloader, testloader, device, epochs=50):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct_train, total_train = 0, 0, 0
        start = time.time()

        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            wrong_y = (y + torch.randint(1, 10, y.shape, device=device)) % 10

            pos = embed_labels(x.clone(), y).to(device)
            with torch.no_grad():
                neg = embed_labels(x.clone(), wrong_y).to(device)

            pos_g = model(pos)
            neg_g = model(neg)

            pscore = pos_g.gather(2, y.view(-1, 1, 1).expand(-1, pos_g.size(1), 1)).squeeze(2)
            nscore = neg_g.gather(2, wrong_y.view(-1, 1, 1).expand(-1, pos_g.size(1), 1)).squeeze(2)
            loss = relu_margin_loss(pscore, nscore)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            pred = pos_g.sum(1).argmax(1)
            correct_train += (pred == y).sum().item()
            total_train += y.size(0)

            del x, y, pos, neg, pos_g, neg_g, pscore, nscore
            torch.cuda.empty_cache()

        train_acc = 100 * correct_train / total_train

        # Evaluation
        model.eval()
        correct_test, total_test = 0, 0
        with torch.no_grad():
            for x, y in testloader:
                x, y = x.to(device), y.to(device)
                x = embed_labels(x, y).to(device)
                g = model(x)
                pred = g.sum(1).argmax(1)
                correct_test += (pred == y).sum().item()
                total_test += y.size(0)
        test_acc = 100 * correct_test / total_test

        runtime = time.time() - start
        print(f"Epoch {epoch:02d} | Loss: {total_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% | Time: {runtime:.1f}s")
        log_metrics(epoch, train_acc, test_acc, total_loss, runtime)

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainloader, testloader = get_dataloaders()
    model = FFModel()
    print("Training FF with Spatial Labels + ReLU Margin Loss")
    train(model, trainloader, testloader, device)


