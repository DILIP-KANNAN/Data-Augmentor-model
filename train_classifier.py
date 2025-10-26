# train_classifier.py
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.transforms.functional as TF
from torchvision.datasets import ImageFolder

# Simple CNN
class SimpleCNN(nn.Module):
    def __init__(self, nc=3, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(nc, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4,4))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*4*4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def get_transform(img_size=64, nc=3):
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*nc, (0.5,)*nc)
    ])

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = get_transform(img_size=args.img_size, nc=args.nc)
    if args.dataset.lower() == 'mnist':
        train_ds = datasets.MNIST(root=args.data_root, train=True, download=True, transform=transform)
        test_ds = datasets.MNIST(root=args.data_root, train=False, download=True, transform=transform)
        num_classes = 10
    elif args.dataset.lower() == 'cifar10':
        train_ds = datasets.CIFAR10(root=args.data_root, train=True, download=True, transform=transform)
        test_ds = datasets.CIFAR10(root=args.data_root, train=False, download=True, transform=transform)
        num_classes = 10
    else:
        raise ValueError('Unsupported dataset')

    # Optional: add synthetic images (ImageFolder)
    if args.synthetic_dir:
        synth_ds = ImageFolder(root=args.synthetic_dir, transform=transform)
        # If your synthetic folder uses a flat structure, use a small wrapper to assign labels or create folders per-class.
        train_ds = ConcatDataset([train_ds, synth_ds])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = SimpleCNN(nc=args.nc, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        # Evaluate
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{args.epochs}] Loss: {avg_loss:.4f} Test Acc: {acc:.2f}%")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(args.out, 'best_classifier.pth'))
    print("Best test acc:", best_acc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='MNIST')
    parser.add_argument('--data_root', default='./data')
    parser.add_argument('--synthetic_dir', default=None, help='Path to folder with synthetic images (ImageFolder format recommended)')
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--nc', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--out', default='./models')
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    train(args)
