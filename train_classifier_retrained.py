# train_classifier_retrained.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, ConcatDataset
from PIL import Image

# --------------------------
# Model Definition
# --------------------------
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

# --------------------------
# Transform function
# --------------------------
def get_transform(img_size=64, nc=3):
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*nc, (0.5,)*nc)
    ])

# --------------------------
# Training Loop
# --------------------------
def train_classifier(data_root, synthetic_dir=None, img_size=64, nc=3,
                     batch_size=128, lr=1e-3, epochs=10, out='./models',
                     dataset_name='CIFAR10'):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(out, exist_ok=True)
    
    transform = get_transform(img_size=img_size, nc=nc)
    
    # Load original dataset
    if dataset_name.lower() == 'mnist':
        train_ds = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
        test_ds  = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)
        num_classes = 10
    elif dataset_name.lower() == 'cifar10':
        train_ds = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
        test_ds  = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
        num_classes = 10
    else:
        raise ValueError("Unsupported dataset")
    
    # Add synthetic GAN images if available
    if synthetic_dir:
        synth_ds = ImageFolder(root=synthetic_dir, transform=transform)
        train_ds = ConcatDataset([train_ds, synth_ds])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Model, loss, optimizer
    model = SimpleCNN(nc=nc, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
        avg_loss = total_loss / len(train_loader.dataset)
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f} Val Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(out, 'best_classifier.pth'))
            print(f"Saved best model with val_acc={best_acc:.4f}")
    print("Training complete. Best val_acc:", best_acc)

# --------------------------
# CLI
# --------------------------
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='./data')
    parser.add_argument('--synthetic_dir', default=None)
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--nc', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--out', default='./models')
    parser.add_argument('--dataset', default='CIFAR10')
    args = parser.parse_args()
    
    train_classifier(data_root=args.data_root,
                     synthetic_dir=args.synthetic_dir,
                     img_size=args.img_size,
                     nc=args.nc,
                     batch_size=args.batch_size,
                     lr=args.lr,
                     epochs=args.epochs,
                     out=args.out,
                     dataset_name=args.dataset)
