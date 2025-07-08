import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

#Convert to tensor and flatten (28x28 → 784)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])

#Check if MNIST data already exists
data_root = './data'
download_flag = not os.path.exists(os.path.join(data_root, 'MNIST'))

#Load datasets (download only if missing)
full_train_dataset = datasets.MNIST(root=data_root, train=True, download=download_flag, transform=transform)
test_dataset = datasets.MNIST(root=data_root, train=False, download=download_flag, transform=transform)

# Train and Validation
train_size = int(0.9 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

#Create DataLoaders
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#checks
for images, labels in train_loader:
    print("Train batch:", images.shape, labels.shape)
    break

for images, labels in val_loader:
    print("Validation batch:", images.shape, labels.shape)
    break

for images, labels in test_loader:
    print("Test batch:", images.shape, labels.shape)
    break




