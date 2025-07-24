import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

transform = transforms.Compose([
    transforms.ToTensor(),  
    transforms.Normalize((0.1307,), (0.3081,)),  
    transforms.Lambda(lambda x: x.view(-1)) 
])

data_root = './data'
download_flag = not os.path.exists(os.path.join(data_root, 'MNIST'))

full_train_dataset = datasets.MNIST(root=data_root, train=True, download=download_flag, transform=transform)
test_dataset = datasets.MNIST(root=data_root, train=False, download=download_flag, transform=transform)

train_size = int(0.9 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

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




