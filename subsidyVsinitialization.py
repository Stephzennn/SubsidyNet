import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from LayerLearnSkeleton import VanillaNet, SubsidyNet, SubsidyNetV2, SubsidyNetV3
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from Dataset import train_loader, test_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

#depths = list(range(5, 130, 5))

depths = [5, 15, 30, 50, 75, 100, 125]

input_dim, output_dim, hidden_dim = 784, 10, 100

init_types = [
    "glorot_uniform",
    "glorot_normal",
    "he_normal",
    "he_uniform",
    "he_truncated",
]

colors = {
    "glorot_uniform": 'orange',
    "glorot_normal": 'green',
    "he_normal": 'blue',
    "he_uniform": 'black',
    "he_truncated": 'cyan',
    "subsidy": 'yellow',
    "subsidy2_mds": 'red'
}

linestyles = {
    "glorot_uniform": '-',
    "glorot_normal": '--',
    "he_normal": '-.',
    "he_uniform": ':',
    "he_truncated": (0, (5, 1)),
    "subsidy": (0, (1, 1)),
    "subsidy2_mds": 'solid',
}

all_mds = {init: [] for init in init_types}
all_mds["subsidy"] = []
all_mds["subsidy2_mds"] = []

all_vsi = {init: [] for init in init_types}
all_vsi["subsidy"] = []
all_vsi["subsidy2_mds"] = []

all_loss = {init: [] for init in init_types}
all_loss["subsidy"] = []
all_loss["subsidy2_mds"] = []

all_convergence = {init: [] for init in init_types}
all_convergence["subsidy"] = []
all_convergence["subsidy2_mds"] = []

# --- VanillaNet Variants ---
for init_type in init_types:
    print(f"\n=== Running VanillaNet for init: {init_type} ===")
    mds, vsi, losses, convergence_epochs = [], [], [], []
    criterion = nn.CrossEntropyLoss()

    for depth in depths:
        hidden_dims = [depth]
        model = VanillaNet(input_dim, hidden_dims, output_dim, init_type=init_type).to(device)
        model.train()
        images, labels = next(iter(train_loader))
        images, labels = images.to(device), labels.to(device)

        output = model(images)
        metrics = model.get_layer_metrics()
        md = metrics['mean_squared_length'][-1]
        std = output.std().item()
        loss = criterion(output, labels)

        mds.append(md)
        vsi.append(std)
        losses.append(loss.item())

        print(f"[Vanilla | {init_type} | Depth {depth}]")
        print(f"  Md: {md:.4f} | VSI: {std:.4f} | Loss: {loss.item():.4f}")

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        epoch, reached = 0, False
        while not reached and epoch < 10:
            for batch_images, batch_labels in train_loader:
                batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
                optimizer.zero_grad()
                out = model(batch_images)
                l = criterion(out, batch_labels)
                l.backward()
                optimizer.step()
                acc = (out.argmax(1) == batch_labels).float().mean().item()
                if acc >= 0.65:
                    reached = True
                    break
            epoch += 1
        print(f"  Convergence Epochs: {epoch}")
        convergence_epochs.append(epoch)

    all_mds[init_type] = mds
    all_vsi[init_type] = vsi
    all_loss[init_type] = losses
    all_convergence[init_type] = convergence_epochs

# --- SubsidyNet ---
print(f"\n=== Running SubsidyNet ===")
mds, vsi, losses, convergence_epochs = [], [], [], []

for depth in depths:
    hidden_dims = [depth]
    model = SubsidyNet(input_dim, hidden_dims, output_dim).to(device)
    images, labels = next(iter(train_loader))
    images, labels = images.to(device), labels.to(device)

    output = model(images, step=0)
    metrics = model.get_layer_metrics()
    md = metrics['mean_squared_length'][-1]
    std = output.std().item()
    loss = nn.CrossEntropyLoss()(output, labels)

    mds.append(md)
    vsi.append(std)
    losses.append(loss.item())

    print(f"[SubsidyNet | Depth {depth}]")
    print(f"  Md: {md:.4f} | VSI: {std:.4f} | Loss: {loss.item():.4f}")

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    epoch, reached = 0, False
    while not reached and epoch < 10:
        for batch_images, batch_labels in train_loader:
            batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            out = model(batch_images, step=0)
            l = nn.CrossEntropyLoss()(out, batch_labels)
            l.backward()
            optimizer.step()
            acc = (out.argmax(1) == batch_labels).float().mean().item()
            if acc >= 0.65:
                reached = True
                break
        epoch += 1
    print(f"  Convergence Epochs: {epoch}")
    convergence_epochs.append(epoch)

all_mds["subsidy"] = mds
all_vsi["subsidy"] = vsi
all_loss["subsidy"] = losses
all_convergence["subsidy"] = convergence_epochs

# --- SubsidyNetV3 ---
print(f"\n=== Running SubsidyNetV3 ===")
mds, vsi, losses, convergence_epochs = [], [], [], []

for depth in depths:
    hidden_dims = [depth]
    gamma = 10 * depth
    model = SubsidyNetV3(input_dim, hidden_dims, output_dim, gamma=gamma).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.0, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    images, labels = next(iter(train_loader))
    images, labels = images.to(device), labels.to(device)

    # Step 1: without subsidy
    model.train()
    optimizer.zero_grad()
    output = model(images, step=0, apply_subsidy=False, initial_subsidy=True)
    loss = criterion(output, labels)
    loss.backward(retain_graph=True)
    model.update_gradients()
    optimizer.zero_grad()

    # Step 2: with subsidy
    output = model(images, step=0, apply_subsidy=True, initial_subsidy=True)
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()

    output = model(images, step=1, apply_subsidy=True, initial_subsidy=False)
    metrics = model.get_layer_metrics()
    md = metrics['mean_squared_length'][-1]
    std = output.std().item()

    mds.append(md)
    vsi.append(std)
    losses.append(loss.item())

    print(f"[SubsidyNetV3 | Depth {depth}]")
    print(f"  Md: {md:.4f} | VSI: {std:.4f} | Loss: {loss.item():.4f}")

    epoch, reached = 0, False
    while not reached and epoch < 10:
        for batch_images, batch_labels in train_loader:
            batch_images, batch_labels = batch_images.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            out = model(batch_images, step=1, apply_subsidy=True, initial_subsidy=False)
            l = criterion(out, batch_labels)
            l.backward()
            optimizer.step()
            acc = (out.argmax(1) == batch_labels).float().mean().item()
            if acc >= 0.65:
                reached = True
                break
        epoch += 1
    print(f"  Convergence Epochs: {epoch}")
    convergence_epochs.append(epoch)

all_mds["subsidy2_mds"] = mds
all_vsi["subsidy2_mds"] = vsi
all_loss["subsidy2_mds"] = losses
all_convergence["subsidy2_mds"] = convergence_epochs

# --- Plotting Section ---
#plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(18, 12))

# Mean Squared Length
plt.subplot(2, 2, 1)
for key in all_mds:
    plt.plot(depths, all_mds[key], label=key, color=colors[key], linestyle=linestyles[key])
plt.yscale('log')
plt.xlabel('Network Depth', fontsize=13)
plt.ylabel('Mean Squared Length (Md)', fontsize=13)
plt.title('Signal Stability Across Depths', fontsize=14)
plt.legend()

# VSI
plt.subplot(2, 2, 2)
for key in all_vsi:
    plt.plot(depths, all_vsi[key], label=key, color=colors[key], linestyle=linestyles[key])
plt.xlabel('Network Depth', fontsize=13)
plt.ylabel('Vanishing Signal Index (Std Dev of Outputs)', fontsize=13)
plt.title('Vanishing Signal Index vs Depth', fontsize=14)
plt.legend()

# Training Loss
plt.subplot(2, 2, 3)
for key in all_loss:
    plt.plot(depths, all_loss[key], label=key, color=colors[key], linestyle=linestyles[key])
plt.xlabel('Network Depth', fontsize=13)
plt.ylabel('Training Loss (After 1 Forward Pass)', fontsize=13)
plt.title('Training Loss vs Depth', fontsize=14)
plt.legend()

# Convergence Time
plt.subplot(2, 2, 4)
for key in all_convergence:
    plt.plot(depths, all_convergence[key], label=key, color=colors[key], linestyle=linestyles[key])
plt.xlabel('Network Depth', fontsize=13)
plt.ylabel('Epochs to 65% Batch Accuracy', fontsize=13)##
plt.title('Convergence Time vs Depth', fontsize=14)
plt.legend()

plt.tight_layout()
plt.suptitle('Network Depth Analysis Across Initialization and Subsidy Variants', fontsize=16, y=1.02)
plt.show()
