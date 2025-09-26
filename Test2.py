import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import csv
import os
from torchvision import transforms
from torch.utils.data import DataLoader
from LayerLearnSkeleton import VanillaNet, SubsidyNet, SubsidyNetV2, SubsidyNetV3,  SubsidyNetV4
from Dataset import train_loader, test_loader
import random
# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)



# Settings
depths = list(range(2, 100))
input_dim = 784
output_dim = 10
hidden_dim = 10
target_acc = 0.20
max_epochs = 10
learning_rate = 0.01

init_types = [
    "glorot_normal",
    "he_normal",
    
    
    "he_uniform",
    "glorot_uniform",
    "he_custom",
    "he_truncated",
]

# Storage
epochs_to_20_acc = {init_type: [] for init_type in init_types}
epochs_to_20_acc["subsidy"] = []
epochs_to_20_acc["subsidy2_mds"] = []

# Accuracy helper
def compute_accuracy(outputs, labels):
    preds = torch.argmax(outputs, dim=1)
    correct = (preds == labels).sum().item()
    return correct / labels.size(0)


num_trials = 5


print("[VanillaNet] Baseline Runs")

#Trial 4 

"""
This section of the test is an experimental version, we try out using scaled, normalized fisher information , as a learning signal per layer.
"""

print("[SubsidyNetV4] Training with Persistent Gradient-Based Subsidy, Fisher Information")

subsidy_results = []

for depth in range(35, max(depths) + 1, 10):
    hidden_dims = [depth] * depth
    run_epochs = []

    for run in range(num_trials):
        seed = 42 + run
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Single SubsidyNetV3 model for this run
        model = SubsidyNetV4(input_dim, hidden_dims, output_dim, gamma = 0.6 * depth).to(device) #
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.0, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        reached = False

        for epoch in range(1, max_epochs + 1):
            model.train()
            
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images, step=epoch, apply_subsidy=True, initial_subsidy=False)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                model.update_gradients()
                optimizer.step()
            
            # Evaluate
            model.eval()
            total_correct, total_samples = 0, 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images, step=epoch, apply_subsidy=True, initial_subsidy=False)
                    total_correct += (outputs.argmax(dim=1) == labels).sum().item()
                    total_samples += labels.size(0)

            acc = total_correct / total_samples
            print("Depth =",depth, "Epoch", epoch, "Accuracy",acc)
            model.step_epoch(depth)
            if acc >= target_acc:
                run_epochs.append(epoch)
                reached = True
                break

        if not reached:
            run_epochs.append(max_epochs + 1)

    mean_epochs = np.mean(run_epochs)
    subsidy_results.append((depth, mean_epochs))
    epochs_to_20_acc["subsidy2_mds"].append(mean_epochs)
    print(f"Depth = {depth} | Mean epochs to reach {target_acc*100:.0f}% acc = {mean_epochs:.2f}")

# Save results
filename = "results/epochs_vs_depth_subsidyVersion4_mds_glorot_uniform.csv"
os.makedirs(os.path.dirname(filename), exist_ok=True)
with open(filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["depth", "mean_epochs"])
    writer.writerows(subsidy_results)

print("Saved SubsidyNetV3 results with persistent subsidy.\n")
