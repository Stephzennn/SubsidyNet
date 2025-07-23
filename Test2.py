import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

import csv
import os

from torchvision import transforms
from torch.utils.data import DataLoader

from LayerLearnSkeleton import VanillaNet, SubsidyNet, SubsidyNetV2, SubsidyNetV3
from Dataset import train_loader, test_loader

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Data preprocessing (already done in your loader presumably)
# If you reinitialize loaders, add this:
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Lambda(lambda x: x.view(-1))
# ])

# Settings
depths = list(range(2, 130))
input_dim = 784
output_dim = 10
hidden_dim = 10
target_acc = 0.20
max_epochs = 10
learning_rate = 0.01

init_types = [
    "he_normal",
    "he_uniform",
    "glorot_uniform",
    "he_custom",
    "glorot_normal",
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

# --- VanillaNet ---
num_trials = 5
import random
"""
for init_type in init_types:
    print(f"[VanillaNet] Init: {init_type}")
    
    for depth in range(40, max(depths) + 1, 5):
        #hidden_dim = depth  # You may want to decouple hidden_dim and depth
        hidden_dims = [depth] * depth
        run_epochs = []

        for run in range(num_trials):
            seed = 42 + run  
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            model = VanillaNet(input_dim, [hidden_dim] * depth, output_dim, init_type=init_type).to(device)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.0, weight_decay=1e-4)
            #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            criterion = nn.CrossEntropyLoss()
            reached = False

            for epoch in range(1, max_epochs + 1):

                # Training step
                model.train()
                for images, labels in train_loader:
                    images = images.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                # Evaluation step on test set
                model.eval()
                total_correct = 0
                total_samples = 0
                with torch.no_grad():
                    for images, labels in test_loader:
                        images = images.to(device)
                        labels = labels.to(device)
                        outputs = model(images)
                        preds = outputs.argmax(dim=1)
                        total_correct += (preds == labels).sum().item()
                        total_samples += labels.size(0)

                acc = total_correct / total_samples
                print(f"Run {run}, Epoch {epoch}, Test Accuracy: {acc:.4f}")

                if acc >= target_acc:
                    run_epochs.append(epoch)
                    reached = True
                    break  # stop early if target accuracy reached

            if not reached:
                run_epochs.append(max_epochs + 1)  # did not reach target

        mean_epochs = np.mean(run_epochs)
        epochs_to_20_acc[init_type].append(mean_epochs)
        print(f"Depth = {depth} | Mean epochs to reach {target_acc*100:.0f}% acc over {num_trials} runs = {mean_epochs:.2f}")


os.makedirs("results", exist_ok=True)  

for init_type in init_types:
    filename = f"results/epochs_vs_depth_{init_type}.csv"
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["depth", "mean_epochs"])
        writer.writerows(epochs_to_20_acc[init_type])
    print(f"Saved results to {filename}")

# --- SubsidyNet ---
print("[SubsidyNet] Default Init")
for depth in depths:
    model = SubsidyNet(input_dim, [hidden_dim] * depth, output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    reached = False

    for epoch in range(1, max_epochs + 1):
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images, step=epoch)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            acc = compute_accuracy(outputs, labels)
            if acc >= target_acc:
                epochs_to_20_acc["subsidy"].append(epoch)
                reached = True
                break
        if reached:
            break

    if not reached:
        epochs_to_20_acc["subsidy"].append(max_epochs + 1)

print("[SubsidyNetV3] With Gradient-Based Subsidy")

subsidy_results = []  # list to store (depth, mean_epochs)

#for depth in depths:
for depth in range(20, max(depths) + 1, 10):
    hidden_dim = depth
    run_epochs = []
    
    for run in range(num_trials):
        seed = 42 + run
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        model = SubsidyNetV3(input_dim, [hidden_dim] * depth, output_dim, gamma= 0.6 *depth  ).to(device) #,depth=depths 
        #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.0, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        subsidy_initialized = False
        reached = False
        step = 0

        for epoch in range(1, max_epochs + 1):
            model.train()
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                if not subsidy_initialized:
                    # Step 1: No subsidy
                    outputs = model(images, step=step, apply_subsidy=False, initial_subsidy=True)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    model.update_gradients()

                    optimizer.zero_grad()
                    # Step 2: With subsidy
                    outputs = model(images, step=step, apply_subsidy=True, initial_subsidy=True)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                    subsidy_initialized = True
                else:
                    optimizer.zero_grad()
                    outputs = model(images, step=step, apply_subsidy=True, initial_subsidy=False)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

            # decay gamma once per epoch
            model.step_epoch(depth)

            # Evaluation step
            model.eval()
            total_correct = 0
            total_samples = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images, step=step, apply_subsidy=False, initial_subsidy=False)
                    total_correct += (outputs.argmax(dim=1) == labels).sum().item()
                    total_samples += labels.size(0)

            acc = total_correct / total_samples
            print(f"[SubsidyNetV2] Run {run}, Epoch {epoch}, Test Accuracy: {acc:.4f}")

            if acc >= target_acc:
                run_epochs.append(epoch)
                reached = True
                break
            step += 1
        if not reached:
            run_epochs.append(max_epochs + 1)

    mean_epochs = np.mean(run_epochs)
    subsidy_results.append((depth, mean_epochs))
    epochs_to_20_acc["subsidy2_mds"].append(mean_epochs)
    print(f"Depth = {depth} | Mean epochs to reach {target_acc*100:.0f}% acc over {num_trials} runs = {mean_epochs:.2f}")

#======================================================================================================================================
"""
#Test trial 3
print("[SubsidyNetV3] With Gradient-Based Subsidy")
subsidy_results = []  

for depth in range(20, max(depths) + 1, 10):
    hidden_dim = depth
    run_epochs = []

    for run in range(num_trials):
        seed = 42 + run
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        #Initialize model_init
        model_init = SubsidyNetV3(input_dim, [hidden_dim] * depth, output_dim, gamma=0.6 * depth).to(device)
        optimizer_init = torch.optim.SGD(model_init.parameters(), lr=0.01, momentum=0.0, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        # One batch only
        images, labels = next(iter(train_loader))
        images, labels = images.to(device), labels.to(device)

        # First forward-backward pass (no subsidy)
        model_init.train()
        outputs = model_init(images, step=0, apply_subsidy=False, initial_subsidy=True)
        loss = criterion(outputs, labels)
        loss.backward()
        model_init.update_gradients()
        optimizer_init.zero_grad()

        # Second forward-backward pass (with subsidy)
        outputs = model_init(images, step=0, apply_subsidy=True, initial_subsidy=True)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_init.step()

        #Initialize model with copied weights
        model = SubsidyNetV3(input_dim, [hidden_dim] * depth, output_dim, gamma=0.6 * depth).to(device)
        model.load_state_dict(model_init.state_dict())  
        del model_init 

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.0, weight_decay=1e-4)
        step = 1
        reached = False

        #Train as usual
        for epoch in range(1, max_epochs + 1):
            model.train()
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(images, step=step, apply_subsidy=True, initial_subsidy=False)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            model.step_epoch(depth)

            # Evaluate
            model.eval()
            total_correct = 0
            total_samples = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images, step=step, apply_subsidy=False, initial_subsidy=False)
                    total_correct += (outputs.argmax(dim=1) == labels).sum().item()
                    total_samples += labels.size(0)

            acc = total_correct / total_samples
            print(f"[SubsidyNetV2] Run {run}, Epoch {epoch}, Test Accuracy: {acc:.4f}")

            if acc >= target_acc:
                run_epochs.append(epoch)
                reached = True
                break
            step += 1

        if not reached:
            run_epochs.append(max_epochs + 1)

    mean_epochs = np.mean(run_epochs)
    subsidy_results.append((depth, mean_epochs))
    epochs_to_20_acc["subsidy2_mds"].append(mean_epochs)
    print(f"Depth = {depth} | Mean epochs to reach {target_acc*100:.0f}% acc over {num_trials} runs = {mean_epochs:.2f}")
#===========================================================================================================================

filename = "results/epochs_vs_depth_subsidy2_mds.csv"
with open(filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["depth", "mean_epochs"])
    writer.writerows(subsidy_results)
print(f"Saved SubsidyNetV2 results to {filename}")

# Done
print("\n===> Finished measuring epochs to reach 20% accuracy.")
