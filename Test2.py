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
depths = list(range(2, 35))
input_dim = 784
output_dim = 10
hidden_dim = 10
target_acc = 0.70
max_epochs = 8
learning_rate = 0.01

init_types = [
    "he_normal",
    "glorot_normal",
    
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

for init_type in init_types:
    vanilla_results = []
    print(f"[VanillaNet] Init: {init_type}")
    for depth in range(5, max(depths) + 1, 10):
        hidden_dims = [depth] * depth
        run_epochs = []

        for run in range(num_trials):
            seed = 42 + run
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            model = VanillaNet(input_dim, hidden_dims, output_dim, init_type=init_type).to(device)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.0, weight_decay=1e-4)
            criterion = nn.CrossEntropyLoss()
            reached = False

            for epoch in range(1, max_epochs + 1):
                model.train()
                for images, labels in train_loader:
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                # Evaluate
                model.eval()
                total_correct, total_samples = 0, 0
                with torch.no_grad():
                    for images, labels in test_loader:
                        images, labels = images.to(device), labels.to(device)
                        outputs = model(images)
                        total_correct += (outputs.argmax(dim=1) == labels).sum().item()
                        total_samples += labels.size(0)

                acc = total_correct / total_samples
                

                if acc >= target_acc:
                    run_epochs.append(epoch)
                    reached = True
                    break

            if not reached:
                run_epochs.append(max_epochs + 1)

        mean_epochs = np.mean(run_epochs)
        epochs_to_20_acc[init_type].append(mean_epochs)
        vanilla_results.append((depth, mean_epochs))
        print(f"Depth = {depth} | Mean epochs to reach {target_acc*100:.0f}% acc = {mean_epochs:.2f}")

        
    # Save per-init-type results
    print(vanilla_results)
    os.makedirs("results", exist_ok=True)
    filename = f"results/epochs_vs_depth_80percent{init_type}.csv"
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["depth", "mean_epochs"])
        writer.writerows(vanilla_results)

print("Saved VanillaNet results.")


#SubsidyNet

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

subsidy_results = [] 

for depth in range(20, max(depths) + 1, 10):
    hidden_dim = depth
    run_epochs = []
    
    for run in range(num_trials):
        seed = 42 + run
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        model = SubsidyNetV3(input_dim, [hidden_dim] * depth, output_dim, gamma= 0.6 *depth  ).to(device) 
        """
        In the paper "How to start training, the optimizer used is SGD , with learining rate of 0.01"
        """
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
                    """
                    The gradient clipping below may help limit the impact of the subsidy on exploding gradients.
                    """
                    #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                    subsidy_initialized = True
                else:
                    optimizer.zero_grad()
                    outputs = model(images, step=step, apply_subsidy=True, initial_subsidy=False)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    """
                    The gradient clipping below may help limit the impact of the subsidy on exploding gradients.
                    """
                    #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

            # decay gamma once per epoch
            model.step_epoch(depth)

            # Evaluation step
            model.eval()
            """
            Here model.eval, removes dropout and subsidy
            """
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

#Test trial 3

"""
In this trial, we test whether running SubsidyNet for a few epochs and then transferring its weights to a structurally identical
Vanilla model can help overcome poor weight initialization.
"""

print("[SubsidyNetV3] With Gradient-Based Subsidy")
subsidy_results = []  

for depth in range(30, max(depths) + 1, 10):
    hidden_dim = depth
    run_epochs = []

    for run in range(num_trials):
        seed = 42 + run
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        #Initialize model_init
        model_init = SubsidyNetV3(input_dim, [hidden_dim] * depth, output_dim, gamma=0.5 * depth).to(device)
        optimizer_init = torch.optim.SGD(model_init.parameters(), lr=0.01, momentum=0.0, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        # One batch only
        images, labels = next(iter(train_loader))
        images, labels = images.to(device), labels.to(device)

        # First forward-backward pass (no subsidy)
        model_init.train()
        outputs = model_init(images, step=0, apply_subsidy=False, initial_subsidy=True)
        loss = criterion(outputs, labels)
        loss.backward(retain_graph=True)

        model_init.update_gradients()
        optimizer_init.zero_grad()

        # Second forward-backward pass (with subsidy)
        outputs = model_init(images, step=0, apply_subsidy=True, initial_subsidy=True)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_init.step()

        #Initialize model with copied weights
        
        model = VanillaNet(input_dim, [hidden_dim] * depth, output_dim, init_type="glorot_uniform").to(device)
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
               
                output = model(images)
                loss = criterion(outputs, labels)
                
                loss.backward(retain_graph=True)

                optimizer.step()


            # Evaluate
            model.eval()
            total_correct = 0
            total_samples = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    output = model(images)
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

print("\Finished measuring epochs to reach 20% accuracy.")


#--------------------------------------------------------------------------------------------------------------------------------

print("[SubsidyNetV3] With Gradient-Based Initialization")
subsidy_results = []

for depth in range(60, max(depths) + 1, 10):
    hidden_dims = [depth] * depth
    run_epochs = []

    for run in range(num_trials):
        seed = 42 + run
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # -- Initialization phase using SubsidyNetV3
        model_init = SubsidyNetV3(input_dim, hidden_dims, output_dim, gamma=0.6 * depth).to(device)
        optimizer_init = torch.optim.SGD(model_init.parameters(), lr=0.1, momentum=0.0, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        images, labels = next(iter(train_loader))
        images, labels = images.to(device), labels.to(device)

        # Pass 1: WITHOUT subsidy
        model_init.train()
        outputs = model_init(images, step=0, apply_subsidy=True, initial_subsidy=False)
        loss = criterion(outputs, labels)
        loss.backward(retain_graph=True)
        model_init.update_gradients()
        optimizer_init.zero_grad()

        # Pass 2: WITH subsidy
        outputs = model_init(images, step=0, apply_subsidy=True, initial_subsidy=False)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_init.step()

        # Copy weights to VanillaNet
        model = VanillaNet(input_dim, hidden_dims, output_dim, init_type="he_normal").to(device)
        model.load_state_dict(model_init.state_dict())
        del model_init

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.0, weight_decay=1e-4)
        reached = False

        for epoch in range(1, max_epochs + 1):
            model.train()
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # Evaluate
            model.eval()
            total_correct, total_samples = 0, 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    total_correct += (outputs.argmax(dim=1) == labels).sum().item()
                    total_samples += labels.size(0)

            acc = total_correct / total_samples
            print(f"[VanillaNet-SubsidyInit] Run {run}, Epoch {epoch}, Test Accuracy: {acc:.4f}")

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
filename = "results/epochs_vs_depth_subsidy2_mds.csv"
os.makedirs(os.path.dirname(filename), exist_ok=True)
with open(filename, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["depth", "mean_epochs"])
    writer.writerows(subsidy_results)

print("Saved SubsidyNetV3-to-VanillaNet results.\n")

#===============================================================================================================================

#Trial 4 

"""
This section of the test is an experimental version, we try out using scaled, normalized fisher information , as a learning signal per layer.
"""

print("[SubsidyNetV4] Training with Persistent Gradient-Based Subsidy, Fisher Information")

subsidy_results = []

for depth in range(5, max(depths) + 1, 10):
    hidden_dims = [depth] * depth
    run_epochs = []

    for run in range(num_trials):
        seed = 42 + run
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Single SubsidyNetV3 model for this run
        model = SubsidyNetV4(input_dim, hidden_dims, output_dim, gamma = 0).to(device) #
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
