
import torch
import matplotlib.pyplot as plt
from LayerLearnSkeleton import VanillaNet ,  SubsidyNet ,SubsidyNetV2, SubsidyNetV3
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from Dataset import train_loader, test_loader

import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])

"""
We will use depth like the one mentioned in the paper, this file tests, final layer MD
"""
depths = list(range(2, 130))  
mean_squared_lengths = []

# Get one batch and move to device — models in LayerLearnSkeleton call .to(device) internally
images, labels = next(iter(train_loader))
images, labels = images.to(device), labels.to(device)

input_dim = 784
output_dim = 10
hidden_dim = 100 

import matplotlib.pyplot as plt
import numpy as np

#initialization Types, Colors, and Styles
import matplotlib.pyplot as plt

#initialization Types, Colors, Line Styles
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
    "subsidy2_mds" : 'red'
}

linestyles = {
    "glorot_uniform": '-',
    "glorot_normal": '--',
    "he_normal": '-.',
    "he_uniform": ':',
    "he_truncated": (0, (5, 1)),
    
    "subsidy": (0, (1, 1))  ,
    "subsidy2_mds" : 'solid',
}

#Storage for Results
all_mds = {init_type: [] for init_type in init_types}
all_mds["subsidy"] = []
all_mds["subsidy2_mds"] = []

#loop over depths and inits for VanillaNet
for init_type in init_types:
    print(f"Running VanillaNet for init: {init_type}")
    mds = []

    for depth in depths:
        
        hidden_dims = [depth] * depth
        model = VanillaNet(input_dim, hidden_dims, output_dim, init_type=init_type)
        output = model(images)
        metrics = model.get_layer_metrics()
        mds.append(metrics['mean_squared_length'][-1])

    all_mds[init_type] = mds

print(f"Running SubsidyNet (default init)")
subsidy_mds = []

for depth in depths:
    
    hidden_dims = [depth] * depth
    subsidy_model = SubsidyNet(input_dim, hidden_dims, output_dim)  
    output = subsidy_model(images, step=0)  
    metrics = subsidy_model.get_layer_metrics()
    subsidy_mds.append(metrics['mean_squared_length'][-1])

all_mds["subsidy"] = subsidy_mds


# Trial Second version

print(f"Running SubsidyNet Version 2 (default init)")
subsidy2_mds = []

import torch.nn as nn

"""
In the second version of SubsidyNet, the subsidy is applied only after the network has completed at least one forward and backward pass.
This design choice is necessary because alternative learning indicators, such as Fisher Information and gradient norms, require gradients
to be meaningful. To compute these gradients, we need to first run a complete training step.

The overall flow is as follows:
Random weight initialization → Forward pass → Loss computation → Backward pass → Gradient update → Subsidy application.
"""

#===============================================================================================================================
print("[VanillaNet] MDS Measurement Using SubsidyNetV3 Initialization")
subsidy2_mds = []

for depth in depths:
    hidden_dims = [depth] * depth
    gamma = 10 * depth  

    #Initialize via SubsidyNetV3
    init_model = SubsidyNetV3(input_dim, hidden_dims, output_dim, gamma=gamma).to(device)
    init_optimizer = torch.optim.SGD(init_model.parameters(), lr=0.01, momentum=0.0, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    #One batch
    images, labels = next(iter(train_loader))
    images, labels = images.to(device), labels.to(device)

    #First pass: no subsidy
    init_model.train()
    init_optimizer.zero_grad()
    outputs = init_model(images, step=0, apply_subsidy=False, initial_subsidy=True)
    loss = criterion(outputs, labels)
    loss.backward(retain_graph=True)
    init_model.update_gradients()
    init_optimizer.zero_grad()

    #Second pass: with subsidy
    outputs = init_model(images, step=0, apply_subsidy=True, initial_subsidy=True)
    loss = criterion(outputs, labels)
    loss.backward()
    init_optimizer.step()

    # Continue training with the already-initialized init_model — the VanillaNet
    # load + immediate overwrite was dead code (initialized weights were discarded)
    model = init_model
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images, step=1, apply_subsidy=True, initial_subsidy=False)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # Pass epoch counter (1), not depth — passing depth zeroed gamma immediately
        model.step_epoch(1)
        metrics = model.get_layer_metrics()
        break

    subsidy2_mds.append(metrics['mean_squared_length'][-1])

all_mds["subsidy2_mds"] = subsidy2_mds


#===============================================================================================================================

plt.figure(figsize=(12, 8))

for init_type in list(all_mds.keys()):
    print("Key=",init_type, "size =", len(all_mds[init_type]))
    plt.plot(depths, all_mds[init_type],
             label=init_type,
             color=colors[init_type],
             linestyle=linestyles[init_type])

plt.yscale('log')
plt.xlabel('Network Depth', fontsize=13)
plt.ylabel('Mean Squared Length (Md)', fontsize=13)
plt.title('Mean Squared Length vs Network Depth\nVanillaNet (various inits) vs SubsidyNet', fontsize=15)
plt.legend(fontsize=11)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()



"""
Here, this is just an experimental extra test metric basically, the consistency of variation from depth to depth,
taking into account that the moving mean stays the same.
"""

"""
import numpy as np

md_weighted_deviation_scores = {}

for init_type, mds in all_mds.items():
    mds = np.array(mds)
    mean_md = np.mean(mds)
    # Deviations
    deltas = mds - mean_md  
    # Linear weighting by depth
    weights = np.array(depths)  

    # Weighted squared deviations
    weighted_squared_deltas = weights * (deltas ** 2)
    weighted_var = np.sum(weighted_squared_deltas) / np.sum(weights)

    md_weighted_deviation_scores[init_type] = weighted_var

#results
for init_type, var_value in md_weighted_deviation_scores.items():
    print(f"{init_type}: Depth-weighted Deviation Score = {var_value:.6f}")

"""
