
import torch
import matplotlib.pyplot as plt
from LayerLearnSkeleton import VanillaNet ,  SubsidyNet 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from Dataset import train_loader, test_loader
#Load vectorized MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])
#train_dataset = datasets.MNIST(root='./data', train=True, download=False, transform=transform)
#train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)

#Prepare depths and store results
"""
We will use depth like the one mentioned in the paper
"""
depths = list(range(2, 130))  
mean_squared_lengths = []

# Get one batch
images, _ = next(iter(train_loader))

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
    "he_truncated"
]

colors = {
    "glorot_uniform": 'orange',
    "glorot_normal": 'green',
    "he_normal": 'blue',
    "he_uniform": 'black',
    "he_truncated": 'cyan',
    #"he_normal": 'white',
    #"he_uniform": 'black',
    #"he_truncated": 'white',
    "subsidy": 'red'  # SubsidyNet color
}

linestyles = {
    "glorot_uniform": '-',
    "glorot_normal": '--',
    "he_normal": '-.',
    "he_uniform": ':',
    "he_truncated": (0, (5, 1)),
    # Red dotted line for SubsidyNet
    "subsidy": (0, (1, 1))  
}

#Storage for Results
all_mds = {init_type: [] for init_type in init_types}
all_mds["subsidy"] = []

#loop over depths and inits for VanillaNet
for init_type in init_types:
    print(f"Running VanillaNet for init: {init_type}")
    mds = []

    for depth in depths:
        hidden_dims = [hidden_dim] * depth
        model = VanillaNet(input_dim, hidden_dims, output_dim, init_type=init_type)
        output = model(images)
        metrics = model.get_layer_metrics()
        mds.append(metrics['mean_squared_length'][-1])

    all_mds[init_type] = mds

#loop over depths for SubsidyNet
print(f"Running SubsidyNet (default init)")
subsidy_mds = []

for depth in depths:
    hidden_dims = [hidden_dim] * depth
    subsidy_model = SubsidyNet(input_dim, hidden_dims, output_dim)  
    #static decay (no training)
    output = subsidy_model(images, step=0)  
    metrics = subsidy_model.get_layer_metrics()
    subsidy_mds.append(metrics['mean_squared_length'][-1])

all_mds["subsidy"] = subsidy_mds

#plot (Log scale)
plt.figure(figsize=(12, 8))

for init_type in list(all_mds.keys()):
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
# 4. Plot
plt.figure(figsize=(8,6))
plt.plot(depths, mean_squared_lengths, marker='o', label='Vanilla + ReLU')
plt.yscale('log')
plt.xlabel('Network Depth')
plt.ylabel('Mean Squared Length (Md)')
plt.title('Mean Squared Length vs Depth (Vanilla Net)')
plt.legend()
plt.grid(True)
plt.show()
"""