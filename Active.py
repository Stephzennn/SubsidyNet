
import torch
import matplotlib.pyplot as plt
from LayerLearnSkeleton import VanillaNet ,  SubsidyNet ,SubsidyNetV2
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
images, labels = next(iter(train_loader))

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
    #"he_normal": 'white',
    #"he_uniform": 'black',
    #"he_truncated": 'white',
    "subsidy": 'yellow',  # SubsidyNet color
    "subsidy2_mds" : 'red'
}

linestyles = {
    "glorot_uniform": '-',
    "glorot_normal": '--',
    "he_normal": '-.',
    "he_uniform": ':',
    "he_truncated": (0, (5, 1)),
    # Red dotted line for SubsidyNet
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
    #Here we will let the SubsidyNet go one round before computation
    #static decay (no training)
    output = subsidy_model(images, step=0)  
    metrics = subsidy_model.get_layer_metrics()
    subsidy_mds.append(metrics['mean_squared_length'][-1])

all_mds["subsidy"] = subsidy_mds


# Trial Second version

#loop over depths for SubsidyNet
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

num_epochs = 1
learning_rate = 0.001
# Test learning_rate = 1.0
for depth in depths:
    #print("depth", depth)
    subsidy_model = SubsidyNetV2(input_dim, hidden_dims, output_dim)
    optimizer = torch.optim.Adam(subsidy_model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    step = 0
    subsidy_initialized = False
    metrics  = []
    for epoch in range(num_epochs):
        
        for images, labels in train_loader:
            optimizer.zero_grad()

            if not subsidy_initialized:
                # First pass: no subsidy 
                outputs = subsidy_model(images, step=step, apply_subsidy=False, initial_subsidy=True)
                loss = criterion(outputs, labels)
                loss.backward()

                #Update metrics for subsidy allocation 
                subsidy_model.update_gradients()

                # Second pass: with subsidy 
                optimizer.zero_grad()
                outputs_subsidy = subsidy_model(images, step=step, apply_subsidy=True,  initial_subsidy=True)
                loss_subsidy = criterion(outputs_subsidy, labels)
                loss_subsidy.backward()
                optimizer.step()

                # Optional: final forward for logging metrics 
                metrics = subsidy_model.get_layer_metrics()

                # Set the flag to avoid repeating
                subsidy_initialized = True
            else:
                # Regular training with subsidy
                outputs = subsidy_model(images, step=step, apply_subsidy=True,  initial_subsidy=False)
                loss = criterion(outputs, labels)
                loss.backward()
                metrics = subsidy_model.get_layer_metrics()
                optimizer.step()
            break    
        
        step += 1
        
    subsidy2_mds.append(metrics['mean_squared_length'][-1])

all_mds["subsidy2_mds"] = subsidy2_mds

#plot (Log scale)
plt.figure(figsize=(12, 8))
print
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
