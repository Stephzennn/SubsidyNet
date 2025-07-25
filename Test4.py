import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from LayerLearnSkeleton import VanillaNet, SubsidyNet, SubsidyNetV2, SubsidyNetV4
from Dataset import train_loader, test_loader



# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Settings
depths = list(range(2, 130))
input_dim = 784
output_dim = 10
hidden_dims = 10
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

depth = 60
hidden_dims = [depth] * depth
def get_layer_pre_activation_variances(model, x, apply_subsidy=False):
    variances = []
    model.eval()
    with torch.no_grad():
        for layer in model.layers:
            x = layer.linear(x)  # get pre-activation (Z)
            variances.append(x.var().item())
            if hasattr(layer, 'subsidy_value') and apply_subsidy:
                z = x + layer.subsidy_value
                x = torch.relu(z)
            else:
                x = torch.relu(x)
    return variances

def train_model(model, train_loader, optimizer, criterion, epochs=3, apply_subsidy=False, vanilla =False):
    model.train()
    for epoch in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            if vanilla == True:
                outputs = model(images)
            else:
                outputs = model(images, step=epoch+1, apply_subsidy=apply_subsidy)
            loss = criterion(outputs, labels)
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            model.update_gradients()  # for SubsidyNet
            optimizer.step()


def train_modell(model, train_loader, optimizer, criterion, epochs=3, apply_subsidy=False, vanilla=False):
    model.train()
    first_batch = True
    for epoch in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            if vanilla:
                outputs = model(images)
            else:
                outputs = model(images, step=epoch+1, apply_subsidy=apply_subsidy)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            model.update_gradients()  # for SubsidyNet
            
            if first_batch:
                # Morph weights here based on Fisher info
                with torch.no_grad():
                    for layer in model.layers:
                        fisher = layer.activation_variance
                        eps = 1e-6
                        fisher_clipped = min(max(fisher, eps), 1.0)
                        inverse_fisher = 1.0 - fisher_clipped
                        morph_scale = 1 + inverse_fisher
                        layer.linear.weight.data *= morph_scale
                        if layer.linear.bias is not None:
                            layer.linear.bias.data *= morph_scale
                first_batch = False
            
            optimizer.step()

# Assume input_dim, output_dim, train_loader, etc. are defined

# Initialize models
model_subsidy = SubsidyNetV4(input_dim, hidden_dims, output_dim, gamma=0.06 * depth).to(device)
model_he = VanillaNet(input_dim, hidden_dims, output_dim, init_type="he_normal").to(device)
optimizer_subsidy = torch.optim.SGD(model_subsidy.parameters(), lr=0.01)
optimizer_he = torch.optim.SGD(model_he.parameters(), lr=0.01)

criterion = nn.CrossEntropyLoss()

# Train both models for 3 epochs
train_model(model_he, train_loader, optimizer_he, criterion, epochs=3, apply_subsidy=False, vanilla =True)
train_modell(model_subsidy, train_loader, optimizer_subsidy, criterion, epochs=3, apply_subsidy=True)


# Get a batch of data for variance measurement
images, _ = next(iter(train_loader))
images = images.to(device)

# Get pre-activation variances
var_subsidy = get_layer_pre_activation_variances(model_subsidy, images, apply_subsidy=True)
var_he = get_layer_pre_activation_variances(model_he, images, apply_subsidy=False)

print("Pre-activation variances after 3 epochs:")
for i, (v_sub, v_he) in enumerate(zip(var_subsidy, var_he)):
    print(f"Layer {i}: SubsidyNet variance = {v_sub:.4f}, He init variance = {v_he:.4f}")
