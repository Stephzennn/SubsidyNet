import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""
This Test3.py file is not important, it is just an experimental file

"""
def init_he_normal(layer):
    if isinstance(layer, nn.Linear):
        nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
        nn.init.zeros_(layer.bias)

def init_he_uniform(layer):
    if isinstance(layer, nn.Linear):
        nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
        nn.init.zeros_(layer.bias)

def init_he_normal_truncated(layer):
    if isinstance(layer, nn.Linear):
        fan_in = layer.weight.size(1)
        std = torch.sqrt(torch.tensor(2.0 / fan_in)).to(layer.weight.device)
        with torch.no_grad():
            layer.weight.normal_(0, std)
            layer.weight.clamp_(-2*std, 2*std)
        nn.init.zeros_(layer.bias)

def init_glorot_uniform(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)

def init_glorot_normal(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_normal_(layer.weight)
        nn.init.zeros_(layer.bias)

class DecayScheduler:
    def __init__(self, decay_type='exponential', beta=0.01):
        self.decay_type = decay_type
        self.beta = beta

    def get_decay(self, step):
        if self.decay_type == 'exponential':
            return torch.exp(torch.tensor(-self.beta * step, dtype=torch.float32)).item()
        elif self.decay_type == 'linear':
            return max(0.0, 1 - self.beta * step)
        else:
            return 1.0

def compute_activation_variance(activations):
    return torch.var(activations, unbiased=False).item()

def compute_gradient_norm(param):
    if param.grad is None:
        return 0.0
    return torch.norm(param.grad, p=2).item()

def compute_fisher_information(param):
    if param.grad is None:
        return 0.0
    grad = param.grad.view(-1)
    return torch.sum(grad ** 2).item()

def allocate_subsidy(signal_value, epsilon, gamma, decay_value):
    gap = max(0.0, epsilon - signal_value)
    return gamma * gap * decay_value

def allocate_subsidy_gradient(grad_norm, epsilon, gamma, decay_value):
    gap = max(0.0, epsilon - grad_norm)
    return gamma * gap * decay_value

class SubsidyLinear(nn.Module):
    def __init__(self, in_features, out_features, layer_idx, init_type="glorot_uniform", epsilon=0.05, gamma=1.0, decay_scheduler=None):
        super(SubsidyLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.layer_idx = layer_idx
        self.epsilon = epsilon
        self.gamma = gamma
        self.decay_scheduler = decay_scheduler
        self.init_type = init_type

        if init_type == "glorot_uniform":
            init_glorot_uniform(self.linear)
        elif init_type == "glorot_normal":
            init_glorot_normal(self.linear)
        elif init_type == "he_normal":
            init_he_normal(self.linear)
        elif init_type == "he_uniform":
            init_he_uniform(self.linear)
        elif init_type == "he_truncated":
            init_he_normal_truncated(self.linear)
        elif init_type == "bad_uniform":
            nn.init.uniform_(self.linear.weight, a=0.1, b=1.0)
            nn.init.uniform_(self.linear.bias, a=0.1, b=1.0)

        self.to(device)

        self.subsidy_value = 0.0
        self.mean_squared_length = 0.0
        self.activation_variance = 0.0
        self.gradient_norm = 0.0

    def forward(self, x, current_step):
        z = self.linear(x)
        squared_length = (z.pow(2).sum(dim=1) / z.size(1)).mean().item()
        self.mean_squared_length = squared_length
        self.activation_variance = torch.var(z, unbiased=False).item()
        decay = self.decay_scheduler.get_decay(current_step) if self.decay_scheduler else 1.0
        self.subsidy_value = allocate_subsidy(self.activation_variance, self.epsilon, self.gamma, decay)
        z = z + self.subsidy_value
        return F.relu(z)

    def compute_gradient_info(self):
        if self.linear.weight.grad is not None:
            self.gradient_norm = torch.norm(self.linear.weight.grad, p=2).item()

class SubsidyNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, init_type="he_normal", epsilon=0.05, gamma=1.0, beta=0.01):
        super(SubsidyNet, self).__init__()
        self.decay_scheduler = DecayScheduler(beta=beta)
        self.layers = nn.ModuleList()
        dims = [input_dim] + hidden_dims + [output_dim]

        for idx in range(len(dims) - 1):
            self.layers.append(
                SubsidyLinear(dims[idx], dims[idx+1], layer_idx=idx,
                              init_type=init_type, epsilon=epsilon,
                              gamma=gamma, decay_scheduler=self.decay_scheduler)
            )
        self.to(device)

    def forward(self, x, step):
        x = x.to(device)
        for layer in self.layers[:-1]:
            x = layer(x, step)
        return self.layers[-1](x, step)

    def update_gradients(self):
        for layer in self.layers:
            layer.compute_gradient_info()

    def get_layer_metrics(self):
        return {
            "mean_squared_length": [l.mean_squared_length for l in self.layers],
            "activation_variance": [l.activation_variance for l in self.layers],
            "gradient_norm": [l.gradient_norm for l in self.layers],
        }

class VanillaLinear(nn.Module):
    def __init__(self, in_features, out_features, init_type="he_normal"):
        super(VanillaLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.init_type = init_type

        if init_type == "glorot_uniform":
            init_glorot_uniform(self.linear)
        elif init_type == "glorot_normal":
            init_glorot_normal(self.linear)
        elif init_type == "he_normal":
            init_he_normal(self.linear)
        elif init_type == "he_uniform":
            init_he_uniform(self.linear)
        elif init_type == "he_truncated":
            init_he_normal_truncated(self.linear)
        elif init_type == "bad_uniform":
            nn.init.uniform_(self.linear.weight, a=0.1, b=1.0)
            nn.init.uniform_(self.linear.bias, a=0.1, b=1.0)

        self.to(device)

        self.mean_squared_length = 0.0
        self.activation_variance = 0.0
        self.gradient_norm = 0.0

    def forward(self, x):
        z = self.linear(x)
        self.mean_squared_length = (z.pow(2).sum(dim=1) / z.size(1)).mean().item()
        self.activation_variance = torch.var(z, unbiased=False).item()
        return F.relu(z)

    def compute_gradient_info(self):
        if self.linear.weight.grad is not None:
            self.gradient_norm = torch.norm(self.linear.weight.grad, p=2).item()

class VanillaNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, init_type="he_normal"):
        super(VanillaNet, self).__init__()
        self.layers = nn.ModuleList()
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            self.layers.append(VanillaLinear(dims[i], dims[i+1], init_type))
        self.to(device)

    def forward(self, x):
        x = x.to(device)
        for layer in self.layers[:-1]:
            x = layer(x)
        return self.layers[-1](x)

    def update_gradients(self):
        for layer in self.layers:
            layer.compute_gradient_info()

    def get_layer_metrics(self):
        return {
            "mean_squared_length": [l.mean_squared_length for l in self.layers],
            "activation_variance": [l.activation_variance for l in self.layers],
            "gradient_norm": [l.gradient_norm for l in self.layers],
        }
