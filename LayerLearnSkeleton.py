import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn as nn
import torch

"""
The following code is a rough skeleton for the paper we will be doing for the Deep Learning final project. We have several classes and functions that we will
use to test our hypothesis. If you want to add more functions and classes for testing, this is where they should be located. So far, we have a normal vanilla
neural network module that uses PyTorch's functionalities with some modifications so that we can try out different initializations of weights.
These are (He normal, He uniform, He Normal truncated (LowerBound : -2*std, UpperBound: 2*std ), Glorot Uniform and Glorot Normal ).

The Glorot initialization will serve as our baseline for a suboptimal initialization, as discussed in the paper "How to Start Training: The Effect of Initialization
and Architecture" by Hanin and Rolnick (2018). The metric functions are tools we will use to test our subsidy theory. Since subsidy is discretionary, it must be
based on feedback indicating initialization imbalances, such as activation variance. Notably, using activation variance to adjust the initialized weights is
conceptually similar to what He initialization achieves. However, our subsidy approach differs in that, based on periodic feedback from variance (or other metrics), we
will continue to adjust layers with poor learning by a decaying amount, controlled by a specific hyperparameter. 

To give an analogy, if an earlier layer is disproportionately underperforming compared to its peers, that feedback will prompt us to subsidize its learning so that
the layer can catch up, especially during the initial stages. However, if we observe similar learning stagnation later in the training cycle, while we may still apply
subsidy, the decay parameter will gradually reduce its effect. This reflects the assumption that the layer’s lack of learning is no longer due to an architectural problem
but because it has already learned what it can and has reached its optimum. This is similar to how governments subsidize sensitive and risky sectors or companies early on,
so that once those sectors or companies have built sufficient capacity, the subsidy can be lifted and they can compete fairly in the open market.


The decay scheduler is a function that has not yet been tested, but its purpose is to allow us to tune the type of decay we want to apply to the model (e.g., exponential, linear)
as well as the decay amount.


"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# 1. He Normal (Kaiming Normal)
def init_he_normal(layer):
    if isinstance(layer, nn.Linear):
        #nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
        torch.nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(layer.bias)
        

# 2. He Uniform (Kaiming Uniform)
def init_he_uniform(layer):
    if isinstance(layer, nn.Linear):
        torch.nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
        #nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
        nn.init.zeros_(layer.bias)

# 3. He Normal Truncated (values clipped at 2 standard deviations)
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
        torch.nn.init.xavier_uniform_(layer.weight, gain=torch.nn.init.calculate_gain('relu'))
        nn.init.zeros_(layer.bias)


def init_glorot_normal(layer):
    if isinstance(layer, nn.Linear):
        torch.nn.init.xavier_normal_(layer.weight, gain=torch.nn.init.calculate_gain('relu'))
        nn.init.zeros_(layer.bias)



# ===== Decay Scheduler =====
class DecayScheduler:
    def __init__(self, decay_type='exponential', beta=0.01):
        self.decay_type = decay_type
        self.beta = beta

    def get_decay(self, step):
        if self.decay_type == 'exponential':
            #return torch.exp(-self.beta * step)
            return torch.exp(torch.tensor(-self.beta * step, dtype=torch.float32))

        elif self.decay_type == 'linear':
            return max(0.0, 1 - self.beta * step)
        else:
            return 1.0

# Metric Functions 
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

#Subsidy Allocation Function 
def allocate_subsidy(signal_value, epsilon, gamma, decay_value):
    """
    signal_value:
        A numeric value representing a measured quantity or metric
        related to the current state of the model or layer. For example,
        it could be variance, accuracy, or any feedback signal—such as
        those listed above in the Metric Functions section
        (e.g., gradient norm, Fisher information),that indicates how well
        a part of the model is performing.

    epsilon:
        A threshold or target value. This acts as a benchmark or minimal
        acceptable level for the signal_value.

    gamma:
        A scaling factor (hyperparameter) controlling the overall magnitude
        of the subsidy. It determines how strongly we want to respond when 
        signal_value is below the threshold.

    decay_value:
        A decay multiplier (between 0 and 1) that reduces the subsidy amount
        over time or iterations, allowing the subsidy to gradually diminish
        as training progresses or as conditions improve.
    """
    gap = max(0.0000001, epsilon - signal_value)
    return gamma * gap * decay_value


"""
# ===== Subsidy Layer Version Old =====
class SubsidyLinear(nn.Module):
    def __init__(self, in_features, out_features, layer_idx, epsilon=0.05, gamma=1.0, decay_scheduler=None):
        super(SubsidyLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.layer_idx = layer_idx
        self.epsilon = epsilon
        self.gamma = gamma
        self.decay_scheduler = decay_scheduler
        self.subsidy_value = 0.0  # will update dynamically
        self.activation_variance = 0.0
        self.gradient_norm = 0.0

    def forward(self, x, current_step):
        # Measure pre-activation
        z = self.linear(x)

        # Compute activation variance
        self.activation_variance = compute_activation_variance(z)

        # Compute decay
        decay = self.decay_scheduler.get_decay(current_step) if self.decay_scheduler else 1.0

        # Allocate subsidy
        self.subsidy_value = allocate_subsidy(self.activation_variance, self.epsilon, self.gamma, decay)

        # Apply subsidy (pre-activation)
        z = z + self.subsidy_value

        # Apply activation
        a = F.relu(z)
        return a

    def compute_gradient_info(self):
        self.gradient_norm = compute_gradient_norm(self.linear.weight)

# ===== Full Network =====
class SubsidyNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, epsilon=0.05, gamma=1.0, beta=0.01):
        super(SubsidyNet, self).__init__()
        self.decay_scheduler = DecayScheduler(beta=beta)
        self.layers = nn.ModuleList()
        dims = [input_dim] + hidden_dims + [output_dim]

        for idx in range(len(dims) - 1):
            self.layers.append(SubsidyLinear(dims[idx], dims[idx+1], layer_idx=idx, 
                                             epsilon=epsilon, gamma=gamma, decay_scheduler=self.decay_scheduler))

    def forward(self, x, step):
        for layer in self.layers[:-1]:
            x = layer(x, step)
        # Final layer without ReLU
        x = self.layers[-1](x, step)
        return x

    def update_gradients(self):
        for layer in self.layers:
            layer.compute_gradient_info()
"""
class SubsidyLinear(nn.Module):
    def __init__(self, in_features, out_features, layer_idx, init_type="glorot_uniform", epsilon=0.05, gamma=1.0, decay_scheduler=None):
        super(SubsidyLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.layer_idx = layer_idx
        self.epsilon = epsilon
        self.gamma = gamma
        self.decay_scheduler = decay_scheduler
        self.init_type = init_type

        # Apply selected initialization
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
        else:
            pass  # Default PyTorch init
        
        self.to(device)
        # Metrics for tracking
        self.subsidy_value = 0.0
        self.mean_squared_length = 0.0
        self.activation_variance = 0.0
        self.gradient_norm = 0.0

    def forward(self, x, current_step):
        z = self.linear(x)

        # Compute pre-activation metrics
        squared_length = (z.pow(2).sum(dim=1) / z.size(1)).mean().item()
        self.mean_squared_length = squared_length
        self.activation_variance = torch.var(z, unbiased=False).item()

        # Compute decay
        decay = self.decay_scheduler.get_decay(current_step) if self.decay_scheduler else 1.0

        # Compute and apply subsidy
        self.subsidy_value = allocate_subsidy(self.activation_variance, self.epsilon, self.gamma, decay)
        z = z + self.subsidy_value

        a = F.relu(z)
        return a

    def compute_gradient_info(self):
        if self.linear.weight.grad is not None:
            self.gradient_norm = torch.norm(self.linear.weight.grad, p=2).item()
        else:
            self.gradient_norm = 0.0
class SubsidyNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, init_type="he_normal", epsilon=0.05, gamma=1.0, beta=0.01):
        super(SubsidyNet, self).__init__()
        self.decay_scheduler = DecayScheduler(beta=beta)
        self.layers = nn.ModuleList()
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for idx in range(len(dims) - 1):
            self.layers.append(SubsidyLinear(dims[idx], dims[idx+1], layer_idx=idx, 
                                             init_type=init_type,
                                             epsilon=epsilon, gamma=gamma, decay_scheduler=self.decay_scheduler))
        self.to(device)

    def forward(self, x, step):
        x = x.to(device)
        for layer in self.layers[:-1]:
            x = layer(x, step)
        x = self.layers[-1](x, step)  # No ReLU at the end
        return x

    def update_gradients(self):
        for layer in self.layers:
            layer.compute_gradient_info()

    def get_layer_metrics(self):
        mean_sq_lengths = [layer.mean_squared_length for layer in self.layers]
        act_vars = [layer.activation_variance for layer in self.layers]
        grad_norms = [layer.gradient_norm for layer in self.layers]

        return {
            "mean_squared_length": mean_sq_lengths,
            "activation_variance": act_vars,
            "gradient_norm": grad_norms,
        }


# ===== He-Initialized Linear Layer =====


class VanillaLinear(nn.Module):
    def __init__(self, in_features, out_features, init_type="he_normal"):
        super(VanillaLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features,bias=True)
        # Store the initialization type as a string
        self.init_type = init_type  

        # Apply selected initialization
        if init_type == "glorot_uniform":
            init_glorot_uniform(self.linear)
            
            #init_glorot_uniform(self.linear.bias)
        elif init_type == "glorot_normal":
            init_glorot_normal(self.linear)
            #init_glorot_normal(self.linear.bias)
        elif init_type == "he_normal":
            init_he_normal(self.linear)
            #init_he_normal(self.linear.bias) 
        elif init_type == "he_uniform":
            init_he_uniform(self.linear)
            #init_he_uniform(self.linear.bias)
        elif init_type == "he_truncated":
            init_he_normal_truncated(self.linear)
            #init_he_normal_truncated(self.linear.bias)
        elif init_type == "he_custom":
            self.init_he_normal_full(self.linear)
            #self.init_he_normal_full(self.linear.bias)
        elif init_type == "bad_uniform":
            nn.init.uniform_(self.linear.weight, a=0.1, b=1.0)
            nn.init.uniform_(self.linear.bias, a=0.1, b=1.0)
        else:
            # Default PyTorch init
            pass  
        #if self.linear.bias is not None:
        #    self.linear.bias.data.zero_()

        if self.linear.bias is not None:
            with torch.no_grad():
                self.linear.bias.zero_()

        self.mean_squared_length = 0.0
        self.activation_variance = 0.0
        self.gradient_norm = 0.0
        self.to(device)
    @staticmethod
    def init_he_normal_full(layer):
        if isinstance(layer, nn.Linear):
            fan_in = layer.weight.size(1)  
            std = (2.0 / fan_in) ** 0.5
            with torch.no_grad():
                layer.weight.normal_(0.0, std)  
                if layer.bias is not None:
                    layer.bias.zero_()

    def forward(self, x):
        # pre-activation
        z = self.linear(x)  

        # Compute Mean Squared Length (pre-activation)
        squared_length = (z.pow(2).sum(dim=1) / z.size(1)).mean().item()
        self.mean_squared_length = squared_length

        # Compute Activation Variance (pre-activation)
        """
        The reason we record the variance even though this is a vanilla
        architecture is that we want the different functions to be consistent
        with each other. This way, in the comparison section, it will be easier
        to use a single codebase for both SubsidyNet and VanillaNet.
        """
        self.activation_variance = torch.var(z, unbiased=False).item()

        # ReLU activation
        a = F.relu(z)  
        return a

    def compute_gradient_info(self):
        if self.linear.weight.grad is not None:
            self.gradient_norm = torch.norm(self.linear.weight.grad, p=2).item()
        else:
            self.gradient_norm = 0.0
"""
class VanillaNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, init_type="he_normal"):
        super(VanillaNet, self).__init__()
        self.layers = nn.ModuleList()
        dims = [input_dim] + hidden_dims + [output_dim]

        for idx in range(len(dims) - 1):
            self.layers.append(VanillaLinear(dims[idx], dims[idx+1], init_type))
        self.to(device)
    def forward(self, x):
        x = x.to(device)
        for layer in self.layers[:-1]:
            x = layer(x)
        x = self.layers[-1](x) 
        return x

    def update_gradients(self):
        for layer in self.layers:
            layer.compute_gradient_info()

    def get_layer_metrics(self):
        
        mean_sq_lengths = [layer.mean_squared_length for layer in self.layers]
        act_vars = [layer.activation_variance for layer in self.layers]
        grad_norms = [layer.gradient_norm for layer in self.layers]

        return {
            "mean_squared_length": mean_sq_lengths,
            "activation_variance": act_vars,
            "gradient_norm": grad_norms,
        }
"""
class VanillaNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, init_type="he_normal"):
        super(VanillaNet, self).__init__()
        self.hidden_layers = nn.ModuleList()
        dims = [input_dim] + hidden_dims

        # Custom hidden layers
        for idx in range(len(dims) - 1):
            self.hidden_layers.append(VanillaLinear(dims[idx], dims[idx+1], init_type))

        # Standard linear output layer
        self.output_layer = nn.Linear(dims[-1], output_dim)

        if self.output_layer.bias is not None:
            with torch.no_grad():
                self.output_layer.bias.zero_()

        # Apply selected initialization
        if init_type == "glorot_uniform":
            init_glorot_uniform(self.output_layer)
            #init_glorot_uniform(self.linear.bias)
        elif init_type == "glorot_normal":
            init_glorot_normal(self.output_layer)
            #init_glorot_normal(self.linear.bias)
        elif init_type == "he_normal":
            init_he_normal(self.output_layer)
            #init_he_normal(self.linear.bias) 
        elif init_type == "he_uniform":
            init_he_uniform(self.output_layer)
            #init_he_uniform(self.linear.bias)
        elif init_type == "he_truncated":
            init_he_normal_truncated(self.output_layer)
            #init_he_normal_truncated(self.linear.bias)
        elif init_type == "he_custom":
            self.init_he_normal_full(self.output_layer)
            #self.init_he_normal_full(self.linear.bias)
        elif init_type == "bad_uniform":
            nn.init.uniform_(self.output_layer.weight, a=0.1, b=1.0)
            nn.init.uniform_(self.output_layer.bias, a=0.1, b=1.0)
        else:
            # Default PyTorch init
            pass  

        self.to(device)

    def forward(self, x):
        x = x.to(device)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)  
        return x

    def update_gradients(self):
        for layer in self.hidden_layers:
            layer.compute_gradient_info()

    def get_layer_metrics(self):
        mean_sq_lengths = [layer.mean_squared_length for layer in self.hidden_layers]
        act_vars = [layer.activation_variance for layer in self.hidden_layers]
        grad_norms = [layer.gradient_norm for layer in self.hidden_layers]

        return {
            "mean_squared_length": mean_sq_lengths,
            "activation_variance": act_vars,
            "gradient_norm": grad_norms,
        }



"""
model = SubsidyNet(input_dim=784, hidden_dims=[256, 256], output_dim=10, epsilon=0.1, gamma=1.0, beta=0.001)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        step = epoch * len(train_loader) + batch_idx
        output = model(data, step)
        loss = F.cross_entropy(output, target)
        optimizer.zero_grad()
        loss.backward()
        model.update_gradients()  # Capture per-layer gradient norms
        optimizer.step()



"""

#====================================================

# === Decay Scheduler ===
class DecayScheduler:
    def __init__(self, beta=0.01, decay_type='exponential'):
        self.beta = beta
        self.decay_type = decay_type

    def get_decay(self, step):
        if self.decay_type == 'exponential':
            return torch.exp(torch.tensor(-self.beta * step, dtype=torch.float32)).item()
        elif self.decay_type == 'linear':
            return max(0.0, 1 - self.beta * step)
        else:
            return 1.0

#Subsidy Allocation using Gradient Norm 
def allocate_subsidy_gradient(grad_norm, epsilon, gamma, decay_value):
    gap = max(0.0, epsilon - grad_norm)
    return gamma * gap * decay_value

class SubsidyLinearV2(nn.Module):
    def __init__(self, in_features, out_features, layer_idx, init_type="glorot_uniform", epsilon=0.05, gamma=1.0, decay_scheduler=None, is_output_layer=False, depth = 1):
        super(SubsidyLinearV2, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.layer_idx = layer_idx
        self.epsilon = epsilon
        #self.gamma = gamma
        self.gamma = float(int(gamma) * depth)
        self.decay_scheduler = decay_scheduler
        self.init_type = init_type
        self.is_output_layer = is_output_layer

        # Add LayerNorm except for output layer
        self.layer_norm = nn.LayerNorm(out_features)
        
        # Apply initialization (same as before)
        if init_type == "glorot_uniform":
            nn.init.xavier_uniform_(self.linear.weight)
        elif init_type == "glorot_normal":
            nn.init.xavier_normal_(self.linear.weight)
        elif init_type == "he_normal":
            nn.init.kaiming_normal_(self.linear.weight, nonlinearity='relu')
        elif init_type == "he_uniform":
            nn.init.kaiming_uniform_(self.linear.weight, nonlinearity='relu')
        elif init_type == "he_truncated":
            nn.init.trunc_normal_(self.linear.weight, std=0.02)
        elif init_type == "bad_uniform":
            nn.init.uniform_(self.linear.weight, a=0.1, b=1.0)
            nn.init.uniform_(self.linear.bias, a=0.1, b=1.0)

        self.subsidy_value = 0.0
        self.mean_squared_length = 0.0
        self.activation_variance = 0.0
        self.gradient_norm = 0.0

    def forward(self, x, current_step, apply_subsidy=False, initial_subsidy=False):
        z = self.linear(x)
        
        # Apply layer norm if not output layer
        if self.layer_norm is not None:
            z = self.layer_norm(z)

        if apply_subsidy and (initial_subsidy or not self.is_output_layer):
            self.mean_squared_length = (z.pow(2).sum(dim=1) / z.size(1)).mean().item()
            self.activation_variance = compute_activation_variance(z)

            decay = self.decay_scheduler.get_decay(current_step) if self.decay_scheduler else 1.0
            self.subsidy_value = allocate_subsidy(self.activation_variance, self.epsilon, self.gamma, decay)
            #print("Subsidy value",self.subsidy_value)
            subsidy_vector = torch.full_like(z, self.subsidy_value / z.size(1))
            z = z + subsidy_vector

            #z = z + self.subsidy_value

        return z if self.is_output_layer else F.relu(z)

    def compute_gradient_info(self):
        if self.linear.weight.grad is not None:
            self.gradient_norm = torch.norm(self.linear.weight.grad, p=2).item()
        else:
            self.gradient_norm = 0.0


#Network
class SubsidyNetV2(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, init_type="glorot_normal", epsilon=0.05, gamma=1.0, beta=0.01, depth =1):
        super(SubsidyNetV2, self).__init__()
        self.decay_scheduler = DecayScheduler(beta=beta,decay_type='exponential')
        self.layers = nn.ModuleList()

        dims = [input_dim] + hidden_dims + [output_dim]

        for idx in range(len(dims) - 1):
            is_output = (idx == len(dims) - 2) 
            self.layers.append(SubsidyLinearV2(dims[idx], dims[idx+1], layer_idx=idx,
                                               init_type=init_type,
                                               epsilon=epsilon, gamma=gamma,
                                               decay_scheduler=self.decay_scheduler,
                                               is_output_layer=is_output, depth=depth))

    def forward(self, x, step,apply_subsidy=False, initial_subsidy=False):
        
        
        for layer in self.layers[:-1]:
            x = layer(x, step,apply_subsidy,initial_subsidy)
        x = self.layers[-1](x, step,apply_subsidy,initial_subsidy)
        return x

    def update_gradients(self):
        for layer in self.layers:
            layer.compute_gradient_info()

    def get_layer_metrics(self):
        mean_sq_lengths = [layer.mean_squared_length for layer in self.layers]
        act_vars = [layer.activation_variance for layer in self.layers]
        grad_norms = [layer.gradient_norm for layer in self.layers]

        return {
            "mean_squared_length": mean_sq_lengths,
            "activation_variance": act_vars,
            "gradient_norm": grad_norms,
        }


#================================
class SubsidyLinearV3(nn.Module):
    def __init__(self, in_features, out_features, layer_idx, init_type="glorot_uniform",
                 epsilon=0.05, decay_scheduler=None, is_output_layer=False):
        super(SubsidyLinearV3, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.layer_idx = layer_idx
        self.epsilon = epsilon
        self.decay_scheduler = decay_scheduler
        self.is_output_layer = is_output_layer

        self.layer_norm = nn.LayerNorm(out_features) 

        # Apply selected initialization
        if init_type == "glorot_uniform":
            init_glorot_uniform(self.linear.weight)
            init_glorot_uniform(self.linear.bias)
        elif init_type == self.subsidy_value:
            init_glorot_normal(self.linear.weight)
            init_glorot_normal(self.linear.bias)
        elif init_type == "he_normal":
            init_he_normal(self.linear.weight)
            init_he_normal(self.linear.bias) 
        elif init_type == "he_uniform":
            init_he_uniform(self.linear.weight)
            init_he_uniform(self.linear.bias)
        elif init_type == "he_truncated":
            init_he_normal_truncated(self.linear.weight)
            init_he_normal_truncated(self.linear.bias)
        elif init_type == "he_custom":
            self.init_he_normal_full(self.linear)
            #self.init_he_normal_full(self.linear.bias)
        elif init_type == "bad_uniform":
            nn.init.uniform_(self.linear.weight, a=0.1, b=1.0)
            nn.init.uniform_(self.linear.bias, a=0.1, b=1.0)
        else:
            # Default PyTorch init
            pass  

        self.subsidy_value = 1e-10
        self.activation_variance = 1e-7  
        self.mean_squared_length = 0.0
        self.gradient_norm = 0.0

    def forward(self, x, current_step, apply_subsidy=False, initial_subsidy=False):
        
        
        z = self.linear(x)
        
        
        #if x == x:
        if apply_subsidy== True:
            self.mean_squared_length = (z.pow(2).sum(dim=1) / z.size(1)).mean().item()
            # Here we can use other signals besides activation variance, like fisher info ..
            #self.activation_variance = compute_activation_variance(z)
            #self.activation_variance =  compute_fisher_information(self.linear.weight)
            self.activation_variance =  compute_gradient_norm(self.linear.weight)
            
            #with torch.no_grad():
            #    noise = torch.randn_like(self.linear.weight) * self.subsidy_value
                #self.linear.weight += noise 
            
            #subsidized_weight = self.linear.weight + noise
            #z = F.linear(x, subsidized_weight, self.linear.bias)
            
            #fisher_bias = compute_fisher_information(self.linear.bias)

            #self.activation_variance =compute_gradient_norm(self.linear.weight.grad)
            
            # Random tensor with small values, scaled by subsidy_value
            #noise = torch.randn_like(z) * 0.01  # or some other small scale
            #subsidy_vector = noise * self.subsidy_value

            #subsidy_strength = self.subsidy_value / self.activation_variance  # inverse weight importance
            #subsidy_vector = noise * subsidy_strength 
            #subsidy_vector = noise * subsidy_strength
            # Inject controlled randomness based on subsidy_value

            # Higher subsidy → more noise injected
            #print("the subsidy", self.subsidy_value)
            #z = F.relu(z)
            #noise_scale = 1/ self.subsidy_value  
            #noise = torch.randn_like(z) * noise_scale # self.subsidy_value  #noise_scale
            #z = z + noise
            
            #z = z + self.subsidy_value #subsidy_vector
            # Normalize z across features for each sample (row-wise)
            #z_norm = z.norm(p=2, dim=1, keepdim=True) + 1e-8  
            #z = z / z_norm
            z_mean = z.mean(dim=1, keepdim=True)        # shape: (batch_size, 1)
            z_std = z.std(dim=1, keepdim=True)  
            #alive_ratio = (z > 0).float().mean().item()
            #subsidy2 = (1- (alive_ratio + 1e-9)) * self.subsidy_value 
            z = z +  self.subsidy_value 
            z_upper_clip = z_mean + 2 * z_std
            z = torch.minimum(z, z_upper_clip)
            #z_std = z.std().item()
            #subsidy = z_std * self.subsidy_value   
            #if self.subsidy_value >= 0.05:
            #subsidy_vector = torch.full_like(z, subsidy / z.size(1))
            #    z_std = z.std().item()
            
            

        
        """
        if apply_subsidy:
            delta_W = torch.randn_like(self.linear.weight) * self.subsidy_value
            delta_b = torch.randn_like(self.linear.bias) * self.subsidy_value
            z = F.linear(x, self.linear.weight + delta_W, self.linear.bias + delta_b)
        else:
            z = self.linear(x)

        if self.layer_norm is not None:
            
        
        """
        #z = self.layer_norm(z)
        z = F.relu(z)
        
        #z = z +  self.subsidy_value
        #return z if self.is_output_layer else F.relu(z)
        return z

    def compute_gradient_info(self):
        if self.linear.weight.grad is not None:
            self.gradient_norm = torch.norm(self.linear.weight.grad, p=2).item()
        else:
            self.gradient_norm = 0.0

"""
class SubsidyNetV3(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, depth=1, init_type="glorot_uniform",
                 epsilon=0.05, gamma=100.0, beta=0.001,):
        super(SubsidyNetV3, self).__init__()
        self.decay_scheduler = DecayScheduler(beta= (beta  * depth), decay_type='linear')
        self.initialGamma = gamma
        self.gamma = gamma
        
        self.epsilon = epsilon

        dims = [input_dim] + hidden_dims + [output_dim]
        self.layers = nn.ModuleList()

        for idx in range(len(dims) - 1):
            is_output = (idx == len(dims) - 2)
            self.layers.append(SubsidyLinearV3(
                in_features=dims[idx],
                out_features=dims[idx + 1],
                layer_idx=idx,
                init_type=init_type,
                epsilon=epsilon,
                decay_scheduler=self.decay_scheduler,
                is_output_layer=is_output
            ))

    def forward(self, x, step, apply_subsidy=False, initial_subsidy=False):

        if self.training and apply_subsidy:
            # Collect variances and normalize
            act_vars = [layer.activation_variance for layer in self.layers]
            total_var = sum(act_vars) + 1e-8
            norm_weights = [v / total_var for v in act_vars]
            #print("Min weight norm:", min(norm_weights)) 
             
            # Allocate subsidy
            for layer, weight in zip(self.layers, norm_weights):
                layer.subsidy_value = self.gamma * weight

            # Apply decay after allocation
            #decay = self.decay_scheduler.get_decay(step) if self.decay_scheduler else 1.0
            #self.gamma *= decay
        else:
            for layer in self.layers:
                layer.subsidy_value = 1e-10

        # Forward pass
        for layer in self.layers[:-1]:
            x = layer(x, step, apply_subsidy=apply_subsidy, initial_subsidy=initial_subsidy)
        x = self.layers[-1](x, step, apply_subsidy=apply_subsidy, initial_subsidy=initial_subsidy)
        return x

    def update_gradients(self):
        for layer in self.layers:
            layer.compute_gradient_info()

    def get_layer_metrics(self):
        mean_sq_lengths = [layer.mean_squared_length for layer in self.layers]
        act_vars = [layer.activation_variance for layer in self.layers]
        grad_norms = [layer.gradient_norm for layer in self.layers]

        return {
            "mean_squared_length": mean_sq_lengths,
            "activation_variance": act_vars,
            "gradient_norm": grad_norms,
        }
    def step_epoch(self, epoch):
        print("Current gamma = ",self.gamma)
        decay = self.decay_scheduler.get_decay(epoch) if self.decay_scheduler else 1.0
        self.gamma *= decay
        print("Current gamma = ",self.gamma)
"""

class SubsidyNetV3(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, depth=1, init_type="glorot_uniform",
                 epsilon=0.05, gamma=100.0, beta=0.001):
        super(SubsidyNetV3, self).__init__()
        self.decay_scheduler = DecayScheduler(beta=(beta * depth), decay_type='linear')
        self.initialGamma = gamma
        self.gamma = gamma
        self.epsilon = epsilon

        dims = [input_dim] + hidden_dims
        self.layers = nn.ModuleList()

        # Use SubsidyLinearV3 for hidden layers
        for idx in range(len(dims) - 1):
            self.layers.append(SubsidyLinearV3(
                in_features=dims[idx],
                out_features=dims[idx + 1],
                layer_idx=idx,
                init_type=init_type,
                epsilon=epsilon,
                decay_scheduler=self.decay_scheduler,
                is_output_layer=False
            ))

        # Use a regular output layer
        self.output_layer = nn.Linear(dims[-1], output_dim)
        # Apply selected initialization
        if init_type == "glorot_uniform":
            init_glorot_uniform(self.output_layer)
            #init_glorot_uniform(self.linear.bias)
        elif init_type == "glorot_normal":
            init_glorot_normal(self.output_layer)
            #init_glorot_normal(self.linear.bias)
        elif init_type == "he_normal":
            init_he_normal(self.output_layer)
            #init_he_normal(self.linear.bias) 
        elif init_type == "he_uniform":
            init_he_uniform(self.output_layer)
            #init_he_uniform(self.linear.bias)
        elif init_type == "he_truncated":
            init_he_normal_truncated(self.output_layer)
            #init_he_normal_truncated(self.linear.bias)
        elif init_type == "he_custom":
            self.init_he_normal_full(self.output_layer)
            #self.init_he_normal_full(self.linear.bias)
        elif init_type == "bad_uniform":
            nn.init.uniform_(self.output_layer.weight, a=0.1, b=1.0)
            nn.init.uniform_(self.output_layer.bias, a=0.1, b=1.0)
        else:
            # Default PyTorch init
            pass  
        
        self.to(device)
    def forward(self, x, step, apply_subsidy=False, initial_subsidy=False):
        if self.training and apply_subsidy:
            act_vars = [layer.activation_variance for layer in self.layers]
            total_var = sum(act_vars) + 1e-8
            norm_weights = [v / total_var for v in act_vars]
            for layer, weight in zip(self.layers, norm_weights):
                layer.subsidy_value = self.gamma * weight
        else:
            for layer in self.layers:
                layer.subsidy_value = 0

        # Forward through Subsidy layers
        for layer in self.layers:
            x = layer(x, step, apply_subsidy=apply_subsidy, initial_subsidy=initial_subsidy)

        # Final linear layer
        x = self.output_layer(x)

        return x 
    def update_gradients(self):
        for layer in self.layers:
            layer.compute_gradient_info()

    def get_layer_metrics(self):
        mean_sq_lengths = [layer.mean_squared_length for layer in self.layers]
        act_vars = [layer.activation_variance for layer in self.layers]
        grad_norms = [layer.gradient_norm for layer in self.layers]

        return {
            "mean_squared_length": mean_sq_lengths,
            "activation_variance": act_vars,
            "gradient_norm": grad_norms,
        }
    def step_epoch(self, epoch):
        print("Current gamma = ",self.gamma)
        decay = self.decay_scheduler.get_decay(epoch) if self.decay_scheduler else 1.0
        self.gamma *= decay
        print("Current gamma = ",self.gamma)
