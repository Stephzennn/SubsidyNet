import torch
import torch.nn as nn
import torch.nn.functional as F
import math
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
        
        torch.nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(layer.bias)
        

# 2. He Uniform (Kaiming Uniform)
def init_he_uniform(layer):
    if isinstance(layer, nn.Linear):
        torch.nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
        
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
        torch.nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)


def init_glorot_normal(layer):
    if isinstance(layer, nn.Linear):
        torch.nn.init.xavier_normal_(layer.weight)
        nn.init.zeros_(layer.bias)



#Decay Scheduler
class DecayScheduler:
    def __init__(self, decay_type='exponential', beta=0.01):
        self.decay_type = decay_type
        self.beta = beta

    def get_decay(self, step):
        if self.decay_type == 'exponential':
           
            return torch.exp(torch.tensor(-self.beta * step, dtype=torch.float32))

        elif self.decay_type == 'linear':
            return max(0.0, 1 - self.beta * step)
        else:
            return 1.0



# Metric Functions 

"""
Below are function we use as a conduit or a signal for learning per layer.
"""
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

def normalize_fisher(FI_raw, min_val=0, max_val=90000):
    FI_clipped = max(min(FI_raw, max_val), min_val)
    return (FI_clipped - min_val) / (max_val - min_val)


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
            pass  
        
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
        x = self.layers[-1](x, step)  
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


#Vanilla Layer Class that can use several initializations


class VanillaLinear(nn.Module):
    def __init__(self, in_features, out_features, init_type="he_normal"):
        super(VanillaLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features,bias=True)

        # Store the initialization type as a string
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
        elif init_type == "he_custom":
            self.init_he_normal_full(self.linear)
        elif init_type == "bad_uniform":
            nn.init.uniform_(self.linear.weight, a=0.1, b=1.0)
            nn.init.uniform_(self.linear.bias, a=0.1, b=1.0)
        else:
            
            pass  
        

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
        """
        Here as we can see , we calculate the Mean Squared Length, at exactly the same place we do it for the SubsidyNet. 
        They must be congruent so that we can compaer both of them.
        """

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
        """
        This function is called from the outside after loss.backward ()
        """
        if self.linear.weight.grad is not None:
            self.gradient_norm = torch.norm(self.linear.weight.grad, p=2).item()
        else:
            self.gradient_norm = 0.0

class VanillaNet(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, init_type="he_normal"):
        super(VanillaNet, self).__init__()
        self.layers = nn.ModuleList()
        dims = [input_dim] + hidden_dims

        # Custom hidden layers
        for idx in range(len(dims) - 1):
            self.layers.append(VanillaLinear(dims[idx], dims[idx+1], init_type))

        # Standard linear output layer
        self.output_layer = nn.Linear(dims[-1], output_dim)

        if self.output_layer.bias is not None:
            with torch.no_grad():
                self.output_layer.bias.zero_()
        init_he_normal(self.output_layer)
        
        self.to(device)

    def forward(self, x):
        x = x.to(device)
        for layer in self.layers:
            x = layer(x)
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

#====================================================

#Subsidy Allocation using Gradient Norm 
def allocate_subsidy_gradient(grad_norm, epsilon, gamma, decay_value):
    gap = max(0.0, epsilon - grad_norm)
    return gamma * gap * decay_value


"""
Below is SubsidyNet version 2. The main difference is that the subsidy is applied by scaling it with a randomly
generated tensor matching the size of Z. This approach tests whether injecting randomness into the subsidy can
help the model escape the local minima discussed earlier.
"""
class SubsidyLinearV2(nn.Module):
    def __init__(self, in_features, out_features, layer_idx, init_type="glorot_uniform", epsilon=0.05, gamma=1.0, decay_scheduler=None, is_output_layer=False, depth = 1):
        super(SubsidyLinearV2, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.layer_idx = layer_idx
        self.epsilon = epsilon
        
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
            
            subsidy_vector = torch.full_like(z, self.subsidy_value / z.size(1))
            z = z + subsidy_vector

            

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
    

#below is version 3

class SubsidyLinearV3(nn.Module):
    def __init__(self, in_features, out_features, layer_idx, init_type="he_uniform",
                 epsilon=0.05, decay_scheduler=None, is_output_layer=False):
        super(SubsidyLinearV3, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.layer_idx = layer_idx
        self.epsilon = epsilon
        self.decay_scheduler = decay_scheduler
        self.is_output_layer = is_output_layer

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
           
        elif init_type == "he_custom":
            self.init_he_normal_full(self.linear)
            
        elif init_type == "bad_uniform":
            nn.init.uniform_(self.linear.weight, a=0.1, b=1.0)
            nn.init.uniform_(self.linear.bias, a=0.1, b=1.0)
        else:
            
            pass  

       
        self.subsidy_value = 0.0
        self.activation_variance = 1e-7  
        self.mean_squared_length = 0.0
        self.gradient_norm = 0.0

    def forward(self, x, current_step, apply_subsidy=False, initial_subsidy=False):
        z = self.linear(x)
        self.mean_squared_length = (z.pow(2).sum(dim=1) / z.size(1)).mean().item()
        self.activation_variance = compute_activation_variance(z)
        if apply_subsidy and not initial_subsidy:
            if self.subsidy_value != 0:
                z = z + self.subsidy_value
        z = F.relu(z)
        return z
    def compute_gradient_info(self):
        if self.linear.weight.grad is not None:
            self.gradient_norm = torch.norm(self.linear.weight.grad, p=2).item()
        else:
            self.gradient_norm = 0.0

"""
One of the bugs we encountered was that we had inadvertently included a ReLU activation at the output layer.
We initially attempted to fix this within the Linear class. However, the issue was deeper: while we apply the
subsidy to hidden layers, we do not apply it to the output layer. Therefore, the output layer’s initialization
should be handled separately, especially when using Glorot or other sub-optimal initializations. In this version,
we separate the output layer from the Linear class and always use He initialization for its weights. This results
in a more robust implementation.
"""
class SubsidyNetV3(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, depth=1, init_type="he_uniform",
                 epsilon=0.05, gamma=100.0, beta=0.01):
        super(SubsidyNetV3, self).__init__()
        self.decay_scheduler = DecayScheduler(beta=(beta * depth), decay_type='linear')
        self.initialGamma = gamma
        self.gamma = gamma
        self.epsilon = epsilon

        dims = [input_dim] + hidden_dims
        self.layers = nn.ModuleList()

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
        
        init_he_normal(self.output_layer)
        
        self.to(device)
    def forward(self, x, step, apply_subsidy=False, initial_subsidy=False):
        
        if self.training and apply_subsidy and self.gamma > 0:

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
        
        decay = self.decay_scheduler.get_decay(epoch) if self.decay_scheduler else 1.0
        
        self.gamma *= decay
        









#Below is version 4

#================================
"""
The below is a trial testing, in reference of this paper {cite the paper}
"""
def compute_c_and_stats(layer, z,  x, depth):
    with torch.no_grad():
        # Pre-activation output z
        #z = layer(x)  # shape: (batch_size, out_features)

        # Probability neuron is active (p(z > 0))
        p_active = (z > 0).float().mean().item()

        # Weight variance (unbiased=False for population variance)
        var_w = layer.weight.var(unbiased=False).item()

        # Compute depth scale c = 1 / |log(chi)| where chi = var_w * p_active
        chi = var_w * p_active
        if chi <= 0:
            c = float('inf')  # Avoid log(0) or negative
        else:
            c = 1.0 / abs(math.log(chi))

        # Also compute subsidized z, example subsidy shift s (you can customize)
        s = 2.0  # subsidy value, e.g. add 2 to z
        z_subsidized = z + s

        # Probability active after subsidy
        p_active_subsidized = (z_subsidized > 0).float().mean().item()

        return {
            'z': z,
            'z_subsidized': z_subsidized,
            'var_w': var_w,
            'p_active': p_active,
            'p_active_subsidized': p_active_subsidized,
            'c': c,
            'depth': depth
        }
    

class SubsidyLinearV4(nn.Module):
    def __init__(self, in_features, out_features, layer_idx, init_type="glorot_normal",
                 epsilon=0.05, decay_scheduler=None, is_output_layer=False):
        super(SubsidyLinearV4, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.layer_idx = layer_idx
        self.epsilon = epsilon
        self.decay_scheduler = decay_scheduler
        self.is_output_layer = is_output_layer

         

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
           
        elif init_type == "he_custom":
            self.init_he_normal_full(self.linear)
            #self.init_he_normal_full(self.linear.bias)
        elif init_type == "bad_uniform":
            nn.init.uniform_(self.linear.weight, a=0.1, b=1.0)
            nn.init.uniform_(self.linear.bias, a=0.1, b=1.0)
        else:
            # Default PyTorch init
            pass  

        if self.linear.bias is not None:
            with torch.no_grad():
                self.linear.bias.zero_()
       

        self.subsidy_value = 0.0
        self.activation_variance = 1e-7  
        self.mean_squared_length = 0.0
        self.gradient_norm = 0.0

    def forward(self, x, current_step, apply_subsidy=False, initial_subsidy=False, roundSubsidy= 1):

        z = self.linear(x)
        
        self.mean_squared_length = (z.pow(2).sum(dim=1) / z.size(1)).mean().item()
        """
        The below are the different trials we are doing 
        self.activation_variance = compute_fisher_information(self.linear.weight)
        self.activation_variance = compute_fisher_information(self.linear.weight)
        fisher_info_clipped = normalize_fisher(self.activation_variance) 
        self.activation_variance = 1.0 - fisher_info_clipped  #  
        # Use previously computed Fisher info (computed after .backward())
        """
        
        if self.training:
            scaled_subsidy = self.subsidy_value * self.activation_variance
            eps = 1e-6
            """
            THE BELOW ARE THE DIFFERENT TRIALS WE ARE DOING
            fisher_info_clipped = min(max(self.activation_variance, eps), 1.0)  
            p = torch.clamp((z.var().item()), 0.0, 1.0)
            p = 1 -  p
            scaled_subsidy =self.mean_squared_length
            Compute mean and std of pre-activation z (over batch and features)
            z_mean = z.mean().item()
            z_std = z.std().item()
            max_allowed_subsidy = z_mean + (z_std *2)
            # Clamp subsidy so it does not exceed mean + std
            if scaled_subsidy > max_allowed_subsidy:
                scaled_subsidy = max_allowed_subsidy
            scaled_subsidy = p
            z = z * (1.0 + scaled_subsidy)
            """
            z_std = z.std().item()
            negative_mask = (z < 0).float()  # 1 where z < 0, 0 elsewhere
            z_updated = z + scaled_subsidy * negative_mask
            #z = z + self.subsidy_value * negative_mask
            over_limit_mask = (z_updated > z_std) & (negative_mask.bool())
            z = torch.where(over_limit_mask, z_std, z_updated)
            z = z + self.subsidy_value
            

        """
        if apply_subsidy and not initial_subsidy:
            self.activation_variance = compute_fisher_information(self.linear.weight)
            if self.subsidy_value != 0:
                z = z + self.subsidy_value
                stats = compute_c_and_stats(layer, x, depth)

        z = self.linear(x)  
        """
        z = F.relu(z)
        return z

    """
    def compute_gradient_info(self):
        if self.linear.weight.grad is not None:
            self.gradient_norm = torch.norm(self.linear.weight.grad, p=2).item()
        else:
            self.gradient_norm = 0.0
    """
    def compute_gradient_info(self):
        
        if self.linear.weight.grad is not None:
            
            self.gradient_norm = torch.norm(self.linear.weight.grad, p=2).item()
            # Compute and store Fisher info
            self.activation_variance = self.linear.weight.var().item()
            """
            #fisher_info_clipped = normalize_fisher(activation_variance) 
            #self.activation_variance = 1.0 - fisher_info_clipped
            """
        
        else:
            self.gradient_norm = 0.0
            self.activation_variance = 0.0
            
        


class SubsidyNetV4(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, depth=1, init_type="he_normal",
                 epsilon=0.05, gamma=100.0, beta=0.01):
        super(SubsidyNetV4, self).__init__()
        self.decay_scheduler = DecayScheduler(beta=(beta * depth), decay_type='linear')
        self.initialGamma = gamma
        self.gamma = gamma
        self.epsilon = epsilon

        self.dims = [input_dim] + hidden_dims
        self.layers = nn.ModuleList()

        # Use SubsidyLinearV3 for hidden layers
        for idx in range(len(self.dims) - 1):
            self.layers.append(SubsidyLinearV4(
                in_features= self.dims[idx],
                out_features=self.dims[idx + 1],
                layer_idx=idx,
                init_type=init_type,
                epsilon=epsilon,
                decay_scheduler=self.decay_scheduler,
                is_output_layer=False
            ))

        # Use a regular output layer
        self.output_layer = nn.Linear(self.dims[-1], output_dim)

        
        init_he_normal(self.output_layer)
        
        self.to(device)
    def forward(self, x, step =1, apply_subsidy=False, initial_subsidy=False):
        
        if self.training and apply_subsidy and self.gamma > 0:

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
            x = layer(x, step, apply_subsidy=apply_subsidy, initial_subsidy=initial_subsidy, roundSubsidy = self.gamma/ (len(self.dims) - 1)
 )

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
        
        decay = self.decay_scheduler.get_decay(epoch) if self.decay_scheduler else 1.0
        
        self.gamma *= decay
        

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional

# NOTE: The following names are expected to exist elsewhere in your codebase:
# - DecayScheduler: schedules a multiplicative decay factor over epochs/steps
# - SubsidyLinearV4: your custom linear layer that consumes `subsidy_value`
# - init_he_normal: a He-normal initializer for nn.Linear
# - `device` (optional): torch.device; if not defined, pass `device=` to the ctor


class SubsidyNetV4(nn.Module):
    """
    A feed-forward network whose hidden layers are SubsidyLinearV4 blocks
    that can receive a per-layer "subsidy" signal. The total subsidy budget
    `gamma` is distributed across layers based on their activation variances.

    Args
    ----
    input_dim : int
        Dimensionality of the input.
    hidden_dims : List[int]
        Sizes of hidden layers (each becomes a SubsidyLinearV4).
    output_dim : int
        Dimensionality of the final prediction head (plain Linear).
    depth : int, default=1
        Used to scale the decay scheduler's beta as (beta * depth).
    init_type : str, default="he_normal"
        Initialization type passed to SubsidyLinearV4.
    epsilon : float, default=0.05
        Small constant many layers use internally (passed through).
    gamma : float, default=100.0
        Total subsidy budget to distribute across hidden layers while training.
    beta : float, default=0.01
        Base decay-rate factor; effective rate is (beta * depth).

    Forward Flags
    -------------
    step : int
        Training step index you propagate to the hidden layers.
    apply_subsidy : bool
        If True (and model is in .train() mode and gamma > 0),
        distributes subsidy to hidden layers before the pass.
    initial_subsidy : bool
        Optional flag forwarded to SubsidyLinearV4 to treat the first rounds
        of training differently (if that layer supports it).

    Notes
    -----
    - Subsidy distribution is **variance-weighted**:
        weight_i = activation_variance_i / sum_j activation_variance_j
        subsidy_i = gamma * weight_i
    - If subsidies are disabled, each layer gets 0.
    - The output layer is a standard nn.Linear with He-normal init.
    """

    # ──────────────────────────────────────────────────────────────────────
    # [A] Construction & hyper-parameters
    # ──────────────────────────────────────────────────────────────────────
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        depth: int = 1,
        init_type: str = "he_normal",
        epsilon: float = 0.05,
        gamma: float = 100.0,
        beta: float = 0.01,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        # [A1] Global knobs for subsidy behavior
        self.initialGamma = float(gamma)     # keep a record of the starting budget
        self.gamma = float(gamma)            # mutable budget (decays over time)
        self.epsilon = float(epsilon)

        # [A2] Decay scheduler for gamma (external component)
        self.decay_scheduler = DecayScheduler(beta=(beta * depth), decay_type="linear")

        # [A3] Layer layout
        self.dims = [int(input_dim)] + [int(h) for h in hidden_dims]
        self.layers = nn.ModuleList()

        # ──────────────────────────────────────────────────────────────────
        # [B] Build hidden stack from SubsidyLinearV4
        # ──────────────────────────────────────────────────────────────────
        for idx in range(len(self.dims) - 1):
            self.layers.append(
                SubsidyLinearV4(
                    in_features=self.dims[idx],
                    out_features=self.dims[idx + 1],
                    layer_idx=idx,
                    init_type=init_type,
                    epsilon=self.epsilon,
                    decay_scheduler=self.decay_scheduler,
                    is_output_layer=False,
                )
            )

        # ──────────────────────────────────────────────────────────────────
        # [C] Output head (plain Linear) + initialization
        # ──────────────────────────────────────────────────────────────────
        self.output_layer = nn.Linear(self.dims[-1], output_dim)
        init_he_normal(self.output_layer)

        # [A4] Optional device placement
        if device is not None:
            self.to(device)

    # ──────────────────────────────────────────────────────────────────────
    # [D] Forward pass: allocate subsidy (variance-weighted) and run layers
    # ──────────────────────────────────────────────────────────────────────
    def forward(
        self,
        x: torch.Tensor,
        step: int = 1,
        apply_subsidy: bool = False,
        initial_subsidy: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through all hidden SubsidyLinearV4 layers + output layer.

        Important switches:
        - If training AND apply_subsidy AND gamma > 0:
            Distribute `self.gamma` across layers proportional to their
            `activation_variance` attribute (expected to be updated by layers).
        - Else:
            Zero-out the per-layer `subsidy_value`.
        """
        # [D1] Decide & distribute subsidy
        if self.training and apply_subsidy and (self.gamma > 0):
            # Gather per-layer activation variances (assumed scalar floats)
            act_vars = [float(getattr(layer, "activation_variance", 0.0)) for layer in self.layers]

            # Robust normalization to avoid division-by-zero
            total_var = sum(act_vars)
            if total_var <= 0.0:
                # Edge case: if we have no variance yet, distribute uniformly
                norm_weights = [1.0 / max(len(self.layers), 1)] * len(self.layers)
            else:
                norm_weights = [v / total_var for v in act_vars]

            # Assign each hidden layer its share of the subsidy budget
            for layer, weight in zip(self.layers, norm_weights):
                layer.subsidy_value = self.gamma * weight
        else:
            # No subsidies in eval mode or when disabled
            for layer in self.layers:
                layer.subsidy_value = 0.0

        # [D2] Run hidden stack (each layer may consume `subsidy_value`)
        #      We also pass a "roundSubsidy" convenience value.
        num_hidden = max(len(self.layers), 1)
        per_round_share = self.gamma / num_hidden
        for layer in self.layers:
            x = layer(
                x,
                step,
                apply_subsidy=apply_subsidy,
                initial_subsidy=initial_subsidy,
                roundSubsidy=per_round_share,
            )

        # ──────────────────────────────────────────────────────────────────
        # [F] Final projection
        # ──────────────────────────────────────────────────────────────────
        x = self.output_layer(x)
        return x

    # ──────────────────────────────────────────────────────────────────────
    # [G] Utility: tell layers to snapshot/compute gradient diagnostics
    # ──────────────────────────────────────────────────────────────────────
    def update_gradients(self) -> None:
        for layer in self.layers:
            # Expects SubsidyLinearV4 to implement `compute_gradient_info()`
            layer.compute_gradient_info()

    # ──────────────────────────────────────────────────────────────────────
    # [H] Utility: collect per-layer diagnostics for logging/plots
    # ──────────────────────────────────────────────────────────────────────
    def get_layer_metrics(self) -> Dict[str, List[float]]:
        mean_sq_lengths = [float(getattr(layer, "mean_squared_length", 0.0)) for layer in self.layers]
        act_vars = [float(getattr(layer, "activation_variance", 0.0)) for layer in self.layers]
        grad_norms = [float(getattr(layer, "gradient_norm", 0.0)) for layer in self.layers]

        return {
            "mean_squared_length": mean_sq_lengths,
            "activation_variance": act_vars,
            "gradient_norm": grad_norms,
        }

    # ──────────────────────────────────────────────────────────────────────
    # [I] Scheduler step: decay the total subsidy budget `gamma`
    # ──────────────────────────────────────────────────────────────────────
    def step_epoch(self, epoch: int) -> None:
        """
        Update gamma with the decay factor provided by DecayScheduler.
        Call this once per epoch (or however your training loop defines epochs).
        """
        decay = self.decay_scheduler.get_decay(epoch) if self.decay_scheduler else 1.0
        self.gamma *= float(decay)
        # (Optional) Clamp if you never want gamma to go negative due to FP noise:
        # self.gamma = max(self.gamma, 0.0)
