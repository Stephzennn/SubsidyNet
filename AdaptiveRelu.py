import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import TensorDataset, DataLoader

SEED = 0
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ThresholdReLUFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, t):
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=x.dtype, device=x.device)
        else:
            t = t.to(dtype=x.dtype, device=x.device)
        ctx.save_for_backward(x, t)
        y = torch.where(x > t, x, torch.zeros_like(x))
        return y

    @staticmethod
    def backward(ctx, grad_out):
        x, t = ctx.saved_tensors
        mask = (x > t).to(grad_out.dtype)
        grad_x = grad_out * mask
        # t is not learnable -> return None
        return grad_x, None

class ThresholdReLU(nn.Module):
    def forward(self, x, threshold):
        return ThresholdReLUFn.apply(x, threshold)

# quick test
act = ThresholdReLU()
x = torch.tensor([[-1.0, 0.0, 0.5, 2.0, 7.0, 5.5, 0.9]], requires_grad=True)
t = 3.0
y = act(x, t)         # -> [[0., 0., 0., 0., 7.0, 5.5, 0.]] (only values strictly > 3.0 pass)
y.sum().backward()    # OK

# %% ---------- data ----------
# pip install torchvision if needed
import torch
from torch.utils.data import DataLoader, random_split, TensorDataset
from torchvision import datasets, transforms

SEED = 1337
torch.manual_seed(SEED)
class Flatten:
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return t.view(-1)
def get_loaders(
    name="CIFAR10",
    data_dir="./data",
    batch_train=128,
    batch_val=256,
    val_frac=0.2,
    shuffle_train=True,
):
    """
    Returns loaders with each sample flattened to 1D for MLPs.
    Also returns (num_classes, input_shape) where input_shape=(flat_dim,)
    """
    name = name.upper()
    FlattenT = Flatten()

    if name in ["MNIST", "FASHIONMNIST"]:
        # 1×28×28 grayscale
        if name == "MNIST":
            mean, std = (0.1307,), (0.3081,)
            DatasetClass = datasets.MNIST
        else:  # FASHIONMNIST
            mean, std = (0.2860,), (0.3530,)
            DatasetClass = datasets.FashionMNIST

        T_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            FlattenT,  # <---- flatten for MLP
        ])
        T_eval = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            FlattenT,  # <---- flatten for MLP
        ])

        full_train_aug  = DatasetClass(root=data_dir, train=True,  download=True, transform=T_train)
        full_train_eval = DatasetClass(root=data_dir, train=True,  download=True, transform=T_eval)
        test_ds         = DatasetClass(root=data_dir, train=False, download=True, transform=T_eval)

        num_classes, input_shape = 10, (1 * 28 * 28,)

        # Split indices for train/val
        n_val = int(len(full_train_aug) * val_frac)
        n_tr  = len(full_train_aug) - n_val
        gen = torch.Generator().manual_seed(SEED)
        train_idx, val_idx = torch.utils.data.random_split(range(len(full_train_aug)), [n_tr, n_val], generator=gen)

        train_ds = torch.utils.data.Subset(full_train_aug, train_idx.indices)
        val_ds   = torch.utils.data.Subset(full_train_eval, val_idx.indices)

        train_loader = DataLoader(train_ds, batch_size=batch_train, shuffle=shuffle_train,
                                  num_workers=0, pin_memory=False)
        val_loader   = DataLoader(val_ds,   batch_size=batch_val,   shuffle=False,
                                  num_workers=0, pin_memory=False)
        test_loader  = DataLoader(test_ds,  batch_size=batch_val,   shuffle=False,
                                  num_workers=0, pin_memory=False)
        return train_loader, val_loader, test_loader, num_classes, input_shape

    elif name in ["CIFAR10", "CIFAR-10"]:
        # 3×32×32 RGB
        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        T_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            FlattenT,  # <---- flatten for MLP
        ])
        T_eval = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            FlattenT,  # <---- flatten for MLP
        ])

        full_train_aug  = datasets.CIFAR10(root=data_dir, train=True,  download=True, transform=T_train)
        full_train_eval = datasets.CIFAR10(root=data_dir, train=True,  download=True, transform=T_eval)
        test_ds         = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=T_eval)

        num_classes, input_shape = 10, (3 * 32 * 32,)

        n_val = int(len(full_train_aug) * val_frac)
        n_tr  = len(full_train_aug) - n_val
        gen = torch.Generator().manual_seed(SEED)
        train_idx, val_idx = torch.utils.data.random_split(range(len(full_train_aug)), [n_tr, n_val], generator=gen)

        train_ds = torch.utils.data.Subset(full_train_aug, train_idx.indices)
        val_ds   = torch.utils.data.Subset(full_train_eval, val_idx.indices)

        train_loader = DataLoader(train_ds, batch_size=batch_train, shuffle=shuffle_train, num_workers=0, pin_memory=False)
        val_loader   = DataLoader(val_ds,   batch_size=batch_val,   shuffle=False,        num_workers=0, pin_memory=False)
        test_loader  = DataLoader(test_ds,  batch_size=batch_val,   shuffle=False,        num_workers=0, pin_memory=False)
        return train_loader, val_loader, test_loader, num_classes, input_shape

    elif name in ["CIFAR100", "CIFAR-100"]:
        # 3×32×32 RGB, 100 fine-grained classes
        # Harder than CIFAR-10 — deep plain MLPs with poor init genuinely collapse here
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        T_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            FlattenT,
        ])
        T_eval = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            FlattenT,
        ])

        full_train_aug  = datasets.CIFAR100(root=data_dir, train=True,  download=True, transform=T_train)
        full_train_eval = datasets.CIFAR100(root=data_dir, train=True,  download=True, transform=T_eval)
        test_ds         = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=T_eval)

        num_classes, input_shape = 100, (3 * 32 * 32,)

        n_val = int(len(full_train_aug) * val_frac)
        n_tr  = len(full_train_aug) - n_val
        gen = torch.Generator().manual_seed(SEED)
        train_idx, val_idx = torch.utils.data.random_split(range(len(full_train_aug)), [n_tr, n_val], generator=gen)

        train_ds = torch.utils.data.Subset(full_train_aug, train_idx.indices)
        val_ds   = torch.utils.data.Subset(full_train_eval, val_idx.indices)

        train_loader = DataLoader(train_ds, batch_size=batch_train, shuffle=shuffle_train, num_workers=0, pin_memory=False)
        val_loader   = DataLoader(val_ds,   batch_size=batch_val,   shuffle=False,        num_workers=0, pin_memory=False)
        test_loader  = DataLoader(test_ds,  batch_size=batch_val,   shuffle=False,        num_workers=0, pin_memory=False)
        return train_loader, val_loader, test_loader, num_classes, input_shape

    elif name == "SVHN":
        # 3×32×32 RGB digits in the wild
        mean, std = (0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)
        T = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            FlattenT,  # <---- flatten for MLP
        ])
        full_train = datasets.SVHN(root=data_dir, split='train', download=True, transform=T)
        test_ds    = datasets.SVHN(root=data_dir, split='test',  download=True, transform=T)
        num_classes, input_shape = 10, (3 * 32 * 32,)

    else:
        raise ValueError("Supported datasets: MNIST, FashionMNIST, CIFAR10, CIFAR100, SVHN")

    # default split path for SVHN
    n_val = int(len(full_train) * val_frac)
    n_tr  = len(full_train) - n_val
    gen = torch.Generator().manual_seed(SEED)
    train_ds, val_ds = random_split(full_train, [n_tr, n_val], generator=gen)

    train_loader = DataLoader(train_ds, batch_size=batch_train, shuffle=shuffle_train, num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_val,   shuffle=False,        num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_val,   shuffle=False,        num_workers=0, pin_memory=False)
    return train_loader, val_loader, test_loader, num_classes, input_shape


# === EXAMPLE USAGE ===
# Choose one of: "MNIST", "FashionMNIST", "CIFAR10", "SVHN"
#train_loader, val_loader, test_loader, num_classes, input_shape = get_loaders("CIFAR10") 


train_loader, val_loader, test_loader, num_classes, input_shape = get_loaders("MNIST")
# If your model is an MLP, you can flatten inputs inside your training loop:
# xb = xb.view(xb.size(0), -1)
# For CNNs, keep the (C,H,W) shape and just pass xb to the conv net.


# %% ---------- models ----------
class AdaptiveThresholdMLP(nn.Module):
    """
    3 hidden layers with ThresholdReLU(threshold_i), followed by Linear output.
    thresholds: list of floats, e.g., [3.0, 2.0, 1.0]
    """
    def __init__(self, input_dim, hidden_dims, output_dim, thresholds):
        super().__init__()
        assert len(hidden_dims) == len(thresholds)
        self.thresholds = thresholds
        self.act = ThresholdReLU()

        dims = [input_dim] + hidden_dims
        self.hidden = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(len(hidden_dims))])
        self.out = nn.Linear(dims[-1], output_dim)

        for layer in self.hidden:
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)
        nn.init.kaiming_normal_(self.out.weight, nonlinearity="linear")
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        for layer, thr in zip(self.hidden, self.thresholds):
            x = layer(x)
            x = self.act(x, thr)   # gate by layer-specific threshold
        return self.out(x)         # no activation on last layer

class ReLUMlp(nn.Module):
    """Same architecture, but standard ReLU activations."""
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        dims = [input_dim] + hidden_dims
        self.hidden = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(len(hidden_dims))])
        self.out = nn.Linear(dims[-1], output_dim)
        self.act = nn.ReLU(inplace=False)

        for layer in self.hidden:
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)
        nn.init.kaiming_normal_(self.out.weight, nonlinearity="linear")
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        for layer in self.hidden:
            x = self.act(layer(x))
        return self.out(x)
    
class LeakyReLUMlp(nn.Module):
    """Same architecture, but LeakyReLU activations."""
    def __init__(self, input_dim, hidden_dims, output_dim, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = float(negative_slope)
        dims = [input_dim] + hidden_dims
        self.hidden = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(len(hidden_dims))])
        self.out = nn.Linear(dims[-1], output_dim)
        self.act = nn.LeakyReLU(negative_slope=self.negative_slope, inplace=False)

        # Kaiming init tuned for LeakyReLU: set nonlinearity='leaky_relu' and pass 'a'
        for layer in self.hidden:
            nn.init.kaiming_normal_(layer.weight, nonlinearity="leaky_relu", a=self.negative_slope)
            nn.init.zeros_(layer.bias)
        nn.init.kaiming_normal_(self.out.weight, nonlinearity="linear")
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        for layer in self.hidden:
            x = self.act(layer(x))
        return self.out(x)


# ── SubsidyNet ────────────────────────────────────────────────────────────────

class DecayScheduler:
    """Linear or exponential decay of the subsidy budget gamma over epochs."""
    def __init__(self, beta=0.01, decay_type='linear'):
        self.beta = beta
        self.decay_type = decay_type

    def get_decay(self, step):
        if self.decay_type == 'exponential':
            return float(torch.exp(torch.tensor(-self.beta * step, dtype=torch.float32)))
        elif self.decay_type == 'linear':
            return max(0.0, 1.0 - self.beta * step)
        return 1.0


class SubsidyLinearBlock(nn.Module):
    """
    Single hidden layer that tracks activation variance and accepts an
    additive subsidy value (set by the parent SubsidyMLP) before ReLU.
    """
    def __init__(self, in_features, out_features, layer_idx):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.layer_idx = layer_idx

        nn.init.kaiming_normal_(self.linear.weight, nonlinearity='relu')
        nn.init.zeros_(self.linear.bias)

        # Set by SubsidyMLP.forward() each pass; read here during forward
        self.subsidy_value = 0.0
        # Metrics tracked each forward pass
        self.activation_variance = 1e-7
        self.mean_squared_length = 0.0
        self.gradient_norm = 0.0

    def forward(self, x, apply_subsidy=False):
        z = self.linear(x)
        self.mean_squared_length = (z.pow(2).sum(dim=1) / z.size(1)).mean().item()
        self.activation_variance = z.var(unbiased=False).item()

        if apply_subsidy and self.subsidy_value != 0.0:
            z = z + self.subsidy_value

        return F.relu(z)

    def compute_gradient_info(self):
        if self.linear.weight.grad is not None:
            self.gradient_norm = torch.norm(self.linear.weight.grad, p=2).item()
        else:
            self.gradient_norm = 0.0


class SubsidyMLP(nn.Module):
    """
    MLP with hidden SubsidyLinearBlock layers and a plain Linear output head.

    Budget gamma is distributed via inverse-variance weighting each forward
    pass: layers with lower activation variance (struggling, near-dead)
    receive more subsidy.  Gamma decays linearly — call step_epoch(ep) once
    per epoch.  Subsidy is only active during training (model.train()).
    """
    def __init__(self, input_dim, hidden_dims, output_dim, gamma=10.0, beta=0.01):
        super().__init__()
        self.gamma = float(gamma)
        self.decay_scheduler = DecayScheduler(beta=beta, decay_type='linear')

        dims = [input_dim] + hidden_dims
        self.layers = nn.ModuleList([
            SubsidyLinearBlock(dims[i], dims[i + 1], layer_idx=i)
            for i in range(len(dims) - 1)
        ])

        self.output_layer = nn.Linear(dims[-1], output_dim)
        nn.init.kaiming_normal_(self.output_layer.weight, nonlinearity='linear')
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x, apply_subsidy=True):
        if self.training and apply_subsidy and self.gamma > 0:
            act_vars = [layer.activation_variance for layer in self.layers]
            inv_vars = [1.0 / (v + 1e-8) for v in act_vars]
            total_inv = sum(inv_vars)
            norm_weights = [iv / total_inv for iv in inv_vars]
            for layer, w in zip(self.layers, norm_weights):
                layer.subsidy_value = self.gamma * w
        else:
            for layer in self.layers:
                layer.subsidy_value = 0.0

        for layer in self.layers:
            x = layer(x, apply_subsidy=apply_subsidy)
        return self.output_layer(x)

    def step_epoch(self, epoch):
        decay = self.decay_scheduler.get_decay(epoch)
        self.gamma = max(0.0, self.gamma * decay)

    def update_gradients(self):
        for layer in self.layers:
            layer.compute_gradient_info()

    def get_layer_metrics(self):
        return {
            "mean_squared_length": [l.mean_squared_length for l in self.layers],
            "activation_variance":  [l.activation_variance  for l in self.layers],
            "gradient_norm":        [l.gradient_norm        for l in self.layers],
        }


# Re-seed so both start comparably

import math
from typing import List

def make_threshold_schedule(
    num_layers: int,
    kind: str,
    scale: float = 10.0,
    *,
    alpha: float = 1.5,     # Pareto shape (>0)
    lam: float = 0.3,       # Exponential rate (>0)
    sigma: float = 0.35,    # Normal width (0<sigma<=1, for 'normal')
    mode: str = "decreasing"  # 'decreasing' (half-normal) or 'centered' bell
) -> List[float]:
    """
    Returns a list of length `num_layers` of thresholds for your layers.

    Mapping to absolute thresholds:
      threshold_i = scale * weight_i
    With scale=10.0, a normalized weight of 0.1 -> threshold 1.0 (your rule).

    kind in {'pareto', 'exponential', 'normal', 'uniform'}.
      pareto:      decreasing heavy-head (w_i ∝ 1/(i+1)^alpha), w_0 ≈ 1
      exponential: decreasing (w_i ∝ exp(-lam * i)), w_0 = 1
      normal:      if mode='decreasing' → half-normal (peak at layer 0, then down)
                   if mode='centered'   → bell across depth (not strictly monotone)
      uniform:     constant across layers

    Tip: You can also multiply the returned list by a global factor if you want
         to shift all thresholds up or down after the fact.
    """
    assert num_layers >= 1, "num_layers must be >= 1"
    kind = kind.lower()

    if num_layers == 1:
        return [scale * 1.0]

    # positions 0..L-1 and normalized 0..1
    L = num_layers
    idx = list(range(L))
    u = [i / (L - 1) for i in idx]  # 0 at first layer → 1 at last layer

    # compute normalized weights in [0, 1] (w_0 ~ 1 for decreasing modes)
    if kind in ("pareto", "powerlaw", "power"):
        # decreasing Pareto-like: w_i ∝ 1 / (i+1)^alpha
        w = [1.0 / ((i + 1) ** alpha) for i in idx]
        m = max(w); w = [wi / m for wi in w]  # normalize to max=1

    elif kind in ("exponential", "exp"):
        # decreasing exponential: w_i ∝ e^{-lam * i}
        w = [math.exp(-lam * i) for i in idx]
        m = max(w); w = [wi / m for wi in w]

    elif kind in ("normal", "gaussian"):
        if mode == "decreasing":
            # half-normal anchored at layer 0: w(u) = exp(-0.5 * (u / sigma)^2)
            w = [math.exp(-0.5 * (ui / max(sigma, 1e-8))**2) for ui in u]
            m = max(w); w = [wi / m for wi in w]
        elif mode == "centered":
            # bell centered at middle: w(u) = exp(-0.5 * ((u-0.5)/sigma)^2)
            w = [math.exp(-0.5 * ((ui - 0.5) / max(sigma, 1e-8))**2) for ui in u]
            m = max(w); w = [wi / m for wi in w]
        else:
            raise ValueError("normal: mode must be 'decreasing' or 'centered'")

    elif kind in ("uniform", "const", "constant"):
        w = [1.0] * L

    else:
        raise ValueError("Unknown kind. Use 'pareto', 'exponential', 'normal', or 'uniform'.")

    # map to absolute thresholds via the 0.1->1.0 scale rule
    thresholds = [scale * wi for wi in w]
    return thresholds
# 10 hidden layers tapering from 512 → 16 (MNIST: 784-dim input, 10 classes)
HIDDEN_DIMS = [512, 256, 256, 128, 128, 64, 64, 32, 32, 16]
NUM_HIDDEN  = len(HIDDEN_DIMS)

thresholdsPareto  = make_threshold_schedule(NUM_HIDDEN, "pareto",      scale=4, alpha=1.9)
thresholdsExpo    = make_threshold_schedule(NUM_HIDDEN, "exponential", scale=4, lam=0.3)
thresholdsnormal  = make_threshold_schedule(NUM_HIDDEN, "normal",      scale=4, sigma=0.35)
thresholdsUniform = make_threshold_schedule(NUM_HIDDEN, "uniform",     scale=4)

print("Pareto thresholds: ",  thresholdsPareto)
print("Expo thresholds:   ",  thresholdsExpo)
print("Normal thresholds: ",  thresholdsnormal)
print("Uniform thresholds:",  thresholdsUniform)

torch.manual_seed(SEED)
thr_model = AdaptiveThresholdMLP(
    input_dim=input_shape[0], hidden_dims=HIDDEN_DIMS, output_dim=num_classes,
    thresholds=thresholdsPareto,
    # thresholds=thresholdsExpo
    # thresholds=thresholdsnormal
    # thresholds=thresholdsUniform
).to(device)

torch.manual_seed(SEED)
relu_model = ReLUMlp(
    input_dim=input_shape[0], hidden_dims=HIDDEN_DIMS, output_dim=num_classes
).to(device)

torch.manual_seed(SEED)
leaky_model = LeakyReLUMlp(
    input_dim=input_shape[0], hidden_dims=HIDDEN_DIMS, output_dim=num_classes, negative_slope=0.01
).to(device)

torch.manual_seed(SEED)
subsidy_model = SubsidyMLP(
    input_dim=input_shape[0], hidden_dims=HIDDEN_DIMS, output_dim=num_classes,
    gamma=10.0, beta=0.01
).to(device)

# %% ---------- training / eval helpers ----------
# CrossEntropyLoss for multi-class classification (expects raw logits + Long targets)
loss_fn = nn.CrossEntropyLoss()

def evaluate(model, loader):
    model.eval()
    total_loss, total_correct, count = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            # yb are class indices (Long) from the DataLoader — CrossEntropyLoss expects that
            loss = loss_fn(logits, yb)
            total_loss += loss.item() * xb.size(0)
            # argmax over class dimension to get predicted label
            total_correct += (logits.argmax(dim=1) == yb).sum().item()
            count += xb.size(0)
    return total_loss / max(count, 1), total_correct / max(count, 1)

def train(model, loader, epochs=50, lr=1e-3, tag="model"):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if ep % 10 == 0 or ep == 1:
            tr_loss, tr_acc = evaluate(model, train_loader)
            va_loss, va_acc = evaluate(model, val_loader)
            print(f"[{tag}] epoch {ep:03d} | train loss: {tr_loss:.4f} acc: {tr_acc:.3f} | val loss: {va_loss:.4f} acc: {va_acc:.3f}")
    return model

def train_subsidy(model, loader, epochs=50, lr=1e-3, tag="SubsidyMLP"):
    """Training loop for SubsidyMLP — passes apply_subsidy=True and calls step_epoch."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb, apply_subsidy=True)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            model.update_gradients()
            opt.step()
        model.step_epoch(ep)
        if ep % 10 == 0 or ep == 1:
            tr_loss, tr_acc = evaluate(model, train_loader)
            va_loss, va_acc = evaluate(model, val_loader)
            print(f"[{tag}] epoch {ep:03d} | train loss: {tr_loss:.4f} acc: {tr_acc:.3f} | val loss: {va_loss:.4f} acc: {va_acc:.3f} | gamma: {model.gamma:.4f}")
    return model

# %% ---------- run all ----------

print("Relu_model")
relu_model = train(relu_model, train_loader, epochs=20, lr=1e-3, tag="ReLU")

print("Thr_model")
thr_model  = train(thr_model,  train_loader, epochs=20, lr=1e-3, tag="ThresholdReLU[3,2,1]")

print("Leaky Relu Model")
leaky_model = train(leaky_model, train_loader, epochs=20, lr=1e-3, tag="LeakyReLU[0.01]")

print("SubsidyMLP")
subsidy_model = train_subsidy(subsidy_model, train_loader, epochs=20, lr=1e-3, tag="SubsidyMLP")

# %% ---------- final comparison ----------

thr_loss,     thr_acc     = evaluate(thr_model,     val_loader)
relu_loss,    relu_acc    = evaluate(relu_model,    val_loader)
leaky_loss,   leaky_acc   = evaluate(leaky_model,   val_loader)
subsidy_loss, subsidy_acc = evaluate(subsidy_model, val_loader)

print("\nFinal validation (CrossEntropy loss / accuracy):")
print(f"  ThresholdReLU (Pareto): loss={thr_loss:.4f}  acc={thr_acc:.3f}")
print(f"  ReLU:                   loss={relu_loss:.4f}  acc={relu_acc:.3f}")
print(f"  LeakyReLU[0.01]:        loss={leaky_loss:.4f}  acc={leaky_acc:.3f}")
print(f"  SubsidyMLP:             loss={subsidy_loss:.4f}  acc={subsidy_acc:.3f}")

# Sample predictions - pull a real batch from the test loader so dimensions match
thr_model.eval(); relu_model.eval(); leaky_model.eval(); subsidy_model.eval()
xb_sample, yb_sample = next(iter(test_loader))
xb_sample = xb_sample[:5].to(device)
with torch.no_grad():
    # argmax over class logits gives the predicted digit
    print("\nSample predicted classes (ThresholdReLU):", thr_model(xb_sample).argmax(dim=1).tolist())
    print("Sample predicted classes (ReLU):         ", relu_model(xb_sample).argmax(dim=1).tolist())
    print("Sample predicted classes (LeakyReLU):    ", leaky_model(xb_sample).argmax(dim=1).tolist())
    print("Sample predicted classes (SubsidyMLP):   ", subsidy_model(xb_sample).argmax(dim=1).tolist())
    print("True labels:                             ", yb_sample[:5].tolist())
