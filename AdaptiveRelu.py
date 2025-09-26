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
y = act(x, t)         # -> [[0., 0., 0.5, 2.0]]
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
                                  num_workers=2, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_val,   shuffle=False,
                                  num_workers=2, pin_memory=True)
        test_loader  = DataLoader(test_ds,  batch_size=batch_val,   shuffle=False,
                                  num_workers=2, pin_memory=True)
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

        train_loader = DataLoader(train_ds, batch_size=batch_train, shuffle=shuffle_train, num_workers=2, pin_memory=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_val,   shuffle=False,        num_workers=2, pin_memory=True)
        test_loader  = DataLoader(test_ds,  batch_size=batch_val,   shuffle=False,        num_workers=2, pin_memory=True)
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
        raise ValueError("Supported datasets: MNIST, FashionMNIST, CIFAR10, SVHN")

    # default split path for SVHN
    n_val = int(len(full_train) * val_frac)
    n_tr  = len(full_train) - n_val
    gen = torch.Generator().manual_seed(SEED)
    train_ds, val_ds = random_split(full_train, [n_tr, n_val], generator=gen)

    train_loader = DataLoader(train_ds, batch_size=batch_train, shuffle=shuffle_train, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_val,   shuffle=False,        num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_val,   shuffle=False,        num_workers=2, pin_memory=True)
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
#thr_pareto = make_threshold_schedule(10, "pareto", scale=10.0, alpha=1.3)
thresholdsPareto = make_threshold_schedule(4, "pareto", scale=4, alpha=1.9)
print(thresholdsPareto )

thresholdsExpo = make_threshold_schedule(4, "exponential", scale=4, alpha=1.3)
print(thresholdsExpo)


thresholdsnormal = make_threshold_schedule(4, "normal", scale=4, alpha=1.3)
print(thresholdsnormal)

thresholdsUniform = make_threshold_schedule(4, "uniform", scale=4, alpha=1.3)
print(thresholdsUniform)
torch.manual_seed(SEED)
thr_model = AdaptiveThresholdMLP(
    input_dim=2, hidden_dims=[128, 64, 32, 32], output_dim=1,
    thresholds= thresholdsPareto 
    #thresholds= thresholdsnormal 
    #thresholds= thresholdsUniform
    #thresholds= thresholdsExpo 
    #thresholds = make_threshold_schedule(4, "pareto", scale=4, alpha=1.3)
).to(device)

torch.manual_seed(SEED)
relu_model = ReLUMlp(
    input_dim=2, hidden_dims=[128, 64, 32, 32], output_dim=1
).to(device)
torch.manual_seed(SEED)
leaky_model = LeakyReLUMlp(   # <-- use a separate var; keep your ReLU model intact
    input_dim=2, hidden_dims=[128, 64, 32, 32], output_dim=1, negative_slope=0.01
).to(device)

# %% ---------- training / eval helpers ----------
loss_fn = nn.MSELoss()

def evaluate(model, loader):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            total += loss.item() * xb.size(0)
            count += xb.size(0)
    return total / max(count, 1)

def train(model, loader, epochs=50, lr=1e-3, tag="model"):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(1, epochs + 1):
        print("Epoch : ", ep)
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if ep % 10 == 0 or ep == 1:
            tr = evaluate(model, train_loader)
            va = evaluate(model, val_loader)
            print(f"[{tag}] epoch {ep:03d} | train MSE: {tr:.4f} | val MSE: {va:.4f}")
    return model

# %% ---------- run all ----------
print("Relu_model")
relu_model = train(relu_model, train_loader, epochs=20, lr=1e-3, tag="ReLU")

print("Thr_model")
thr_model  = train(thr_model,  train_loader, epochs=20, lr=1e-3, tag="ThresholdReLU[3,2,1]")

print("Leaky Relu Model")
leaky_model = train(leaky_model, train_loader, epochs=20, lr=1e-3, tag="LeakyReLU[0.01]")

# %% ---------- final comparison ----------
thr_val   = evaluate(thr_model,   val_loader)
relu_val  = evaluate(relu_model,  val_loader)
leaky_val = evaluate(leaky_model, val_loader)
print("\nFinal validation MSE:")
print(f"  ThresholdReLU[3,2,1]: {thr_val:.4f}")
print(f"  ReLU:                 {relu_val:.4f}")
print(f"  LeakyReLU[0.01]:      {leaky_val:.4f}")

# quick sample preds
thr_model.eval(); relu_model.eval(); leaky_model.eval()
with torch.no_grad():
    xb = torch.tensor([[2.0, 7.5], [9.0, 1.0], [4.5, 4.5]], device=device)
    print("\nSample preds (ThresholdReLU):", thr_model(xb).squeeze().tolist())
    print("Sample preds (ReLU):         ", relu_model(xb).squeeze().tolist())
    print("Sample preds (LeakyReLU):    ", leaky_model(xb).squeeze().tolist())
