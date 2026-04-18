import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

SEED = 0
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# %% ---------- ThresholdReLU ----------

class ThresholdReLUFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, t):
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=x.dtype, device=x.device)
        else:
            t = t.to(dtype=x.dtype, device=x.device)
        ctx.save_for_backward(x, t)
        return torch.where(x > t, x, torch.zeros_like(x))

    @staticmethod
    def backward(ctx, grad_out):
        x, t = ctx.saved_tensors
        mask = (x > t).to(grad_out.dtype)
        return grad_out * mask, None


class ThresholdReLU(nn.Module):
    def forward(self, x, threshold):
        return ThresholdReLUFn.apply(x, threshold)


# %% ---------- data ----------

SEED = 1337
torch.manual_seed(SEED)

class Flatten:
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return t.view(-1)

def get_loaders(
    name="CIFAR100",
    data_dir="./data",
    batch_train=128,
    batch_val=256,
    val_frac=0.2,
    shuffle_train=True,
):
    name = name.upper()
    FlattenT = Flatten()

    if name in ["MNIST", "FASHIONMNIST"]:
        if name == "MNIST":
            mean, std = (0.1307,), (0.3081,)
            DatasetClass = datasets.MNIST
        else:
            mean, std = (0.2860,), (0.3530,)
            DatasetClass = datasets.FashionMNIST
        T_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std), FlattenT])
        T_eval  = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std), FlattenT])
        full_train_aug  = DatasetClass(root=data_dir, train=True,  download=True, transform=T_train)
        full_train_eval = DatasetClass(root=data_dir, train=True,  download=True, transform=T_eval)
        test_ds         = DatasetClass(root=data_dir, train=False, download=True, transform=T_eval)
        num_classes, input_shape = 10, (1 * 28 * 28,)

    elif name in ["CIFAR10", "CIFAR-10"]:
        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        T_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(), transforms.Normalize(mean, std), FlattenT])
        T_eval  = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std), FlattenT])
        full_train_aug  = datasets.CIFAR10(root=data_dir, train=True,  download=True, transform=T_train)
        full_train_eval = datasets.CIFAR10(root=data_dir, train=True,  download=True, transform=T_eval)
        test_ds         = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=T_eval)
        num_classes, input_shape = 10, (3 * 32 * 32,)

    elif name in ["CIFAR100", "CIFAR-100"]:
        # 3×32×32 RGB, 100 fine-grained classes — harder problem, deeper separation needed
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

    elif name == "SVHN":
        mean, std = (0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)
        T = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std), FlattenT])
        full_train_aug  = datasets.SVHN(root=data_dir, split='train', download=True, transform=T)
        full_train_eval = full_train_aug
        test_ds         = datasets.SVHN(root=data_dir, split='test',  download=True, transform=T)
        num_classes, input_shape = 10, (3 * 32 * 32,)

    else:
        raise ValueError("Supported datasets: MNIST, FashionMNIST, CIFAR10, CIFAR100, SVHN")

    n_val = int(len(full_train_aug) * val_frac)
    n_tr  = len(full_train_aug) - n_val
    gen   = torch.Generator().manual_seed(SEED)
    train_idx, val_idx = torch.utils.data.random_split(range(len(full_train_aug)), [n_tr, n_val], generator=gen)

    train_ds = torch.utils.data.Subset(full_train_aug,  train_idx.indices)
    val_ds   = torch.utils.data.Subset(full_train_eval, val_idx.indices)

    train_loader = DataLoader(train_ds, batch_size=batch_train, shuffle=shuffle_train, num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_val,   shuffle=False,         num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_val,   shuffle=False,         num_workers=0, pin_memory=False)
    return train_loader, val_loader, test_loader, num_classes, input_shape


train_loader, val_loader, test_loader, num_classes, input_shape = get_loaders("CIFAR100")
# CIFAR-100: input_shape=(3072,), num_classes=100

# %% ---------- models ----------

# CIFAR-100 architecture: 10 hidden layers tapering from 2048 → 128
# (3072-dim input, 100 classes — depth amplifies the subsidy benefit)
HIDDEN_DIMS = [2048, 2048, 1024, 1024, 512, 512, 256, 256, 128, 128]
NUM_HIDDEN  = len(HIDDEN_DIMS)


class AdaptiveThresholdMLP(nn.Module):
    """MLP with per-layer ThresholdReLU gates (Pareto-scheduled thresholds)."""
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
            x = self.act(layer(x), thr)
        return self.out(x)


class ReLUMlp(nn.Module):
    """Standard ReLU MLP baseline."""
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
    """LeakyReLU MLP baseline."""
    def __init__(self, input_dim, hidden_dims, output_dim, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = float(negative_slope)
        dims = [input_dim] + hidden_dims
        self.hidden = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(len(hidden_dims))])
        self.out = nn.Linear(dims[-1], output_dim)
        self.act = nn.LeakyReLU(negative_slope=self.negative_slope, inplace=False)

        for layer in self.hidden:
            nn.init.kaiming_normal_(layer.weight, nonlinearity="leaky_relu", a=self.negative_slope)
            nn.init.zeros_(layer.bias)
        nn.init.kaiming_normal_(self.out.weight, nonlinearity="linear")
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        for layer in self.hidden:
            x = self.act(layer(x))
        return self.out(x)


# ── ResNet (residual MLP) ─────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """
    One residual block: ReLU( Linear(x) + shortcut(x) ).
    When in_features == out_features the shortcut is an identity (zero extra params).
    When they differ a bias-free linear projection aligns the dimensions,
    matching the ResNet convention for dimension-changing shortcuts.
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

        if in_features != out_features:
            self.shortcut = nn.Linear(in_features, out_features, bias=False)
            nn.init.kaiming_normal_(self.shortcut.weight, nonlinearity='linear')
        else:
            self.shortcut = nn.Identity()

        nn.init.kaiming_normal_(self.linear.weight, nonlinearity='relu')
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return F.relu(self.linear(x) + self.shortcut(x))


class ResidualMLP(nn.Module):
    """
    10-block residual MLP with the same hidden-dim layout as the other models.
    Each block: ReLU( W*x + shortcut(x) ), where the shortcut is identity when
    dims match and a linear projection when they change (same as ResNet-v1).
    Plain Linear output head — no activation, no residual.
    """
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        dims = [input_dim] + hidden_dims
        self.blocks = nn.ModuleList([
            ResidualBlock(dims[i], dims[i + 1])
            for i in range(len(dims) - 1)
        ])
        self.output_layer = nn.Linear(dims[-1], output_dim)
        nn.init.kaiming_normal_(self.output_layer.weight, nonlinearity='linear')
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.output_layer(x)


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
    additive subsidy (set by parent SubsidyMLP) before ReLU.
    """
    def __init__(self, in_features, out_features, layer_idx):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.layer_idx = layer_idx

        nn.init.kaiming_normal_(self.linear.weight, nonlinearity='relu')
        nn.init.zeros_(self.linear.bias)

        self.subsidy_value    = 0.0
        self.activation_variance = 1e-7
        self.mean_squared_length = 0.0
        self.gradient_norm    = 0.0

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
    def __init__(self, input_dim, hidden_dims, output_dim, gamma=50.0, beta=0.01):
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


# %% ---------- threshold schedules ----------

def make_threshold_schedule(
    num_layers: int,
    kind: str,
    scale: float = 10.0,
    *,
    alpha: float = 1.5,
    lam: float = 0.3,
    sigma: float = 0.35,
    mode: str = "decreasing",
) -> List[float]:
    assert num_layers >= 1
    kind = kind.lower()
    if num_layers == 1:
        return [scale * 1.0]

    L   = num_layers
    idx = list(range(L))
    u   = [i / (L - 1) for i in idx]

    if kind in ("pareto", "powerlaw", "power"):
        w = [1.0 / ((i + 1) ** alpha) for i in idx]
        m = max(w); w = [wi / m for wi in w]
    elif kind in ("exponential", "exp"):
        w = [math.exp(-lam * i) for i in idx]
        m = max(w); w = [wi / m for wi in w]
    elif kind in ("normal", "gaussian"):
        if mode == "decreasing":
            w = [math.exp(-0.5 * (ui / max(sigma, 1e-8))**2) for ui in u]
        elif mode == "centered":
            w = [math.exp(-0.5 * ((ui - 0.5) / max(sigma, 1e-8))**2) for ui in u]
        else:
            raise ValueError("normal: mode must be 'decreasing' or 'centered'")
        m = max(w); w = [wi / m for wi in w]
    elif kind in ("uniform", "const", "constant"):
        w = [1.0] * L
    else:
        raise ValueError("Unknown kind. Use 'pareto', 'exponential', 'normal', or 'uniform'.")

    return [scale * wi for wi in w]


# %% ---------- model instantiation ----------

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
    gamma=120.0, beta=0.01,
).to(device)

torch.manual_seed(SEED)
resnet_model = ResidualMLP(
    input_dim=input_shape[0], hidden_dims=HIDDEN_DIMS, output_dim=num_classes
).to(device)

# %% ---------- training / eval helpers ----------

loss_fn = nn.CrossEntropyLoss()

def evaluate(model, loader):
    model.eval()
    total_loss, total_correct, count = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            total_loss   += loss.item() * xb.size(0)
            total_correct += (logits.argmax(dim=1) == yb).sum().item()
            count += xb.size(0)
    return total_loss / max(count, 1), total_correct / max(count, 1)


def train(model, loader, epochs=100, lr=1e-3, tag="model"):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_acc  = 0.0
    best_val_loss = float('inf')
    best_tr_acc   = 0.0
    best_ep       = 0
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
            _, tr_acc = evaluate(model, train_loader)
            va_loss, va_acc = evaluate(model, val_loader)
            if va_acc > best_val_acc:
                best_val_acc  = va_acc
                best_val_loss = va_loss
                best_tr_acc   = tr_acc
                best_ep       = ep
            print(f"[{tag}] epoch {ep:03d} | best so far → ep {best_ep:03d} | train acc: {best_tr_acc:.3f} | val loss: {best_val_loss:.4f} acc: {best_val_acc:.3f}")
    return model


def train_subsidy(model, loader, epochs=100, lr=1e-3, tag="SubsidyMLP"):
    """Training loop for SubsidyMLP — passes apply_subsidy=True and calls step_epoch."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_acc  = 0.0
    best_val_loss = float('inf')
    best_tr_acc   = 0.0
    best_ep       = 0
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
            _, tr_acc = evaluate(model, train_loader)
            va_loss, va_acc = evaluate(model, val_loader)
            if va_acc > best_val_acc:
                best_val_acc  = va_acc
                best_val_loss = va_loss
                best_tr_acc   = tr_acc
                best_ep       = ep
            print(f"[{tag}] epoch {ep:03d} | best so far → ep {best_ep:03d} | train acc: {best_tr_acc:.3f} | val loss: {best_val_loss:.4f} acc: {best_val_acc:.3f} | gamma: {model.gamma:.4f}")
    return model


# %% ---------- run all ----------
# Order: SubsidyMLP → ResidualMLP → ReLU → LeakyReLU

print("SubsidyMLP")
subsidy_model = train_subsidy(subsidy_model, train_loader, epochs=100, lr=1e-3, tag="SubsidyMLP")

print("ResidualMLP")
resnet_model = train(resnet_model, train_loader, epochs=100, lr=1e-3, tag="ResidualMLP")

print("ReLU model")
relu_model = train(relu_model, train_loader, epochs=100, lr=1e-3, tag="ReLU")

print("LeakyReLU model")
leaky_model = train(leaky_model, train_loader, epochs=100, lr=1e-3, tag="LeakyReLU[0.01]")

# %% ---------- final comparison ----------

subsidy_loss, subsidy_acc = evaluate(subsidy_model, val_loader)
resnet_loss,  resnet_acc  = evaluate(resnet_model,  val_loader)
relu_loss,    relu_acc    = evaluate(relu_model,    val_loader)
leaky_loss,   leaky_acc   = evaluate(leaky_model,   val_loader)

print("\nFinal validation on CIFAR-100 (CrossEntropy loss / accuracy):")
print(f"  SubsidyMLP:       loss={subsidy_loss:.4f}  acc={subsidy_acc:.3f}")
print(f"  ResidualMLP:      loss={resnet_loss:.4f}  acc={resnet_acc:.3f}")
print(f"  ReLU:             loss={relu_loss:.4f}  acc={relu_acc:.3f}")
print(f"  LeakyReLU[0.01]:  loss={leaky_loss:.4f}  acc={leaky_acc:.3f}")

# Sample predictions from the test set
subsidy_model.eval(); resnet_model.eval(); relu_model.eval(); leaky_model.eval()
xb_sample, yb_sample = next(iter(test_loader))
xb_sample = xb_sample[:5].to(device)
with torch.no_grad():
    print("\nSample predicted classes (SubsidyMLP):   ", subsidy_model(xb_sample).argmax(dim=1).tolist())
    print("Sample predicted classes (ResidualMLP):  ", resnet_model(xb_sample).argmax(dim=1).tolist())
    print("Sample predicted classes (ReLU):         ", relu_model(xb_sample).argmax(dim=1).tolist())
    print("Sample predicted classes (LeakyReLU):    ", leaky_model(xb_sample).argmax(dim=1).tolist())
    print("True labels:                             ", yb_sample[:5].tolist())
