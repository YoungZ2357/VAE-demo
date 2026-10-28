import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

BINARIZE_MODES = ("none", "dynamic", "static")


class BinarizedMNIST(Dataset):
    """MNIST with {0,1} pixels, so the Bernoulli decoder's BCE is a true log-likelihood.
    "dynamic":   a fresh x ~ Bernoulli(p) every time an image is read (train set);
    "fixed":     one seeded draw, the same every epoch (test set);
    "threshold": p > 0.5."""

    def __init__(self, base, mode, seed=0):
        self.p = base.data.float().div(255).unsqueeze(1)  # [N, 1, 28, 28], as ToTensor gives
        self.targets = base.targets
        self.mode = mode
        if mode == "fixed":
            self.x = torch.bernoulli(self.p, generator=torch.Generator().manual_seed(seed))
        elif mode == "threshold":
            self.x = (self.p > 0.5).float()
        elif mode != "dynamic":
            raise ValueError(f"unknown mode {mode!r}")

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        x = torch.bernoulli(self.p[i]) if self.mode == "dynamic" else self.x[i]
        return x, int(self.targets[i])


def get_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_test_set(data_dir="./data", binarize="none", seed=0):
    """Test images exactly as get_loaders feeds them: (x [N, 784] float in [0, 1], labels [N])."""
    base = datasets.MNIST(data_dir, train=False, download=True)
    if binarize == "dynamic":
        x = BinarizedMNIST(base, "fixed", seed).x
    elif binarize == "static":
        x = BinarizedMNIST(base, "threshold").x
    elif binarize == "none":
        x = base.data.float().div(255)
    else:
        raise ValueError(f"binarize must be one of {BINARIZE_MODES}, got {binarize!r}")
    return x.reshape(len(x), -1), base.targets.clone()


def get_loaders(data_dir="./data", batch_size=128, num_workers=2, pin_memory=False,
                binarize="none", seed=0):
    """binarize: "none" (grayscale in [0, 1]) | "dynamic" (train resampled, test one fixed draw)
    | "static" (threshold at 0.5, both splits)."""
    if binarize not in BINARIZE_MODES:
        raise ValueError(f"binarize must be one of {BINARIZE_MODES}, got {binarize!r}")
    tf = transforms.ToTensor()  # [0, 1], no Normalize — required for BCE
    train = datasets.MNIST(data_dir, train=True, download=True, transform=tf)
    test = datasets.MNIST(data_dir, train=False, download=True, transform=tf)
    if binarize == "dynamic":
        train, test = BinarizedMNIST(train, "dynamic"), BinarizedMNIST(test, "fixed", seed)
    elif binarize == "static":
        train, test = BinarizedMNIST(train, "threshold"), BinarizedMNIST(test, "threshold")
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory,
                              persistent_workers=num_workers > 0)
    test_loader = DataLoader(test, batch_size=1000, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory,
                             persistent_workers=num_workers > 0)
    return train_loader, test_loader
