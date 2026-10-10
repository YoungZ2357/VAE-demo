import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_loaders(data_dir="./data", batch_size=128, num_workers=2, pin_memory=False):
    tf = transforms.ToTensor()  # [0, 1], no Normalize — required for BCE
    train = datasets.MNIST(data_dir, train=True, download=True, transform=tf)
    test = datasets.MNIST(data_dir, train=False, download=True, transform=tf)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory,
                              persistent_workers=num_workers > 0)
    test_loader = DataLoader(test, batch_size=1000, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory,
                             persistent_workers=num_workers > 0)
    return train_loader, test_loader
