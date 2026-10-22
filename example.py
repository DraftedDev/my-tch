import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
from my_tch import Config, DatasetFetcher, Module, run, DataLoader, Accuracy, ErrorRate, Confidence


class ModelConfig(Config):
    lr = 0
    epochs = 0

    def set_defaults(self):
        self.lr = 0.002
        self.epochs = 3

    def from_dict(self, fields: dict):
        self.lr = fields.get("lr", 0.002)
        self.epochs = fields.get("epochs", 3)

    def to_dict(self) -> dict:
        return {"lr": self.lr, "epochs": self.epochs}


class MNISTDatasetFetcher(DatasetFetcher):
    def get_train_loader(self, arg: str) -> DataLoader:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        return DataLoader(trainset, batch_size=64, shuffle=True)

    def get_test_loader(self, arg: str) -> DataLoader:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        return DataLoader(testset, batch_size=64, shuffle=False)


class MNISTModel(Module):
    def __init__(self, config: Config):
        super().__init__(config)
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        return self.net(x)


config = ModelConfig(path="config.json")
mod = MNISTModel(config)

run(
    model=mod,
    dataset_fetcher=MNISTDatasetFetcher(),
    optimizer=optim.Adam(mod.parameters(), lr=config.lr),
    criterion=nn.CrossEntropyLoss(),
    metrics=[Accuracy(), ErrorRate(), Confidence()],
    epochs=config.epochs
)
