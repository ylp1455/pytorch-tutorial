import torch
from PIL import Image
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor


# Get data
train_data = datasets.MNIST(download=True, root="data", train=True, transform=ToTensor())

datasets = DataLoader(train_data, batch_size=32)


class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 3, 3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 3),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * (28 - 6) * (28 - 6), 10),
        )

    def forward(self, x):
        return self.model(x)


# Instance of NN, loss, optimizer

clf = ImageClassifier().to("cuda")
optimizer = Adam(clf.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()


if __name__ == "__main__":
    for epoch in range(10):
        for batch, (x, y) in enumerate(datasets):
            x = x.to("cuda")
            y = y.to("cuda")
            optimizer.zero_grad()
            pred = clf(x)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch} Batch {batch} Loss {loss.item()}")

    # Saving the model

    with open("model_state.pth", "wb") as f:
        save(clf.state_dict(), f)


# Loading the model

if __name__ == "__main__":
    with open("model_state.pth", "rb") as f:
        clf.load_state_dict(load(f))
        img = Image.open("images.jpeg")
        img_tensor = ToTensor()(img).unsqueeze(0).to("cuda")
        print(torch.argmax(clf(img_tensor)))
