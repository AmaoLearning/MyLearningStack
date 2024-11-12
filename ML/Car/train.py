from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms
from Network import NeuralNetwork
from torch import nn
from torch.optim.lr_scheduler import StepLR
import torch



def train_loop(dataloader, model, loss_fn, optimizer, device='cuda'):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        X, y = X.to(device), y.to(device) #将图像转移到gpu上
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn, device='cuda'):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

if __name__ == '__main__':
    device = (
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    num_workers = 4
    learning_rate = 1e-4
    batch_size = 32

    model = NeuralNetwork().to(device=device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    epochs = 10

    my_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # 确保图像是单通道灰度图
        transforms.Resize((28, 28)),  # 调整图像大小为 28x28
        transforms.RandomRotation(8),
        transforms.ToTensor(),  # 将图像转换为张量
        transforms.Normalize((0.5,), (0.5,))  # 归一化图像数据
    ])

    try:
        dict = torch.load("model.pth", weights_only=True)
        model.load_state_dict(dict["model_state"])
        optimizer.load_state_dict(dict["optimizer_state"])
    except FileNotFoundError:
        print("weights not found, train from the beginning")



    train_data = datasets.ImageFolder(
        root=".\\raw_data\\train",
        transform=my_transform
    )
    test_data = datasets.ImageFolder(
        root=".\\raw_data\\val",
        transform=my_transform
    )
    print(train_data.class_to_idx)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_loader, model, loss_function, optimizer, device)
        test_loop(test_loader, model, loss_function, device)
        scheduler.step()
    print("Done!")

    torch.save({
    'model_state': model.state_dict(),
    'optimizer_state': optimizer.state_dict(),
}, 'model.pth')