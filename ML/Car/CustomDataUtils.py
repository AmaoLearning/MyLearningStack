from torch.utils.data import DataLoader
from CustomDataset import SignImageDataset
from torchvision import transforms
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

training_data = SignImageDataset(
    annotations_file=".\\label.csv",
    img_dir=".\\raw_data\\train\\left",
    transform=transform
)


train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)

# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
figure = plt.figure(figsize=(640, 640))
for i in range(1, 65):
    img, label = train_features[i].numpy().transpose(1, 2, 0), train_labels[i]
    figure.add_subplot(8, 8, i)
    plt.title(label)
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()