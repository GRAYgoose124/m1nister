import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from grid.load import train_loader, test_loader
from grid.viz import GradCam
from grid.model import GridNetwork


def train(model, device, train_loader, optimizer, criterion):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print(
        f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)"
    )


def visualize(model, device):
    cases = {i: [] for i in range(10)}
    cases_collected = {i: 0 for i in range(10)}

    for images, labels in test_loader:
        for i, label in enumerate(labels):
            label_item = label.item()
            if cases_collected[label_item] < 5:
                # accounting for batch size:
                cases[label_item].append(images[i].unsqueeze(0))
                cases_collected[label_item] += 1
            if all(count == 5 for count in cases_collected.values()):
                break
        if all(count == 5 for count in cases_collected.values()):
            break

    # Generate CAMs
    grad_cam = GradCam(model=model, target_layer=model.path1.conv2)
    cams = {}
    predictions = {}
    for digit, images in cases.items():
        cams[digit] = []
        predictions[digit] = []
        for image in images:
            img = image.to(device)
            output = model(img)

            cam = grad_cam.generate_cam(image, output, target_class=digit)
            cams[digit].append(cam)

            pred = output.max(1, keepdim=True)[1].item()
            predictions[digit].append(pred)

    # plotting
    fig, axs = plt.subplots(10, 10, figsize=(20, 20))

    for digit, images in cases.items():
        for i, image in enumerate(images):
            image_np = image[0][0].cpu().numpy()
            cam = cams[digit][i]
            # norm
            cam = (cam - cam.min()) / (cam.max() - cam.min())

            axs[digit, i * 2].imshow(image_np, cmap="gray")
            axs[digit, i * 2].set_title(
                f"Digit: {digit}, Pred: {predictions[digit][i]}"
            )
            axs[digit, i * 2].axis("off")

            axs[digit, i * 2 + 1].imshow(image_np, cmap="gray")
            axs[digit, i * 2 + 1].imshow(cam, cmap="jet", alpha=0.5)
            axs[digit, i * 2 + 1].set_title(f"CAM")
            axs[digit, i * 2 + 1].axis("off")

    plt.tight_layout()
    plt.show()


def main():
    # load model from pth or create a new one
    model = GridNetwork()

    try:
        model.load_state_dict(torch.load("grid.pth"))
        print("Loaded model from grid.pth")
    except FileNotFoundError:
        print("Creating new model...")

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    try:
        epoch = 0
        while True:
            train(model, device, train_loader, optimizer, criterion)
            test(model, device, test_loader, criterion)
            epoch += 1
    except KeyboardInterrupt:
        print("Interrupted")
        pass

    if epoch > 0:
        print("Saving model...")
        torch.save(model.state_dict(), "grid.pth")

    visualize(model, device)


if __name__ == "__main__":
    main()
