import torch
from torch.utils.data import DataLoader
from datasets.ct_dataset import CTDataset
from losses.dice_loss import dice_loss
from utils import set_seed, save_checkpoint


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0

    for image, mask in loader:
        image = image.to(device)
        mask = mask.to(device)

        # Forward (prompts are assumed to be generated inside model or externally)
        logits = model(image)

        loss = dice_loss(logits, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = CTDataset(root_dir="data/train")
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = torch.load("model_init.pth").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    epochs = 100
    for epoch in range(epochs):
        loss = train_one_epoch(model, loader, optimizer, device)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {loss:.4f}")

        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, optimizer, epoch, "checkpoints")


if __name__ == "__main__":
    main()
