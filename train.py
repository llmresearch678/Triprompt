import os
import torch
from torch.utils.data import DataLoader

from datasets.ct_dataset import CTDataset
from losses.dice_loss import dice_loss
from losses.contrastive_alignment import contrastive_alignment_loss
from utils import set_seed, save_checkpoint


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
    lambda_align: float = 0.1
):
    """
    Train the TRIPROMPT model for one epoch.

    Args:
        model (nn.Module):
            TRIPROMPT segmentation model.
        dataloader (DataLoader):
            Training data loader.
        optimizer (Optimizer):
            Optimizer (AdamW).
        device (torch.device):
            Training device.
        lambda_align (float):
            Weight for contrastive query–prompt alignment loss.

    Returns:
        float:
            Average training loss for the epoch.
    """

    model.train()
    running_loss = 0.0

    for image, mask in dataloader:
        image = image.to(device)
        mask = mask.to(device)

        # ---------------------------------------------------
        # Forward pass
        # ---------------------------------------------------
        # Expected model output:
        #   logits: (B, C, H, W, D)
        #   query_embeddings: (K, C)
        #   prompt_embeddings: (K, C)
        logits, query_emb, prompt_emb = model(image)

        # ---------------------------------------------------
        # Segmentation loss (Dice)
        # ---------------------------------------------------
        seg_loss = dice_loss(logits, mask)

        # ---------------------------------------------------
        # Query–Prompt alignment loss (contrastive)
        # ---------------------------------------------------
        align_loss = contrastive_alignment_loss(
            query_emb=query_emb,
            prompt_emb=prompt_emb
        )

        # ---------------------------------------------------
        # Total loss (Eq. 10 in paper – simplified weighting)
        # ---------------------------------------------------
        loss = seg_loss + lambda_align * align_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)


def main():
    """
    Main training entry point.
    """

    # ---------------------------------------------------
    # Reproducibility
    # ---------------------------------------------------
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------------------------------
    # Dataset & DataLoader
    # ---------------------------------------------------
    train_dataset = CTDataset(root_dir="data/train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # ---------------------------------------------------
    # Model initialization
    # ---------------------------------------------------
    if not os.path.exists("model_init.pth"):
        raise FileNotFoundError(
            "Initial model weights 'model_init.pth' not found."
        )

    model = torch.load("model_init.pth", map_location=device)
    model.to(device)

    # ---------------------------------------------------
    # Optimizer
    # ---------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-5
    )

    # ---------------------------------------------------
    # Training loop
    # ---------------------------------------------------
    num_epochs = 100
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(num_epochs):
        avg_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            lambda_align=0.1
        )

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"- Training Loss: {avg_loss:.4f}"
        )

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                save_dir="checkpoints"
            )


if __name__ == "__main__":
    main()
