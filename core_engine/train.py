"""
Training script for VisionMambaClassifier.

Features:
- Windows safe (default num_workers=0 to prevent multiprocessing hangs)
- Automatic Mixed Precision (AMP) enabled for CUDA, clean standard precision for CPU
- Rich checkpoint saving (model_state_dict, class_names, val_acc, epoch)
- Tqdm progress bar for clean monitoring
- Transfer learning with frozen ResNet-18 early stages and trainable SSM + classifier
"""

import argparse
import os
import random
import time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from vision_mamba_model import VisionMambaClassifier, PLANT_CLASSES

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def maybe_subset(dataset, per_class_cap, seed=42):
    if not per_class_cap or per_class_cap <= 0:
        return dataset
    rng = random.Random(seed)
    by_class = {}
    for idx, (_, label) in enumerate(dataset.samples):
        by_class.setdefault(label, []).append(idx)
    keep = []
    for label, idxs in by_class.items():
        rng.shuffle(idxs)
        keep.extend(idxs[:per_class_cap])
    return Subset(dataset, keep)


def get_loaders(data_dir, batch_size, img_size, subset, num_workers):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_path = os.path.join(data_dir, 'train')
    val_path = os.path.join(data_dir, 'val')

    train_ds = datasets.ImageFolder(train_path, transform=train_tf)
    val_ds = datasets.ImageFolder(val_path, transform=val_tf)

    assert train_ds.classes == PLANT_CLASSES, (
        f"Dataset class folders {train_ds.classes} do not match "
        f"PLANT_CLASSES {PLANT_CLASSES}. Rename folders or reorder "
        f"PLANT_CLASSES so they match exactly."
    )

    train_ds = maybe_subset(train_ds, subset)
    val_ds = maybe_subset(val_ds, max(1, subset // 4) if subset else None)

    is_cuda = torch.cuda.is_available()
    loader_kwargs = {
        'num_workers': num_workers,
        'pin_memory': is_cuda,
    }
    if num_workers > 0:
        loader_kwargs['persistent_workers'] = True

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)
    return train_loader, val_loader


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    use_amp = (device.type == 'cuda')
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if use_amp:
                with torch.autocast(device_type='cuda'):
                    logits, _ = model(x)
            else:
                logits, _ = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser(description="Train Vision Mamba Plant Disease Classifier")
    parser.add_argument('--data_dir', required=True, help="Path to data directory containing train/ and val/")
    parser.add_argument('--epochs', type=int, default=5, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--img_size', type=int, default=176, help="Image resolution for training")
    parser.add_argument('--lr', type=float, default=3e-4, help="Learning rate")
    parser.add_argument('--num_workers', type=int, default=0, help="DataLoader workers (default 0 for Windows CPU)")
    parser.add_argument('--subset', type=int, default=0,
                        help='cap images per class for faster training (e.g. 100); 0 = use all data')
    parser.add_argument('--out', default='checkpoint.pt', help="Output checkpoint file path")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== Plant Mamba Classifier Training ===")
    print(f"Device: {device}")
    print(f"Data directory: {args.data_dir}")
    print(f"Image size: {args.img_size}x{args.img_size}, Batch size: {args.batch_size}")
    if args.subset:
        print(f"Subset mode: {args.subset} images/class for train, ~{max(1, args.subset // 4)} for val")

    train_loader, val_loader = get_loaders(
        args.data_dir, args.batch_size, args.img_size, args.subset, args.num_workers
    )
    print(f"Loaded: {len(train_loader.dataset)} training images, {len(val_loader.dataset)} validation images")

    model = VisionMambaClassifier(num_classes=len(PLANT_CLASSES)).to(device)

    # Freeze earlier stages of ResNet-18 (blocks 0 to 5), keep blocks 6 and 7 + SSM + classifier trainable
    for name, param in model.backbone.named_parameters():
        if not (name.startswith('6.') or name.startswith('7.')):
            param.requires_grad = False

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")

    optimizer = optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    use_cuda = (device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda') if use_cuda else None

    best_acc = 0.0
    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", unit="batch")
        
        for x, y in pbar:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            if use_cuda:
                with torch.autocast(device_type='cuda'):
                    logits, _ = model(x)
                    loss = criterion(logits, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, _ = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * x.size(0)
            pbar.set_postfix({'batch_loss': f"{loss.item():.4f}"})

        scheduler.step()
        train_loss = running_loss / max(len(train_loader.dataset), 1)
        val_acc = evaluate(model, val_loader, device)
        print(f"  -> Epoch {epoch+1} Result: Train Loss = {train_loss:.4f} | Val Accuracy = {val_acc*100:.2f}%")

        if val_acc > best_acc or epoch == 0:
            best_acc = val_acc
            save_payload = {
                'model_state_dict': model.state_dict(),
                'classes': PLANT_CLASSES,
                'val_acc': val_acc,
                'epoch': epoch + 1,
                'img_size': args.img_size,
            }
            torch.save(save_payload, args.out)
            # Also save raw state_dict for simple torch.load compatibility
            raw_out = os.path.splitext(args.out)[0] + "_weights.pt"
            torch.save(model.state_dict(), raw_out)
            print(f"  [Saved Best Checkpoint] -> {args.out} (Val Acc: {val_acc*100:.2f}%)")

    elapsed = time.time() - start_time
    print(f"\nTraining Complete in {elapsed/60:.1f} minutes!")
    print(f"Best Validation Accuracy: {best_acc*100:.2f}%")
    print(f"Checkpoint saved to: {os.path.abspath(args.out)}")


if __name__ == '__main__':
    main()
