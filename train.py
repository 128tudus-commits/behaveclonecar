import argparse
import csv
import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

TEST_SPLIT = 0.1


def parse_args():
    parser = argparse.ArgumentParser(description="Training a driving model on the dataset from record.py")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--width", type=int, default=160, help="Network input width")
    parser.add_argument("--height", type=int, default=120, help="Network input height")
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--data", default="", help="Path to the dataset (default: dataset/ next to the script)")
    return parser.parse_args()


class DriveDataset(Dataset):
    def __init__(self, root, width, height):
        self.width = width
        self.height = height
        self.samples = []
        labels_path = Path(root) / "labels.csv"
        images_dir = Path(root) / "images"
        with open(labels_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                path = images_dir / row["filename"]
                if path.exists():
                    self.samples.append(
                        (str(path), float(row["steering"]), float(row["throttle"]))
                    )
        if not self.samples:
            raise RuntimeError(f"No samples found in {labels_path}. Record data first using record.py")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, steering, throttle = self.samples[idx]
        img = cv2.imread(path)
        if img is None:
            raise RuntimeError(f"Cannot load {path}")
        img = cv2.resize(img, (self.width, self.height))
        if random.random() < 0.5:
            img = cv2.flip(img, 1)
            steering = -steering
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[..., 2] = np.clip(hsv[..., 2].astype(np.int16) * random.uniform(0.7, 1.3), 0, 255).astype(np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        img = img[:, :, ::-1].astype(np.float32) / 255.0
        img = np.ascontiguousarray(img.transpose(2, 0, 1))
        target = np.array([steering, throttle], dtype=np.float32)
        return torch.from_numpy(img), torch.from_numpy(target)


class PilotNet(nn.Module):
    def __init__(self, height, width):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2), nn.ReLU(),
            nn.Conv2d(24, 36, 5, stride=2), nn.ReLU(),
            nn.Conv2d(36, 48, 5, stride=2), nn.ReLU(),
            nn.Conv2d(48, 64, 3), nn.ReLU(),
            nn.Conv2d(64, 64, 3), nn.ReLU(),
        )
        with torch.no_grad():
            flat = self.features(torch.zeros(1, 3, height, width)).flatten(1).shape[1]
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 100), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(100, 50), nn.ReLU(),
            nn.Linear(50, 10), nn.ReLU(),
            nn.Linear(10, 2),
        )

    def forward(self, x):
        return self.head(self.features(x))


def evaluate(model, loader, criterion, device):
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for imgs, targets in loader:
            imgs, targets = imgs.to(device), targets.to(device)
            preds = model(imgs)
            losses = criterion(preds, targets).sum(dim=1)
            total += losses.sum().item()
            count += imgs.size(0)
    return total / max(count, 1)


def test_report(model, loader, device):
    model.eval()
    steer_errs = []
    throttle_errs = []
    with torch.no_grad():
        for imgs, targets in loader:
            preds = model(imgs.to(device)).cpu().numpy()
            targets = targets.numpy()
            steer_errs.append(np.abs(preds[:, 0] - targets[:, 0]))
            throttle_errs.append(np.abs(preds[:, 1] - targets[:, 1]))
    steer_err = np.concatenate(steer_errs)
    throttle_err = np.concatenate(throttle_errs)

    mse = (steer_err ** 2).mean() * 0.5 + (throttle_err ** 2).mean() * 0.5
    print("=== TEST SET REPORT ===")
    print(f"Test samples: {len(steer_err)}")
    print(f"Overall MSE: {mse:.6f}")
    print(f"Steering : MAE {steer_err.mean():.4f} | within +-0.03: {(steer_err <= 0.03).mean() * 100:.1f}% | "
          f"within +-0.07: {(steer_err <= 0.07).mean() * 100:.1f}% | within +-0.15: {(steer_err <= 0.15).mean() * 100:.1f}%")
    print(f"Throttle : MAE {throttle_err.mean():.4f} | within +-0.05: {(throttle_err <= 0.05).mean() * 100:.1f}% | "
          f"within +-0.10: {(throttle_err <= 0.10).mean() * 100:.1f}% | within +-0.25: {(throttle_err <= 0.25).mean() * 100:.1f}%")


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    base = Path(__file__).resolve().parent
    data_root = Path(args.data) if args.data else base / "dataset"

    dataset = DriveDataset(data_root, args.width, args.height)

    groups = {}
    for i, (path, _s, _t) in enumerate(dataset.samples):
        key = Path(path).stem.split("_")[0]
        groups.setdefault(key, []).append(i)
    keys = sorted(groups.keys())
    rng = random.Random(args.seed)
    rng.shuffle(keys)

    test_count = max(1, int(len(keys) * TEST_SPLIT))
    val_count = max(1, int(len(keys) * args.val_split))
    test_keys = keys[:test_count]
    val_keys = keys[test_count:test_count + val_count]
    train_keys = keys[test_count + val_count:]

    def indices_of(key_list):
        return [i for k in key_list for i in groups[k]]

    train_idx = indices_of(train_keys)
    val_idx = indices_of(val_keys)
    test_idx = indices_of(test_keys)
    print(f"Samples: {len(dataset)} total | train: {len(train_idx)} | val: {len(val_idx)} | test: {len(test_idx)} "
          f"(splits are grouped per source frame, so augmented copies never leak between splits)")

    train_set = torch.utils.data.Subset(dataset, train_idx)
    val_set = torch.utils.data.Subset(dataset, val_idx)
    test_set = torch.utils.data.Subset(dataset, test_idx)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = PilotNet(args.height, args.width).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
    criterion = nn.MSELoss(reduction="none")

    best_val = math.inf
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, targets).mean()
            loss.backward()
            optimizer.step()
            bs = imgs.size(0)
            running += loss.item() * bs
            seen += bs
        train_loss = running / max(seen, 1)
        val_loss = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}/{args.epochs} | train MSE: {train_loss:.6f} | val MSE: {val_loss:.6f} | lr: {lr_now:.2e}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"  -> new best score (val MSE: {val_loss:.6f})")

    if best_state is not None:
        model.load_state_dict(best_state)

    test_report(model, test_loader, device)

    onnx_path = base / "model.onnx"
    dummy = torch.zeros(1, 3, args.height, args.width)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["image"],
        output_names=["controls"],
        opset_version=17,
        dynamo=False,
    )
    print(f"Done. Best val MSE: {best_val:.6f}")
    print(f"Model saved as: {onnx_path}")


if __name__ == "__main__":
    main()
