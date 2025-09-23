"""
FairFace multi-task training (Gender + AgeGroup).

Organize fairface data as:
    Root/
      ├── train/
      ├── val/
      ├── fairface_label_train.csv
      └── fairface_label_val.csv

We will:
- Use the official "train" split for training.
- Split the official "val" CSV 80:20 into validation:test.
- Train a single shared backbone with two classification heads.

Usage example:
python train_age_gender.py \
  --root openset2/dataset/fairface \
  --train_csv fairface_label_train.csv \
  --val_csv fairface_label_val.csv \
  --epochs 20 --batch_size 128 --lr 3e-4 --backbone resnet50

Switch backbones using: --backbone resnet50 | convnext_tiny | efficientnet_v2_s | vit_b_16
"""

import os, sys, argparse, json, random, math, time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

#config

AGE_LABELS = [
    '0-2','3-9','10-19','20-29','30-39','40-49','50-59','60-69','more than 70'
]

GENDER_LABELS = ['Male','Female']


#utils
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_age_label(x: str) -> str:
    x = str(x).strip()
    # map common variants
    if x in AGE_LABELS:
        return x
    # some CSVs use '70+' or 'more than 70'
    if x in ['70+', '70 +', 'more_than_70', 'more_than70', '70plus', '>=70']:
        return 'more than 70'
    return x


def normalize_gender_label(x: str) -> str:
    x = str(x).strip().lower()
    if x in ['m','male']:
        return 'Male'
    if x in ['f','female']:
        return 'Female'
    return x.capitalize()


def compute_class_weights(labels: List[int], num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / counts
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)

#dataset
class FairFaceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, root: str, img_col: str = 'file',
                 transform=None):
        self.df = df.reset_index(drop=True)
        self.root = root
        self.img_col = img_col
        self.transform = transform

        # build mappings
        self.age2idx = {a:i for i,a in enumerate(AGE_LABELS)}
        self.gen2idx = {g:i for i,g in enumerate(GENDER_LABELS)}

        # sanity: ensure labels are normalized
        self.df['age'] = self.df['age'].apply(normalize_age_label)
        self.df['gender'] = self.df['gender'].apply(normalize_gender_label)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row[self.img_col]
        if not os.path.isabs(img_path):
            # try root/train or root/val
            cand = [
                os.path.join(self.root, img_path),
                os.path.join(self.root, 'train', img_path),
                os.path.join(self.root, 'val', img_path),
                os.path.join(self.root, 'images', img_path),
            ]
            found = None
            for c in cand:
                if os.path.exists(c):
                    found = c
                    break
            if found is None:
                raise FileNotFoundError(f"Image not found for {img_path}")
            img_path = found

        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)

        age_label = self.age2idx[row['age']]
        gen_label = self.gen2idx[row['gender']]
        return img, torch.tensor(age_label), torch.tensor(gen_label)

#model
class MultiTaskNet(nn.Module):
    def __init__(self, backbone: str = 'resnet50', num_age: int = 9, num_gender: int = 2, pretrained: bool = True):
        super().__init__()
        self.backbone_name = backbone.lower()

        if self.backbone_name == 'resnet50':
            m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
            in_feats = m.fc.in_features
            m.fc = nn.Identity()
            self.feature_extractor = m
        elif self.backbone_name == 'convnext_tiny':
            m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
            in_feats = m.classifier[2].in_features
            m.classifier = nn.Sequential(*list(m.classifier.children())[:-1])  # drop final linear
            self.feature_extractor = m
        elif self.backbone_name == 'efficientnet_v2_s':
            m = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None)
            in_feats = m.classifier[1].in_features
            m.classifier = nn.Sequential(*list(m.classifier.children())[:-1])
            self.feature_extractor = m
        elif self.backbone_name == 'vit_b_16':
            m = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
            in_feats = m.heads.head.in_features
            m.heads.head = nn.Identity()
            self.feature_extractor = m
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.head_age = nn.Sequential(
            nn.Linear(in_feats, 512), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(512, num_age)
        )
        self.head_gender = nn.Sequential(
            nn.Linear(in_feats, 256), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(256, num_gender)
        )

    def forward(self, x):
        feats = self.feature_extractor(x)
        if feats.ndim > 2:
            feats = feats.mean(dim=[2,3])  # safety for conv nets
        logits_age = self.head_age(feats)
        logits_gen = self.head_gender(feats)
        return logits_age, logits_gen

#training
@dataclass
class TrainConfig:
    root: str
    train_csv: str
    val_csv: str
    img_col: str
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    backbone: str
    num_workers: int
    amp: bool
    seed: int

def make_transforms(backbone: str):
    # Pick weights enum + expected input size
    if backbone == 'resnet50':
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        size = 224
    elif backbone == 'convnext_tiny':
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        size = 224
    elif backbone == 'efficientnet_v2_s':
        weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        # EfficientNetV2-S default eval size is 384 in torchvision’s recipe
        size = 384
    elif backbone == 'vit_b_16':
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1
        size = 224
    else:
        raise ValueError(backbone)

    # Build normalize from weights meta (robust across torchvision versions)
    mean = weights.meta.get('mean', (0.485, 0.456, 0.406))
    std  = weights.meta.get('std',  (0.229, 0.224, 0.225))
    normalize = transforms.Normalize(mean=mean, std=std)

    train_tfms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomResizedCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
        transforms.ToTensor(),
        normalize,
    ])

    eval_tfms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        normalize,
    ])
    return train_tfms, eval_tfms



def prepare_data(cfg: TrainConfig):
    train_csv_path = os.path.join(cfg.root, cfg.train_csv) if not os.path.isabs(cfg.train_csv) else cfg.train_csv
    val_csv_path = os.path.join(cfg.root, cfg.val_csv) if not os.path.isabs(cfg.val_csv) else cfg.val_csv

    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)

    # normalize column names
    col_map = {c.lower(): c for c in val_df.columns}
    # image column fallback choices
    img_col = cfg.img_col
    if img_col not in train_df.columns:
        for cand in ['file','image_path','img','path','Filename','filename']:
            if cand in train_df.columns:
                img_col = cand
                break
    # standardize label columns
    for df in [train_df, val_df]:
        # attempt to unify column naming
        rename = {}
        for c in df.columns:
            lc = c.lower()
            if lc in ['gender']:
                rename[c] = 'gender'
            if lc in ['age', 'age_group', 'agegroup']:
                rename[c] = 'age'
            if lc in ['file','image_path','img','path','filename','filepath']:
                rename[c] = img_col
        if rename:
            df.rename(columns=rename, inplace=True)

    # filter to the classes of interest & drop NAs
    train_df = train_df.dropna(subset=['gender','age',img_col]).copy()
    val_df = val_df.dropna(subset=['gender','age',img_col]).copy()

    # split val 80:20 into valid:test using joint stratification (age+gender)
    joint = (val_df['age'].apply(normalize_age_label) + '_' + val_df['gender'].apply(normalize_gender_label))
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=cfg.seed)
    idx_valid, idx_test = next(splitter.split(val_df, joint))
    valid_df = val_df.iloc[idx_valid].reset_index(drop=True)
    test_df  = val_df.iloc[idx_test].reset_index(drop=True)

    return train_df.reset_index(drop=True), valid_df, test_df, img_col


def make_loaders(cfg: TrainConfig):
    train_df, valid_df, test_df, img_col = prepare_data(cfg)
    train_tfms, eval_tfms = make_transforms(cfg.backbone)

    ds_train = FairFaceDataset(train_df, cfg.root, img_col=img_col, transform=train_tfms)
    ds_valid = FairFaceDataset(valid_df, cfg.root, img_col=img_col, transform=eval_tfms)
    ds_test  = FairFaceDataset(test_df,  cfg.root, img_col=img_col, transform=eval_tfms)

    #class weighting for imbalance
    age_labels = [a for _,a,_ in DataLoader(ds_train, batch_size=1, shuffle=False, num_workers=0)]
    gen_labels = [g for *_,g in DataLoader(ds_train, batch_size=1, shuffle=False, num_workers=0)]
    age_labels = torch.stack(age_labels).squeeze().tolist()
    gen_labels = torch.stack(gen_labels).squeeze().tolist()

    age_weights = compute_class_weights(age_labels, num_classes=len(AGE_LABELS))
    gen_weights = compute_class_weights(gen_labels, num_classes=len(GENDER_LABELS))

    loader_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    loader_valid = DataLoader(ds_valid, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    loader_test  = DataLoader(ds_test,  batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    return loader_train, loader_valid, loader_test, age_weights, gen_weights


def train_one_epoch(model, loader, optimizer, device, scaler, loss_age, loss_gen, alpha=0.5):
    model.train()
    total_loss = 0.0
    all_age_pred, all_age_true = [], []
    all_gen_pred, all_gen_true = [], []

    for imgs, age_t, gen_t in loader:
        imgs = imgs.to(device, non_blocking=True)
        age_t = age_t.to(device)
        gen_t = gen_t.to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits_age, logits_gen = model(imgs)
            la = loss_age(logits_age, age_t)
            lg = loss_gen(logits_gen, gen_t)
            loss = alpha*la + (1-alpha)*lg
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward(); optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        all_age_pred.append(logits_age.argmax(1).detach().cpu())
        all_age_true.append(age_t.detach().cpu())
        all_gen_pred.append(logits_gen.argmax(1).detach().cpu())
        all_gen_true.append(gen_t.detach().cpu())

    age_pred = torch.cat(all_age_pred).numpy(); age_true = torch.cat(all_age_true).numpy()
    gen_pred = torch.cat(all_gen_pred).numpy(); gen_true = torch.cat(all_gen_true).numpy()

    metrics = {
        'loss': total_loss / len(loader.dataset),
        'age_acc': accuracy_score(age_true, age_pred),
        'age_f1m': f1_score(age_true, age_pred, average='macro'),
        'gen_acc': accuracy_score(gen_true, gen_pred),
        'gen_f1m': f1_score(gen_true, gen_pred, average='macro'),
    }
    return metrics


def evaluate(model, loader, device, loss_age, loss_gen, alpha=0.5):
    model.eval()
    total_loss = 0.0
    all_age_pred, all_age_true = [], []
    all_gen_pred, all_gen_true = [], []

    with torch.no_grad():
        for imgs, age_t, gen_t in loader:
            imgs = imgs.to(device, non_blocking=True)
            age_t = age_t.to(device)
            gen_t = gen_t.to(device)
            logits_age, logits_gen = model(imgs)
            la = loss_age(logits_age, age_t)
            lg = loss_gen(logits_gen, gen_t)
            loss = alpha*la + (1-alpha)*lg
            total_loss += loss.item() * imgs.size(0)

            all_age_pred.append(logits_age.argmax(1).cpu())
            all_age_true.append(age_t.cpu())
            all_gen_pred.append(logits_gen.argmax(1).cpu())
            all_gen_true.append(gen_t.cpu())

    age_pred = torch.cat(all_age_pred).numpy(); age_true = torch.cat(all_age_true).numpy()
    gen_pred = torch.cat(all_gen_pred).numpy(); gen_true = torch.cat(all_gen_true).numpy()

    metrics = {
        'loss': total_loss / len(loader.dataset),
        'age_acc': accuracy_score(age_true, age_pred),
        'age_f1m': f1_score(age_true, age_pred, average='macro'),
        'gen_acc': accuracy_score(gen_true, gen_pred),
        'gen_f1m': f1_score(gen_true, gen_pred, average='macro'),
        'age_cm': confusion_matrix(age_true, age_pred, labels=list(range(len(AGE_LABELS)))) ,
        'gen_cm': confusion_matrix(gen_true, gen_pred, labels=list(range(len(GENDER_LABELS)))) ,
    }
    return metrics


def save_confusion_matrices(metrics, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    age_cm = metrics['age_cm']
    gen_cm = metrics['gen_cm']
    np.savetxt(os.path.join(out_dir, 'confusion_matrix_age.csv'), age_cm, fmt='%d', delimiter=',')
    np.savetxt(os.path.join(out_dir, 'confusion_matrix_gender.csv'), gen_cm, fmt='%d', delimiter=',')


#main
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True, type=str, help='Root folder that contains images/CSVs')
    p.add_argument('--train_csv', default='fairface_label_train.csv', type=str)
    p.add_argument('--val_csv', default='fairface_label_val.csv', type=str)
    p.add_argument('--img_col', default='file', type=str, help='CSV column with relative image path')
    p.add_argument('--epochs', default=20, type=int)
    p.add_argument('--batch_size', default=128, type=int)
    p.add_argument('--lr', default=3e-4, type=float)
    p.add_argument('--weight_decay', default=1e-4, type=float)
    p.add_argument('--backbone', default='resnet50', type=str, choices=['resnet50','convnext_tiny','efficientnet_v2_s','vit_b_16'])
    p.add_argument('--num_workers', default=8, type=int)
    p.add_argument('--no_amp', action='store_true')
    p.add_argument('--seed', default=42, type=int)
    p.add_argument('--out_dir', default='runs/fairface_mt', type=str)
    p.add_argument('--results_csv', default='runs/benchmark_results.csv', type=str)
    args = p.parse_args()


    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cfg = TrainConfig(
        root=args.root,
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        img_col=args.img_col,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        backbone=args.backbone,
        num_workers=args.num_workers,
        amp=(not args.no_amp),
        seed=args.seed,
    )

    loader_train, loader_valid, loader_test, age_w, gen_w = make_loaders(cfg)

    model = MultiTaskNet(backbone=cfg.backbone, num_age=len(AGE_LABELS), num_gender=len(GENDER_LABELS), pretrained=True)
    model.to(device)

    # Losses (with class weights)
    loss_age = nn.CrossEntropyLoss(weight=age_w.to(device))
    loss_gen = nn.CrossEntropyLoss(weight=gen_w.to(device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    scaler = torch.cuda.amp.GradScaler() if (cfg.amp and device.type=='cuda') else None

    best_score = -1
    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(cfg.epochs):
        t0 = time.time()
        tr = train_one_epoch(model, loader_train, optimizer, device, scaler, loss_age, loss_gen, alpha=0.5)
        va = evaluate(model, loader_valid, device, loss_age, loss_gen, alpha=0.5)
        scheduler.step()
        dt = time.time()-t0

        # combined validation score
        combined = (va['age_f1m'] + va['gen_f1m'])/2.0
        is_best = combined > best_score
        if is_best:
            best_score = combined
            ckpt = {'model': model.state_dict(), 'cfg': cfg.__dict__, 'epoch': epoch}
            torch.save(ckpt, os.path.join(args.out_dir, 'best.pt'))
            torch.save(ckpt, os.path.join(args.out_dir, f"best_{cfg.backbone}_seed{cfg.seed}.pt"))


        print(f"Epoch {epoch+1:03d}/{cfg.epochs} | time {dt:.1f}s\n"
              f"  Train: loss={tr['loss']:.4f} age_acc={tr['age_acc']:.3f} age_f1={tr['age_f1m']:.3f} gen_acc={tr['gen_acc']:.3f} gen_f1={tr['gen_f1m']:.3f}\n"
              f"  Valid: loss={va['loss']:.4f} age_acc={va['age_acc']:.3f} age_f1={va['age_f1m']:.3f} gen_acc={va['gen_acc']:.3f} gen_f1={va['gen_f1m']:.3f}  best={best_score:.3f}")

    # final test eval (best checkpoint)
    ckpt = torch.load(os.path.join(args.out_dir, 'best.pt'), map_location='cpu')
    model.load_state_dict(ckpt['model'])
    te = evaluate(model, loader_test, device, loss_age, loss_gen, alpha=0.5)
    print(f"Test: loss={te['loss']:.4f} age_acc={te['age_acc']:.3f} age_f1 {te['age_f1m']:.3f} gen_acc={te['gen_acc']:.3f} gen_f1={te['gen_f1m']:.3f}")
    try:
        print('Confusion matrix (Age):')
        print(te['age_cm'])
        print('Confusion matrix (Gender):')
        print(te['gen_cm'])
    except Exception:
        pass


        #save CMs & a small report
    save_confusion_matrices(te, args.out_dir)
    report = {
        'age_labels': AGE_LABELS,
        'gender_labels': GENDER_LABELS,
        'test_metrics': {
            'loss': te['loss'],
            'age_acc': te['age_acc'], 'age_f1m': te['age_f1m'],
            'gen_acc': te['gen_acc'], 'gen_f1m': te['gen_f1m'],
        }
    }
    with open(os.path.join(args.out_dir, 'report.json'), 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Saved confusion matrices and report to: {args.out_dir}")

    #append a summary row to a global CSV for cross-backbone comparison
    try:
        os.makedirs(os.path.dirname(args.results_csv), exist_ok=True)
        summary = {
            'backbone': cfg.backbone,
            'epochs': cfg.epochs,
            'batch_size': cfg.batch_size,
            'lr': cfg.lr,
            'seed': cfg.seed,
            'val_best_score': float(best_score),
            'test_loss': float(te['loss']),
            'test_age_acc': float(te['age_acc']),
            'test_age_f1m': float(te['age_f1m']),
            'test_gen_acc': float(te['gen_acc']),
            'test_gen_f1m': float(te['gen_f1m']),
            'out_dir': args.out_dir,
            'ckpt_path': os.path.join(args.out_dir, 'best.pt')
        }
        df = pd.DataFrame([summary])
        if not os.path.exists(args.results_csv):
            df.to_csv(args.results_csv, index=False)
        else:
            df.to_csv(args.results_csv, mode='a', header=False, index=False)
        print(f"Results appended to {args.results_csv}")
    except Exception as e:
        print(f"WARNING: could not write results CSV: {e}")


if __name__ == '__main__':
    main()
    
"""
To run:

ConvNeXt-Tiny
python train_age_gender.py --root /home/jupyter-swt224/openset2/dataset/fairface \
  --backbone convnext_tiny --epochs 20 --batch_size 128 --lr 3e-4 --out_dir runs/convnext_t

EfficientNetV2-S
python train_age_gender.py --root /home/jupyter-swt224/openset2/dataset/fairface \
  --backbone efficientnet_v2_s --epochs 20 --batch_size 128 --lr 3e-4 --out_dir runs/effv2s

ResNet-50 (baseline)
python train_age_gender.py --root /home/jupyter-swt224/openset2/dataset/fairface \
  --backbone resnet50 --epochs 20 --batch_size 128 --lr 3e-4 --out_dir runs/rn50

ViT-B/16 
python train_age_gender.py --root /home/jupyter-swt224/openset2/dataset/fairface \
  --backbone vit_b_16 --epochs 20 --batch_size 64 --lr 3e-4 --out_dir runs/vit_b16
  
"""