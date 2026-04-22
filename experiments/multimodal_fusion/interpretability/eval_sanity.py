#!/usr/bin/env python3
"""
Sanity check: load a saved paper checkpoint and reproduce per-fold metrics.

Evaluates cross_task_cv_results/model_seed{S}_fold{F}.pth on the matching
ablation_mri_only/cv_results/seed_{S}/fold_{F}/ test and cn_ad_test CSVs.

Replicates the training-time Test-Time Augmentation (8 flip combinations)
used to produce the numbers in cv_results/cv_summary.json.

Usage:
    python eval_sanity.py --seed 42 --fold 1 [--no-tta] [--batch-size 4]
"""

import argparse
import sys
from pathlib import Path

import torch
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score, confusion_matrix,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
from dataset import MultiModalDataset  # noqa: E402
from interpretability.ft_attention import (  # noqa: E402
    TABULAR_FEATURES, build_paper_model,
)


# <repo>/experiments/multimodal_fusion/interpretability/ -> <repo>
REPO = SCRIPT_DIR.parent.parent.parent
EXP = REPO / "experiments" / "multimodal_fusion"


def tta_flip(x, idx):
    """8 geometric augmentations matching train.py:test_with_tta."""
    if idx == 0:
        return x
    flips = {
        1: [2], 2: [3], 3: [4],
        4: [2, 3], 5: [2, 4], 6: [3, 4],
        7: [2, 3, 4],
    }
    return torch.flip(x, dims=flips[idx])


@torch.no_grad()
def evaluate(model, loader, device, use_tta=True, num_aug=8):
    model.eval()
    all_logits, all_labels = [], []
    for mri, tab, y in tqdm(loader, desc="TTA eval" if use_tta else "eval"):
        mri = mri.to(device, non_blocking=True)
        tab = tab.to(device, non_blocking=True)
        if use_tta:
            logits_stack = [model(tta_flip(mri, a), tab) for a in range(num_aug)]
            logits = torch.stack(logits_stack).mean(dim=0)
        else:
            logits = model(mri, tab)
        all_logits.append(logits.cpu())
        all_labels.append(y)
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels).numpy()
    probs = torch.softmax(logits, dim=-1)[:, 1].numpy()
    preds = logits.argmax(dim=-1).numpy()
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    return {
        "accuracy": 100 * accuracy_score(labels, preds),
        "balanced_accuracy": 100 * balanced_accuracy_score(labels, preds),
        "sensitivity": 100 * tp / (tp + fn) if (tp + fn) else 0.0,
        "specificity": 100 * tn / (tn + fp) if (tn + fp) else 0.0,
        "auc": roc_auc_score(labels, probs),
        "n": len(labels),
        "n_pos": int((labels == 1).sum()),
        "n_neg": int((labels == 0).sum()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-tta", action="store_true")
    p.add_argument(
        "--preset",
        choices=["new-cv", "old-fixed"],
        default="new-cv",
        help=(
            "new-cv: eval on ablation_mri_only/cv_results/seed_S/fold_F/ CSVs "
            "(post 'fix cross val' splits, contaminated for orphan checkpoints). "
            "old-fixed: eval on data/combined_trajectory/{train,val,test}.csv "
            "(pre-fix logic, the proper held-out test for orphan checkpoints)."
        ),
    )
    args = p.parse_args()

    ckpt_path = EXP / "cross_task_cv_results" / f"model_seed{args.seed}_fold{args.fold}.pth"

    if args.preset == "new-cv":
        fold_dir = (
            REPO / "experiments" / "ablation_mri_only" / "cv_results"
            / f"seed_{args.seed}" / f"fold_{args.fold}"
        )
        train_csv = fold_dir / "train.csv"
        val_csv = fold_dir / "val.csv"
        test_csv = fold_dir / "test.csv"
        cn_ad_csv = fold_dir / "cn_ad_test.csv"
        eval_pairs = [
            ("Val (identity check)", val_csv),
            ("AD-trajectory", test_csv),
            ("Established AD", cn_ad_csv),
        ]
    else:
        # old-fixed preset: use combined_trajectory/ as the proper test for orphan ckpts
        data_dir = EXP / "data" / "combined_trajectory"
        train_csv = data_dir / "train.csv"
        val_csv = data_dir / "val.csv"
        test_csv = data_dir / "test.csv"
        cn_ad_data_dir = EXP / "data" / "combined_cn_ad"
        cn_ad_csv = cn_ad_data_dir / "test.csv" if cn_ad_data_dir.exists() else None
        eval_pairs = [
            ("Val (old fixed)", val_csv),
            ("Test (old fixed, held-out)", test_csv),
        ]
        if cn_ad_csv is not None and cn_ad_csv.exists():
            eval_pairs.append(("CN vs Established AD (old fixed)", cn_ad_csv))

    for x in (ckpt_path, train_csv, val_csv, test_csv):
        if not x.exists():
            raise FileNotFoundError(f"missing: {x}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device}")
    print(f"[*] Checkpoint: {ckpt_path.name}, TTA: {'no' if args.no_tta else 'yes (8 flips)'}")

    print("[*] Fitting scaler on train set...")
    train_ds = MultiModalDataset(
        str(train_csv),
        tabular_features=TABULAR_FEATURES,
        target_shape=(128, 128, 128),
        augment=False,
        normalize_tabular=True,
        scaler=None,
        use_paper_preprocessing=True,
        target_spacing=1.75,
    )
    scaler = train_ds.get_scaler()

    print("[*] Building model and loading checkpoint...")
    model = build_paper_model()
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    out = model.load_state_dict(sd, strict=False)
    miss = [k for k in out.missing_keys if not k.startswith("vit.")]
    unexp = [k for k in out.unexpected_keys if not k.startswith("vit.")]
    print(f"    missing (excl. vit alias)={len(miss)}  unexpected={len(unexp)}")
    model.to(device)

    for name, csv in eval_pairs:
        print(f"\n[*] Evaluating on {name} ({csv.name})")
        ds = MultiModalDataset(
            str(csv),
            tabular_features=TABULAR_FEATURES,
            target_shape=(128, 128, 128),
            augment=False,
            normalize_tabular=True,
            scaler=scaler,
            use_paper_preprocessing=True,
            target_spacing=1.75,
        )
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        m = evaluate(model, loader, device, use_tta=not args.no_tta)
        print(f"    N={m['n']} (CN={m['n_neg']}, AD={m['n_pos']})")
        print(f"    accuracy          : {m['accuracy']:.2f}%")
        print(f"    balanced accuracy : {m['balanced_accuracy']:.2f}%")
        print(f"    sensitivity       : {m['sensitivity']:.2f}%")
        print(f"    specificity       : {m['specificity']:.2f}%")
        print(f"    AUC               : {m['auc']:.4f}")

    if args.preset == "new-cv":
        print("\n[*] Expected paper values (cv_summary.json, fold 1, seed 42):")
        print("    val   : acc=92.99% (identity check — if ~equal, checkpoint matches the paper run)")
        print("    traj  : acc=93.57%  sens=79.85%  spec=97.37%  AUC=0.9677")
        print("    cn_ad : acc=94.08%  AUC=0.9662")
    else:
        print("\n[*] Orphan-checkpoint preset (old-fixed). No paper value to match;")
        print("    these metrics reflect what the checkpoint actually achieves on its")
        print("    proper held-out set (pre 'fix cross val' split logic).")


if __name__ == "__main__":
    main()
