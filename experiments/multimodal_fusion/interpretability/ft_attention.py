#!/usr/bin/env python3
"""
FT-Transformer attention extraction for tabular interpretability.

Loads the paper's multimodal checkpoint (seed 42, fold 1), extracts the
self-attention weights from each TransformerEncoder layer of the FT-Transformer
branch, and produces per-class (CN vs AD-trajectory) feature attention plots.

CPU-only (MRI branch not executed).

Usage:
    python ft_attention.py [--fold N] [--seed S]
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
from model import MultiModalFusion  # noqa: E402

REPO = Path("/home/tanguy/medical/alzheimer")
EXP = REPO / "experiments" / "multimodal_fusion"

TABULAR_FEATURES = [
    "AGE", "PTGENDER", "PTEDUCAT", "PTMARRY",
    "CATANIMSC", "TRAASCOR", "TRABSCOR",
    "DSPANFOR", "DSPANBAC", "BNTTOTAL",
    "VSWEIGHT", "BMI",
    "MH14ALCH", "MH16SMOK", "MH4CARD", "MH2NEURL",
]

FEATURE_LABELS = {
    "AGE": "Age", "PTGENDER": "Sex", "PTEDUCAT": "Education", "PTMARRY": "Marital",
    "CATANIMSC": "Cat. fluency (animals)", "TRAASCOR": "TMT-A", "TRABSCOR": "TMT-B",
    "DSPANFOR": "Digit Span Fwd", "DSPANBAC": "Digit Span Bwd", "BNTTOTAL": "Boston Naming",
    "VSWEIGHT": "Weight", "BMI": "BMI",
    "MH14ALCH": "Alcohol Hx", "MH16SMOK": "Smoking Hx",
    "MH4CARD": "Cardiac Hx", "MH2NEURL": "Neurological Hx",
}


def build_paper_model():
    """Reconstruct the paper's multimodal model (architecture matches checkpoint state_dict)."""
    model = MultiModalFusion(
        num_tabular_features=len(TABULAR_FEATURES),
        num_classes=2,
        backbone_config={
            "type": "vit",
            "architecture": "vit_base",
            "image_size": 128,
            "feature_dim": 768,
        },
        tabular_config={
            "type": "ft_transformer",
            "embed_dim": 64,
            "num_heads": 4,
            "num_layers": 3,
            "dropout": 0.1,
            "output_dim": 64,
        },
        fusion_config={
            "method": "cross_modal",
            "hidden_dim": 512,
            "num_heads": 8,
            "dropout": 0.3,
            "auxiliary_losses": True,
        },
        pretrained_path=None,
        freeze_backbone=False,
    )
    return model


class AttentionCapturingLayer(nn.TransformerEncoderLayer):
    """TransformerEncoderLayer that captures self-attention weights.

    Overrides `forward` entirely to bypass the C++ fused fast-path
    (`torch._transformer_encoder_layer_fwd`) which would silently skip
    our `_sa_block` override.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.captured_attn = None

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        x = src
        if self.norm_first:
            normed = self.norm1(x)
            attn_out, attn_w = self.self_attn(
                normed, normed, normed,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                need_weights=True,
                average_attn_weights=False,
                is_causal=is_causal,
            )
            self.captured_attn = attn_w.detach()
            x = x + self.dropout1(attn_out)
            x = x + self._ff_block(self.norm2(x))
        else:
            attn_out, attn_w = self.self_attn(
                x, x, x,
                attn_mask=src_mask,
                key_padding_mask=src_key_padding_mask,
                need_weights=True,
                average_attn_weights=False,
                is_causal=is_causal,
            )
            self.captured_attn = attn_w.detach()
            x = self.norm1(x + self.dropout1(attn_out))
            x = self.norm2(x + self._ff_block(x))
        return x


def swap_layers_for_capture(transformer_encoder, embed_dim, num_heads, dropout):
    """Replace TransformerEncoderLayer instances with capture-enabled subclass."""
    new_layers = nn.ModuleList()
    for old in transformer_encoder.layers:
        new = AttentionCapturingLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        new.load_state_dict(old.state_dict())
        new_layers.append(new)
    transformer_encoder.layers = new_layers
    return transformer_encoder


def prepare_tabular(train_df, test_df, features):
    """Impute (per-dataset median, matching training-time behavior) and standardize."""
    train_tab = train_df[features].copy()
    test_tab = test_df[features].copy()
    for col in features:
        train_tab[col] = train_tab[col].fillna(train_tab[col].median())
        test_tab[col] = test_tab[col].fillna(test_tab[col].median())
    scaler = StandardScaler().fit(train_tab.values.astype(np.float32))
    return scaler.transform(test_tab.values.astype(np.float32)), scaler


def collect_attention(model, X_test):
    """Forward each test sample through the FT-Transformer branch, collect attention per layer."""
    model.eval()
    num_layers = len(model.tabular_encoder.transformer.layers)
    per_sample = []
    with torch.no_grad():
        for i in range(X_test.shape[0]):
            x = torch.from_numpy(X_test[i : i + 1]).float()
            _ = model.tabular_encoder(x)
            layers_attn = [
                layer.captured_attn[0].cpu().numpy()  # (heads, L, L)
                for layer in model.tabular_encoder.transformer.layers
            ]
            per_sample.append(np.stack(layers_attn, axis=0))  # (num_layers, heads, L, L)
    return np.stack(per_sample, axis=0)  # (N, num_layers, heads, L, L)


def attention_rollout(attn_per_sample):
    """Abnar & Zuidema rollout: per sample, product of (attn_layer + I)/2 across layers.

    attn_per_sample: (N, num_layers, heads, L, L)
    Returns: (N, L, L) rollout matrix
    """
    N, num_layers, heads, L, _ = attn_per_sample.shape
    avg_attn = attn_per_sample.mean(axis=2)  # (N, num_layers, L, L)
    I = np.eye(L)[None]  # (1, L, L)
    rolled = np.broadcast_to(I, (N, L, L)).copy()
    for li in range(num_layers):
        A = (avg_attn[:, li] + I) / 2.0
        # row-normalize
        A = A / A.sum(axis=-1, keepdims=True)
        rolled = np.einsum("nij,njk->nik", A, rolled)
    return rolled


def plot_attention(cn_attn, ad_attn, out_path, title_suffix=""):
    order = np.argsort(-(cn_attn + ad_attn))
    y = np.arange(len(TABULAR_FEATURES))
    labels_ordered = [FEATURE_LABELS[TABULAR_FEATURES[i]] for i in order]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(y - 0.2, cn_attn[order], 0.4, label="CN", color="steelblue")
    ax.barh(y + 0.2, ad_attn[order], 0.4, label="AD-trajectory", color="coral")
    ax.set_yticks(y)
    ax.set_yticklabels(labels_ordered)
    ax.invert_yaxis()
    ax.set_xlabel("Attention weight (CLS → feature, mean over test samples)")
    ax.set_title(f"FT-Transformer feature attention{title_suffix}")
    ax.axvline(1.0 / (len(TABULAR_FEATURES) + 1), color="gray", linestyle="--",
               alpha=0.5, label="Uniform baseline")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def regenerate_fold_splits(seed, fold, n_folds=5):
    """Regenerate the fold's train/val/test indices deterministically from all_df.

    Must match train_cv.py logic exactly: concat(train, val, test) then
    StratifiedKFold(n_folds, shuffle=True, random_state=seed), then 90/10
    train/val split per-fold with seed = seed+fold.
    """
    from sklearn.model_selection import StratifiedKFold
    train_df = pd.read_csv(EXP / "data/combined_trajectory/train.csv")
    val_df = pd.read_csv(EXP / "data/combined_trajectory/val.csv")
    test_df = pd.read_csv(EXP / "data/combined_trajectory/test.csv")
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    labels = all_df["label"].values
    indices = np.arange(len(all_df))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fi, (trn_idx, tst_idx) in enumerate(skf.split(indices, labels)):
        if fi != fold:
            continue
        np.random.seed(seed + fold)
        perm = np.random.permutation(len(trn_idx))
        vs = int(0.1 * len(trn_idx))
        val_local = trn_idx[perm[:vs]]
        train_local = trn_idx[perm[vs:]]
        return all_df.iloc[train_local].reset_index(drop=True), \
               all_df.iloc[val_local].reset_index(drop=True), \
               all_df.iloc[tst_idx].reset_index(drop=True)
    raise ValueError(f"fold {fold} not found")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument(
        "--preset",
        choices=["orphan", "new-cv"],
        default="new-cv",
        help="orphan: pre-fix checkpoints in cross_task_cv_results/. "
             "new-cv: paper-grade checkpoints in cv_results/seed_X/fold_Y/.",
    )
    args = parser.parse_args()

    if args.preset == "orphan":
        ckpt_path = EXP / "cross_task_cv_results" / f"model_seed{args.seed}_fold{args.fold}.pth"
        fold_dir = REPO / "experiments" / "ablation_mri_only" / "cv_results" / f"seed_{args.seed}" / f"fold_{args.fold}"
        train_df = pd.read_csv(fold_dir / "train.csv")
        test_df = pd.read_csv(fold_dir / "test.csv")
        cn_ad_df = pd.read_csv(fold_dir / "cn_ad_test.csv")
        scaler = None
        out_dir = SCRIPT_DIR / "results"
    else:
        # new-cv: use paper-grade checkpoints + regenerate splits
        ckpt_path = EXP / "cv_results" / f"seed_{args.seed}" / f"fold_{args.fold}" / "model.pth"
        scaler_path = EXP / "cv_results" / f"seed_{args.seed}" / f"fold_{args.fold}" / "scaler.pkl"
        print(f"[*] Regenerating fold splits (seed={args.seed}, fold={args.fold})...")
        train_df, val_df, test_df = regenerate_fold_splits(args.seed, args.fold)
        # Load saved scaler (trained on train.csv during the CV training)
        import pickle
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        # Cross-task eval: filter test fold to stable CN_AD subjects
        cn_ad_dir = EXP / "data" / "combined_cn_ad"
        stable_ids = set()
        for n in ["train.csv", "val.csv", "test.csv"]:
            p = cn_ad_dir / n
            if p.exists():
                stable_ids |= set(pd.read_csv(p)["subject_id"].values)
        cn_ad_df = test_df[test_df["subject_id"].isin(stable_ids)].reset_index(drop=True)
        out_dir = SCRIPT_DIR / "results_new"
    out_dir.mkdir(exist_ok=True, parents=True)

    print(f"[*] Checkpoint: {ckpt_path}")
    print(f"[*] Preset: {args.preset}")
    print(f"[*] Output dir: {out_dir}")

    print(f"[*] Train={len(train_df)}  Test(traj)={len(test_df)}  Test(cn_ad)={len(cn_ad_df)}")
    print(f"    Test label dist (traj): {dict(test_df['label'].value_counts().sort_index())}")
    print(f"    Test label dist (cn_ad): {dict(cn_ad_df['label'].value_counts().sort_index())}")

    if scaler is None:
        X_test_traj, scaler = prepare_tabular(train_df, test_df, TABULAR_FEATURES)
    else:
        X_test_traj = scaler.transform(
            test_df[TABULAR_FEATURES]
            .apply(lambda col: col.fillna(col.median()))
            .values.astype(np.float32)
        )
    X_test_cn_ad = scaler.transform(
        cn_ad_df[TABULAR_FEATURES]
        .apply(lambda col: col.fillna(col.median()))
        .values.astype(np.float32)
    )

    y_traj = test_df["label"].values.astype(int)
    y_cn_ad = cn_ad_df["label"].values.astype(int)

    # Model
    print("[*] Building model and loading checkpoint...")
    model = build_paper_model()
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    result = model.load_state_dict(state_dict, strict=False)
    missing = [k for k in result.missing_keys if not k.startswith("vit.")]
    unexpected = [k for k in result.unexpected_keys if not k.startswith("vit.")]
    print(f"    missing (excl. vit alias): {len(missing)}  |  unexpected: {len(unexpected)}")
    if missing[:3]:
        print(f"    sample missing: {missing[:3]}")
    if unexpected[:3]:
        print(f"    sample unexpected: {unexpected[:3]}")

    # Swap FT-T encoder layers to capture attention
    model.tabular_encoder.transformer = swap_layers_for_capture(
        model.tabular_encoder.transformer,
        embed_dim=64, num_heads=4, dropout=0.1,
    )

    # Sanity: standalone aux tab_classifier accuracy on test (cheap, no MRI)
    print("[*] Tabular-branch sanity (aux classifier, no MRI):")
    for name, X, y in [("traj", X_test_traj, y_traj), ("cn_ad", X_test_cn_ad, y_cn_ad)]:
        with torch.no_grad():
            tab_feats = model.tabular_encoder(torch.from_numpy(X).float())
            logits = model.tab_classifier(tab_feats)
            preds = logits.argmax(dim=-1).numpy()
        acc = (preds == y).mean() * 100
        print(f"    {name}: aux_tab_acc={acc:.2f}% (N={len(y)})")

    # Collect attention on both test sets
    for test_name, X, y in [("traj", X_test_traj, y_traj), ("cn_ad", X_test_cn_ad, y_cn_ad)]:
        print(f"[*] Collecting attention ({test_name}, N={len(y)})...")
        attns = collect_attention(model, X)
        print(f"    attns.shape = {attns.shape} (N, layers, heads, L, L)")

        np.save(out_dir / f"attn_seed{args.seed}_fold{args.fold}_{test_name}.npy", attns)

        # Last-layer CLS→feature attention, averaged over heads
        cls_last = attns[:, -1, :, 0, 1:].mean(axis=1)  # (N, 16)
        # Rollout (Abnar & Zuidema)
        rolled = attention_rollout(attns)
        rolled_cls = rolled[:, 0, 1:]  # CLS → features (N, 16)

        cn_mask, ad_mask = (y == 0), (y == 1)
        for variant, tensor in [("last_layer", cls_last), ("rollout", rolled_cls)]:
            cn_mean = tensor[cn_mask].mean(axis=0)
            ad_mean = tensor[ad_mask].mean(axis=0)
            df = pd.DataFrame({
                "feature": TABULAR_FEATURES,
                "label": [FEATURE_LABELS[f] for f in TABULAR_FEATURES],
                "cn_mean": cn_mean,
                "ad_mean": ad_mean,
                "diff": ad_mean - cn_mean,
            }).sort_values("diff", ascending=False)
            csv_path = out_dir / f"ft_attn_{variant}_seed{args.seed}_fold{args.fold}_{test_name}.csv"
            df.to_csv(csv_path, index=False)
            fig_path = out_dir / f"ft_attn_{variant}_seed{args.seed}_fold{args.fold}_{test_name}.png"
            plot_attention(
                cn_mean, ad_mean, fig_path,
                title_suffix=f" ({variant}, {test_name}, seed={args.seed}, fold={args.fold})",
            )
            print(f"    [{variant}] → {csv_path.name}")
            top3 = df.head(3)[["label", "diff"]].values
            bot3 = df.tail(3)[["label", "diff"]].values
            print(f"       top↑ AD-CN: {top3.tolist()}")
            print(f"       top↓ AD-CN: {bot3.tolist()}")

    print("[*] Done.")


if __name__ == "__main__":
    main()
