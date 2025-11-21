# ======================================================
# iSyncTab.py — multimodal model with NS-PFS (feature sequencing)
# ======================================================
# Requirements:
#   pip install torch torchvision linformer scipy
# ======================================================
import os, random, numpy as np
from typing import Optional, Tuple, List
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision
from linformer import Linformer
from scipy.optimize import linear_sum_assignment

# ======================================================
# Utility
# ======================================================
def set_seed(seed=42, deterministic=True):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)

# ======================================================
# Image Encoder (supports RGB + Grayscale)
# ======================================================
class ImageTokenEncoder(nn.Module):
    def __init__(self, d_model=256, pretrained=False, in_channels=3):
        super().__init__()
        weights = torchvision.models.ResNet50_Weights.DEFAULT if pretrained else None
        base = torchvision.models.resnet50(weights=weights)

        # --- handle grayscale ---
        if in_channels != 3:
            old_conv = base.conv1
            base.conv1 = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )
            if pretrained and in_channels == 1:
                with torch.no_grad():
                    base.conv1.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)

        # --- backbone ---
        self.stem = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4
        )

        self.fixpool = nn.AdaptiveAvgPool2d((7, 7))  # 49 tokens
        self.proj = nn.Linear(2048, d_model)

        # --- normalization fallback ---
        if pretrained and weights is not None:
            meta = getattr(weights, "meta", None)
            if meta and "mean" in meta and "std" in meta:
                mean, std = meta["mean"], meta["std"]
            else:
                mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))
        else:
            self.mean, self.std = None, None

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        if (self.mean is not None) and (self.std is not None):
            x = (x - self.mean) / self.std

        f = self.stem(x)
        f = self.fixpool(f)
        B, C, H, W = f.shape
        f = f.view(B, C, H*W).transpose(1, 2)
        return self.proj(f)  # (B,49,d_model)

# ======================================================
# Tabular Encoder
# ======================================================
class TabularTokenEncoder(nn.Module):
    def __init__(self, num_features=None, d_model=256, depth=2, heads=4,
                 vocab_size_text=5000, max_cat_card=50):
        super().__init__()
        self.d_model = d_model
        self.vocab_size_text = vocab_size_text
        self.max_cat_card = max_cat_card

        self.scalar_to_token = nn.Linear(1, d_model)
        self.cat_embed = nn.Embedding(max_cat_card + 2, d_model)
        self.text_embed = nn.EmbeddingBag(vocab_size_text, d_model, mode='mean')

        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=depth)
        self.pos = None

    def forward(self, x_tab):
        if isinstance(x_tab, dict):
            x_num = x_tab.get("num", None)
            x_cat = x_tab.get("cat", None)
            x_text = x_tab.get("text", None)
        else:
            x_num, x_cat, x_text = x_tab, None, None

        B = x_num.shape[0] if x_num is not None else \
            x_cat.shape[0] if x_cat is not None else x_text.shape[0]

        tokens = []

        # --- numeric ---
        if x_num is not None:
            col_median = torch.nanmedian(x_num, dim=0).values
            col_median = torch.where(torch.isfinite(col_median), col_median, torch.zeros_like(col_median))
            x_num = x_num.clone()
            mask = torch.isnan(x_num)
            if mask.any():
                x_num[mask] = col_median.expand_as(x_num)[mask]
            tok_num = self.scalar_to_token(x_num.unsqueeze(-1))
            tokens.append(tok_num)

        # --- categorical ---
        if x_cat is not None:
            x_cat = x_cat.clone().to(torch.long)
            x_cat = torch.clamp(x_cat, min=-1, max=self.max_cat_card-1)
            x_cat[x_cat < 0] = self.max_cat_card + 1
            tok_cat = self.cat_embed(x_cat)
            tokens.append(tok_cat)

        # --- text ---
        if x_text is not None:
            B, O_text, seq_len = x_text.shape
            x_text_flat = x_text.view(B * O_text, seq_len).to(torch.long)
            tok_text = self.text_embed(x_text_flat)
            tok_text = tok_text.view(B, O_text, self.d_model)
            tokens.append(tok_text)

        if not tokens:
            raise ValueError("No valid input provided to TabularTokenEncoder.")

        x = torch.cat(tokens, dim=1)
        O = x.size(1)
        if (self.pos is None) or (self.pos.size(1) < O) or (self.pos.device != x.device):
            self.pos = nn.Parameter(torch.randn(1, O, self.d_model, device=x.device) * 0.02)
        x = x + self.pos[:, :O, :]
        return self.encoder(x)

# ======================================================
# GPU KMeans
# ======================================================
@torch.no_grad()
def kmeans_torch(X_points, k, iters=50, tol=1e-4, device=None):
    device = device or X_points.device
    P, D = X_points.shape
    if P < k: k = P
    idx = torch.randperm(P, device=device)[:k]
    C = X_points[idx].clone()
    prev = None
    for _ in range(iters):
        dists = torch.cdist(X_points, C, p=2)
        labels = dists.argmin(dim=1)
        newC = torch.zeros_like(C)
        for ci in range(k):
            mask = (labels == ci)
            newC[ci] = X_points[mask].mean(dim=0) if mask.any() else X_points[torch.randint(0, P, (1,), device=device)]
        shift = (newC - C).abs().mean().item()
        C = newC
        if prev is not None and abs(prev - shift) < tol: break
        prev = shift
    return labels, C

# ======================================================
# NS-PFS (Feature Sequencing)
# ======================================================
class NSPFS_GPU(nn.Module):
    def __init__(self, num_clusters=5, metric="variance", bins=32, mi_chunk=128, device=None):
        super().__init__()
        self.k = int(num_clusters)
        self.metric = metric.lower()
        self.bins = int(bins)
        self.mi_chunk = int(mi_chunk)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # pairwise metrics
    @torch.no_grad()
    def _pairwise_variance(self, X):
        v = torch.var(X, dim=0, unbiased=False)
        M = torch.abs(v[:, None] - v[None, :]); M.fill_diagonal_(0.0)
        return M
    @torch.no_grad()
    def _pairwise_euclidean(self, X): return torch.cdist(X.T.contiguous(), X.T.contiguous(), p=2)
    @torch.no_grad()
    def _pairwise_cosine(self, X):
        Z = F.normalize(X.T.contiguous(), dim=1)
        sim = Z @ Z.T
        return 1.0 - sim.clamp(-1, 1)
    @torch.no_grad()
    def _pairwise_correlation(self, X):
        N, D = X.shape
        Xm = X - X.mean(dim=0, keepdim=True)
        std = Xm.std(dim=0, unbiased=False).clamp_min(1e-12)
        Z = Xm / std
        C = (Z.T @ Z) / float(N)
        Dcorr = 1.0 - C.abs(); Dcorr.fill_diagonal_(0.0)
        return Dcorr
    @torch.no_grad()
    def _histograms_shared_bins(self, X):
        xmin, xmax = X.min(), X.max()
        D = X.shape[1]
        if float(xmax - xmin) < 1e-12:
            P = torch.zeros((D, self.bins), device=X.device); P[:, 0] = 1.0
            return torch.zeros_like(X, dtype=torch.long), P
        edges = torch.linspace(xmin, xmax, self.bins + 1, device=X.device)
        idx = torch.bucketize(X, edges) - 1; idx = idx.clamp(0, self.bins - 1)
        counts = torch.zeros((D, self.bins), device=X.device)
        counts.scatter_add_(1, idx.T, torch.ones_like(idx, dtype=torch.float32).T)
        P = counts / counts.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return idx, P
    @torch.no_grad()
    def _kl_matrix(self, P):
        P = P.clamp_min(1e-12)
        Hself = (P * P.log()).sum(dim=1)
        XEnt = P @ P.log().T
        K = Hself[:, None] - XEnt; K.fill_diagonal_(0.0)
        return K
    @torch.no_grad()
    def _construct_graph(self, X):
        m = self.metric
        if m in ("variance","var"): return self._pairwise_variance(X)
        if m in ("euclidean","l2"): return self._pairwise_euclidean(X)
        if m in ("cosine","cos"): return self._pairwise_cosine(X)
        if m in ("correlation","corr","pearson"): return self._pairwise_correlation(X)
        if m in ("kl","kl_divergence"):
            _, P = self._histograms_shared_bins(X); return self._kl_matrix(P)
        raise ValueError(f"Unknown metric: {self.metric}")

    @torch.no_grad()
    def forward(self, tab_tokens, img_tokens):
        device = tab_tokens.device
        B, O, Ht = tab_tokens.shape; B2, D, Hi = img_tokens.shape
        assert B == B2
        tab_scalar, img_scalar = tab_tokens.mean(2), img_tokens.mean(2)
        lab_tab, _ = kmeans_torch(tab_scalar.T, k=self.k, device=device)
        lab_img, _ = kmeans_torch(img_scalar.T, k=self.k, device=device)
        tab_idx = [torch.nonzero(lab_tab == i, as_tuple=False).flatten() for i in range(self.k)]
        img_idx = [torch.nonzero(lab_img == i, as_tuple=False).flatten() for i in range(self.k)]
        Psi = torch.zeros((self.k, self.k), device=device)
        for r in range(self.k):
            for c in range(self.k):
                if tab_idx[r].numel() == 0 or img_idx[c].numel() == 0: continue
                Psi[r, c] = torch.rand(1).item()
        rr, cc = linear_sum_assignment((-Psi).cpu().numpy())
        seq = torch.cat([torch.tensor(tab_idx[r].tolist() + (O + img_idx[c]).tolist(), device=device)
                         for r, c in zip(rr, cc)])
        #perm = torch.unique(seq, consecutive=False)
        try:
            perm = torch.unique(seq)
        except TypeError:
            #Fallback to older versions that require 'sorted' kwargs
            perm = torch.unique(seq, sorted=False)
        all_idx = torch.arange(O + D, device=device)
        missing = all_idx[~torch.isin(all_idx, perm)]
        if missing.numel() > 0: perm = torch.cat([perm, missing])
        return perm

# ======================================================
# iSyncTab Model (Frozen Feature Tokens + Global Perm)
# ======================================================
# ======================================================
# iSyncTab Model
# ======================================================
class iSyncTab(nn.Module):
    def __init__(self, num_tab_features, num_classes,
                 d_model=128, num_clusters=4, metric="variance",
                 linformer_depth=4, linformer_heads=4, linformer_k=32,
                 lambda_fs=0.1, pretrained_resnet=False, device=None):
        super().__init__()
        self.lambda_fs = float(lambda_fs)
        self.tab_enc = TabularTokenEncoder(num_features=num_tab_features, d_model=d_model)
        self.img_enc = ImageTokenEncoder(d_model, pretrained=pretrained_resnet)
        self.nspfs = NSPFS_GPU(num_clusters=num_clusters, metric=metric, device=device)
        self.seq_len_max = int(num_tab_features) + 49
        self.linformer = Linformer(
            dim=d_model, seq_len=self.seq_len_max,
            depth=linformer_depth, heads=linformer_heads, k=linformer_k,
            one_kv_head=True, share_kv=True
        )
        self.cls_head = nn.Linear(d_model, num_classes)
        self.seq_head = nn.Linear(d_model, 1)

    def forward(self, x_tab, x_img, y=None):
        t_tok = self.tab_enc(x_tab)
        i_tok = self.img_enc(x_img)
        B, O, _ = t_tok.shape; L_img = i_tok.shape[1]
        assert O + L_img <= self.seq_len_max, f"Sequence too long ({O+L_img})"
        perm = self.nspfs(t_tok, i_tok)
        tokens = torch.cat([t_tok, i_tok], dim=1)[:, perm, :]
        z = self.linformer(tokens)
        z_cls = z.mean(1)
        logits = self.cls_head(z_cls)
        seq_scores = self.seq_head(z).squeeze(-1)
        ranks = torch.arange(perm.numel(), device=logits.device, dtype=torch.float32)
        beta = (ranks / (ranks[-1] if ranks[-1] > 0 else 1.0)).unsqueeze(0).expand(B, -1)
        out = {"logits": logits, "perm": perm, "seq_scores": seq_scores, "beta": beta}
        if y is not None:
            ce = F.cross_entropy(logits, y)
            fs = F.mse_loss(seq_scores, beta)
            out["loss"] = ce + self.lambda_fs * fs
            out["loss_ce"] = ce; out["loss_fs"] = fs
        return out