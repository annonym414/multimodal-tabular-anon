# ======================================================
# iSyncTab_refined.py — iSyncTab with stable, theory-aligned NS–PFS
# ======================================================
# pip install torch torchvision linformer scipy
# ======================================================
from typing import Optional, List, Tuple, Dict
import math, random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from linformer import Linformer
from scipy.optimize import linear_sum_assignment


# ======================================================
# Repro
# ======================================================
def set_seed(seed: int = 42, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


# ======================================================
# Image Encoder (ResNet50 → 49 tokens → d_model)
#   - fixed grayscale handling
# ======================================================
class ImageTokenEncoder(nn.Module):
    def __init__(self, d_model: int = 256, pretrained: bool = False, in_channels: int = 3):
        super().__init__()
        weights = torchvision.models.ResNet50_Weights.DEFAULT if pretrained else None
        base = torchvision.models.resnet50(weights=weights)

        # swap first conv if needed
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
                    # average pretrained RGB weights to init 1ch conv
                    base.conv1.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)

        self.stem = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2, base.layer3, base.layer4
        )
        self.fixpool = nn.AdaptiveAvgPool2d((7, 7))  # 49 spatial tokens
        self.proj = nn.Linear(2048, d_model)

        # in-model normalization if pretrained
        if pretrained and weights is not None and getattr(weights, "meta", None):
            mean = weights.meta.get("mean", [0.485, 0.456, 0.406])
            std = weights.meta.get("std", [0.229, 0.224, 0.225])
        else:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)
        self.in_channels = in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W) or (B,1,H,W)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if x.shape[1] == 1 and self.in_channels == 3:
            # expand grayscale to RGB if encoder expects 3ch
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] != self.in_channels:
            raise ValueError(f"ImageTokenEncoder expected {self.in_channels} channels, got {x.shape[1]}")

        x = (x - self.mean) / self.std
        f = self.stem(x)                  # (B, 2048, h, w)
        f = self.fixpool(f)               # (B, 2048, 7, 7)
        B, C, H, W = f.shape
        f = f.view(B, C, H * W).transpose(1, 2)  # (B, 49, 2048)
        return self.proj(f)               # (B, 49, d_model)


# ======================================================
# Tabular Token Encoder  (numeric / categorical / text)
# ======================================================
class TabularTokenEncoder(nn.Module):
    def __init__(self, d_model: int = 256, depth: int = 2, heads: int = 4,
                 vocab_size_text: int = 5000, max_cat_card: int = 50):
        super().__init__()
        self.d_model = d_model
        self.scalar_to_token = nn.Linear(1, d_model)
        self.cat_embed = nn.Embedding(max_cat_card + 2, d_model)  # last = missing bucket
        self.text_embed = nn.EmbeddingBag(vocab_size_text, d_model, mode='mean')

        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=depth)
        self.pos = None
        self.max_cat_card = max_cat_card

    def forward(self, x_tab) -> torch.Tensor:
        """
        x_tab can be:
          - Tensor numeric: (B, O_num)
          - Dict with keys: {"num": (B,O_num), "cat": (B,O_cat), "text": (B,O_txt, L)}
        """
        if isinstance(x_tab, dict):
            x_num = x_tab.get("num")
            x_cat = x_tab.get("cat")
            x_text = x_tab.get("text")
        else:
            x_num, x_cat, x_text = x_tab, None, None

        B = None
        tokens = []

        # numeric
        if x_num is not None:
            assert x_num.dim() == 2, "numeric must be (B, O_num)"
            B = x_num.shape[0] if B is None else B
            x_num = x_num.clone()
            # impute per-batch median
            med = torch.nanmedian(x_num, dim=0).values
            med = torch.where(torch.isfinite(med), med, torch.zeros_like(med))
            mask = torch.isnan(x_num)
            if mask.any():
                x_num[mask] = med.expand_as(x_num)[mask]
            tok_num = self.scalar_to_token(x_num.unsqueeze(-1))  # (B,O_num, d)
            tokens.append(tok_num)

        # categorical
        if x_cat is not None:
            assert x_cat.dim() == 2, "categorical must be (B, O_cat)"
            B = x_cat.shape[0] if B is None else B
            x_cat = x_cat.clone().to(torch.long)
            x_cat = torch.clamp(x_cat, min=-1, max=self.max_cat_card - 1)
            x_cat[x_cat < 0] = self.max_cat_card + 1  # missing bucket
            tok_cat = self.cat_embed(x_cat)  # (B,O_cat,d)
            tokens.append(tok_cat)

        # text
        if x_text is not None:
            assert x_text.dim() == 3, "text must be (B, O_txt, L)"
            B = x_text.shape[0] if B is None else B
            B_, O_txt, L = x_text.shape
            xt = x_text.view(B_ * O_txt, L).to(torch.long)
            tok_text = self.text_embed(xt)        # (B_*O_txt, d)
            tok_text = tok_text.view(B_, O_txt, -1)
            tokens.append(tok_text)

        if not tokens:
            raise ValueError("No valid tabular fields provided.")
        x = torch.cat(tokens, dim=1)  # (B, O_total, d)

        # learned positional tokens (expand if needed)
        O = x.size(1)
        if (self.pos is None) or (self.pos.size(1) < O) or (self.pos.device != x.device):
            self.pos = nn.Parameter(torch.randn(1, O, self.d_model, device=x.device) * 0.02)
        x = x + self.pos[:, :O, :]

        return self.encoder(x)  # (B, O_total, d)


# ======================================================
# Lightweight k-means (torch)
# ======================================================
@torch.no_grad()
def kmeans_torch(points: torch.Tensor, k: int, iters: int = 50, tol: float = 1e-4) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    points: (P, D)
    returns: labels (P,), centroids (k, D)
    """
    device = points.device
    P, D = points.shape
    k = min(k, max(1, P))
    idx = torch.randperm(P, device=device)[:k]
    C = points[idx].clone()
    prev_shift = None
    for _ in range(iters):
        d = torch.cdist(points, C, p=2)
        labels = d.argmin(dim=1)
        newC = torch.zeros_like(C)
        for ci in range(k):
            mask = (labels == ci)
            if mask.any():
                newC[ci] = points[mask].mean(dim=0)
            else:
                newC[ci] = points[torch.randint(0, P, (1,), device=device)]
        shift = (newC - C).abs().mean().item()
        C = newC
        if prev_shift is not None and abs(prev_shift - shift) < tol:
            break
        prev_shift = shift
    return labels, C


# ======================================================
# Metrics & utilities for NS–PFS
# ======================================================
@torch.no_grad()
def shared_histograms(X: torch.Tensor, bins: int = 32) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    X: (N, m) scalar matrix (N samples across batch, m features/tokens)
    returns:
        idx: (N, m) bin indices
        P:   (m, bins) normalized histograms
    """
    device = X.device
    xmin, xmax = X.min(), X.max()
    m = X.shape[1]
    if float(xmax - xmin) < 1e-12:
        P = torch.zeros((m, bins), device=device)
        P[:, 0] = 1.0
        return torch.zeros_like(X, dtype=torch.long), P
    edges = torch.linspace(xmin, xmax, bins + 1, device=device)
    idx = torch.bucketize(X, edges) - 1
    idx = idx.clamp(0, bins - 1)
    counts = torch.zeros((m, bins), device=device)
    counts.scatter_add_(1, idx.T, torch.ones_like(idx, dtype=torch.float32).T)
    P = counts / counts.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return idx, P


@torch.no_grad()
def kl_matrix(P: torch.Tensor) -> torch.Tensor:
    """
    P: (m, bins) row-stochastic
    returns pairwise KL(f||g) asym → use symmetrized later if needed
    """
    P = P.clamp_min(1e-12)
    Hself = (P * P.log()).sum(dim=1)          # (m,)
    XEnt = P @ P.log().T                      # (m, m)
    K = Hself[:, None] - XEnt                 # (m, m)
    K.fill_diagonal_(0.0)
    return K


@torch.no_grad()
def js_divergence(P: torch.Tensor) -> torch.Tensor:
    """
    Jensen-Shannon divergence matrix using shared histograms
    """
    P = P.clamp_min(1e-12)
    M = 0.5 * (P[:, None, :] + P[None, :, :])     # (m,m,bins)
    KL_PM = (P[:, None, :] * (P[:, None, :].log() - M.log())).sum(dim=2)
    KL_QM = (P[None, :, :] * (P[None, :, :].log() - M.log())).sum(dim=2)
    JS = 0.5 * (KL_PM + KL_QM)
    JS.fill_diagonal_(0.0)
    return JS


@torch.no_grad()
def pairwise_metric_matrix(X_scalar: torch.Tensor, metric: str, bins: int = 32) -> torch.Tensor:
    """
    X_scalar: (N, m)   per-token scalar across the batch/samples
    metric in {"variance","euclidean","manhattan","cosine","correlation","kl","js"}
    returns: (m, m) distances / dissimilarity
    """
    metric = metric.lower()
    N, m = X_scalar.shape
    device = X_scalar.device

    if metric in ("variance", "var"):
        v = X_scalar.var(dim=0, unbiased=False)              # (m,)
        D = (v[:, None] - v[None, :]).abs()
        D.fill_diagonal_(0.0)
        return D

    if metric in ("euclidean", "l2"):
        X = X_scalar.T.contiguous()                          # (m, N)
        return torch.cdist(X, X, p=2)

    if metric in ("manhattan", "l1"):
        X = X_scalar.T.contiguous()
        # pairwise L1 via trick
        # (m, N) → (m,1,N) - (1,m,N) → abs → sum_N
        diffs = (X[:, None, :] - X[None, :, :]).abs().sum(dim=2)
        diffs.fill_diagonal_(0.0)
        return diffs

    if metric in ("cosine", "cos"):
        Z = F.normalize(X_scalar.T.contiguous(), dim=1)      # (m,N)
        sim = (Z @ Z.T).clamp(-1, 1)
        D = 1.0 - sim
        D.fill_diagonal_(0.0)
        return D

    if metric in ("correlation", "corr", "pearson"):
        X = X_scalar - X_scalar.mean(dim=0, keepdim=True)
        std = X.std(dim=0, unbiased=False).clamp_min(1e-12)
        Z = X / std
        C = (Z.T @ Z) / float(N)                             # (m,m)
        D = 1.0 - C.abs()
        D.fill_diagonal_(0.0)
        return D

    if metric in ("kl", "kl_divergence", "js", "jensen-shannon"):
        _, P = shared_histograms(X_scalar, bins=bins)        # (m,bins)
        if metric.startswith("kl"):
            K = kl_matrix(P)
            # symmetrize
            return 0.5 * (K + K.T)
        else:
            return js_divergence(P)

    raise ValueError(f"Unknown metric: {metric}")


# ======================================================
# Spectral seriation (Fiedler vector ordering)
# Minimizes a linear arrangement objective approx. to Σ W_ij |π(i)-π(j)|
# ======================================================
@torch.no_grad()
def spectral_seriation(weights: torch.Tensor) -> torch.Tensor:
    """
    weights: (q, q) nonnegative, symmetric sparse/dense
    return: indices (q,) — order by Fiedler vector
    """
    q = weights.shape[0]
    if q <= 1:
        return torch.arange(q, device=weights.device)

    W = weights.clone()
    W = (W + W.T) * 0.5
    W.fill_diagonal_(0.0)
    d = W.sum(dim=1)
    L = torch.diag(d) - W

    # use smallest nonzero eigenvector (2nd smallest eig)
    # torch.linalg.eigh returns ascending eigenvalues
    evals, evecs = torch.linalg.eigh(L)
    if evals.numel() < 2:
        return torch.arange(q, device=W.device)
    fiedler = evecs[:, 1]  # (q,)
    order = torch.argsort(fiedler)
    return order


# ======================================================
# NS–PFS (Refined, one-time fit, cached permutation)
# ======================================================
class NSPFSRefined(nn.Module):
    def __init__(self,
                 num_clusters: int = 5,
                 metric: str = "variance",
                 bins: int = 32,
                 sparsity_q: float = 0.5,        # quantile for edge sparsification
                 tauH_q: float = 0.75,           # quantile for "high" thresholds
                 tauL_q: float = 0.25,           # quantile for "low" thresholds
                 gamma_eta: float = 0.8,         # cap for HL/LH
                 gamma_zeta: float = 0.5):       # cap for LL
        super().__init__()
        self.k = int(num_clusters)
        self.metric = metric
        self.bins = int(bins)
        self.sparsity_q = float(sparsity_q)
        self.tauH_q = float(tauH_q)
        self.tauL_q = float(tauL_q)
        self.gamma_eta = float(gamma_eta)
        self.gamma_zeta = float(gamma_zeta)

        # cached after fit()
        self.register_buffer("perm_cached", torch.tensor([], dtype=torch.long), persistent=True)
        self.register_buffer("beta_cached", torch.tensor([], dtype=torch.float32), persistent=True)

    def has_perm(self) -> bool:
        return self.perm_cached.numel() > 0

    @torch.no_grad()
    def reset(self):
        self.perm_cached = torch.tensor([], dtype=torch.long, device=self.perm_cached.device if self.perm_cached.is_cuda else None)
        self.beta_cached = torch.tensor([], dtype=torch.float32, device=self.beta_cached.device if self.beta_cached.is_cuda else None)

    @torch.no_grad()
    def _compute_cluster_stats(self, X_scalar: torch.Tensor, labels: torch.Tensor, token_vecs: torch.Tensor
                               ) -> Tuple[List[torch.Tensor], List[float], List[torch.Tensor]]:
        """
        For each cluster:
          - indices
          - energy ξ = sum_i M(f_i) (here: sum of per-feature metric "magnitude")
          - centroid vector c (mean of token vectors)
        X_scalar: (N, m) per-token scalars across batch
        labels: (m,) cluster labels
        token_vecs: (m, d) token vectors averaged over batch
        """
        device = X_scalar.device
        m = X_scalar.shape[1]
        idx_by_cluster = []
        energy = []
        centroid = []
        # per-feature scalar "magnitude" under current metric
        # for "variance": use var over N; for distances: use L2 norm over sample vector
        metric = self.metric.lower()
        if metric in ("variance", "var"):
            m_val = X_scalar.var(dim=0, unbiased=False)              # (m,)
        elif metric in ("euclidean", "l2", "manhattan", "l1", "cosine", "cos", "correlation", "corr", "pearson"):
            m_val = X_scalar.norm(p=2, dim=0)                        # (m,)
        elif metric in ("kl", "kl_divergence", "js", "jensen-shannon"):
            # proxy magnitude = entropy of per-feature histogram distribution
            _, P = shared_histograms(X_scalar, bins=self.bins)       # (m,bins)
            m_val = -(P * P.clamp_min(1e-12).log()).sum(dim=1)       # (m,) entropy
        else:
            raise ValueError(f"Unknown metric {self.metric}")

        for ci in range(self.k):
            mask = (labels == ci)
            idx = torch.nonzero(mask, as_tuple=False).flatten()
            idx_by_cluster.append(idx)

            if idx.numel() == 0:
                energy.append(0.0)
                centroid.append(torch.zeros(token_vecs.shape[1], device=device))
            else:
                energy.append(float(m_val[idx].sum().item()))
                centroid.append(token_vecs[idx].mean(dim=0))

        return idx_by_cluster, energy, centroid

    @torch.no_grad()
    def _sync_matrix(self,
                     energy_I: List[float], cent_I: List[torch.Tensor],
                     energy_T: List[float], cent_T: List[torch.Tensor]) -> torch.Tensor:
        """
        Build Ψ_{km} = E_{km} * S_{km}
        E_{km} = 1 - |ξ_I - ξ_T| / (ξ_I + ξ_T + eps)
        S_{km} = cosine_sim(c_I, c_T)
        """
        device = cent_I[0].device
        K = self.k
        eps = 1e-12
        Psi = torch.zeros((K, K), device=device)
        # stack centroids for cosine
        CI = torch.stack(cent_I, dim=0)  # (K,d)
        CT = torch.stack(cent_T, dim=0)  # (K,d)
        CI = F.normalize(CI, dim=1)
        CT = F.normalize(CT, dim=1)
        S = (CI @ CT.T).clamp(-1, 1)     # (K,K)
        for i in range(K):
            for j in range(K):
                xi, xt = energy_I[i], energy_T[j]
                if xi <= 0.0 and xt <= 0.0:
                    E = 0.0
                else:
                    E = 1.0 - abs(xi - xt) / (xi + xt + eps)
                Psi[i, j] = E * S[i, j]
        return Psi

    @torch.no_grad()
    def _regime_gamma(self, energies: List[float], Psi: torch.Tensor) -> Dict[Tuple[int, int], float]:
        """
        Assign HH/HL/LH/LL regimes using energy and Ψ ranks; return γ per pair.
        """
        K = self.k
        allE = torch.tensor(energies, device=Psi.device)
        eH = torch.quantile(allE, q=self.tauH_q).item()
        eL = torch.quantile(allE, q=self.tauL_q).item()

        PsiH = torch.quantile(Psi.flatten(), q=self.tauH_q).item()
        PsiL = torch.quantile(Psi.flatten(), q=self.tauL_q).item()

        gamma = {}
        for i in range(K):
            for j in range(K):
                # crude regime based on whether each side's energy is high and Psi high
                # (use min energy of the pair for conservativeness)
                e_pair = min(allE[i].item(), allE[j].item())  # proxy
                p = Psi[i, j].item()
                if (e_pair >= eH) and (p >= PsiH):
                    g = p  # g_H(Ψ)=Ψ
                elif (e_pair <= eL) and (p <= PsiL):
                    g = min(p, self.gamma_zeta)  # g_L(Ψ)=min(Ψ, ζ)
                else:
                    g = min(p, self.gamma_eta)   # g_A(Ψ)=min(Ψ, η)
                gamma[(i, j)] = float(max(0.0, min(1.0, g)))
        return gamma

    @torch.no_grad()
    def fit_from_tokens(self,
                        tab_tokens: torch.Tensor,   # (B, O, d)
                        img_tokens: torch.Tensor    # (B, 49, d)
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        One-time NS–PFS to produce a global permutation over [tab | image] tokens.
        Returns (perm, beta), and caches them.
        """
        device = tab_tokens.device
        B, O, d = tab_tokens.shape
        B2, D, di = img_tokens.shape
        assert B == B2, "Tab/Image batch mismatch"

        # build per-token scalars over batch (mean over hidden dim → (B,O)->scalar per token)
        tab_scalar = tab_tokens.mean(dim=2)      # (B, O)
        img_scalar = img_tokens.mean(dim=2)      # (B, D)
        X_tab = tab_scalar                       # (B, O)
        X_img = img_scalar                       # (B, D)

        # token vectors averaged over batch (for centroids)
        Tvec = tab_tokens.mean(dim=0)            # (O, d)
        Ivec = img_tokens.mean(dim=0)            # (D, d)

        # cluster separately
        # treat features as points across batch → (O,B) and (D,B)
        lab_tab, _ = kmeans_torch(X_tab.T.contiguous(), k=self.k)
        lab_img, _ = kmeans_torch(X_img.T.contiguous(), k=self.k)

        # cluster stats
        tab_idx, tab_energy, tab_cent = self._compute_cluster_stats(X_tab, lab_tab, Tvec)
        img_idx, img_energy, img_cent = self._compute_cluster_stats(X_img, lab_img, Ivec)

        # synchrony Ψ and Hungarian matching (maximize Ψ)
        Psi = self._sync_matrix(img_energy, img_cent, tab_energy, tab_cent)  # note (I,T)
        rr, cc = linear_sum_assignment((-Psi).cpu().numpy())  # maximize → minimize negative

        # regime γ per pair, using combined energies (simple merge of both modality energies)
        merged_energy = [float(e) for e in (img_energy + tab_energy)]
        gamma_map = self._regime_gamma(merged_energy, Psi)

        # build local sequences for each joint cluster
        local_orders: List[torch.Tensor] = []
        local_weights_sums: List[float] = []  # to compute α_J

        for (iI, iT) in zip(rr, cc):
            idxI = img_idx[iI]  # indices in image token space [0..D-1]
            idxT = tab_idx[iT]  # indices in tab token space   [0..O-1]
            if idxI.numel() == 0 and idxT.numel() == 0:
                continue

            # joint cluster indices in the final concatenated space [0..O+D-1]
            # first all tab [0..O-1], then image offset +O
            joint = []
            if idxT.numel() > 0:
                joint += idxT.tolist()
            if idxI.numel() > 0:
                joint += (idxI + O).tolist()
            joint = torch.tensor(joint, device=device, dtype=torch.long)

            # Within-cluster weights W_ij = |M(f_i) - M(f_j)|
            # Compute per-feature scalar "magnitude" M(f) for these features
            # Use the same magnitude as in _compute_cluster_stats for consistency
            metric = self.metric.lower()
            if metric in ("variance", "var"):
                # compute variance over batch of the scalar value for each token
                all_scalar = torch.cat([X_tab, X_img], dim=1)  # (B, O+D)
                Mval = all_scalar.var(dim=0, unbiased=False)   # (O+D,)
            elif metric in ("euclidean", "l2", "manhattan", "l1", "cosine", "cos", "correlation", "corr", "pearson"):
                all_scalar = torch.cat([X_tab, X_img], dim=1)  # (B, O+D)
                Mval = all_scalar.norm(p=2, dim=0)             # (O+D,)
            elif metric in ("kl", "kl_divergence", "js", "jensen-shannon"):
                all_scalar = torch.cat([X_tab, X_img], dim=1)
                _, P = shared_histograms(all_scalar, bins=self.bins)
                Mval = -(P * P.clamp_min(1e-12).log()).sum(dim=1)
            else:
                raise ValueError(f"Unknown metric {self.metric}")

            Mj = Mval[joint]                                    # (q,)
            W_full = (Mj[:, None] - Mj[None, :]).abs()          # (q,q)

            # sparsify edges by quantile threshold
            thr = torch.quantile(W_full.flatten(), q=self.sparsity_q).item()
            W_sparse = W_full.clone()
            W_sparse[W_sparse <= thr] = 0.0
            W_sparse.fill_diagonal_(0.0)

            # regime scaling γ(ρ, Ψ)
            g = gamma_map[(iI, iT)]
            W_sparse = g * W_sparse

            # spectral seriation → local order over 'joint'
            if W_sparse.sum() <= 0:
                # fallback: keep as-is to remain stable
                local = joint
                Wsum = float(Mj.sum().item())
            else:
                order_loc = spectral_seriation(W_sparse)        # (q,)
                local = joint[order_loc]                        # indices in [0..O+D-1]
                Wsum = float(W_sparse.sum().item())

            local_orders.append(local)
            local_weights_sums.append(max(1e-12, Wsum))

        if not local_orders:
            # degenerate: identity permutation
            perm = torch.arange(O + D, device=device)
        else:
            # α_J ranking by normalized energy proxy (sum of weights)
            Wtot = float(sum(local_weights_sums))
            alpha = [w / Wtot for w in local_weights_sums]

            # sort joint clusters by descending α
            order_clusters = sorted(range(len(alpha)), key=lambda i: alpha[i], reverse=True)
            # concatenate local sequences
            perm = torch.cat([local_orders[i] for i in order_clusters], dim=0)
            # deduplicate while preserving order (stable unique)
            seen = torch.zeros(O + D, dtype=torch.bool, device=device)
            perm_stable = []
            for idx in perm:
                if not seen[idx]:
                    seen[idx] = True
                    perm_stable.append(idx)
            # append any missing tokens (unassigned) in natural order to ensure full coverage
            if len(perm_stable) < (O + D):
                all_idx = torch.arange(O + D, device=device)
                missing = all_idx[~seen]
                perm = torch.cat([torch.stack(perm_stable), missing], dim=0)
            else:
                perm = torch.stack(perm_stable)

        # beta (normalized rank targets) for order head (no memory tokens yet)
        ranks = torch.arange(perm.numel(), device=device, dtype=torch.float32)
        beta = ranks / (ranks[-1] if ranks[-1] > 0 else 1.0)

        # cache
        self.perm_cached = perm.detach()
        self.beta_cached = beta.detach()
        return self.perm_cached, self.beta_cached


# ======================================================
# iSyncTab (Refined): Encoders → one-time NS–PFS → Linformer(+mem) → heads
# ======================================================
class iSyncTab(nn.Module):
    def __init__(self,
                 num_tab_features_hint: int,  # used to size Linformer; actual tokens can be >= this
                 num_classes: int,
                 d_model: int = 128,
                 num_clusters: int = 4,
                 metric: str = "variance",
                 bins: int = 32,
                 linformer_depth: int = 4,
                 linformer_heads: int = 4,
                 linformer_k: int = 32,
                 lambda_fs: float = 0.1,
                 num_memory_tokens: int = 2,
                 pretrained_resnet: bool = False,
                 image_in_channels: int = 3,
                 device: Optional[str] = None):
        super().__init__()
        self.lambda_fs = float(lambda_fs)
        self.num_memory_tokens = int(num_memory_tokens)

        self.tab_enc = TabularTokenEncoder(d_model=d_model, depth=2, heads=4)
        self.img_enc = ImageTokenEncoder(d_model=d_model, pretrained=pretrained_resnet,
                                         in_channels=image_in_channels)

        self.nspfs = NSPFSRefined(num_clusters=num_clusters, metric=metric, bins=bins)

        # NOTE: we allocate a safe upper bound for seq_len:
        #   hint + 49 image tokens + memory tokens
        self.seq_len_max = int(num_tab_features_hint) + 49 + self.num_memory_tokens

        self.linformer = Linformer(
            dim=d_model, seq_len=self.seq_len_max,
            depth=linformer_depth, heads=linformer_heads, k=linformer_k,
            one_kv_head=True, share_kv=True
        )
        self.cls_head = nn.Linear(d_model, num_classes)
        self.seq_head = nn.Linear(d_model, 1)

        # memory tokens (trainable, prepended)
        self.mem_tokens = nn.Parameter(torch.randn(self.num_memory_tokens, d_model) * 0.02)

        # cached sizes to sanity-check perm length
        self.cached_total_tokens: Optional[int] = None

    @torch.no_grad()
    def reset_order(self):
        """Reset cached permutation so it will be recomputed on next forward."""
        self.nspfs.reset()
        self.cached_total_tokens = None

    def _ensure_perm(self, t_tok: torch.Tensor, i_tok: torch.Tensor):
        """
        Fit NS–PFS once (encoders in eval/no_grad) and cache perm & beta.
        """
        if not self.nspfs.has_perm():
            self.tab_enc.eval(); self.img_enc.eval()
            self.nspfs.fit_from_tokens(t_tok, i_tok)
            self.tab_enc.train(); self.img_enc.train()

    def forward(self, x_tab, x_img, y: Optional[torch.Tensor] = None):
        """
        x_tab: tensor or dict for tabular fields (see TabularTokenEncoder)
        x_img: images (B,C,H,W) or (B,1,H,W)
        """
        B = x_img.shape[0]
        t_tok = self.tab_enc(x_tab)                     # (B, O, d)
        i_tok = self.img_enc(x_img)                     # (B, 49, d)

        # fit once
        self._ensure_perm(t_tok.detach(), i_tok.detach())

        O = t_tok.shape[1]
        D = i_tok.shape[1]
        L = O + D
        perm = self.nspfs.perm_cached
        beta = self.nspfs.beta_cached                   # length L (no mem)

        # cache and check Linformer maximum length
        if (self.cached_total_tokens is None) or (self.cached_total_tokens != L):
            self.cached_total_tokens = L
            assert (L + self.num_memory_tokens) <= self.seq_len_max, \
                f"seq_len {L+self.num_memory_tokens} exceeds Linformer bound {self.seq_len_max}. " \
                f"Increase num_tab_features_hint."

        # apply global permutation to concatenated tokens
        tokens = torch.cat([t_tok, i_tok], dim=1)       # (B, L, d)
        tokens = tokens[:, perm, :]                     # (B, L, d)

        # prepend memory tokens
        mem = self.mem_tokens.unsqueeze(0).expand(B, -1, -1)  # (B, M, d)
        z_in = torch.cat([mem, tokens], dim=1)                 # (B, M+L, d)

        # encode with Linformer
        z = self.linformer(z_in)                         # (B, M+L, d)

        # classification head (use memory-pooled representation)
        # you can also use z.mean(1); we take mean over memory tokens for a dedicated "controller" rep
        z_mem = z[:, :self.num_memory_tokens, :].mean(dim=1)   # (B,d)
        logits = self.cls_head(z_mem)                           # (B,C)

        # sequence order head (only supervise non-memory tokens)
        z_seq = z[:, self.num_memory_tokens:, :]                # (B, L, d)
        seq_scores = self.seq_head(z_seq).squeeze(-1)           # (B, L)

        out = {"logits": logits, "perm": perm, "seq_scores": seq_scores, "beta": beta}

        if y is not None:
            ce = F.cross_entropy(logits, y)
            # match each sample's score against the same beta target (fixed global order)
            beta_exp = beta.unsqueeze(0).expand(B, -1)         # (B, L)
            fs = F.mse_loss(seq_scores, beta_exp)
            loss = ce + self.lambda_fs * fs
            out.update({"loss": loss, "loss_ce": ce, "loss_fs": fs})

        return out