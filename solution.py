import os
os.environ.setdefault("HF_HUB_OFFLINE", "0")
import time, random, copy
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 1234
RES = 224
NFOLD = 5
PCA_DIM = 96
XA_DEPTH = 1
XA_DH = 128
XA_EP = 18
XA_SEEDS = 2
LAM_FLOOR = 0.08
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AMP = (DEVICE == "cuda")
if DEVICE == "cpu":
    XA_EP = 8
    XA_SEEDS = 1


def find_base():
    here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    for c in [os.getcwd(), here, os.path.join(here, "dataset", "public"),
              os.path.join(os.getcwd(), "dataset", "public"), os.path.join(here, "..")]:
        if os.path.exists(os.path.join(c, "train.csv")) and os.path.exists(os.path.join(c, "test.csv")):
            return os.path.abspath(c)
    return os.path.abspath(os.getcwd())


BASE = find_base()
train_df = pd.read_csv(os.path.join(BASE, "train.csv"))
test_df = pd.read_csv(os.path.join(BASE, "test.csv"))
OUT_DIR = os.path.join(os.getcwd(), "working")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = os.path.join(OUT_DIR, "submission.csv")


def write_sub(probs):
    probs = np.clip(np.asarray(probs, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    pd.DataFrame({"id": test_df["id"].values,
                  "prob_left_higher_division_pressure": probs}).to_csv(OUT_PATH, index=False)


write_sub(np.full(len(test_df), 0.5))


def resolve_root(rel):
    here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    for r in [BASE, os.getcwd(), here, os.path.join(BASE, "dataset", "public"), os.path.join(BASE, "public")]:
        if os.path.exists(os.path.join(r, rel)):
            return r
    return BASE


IMG_ROOT = resolve_root(train_df.iloc[0]["left_image_path"])
all_paths = sorted(set(train_df.left_image_path).union(train_df.right_image_path)
                   .union(test_df.left_image_path).union(test_df.right_image_path))
pidx = {p: i for i, p in enumerate(all_paths)}
N = len(all_paths)


def load_img(rel):
    a = Image.open(os.path.join(IMG_ROOT, rel)).convert("L").resize((RES, RES), Image.BILINEAR)
    a = np.asarray(a, dtype=np.float32)
    return (a - a.mean()) / (a.std() + 1e-6)


cache = torch.from_numpy(np.stack([load_img(p) for p in all_paths]).astype(np.float32))
if DEVICE == "cuda":
    cache = cache.to(DEVICE)

trL = np.array([pidx[p] for p in train_df.left_image_path])
trR = np.array([pidx[p] for p in train_df.right_image_path])
trY = train_df.left_higher_division_pressure.values.astype(np.float64)
trW = train_df.pair_weight.values.astype(np.float64)
teL = np.array([pidx[p] for p in test_df.left_image_path])
teR = np.array([pidx[p] for p in test_df.right_image_path])
train_imgs = sorted(set(trL.tolist()) | set(trR.tolist()))


def d4_global(forward_fn, chunk):
    feats = None
    with torch.no_grad():
        for fh in [False, True]:
            for rk in range(4):
                base = cache.unsqueeze(1).repeat(1, 3, 1, 1)
                if fh:
                    base = torch.flip(base, [3])
                if rk:
                    base = torch.rot90(base, rk, [2, 3])
                outs = []
                for j in range(0, N, chunk):
                    with torch.autocast(device_type=DEVICE, enabled=AMP):
                        outs.append(forward_fn(base[j:j + chunk]).float())
                cat = torch.cat(outs, 0)
                feats = cat if feats is None else feats + cat
    return (feats / 8.0).cpu().numpy()


PER_FEATS = []
TOK = None
dino = None
try:
    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=True, source="github").to(DEVICE).eval()
    PER_FEATS.append(d4_global(lambda x: torch.cat([dino.forward_features(x)["x_norm_clstoken"],
                                                    dino.forward_features(x)["x_norm_patchtokens"].mean(1)], 1), 32))
    with torch.no_grad():
        nt = dino.forward_features(cache[:2].unsqueeze(1).repeat(1, 3, 1, 1))["x_norm_patchtokens"].shape[1]
        TOK = np.zeros((N, 4, nt, 384), np.float16)
        for rk in range(4):
            x = cache.unsqueeze(1).repeat(1, 3, 1, 1)
            if rk:
                x = torch.rot90(x, rk, [2, 3])
            for j in range(0, N, 24):
                with torch.autocast(device_type=DEVICE, enabled=AMP):
                    TOK[j:j + 24, rk] = dino.forward_features(x[j:j + 24])["x_norm_patchtokens"].float().cpu().numpy().astype(np.float16)
    del dino
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    print("dino ok; tokens", None if TOK is None else TOK.shape)
except Exception as e:
    print("dino unavailable:", repr(e)[:120])
try:
    from torchvision import models
    em = models.efficientnet_b0(weights="IMAGENET1K_V1").features.to(DEVICE).eval()
    PER_FEATS.append(d4_global(lambda x: F.adaptive_avg_pool2d(em(x), 1).flatten(1), 96))
    del em
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    print("effb0 ok")
except Exception as e:
    print("effb0 unavailable:", repr(e)[:120])

TOKt = torch.from_numpy(TOK) if TOK is not None else None


def spectral_folds(K, seed):
    idx = {v: i for i, v in enumerate(train_imgs)}
    M = len(train_imgs)
    A = np.zeros((M, M))
    for a, b in zip(trL, trR):
        A[idx[a], idx[b]] += 1.0
        A[idx[b], idx[a]] += 1.0
    D = A.sum(1)
    dinv = 1.0 / np.sqrt(np.maximum(D, 1e-9))
    Lap = np.eye(M) - (dinv[:, None] * A * dinv[None, :])
    w, v = np.linalg.eigh(Lap)
    emb = v[:, 1:K]
    rng = np.random.RandomState(seed)
    C = emb[rng.choice(M, K, replace=False)].copy()
    lab = np.zeros(M, dtype=int)
    for _ in range(60):
        d = ((emb[:, None, :] - C[None, :, :]) ** 2).sum(2)
        lab = d.argmin(1)
        for k in range(K):
            mm = lab == k
            if mm.any():
                C[k] = emb[mm].mean(0)
    out = np.full(N, -1, dtype=int)
    for v2, i in idx.items():
        out[v2] = lab[i]
    return out


def sig(t):
    return 1.0 / (1.0 + np.exp(-np.clip(t, -30, 30)))


def wll(y, p, w):
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return 100.0 * float((w * (-(y * np.log(p) + (1.0 - y) * np.log1p(-p)))).sum() / w.sum())


def pca(F0, k):
    fc = F0 - F0.mean(0)
    U, S, Vt = np.linalg.svd(fc, full_matrices=False)
    P = fc @ Vt[:min(k, Vt.shape[0])].T
    return (P - P.mean(0)) / (P.std(0) + 1e-9)


def per_ranker(P, tri, vate_idx, te_idx):
    X = P[trL] - P[trR]
    w = np.zeros(X.shape[1])
    b = 0.0
    for _ in range(2500):
        p = sig(X[tri] @ w + b)
        g = (p - trY[tri]) * trW[tri]
        w -= 0.3 * (X[tri].T @ g / trW[tri].sum() + 1e-2 * w)
        b -= 0.3 * (g.sum() / trW[tri].sum())
    Xte = P[te_idx[0]] - P[te_idx[1]]
    return X[vate_idx] @ w + b, Xte @ w + b


class CrossPair(nn.Module):
    def __init__(self, d=384, dh=128, heads=4, depth=1):
        super().__init__()
        self.proj = nn.Linear(d, dh)
        self.ln = nn.LayerNorm(dh)
        self.cross = nn.ModuleList([nn.MultiheadAttention(dh, heads, batch_first=True, dropout=0.1) for _ in range(depth)])
        self.q = nn.Parameter(torch.randn(1, 1, dh) * 0.02)
        self.pool = nn.MultiheadAttention(dh, heads, batch_first=True)
        self.head = nn.Sequential(nn.Linear(dh * 3, dh), nn.GELU(), nn.Dropout(0.3), nn.Linear(dh, 1))

    def forward(self, A, B):
        a = self.ln(self.proj(A))
        b = self.ln(self.proj(B))
        for cr in self.cross:
            a2, _ = cr(a, b, b)
            a = a + a2
        q = self.q.expand(a.shape[0], -1, -1)
        pooled, _ = self.pool(q, a, a)
        pooled = pooled[:, 0]
        bm = b.mean(1)
        return self.head(torch.cat([pooled, pooled - bm, pooled * bm], 1)).squeeze(1)


def train_cross(tri, seed):
    torch.manual_seed(seed)
    net = CrossPair(dh=XA_DH, depth=XA_DEPTH).to(DEVICE)
    ema = copy.deepcopy(net)
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(net.parameters(), lr=8e-4, weight_decay=1e-2)
    nb = max(1, len(tri) // 32 + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=8e-4, total_steps=XA_EP * nb, pct_start=0.2)
    scaler = torch.cuda.amp.GradScaler(enabled=AMP)
    Lt, Rt = trL[tri], trR[tri]
    Yt, Wt = trY[tri].astype(np.float32), trW[tri].astype(np.float32)
    for ep in range(XA_EP):
        net.train()
        perm = np.random.permutation(len(tri))
        for i in range(0, len(tri), 32):
            bi = perm[i:i + 32]
            ro = np.random.randint(0, 4, len(bi))
            A = TOKt[Lt[bi], ro].to(DEVICE).float()
            B = TOKt[Rt[bi], ro].to(DEVICE).float()
            A = A * (torch.rand_like(A[:, :, :1]) > 0.15).float() + torch.randn_like(A) * 0.05
            B = B * (torch.rand_like(B[:, :, :1]) > 0.15).float() + torch.randn_like(B) * 0.05
            yy = torch.from_numpy(Yt[bi]).to(DEVICE)
            ww = torch.from_numpy(Wt[bi]).to(DEVICE)
            ww = ww / ww.mean()
            with torch.autocast(device_type=DEVICE, enabled=AMP):
                zAB = net(A, B)
                zBA = net(B, A)
                loss = (F.binary_cross_entropy_with_logits(zAB, yy, reduction="none") * ww).mean() \
                    + (F.binary_cross_entropy_with_logits(zBA, 1 - yy, reduction="none") * ww).mean() \
                    + 0.1 * ((zAB + zBA) ** 2).mean()
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            with torch.no_grad():
                for pe, pn in zip(ema.parameters(), net.parameters()):
                    pe.mul_(0.997).add_(pn, alpha=0.003)
    return ema


def cross_predict(model, li, ri):
    model.eval()
    out = np.zeros(len(li))
    with torch.no_grad():
        for i in range(0, len(li), 64):
            A = TOKt[li[i:i + 64], 0].to(DEVICE).float()
            B = TOKt[ri[i:i + 64], 0].to(DEVICE).float()
            with torch.autocast(device_type=DEVICE, enabled=AMP):
                z = 0.5 * (model(A, B).float() - model(B, A).float())
            out[i:i + 64] = z.cpu().numpy()
    return out


fold_of = spectral_folds(NFOLD, 0)
lf = fold_of[trL]
rf = fold_of[trR]
PCAS = [pca(F0, PCA_DIM) for F0 in PER_FEATS]

oof_logit = np.zeros(len(trY))
test_logit = np.zeros(len(teL))
scored = np.zeros(len(trY), dtype=bool)
n_streams = 0


def add_stream(oof, test, sc):
    global oof_logit, test_logit, n_streams
    sd = np.nanstd(oof[sc]) + 1e-9
    oof_logit[sc] += oof[sc] / sd
    test_logit += test / sd
    n_streams += 1


for P in PCAS:
    oof = np.full(len(trY), np.nan)
    tst = np.zeros(len(teL))
    for k in range(NFOLD):
        tri = np.where((lf != k) & (rf != k))[0]
        vai = np.where((lf == k) & (rf == k))[0]
        scored[vai] = True
        ov, tv = per_ranker(P, tri, vai, (teL, teR))
        oof[vai] = ov
        tst += tv
    tst /= NFOLD
    add_stream(oof, tst, scored)

if TOKt is not None:
    try:
        oof = np.full(len(trY), np.nan)
        tst = np.zeros(len(teL))
        cnt_t = 0
        for k in range(NFOLD):
            tri = np.where((lf != k) & (rf != k))[0]
            vai = np.where((lf == k) & (rf == k))[0]
            scored[vai] = True
            for s in range(XA_SEEDS):
                m = train_cross(tri, SEED + 100 * s + k)
                vpred = cross_predict(m, trL[vai], trR[vai])
                oof[vai] = vpred if s == 0 else (oof[vai] + vpred)
                tst += cross_predict(m, teL, teR)
                cnt_t += 1
            oof[vai] /= XA_SEEDS
        tst /= max(1, cnt_t)
        add_stream(oof, tst, scored)
        print("cross-attention stream added")
    except Exception as e:
        print("cross-attention failed:", repr(e)[:160])

a, best_lam, best_s = 1.0, LAM_FLOOR, float("nan")
if n_streams > 0:
    oof_logit[scored] /= n_streams
    test_logit /= n_streams
    try:
        zy, zw, zz = trY[scored], trW[scored], oof_logit[scored]
        lo, hi = 0.0, 8.0
        for _ in range(80):
            m1 = lo + (hi - lo) / 3
            m2 = hi - (hi - lo) / 3
            if wll(zy, sig(m1 * zz), zw) < wll(zy, sig(m2 * zz), zw):
                hi = m2
            else:
                lo = m1
        a = 0.5 * (lo + hi)
        pcal = sig(a * zz)
        best_lam, best_s = 0.0, wll(zy, pcal, zw)
        for lam in np.linspace(0, 0.6, 61):
            s2 = wll(zy, (1 - lam) * pcal + lam * 0.5, zw)
            if s2 < best_s:
                best_s, best_lam = s2, lam
        best_lam = max(best_lam, LAM_FLOOR)
        print("streams:", n_streams, "OOF pairs:", int(scored.sum()),
              "calib:", round(best_s, 3), "a:", round(a, 3), "lam:", round(best_lam, 3))
    except Exception as e:
        print("calibration failed:", repr(e)[:160])

test_prob = (1 - best_lam) * sig(a * test_logit) + best_lam * 0.5
write_sub(test_prob)
print("wrote", OUT_PATH, "rows:", len(test_df))
