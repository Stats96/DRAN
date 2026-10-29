"""
DRAEN-SA: complete, leakage-free analysis pipeline
==================================================

One script that reproduces every analysis described in the manuscript:

  sim     Simulation study (Table 2): ridge-Cox, random survival forest (RSF) and DRAEN-SA,
          200 replicates x 3 scenarios, 70/30 split, ~50% censoring (calibrated), oracle reference.
  real    SCAN-B (GSE96058) analysis (Tables 3-4, Figs 3-4): ten-fold cross-validation with
          patient-grouped folds shared by ALL models, in-fold gene screening and scaling,
          inner validation split for early stopping, inner CV for the comparators' hyper-parameters,
          C-index (Harrell, Uno), time-dependent AUC, Brier score (+ Kaplan-Meier reference),
          paired fold-wise tests, and gradient x input feature importance.
  report  Aggregates the saved per-fold results into CSV/LaTeX tables and figures.

Design rules that make the estimates honest
-------------------------------------------
* Gene screening (univariable Cox score test) and feature scaling are fitted INSIDE each training fold.
* Folds are stratified by event status and grouped by patient (technical replicates never straddle folds);
  one fold file is written and used by every model and every feature set.
* DRAEN-SA early-stops on an inner validation split taken from the TRAINING folds; the held-out fold is
  used only once, for evaluation. Comparators tune their penalties by inner 3-fold CV on the training folds.
* The training loss is the exact Efron negative log partial likelihood (verified against lifelines).
* Everything is seeded and resumable (finished fold/model pairs are skipped on restart).

Input file (tab- or comma-separated; one row per RNA-seq profile)
-----------------------------------------------------------------
Columns: OS_days, OS_event, the 11 clinical covariates (names in CLINICAL below), every other column is a gene.
Optional: a patient-id column (--id_col) so that technical replicates share a fold.

Examples
--------
  python draen_pipeline.py real --data full_dataset.tsv --id_col patient_id --out results_real
  python draen_pipeline.py real --data p1e-11.txt p1e-10.txt p1e-9.txt --screen none --out results_prescreened
  python draen_pipeline.py sim  --out results_sim
  python draen_pipeline.py report --real_dir results_real --sim_dir results_sim
  python draen_pipeline.py real --data full_dataset.tsv --smoke --out smoke      # quick test (minutes)

Requires: python>=3.10, tensorflow>=2.16 (Keras 3), scikit-survival, scikit-learn, lifelines (optional test),
numpy, pandas, scipy, matplotlib.
"""
import argparse
import json
import os
import platform
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq
from scipy.stats import multivariate_normal
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import RobustScaler
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
from sksurv.metrics import (brier_score, concordance_index_censored, concordance_index_ipcw,
                            cumulative_dynamic_auc)
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.util import Surv

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf  # noqa: E402
from tensorflow.keras import layers, models, optimizers, regularizers  # noqa: E402

CLINICAL = ["age_at_diagnosis", "tumor_size_mm", "lymph_node_status", "er_status", "pgr_status",
            "her2_status", "ki67_status", "nhg_grade", "pam50_subtype", "endocrine_treated", "chemo_treated"]
TIME, EVENT = "OS_days", "OS_event"
DAYS_PER_YEAR = 365.25
HORIZONS = [1, 2, 3, 4, 5]


# ============================================================================================
# 1. Efron loss (exact, vectorised, graph-compatible)
# ============================================================================================
def efron_loss(y_true, y_pred):
    """Negative Efron log partial likelihood, averaged over events in the batch.
    y_true[:,0]=time, y_true[:,1]=event, y_pred=log-risk.  Tied events at time t (m of them): the l-th gets
    eta_i - log( sum_{risk set} e^eta - (l/m) sum_{tied events} e^eta )."""
    t = tf.cast(y_true[:, 0], tf.float32)
    d = tf.cast(y_true[:, 1], tf.float32)
    eta = tf.reshape(tf.cast(y_pred, tf.float32), [-1])
    eta = eta - tf.reduce_max(eta)
    w = tf.exp(eta)
    risk = tf.cast(t[None, :] >= t[:, None], tf.float32)
    same = tf.cast(tf.equal(t[None, :], t[:, None]), tf.float32) * d[None, :]
    s_risk = tf.linalg.matvec(risk, w)
    s_tied = tf.linalg.matvec(same, w)
    m = tf.reduce_sum(same, axis=1)
    idx = tf.range(tf.shape(t)[0])
    earlier = tf.cast(idx[None, :] < idx[:, None], tf.float32)
    l = tf.reduce_sum(same * earlier, axis=1)
    denom = s_risk - (l / tf.maximum(m, 1.0)) * s_tied
    ll = d * (eta - tf.math.log(tf.maximum(denom, 1e-12)))
    return -tf.reduce_sum(ll) / (tf.reduce_sum(d) + 1e-7)


# ============================================================================================
# 2. Breslow baseline hazard and survival probabilities
# ============================================================================================
class Breslow:
    def fit(self, risk, t, e):
        risk = np.clip(np.asarray(risk, float), -30, 30)
        ut = np.unique(t[e == 1])
        w = np.exp(risk)
        order = np.argsort(t)
        ts, ws = t[order], w[order]
        suffix = np.cumsum(ws[::-1])[::-1]
        first = np.searchsorted(ts, ut, side="left")
        denom = suffix[first]
        dj = np.array([np.sum((t == u) & (e == 1)) for u in ut], float)
        self.ut, self.H = ut, np.cumsum(dj / denom)
        return self

    def surv(self, risk, times):
        risk = np.clip(np.asarray(risk, float), -30, 30)
        pos = np.searchsorted(self.ut, times, side="right") - 1
        H = np.where(pos >= 0, self.H[np.clip(pos, 0, None)], 0.0)
        return np.exp(-H[None, :] * np.exp(risk)[:, None])


# ============================================================================================
# 3. DRAEN-SA network (and ablation switches)
# ============================================================================================
def build_draen(d, k=1024, n_blocks=4, attention=True, residual=True, l2=5e-4, noise=0.1, in_drop=0.35,
                block_drops=(0.20, 0.25, 0.30, 0.35), head=(512, 256, 128), head_drops=(0.30, 0.25, 0.20)):
    reg = regularizers.l2(l2)
    inp = tf.keras.Input(shape=(d,))
    x = layers.GaussianNoise(noise)(inp)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(in_drop)(x)
    x = layers.Dense(2 * k, activation="gelu")(x)
    x = layers.Dense(k)(x)
    for i in range(n_blocks):
        h = layers.Dense(k, kernel_regularizer=reg)(x)
        h = layers.Activation("gelu")(h)
        h = layers.Dropout(block_drops[min(i, len(block_drops) - 1)])(h)
        h = layers.Dense(k, kernel_regularizer=reg)(h)
        x = layers.Add()([h, x]) if residual else h
        x = layers.Activation("gelu")(x)
    if attention:
        a = layers.Dense(k, activation="tanh")(x)
        a = layers.Dense(k, activation="softmax")(a)
        x = layers.Multiply()([x, a])
    for u, p in zip(head, head_drops):
        x = layers.Dense(u, activation="gelu")(x)
        x = layers.Dropout(p)(x)
    out = layers.Dense(1)(x)
    return models.Model(inp, out)


class DraenEnsemble:
    """Ensemble of independently initialised networks; early stopping on an inner validation split."""

    def __init__(self, cfg, variant="full", seed=0):
        self.cfg, self.variant, self.seed = cfg, variant, seed
        self.members, self.breslow = [], Breslow()

    def _arch(self):
        kw = dict(k=self.cfg.latent)
        if self.variant == "noattn":
            kw["attention"] = False
        if self.variant == "nores":
            kw["residual"] = False
        return kw

    def fit(self, X, t, e):
        self.members = []
        n_members = 1 if self.variant == "single" else self.cfg.n_ensemble
        tr, va = train_test_split(np.arange(len(t)), test_size=self.cfg.val_frac, stratify=e, random_state=self.seed)
        Xtr, Xva = X[tr].astype("float32"), X[va].astype("float32")
        ytr = np.stack([t[tr], e[tr]], 1).astype("float32")
        yva = np.stack([t[va], e[va]], 1).astype("float32")
        for m in range(n_members):
            tf.keras.utils.set_random_seed(self.seed * 1000 + m)
            net = build_draen(X.shape[1], **self._arch())
            net.compile(optimizer=optimizers.AdamW(learning_rate=self.cfg.lr, weight_decay=self.cfg.wd), loss=efron_loss)
            net.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=self.cfg.epochs, batch_size=self.cfg.batch,
                    validation_batch_size=len(va), verbose=0,
                    callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=self.cfg.patience,
                                                                restore_best_weights=True)])
            self.members.append(net)
        self.breslow.fit(self.risk(X), t, e)
        return self

    def risk(self, X):
        X = np.asarray(X, "float32")
        return np.mean([m.predict(X, verbose=0, batch_size=512).ravel() for m in self.members], axis=0)

    def surv(self, X, times):
        return self.breslow.surv(self.risk(X), times)

    def grad_x_input(self, X):
        """Signed gradient x input attribution of the ensemble log-risk, shape (n, d)."""
        Xt = tf.convert_to_tensor(np.asarray(X, "float32"))
        out = np.zeros(Xt.shape, "float32")
        for m in self.members:
            with tf.GradientTape() as tape:
                tape.watch(Xt)
                r = tf.reduce_sum(m(Xt, training=False))
            out += (tape.gradient(r, Xt) * Xt).numpy()
        return out / len(self.members)


# ============================================================================================
# 4. Comparison models (scikit-survival), tuned by inner 3-fold CV inside the training data
# ============================================================================================
def _step_eval(fns, times):
    rows = []
    for f in fns:
        lo, hi = f.domain
        rows.append(f(np.clip(times, lo, hi)))
    return np.vstack(rows)


def _inner_cv_score(make, params, X, t, e, seed):
    skf = StratifiedKFold(3, shuffle=True, random_state=seed)
    sc = []
    for a, b in skf.split(X, e):
        try:
            m = make(**params).fit(X[a], Surv.from_arrays(e[a].astype(bool), t[a]))
            sc.append(concordance_index_censored(e[b].astype(bool), t[b], m.predict(X[b]))[0])
        except Exception:
            sc.append(np.nan)
    return np.nanmean(sc) if not np.all(np.isnan(sc)) else -np.inf


class SkModel:
    def __init__(self, kind, cfg, seed=0):
        self.kind, self.cfg, self.seed = kind, cfg, seed

    def fit(self, X, t, e):
        y = Surv.from_arrays(e.astype(bool), t)
        if self.kind == "ridge":
            make = lambda alpha: CoxPHSurvivalAnalysis(alpha=alpha, ties="efron", n_iter=200)
            grid = [{"alpha": a} for a in (0.1, 1, 10, 100, 1000, 10000)]
        elif self.kind == "lasso":
            make = lambda alpha: CoxnetSurvivalAnalysis(l1_ratio=1.0, alphas=[alpha], fit_baseline_model=True,
                                                        max_iter=100000)
            grid = [{"alpha": a} for a in (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1)]
        elif self.kind == "clinical":      # clinical-only ridge-Cox (reference model)
            self.cols = [i for i, c in enumerate(self.cfg.feature_names) if c in CLINICAL]
            make = lambda alpha: CoxPHSurvivalAnalysis(alpha=alpha, ties="efron", n_iter=200)
            grid = [{"alpha": a} for a in (0.1, 1, 10, 100)]
            X = X[:, self.cols]
        elif self.kind == "rsf":
            trees = self.cfg.rsf_trees
            make = lambda min_samples_leaf: RandomSurvivalForest(
                n_estimators=max(50, trees // 4), min_samples_leaf=min_samples_leaf, max_features="sqrt",
                n_jobs=-1, random_state=self.seed)
            grid = [{"min_samples_leaf": s} for s in (5, 15, 30)]
        else:
            raise ValueError(self.kind)
        scores = [_inner_cv_score(make, g, X, t, e, self.seed) for g in grid]
        best = grid[int(np.argmax(scores))]
        self.best = best
        if self.kind == "rsf":
            self.m = RandomSurvivalForest(n_estimators=self.cfg.rsf_trees, max_features="sqrt", n_jobs=-1,
                                          random_state=self.seed, **best).fit(X, y)
        else:
            self.m = make(**best).fit(X, y)
        return self

    def _x(self, X):
        return X[:, self.cols] if self.kind == "clinical" else X

    def risk(self, X):
        return self.m.predict(self._x(X))

    def surv(self, X, times):
        return _step_eval(self.m.predict_survival_function(self._x(X)), times)


def make_model(name, cfg, seed):
    if name in ("ridge", "lasso", "rsf", "clinical"):
        return SkModel(name, cfg, seed)
    if name == "draen":
        return DraenEnsemble(cfg, "full", seed)
    if name in ("draen_noattn", "draen_nores", "draen_single"):
        return DraenEnsemble(cfg, name.split("_")[1], seed)
    if name == "mlp":                       # DeepSurv-style baseline: plain MLP with the same Efron loss
        return MlpCox(cfg, seed)
    raise ValueError(name)


class MlpCox(DraenEnsemble):
    def __init__(self, cfg, seed=0):
        super().__init__(cfg, "mlp", seed)

    def fit(self, X, t, e):
        tr, va = train_test_split(np.arange(len(t)), test_size=self.cfg.val_frac, stratify=e, random_state=self.seed)
        tf.keras.utils.set_random_seed(self.seed)
        inp = tf.keras.Input(shape=(X.shape[1],))
        x = inp
        for u in (256, 128, 64):
            x = layers.Dense(u, activation="gelu")(x)
            x = layers.Dropout(0.3)(x)
        net = models.Model(inp, layers.Dense(1)(x))
        net.compile(optimizer=optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4), loss=efron_loss)
        net.fit(X[tr].astype("float32"), np.stack([t[tr], e[tr]], 1).astype("float32"),
                validation_data=(X[va].astype("float32"), np.stack([t[va], e[va]], 1).astype("float32")),
                epochs=self.cfg.epochs, batch_size=self.cfg.batch, validation_batch_size=len(va), verbose=0,
                callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=self.cfg.patience,
                                                            restore_best_weights=True)])
        self.members = [net]
        self.breslow.fit(self.risk(X), t, e)
        return self


# ============================================================================================
# 5. Metrics
# ============================================================================================
def evaluate(model, Xtr, ttr, etr, Xte, tte, ete, horizons_years):
    ytr, yte = Surv.from_arrays(etr.astype(bool), ttr), Surv.from_arrays(ete.astype(bool), tte)
    r_te, r_tr = model.risk(Xte), model.risk(Xtr)
    out = {"c_train": concordance_index_censored(etr.astype(bool), ttr, r_tr)[0],
           "c_harrell": concordance_index_censored(ete.astype(bool), tte, r_te)[0]}
    tau = min(tte.max(), ttr.max()) * 0.999
    try:
        out["c_uno"] = concordance_index_ipcw(ytr, yte, r_te, tau=tau)[0]
    except Exception:
        out["c_uno"] = np.nan
    lo = max(tte[ete == 1].min(), ttr.min()) if (ete == 1).any() else tte.min()
    hi = min(tte.max(), ttr.max()) * 0.999
    km_t, km_s = kaplan_meier_estimator(etr.astype(bool), ttr)
    for y in horizons_years:
        h = y * DAYS_PER_YEAR
        auc = bs = bs0 = np.nan
        if lo < h < hi:
            try:
                auc = cumulative_dynamic_auc(ytr, yte, r_te, np.array([h]))[0][0]
            except Exception:
                pass
            try:
                S = model.surv(Xte, np.array([h]))
                bs = brier_score(ytr, yte, S, np.array([h]))[1][0]
                s0 = km_s[max(np.searchsorted(km_t, h, side="right") - 1, 0)]
                bs0 = brier_score(ytr, yte, np.full((len(tte), 1), s0), np.array([h]))[1][0]
            except Exception:
                pass
        out[f"auc_{y}y"], out[f"brier_{y}y"], out[f"brier_null_{y}y"] = auc, bs, bs0
    return out


# ============================================================================================
# 6. In-fold gene screening: fast univariable Cox score test (vectorised over genes)
# ============================================================================================
def cox_score_pvalues(X, t, e):
    """Univariable Cox score-test p-values for every column of X (Breslow handling of ties)."""
    order = np.argsort(t, kind="stable")
    ts, es, Xs = t[order], e[order], X[order].astype(np.float64)
    n = len(ts)
    first = np.searchsorted(ts, ts, side="left")
    S1 = np.cumsum(Xs[::-1], axis=0)[::-1]
    S2 = np.cumsum((Xs ** 2)[::-1], axis=0)[::-1]
    ev = es == 1
    nr = (n - first[ev])[:, None].astype(np.float64)
    m1 = S1[first[ev]] / nr
    v = S2[first[ev]] / nr - m1 ** 2
    U = (Xs[ev] - m1).sum(0)
    I = v.sum(0)
    chi2 = np.where(I > 1e-12, U ** 2 / np.maximum(I, 1e-12), 0.0)
    return stats.chi2.sf(chi2, 1)


# ============================================================================================
# 7. Real-data analysis
# ============================================================================================
def load_table(path):
    sep = "\t" if path.endswith((".txt", ".tsv")) else ","
    return pd.read_csv(path, sep=sep)


def prepare_real(args):
    dfs = [load_table(p) for p in args.data]
    for d in dfs:
        d.dropna(subset=[TIME, EVENT], inplace=True)
        d.reset_index(drop=True, inplace=True)
    df0 = dfs[0]
    groups = df0[args.id_col].values if args.id_col and args.id_col in df0 else np.arange(len(df0))
    if not (args.id_col and args.id_col in df0):
        print("WARNING: no patient-id column: folds are formed at the profile level (technical replicates may "
              "straddle folds). Pass --id_col to group by patient.")
    return dfs, groups


def make_folds(e, groups, k, seed, path):
    if os.path.exists(path):
        return pd.read_csv(path)["fold"].values
    fold = np.zeros(len(e), int)
    sgk = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    for f, (_, te) in enumerate(sgk.split(np.zeros(len(e)), e, groups)):
        fold[te] = f
    pd.DataFrame({"fold": fold}).to_csv(path, index=False)
    return fold


def run_real(args):
    os.makedirs(args.out, exist_ok=True)
    cfg = args
    dfs, groups = prepare_real(args)
    df0 = dfs[0]
    t, e = df0[TIME].values.astype(float), df0[EVENT].values.astype(int)
    fold = make_folds(e, groups, args.folds, args.seed, os.path.join(args.out, "folds.csv"))
    clin = [c for c in CLINICAL if c in df0.columns]
    gene_cols = [c for c in df0.columns if c not in clin + [TIME, EVENT, args.id_col]]
    # feature-set definitions
    if args.screen == "infold":
        sets = [(f"top{k}", k) for k in args.n_genes]
    else:
        sets = [(os.path.basename(p), None) for p in args.data]
    res_path = os.path.join(args.out, "fold_results.csv")
    done = set()
    rows = pd.read_csv(res_path).to_dict("records") if os.path.exists(res_path) else []
    for r in rows:
        done.add((r["set"], r["fold"], r["model"]))
    imp_store = {}
    for si, (sname, K) in enumerate(sets):
        df = df0 if args.screen == "infold" else dfs[si]
        feats_all = [c for c in df.columns if c not in [TIME, EVENT, args.id_col]]
        genes = [c for c in feats_all if c not in clin]
        for f in range(args.folds):
            tr, te = np.where(fold != f)[0], np.where(fold == f)[0]
            if args.screen == "infold":
                p = cox_score_pvalues(df[genes].values[tr], t[tr], e[tr])
                sel = [genes[i] for i in np.argsort(p)[:K]]
            else:
                sel = genes
            feat = sel + clin
            cfg.feature_names = feat
            sc = RobustScaler().fit(df.loc[tr, feat].values)
            Xtr, Xte = sc.transform(df.loc[tr, feat].values), sc.transform(df.loc[te, feat].values)
            Xtr, Xte = np.clip(Xtr, -10, 10), np.clip(Xte, -10, 10)
            pd.Series(sel).to_csv(os.path.join(args.out, f"selected_{sname}_fold{f}.csv"), index=False, header=False)
            for name in args.models:
                if (sname, f, name) in done:
                    continue
                t0 = time.time()
                mdl = make_model(name, cfg, args.seed + f).fit(Xtr, t[tr], e[tr])
                r = evaluate(mdl, Xtr, t[tr], e[tr], Xte, t[te], e[te], HORIZONS)
                r.update({"set": sname, "fold": f, "model": name, "n_features": len(feat), "n_train": len(tr),
                          "n_test": len(te), "seconds": round(time.time() - t0, 1)})
                rows.append(r)
                pd.DataFrame(rows).to_csv(res_path, index=False)
                print(f"[{sname}] fold {f + 1}/{args.folds} {name:12s} C={r['c_harrell']:.3f} "
                      f"AUC5y={r.get('auc_5y', np.nan):.3f} BS3y={r.get('brier_3y', np.nan):.4f} ({r['seconds']}s)",
                      flush=True)
                if name == "draen" and args.importance and si == 0:
                    A = mdl.grad_x_input(Xte)
                    imp_store[f] = (feat, A, df.index[te].values)
                    np.savez_compressed(os.path.join(args.out, f"importance_fold{f}.npz"),
                                        features=np.array(feat), attribution=A, rows=df.index[te].values)
    json.dump({k: v for k, v in vars(args).items() if k != "func"}, open(os.path.join(args.out, "config.json"), "w"),
              indent=2, default=str)
    write_env(args.out)
    print("Done. Run:  python draen_pipeline.py report --real_dir", args.out)


# ============================================================================================
# 8. Simulation study
# ============================================================================================
SCENARIOS = {1: dict(p=20, pct=0.8, bs=(0.5, 0.1), bn=(0.0, 0.01), nl="full"),
             2: dict(p=50, pct=0.5, bs=(0.3, 0.08), bn=(0.0, 0.02), nl="full"),
             3: dict(p=100, pct=0.3, bs=(0.15, 0.05), bn=(0.0, 0.01), nl="mild")}


def gen_sim(sc, seed, n=200, cens=0.5):
    P = SCENARIOS[sc]
    rng = np.random.default_rng(seed)
    p = P["p"]
    Sig = np.exp(-np.abs(np.arange(p)[:, None] - np.arange(p)[None, :]))
    Z = rng.multivariate_normal(np.zeros(p), Sig, size=n)
    ns = int(p * P["pct"])
    beta = np.concatenate([rng.normal(*P["bs"], ns), rng.normal(*P["bn"], p - ns)])
    rng.shuffle(beta)
    eta = Z @ beta
    if P["nl"] == "full":
        eta = eta + 0.15 * Z[:, 0] ** 2 + 0.1 * Z[:, 1] ** 2 + 0.2 * Z[:, 2] * Z[:, 3] + 0.1 * (Z[:, 4] < -0.5)
    else:
        eta = eta + 0.1 * Z[:, 0] ** 2 + 0.05 * Z[:, 1] ** 2 + 0.05 * Z[:, 2] * Z[:, 3]
    U = rng.uniform(size=n)
    T = 0.1 * (-np.log(U) / np.exp(eta)) ** 0.5                  # Weibull(shape 2, scale 0.1) baseline hazard
    f = lambda s: np.mean(1 - np.exp(-T / s)) - cens             # exponential censoring, mean s, target rate
    s = brentq(f, 1e-6 * T.mean(), 1e6 * T.mean())
    C = rng.exponential(s, size=n)
    return Z, np.minimum(T, C), (T <= C).astype(int), eta


def run_sim(args):
    os.makedirs(args.out, exist_ok=True)
    res_path = os.path.join(args.out, "sim_results.csv")
    rows = pd.read_csv(res_path).to_dict("records") if os.path.exists(res_path) else []
    done = {(r["scenario"], r["rep"], r["model"]) for r in rows}
    cfg = args
    cfg.epochs, cfg.patience = args.sim_epochs, args.sim_patience
    for sc in args.scenarios:
        for rep in range(1, args.sim_reps + 1):
            Z, t, e, eta = gen_sim(sc, args.seed + sc * 1000 + rep)
            tr, te = train_test_split(np.arange(len(t)), test_size=0.3, stratify=e, random_state=args.seed + rep)
            cfg.feature_names = [f"z{i}" for i in range(Z.shape[1])]
            if ("oracle", rep) and (sc, rep, "oracle") not in done:
                rows.append({"scenario": sc, "rep": rep, "model": "oracle", "c_train": np.nan,
                             "c_test": concordance_index_censored(e[te].astype(bool), t[te], eta[te])[0]})
            for name in args.sim_models:
                if (sc, rep, name) in done:
                    continue
                m = make_model(name, cfg, args.seed + rep).fit(Z[tr], t[tr], e[tr])
                rows.append({"scenario": sc, "rep": rep, "model": name,
                             "c_train": concordance_index_censored(e[tr].astype(bool), t[tr], m.risk(Z[tr]))[0],
                             "c_test": concordance_index_censored(e[te].astype(bool), t[te], m.risk(Z[te]))[0],
                             "censoring": 1 - e.mean()})
            pd.DataFrame(rows).to_csv(res_path, index=False)
            print(f"scenario {sc} replicate {rep}/{args.sim_reps} done", flush=True)
    write_env(args.out)


# ============================================================================================
# 9. Reporting: tables, paired tests, figures
# ============================================================================================
def ms(x):
    return f"{100 * np.nanmean(x):.2f} ({100 * np.nanstd(x, ddof=1):.2f})"


def paired_tests(df, metric, ref="draen", n_te_over_n_tr=1 / 9):
    out = []
    for (s,), g in df.groupby(["set"]):
        piv = g.pivot(index="fold", columns="model", values=metric)
        if ref not in piv:
            continue
        for other in [c for c in piv.columns if c != ref]:
            d = (piv[ref] - piv[other]).dropna().values
            if len(d) < 3:
                continue
            k = len(d)
            tstat = d.mean() / np.sqrt((1 / k + n_te_over_n_tr) * d.var(ddof=1) + 1e-18)   # Nadeau-Bengio
            p_nb = 2 * stats.t.sf(abs(tstat), k - 1)
            try:
                p_w = stats.wilcoxon(d).pvalue
            except Exception:
                p_w = np.nan
            out.append({"set": s, "metric": metric, "ref": ref, "other": other, "mean_diff": d.mean(),
                        "p_corrected_t": p_nb, "p_wilcoxon": p_w, "folds_ref_better": int((d > 0).sum()), "k": k})
    o = pd.DataFrame(out)
    if len(o):                                                                # Holm correction within metric
        order = o["p_corrected_t"].rank(method="first").astype(int) - 1
        m = len(o)
        o["p_holm"] = np.minimum(1, o["p_corrected_t"] * (m - order))
    return o


def report(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if args.real_dir and os.path.exists(os.path.join(args.real_dir, "fold_results.csv")):
        d = pd.read_csv(os.path.join(args.real_dir, "fold_results.csv"))
        names = {"ridge": "Ridge-Cox", "lasso": "Lasso-Cox", "rsf": "RSF", "draen": "DRAEN-SA", "mlp": "MLP-Cox",
                 "clinical": "Clinical-only Cox", "draen_noattn": "DRAEN-SA w/o attention",
                 "draen_nores": "DRAEN-SA w/o residual", "draen_single": "DRAEN-SA single network"}
        # Table: C-index
        lines = []
        for s in d["set"].unique():
            for m in d["model"].unique():
                g = d[(d.set == s) & (d.model == m)]
                lines.append(f"{names.get(m, m)} & {s} & {ms(g.c_train)} & {ms(g.c_harrell)} & {ms(g.c_uno)} \\\\")
        open(os.path.join(args.real_dir, "table_cindex.tex"), "w").write("\n".join(lines))
        # Table: time-dependent
        first = d["set"].unique()[0]
        tl = []
        for m in d["model"].unique():
            g = d[(d.set == first) & (d.model == m)]
            auc = [100 * g[f"auc_{y}y"].mean() for y in HORIZONS]
            bs = [g[f"brier_{y}y"].mean() for y in HORIZONS]
            bs0 = [g[f"brier_null_{y}y"].mean() for y in HORIZONS]
            tl.append(f"{names.get(m, m)} & AUC & " + " & ".join(f"{a:.2f}" for a in auc) + f" & {np.mean(auc):.2f} \\\\")
            tl.append(f" & Brier & " + " & ".join(f"{b:.4f}" for b in bs) + f" & {np.mean(bs):.3f} \\\\")
            tl.append(f" & Brier skill vs KM & " + " & ".join(f"{100 * (1 - b / z):.1f}\\%" for b, z in zip(bs, bs0))
                      + f" & {100 * (1 - np.mean(bs) / np.mean(bs0)):.1f}\\% \\\\")
        open(os.path.join(args.real_dir, "table_timedep.tex"), "w").write("\n".join(tl))
        # paired tests
        pt = pd.concat([paired_tests(d, "c_harrell")] + [paired_tests(d, f"auc_{y}y") for y in (1, 3, 5)] +
                       [paired_tests(d, f"brier_{y}y") for y in (1, 3, 5)], ignore_index=True)
        pt.to_csv(os.path.join(args.real_dir, "paired_tests.csv"), index=False)
        d.groupby(["set", "model"]).agg(["mean", "std"]).to_csv(os.path.join(args.real_dir, "summary_mean_sd.csv"))
        # performance figure
        cols = {"ridge": "#4C78A8", "lasso": "#F58518", "rsf": "#54A24B", "draen": "#7B3FA0"}
        fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.7))
        for m, c in cols.items():
            g = d[(d.set == first) & (d.model == m)]
            if g.empty:
                continue
            ax[0].errorbar(HORIZONS, [100 * g[f"auc_{y}y"].mean() for y in HORIZONS],
                           [100 * g[f"auc_{y}y"].std() for y in HORIZONS], color=c, marker="o", capsize=2, label=names[m])
            ax[1].errorbar(HORIZONS, [g[f"brier_{y}y"].mean() for y in HORIZONS],
                           [g[f"brier_{y}y"].std() for y in HORIZONS], color=c, marker="o", capsize=2)
        w = 0.2
        for i, (m, c) in enumerate(cols.items()):
            v = [d[(d.set == s) & (d.model == m)].c_harrell for s in d["set"].unique()]
            ax[2].bar(np.arange(len(v)) + (i - 1.5) * w, [100 * x.mean() for x in v], w, yerr=[100 * x.std() for x in v],
                      color=c, capsize=2)
        ax[0].set_ylabel("Time-dependent AUC (%)"); ax[1].set_ylabel("Brier score"); ax[2].set_ylabel("C-index (%)")
        ax[0].set_xlabel("Years"); ax[1].set_xlabel("Years"); ax[2].set_xticks(range(len(d["set"].unique())))
        ax[2].set_xticklabels(d["set"].unique()); ax[0].legend(frameon=False, fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(args.real_dir, "fig_performance.pdf"))
        # importance (gradient x input) aggregated across folds
        files = sorted(f for f in os.listdir(args.real_dir) if f.startswith("importance_fold"))
        if files:
            acc, signed, freq = {}, {}, {}
            for f in files:
                z = np.load(os.path.join(args.real_dir, f), allow_pickle=True)
                feat, A = list(z["features"]), z["attribution"]
                for j, nme in enumerate(feat):
                    acc.setdefault(nme, []).append(np.mean(np.abs(A[:, j])))
                    freq[nme] = freq.get(nme, 0) + 1
            imp = pd.DataFrame({"feature": list(acc), "importance": [np.mean(v) for v in acc.values()],
                                "folds_selected": [freq[k] for k in acc]}).sort_values("importance", ascending=False)
            imp.to_csv(os.path.join(args.real_dir, "feature_importance_all.csv"), index=False)
            top = imp.head(20)
            top.to_csv(os.path.join(args.real_dir, "top20_features.csv"), index=False)
            fig, ax = plt.subplots(figsize=(3.6, 3.9))
            ax.barh(top.feature[::-1], top.importance[::-1] * 1e4, color="#8FB4D9")
            ax.set_xlabel("Mean |gradient x input| (x1e-4)"); fig.tight_layout()
            fig.savefig(os.path.join(args.real_dir, "fig_importance.pdf"))
    if args.sim_dir and os.path.exists(os.path.join(args.sim_dir, "sim_results.csv")):
        s = pd.read_csv(os.path.join(args.sim_dir, "sim_results.csv"))
        lines = []
        for m in s.model.unique():
            cells = []
            for sc in sorted(s.scenario.unique()):
                g = s[(s.scenario == sc) & (s.model == m)]
                cells.append(f"{ms(g.c_train)} / {ms(g.c_test)}" if g.c_train.notna().any() else f"-- / {ms(g.c_test)}")
            lines.append(f"{m} & " + " & ".join(cells) + " \\\\")
        open(os.path.join(args.sim_dir, "table_sim.tex"), "w").write("\n".join(lines))
        print(open(os.path.join(args.sim_dir, "table_sim.tex")).read())
    print("Report written.")


def write_env(out):
    import sksurv, sklearn
    with open(os.path.join(out, "environment.txt"), "w") as fh:
        fh.write(f"python {sys.version}\nplatform {platform.platform()}\nnumpy {np.__version__}\n"
                 f"pandas {pd.__version__}\nscikit-learn {sklearn.__version__}\nscikit-survival {sksurv.__version__}\n"
                 f"tensorflow {tf.__version__}\nGPUs {tf.config.list_physical_devices('GPU')}\n")


# ============================================================================================
# 10. Command line
# ============================================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", default="results")
    common.add_argument("--seed", type=int, default=42)
    common.add_argument("--latent", type=int, default=1024)
    common.add_argument("--lr", type=float, default=2e-5)
    common.add_argument("--wd", type=float, default=1e-5)
    common.add_argument("--batch", type=int, default=128)
    common.add_argument("--epochs", type=int, default=1200)
    common.add_argument("--patience", type=int, default=100)
    common.add_argument("--val_frac", type=float, default=0.15)
    common.add_argument("--n_ensemble", type=int, default=5)
    common.add_argument("--rsf_trees", type=int, default=500)
    sr = sub.add_parser("real", parents=[common])
    sr.add_argument("--data", nargs="+", required=True)
    sr.add_argument("--id_col", default=None)
    sr.add_argument("--screen", choices=["infold", "none"], default="infold")
    sr.add_argument("--n_genes", type=int, nargs="+", default=[154, 258, 441])
    sr.add_argument("--folds", type=int, default=10)
    sr.add_argument("--models", nargs="+", default=["ridge", "lasso", "rsf", "draen"],
                    help="ridge lasso rsf draen | optional: clinical mlp draen_noattn draen_nores draen_single")
    sr.add_argument("--importance", action="store_true", default=True)
    sr.add_argument("--smoke", action="store_true", help="tiny settings for a quick functional test")
    ss = sub.add_parser("sim", parents=[common])
    ss.add_argument("--scenarios", type=int, nargs="+", default=[1, 2, 3])
    ss.add_argument("--sim_reps", type=int, default=200)
    ss.add_argument("--sim_models", nargs="+", default=["ridge", "rsf", "draen"])
    ss.add_argument("--sim_epochs", type=int, default=300)
    ss.add_argument("--sim_patience", type=int, default=30)
    ss.add_argument("--smoke", action="store_true")
    sp = sub.add_parser("report")
    sp.add_argument("--real_dir", default=None)
    sp.add_argument("--sim_dir", default=None)
    args = ap.parse_args()
    if getattr(args, "smoke", False):
        args.epochs, args.patience, args.rsf_trees, args.n_ensemble, args.latent = 4, 2, 20, 2, 64
        if args.cmd == "real":
            args.folds = 3
        else:
            args.sim_reps, args.sim_epochs, args.sim_patience = 2, 4, 2
    if args.cmd == "real":
        run_real(args)
    elif args.cmd == "sim":
        run_sim(args)
    else:
        report(args)


if __name__ == "__main__":
    main()
