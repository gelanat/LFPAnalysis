"""Latent brain-state models (HMM) over multichannel LFP envelope time-courses.

Upgrades the project's deferred transient-state HMM (the ``GaussianMixture`` stopgap in
``directionality_transient_states.py``) to a proper Gaussian **hidden Markov model** fit at the
time-sample level: given a per-subject emission matrix ``X`` (n_samples x n_channels, the decimated
band-power / HFA envelope) and ``lengths`` (per-trial sample counts, so state transitions never cross
trial boundaries), :func:`fit_gaussian_hmm` discovers discrete latent states and returns the state
posterior and Viterbi path. :func:`state_occupancy` and :func:`state_metrics` reduce those to the
per-trial statistics the driver relates to behaviour.

The fit is **label-blind** (unsupervised, no behaviour) so the driver can permute only the relation
to behaviour without double-dipping. Sample-level fitting gives thousands of observations per subject
— the well-powered regime, unlike the ~60 trial-level observations that constrain the decoders.

Backends: ``"hmm"`` uses ``hmmlearn.GaussianHMM`` (a real temporal model; primary). ``"gmm"`` is a
zero-dependency proxy — a ``sklearn`` Gaussian mixture whose components are treated as states, with an
empirical transition matrix estimated from the hard-assigned sequence (no temporal smoothing; the same
coarse proxy the project used before, kept as a fallback). ``"auto"`` prefers ``hmm`` if importable.
"""
from __future__ import annotations

import numpy as np

__all__ = ["fit_gaussian_hmm", "state_occupancy", "state_metrics"]


def _seq_starts(lengths):
    """Row index at which each variable-length sequence begins."""
    return np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(int)


def _empirical_transmat(states, lengths, n_states):
    """Row-normalised transition-count matrix from hard states, within sequences only."""
    T = np.zeros((n_states, n_states), dtype=float)
    off = 0
    for L in lengths:
        s = states[off:off + L]
        for a, b in zip(s[:-1], s[1:]):
            T[a, b] += 1.0
        off += L
    rs = T.sum(axis=1, keepdims=True)
    return np.divide(T, rs, out=np.full_like(T, 1.0 / n_states), where=rs > 0)


def _resolve_backend(backend: str) -> str:
    if backend in ("hmm", "gmm"):
        return backend
    if backend != "auto":
        raise ValueError(f"backend must be 'auto'|'hmm'|'gmm', got {backend!r}")
    try:
        import hmmlearn  # noqa: F401
        return "hmm"
    except Exception:
        return "gmm"


def fit_gaussian_hmm(X, lengths, *, n_states: int = 4, cov: str = "diag", n_init: int = 5,
                     n_iter: int = 100, random_state: int = 0, backend: str = "auto") -> dict:
    """Fit a Gaussian HMM (or GMM proxy) to sequence data and return states + posterior.

    Parameters
    ----------
    X
        ``(n_samples, n_features)`` stacked emissions across all sequences (trials).
    lengths
        Per-sequence sample counts; ``sum(lengths) == n_samples``. Transitions are confined
        within a sequence (trial), never across the concatenation boundary.
    n_states, cov
        Number of latent states and covariance type (``"diag"`` default — robust at this
        channel count). ``n_init`` restarts keep the best log-likelihood; ``n_iter`` EM steps.

    Returns
    -------
    dict
        ``{backend, n_states, loglik, means, covars, transmat, startprob, posterior, states,
        n_samples, n_features}``. ``posterior`` is ``(n_samples, n_states)``; ``states`` the
        Viterbi (hmm) or MAP (gmm) hard path. Empty/degenerate input returns ``None``-filled fields.
    """
    X = np.asarray(X, dtype=float)
    lengths = [int(v) for v in lengths]
    if X.ndim != 2 or X.shape[0] == 0 or sum(lengths) != X.shape[0]:
        raise ValueError(f"X {X.shape} inconsistent with lengths sum {sum(lengths)}")
    n_samples, n_features = X.shape
    resolved = _resolve_backend(backend)

    if resolved == "hmm":
        from hmmlearn.hmm import GaussianHMM
        best = None
        for i in range(n_init):
            m = GaussianHMM(n_components=n_states, covariance_type=cov, n_iter=n_iter,
                            tol=1e-3, random_state=random_state + i, init_params="stmc")
            try:
                m.fit(X, lengths)
                ll = float(m.score(X, lengths))
            except Exception:
                continue
            if np.isfinite(ll) and (best is None or ll > best[0]):
                best = (ll, m)
        if best is not None:
            ll, m = best
            return dict(backend="hmm", n_states=n_states, loglik=ll, means=m.means_,
                        covars=m.covars_, transmat=m.transmat_, startprob=m.startprob_,
                        posterior=m.predict_proba(X, lengths), states=m.predict(X, lengths),
                        n_samples=n_samples, n_features=n_features)
        resolved = "gmm"  # hmm failed to fit -> proxy

    from sklearn.mixture import GaussianMixture
    g = GaussianMixture(n_components=n_states, covariance_type=cov, n_init=n_init,
                        random_state=random_state).fit(X)
    states = g.predict(X)
    starts = _seq_starts(lengths)
    return dict(backend="gmm", n_states=n_states, loglik=float(g.score(X) * n_samples),
                means=g.means_, covars=g.covariances_,
                transmat=_empirical_transmat(states, lengths, n_states),
                startprob=np.bincount(states[starts], minlength=n_states) / max(len(lengths), 1),
                posterior=g.predict_proba(X), states=states,
                n_samples=n_samples, n_features=n_features)


def state_occupancy(posterior, lengths) -> np.ndarray:
    """Per-trial soft occupancy — mean posterior over each trial's samples. ``(n_trials, n_states)``."""
    P = np.asarray(posterior, dtype=float)
    out = np.zeros((len(lengths), P.shape[1]), dtype=float)
    off = 0
    for i, L in enumerate(lengths):
        out[i] = P[off:off + L].mean(axis=0) if L > 0 else np.nan
        off += L
    return out


def state_metrics(states, lengths) -> dict:
    """Per-trial dynamics: ``n_transitions`` and ``mean_dwell`` (samples) per trial."""
    states = np.asarray(states)
    n_tr = len(lengths)
    n_trans = np.zeros(n_tr, dtype=float)
    mean_dwell = np.zeros(n_tr, dtype=float)
    off = 0
    for i, L in enumerate(lengths):
        s = states[off:off + L]
        if L > 1:
            switches = int(np.sum(s[1:] != s[:-1]))
            n_trans[i] = switches
            mean_dwell[i] = L / (switches + 1)
        else:
            n_trans[i] = 0.0
            mean_dwell[i] = float(L)
        off += L
    return dict(n_transitions=n_trans, mean_dwell=mean_dwell)
