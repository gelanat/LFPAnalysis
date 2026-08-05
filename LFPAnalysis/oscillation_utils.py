"""
eBOSC IN PROGRESS: 

Pulled largely from Julian Q. Kosciessa at
https://github.com/jkosciessa/eBOSC_py 

"""



import numpy as np
import pandas as pd
import numpy.matlib
import scipy.io as sio
from pathlib import Path
import statsmodels.api as sm
from scipy.stats.distributions import chi2
from mne_connectivity import phase_slope_index, seed_target_indices, spectral_connectivity_epochs, spectral_connectivity_time
import mne
from scipy.signal import hilbert
from mne.filter import next_fast_len
from tqdm import tqdm
from scipy.stats import zscore, circstd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from joblib import delayed, Parallel


# Helper functions 

def find_nearest_value(array, value):
    """Find nearest value and index of float in array
    Parameters:
    array : Array of values [1d array]
    value : Value of interest [float]
    Returns:
    array[idx] : Nearest value [1d float]
    idx : Nearest index [1d float]
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

def getTimeFromFTmat(fname, var_name='data'):
    """
    Get original timing from FieldTrip structure
    Solution based on https://github.com/mne-tools/mne-python/issues/2476
    """
    # load Matlab/Fieldtrip data
    mat = sio.loadmat(fname, squeeze_me=True, struct_as_record=False)
    ft_data = mat[var_name]
    # convert to mne
    n_trial = len(ft_data.trial)
    n_chans, n_time = ft_data.trial[0].shape
    #data = np.zeros((n_trial, n_chans, n_time))
    time = np.zeros((n_trial, n_time))
    for trial in range(n_trial):
        # data[trial, :, :] = ft_data.trial[trial]
        # Note that this indexes time_orig in the adapted structure
        time[trial, :] = ft_data.time_orig[trial]
    return time

def get_project_root() -> Path:
    return Path(__file__)
    
def swap_time_blocks(data, random_state=None):

    """Compute surrogates by swapping time blocks.
    This function cuts the timeseries at a random time point. Then, both time
    blocks are swapped.
    Parameters
    ----------
    data : array_like
        Array of shape (n_chan, ..., n_times).
    random_state : int | None
        Fix the random state of the machine for reproducible results.
    Returns
    -------
    surr : array_like
        Swapped timeseries to use to compute the distribution of
        permutations
    References
    ----------
    Source: Bahramisharif et al. 2013 
    Justification: https://www.sciencedirect.com/science/article/pii/S0959438814001640
    """
    
    if random_state is None:
        random_state = int(np.random.randint(0, 10000, size=1))
    rnd = np.random.RandomState(random_state)
    
    # get the minimum / maximum shift
    min_shift, max_shift = 1, None
    if not isinstance(max_shift, (int, float)):
        max_shift = data.shape[-1]
    # random cutting point along time axis
    cut_at = rnd.randint(min_shift, max_shift, (1,))
    # split amplitude across time into two parts
    surr = np.array_split(data, cut_at, axis=-1)
    # revered elements
    surr.reverse()

    return np.concatenate(surr, axis=-1)


# ---------------------------------------------------------------------------
# Phase transfer entropy (Lobier et al. 2014 NeuroImage; Hillebrand et al. 2016
# PNAS dPTE normalisation). Phase-based, model-free directed connectivity.
#
# Aligned with the reference implementation used by Gattas et al. 2023
# Nat Commun (github.com/Yassa-TNL/theta_physiology, TransferEntropy/
# B001_TransferEntropy.m): instantaneous Hilbert phase, Scott's-rule circular
# binning, fixed prediction delay, log2 (bits) Shannon entropies, and the
# PTE = H(Y_t,Y_{t+d}) + H(Y_t,X_t) - H(Y_t) - H(Y_{t+d},Y_t,X_t) decomposition.
#
# Robustness choices that tighten the *estimator* without changing the
# *definition* (callers can opt out): closed-right bin edges (the validated
# pac_utils._tort_mi idiom, fixes the reference's linspace off-by-one);
# boundary-safe (t, t+delta) pairing that never straddles trial boundaries;
# all four entropy terms marginalised from one joint count tensor so they share
# identical bin edges; and an effective-transfer-entropy surrogate for the
# finite-sample positive bias (Marschinski & Kantz 2002).
# ---------------------------------------------------------------------------

PTE_MIN_BINS = 4
PTE_MAX_BINS = 32
PTE_MIN_PER_CELL = 5


def scott_n_bins(phase, min_bins=PTE_MIN_BINS, max_bins=PTE_MAX_BINS):
    """Scott's-rule bin count for circular phase, as in the reference.

    ``n_bins = round(2*pi / (3.5 * circ_std(phase) / N**(1/3)))`` clamped to
    ``[min_bins, max_bins]``. Degenerate (constant / too-short) phase falls back
    to ``min_bins``.
    """
    phase = np.asarray(phase, dtype=float).ravel()
    n = phase.size
    if n < 2:
        return int(min_bins)
    sd = float(circstd(phase, high=np.pi, low=-np.pi))
    bw = 3.5 * sd / (n ** (1.0 / 3.0))
    if not np.isfinite(bw) or bw <= 0:
        return int(min_bins)
    nb = int(round(2.0 * np.pi / bw))
    return int(np.clip(nb, min_bins, max_bins))


def choose_pte_bins(phase_seed, phase_target, delay,
                    min_per_cell=PTE_MIN_PER_CELL,
                    min_bins=PTE_MIN_BINS, max_bins=PTE_MAX_BINS):
    """Per-channel Scott bin counts, shrunk so the 3-D joint histogram is sampled.

    Returns ``(n_bins_seed, n_bins_target, n_triples)`` or ``None`` if the window
    is too short to form a single (t, t+delay) pair. The shrink keeps
    ``min_per_cell * receiver_bins**2 * sender_bins <= n_triples`` for *both*
    directions (each PTE direction squares its receiver's bin count).
    """
    ps = np.atleast_2d(np.asarray(phase_seed, dtype=float))
    pt = np.atleast_2d(np.asarray(phase_target, dtype=float))
    n_win = pt.shape[1]
    delay = int(delay)
    if n_win <= delay:
        return None
    n = ps.shape[0] * (n_win - delay)
    nb_seed = scott_n_bins(ps.reshape(-1), min_bins, max_bins)
    nb_target = scott_n_bins(pt.reshape(-1), min_bins, max_bins)
    while max(nb_seed, nb_target) > min_bins and (
        min_per_cell * nb_target * nb_target * nb_seed > n
        or min_per_cell * nb_seed * nb_seed * nb_target > n
    ):
        if nb_seed >= nb_target:
            nb_seed -= 1
        else:
            nb_target -= 1
    return int(nb_seed), int(nb_target), int(n)


def _digitize_phase(phase, n_bins):
    """Map phase in (-pi, pi] to integer codes in [0, n_bins-1] (closed right)."""
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    return np.clip(np.digitize(phase, edges) - 1, 0, n_bins - 1)


def _shannon_entropy(p, base=2):
    """Shannon entropy of a probability tensor; empty bins contribute 0."""
    nz = p[p > 0]
    if nz.size == 0:
        return 0.0
    logf = np.log2 if base == 2 else np.log
    return float(-np.sum(nz * logf(nz)))


def _phase_te_triples(phase_x, phase_y, delay):
    """Boundary-safe (Y_t, Y_{t+delay}, X_t) samples pooled across trials.

    Slices within each trial *before* flattening, so no (t, t+delay) pair ever
    straddles a trial boundary. ``phase_x`` (sender) and ``phase_y`` (receiver)
    are ``(n_trials, n_win)``. Returns ``None`` if ``n_win <= delay``.
    """
    n_win = phase_y.shape[1]
    if n_win <= delay:
        return None
    yt = phase_y[:, : n_win - delay].reshape(-1)   # Y_t
    yd = phase_y[:, delay:].reshape(-1)            # Y_{t+delay}
    xt = phase_x[:, : n_win - delay].reshape(-1)   # X_t (aligned to Y_t)
    return yt, yd, xt


def phase_transfer_entropy(phase_x, phase_y, delay,
                          n_bins_x=None, n_bins_y=None, base=2):
    """Phase transfer entropy from X to Y, in bits (base 2) by default.

    ``PTE_{X->Y} = H(Y_t, Y_{t+d}) + H(Y_t, X_t) - H(Y_t) - H(Y_{t+d}, Y_t, X_t)``
    ``           = H(Y_{t+d} | Y_t) - H(Y_{t+d} | Y_t, X_t)``.

    Parameters
    ----------
    phase_x, phase_y : array
        Instantaneous phase in (-pi, pi], shape ``(n_trials, n_win)`` (1-D is
        promoted to a single trial). ``phase_x`` is the sender, ``phase_y`` the
        receiver whose future is predicted.
    delay : int
        Prediction delay in samples (the reference default is ``round(0.1*fs)``).
    n_bins_x, n_bins_y : int | None
        Sender / receiver bin counts. ``None`` derives them per channel via
        Scott's rule (callers usually pass counts from :func:`choose_pte_bins`
        so both directions of a pair share a fixed per-channel resolution).

    Returns ``nan`` for non-finite phase or windows shorter than ``delay``.
    """
    phase_x = np.atleast_2d(np.asarray(phase_x, dtype=float))
    phase_y = np.atleast_2d(np.asarray(phase_y, dtype=float))
    if not (np.all(np.isfinite(phase_x)) and np.all(np.isfinite(phase_y))):
        return float("nan")
    triples = _phase_te_triples(phase_x, phase_y, int(delay))
    if triples is None:
        return float("nan")
    yt, yd, xt = triples
    n = yt.size
    if n < 2:
        return float("nan")
    nby = int(n_bins_y) if n_bins_y is not None else scott_n_bins(np.concatenate([yt, yd]))
    nbx = int(n_bins_x) if n_bins_x is not None else scott_n_bins(xt)
    iy = _digitize_phase(yt, nby)
    iyd = _digitize_phase(yd, nby)
    ix = _digitize_phase(xt, nbx)
    codes = np.ravel_multi_index((iyd, iy, ix), (nby, nby, nbx))
    counts = np.bincount(codes, minlength=nby * nby * nbx).astype(float)
    p3 = counts.reshape(nby, nby, nbx) / n            # P(Y_{t+d}, Y_t, X_t)
    h_yd_y = _shannon_entropy(p3.sum(axis=2), base)   # H(Y_{t+d}, Y_t)
    h_y_x = _shannon_entropy(p3.sum(axis=0), base)    # H(Y_t, X_t)
    h_y = _shannon_entropy(p3.sum(axis=(0, 2)), base)  # H(Y_t)
    h_yd_y_x = _shannon_entropy(p3, base)             # H(Y_{t+d}, Y_t, X_t)
    return float(h_yd_y + h_y_x - h_y - h_yd_y_x)


def pte_surrogate_dist(phase_x, phase_y, delay, n_bins_x=None, n_bins_y=None,
                      kind="circshift", n_surr=200, rng=None, base=2):
    """Surrogate ``PTE_{X->Y}`` distribution (sender disrupted, receiver intact).

    The *sender* (``phase_x``) is disrupted per trial while the receiver stays
    intact, so only the directed term ``H(Y_{t+d}|Y_t,X_t)`` is nulled. The mean
    is the finite-sample bias floor for that direction (effective transfer
    entropy, Marschinski & Kantz 2002); the spread gives a per-pair null.

    - ``"circshift"`` (default): per-trial circular time-shift of the sender,
      preserving its autocorrelation/marginal — the conservative effective-TE
      null.
    - ``"shuffle"``: per-trial random permutation of the sender timepoints — the
      Gattas et al. 2023 reference null (destroys sender autocorrelation too).

    Returns an ``(n_surr,)`` array (``nan`` if the window is too short).
    """
    phase_x = np.atleast_2d(np.asarray(phase_x, dtype=float))
    phase_y = np.atleast_2d(np.asarray(phase_y, dtype=float))
    if rng is None:
        rng = np.random.default_rng()
    n_trials, n_win = phase_x.shape
    if n_win <= int(delay) or int(n_surr) <= 0:
        return np.full(max(int(n_surr), 1), np.nan)
    vals = np.empty(int(n_surr), dtype=float)
    for s in range(int(n_surr)):
        surr = np.empty_like(phase_x)
        for k in range(n_trials):
            if kind == "circshift":
                surr[k] = np.roll(phase_x[k], int(rng.integers(1, n_win)))
            elif kind == "shuffle":
                surr[k] = phase_x[k, rng.permutation(n_win)]
            else:
                raise ValueError(f"unknown surrogate kind {kind!r}; use 'circshift' or 'shuffle'")
        vals[s] = phase_transfer_entropy(surr, phase_y, delay, n_bins_x, n_bins_y, base)
    return vals


def pte_surrogate_mean(phase_x, phase_y, delay, n_bins_x=None, n_bins_y=None,
                      kind="circshift", n_surr=200, rng=None, base=2):
    """Mean of :func:`pte_surrogate_dist` — the directional bias floor."""
    return float(np.nanmean(
        pte_surrogate_dist(phase_x, phase_y, delay, n_bins_x, n_bins_y,
                           kind=kind, n_surr=n_surr, rng=rng, base=base)
    ))


def make_seed_target_df(elec_df, epochs, source_roi, target_roi):
    
    """
    Create arrays of indices for mapping electrodes for connectivity analyses. Use the Epoch
    ch_names list itself to find the index of the electrode within the mne object 
    
    """
    
    seed_target_df = pd.DataFrame(columns=['seed', 'target'], index=['l', 'r'])

    for hemi in ['l', 'r']:
        source_ix = elec_df[(elec_df.hemisphere.str.lower()==hemi) & (elec_df.SNT_region==source_roi)].label.values
        target_ix = elec_df[(elec_df.hemisphere.str.lower()==hemi) & (elec_df.SNT_region==target_roi)].label.values
        
        if (len(source_ix) == 0) | (len(target_ix)==0):
            seed_target_df.at[hemi, 'seed'] = []
            seed_target_df.at[hemi, 'target'] = []
        else:   
            # .at + list() avoid chained-assignment squeezing a single-channel
            # region/hemi (1-element pick) into a 0-d scalar, which then breaks
            # the len(d) filter below (TypeError: len() of unsized object).
            seed_target_df.at[hemi, 'seed'] = list(mne.pick_channels(epochs.ch_names, source_ix))
            seed_target_df.at[hemi, 'target'] = list(mne.pick_channels(epochs.ch_names, target_ix))

    seed_target_df = seed_target_df[
                (seed_target_df['seed'].map(lambda d: len(d) > 0)) & (seed_target_df['target'].map(lambda d: len(d) > 0))]

    
    return seed_target_df
    

def amp_amp_coupling(mne_data, seed_to_target, freqs0, freqs1=None):
    """
    Compute the correlation between the amplitude envelope of two signals. 
    Can be within-frequency or between-frequency coupling.

    Parameters
    ----------
    mne_data : epochs object
        MNE epochs object containing the data to be analyzed.
    seed_to_target : list of tuples
        List of tuples containing the indices of the seed and target electrodes.
    freqs0 : list or tuple
        Frequency range for the first signal.
    freqs1 : list or tuple
        Frequency range for the second signal. If None, assume within-frequency coupling.

    Note: inspired by MNE's pairwise orthogonal envelope connectivity metric but altered for iEEG data 
    """

    nevents = mne_data._data.shape[0]
    ntimes = mne_data._data.shape[-1] 
    nfft = next_fast_len(ntimes)  
    # npairs = len(seed_to_target[0])
    nsource = len(np.unique(seed_to_target[0]))
    ntarget = len(np.unique(seed_to_target[1]))

    if freqs1 is None: 
        # Assume within-frequency coupling
        freqs1 = freqs0
    
    signal0 = mne_data._data[:, np.unique(seed_to_target[0]), :]
    signal1 = mne_data._data[:, np.unique(seed_to_target[1]), :]

    signal0_filt = mne.filter.filter_data(signal0, 
                     mne_data.info['sfreq'], 
                     l_freq=freqs0[0], 
                     h_freq=freqs0[1])
    
    signal1_filt = mne.filter.filter_data(signal1,
                        mne_data.info['sfreq'],
                        l_freq=freqs0[0],
                        h_freq=freqs0[1])
    
    corrs = []

    for ei in range(nevents):
        signal0_hilbert = hilbert(signal0_filt[ei, :, :], N=nfft, axis=-1)[..., :ntimes]
        signal0_amp = np.abs(signal0_hilbert)
        signal1_hilbert = hilbert(signal1_filt[ei, :, :], N=nfft, axis=-1)[..., :ntimes]
        signal1_amp = np.abs(signal1_hilbert)

        # Square and log the analytical amplitude: https://www.nature.com/articles/nn.3101#Sec15
        signal0_amp *= signal0_amp
        np.log(signal0, out=signal0)
        signal1_amp *= signal1_amp
        np.log(signal1, out=signal1)

        # subtract mean 
        signal0_amp_nomean = signal0_amp - np.mean(signal0_amp, axis=-1, keepdims=True)
        signal1_amp_nomean = signal1_amp - np.mean(signal1_amp, axis=-1, keepdims=True)

        # compute variances using linalg.norm (square, sum, sqrt) since mean=0
        signal0_amp_std = np.linalg.norm(signal0_amp_nomean, axis=-1)
        signal0_amp_std[signal0_amp_std == 0] = 1
        signal1_amp_std = np.linalg.norm(signal1_amp_nomean, axis=-1)
        signal1_amp_std[signal1_amp_std == 0] = 1

        # compute correlation for each source to all targets
        corr_mat = []
        for source_ix in range(nsource):
            for target_ix in range(ntarget): 
                signal0_amp_elec = np.squeeze(signal0_amp_nomean[source_ix, :])
                signal1_amp_elec = np.squeeze(signal1_amp_nomean[target_ix, :])
                corr = np.sum(signal1_amp_elec * signal0_amp_elec)
                corr /= signal0_amp_std[source_ix]
                corr /= signal1_amp_std[target_ix]
                corr_mat.append(corr)
                
        corrs.append(corr_mat)

    pairwise_connectivity = np.stack(corrs) # size is (nevents, ntarget, nsource)
    # reshape so all pairs are in order:


    return pairwise_connectivity

def compute_gc_tr(mne_data=None, 
                band=None,
                indices=None, 
                freqs=None, 
                n_cycles=None,
                rank=None, 
                gc_n_lags=15, 
                buf_ms=1000, 
                avg_over_dim='time'): 
    """
    Following https://mne.tools/mne-connectivity/stable/auto_examples/granger_causality.html#sphx-glr-auto-examples-granger-causality-py
    """

    indices_ab = (np.array([np.unique(indices[0]).tolist()]), np.array([np.unique(indices[1]).tolist()]))  # A => B
    indices_ba = (np.array([np.unique(indices[1]).tolist()]), np.array([np.unique(indices[0]).tolist()]))  # B => A
    
    if avg_over_dim == 'epochs':
        # compute Granger causality
        gc_ab = spectral_connectivity_epochs(
            mne_data,
            sfreq = mne_data.info['sfreq'],
            method=["gc"],
            indices=indices_ab,
            fmin=band[0], fmax=band[1],
            rank=rank,
            gc_n_lags=gc_n_lags,
            verbose='ERROR') 
        # A => B
        gc_ba = spectral_connectivity_epochs(
            mne_data,
            sfreq = mne_data.info['sfreq'],
            method=["gc"],
            indices=indices_ba,
            fmin=band[0], fmax=band[1],
            rank=rank,
            gc_n_lags=gc_n_lags,
            verbose='ERROR')  
        # B => A
                    
        # compute GC on time-reversed signals
        gc_tr_ab = spectral_connectivity_epochs(
            mne_data,
            sfreq = mne_data.info['sfreq'],        
            method=["gc_tr"],
            indices=indices_ab,
            fmin=band[0], fmax=band[1],
            rank=rank,
            gc_n_lags=gc_n_lags,
            verbose='ERROR')  
        # TR[A => B]

        gc_tr_ba = spectral_connectivity_epochs(
            mne_data,
            sfreq = mne_data.info['sfreq'],                
            method=["gc_tr"],
            indices=indices_ba,
            fmin=band[0], fmax=band[1],
            rank=rank,
            gc_n_lags=gc_n_lags,
            verbose='ERROR')  
        # TR[B => A]
    elif avg_over_dim =='time':
        # compute Granger causality
        gc_ab = spectral_connectivity_time(
            mne_data,
            sfreq = mne_data.info['sfreq'],
            method=["gc"],
            indices=indices_ab,
            fmin=band[0], fmax=band[1],
            freqs=freqs[(freqs>=band[0]) & (freqs<=band[1])],
            rank=rank,
            padding=(buf_ms / 1000), 
            n_cycles=n_cycles[(freqs>=band[0]) & (freqs<=band[1])],
            gc_n_lags=gc_n_lags,
            verbose='ERROR') 

        # A => B
        gc_ba = spectral_connectivity_time(
            mne_data,
            sfreq = mne_data.info['sfreq'],
            method=["gc"],
            indices=indices_ba,
            fmin=band[0], fmax=band[1],
            freqs=freqs[(freqs>=band[0]) & (freqs<=band[1])],
            rank=rank,
            padding=(buf_ms / 1000), 
            n_cycles=n_cycles[(freqs>=band[0]) & (freqs<=band[1])],
            gc_n_lags=gc_n_lags,
            verbose='ERROR')  
        # B => A
                    
        # compute GC on time-reversed signals
        gc_tr_ab = spectral_connectivity_time(
            mne_data,
            sfreq = mne_data.info['sfreq'],        
            method=["gc_tr"],
            indices=indices_ab,
            fmin=band[0], fmax=band[1],
            freqs=freqs[(freqs>=band[0]) & (freqs<=band[1])],
            rank=rank,
            padding=(buf_ms / 1000), 
            n_cycles=n_cycles[(freqs>=band[0]) & (freqs<=band[1])],
            gc_n_lags=gc_n_lags,
            verbose='ERROR')  
        # TR[A => B]

        gc_tr_ba = spectral_connectivity_time(
            mne_data,
            sfreq = mne_data.info['sfreq'],                
            method=["gc_tr"],
            indices=indices_ba,
            fmin=band[0], fmax=band[1],
            freqs=freqs[(freqs>=band[0]) & (freqs<=band[1])],
            rank=rank,
            padding=(buf_ms / 1000), 
            n_cycles=n_cycles[(freqs>=band[0]) & (freqs<=band[1])],
            gc_n_lags=gc_n_lags,
            verbose='ERROR')  
        # TR[B => A]

    net_gc = gc_ab.get_data() - gc_ba.get_data()  # [A => B] - [B => A]

    # compute net GC on time-reversed signals (TR[A => B] - TR[B => A])
    net_gc_tr = gc_tr_ab.get_data() - gc_tr_ba.get_data()

    # compute TRGC
    gc_tr = net_gc - net_gc_tr

    if avg_over_dim =='time':
        return gc_tr.mean(axis=-1)
    else:
        return np.squeeze(gc_tr)


def compute_psi_per_trial(mne_data, indices, band, freqs, n_cycles,
                          tmin=None, tmax=None, verbose='ERROR'):
    """Per-trial phase slope index over a band. Returns ``(n_epochs, n_pairs)``.

    ``phase_slope_index`` normally averages over epochs; this keeps the trial axis by slicing
    each single trial to ``[tmin, tmax)`` (optional), wrapping it as a 1-epoch ``EpochsArray``,
    running ``phase_slope_index`` (cwt_morlet) on it, and averaging the band/time bins per pair.
    Positive PSI = seed (``indices[0]``) leads target (``indices[1]``). ``freqs``/``n_cycles`` are
    the full cwt grid; the band is selected internally via ``fmin``/``fmax`` (matches the
    epoch-averaged ``compute_connectivity`` PSI path).
    """
    X = mne_data.get_data(copy=False)
    info = mne_data.info
    times = mne_data.times
    sfreq = info['sfreq']
    if (tmin is not None) or (tmax is not None):
        lo = -np.inf if tmin is None else tmin
        hi = np.inf if tmax is None else tmax
        tidx = (times >= lo) & (times < hi)
    else:
        tidx = np.ones(times.shape[0], dtype=bool)
    t0 = times[np.where(tidx)[0][0]]
    srcs, tgts = indices
    npair = len(srcs)
    out = np.full((X.shape[0], npair), np.nan, float)
    for ei in range(X.shape[0]):
        ep = mne.EpochsArray(X[ei:ei + 1, :, tidx], info, tmin=t0, verbose=verbose)
        conn = np.asarray(phase_slope_index(
            ep, indices=(srcs, tgts), sfreq=sfreq, mode='cwt_morlet',
            fmin=band[0], fmax=band[1], cwt_freqs=freqs, cwt_n_cycles=n_cycles,
            verbose=verbose).get_data())
        axes = [a for a, s in enumerate(conn.shape) if s == npair]
        if axes and axes[0] != 0:
            conn = np.moveaxis(conn, axes[0], 0)
        out[ei, :] = conn.reshape(npair, -1).mean(axis=1)
    return out


def compute_pte_per_trial(mne_data, indices, band, delay=None, n_bins=4,
                          tmin=None, tmax=None, net=True):
    """Per-trial phase transfer entropy over a band. Returns ``(n_epochs, n_pairs)``.

    Band-filters (FIR), takes the Hilbert phase, and for each trial computes PTE between each
    seed→target pair using only that trial's samples. A single trial is short, so a small fixed
    bin count is used (default 4) rather than Scott's rule. ``net=True`` returns
    ``PTE_{seed→target} - PTE_{target→seed}`` (positive = seed leads); ``net=False`` returns the
    raw seed→target PTE. Per-trial PTE is inherently noisy (one short window) — intended for
    trial-resolved regression with subject-level inference, not single-trial significance.
    """
    from scipy.signal import hilbert
    sfreq = mne_data.info['sfreq']
    if delay is None:
        delay = max(1, int(round(0.1 * sfreq)))
    dat = mne_data.get_data(copy=True)
    n_ep, n_ch, n_t = dat.shape
    filt = mne.filter.filter_data(dat.reshape(-1, n_t), sfreq, band[0], band[1], verbose='ERROR')
    phase = np.angle(hilbert(filt.reshape(n_ep, n_ch, n_t), axis=-1))
    times = mne_data.times
    if (tmin is not None) or (tmax is not None):
        lo = -np.inf if tmin is None else tmin
        hi = np.inf if tmax is None else tmax
        phase = phase[:, :, (times >= lo) & (times < hi)]
    srcs, tgts = indices
    npair = len(srcs)
    out = np.full((n_ep, npair), np.nan, float)
    for p in range(npair):
        ps, pt = phase[:, srcs[p], :], phase[:, tgts[p], :]
        for ei in range(n_ep):
            fwd = phase_transfer_entropy(ps[ei], pt[ei], delay, n_bins_x=n_bins, n_bins_y=n_bins)
            if net:
                rev = phase_transfer_entropy(pt[ei], ps[ei], delay, n_bins_x=n_bins, n_bins_y=n_bins)
                out[ei, p] = fwd - rev
            else:
                out[ei, p] = fwd
    return out


def _per_trial_time_corr(a, b):
    """Row-wise (per-trial) Pearson correlation over the time axis.

    ``a``, ``b`` : real arrays ``(n_trials, n_times)``. Returns ``(n_trials,)`` with the
    Pearson r of each trial's two series across time; zero-variance rows -> NaN.
    """
    a = a - a.mean(axis=-1, keepdims=True)
    b = b - b.mean(axis=-1, keepdims=True)
    num = (a * b).sum(axis=-1)
    den = np.sqrt((a * a).sum(axis=-1) * (b * b).sum(axis=-1))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / den
    r = np.asarray(r, dtype=float)
    r[~np.isfinite(r)] = np.nan
    return r


def compute_aec_per_trial(mne_data, indices, band, *, tmin=None, tmax=None,
                          n_subbands=8, orthogonalize=True, kind="power",
                          filter_kind="butter", order=4):
    """Per-trial amplitude-envelope correlation (AEC) over a band. Returns ``(n_epochs, n_pairs)``.

    The amplitude-domain analogue of ``compute_psi_per_trial`` / ``compute_pte_per_trial`` and the
    same contract: one scalar per trial per seed->target pair, correlating the two channels' band
    envelopes **across time within the trial window** ``[tmin, tmax)``. Intended for trial-resolved
    regression with subject-level inference, not single-trial significance.

    ``band`` is split into ``n_subbands`` equal sub-bands (8 -> the canonical 1/f-robust broadband
    HFA when ``band=(70, 150)``; 1 -> a single band-pass for lower narrowbands). Each sub-band is
    band-pass + Hilbert filtered over the *whole* epoch (so the analysis window is free of filter
    edge transients, matching ``representational_utils._band_envelope``), then cropped to the
    window; the per-trial correlation is computed per sub-band and **averaged over sub-bands**.

    ``orthogonalize=True`` (default) applies the pairwise, leakage-rejecting orthogonalisation of
    Hipp et al. 2012 (Nat Neurosci) before correlating, symmetrised over the two directions::

        Y_perp = |Im(Y * conj(X) / |X|)|,   X_perp = |Im(X * conj(Y) / |Y|)|
        r      = 0.5 * ( corr(env(X), env(Y_perp)) + corr(env(Y), env(X_perp)) )

    which removes the zero-lag shared component (volume conduction / a shared bipolar contact) that
    inflates raw AEC. ``orthogonalize=False`` is the raw envelope correlation ``corr(env(X), env(Y))``
    -- kept as a leakage comparator. ``kind='power'`` (default) correlates squared envelopes; the
    per-electrode-pair correlations follow ``indices`` (pair ``k`` = ``indices[0][k]`` <->
    ``indices[1][k]``), so ``n_pairs = len(indices[0])``.
    """
    from LFPAnalysis.pac_utils import _filter_hilbert  # local import: avoids a module-load cycle

    if kind not in ("power", "amplitude"):
        raise ValueError(f"kind must be 'power' or 'amplitude', got {kind!r}")
    if n_subbands < 1:
        raise ValueError(f"n_subbands must be >= 1, got {n_subbands}")

    sfreq = float(mne_data.info["sfreq"])
    dat = mne_data.get_data(copy=True)          # (n_ep, n_ch, n_t)
    n_ep, _, n_t = dat.shape
    times = mne_data.times

    srcs, tgts = np.asarray(indices[0]), np.asarray(indices[1])
    npair = len(srcs)
    # Filter only the channels actually referenced (unique over seeds+targets), then remap indices.
    used = np.unique(np.concatenate([srcs, tgts]))
    pos = {int(c): i for i, c in enumerate(used)}
    dat = dat[:, used, :]
    srcs = np.array([pos[int(c)] for c in srcs])
    tgts = np.array([pos[int(c)] for c in tgts])

    if (tmin is not None) or (tmax is not None):
        lo = -np.inf if tmin is None else tmin
        hi = np.inf if tmax is None else tmax
        tmask = (times >= lo) & (times < hi)
    else:
        tmask = np.ones(n_t, dtype=bool)

    edges = np.linspace(band[0], band[1], n_subbands + 1)
    tiny = np.finfo(float).tiny
    acc = np.zeros((n_ep, npair), dtype=float)
    cnt = np.zeros((n_ep, npair), dtype=float)

    for lo_e, hi_e in zip(edges[:-1], edges[1:]):
        analytic = _filter_hilbert(dat, sfreq, (float(lo_e), float(hi_e)),
                                   order=order, filter_kind=filter_kind)   # complex (n_ep, n_ch, n_t)
        analytic = analytic[:, :, tmask]
        for p in range(npair):
            X = analytic[:, srcs[p], :]
            Y = analytic[:, tgts[p], :]
            aX, aY = np.abs(X), np.abs(Y)
            if orthogonalize:
                yperp = np.abs(np.imag(Y * np.conj(X) / (aX + tiny)))
                xperp = np.abs(np.imag(X * np.conj(Y) / (aY + tiny)))
                if kind == "power":
                    aX, aY, yperp, xperp = aX ** 2, aY ** 2, yperp ** 2, xperp ** 2
                r = 0.5 * (_per_trial_time_corr(aX, yperp) + _per_trial_time_corr(aY, xperp))
            else:
                if kind == "power":
                    aX, aY = aX ** 2, aY ** 2
                r = _per_trial_time_corr(aX, aY)
            valid = np.isfinite(r)
            acc[valid, p] += r[valid]
            cnt[valid, p] += 1

    with np.errstate(invalid="ignore"):
        out = np.where(cnt > 0, acc / np.where(cnt > 0, cnt, 1), np.nan)
    return out


def compute_surr_connectivity_epochs(mne_data, indices, metric, band, freqs, n_cycles, gc_n_lags=15, buf_ms=1000):

    n_pairs = len(indices[0])
    data = np.swapaxes(mne_data.get_data(copy=False), 0, 1) # swap so now it's chan, events, times 

    surr_dat = np.zeros_like(data) # allocate space for the surrogate channels 

    for ix, ch_dat in enumerate(data): # apply the same swap to every event in a channel, but differ between channels 
        surr_ch = swap_time_blocks(ch_dat, random_state=None)
        surr_dat[ix, :, :] = surr_ch

    surr_dat = np.swapaxes(surr_dat, 0, 1) # swap back so it's events, chan, times 

    # make a new EpochArray from it
    surr_mne = mne.EpochsArray(surr_dat, 
                mne_data.info, 
                tmin=mne_data.tmin, 
                events = mne_data.events, 
                event_id = mne_data.event_id,
                verbose='ERROR')

    if metric == 'psi':
        surr_conn = np.squeeze(phase_slope_index(surr_mne,
                                                    indices=indices,
                                                    sfreq=surr_mne.info['sfreq'],
                                                    mode='cwt_morlet',
                                                    fmin=band[0], fmax=band[1],
                                                    cwt_freqs=freqs,
                                                    cwt_n_cycles=n_cycles,
                                                    verbose='warning').get_data()[:, 0])

    elif metric == 'granger':
        # I don't want to compute multivariate GC, so refactor the indices: 
        surr_conn = []

        for ix, _ in enumerate(indices[0]):
            gc_indices = (np.array([[indices[0][ix]]]), np.array([[indices[1][ix]]]))
        
            surr_gc = compute_gc_tr(mne_data=surr_mne, 
                    band=band,
                    indices=gc_indices, 
                    freqs=freqs[(freqs>=band[0]) & (freqs<=band[1])], 
                    n_cycles=n_cycles[(freqs>=band[0]) & (freqs<=band[1])],
                    rank=None, 
                    gc_n_lags=gc_n_lags, 
                    buf_ms=buf_ms, 
                    avg_over_dim='epochs')
            
            surr_conn.append(surr_gc)
            
        surr_conn = np.vstack(surr_conn)
    else:
        surr_conn = np.squeeze(spectral_connectivity_epochs(surr_mne,
                                                        indices=indices,
                                                        method=metric,
                                                        sfreq=surr_mne.info['sfreq'],
                                                        mode='cwt_morlet',
                                                        fmin=band[0], fmax=band[1], faverage=True,
                                                        cwt_freqs=freqs,
                                                        cwt_n_cycles=n_cycles,
                                                       verbose='ERROR').get_data()[:, 0])
    if metric != 'granger':
        if n_pairs == 1:
            # n_pairs==1: restore (n_pairs, n_times) so the buffer crop slices the
            # TIME axis (was (n_times, 1) -> crop hit the size-1 pair axis -> empty).
            surr_conn = surr_conn.reshape((n_pairs, surr_conn.shape[0]))

        # crop the buffer now:
        buf_rs = int((buf_ms/1000) * surr_mne.info['sfreq'])
        surr_conn = surr_conn[:, buf_rs:-buf_rs]

    return surr_conn


def compute_surr_connectivity_time(mne_data, indices, metric, band, freqs, n_cycles, buf_ms, gc_n_lags=15):

    n_pairs = len(indices[0])
    data = np.swapaxes(mne_data.get_data(copy=False), 0, 1) # swap so now it's chan, events, times 

    surr_dat = np.zeros_like(data) # allocate space for the surrogate channels 

    for ix, ch_dat in enumerate(data): # apply the same swap to every event in a channel, but differ between channels 
        surr_ch = swap_time_blocks(ch_dat, random_state=None)
        surr_dat[ix, :, :] = surr_ch

    surr_dat = np.swapaxes(surr_dat, 0, 1) # swap back so it's events, chan, times 

    # make a new EpochArray from it
    surr_mne = mne.EpochsArray(surr_dat, 
                mne_data.info, 
                tmin=mne_data.tmin, 
                events = mne_data.events, 
                event_id = mne_data.event_id,
                verbose='ERROR')

    if metric == 'granger':
        # I don't want to compute multivariate GC, so refactor the indices: 
        surr_conn = []

        for ix, _ in enumerate(indices[0]):
            gc_indices = (np.array([[indices[0][ix]]]), np.array([[indices[1][ix]]]))
        
            gc = compute_gc_tr(mne_data=surr_mne, 
                    band=band,
                    indices=gc_indices, 
                    freqs=freqs[(freqs>=band[0]) & (freqs<=band[1])], 
                    n_cycles=n_cycles[(freqs>=band[0]) & (freqs<=band[1])],
                    rank=None, 
                    gc_n_lags=gc_n_lags, 
                    buf_ms=buf_ms, 
                    avg_over_dim='time')
            
            surr_conn.append(gc)
            
        surr_conn = np.hstack(surr_conn)
    else:
        surr_conn = np.squeeze(spectral_connectivity_time(data=surr_mne, 
                                    freqs=freqs[(freqs>=band[0]) & (freqs<=band[1])], 
                                    average=False, 
                                    indices=indices, 
                                    method=metric, 
                                    sfreq=surr_mne.info['sfreq'], 
                                    mode='cwt_morlet', 
                                    fmin=band[0], fmax=band[1], faverage=True, 
                                    padding=(buf_ms / 1000), 
                                    n_cycles=n_cycles[(freqs>=band[0]) & (freqs<=band[1])],
                                    rank=None, 
                                    gc_n_lags=gc_n_lags,
                                    verbose='warning').get_data())
    
    if n_pairs == 1:
        # reshape data
        surr_conn = surr_conn.reshape((surr_conn.shape[0], n_pairs))

    return surr_conn


def compute_connectivity(mne_data=None, 
                        band=None,
                        metric=None, 
                        indices=None, 
                        freqs=None, 
                        n_cycles=None, 
                        buf_ms=1000, 
                        avg_over_dim='time',
                        n_surr=500,
                        parallelize=False,
                        band1=None,
                        gc_n_lags=7):
    """
    Compute different connectivity metrics using mne.
    :param eeg_mne: MNE formatted EEG
    :param samplerate: sample rate of the data
    :param band: tuple of band of interest
    :param metric: 'psi' for directional, or for non_directional: ['coh', 'cohy', 'imcoh', 'plv', 'ciplv', 'ppc', 'pli', pli2_unbiased', 'dpli', 'wpli', 'wpli2_debiased']
    see: https://mne.tools/mne-connectivity/stable/generated/mne_connectivity.spectral_connectivity_epochs.html
    :param indices: determine the source and target for connectivity. Matters most for directional metrics i.e. 'psi'
    :return:
    pairwise connectivity: array of pairwise weights for the connectivity metric with some number of timepoints
    """
    if metric == 'gr_tc':
        return (ValueError('Use the function compute_gc_tr'))

    elif metric in ['gc', 'imcoh']: 
        indices = (np.array([np.unique(indices[0]).tolist()]), np.array([np.unique(indices[1]).tolist()]))

    if avg_over_dim == 'epochs':
        if metric == 'amp': 
            return (ValueError('Cannot compute amplitude-amplitude coupling over epochs.'))
        if metric == 'psi': 
            pairwise_connectivity = np.squeeze(phase_slope_index(mne_data,
                                                                    indices=indices,
                                                                    sfreq=mne_data.info['sfreq'],
                                                                    mode='cwt_morlet',
                                                                    fmin=band[0], fmax=band[1],
                                                                    cwt_freqs=freqs,
                                                                    cwt_n_cycles=n_cycles,
                                                                    verbose='warning').get_data()[:, 0])
            # return pairwise_connectivity
        elif metric == 'granger':
            # I don't want to compute multivariate GC, so refactor the indices: 
            pairwise_connectivity = []

            for ix, _ in enumerate(indices[0]):
                gc_indices = (np.array([[indices[0][ix]]]), np.array([[indices[1][ix]]]))
            
                gc = compute_gc_tr(mne_data=mne_data, 
                        band=band,
                        indices=gc_indices, 
                        freqs=freqs[(freqs>=band[0]) & (freqs<=band[1])], 
                        n_cycles=n_cycles[(freqs>=band[0]) & (freqs<=band[1])],
                        rank=None, 
                        gc_n_lags=gc_n_lags, 
                        buf_ms=buf_ms, 
                        avg_over_dim='epochs')
                
                pairwise_connectivity.append(gc)
                
            pairwise_connectivity = np.vstack(pairwise_connectivity)

        else:
            pairwise_connectivity = np.squeeze(spectral_connectivity_epochs(mne_data,
                                                            indices=indices,
                                                            method=metric,
                                                            sfreq=mne_data.info['sfreq'],
                                                            mode='cwt_morlet',
                                                            fmin=band[0], fmax=band[1], faverage=True,
                                                            cwt_freqs=freqs,
                                                            cwt_n_cycles=n_cycles,
                                                            verbose='warning').get_data()[:, 0])
        if metric in ['gc', 'imcoh']:
            # no pairs here: computed over whole multivariate state space 
            n_pairs=1
        else: 
            n_pairs = len(indices[0])

        if metric != 'granger':
            if n_pairs == 1:
                # n_pairs==1 squeezes to (n_times,); restore (n_pairs, n_times) so
                # the buffer crop below slices the TIME axis (was (n_times, 1) ->
                # crop hit the size-1 pair axis -> empty (n_times, 0)).
                pairwise_connectivity = pairwise_connectivity.reshape((n_pairs, pairwise_connectivity.shape[0]))
            # # crop the buffer now:
            buf_rs = int((buf_ms/1000) * mne_data.info['sfreq'])
            pairwise_connectivity = pairwise_connectivity[:, buf_rs:-buf_rs]

        if n_surr > 0:
            if parallelize == True:
                def _process_surrogate_epochs(ns):
                    # print(f'Computing surrogate # {ns} - parallel')
                    surrogate_result = compute_surr_connectivity_epochs(mne_data, indices, metric, band, freqs, n_cycles, gc_n_lags=gc_n_lags, buf_ms=buf_ms)
                    return surrogate_result

                surrogates = Parallel(n_jobs=-1)(delayed(_process_surrogate_epochs)(ns) for ns in range(n_surr))
                surr_struct = np.stack(surrogates, axis=-1)
            else: 
                data = np.swapaxes(mne_data.get_data(copy=False), 0, 1) # swap so now it's chan, events, times 

                surr_struct = np.zeros([pairwise_connectivity.shape[0], n_pairs, n_surr]) # allocate space for all the surrogates 

                # progress_bar = tqdm(np.arange(n_surr), ascii=True, desc='Computing connectivity surrogates')

                for ns in range(n_surr): 
                    # print(f'Computing surrogate # {ns}')
                    surr_dat = np.zeros_like(data) # allocate space for the surrogate channels 
                    for ix, ch_dat in enumerate(data): # apply the same swap to every event in a channel, but differ between channels 
                        surr_ch = swap_time_blocks(ch_dat, random_state=None)
                        surr_dat[ix, :, :] = surr_ch
                    surr_dat = np.swapaxes(surr_dat, 0, 1) # swap back so it's events, chan, times 
                    # make a new EpochArray from it
                    surr_mne = mne.EpochsArray(surr_dat, 
                                mne_data.info, 
                                tmin=mne_data.tmin, 
                                events = mne_data.events, 
                                event_id = mne_data.event_id)

                    if metric == 'psi':
                        surr_conn = np.squeeze(phase_slope_index(surr_mne,
                                                                    indices=indices,
                                                                    sfreq=surr_mne.info['sfreq'],
                                                                    mode='cwt_morlet',
                                                                    fmin=band[0], fmax=band[1],
                                                                    cwt_freqs=freqs,
                                                                    cwt_n_cycles=n_cycles,
                                                                    verbose='warning').get_data()[:, 0])
                    elif metric == 'granger':
                        # I don't want to compute multivariate GC, so refactor the indices: 
                        surr_conn = []

                        for ix, _ in enumerate(indices[0]):
                            gc_indices = (np.array([[indices[0][ix]]]), np.array([[indices[1][ix]]]))
                        
                            surr_gc = compute_gc_tr(mne_data=surr_mne, 
                                    band=band,
                                    indices=gc_indices, 
                                    freqs=freqs[(freqs>=band[0]) & (freqs<=band[1])], 
                                    n_cycles=n_cycles[(freqs>=band[0]) & (freqs<=band[1])],
                                    rank=None, 
                                    gc_n_lags=gc_n_lags, 
                                    buf_ms=buf_ms, 
                                    avg_over_dim='epochs')
                            
                            surr_conn.append(surr_gc)
                            
                        surr_conn = np.vstack(surr_conn)

                    else:
                        surr_conn = np.squeeze(spectral_connectivity_epochs(surr_mne,
                                                                        indices=indices,
                                                                        method=metric,
                                                                        sfreq=surr_mne.info['sfreq'],
                                                                        mode='cwt_morlet',
                                                                        fmin=band[0], fmax=band[1], faverage=True,
                                                                        cwt_freqs=freqs,
                                                                        cwt_n_cycles=n_cycles,
                                                                        verbose='warning').get_data()[:, 0])
                    if metric != 'granger':
                        if n_pairs == 1:
                            # reshape data
                            surr_conn = surr_conn.reshape((surr_conn.shape[0], n_pairs))
                        # crop the surrogate:
                        surr_conn = surr_conn[:, buf_rs:-buf_rs]

                    surr_struct[:, :, ns] = surr_conn
                    clear_output(wait=True)

            surr_mean = np.nanmean(surr_struct, axis=-1)
            surr_std = np.nanstd(surr_struct, axis=-1)
            pairwise_connectivity = (pairwise_connectivity - surr_mean) / (surr_std)
            
            # surr_struct[:, :, -1] = pairwise_connectivity # add the real data in as the last entry 
            # z_struct = zscore(surr_struct, axis=-1) # take the zscore across surrogate runs and the real data 
            # pairwise_connectivity = z_struct[:, :, -1] # extract the real data
    elif avg_over_dim == 'time':    
        if metric == 'psi': 
            return (ValueError('Cannot compute psi over time.'))
        elif metric == 'amp': 
            
            # crop the buffer first:
            buf_s = buf_ms / 1000
            mne_data.crop(tmin=mne_data.tmin + buf_s,
                          tmax=mne_data.tmax - buf_s)

            pairwise_connectivity = amp_amp_coupling(mne_data, 
                                                     indices, 
                                                     freqs0=band,
                                                     freqs1=band1)
            if metric in ['gc', 'imcoh']:
                # no pairs here: computed over whole multivariate state space 
                n_pairs=1
            else: 
                n_pairs = len(indices[0])

            if n_pairs == 1:
                # reshape data
                pairwise_connectivity = pairwise_connectivity.reshape((pairwise_connectivity.shape[0], n_pairs))

            if n_surr > 0:
                data = np.swapaxes(mne_data.get_data(copy=False), 0, 1) # swap so now it's chan, events, times 

                surr_struct = np.zeros([pairwise_connectivity.shape[0], n_pairs, n_surr]) # allocate space for all the surrogates 

                # progress_bar = tqdm(np.arange(n_surr), ascii=True, desc='Computing connectivity surrogates')

                for ns in range(n_surr): 
                    # print(f'Computing surrogate # {ns}')
                    surr_dat = np.zeros_like(data) # allocate space for the surrogate channels 
                    for ix, ch_dat in enumerate(data): # apply the same swap to every event in a channel, but differ between channels 
                        surr_ch = swap_time_blocks(ch_dat, random_state=None)
                        surr_dat[ix, :, :] = surr_ch
                    surr_dat = np.swapaxes(surr_dat, 0, 1) # swap back so it's events, chan, times 
                    # make a new EpochArray from it
                    surr_mne = mne.EpochsArray(surr_dat, 
                                mne_data.info, 
                                tmin=mne_data.tmin, 
                                events = mne_data.events, 
                                event_id = mne_data.event_id)

                    surr_conn = amp_amp_coupling(surr_mne, 
                                                 indices, 
                                                 freqs0=band,
                                                 freqs1=band1)
                    if n_pairs == 1:
                        # reshape data
                        surr_conn = surr_conn.reshape((surr_conn.shape[0], n_pairs))

                    surr_struct[:, :, ns] = surr_conn
                    clear_output(wait=True)

                surr_mean = np.nanmean(surr_struct, axis=-1)
                surr_std = np.nanstd(surr_struct, axis=-1)
                pairwise_connectivity = (pairwise_connectivity - surr_mean) / (surr_std)
                # surr_struct[:, :, -1] = pairwise_connectivity # add the real data in as the last entry
                # z_struct = zscore(surr_struct, axis=-1) # take the zscore across surrogate runs and the real data
                # pairwise_connectivity = z_struct[:, :, -1] # extract the real data      
        else:
            if metric == 'granger':
                # I don't want to compute multivariate GC, so refactor the indices: 
                pairwise_connectivity = []

                for ix, _ in enumerate(indices[0]):
                    gc_indices = (np.array([[indices[0][ix]]]), np.array([[indices[1][ix]]]))
                
                    gc = compute_gc_tr(mne_data=mne_data, 
                            band=band,
                            indices=gc_indices, 
                            freqs=freqs[(freqs>=band[0]) & (freqs<=band[1])], 
                            n_cycles=n_cycles[(freqs>=band[0]) & (freqs<=band[1])],
                            rank=None, 
                            gc_n_lags=gc_n_lags, 
                            buf_ms=buf_ms, 
                            avg_over_dim='time')
                    
                    pairwise_connectivity.append(gc)
                    
                pairwise_connectivity = np.hstack(pairwise_connectivity)
            else:
                pairwise_connectivity = np.squeeze(spectral_connectivity_time(data=mne_data, 
                                                    freqs=freqs[(freqs>=band[0]) & (freqs<=band[1])], 
                                                    average=False, 
                                                    indices=indices, 
                                                    method=metric, 
                                                    sfreq=mne_data.info['sfreq'], 
                                                    mode='cwt_morlet', 
                                                    fmin=band[0], fmax=band[1], faverage=True, 
                                                    padding=(buf_ms / 1000), 
                                                    n_cycles=n_cycles[(freqs>=band[0]) & (freqs<=band[1])],
                                                    rank=None,
                                                    gc_n_lags=gc_n_lags,
                                                    verbose='warning').get_data())
                # This returns an array of shape (n_events, n_pairs) 
                # where n_pairs is the number of pairs of channels in indices
                # and n_events is the number of events in the data

            
            if metric in ['gc', 'imcoh']:
                # no pairs here: computed over whole multivariate state space 
                n_pairs=1
            else: 
                n_pairs = len(indices[0])

            if n_pairs == 1:
                # reshape data
                pairwise_connectivity = pairwise_connectivity.reshape((pairwise_connectivity.shape[0], n_pairs))

            if n_surr > 0:
                if parallelize == True:
                    def _process_surrogate_time(ns):
                        # print(f'Computing surrogate # {ns} - parallel')
                        surrogate_result = compute_surr_connectivity_time(mne_data, indices, metric, band, freqs, n_cycles, buf_ms, gc_n_lags)
                        return surrogate_result

                    surrogates = Parallel(n_jobs=-1)(delayed(_process_surrogate_time)(ns) for ns in range(n_surr))
                    surr_struct = np.stack(surrogates, axis=-1)
                else:
                    data = np.swapaxes(mne_data.get_data(copy=False), 0, 1) # swap so now it's chan, events, times 

                    surr_struct = np.zeros([pairwise_connectivity.shape[0], n_pairs, n_surr]) # allocate space for all the surrogates 

                    # progress_bar = tqdm(np.arange(n_surr), ascii=True, desc='Computing connectivity surrogates')

                    for ns in range(n_surr): 
                        # print(f'Computing surrogate # {ns}')
                        surr_dat = np.zeros_like(data) # allocate space for the surrogate channels 
                        for ix, ch_dat in enumerate(data): # apply the same swap to every event in a channel, but differ between channels 
                            surr_ch = swap_time_blocks(ch_dat, random_state=None)
                            surr_dat[ix, :, :] = surr_ch
                        surr_dat = np.swapaxes(surr_dat, 0, 1) # swap back so it's events, chan, times 
                        # make a new EpochArray from it
                        surr_mne = mne.EpochsArray(surr_dat, 
                                    mne_data.info, 
                                    tmin=mne_data.tmin, 
                                    events = mne_data.events, 
                                    event_id = mne_data.event_id)
                        
                        surr_conn = np.squeeze(spectral_connectivity_time(data=surr_mne, 
                                                    freqs=freqs[(freqs>=band[0]) & (freqs<=band[1])], 
                                                    average=False, 
                                                    indices=indices, 
                                                    method=metric, 
                                                    sfreq=surr_mne.info['sfreq'], 
                                                    mode='cwt_morlet', 
                                                    fmin=band[0], fmax=band[1], faverage=True, 
                                                    padding=(buf_ms / 1000), 
                                                    n_cycles=n_cycles[(freqs>=band[0]) & (freqs<=band[1])],
                                                    gc_n_lags=gc_n_lags,
                                                    verbose='warning').get_data())
                        
                        if n_pairs == 1:
                            # reshape data
                            surr_conn = surr_conn.reshape((surr_conn.shape[0], n_pairs))

                        surr_struct[:, :, ns] = surr_conn
                        clear_output(wait=True)

                surr_mean = np.nanmean(surr_struct, axis=-1)
                surr_std = np.nanstd(surr_struct, axis=-1)
                pairwise_connectivity = (pairwise_connectivity - surr_mean) / (surr_std)
                # surr_struct[:, :, -1] = pairwise_connectivity # add the real data in as the last entry
                # z_struct = zscore(surr_struct, axis=-1) # take the zscore across surrogate runs and the real data
                # pairwise_connectivity = z_struct[:, :, -1] # extract the real data            

    return pairwise_connectivity


# def compute_indices(elec_df, roi = ['hippocampus', 'anterior_cingulate'], band=[8, 13], band_name='beta'): 
#     """
#     Use mne connectivity to compute the spectral connectivity between electrodes 
#     the first roi is the seed. the second roi is the target. 
#     """


#         # set a mask for the right electrodes 
#         right_elec_mask = [elec_df.hemisphere=='r']
#         seed_target_df = pd.DataFrame(columns=['seed', 'target'], index=['left', 'right'])
#         seed_target_df['seed']['left'] = np.where(elec_df[~right_elec_mask].region == roi[0])[0]
#         seed_target_df['target']['left'] = np.where(elec_df[~right_elec_mask].region == self.roi[1])[0]
#         seed_target_df['seed']['right'] = np.where(elec_df[right_elec_mask].region == self.roi[0])[0]
#         seed_target_df['target']['right'] = np.where(elec_df[right_elec_mask].region == self.roi[1])[0]


#         seed_target_df = seed_target_df[
#             (seed_target_df['seed'].map(lambda d: len(d) > 0)) & (seed_target_df['target'].map(lambda d: len(d) > 0))]

#             # Dealing with multiple channels: stack them, don't average them
#         psi = {}
#         for hemi in ['left', 'right']:
#             # first determine if ipsi connectivity is even possible; if not, move on
#             if hemi not in seed_target_df.index.tolist():
#                 continue
#             else:
#                 seed_to_target = seed_target_indices(
#                     seed_target_df['seed'][hemi],
#                     seed_target_df['target'][hemi])

#                     self.compute_connectivity(eeg_mne[recalled],
#                                                           samplerate=self.resample_freq,
#                                                           band=[self.fb[band][0], self.fb[band][1]],
#                                                           metric=self.metric,
#                                                           indices=seed_to_source,
#                                                           n_cycles=self.n_cycles)
        

"""
BOSC (Better Oscillation Detection) function library
Rewritten from MATLAB to Python by Julian Q. Kosciessa

The original license information follows:
---
This file is part of the Better OSCillation detection (BOSC) library.

The BOSC library is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

The BOSC library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Foobar.  If not, see <http://www.gnu.org/licenses/>.

Copyright 2010 Jeremy B. Caplan, Adam M. Hughes, Tara A. Whitten
and Clayton T. Dickson.
---
"""

def BOSC_tf(eegsignal,F,Fsample,wavenumber):
    """
    Computes the Better Oscillation Detection (BOSC) time-frequency matrix for a given LFP signal.

    Args:
    - eegsignal (numpy.ndarray): The LFP signal to compute the BOSC time-frequency matrix for.
    - F (numpy.ndarray): The frequency range to compute the BOSC time-frequency matrix over.
    - Fsample (float): The sampling frequency of the LFP signal.
    - wavenumber (float): The wavenumber to use for the Morlet wavelet.

    Returns:
    - B (numpy.ndarray): The BOSC time-frequency matrix.
    - T (numpy.ndarray): The time vector corresponding to the BOSC time-frequency matrix.
    - F (numpy.ndarray): The frequency vector corresponding to the BOSC time-frequency matrix.
    """

    st=1./(2*np.pi*(F/wavenumber))
    A=1./np.sqrt(st*np.sqrt(np.pi))
    # initialize the time-frequency matrix
    B = np.zeros((len(F),len(eegsignal)))
    B[:] = np.nan
    # loop through sampled frequencies
    for f in range(len(F)):
        #print(f)
        t=np.arange(-3.6*st[f],(3.6*st[f]),1/Fsample)
        # define Morlet wavelet
        m=A[f]*np.exp(-t**2/(2*st[f]**2))*np.exp(1j*2*np.pi*F[f]*t)
        y=np.convolve(eegsignal,m, 'full')
        y=abs(y)**2
        B[f,:]=y[np.arange(int(np.ceil(len(m)/2))-1, len(y)-int(np.floor(len(m)/2)), 1)]
        T=np.arange(1,len(eegsignal)+1,1)/Fsample
    return B, T, F


def BOSC_detect(b,powthresh,durthresh,Fsample):
    """
    detected=BOSC_detect(b,powthresh,durthresh,Fsample)
    This function detects oscillations based on a wavelet power
    timecourse, b, a power threshold (powthresh) and duration
    threshold (durthresh) returned from BOSC_thresholds.m.
    
    It now returns the detected vector which is already episode-detected.
    
    b - the power timecourse (at one frequency of interest)
    
    durthresh - duration threshold in  required to be deemed oscillatory
    powthresh - power threshold
    
    returns:
    detected - a binary vector containing the value 1 for times at
               which oscillations (at the frequency of interest) were
               detected and 0 where no oscillations were detected.
    
    note: Remember to account for edge effects by including
    "shoulder" data and accounting for it afterwards!
    
    To calculate Pepisode:
    Pepisode=length(find(detected))/(length(detected));
    """                           

    # number of time points
    nT=len(b)
    #t=np.arange(1,nT+1,1)/Fsample
    
    # Step 1: power threshold
    x=b>powthresh
    # we have to turn the boolean to numeric
    x = np.array(list(map(int, x)))
    # show the +1 and -1 edges
    dx=np.diff(x)
    if np.size(np.where(dx==1))!=0:
        pos=np.where(dx==1)[0]+1
        #pos = pos[0]
    else: pos = []
    if np.size(np.where(dx==-1))!=0:
        neg=np.where(dx==-1)[0]+1
        #neg = neg[0]
    else: neg = []

    # now do all the special cases to handle the edges
    detected=np.zeros(b.shape)
    if not any(pos) and not any(neg):
        # either all time points are rhythmic or none
        if all(x==1):
            H = np.array([[0],[nT]])
        elif all(x==0):
            H = np.array([])
    elif not any(pos):
        # i.e., starts on an episode, then stops
        H = np.array([[0],neg])
        #np.concatenate(([1],neg), axis=0)
    elif not any(neg):
        # starts, then ends on an ep.
        H = np.array([pos,[nT]])
        #np.concatenate((pos,[nT]), axis=0)
    else:
        # special-case, create the H double-vector
        if pos[0]>neg[0]:
            # we start with an episode
            pos = np.append(0,pos)
        if neg[-1]<pos[-1]:
            # we end with an episode
            neg = np.append(neg,nT)
        # NOTE: by this time, length(pos)==length(neg), necessarily
        H = np.array([pos,neg])
        #np.concatenate((pos,neg), axis=0)
    
    if H.shape[0]>0: 
        # more than one "hole"
        # find epochs lasting longer than minNcycles*period
        goodep=H[1,]-H[0,]>=durthresh
        if not any(goodep):
            H = [] 
        else: 
            H = H[:,goodep.nonzero()][:,0]
            # mark detected episode on the detected vector
            for h in range(H.shape[1]):
                detected[np.arange(H[0][h], H[1][h],1)]=1
        
    # ensure that outputs are integer
    detected = np.array(list(map(int, detected)))
    return detected

def _bosc_background(freqs, mean_log_pow, wavenumber=6.0, exclude_peak=None):
    """Robust log-log 1/f fit of a wavelet power spectrum for BOSC/eBOSC thresholding.

    Parameters
    ----------
    freqs : (n_freq,) wavelet center frequencies (Hz).
    mean_log_pow : (n_freq,) mean over trials & time of log10(wavelet power).
    wavenumber : Morlet wavenumber (sets the FWHM used for peak exclusion).
    exclude_peak : None, or (flo, fhi) Hz. When given, the empirical peak in
        [flo, fhi] and its wavelet FWHM are dropped from the aperiodic fit so a
        rhythm does not bias its own background (the eBOSC peak-exclusion step).

    Returns
    -------
    slope, intercept : robust (Tukey biweight) log-log fit parameters.
    mp : (n_freq,) linear background power at each frequency (10**fit).
    """
    freqs = np.asarray(freqs, float)
    mean_log_pow = np.asarray(mean_log_pow, float)
    keep = np.ones(freqs.shape, bool)
    if exclude_peak is not None:
        flo, fhi = exclude_peak
        in_rng = np.where((freqs >= flo) & (freqs <= fhi))[0]
        if in_rng.size:
            ipk = in_rng[int(np.argmax(mean_log_pow[in_rng]))]
            fwhm = (2.0 / wavenumber) * freqs[ipk]
            keep[(freqs >= freqs[ipk] - fwhm / 2.0) & (freqs <= freqs[ipk] + fwhm / 2.0)] = False
    if keep.sum() < 2:                              # never fit on <2 points
        keep = np.ones(freqs.shape, bool)
    exog = sm.add_constant(np.log10(freqs[keep]))
    res = sm.RLM(mean_log_pow[keep], exog, M=sm.robust.norms.TukeyBiweight()).fit()
    intercept, slope = float(res.params[0]), float(res.params[1])
    mp = 10.0 ** (slope * np.log10(freqs) + intercept)
    return slope, intercept, mp


def _detect_sustained(power, powthresh, durthresh):
    """Binary detection: power above `powthresh` for runs >= `durthresh` samples.

    Robust run-length implementation (handles runs touching either edge); replaces
    the edge-case-fragile BOSC_detect on the P-episode path.
    """
    above = np.asarray(power, float) > powthresh
    detected = np.zeros(above.shape[0], dtype=int)
    if not above.any():
        return detected
    edges = np.diff(np.concatenate(([0], above.astype(int), [0])))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    for s, e in zip(starts, ends):
        if (e - s) >= durthresh:
            detected[s:e] = 1
    return detected


def pepisode_spectrum(data, sfreq, freqs, wavenumber=6.0, percentile=0.95,
                      n_cycles=3.0, exclude_peak=None, pad_s=0.0,
                      times=None, win=None):
    """BOSC / eBOSC P-episode (rhythmic abundance) spectrum for one channel.

    P-episode at frequency f is the fraction of (trial x time) samples at which a
    *sustained* rhythm is detected: wavelet power exceeds a chi-square(2) threshold
    on the robust aperiodic 1/f background AND persists for at least `n_cycles`
    cycles. This separates a genuine, temporally sustained oscillation from
    aperiodic 1/f power that merely passes a fixed-band filter -- the question the
    Preston/Smith/Voytek aperiodic review forces for any "theta" claim.

    The rhythm detection always runs on the WHOLE input timecourse (so episodes
    spanning the window edge use the surrounding data as buffer, BOSC-style); the
    abundance is then averaged only over the analysis window (`win`) if given, or
    over the centre after trimming `pad_s` from each edge, or over everything.

    Parameters
    ----------
    data : (n_trials, n_times) or (n_times,) raw signal for one channel (NOT
        z-scored / baseline-normalised -- BOSC needs raw wavelet power). Pass the
        full buffered epoch; restrict the readout with `win`.
    sfreq : sampling frequency (Hz).
    freqs : center frequencies (Hz). Use a range WIDER than the band of interest
        (e.g. 2-40) so the 1/f background fit is stable; report the sub-band.
    wavenumber : Morlet wavenumber (default 6).
    percentile : background percentile for the power threshold (default 0.95).
    n_cycles : duration threshold in cycles (default 3).
    exclude_peak : (flo, fhi) Hz removed from the background fit, or None.
    pad_s : seconds trimmed from each edge before averaging (used only when `win`
        is not given).
    times : (n_times,) time vector (s) matching `data`; required to use `win`.
    win : (tmin, tmax) seconds -- average abundance only within this window, using
        the rest of the epoch as wavelet/detection buffer.

    Returns
    -------
    dict: freqs, pepisode (per freq), bg_mp (linear background power), pt (power
        threshold per freq), slope, intercept, n_trials, n_clean_samples,
        win (the analysis window actually used, or None).
    """
    data = np.asarray(data, float)
    if data.ndim == 1:
        data = data[None, :]
    freqs = np.asarray(freqs, float)
    nfreq = freqs.size
    ntime = data.shape[1]

    # per-trial wavelet power on the full (buffered) epoch: (n_trial, n_freq, n_time)
    P = np.empty((data.shape[0], nfreq, ntime))
    for tr in range(data.shape[0]):
        B, _, _ = BOSC_tf(data[tr], freqs, sfreq, wavenumber)
        P[tr] = B

    # robust aperiodic background + chi2(2) power threshold (per freq)
    mean_log_pow = np.log10(P).mean(axis=(0, 2))
    slope, intercept, mp = _bosc_background(freqs, mean_log_pow, wavenumber, exclude_peak)
    pt = chi2.ppf(percentile, 2) * mp / 2.0                # per-freq power threshold
    dt = n_cycles * sfreq / freqs                          # per-freq duration (samples)

    # samples to COUNT (detection itself always uses the full timecourse as buffer)
    if win is not None and times is not None:
        times = np.asarray(times, float)
        cmask = (times >= win[0]) & (times <= win[1])
    elif pad_s > 0:
        pad = int(round(pad_s * sfreq))
        cmask = np.zeros(ntime, bool)
        cmask[pad:ntime - pad] = True
    else:
        cmask = np.ones(ntime, bool)
    n_count = int(cmask.sum())

    det_sum = np.zeros(nfreq)
    for tr in range(P.shape[0]):
        for fi in range(nfreq):
            det = _detect_sustained(P[tr, fi, :], pt[fi], dt[fi])
            det_sum[fi] += det[cmask].sum()
    pepisode = det_sum / max(P.shape[0] * n_count, 1)

    return {"freqs": freqs, "pepisode": pepisode, "bg_mp": mp, "pt": pt,
            "slope": slope, "intercept": intercept,
            "n_trials": int(P.shape[0]), "n_clean_samples": int(P.shape[0] * n_count),
            "win": win}


def eBOSC_getThresholds(cfg_eBOSC, TFR, eBOSC):
    """This function estimates the static duration and power thresholds and
    saves information regarding the overall spectrum and background.
    Inputs: 
               cfg | config structure with cfg.eBOSC field
               TFR | time-frequency matrix
               eBOSC | main eBOSC output structure; will be updated
    
    Outputs: 
               eBOSC   | updated w.r.t. background info (see below)
                       | bg_pow: overall power spectrum
                       | bg_log10_pow: overall power spectrum (log10)
                       | pv: intercept and slope of fit
                       | mp: linear background power
                       | pt: power threshold
               pt | empirical power threshold
               dt | duration threshold
    """

    # concatenate power estimates in time across trials of interest
    
    trial2extract = cfg_eBOSC['trial_background']
    # remove BGpad at beginning and end to avoid edge artifacts
    time2extract = np.arange(cfg_eBOSC['pad.background_sample'], TFR.shape[2]-cfg_eBOSC['pad.background_sample'],1)
    # index both trial and time dimension simultaneously
    TFR = TFR[np.ix_(trial2extract,range(TFR.shape[1]),time2extract)]
    # concatenate trials in time dimension: permute dimensions, then reshape
    TFR_t = np.transpose(TFR, [1,2,0])
    BG = TFR_t.reshape(TFR_t.shape[0],TFR_t.shape[1]*TFR_t.shape[2])
    del TFR_t, trial2extract, time2extract    
    # plt.imshow(BG[:,0:100], extent=[0, 1, 0, 1])
    
    # if frequency ranges should be exluded to reduce the influence of
    # rhythmic peaks on the estimation of the linear background, the
    # following section removes these specified ranges
    freqKeep = np.ones(cfg_eBOSC['F'].shape, dtype=bool)
    # allow for no peak removal
    if cfg_eBOSC['threshold.excludePeak'].size == 0:
        print("NOT removing frequency peaks from the background")
    else:
        print("Removing frequency peaks from the background")
        # n-dimensional arrays allow for the removal of multiple peaks
        for indExFreq in range(cfg_eBOSC['threshold.excludePeak'].shape[0]):
            # find empirical peak in specified range
            freqInd1 = np.where(cfg_eBOSC['F'] >= cfg_eBOSC['threshold.excludePeak'][indExFreq,0])[0][0]
            freqInd2 = np.where(cfg_eBOSC['F'] <= cfg_eBOSC['threshold.excludePeak'][indExFreq,1])[-1][-1]
            freqidx = np.arange(freqInd1,freqInd2+1)
            meanbg_within_range = list(BG[freqidx,:].mean(1))
            indPos = meanbg_within_range.index(max(meanbg_within_range))
            indPos = freqidx[indPos]
            # approximate wavelet extension in frequency domain
            # note: we do not remove the specified range, but the FWHM
            # around the empirical peak
            LowFreq = cfg_eBOSC['F'][indPos]-(((2/cfg_eBOSC['wavenumber'])*cfg_eBOSC['F'][indPos])/2)
            UpFreq = cfg_eBOSC['F'][indPos]+(((2/cfg_eBOSC['wavenumber'])*cfg_eBOSC['F'][indPos])/2)
            # index power estimates within the above range to remove from BG fit
            freqKeep[np.logical_and(cfg_eBOSC['F'] >= LowFreq, cfg_eBOSC['F'] <= UpFreq)] = False

    fitInput = {}
    fitInput['f_'] = cfg_eBOSC['F'][freqKeep]
    fitInput['BG_'] = BG[freqKeep, :]
   
    dataForBG = np.log10(fitInput['BG_']).mean(1)
    
    # perform the robust linear fit, only including putatively aperiodic components (i.e., peak exclusion)
    # replicate TukeyBiweight from MATLABs robustfit function
    exog = np.log10(fitInput['f_'])
    exog = sm.add_constant(exog)
    endog = dataForBG
    rlm_model = sm.RLM(endog, exog, M=sm.robust.norms.TukeyBiweight())
    rlm_results = rlm_model.fit()
    # MATLAB: b = robustfit(np.log10(fitInput['f_']),dataForBG)
    pv = np.zeros(2)
    pv[0] = rlm_results.params[1]
    pv[1] = rlm_results.params[0]
    mp = 10**(np.polyval(pv,np.log10(cfg_eBOSC['F'])))

    # compute eBOSC power (pt) and duration (dt) thresholds: 
    # power threshold is based on a chi-square distribution with df=2 and mean as estimated above
    pt=chi2.ppf(cfg_eBOSC['threshold.percentile'],2)*mp/2
    # duration threshold is the specified number of cycles, so it scales with frequency
    dt=(cfg_eBOSC['threshold.duration']*cfg_eBOSC['fsample']/cfg_eBOSC['F'])
    dt=np.transpose(dt, [1,0])

    # save multiple time-invariant estimates that could be of interest:
    # overall wavelet power spectrum (NOT only background)
    time2encode = np.arange(cfg_eBOSC['pad.total_sample'], BG.shape[1]-cfg_eBOSC['pad.total_sample'],1)
    eBOSC['static.bg_pow'].loc[cfg_eBOSC['tmp_channel'],:] = BG[:,time2encode].mean(1)
    # eBOSC[cfg_eBOSC['tmp_channelID']] = {'static.bg_pow': BG[:,time2encode].mean(1)}
    # log10-transformed wavelet power spectrum (NOT only background)
    eBOSC['static.bg_log10_pow'].loc[cfg_eBOSC['tmp_channel'],:] = np.log10(BG[:,time2encode]).mean(1)
    # intercept and slope parameters of the robust linear 1/f fit (log-log)
    eBOSC['static.pv'].loc[cfg_eBOSC['tmp_channel'],:] = pv
    # linear background power at each estimated frequency
    eBOSC['static.mp'].loc[cfg_eBOSC['tmp_channel'],:] = mp
    # statistical power threshold
    eBOSC['static.pt'].loc[cfg_eBOSC['tmp_channel'],:] = pt

    return eBOSC, pt, dt

def eBOSC_episode_sparsefreq(cfg_eBOSC, detected, TFR):
    """Sparsen the detected matrix along the frequency dimension
    """    
    # print('Creating sparse detected matrix ...')
    
    freqWidth = (2/cfg_eBOSC['wavenumber'])*cfg_eBOSC['F']
    lowFreq = cfg_eBOSC['F']-(freqWidth/2)
    highFreq = cfg_eBOSC['F']+(freqWidth/2)
    # %% define range for each frequency across which max. is detected
    fmat = np.zeros([cfg_eBOSC['F'].shape[0],3])
    for [indF,valF] in enumerate(cfg_eBOSC['F']):
        #print(indF)
        lastVal = np.where(cfg_eBOSC['F']<=lowFreq[indF])[0]
        if len(lastVal)>0:
            # first freq falling into range
            fmat[indF,0] = lastVal[-1]+1
        else: fmat[indF,0] = 0
        firstVal = np.where(cfg_eBOSC['F']>=highFreq[indF])[0]
        if len(firstVal)>0:
            # last freq falling into range
            fmat[indF,2] = firstVal[0]-1
        else: fmat[indF,2] = cfg_eBOSC['F'].shape[0]-1
    fmat[:,1] = np.arange(0, cfg_eBOSC['F'].shape[0],1)
    del indF
    range_cur = np.diff(fmat, axis=1)
    range_cur = [int(np.max(range_cur[:,0])), int(np.max(range_cur[:,1]))]
    # %% perform the actual search
    # initialize variables
    # append frequency search space (i.e. range at both ends. first index refers to lower range
    c1 = np.zeros([int(range_cur[0]),TFR.shape[1]])
    c2 = TFR*detected
    c3 = np.zeros([int(range_cur[1]),TFR.shape[1]])
    tmp_B = np.concatenate([c1, c2, c3])
    del c1,c2,c3
    # preallocate matrix (incl. padding , which will be removed)
    detected = np.zeros(tmp_B.shape)
    # loop across frequencies. note that indexing respects the appended segments
    freqs_to_search = np.arange(int(range_cur[0]), int(tmp_B.shape[0]-range_cur[1]),1)
    for f in freqs_to_search:
        # encode detected positions where power is higher than in LOWER and HIGHER ranges
        range1 = [f+np.arange(1,int(range_cur[1])+1)][0]
        range2 = [f-np.arange(1,int(range_cur[0])+1)][0]
        ranges = np.concatenate([range1,range2])
        detected[f,:] = np.logical_and(tmp_B[f,:] != 0, np.min(tmp_B[f,:] >= tmp_B[ranges,:],axis=0))
    # only retain data without padded zeros
    detected = detected[freqs_to_search,:]
    return detected

def eBOSC_episode_postproc_fwhm(cfg_eBOSC, episodes, TFR):
    """
    % This function performs post-processing of input episodes by checking
    % whether 'detected' time points can trivially be explained by the FWHM of
    % the wavelet used in the time-frequency transform.
    %
    % Inputs: 
    %           cfg | config structure with cfg.eBOSC field
    %           episodes | table of episodes
    %           TFR | time-frequency matrix
    %
    % Outputs: 
    %           episodes_new | updated table of episodes
    %           detected_new | updated binary detected matrix
    """
    
    print("Applying FWHM post-processing ...")
    
    # re-initialize detected_new (for post-proc results)
    detected_new = np.zeros(TFR.shape)
    # initialize new dictionary to save results in
    episodesTable = {}
    for entry in episodes:
        episodesTable[entry] = []

    for e in range(len(episodes['Trial'])):
        # get temporary frequency vector
        f_ = episodes['Frequency'][e]
        f_unique = np.unique(f_)           
        # find index within minor tolerance (float arrays)
        f_ind_unique = np.where(np.abs(cfg_eBOSC['F'][:,None] - f_unique) < 1e-5)
        f_ind_unique = f_ind_unique[0]
        # get temporary amplitude vector
        a_ = episodes['Power'][e]
        # location in time with reference to matrix TFR
        t_ind = np.int_(np.arange(episodes['ColID'][e][0], episodes['ColID'][e][-1]+1))
        # initiate bias matrix (only requires to encode frequencies occuring within episode)
        biasMat = np.zeros([len(f_unique),len(a_)])

        for tp in range(len(a_)):
            # The FWHM correction is done independently at each
            # frequency. To accomplish this, we actually reference
            # to the original data in the TF matrix.
            # search within frequencies that occur within the episode
            for f in range(len(f_unique)):
                # create wavelet with center frequency and amplitude at time point
                st=1/(2*np.pi*(f_unique/cfg_eBOSC['wavenumber']))
                step_size = 1/cfg_eBOSC['fsample']
                t=np.arange(-3.6*st[f],3.6*st[f]+step_size,step_size)
                wave = np.exp(-t**2/(2*st[f]**2))*np.exp(1j*2*np.pi*f_unique[f]*t)                
                if cfg_eBOSC['postproc.effSignal'] == 'all':
                    # Morlet wavelet with amplitude-power threshold modulation
                    m = TFR[f_ind_unique[f], int(t_ind[tp])]*wave
                elif cfg_eBOSC['postproc.effSignal'] == 'PT':
                    m = (TFR[f_ind_unique[f], int(t_ind[tp])]-
                         cfg_eBOSC['tmp.pt'][f_ind_unique[f]])*wave
                # amplitude of wavelet
                wl_a = abs(m)
                maxval = max(wl_a)
                maxloc = np.where(np.abs(wl_a[:,None] - maxval) < 1e-5)[0][0]
                index_fwhm = np.where(wl_a>= maxval/2)[0][0]
                # amplitude at fwhm, freq
                fwhm_a = wl_a[index_fwhm]
                if cfg_eBOSC['postproc.effSignal'] =='PT':
                    # re-add power threshold
                    fwhm_a = fwhm_a+cfg_eBOSC['tmp.pt'][f_ind_unique[f]]
                correctionDist = maxloc-index_fwhm
                # extract FWHM amplitude of frequency- and amplitude-specific wavelet
                # check that lower fwhm is part of signal 
                if tp-correctionDist >= 0:
                    # and that existing value is lower than update
                    if biasMat[f,tp-correctionDist] < fwhm_a:
                        biasMat[f,tp-correctionDist] = fwhm_a
                # check that upper fwhm is part of signal 
                if tp+correctionDist+1 <= biasMat.shape[1]:
                    # and that existing value is lower than update
                    if biasMat[f,tp+correctionDist] < fwhm_a:
                        biasMat[f,tp+correctionDist] = fwhm_a

        # plt.imshow(biasMat, extent=[0, 1, 0, 1])

        # retain only those points that are larger than the FWHM
        aMat_retain = np.zeros(biasMat.shape)
        indFreqs = np.where(np.abs(f_[:,None] - f_unique) < 1e-5)
        indFreqs = indFreqs[1]
        for indF in range(len(f_unique)):
            aMat_retain[indF,np.where(indFreqs == indF)[0]] = np.transpose(a_[indFreqs == indF])
        # anything that is lower than the convolved wavelet is removed
        aMat_retain[aMat_retain <= biasMat] = 0

        # identify which time points to retain and discard
        # Options: only correct at signal edge; correct within entire signal
        keep = aMat_retain.mean(0)>0
        keep = keep>0
        if cfg_eBOSC['postproc.edgeOnly'] == 'yes':
            keepEdgeRemovalOnly = np.zeros([len(keep)],dtype=bool)
            keepEdgeRemovalOnly[np.arange(np.where(keep==1)[0][0],np.where(keep==1)[0][-1]+1)] = True
            keep = keepEdgeRemovalOnly
            del keepEdgeRemovalOnly
            
        # get new episodes
        keep = np.concatenate(([0], keep, [0]))
        d_keep = np.diff(keep.astype(float))
    
        if max(d_keep) == 1 and min(d_keep) == -1:
            # start and end indices
            ind_epsd_begin = np.where(d_keep == 1)[0]
            ind_epsd_end = np.where(d_keep == -1)[0]-1
            for i in range(len(ind_epsd_begin)):
                # check for passing the duration requirement
                # get average frequency
                tmp_col = np.arange(ind_epsd_begin[i],ind_epsd_end[i]+1)
                avg_frq = np.mean(f_[tmp_col])
                # match to closest frequency
                [tmp_a, indF] = find_nearest_value(cfg_eBOSC['F'], avg_frq)
                # check number of data points to fulfill number of cycles criterion
                num_pnt = np.floor((cfg_eBOSC['fsample']/ avg_frq) * int(np.reshape(cfg_eBOSC['threshold.duration'],[-1,1])[indF]))
                # if duration criterion remains fulfilled, encode in table
                if len(tmp_col) >= num_pnt:
                    # update all data in table with new episode limits
                    episodesTable['RowID'].append(episodes['RowID'][e][tmp_col])
                    episodesTable['ColID'].append([t_ind[tmp_col[0]], t_ind[tmp_col[-1]]])
                    episodesTable['Frequency'].append(f_[tmp_col])
                    episodesTable['FrequencyMean'].append(np.mean(episodesTable['Frequency'][-1]))
                    episodesTable['Power'].append(a_[tmp_col])
                    episodesTable['PowerMean'].append(np.mean(episodesTable['Power'][-1]))
                    episodesTable['DurationS'].append(np.diff(episodesTable['ColID'][-1])[0] / cfg_eBOSC['fsample'])
                    episodesTable['DurationC'].append(episodesTable['DurationS'][-1] * episodesTable['FrequencyMean'][-1])
                    episodesTable['Trial'].append(cfg_eBOSC['tmp_trial'])
                    episodesTable['Channel'].append(cfg_eBOSC['tmp_channel']) 
                    episodesTable['Onset'].append(cfg_eBOSC['time.time_tfr'][int(episodesTable['ColID'][-1][0])])
                    episodesTable['Offset'].append(cfg_eBOSC['time.time_tfr'][int(episodesTable['ColID'][-1][-1])])
                    episodesTable['SNR'].append(episodes['SNR'][e][tmp_col])
                    episodesTable['SNRMean'].append(np.mean(episodesTable['SNR'][-1]))
                    # set all detected points to one in binary detected matrix
                    detected_new[episodesTable['RowID'][-1],t_ind[tmp_col]] = 1
                    
    # plt.imshow(detected_new, extent=[0, 1, 0, 1])
    # return post-processed episode dictionary and updated binary detected matrix
    return episodesTable, detected_new

def eBOSC_episode_postproc_maxbias(cfg_eBOSC, episodes, TFR):
    """
    % This function performs post-processing of input episodes by checking
    % whether 'detected' time points can be explained by the simulated extension of
    % the wavelet used in the time-frequency transform.
    %
    % Inputs: 
    %           cfg | config structure with cfg.eBOSC field
    %           episodes | table of episodes
    %           TFR | time-frequency matrix
    %
    % Outputs: 
    %           episodes_new | updated table of episodes
    %           detected_new | updated binary detected matrix
    
    % This method works as follows: we estimate the bias introduced by
    % wavelet convolution. The bias is represented by the amplitudes
    % estimated for the zero-shouldered signal (i.e. for which no real 
    % data was initially available). The influence of episodic
    % amplitudes on neighboring time points is assessed by scaling each
    % time point's amplitude with the last 'rhythmic simulated time
    % point', i.e. the first time wavelet amplitude in the simulated
    % rhythmic time points. At this time point the 'bias' is maximal,
    % although more precisely, this amplitude does not represent a
    % bias per se.
    """
    
    print("Applying maxbias post-processing ...")
    
    # re-initialize detected_new (for post-proc results)
    N_freq = TFR.shape[0]
    N_tp = TFR.shape[1]
    detected_new = np.zeros([N_freq, N_tp]);
    # initialize new dictionary to save results in
    # this is required as episodes may split, thus needing novel entries
    episodesTable = {}
    for entry in episodes:
        episodesTable[entry] = []
    
    # generate "bias" matrix
    # the logic here is as follows: we take a sinusoid, zero-pad it, and get the TFR
    # the bias is the tfr power produced for the padding (where power should be zero)
    B_bias = np.zeros([len(cfg_eBOSC['F']),len(cfg_eBOSC['F']),2*N_tp+1])
    amp_max = np.zeros([len(cfg_eBOSC['F']), len(cfg_eBOSC['F'])])
    for f in range(len(cfg_eBOSC['F'])):
        # temporary time vector and signal
        step_size = 1/cfg_eBOSC['fsample']
        time = np.arange(step_size, 1/cfg_eBOSC['F'][f]+step_size,step_size)
        tmp_sig = np.cos(time*2*np.pi*cfg_eBOSC['F'][f])*-1+1
        # signal for time-frequency analysis
        signal = np.concatenate((np.zeros([N_tp]), tmp_sig, np.zeros([N_tp])))
        [tmp_bias_mat, tmp_time, tmp_freq] = BOSC_tf(signal,cfg_eBOSC['F'],cfg_eBOSC['fsample'],cfg_eBOSC['wavenumber'])
        # bias matrix
        points_begin = np.arange(0,N_tp+1)
        points_end = np.arange(N_tp,B_bias.shape[2]+1)
        # for some reason, we have to transpose the matrix here, as the submatrix dimension order changes???
        B_bias[f,:,points_begin] = np.transpose(tmp_bias_mat[:,points_begin])
        B_bias[f,:,points_end] = np.transpose(np.fliplr(tmp_bias_mat[:,points_begin]))
        # maximum amplitude
        amp_max[f,:] = B_bias[f,:,:].max(1)
        # plt.imshow(amp_max, extent=[0, 1, 0, 1])

    # midpoint index
    ind_mid = N_tp+1
    # loop episodes
    for e in range(len(episodes['Trial'])):
        # get temporary frequency vector
        f_ = episodes['Frequency'][e]
        # get temporary amplitude vector
        a_ = episodes['Power'][e]
        m_ = np.zeros([len(a_),len(a_)])
        # location in time with reference to matrix TFR
        t_ind = np.arange(int(episodes['ColID'][e][0]),int(episodes['ColID'][e][-1]+1))
        # indices of time points' frequencies within "bias" matrix
        f_vec = episodes['RowID'][e]
        # figure; hold on;
        for tp in range(len(a_)):
            # index of current point's frequency within "bias" matrix
            ind_f = f_vec[tp]
            # get bias vector that varies with frequency of the
            # timepoints in the episode
            temporalBiasIndices = np.arange(ind_mid+1-tp,ind_mid+len(a_)-tp+1)
            ind1 = numpy.matlib.repmat(ind_f,len(f_vec),1)
            ind2 = np.reshape(f_vec,[-1,1])
            ind3 = np.reshape(temporalBiasIndices,[-1,1])
            indices = np.ravel_multi_index([ind1, ind2, ind3], 
                                           dims = B_bias.shape, order = 'C')
            tmp_biasVec = B_bias.flatten('C')[indices]
            # temporary "bias" vector (frequency-varying)
            if cfg_eBOSC['postproc.effSignal'] == 'all':
                tmp_bias = ((tmp_biasVec/np.reshape(amp_max[ind_f,f_vec],[-1,1]))*a_[tp])
            elif cfg_eBOSC['postproc.effSignal'] == 'PT':
                tmp_bias = ((tmp_biasVec/np.reshape(amp_max[ind_f,f_vec],[-1,1]))*
                            (a_[tp]-cfg_eBOSC['tmp.pt'][ind_f])) + cfg_eBOSC['tmp.pt'][ind_f]
            # compare to data
            m_[tp,:] = np.transpose(a_ >= tmp_bias)
            #plot(a_', 'k'); hold on; plot(tmp_bias, 'r');

        # identify which time points to retain and discard
        # Options: only correct at signal edge; correct within entire signal
        keep = m_.sum(0) == len(a_)
        if cfg_eBOSC['postproc.edgeOnly'] == 'yes':
            # keep everything that would be kept within the vector,
            # no removal within episode except for edges possible
            keepEdgeRemovalOnly = np.zeros([len(keep)],dtype=bool)
            keepEdgeRemovalOnly[np.arange(np.where(keep==1)[0][0],np.where(keep==1)[0][-1]+1)] = True
            keep = keepEdgeRemovalOnly
            del keepEdgeRemovalOnly

        # get new episodes
        keep = np.concatenate(([0], keep, [0]))
        d_keep = np.diff(keep.astype(float))
    
        if max(d_keep) == 1 and min(d_keep) == -1:
            # start and end indices
            ind_epsd_begin = np.where(d_keep == 1)[0]
            ind_epsd_end = np.where(d_keep == -1)[0]-1
            for i in range(len(ind_epsd_begin)):
                # check for passing the duration requirement
                # get average frequency
                tmp_col = np.arange(ind_epsd_begin[i],ind_epsd_end[i]+1)
                avg_frq = np.mean(f_[tmp_col])
                # match to closest frequency
                [tmp_a, indF] = find_nearest_value(cfg_eBOSC['F'], avg_frq)
                # check number of data points to fulfill number of cycles criterion
                num_pnt = np.floor((cfg_eBOSC['fsample']/ avg_frq) * int(np.reshape(cfg_eBOSC['threshold.duration'],[-1,1])[indF]))
                # if duration criterion remains fulfilled, encode in table
                if len(tmp_col) >= num_pnt:
                    # update all data in table with new episode limits
                    episodesTable['RowID'].append(episodes['RowID'][e][tmp_col])
                    episodesTable['ColID'].append([t_ind[tmp_col[0]], t_ind[tmp_col[-1]]])
                    episodesTable['Frequency'].append(f_[tmp_col])
                    episodesTable['FrequencyMean'].append(np.mean(episodesTable['Frequency'][-1]))
                    episodesTable['Power'].append(a_[tmp_col])
                    episodesTable['PowerMean'].append(np.mean(episodesTable['Power'][-1]))
                    episodesTable['DurationS'].append(np.diff(episodesTable['ColID'][-1])[0] / cfg_eBOSC['fsample'])
                    episodesTable['DurationC'].append(episodesTable['DurationS'][-1] * episodesTable['FrequencyMean'][-1])
                    episodesTable['Trial'].append(cfg_eBOSC['tmp_trial'])
                    episodesTable['Channel'].append(cfg_eBOSC['tmp_channel']) 
                    episodesTable['Onset'].append(cfg_eBOSC['time.time_tfr'][int(episodesTable['ColID'][-1][0])])
                    episodesTable['Offset'].append(cfg_eBOSC['time.time_tfr'][int(episodesTable['ColID'][-1][-1])])
                    episodesTable['SNR'].append(episodes['SNR'][e][tmp_col])
                    episodesTable['SNRMean'].append(np.mean(episodesTable['SNR'][-1]))
                    # set all detected points to one in binary detected matrix
                    detected_new[episodesTable['RowID'][-1],t_ind[tmp_col]] = 1
    # return post-processed episode dictionary and updated binary detected matrix
    return episodesTable, detected_new

def eBOSC_episode_rm_shoulder(cfg_eBOSC,detected1,episodes):
    """ Remove parts of the episode that fall into the 'shoulder' of individual
    trials. There is no check for adherence to a given duration criterion necessary,
    as the point of the padding of the detected matrix is exactly to account
    for allowing the presence of a few cycles.
    """

    print("Removing padding from detected episodes")

    ind1 = cfg_eBOSC['pad.detection_sample']
    ind2 = detected1.shape[1] - cfg_eBOSC['pad.detection_sample']
    rmv = []
    for j in range(len(episodes['Trial'])):
        # get time points of current episode
        tmp_col = np.arange(episodes['ColID'][j][0],episodes['ColID'][j][1]+1)
        # find time points that fall inside the padding (i.e. on- and offset)
        ex = np.where(np.logical_or(tmp_col < ind1, tmp_col >= ind2))[0]
        # remove padded time points from episodes
        tmp_col = np.delete(tmp_col, ex)
        episodes['RowID'][j] = np.delete(episodes['RowID'][j], ex)
        episodes['Power'][j] = np.delete(episodes['Power'][j], ex)
        episodes['Frequency'][j] = np.delete(episodes['Frequency'][j], ex)
        episodes['SNR'][j] = np.delete(episodes['SNR'][j], ex)
        # if nothing remains of episode: retain for later deletion
        if len(tmp_col)==0:
            rmv.append(j)
        else:
            # shift onset according to padding
            # Important: new col index is indexing w.r.t. to matrix AFTER
            # detected padding is removed!
            tmp_col = tmp_col - ind1
            episodes['ColID'][j] = [tmp_col[0], tmp_col[-1]]
            # re-compute mean frequency
            episodes['FrequencyMean'][j] = np.mean(episodes['Frequency'][j])
            # re-compute mean amplitude
            episodes['PowerMean'][j] = np.mean(episodes['Power'][j])
            # re-compute mean SNR
            episodes['SNRMean'][j] = np.mean(episodes['SNR'][j])
            # re-compute duration
            episodes['DurationS'][j] = np.diff(episodes['ColID'][j])[0] / cfg_eBOSC['fsample']
            episodes['DurationC'][j] = episodes['DurationS'][j] * episodes['FrequencyMean'][j]
            # update absolute on-/offsets (should remain the same)
            episodes['Onset'][j] = cfg_eBOSC['time.time_det'][int(episodes['ColID'][j][0])]
            episodes['Offset'][j] = cfg_eBOSC['time.time_det'][int(episodes['ColID'][j][-1])]
    # remove now empty episodes from table    
    for entry in episodes:
        # https://stackoverflow.com/questions/21032034/deleting-multiple-indexes-from-a-list-at-once-python
        episodes[entry] = [v for i, v in enumerate(episodes[entry]) if i not in rmv]
    return episodes

def eBOSC_episode_create(cfg_eBOSC,TFR,detected,eBOSC):
    """This function creates continuous rhythmic "episodes" and attempts to control for the impact of wavelet parameters.
      Time-frequency points that best represent neural rhythms are identified by
      heuristically removing temporal and frequency leakage. 
    
     Frequency leakage: at each frequency x time point, power has to exceed neighboring frequencies.
      Then it is checked whether the detected time-frequency points belong to
      a continuous episode for which (1) the frequency maximally changes by 
      +/- n steps (cfg.eBOSC.fstp) from on time point to the next and (2) that is at 
      least as long as n number of cycles (cfg.eBOSC.threshold.duration) of the average freqency
      of that episode (a priori duration threshold).
    
     Temporal leakage: The impact of the amplitude at each time point within a rhythmic episode on previous
      and following time points is tested with the goal to exclude supra-threshold time
      points that are due to the wavelet extension in time. 
    
    Input:   
               cfg         | config structure with cfg.eBOSC field
               TFR         | time-frequency matrix (excl. WLpadding)
               detected    | detected oscillations in TFR (based on power and duration threshold)
               eBOSC       | main eBOSC output structure; necessary to read in
                               prior eBOSC.episodes if they exist in a loop
    
    Output:  
               detected_new    | new detected matrix with frequency leakage removed
               episodesTable   | table with specific episode information:
                     Trial: trial index (corresponds to cfg.eBOSC.trial)
                     Channel: channel index
                     FrequencyMean: mean frequency of episode (Hz)
                     DurationS: episode duration (in sec)
                     DurationC: episode duration (in cycles, based on mean frequency)
                     PowerMean: mean amplitude of amplitude
                     Onset: episode onset in s
                     Offset: episode onset in s
                     Power: (cell) time-resolved wavelet-based amplitude estimates during episode
                     Frequency: (cell) time-resolved wavelet-based frequency
                     RowID: (cell) row index (frequency dimension): following eBOSC_episode_rm_shoulder relative to data excl. detection padding
                     ColID: (cell) column index (time dimension)
                     SNR: (cell) time-resolved signal-to-noise ratio: momentary amplitude/static background estimate at channel*frequency
                     SNRMean: mean signal-to-noise ratio
    """

    # initialize dictionary to save results in
    episodesTable = {}
    episodesTable['RowID'] = []
    episodesTable['ColID'] = []
    episodesTable['Frequency'] = []
    episodesTable['FrequencyMean'] = []
    episodesTable['Power'] = []
    episodesTable['PowerMean'] = []
    episodesTable['DurationS'] = []
    episodesTable['DurationC'] = []
    episodesTable['Trial'] = []
    episodesTable['Channel'] = []
    episodesTable['Onset'] = []
    episodesTable['Offset'] = []
    episodesTable['SNR'] = []
    episodesTable['SNRMean'] = []
    
    # %% Accounting for the frequency spread of the wavelet
    
    # Here, we compute the bandpass response as given by the wavelet
    # formula and apply half of the BP repsonse on top of the center frequency.
    # Because of log-scaling, the widths are not the same on both sides.
    
    detected = eBOSC_episode_sparsefreq(cfg_eBOSC, detected, TFR)    
    
    # %%  Create continuous rhythmic episodes
    
    # define step size in adjacency matrix
    cfg_eBOSC['fstp'] = 1
        
    # add zeros
    padding = np.zeros([cfg_eBOSC['fstp'],detected.shape[1]])
    detected_remaining = np.vstack([padding, detected, padding])
    detected_remaining[:,0] = 0
    detected_remaining[:,-1] = 0
    # detected_remaining serves as a dummy matrix; unless all entries from detected_remaining are
    # removed, we will continue extracting episodes
    tmp_B1 = np.vstack([padding, TFR*detected, padding])
    tmp_B1[:,0] = 0
    tmp_B1[:,-1] = 0
    detected_new = np.zeros(detected.shape)

    while sum(sum(detected_remaining)) > 0:
        # sampling point counter
        x = []
        y = []
        # find seed (remember that numpy uses row-first format!)
        # we need increasing x-axis sorting here
        [tmp_y,tmp_x] = np.where(np.matrix.transpose(detected_remaining)==1)
        x.append(tmp_x[0])
        y.append(tmp_y[0])
        # check next sampling point
        chck = 0
        while chck == 0:
            # next sampling point
            next_point = y[-1]+1
            next_freqs = np.arange(x[-1]-cfg_eBOSC['fstp'],
                          x[-1]+cfg_eBOSC['fstp']+1)
            tmp = np.where(detected_remaining[next_freqs,next_point]==1)[0]
            if tmp.size > 0:
                y.append(next_point)
                if tmp.size > 1:
                    # JQK 161017: It is possible that an episode is branching 
                    # two ways, hence we follow the 'strongest' branch; 
                    # Note that there is no correction for 1/f here, but 
                    # practically, it leads to satisfying results 
                    # (i.e. following the longer episodes).
                    tmp_data = tmp_B1[next_freqs,next_point]
                    tmp = np.where(tmp_data == max(tmp_data))[0]
                x.append(next_freqs[tmp[0]])
            else:
                chck = 1
            
        # check for passing the duration requirement
        # get average frequency
        avg_frq = np.mean(cfg_eBOSC['F'][np.array(x)-cfg_eBOSC['fstp']])
        # match to closest frequency
        [tmp_a, indF] = find_nearest_value(cfg_eBOSC['F'], avg_frq)
        # check number of data points to fulfill number of cycles criterion
        num_pnt = np.floor((cfg_eBOSC['fsample']/ avg_frq) * int(np.reshape(cfg_eBOSC['threshold.duration'],[-1,1])[indF]))
        if len(y) >= num_pnt:
            # %% encode episode that crosses duration threshold
            episodesTable['RowID'].append(np.array(x)-cfg_eBOSC['fstp'])
            episodesTable['ColID'].append([np.single(y[0]), np.single(y[-1])])
            episodesTable['Frequency'].append(np.single(cfg_eBOSC['F'][episodesTable['RowID'][-1]]))
            episodesTable['FrequencyMean'].append(np.single(avg_frq))
            tmp_x = episodesTable['RowID'][-1]
            tmp_y = np.arange(int(episodesTable['ColID'][-1][0]),int(episodesTable['ColID'][-1][1])+1)
            linIdx = np.ravel_multi_index([np.reshape(tmp_x,[-1,1]),
                                  np.reshape(tmp_y,[-1,1])], 
                                 dims=TFR.shape, order='C')
            episodesTable['Power'].append(np.single(TFR.flatten('C')[linIdx]))
            episodesTable['PowerMean'].append(np.mean(episodesTable['Power'][-1]))
            episodesTable['DurationS'].append(np.single(len(y)/cfg_eBOSC['fsample']))
            episodesTable['DurationC'].append(episodesTable['DurationS'][-1]*episodesTable['FrequencyMean'][-1])
            episodesTable['Trial'].append(cfg_eBOSC['tmp_trial']) # Note that the trial is non-zero-based
            episodesTable['Channel'].append(cfg_eBOSC['tmp_channel']) 
            # episode onset in absolute time
            episodesTable['Onset'].append(cfg_eBOSC['time.time_tfr'][int(episodesTable['ColID'][-1][0])]) 
            # episode offset in absolute time
            episodesTable['Offset'].append(cfg_eBOSC['time.time_tfr'][int(episodesTable['ColID'][-1][-1])]) 
            # extract (static) background power at frequencies
            episodesTable['SNR'].append(episodesTable['Power'][-1]/
                                 eBOSC['static.pt'].iloc[cfg_eBOSC['tmp_channelID'],
                                                         episodesTable['RowID'][-1]].values)
            episodesTable['SNRMean'].append(np.mean(episodesTable['SNR'][-1]))
            
            # remove processed segment from detected matrix
            detected_remaining[x,y] = 0
            # set all detected points to one in binary detected matrix
            rows = episodesTable['RowID'][-1]
            cols = np.arange(int(episodesTable['ColID'][-1][0]),
                                  int(episodesTable['ColID'][-1][1])+1)
            detected_new[rows,cols] = 1
        else:
            # %% remove episode from consideration due to being lower than duration
            detected_remaining[x,y] = 0
        
        # some sanity checks that episode selection was sensible
        #plt.imshow(detected, extent=[0, 1, 0, 1])
        #plt.imshow(detected_new, extent=[0, 1, 0, 1])
    
    # %%  Exclude temporal amplitude "leakage" due to wavelet smearing
    # temporarily pass on power threshold for easier access
    cfg_eBOSC['tmp.pt'] = eBOSC['static.pt'].loc[cfg_eBOSC['tmp_channel']].values
    
    # SQ note: This doesn't work too well so fuck it 
    # only do this if there are any episodes to fine-tune
    if cfg_eBOSC['postproc.use'] == 'yes' and len(episodesTable['Trial']) > 0:
        if cfg_eBOSC['postproc.method'] == 'FWHM':
            [episodesTable, detected_new] = eBOSC_episode_postproc_fwhm(cfg_eBOSC, episodesTable, TFR)
        elif cfg_eBOSC['postproc.method'] == 'MaxBias':
            [episodesTable, detected_new] = eBOSC_episode_postproc_maxbias(cfg_eBOSC, episodesTable, TFR)
        
    # %% remove episodes and part of episodes that fall into 'shoulder'
    
    if len(episodesTable['Trial']) > 0 and cfg_eBOSC['pad.detection_sample']>0:
        episodesTable = eBOSC_episode_rm_shoulder(cfg_eBOSC,detected_new,episodesTable)
    
    # %% if an episode list already exists, append results
    
    if 'episodes' in eBOSC:
        # initialize dictionary entries if not existing
        if not len(eBOSC['episodes']):
            for entry in episodesTable:
                eBOSC['episodes'][entry] = [] 
        # append current results
        for entry in episodesTable:
            episodesTable[entry] = eBOSC['episodes'][entry] + episodesTable[entry]
        
    return episodesTable, detected_new

def eBOSC_wrapper(cfg_eBOSC, data):
    """Main eBOSC wrapper function. Executes eBOSC subfunctions.
    Inputs: 
        cfg_eBOSC | dictionary containing the following entries:
            F                     | frequency sampling
            wavenumber            | wavelet family parameter (time-frequency tradeoff)
            fsample               | current sampling frequency of EEG data
            pad.tfr_s             | padding following wavelet transform to avoid edge artifacts in seconds (bi-lateral)
            pad.detection_s       | padding following rhythm detection in seconds (bi-lateral); 'shoulder' for BOSC eBOSC.detected matrix to account for duration threshold
            pad.total_s           | complete padding (WL + shoulder)
            pad.background_s      | padding of segments for BG (only avoiding edge artifacts)
            threshold.excludePeak | lower and upper bound of frequencies to be excluded during background fit (Hz) (previously: LowFreqExcludeBG HighFreqExcludeBG)
            threshold.duration    | vector of duration thresholds at each frequency (previously: ncyc)
            threshold.percentile  | percentile of background fit for power threshold
            postproc.use          | Post-processing of rhythmic eBOSC.episodes, i.e., wavelet 'deconvolution' (default = 'no')
            postproc.method       | Deconvolution method (default = 'MaxBias', FWHM: 'FWHM')
            postproc.edgeOnly     | Deconvolution only at on- and offsets of eBOSC.episodes? (default = 'yes')
            postproc.effSignal	  | Power deconvolution on whole signal or signal above power threshold? (default = 'PT')
            channel               | Subset of channels? (default: [] = all)
            trial                 | Subset of trials? (default: [] = all)
            trial_background      | Subset of trials for background? (default: [] = all)
        data | input time series data as a Pandas DataFrame: 
            - channels as columns
            - multiindex containing: 'time', 'epoch', 
    Outputs: 
        eBOSC | main eBOSC output dictionary containing the following entries:
            episodes | Dictionary: individual rhythmic episodes (see eBOSC_episode_create)
            detected | DataFrame: binary detected time-frequency points (prior to episode creation), pepisode = temporal average
            detected_ep | DataFrame: binary detected time-frequency points (following episode creation), abundance = temporal average
            cfg | config structure (see input)
    """

    # %% get list of channel names (very manual solution, replace if possible)

    channelNames = list(data.columns.values)
    channelNames.remove('time')
    channelNames.remove('condition')
    channelNames.remove('epoch')

    # %% define some defaults for included channels and trials, if not specified
    
    if not cfg_eBOSC['channel']:
        cfg_eBOSC['channel'] = channelNames # list of channel names
    
    if not cfg_eBOSC['trial']:
        # remember to count trial 1 as zero
        cfg_eBOSC['trial'] = list(np.arange(0,len(pd.unique(data['epoch']))))
    # else: # this ensures the zero count
    #     cfg_eBOSC['trial'] = list(np.array(cfg_eBOSC['trial']))
        
    if not cfg_eBOSC['trial_background']:
        cfg_eBOSC['trial_background'] = list(np.arange(0,len(pd.unique(data['epoch']))))
    # else: # this ensures the zero count
    #     cfg_eBOSC['trial_background'] = list(np.array(cfg_eBOSC['trial_background']) - 1)

    # %% calculate the sample points for paddding
    
    cfg_eBOSC['pad.tfr_sample'] = int(cfg_eBOSC['pad.tfr_s'] * cfg_eBOSC['fsample'])
    cfg_eBOSC['pad.detection_sample'] = int(cfg_eBOSC['pad.detection_s'] * cfg_eBOSC['fsample'])
    cfg_eBOSC['pad.total_s'] = cfg_eBOSC['pad.tfr_s'] + cfg_eBOSC['pad.detection_s']
    cfg_eBOSC['pad.total_sample'] = int(cfg_eBOSC['pad.tfr_sample'] + cfg_eBOSC['pad.detection_sample'])
    cfg_eBOSC['pad.background_sample'] = int(cfg_eBOSC['pad.tfr_sample'])
    
    # %% calculate time vectors (necessary for preallocating data frames)
    
    n_trial = len(cfg_eBOSC['trial'])
    n_freq = len(cfg_eBOSC['F'])
    n_time_total = len(pd.unique(data.loc[data['epoch']==0, ('time')]))
    # copy potentially non-continuous time values (assume that epoch is labeled 0)
    cfg_eBOSC['time.time_total'] = data.loc[data['epoch']==0, ('time')].values
    # alternatively: create a new time vector that is non-continuous and starts at zero
    # np.arange(0, 1/cfg_eBOSC['fsample']*(n_time_total) , 1/cfg_eBOSC['fsample'])
    # get timing and info for post-TFR padding removal
    tfr_time2extract = np.arange(cfg_eBOSC['pad.tfr_sample'], n_time_total-cfg_eBOSC['pad.tfr_sample'],1)
    cfg_eBOSC['time.time_tfr'] = cfg_eBOSC['time.time_total'][tfr_time2extract]
    n_time_tfr = len(cfg_eBOSC['time.time_tfr'])
    # get timing and info for post-detected padding removal
    det_time2extract = np.arange(cfg_eBOSC['pad.detection_sample'], n_time_tfr-cfg_eBOSC['pad.detection_sample'],1)
    cfg_eBOSC['time.time_det'] = cfg_eBOSC['time.time_tfr'][det_time2extract]
    n_time_det = len(cfg_eBOSC['time.time_det'])
        
    # %% preallocate data frames

    eBOSC = {}
    eBOSC['static.bg_pow'] = pd.DataFrame(columns=cfg_eBOSC['F'])
    eBOSC['static.bg_log10_pow'] = pd.DataFrame(columns=cfg_eBOSC['F'])    
    eBOSC['static.pv'] = pd.DataFrame(columns=['slope', 'intercept'])
    eBOSC['static.mp'] = pd.DataFrame(columns=cfg_eBOSC['F'])    
    eBOSC['static.pt'] = pd.DataFrame(columns=cfg_eBOSC['F'])   
    
    # Multiindex for channel x trial x frequency x time
    arrays = np.array([cfg_eBOSC['channel'],cfg_eBOSC['trial'],cfg_eBOSC['F'], cfg_eBOSC['time.time_det']],dtype=object)
    #tuples = list(zip(*arrays))
    names=["channel", "trial", "frequency", "time"]
    index=pd.MultiIndex.from_product(arrays,names=names)
    nullData=np.zeros(len(arrays[0]) * len(arrays[1]) * len(arrays[2]) * len(arrays[3]) )
    eBOSC['detected'] = pd.DataFrame(data=nullData, index=index)
    eBOSC['detected_ep'] = eBOSC['detected'].copy()
    del nullData, index
    
    eBOSC['episodes'] = {}

    # %% main eBOSC loop
    
    for channel in cfg_eBOSC['channel']:
        print('Channel: ' + channel + '; Nr. ' + str(cfg_eBOSC['channel'].index(channel)+1) + '/' + str(len(cfg_eBOSC['channel'])))
        cfg_eBOSC['tmp_channelID'] = cfg_eBOSC['channel'].index(channel)
        cfg_eBOSC['tmp_channel'] = channel
                
        # %% Step 1: time-frequency wavelet decomposition for whole signal to prepare background fit
        n_trial = len(cfg_eBOSC['trial'])
        n_freq = len(cfg_eBOSC['F'])
        n_time = len(pd.unique(data.loc[data['epoch']==0, ('time')]))
        TFR = np.zeros((n_trial, n_freq, n_time))
        TFR[:] = np.nan
        for trial in cfg_eBOSC['trial']:
            eegsignal = data.loc[data['epoch']==trial, (channel)]
            F = cfg_eBOSC['F']
            Fsample = cfg_eBOSC['fsample']
            wavenumber = cfg_eBOSC['wavenumber']
            [TFR[trial,:,:], tmp, tmp] = BOSC_tf(eegsignal,F,Fsample,wavenumber)
            del eegsignal,F,Fsample,wavenumber,tmp
            
        # %% plot example time-frequency spectrograms (only for intuition/debugging) 
        # assumes that multiple trials are present
        # plt.imshow(TFR[0,:,:], extent=[0, 1, 0, 1])
        # plt.imshow(TFR[:,:,:].mean(axis=0), extent=[0, 1, 0, 1])
        # plt.imshow(TFR[:,:,:].mean(axis=1), extent=[0, 1, 0, 1])
        # plt.imshow(TFR[:,:,:].mean(axis=2), extent=[0, 1, 0, 1])
                
        # %% Step 2: robust background power fit (see 2020 NeuroImage paper)
       
        [eBOSC, pt, dt] = eBOSC_getThresholds(cfg_eBOSC, TFR, eBOSC)
         
        # %% application of thresholds to single trials

        for trial in cfg_eBOSC['trial']:
            # print('Trial Nr. ' + str(trial+1) + '/' + str(len(cfg_eBOSC['trial'])))
            # encode current trial ID for later
            cfg_eBOSC['tmp_trialID'] = trial
            # trial ID in the intuitive convention
            cfg_eBOSC['tmp_trial'] = cfg_eBOSC['trial'].index(trial)+1

            # get wavelet transform for single trial
            # tfr padding is removed to avoid edge artifacts from the wavelet
            # transform. Note that a padding fpr detection remains attached so that there
            # is no problems with too few sample points at the edges to
            # fulfill the duration criterion.         
            time2extract = np.arange(cfg_eBOSC['pad.tfr_sample'], TFR.shape[2]-cfg_eBOSC['pad.tfr_sample'],1)
            TFR_ = np.transpose(TFR[trial,:,time2extract],[1,0])
            
            # %% Step 3: detect rhythms and calculate Pepisode
            # The next section applies both the power and the duration
            # threshold to detect individual rhythmic segments in the continuous signals.
            detected = np.zeros((TFR_.shape))
            for f in range(len(cfg_eBOSC['F'])):
                detected[f,:] = BOSC_detect(TFR_[f,:],pt[f],dt[f][0],cfg_eBOSC['fsample'])

            # remove padding for detection (matrix with padding required for refinement)
            time2encode = np.arange(cfg_eBOSC['pad.detection_sample'], detected.shape[1]-cfg_eBOSC['pad.detection_sample'],1)
            eBOSC['detected'].loc[(channel, trial)] = np.reshape(detected[:,time2encode],[-1,1])
            
            # %% Step 4 (optional): create table of separate rhythmic episodes
            [episodes, detected_ep] = eBOSC_episode_create(cfg_eBOSC,TFR_,detected,eBOSC)
            # insert detected episodes into episode structure
            eBOSC['episodes'] = episodes
            
            # remove padding for detection (already done for eBOSC.episodes)
            time2encode = np.arange(cfg_eBOSC['pad.detection_sample'], detected_ep.shape[1]-cfg_eBOSC['pad.detection_sample'],1)
            eBOSC['detected_ep'].loc[(channel, trial)] = np.reshape(detected_ep[:,time2encode],[-1,1])

            # %% Supplementary Plot: original eBOSC.detected vs. sparse episode power
            # import matplotlib.pyplot as plt
            # fig, axes = plt.subplots(nrows=2, ncols=1)
            # detected_cur = eBOSC['detected_ep'].loc[(channel, trial)]
            # detected_cur = detected_cur.pivot_table(index=['frequency'], columns='time')
            # curPlot = detected_cur*TFR_[:,time2encode]
            # axes[0].imshow(curPlot, aspect='auto', vmin = 0, vmax = 1)
            # detected_cur = eBOSC['detected'].loc[(channel, trial)]
            # detected_cur = detected_cur.pivot_table(index=['frequency'], columns='time')
            # curPlot = detected_cur*TFR_[:,time2encode]
            # axes[1].imshow(curPlot, aspect='auto', vmin = 0, vmax = 1)

    # %% return dictionaries back to caller script
    return eBOSC, cfg_eBOSC


def compute_eBOSC_parallel(chan_name, MNE_object, subj_id, elec_df, event_name, ev_dict, conditions, 
                           do_plot=False, save_path='/sc/arion/projects/guLab/Salman/EphysAnalyses', 
                           do_save=False, mean_across_time=False, mean_across_freqs=False, both_dfs=True, **kwargs):
    """

    This function is meant to parallelize our BOSC code to be computed over many channels simultaneously and save the results 
    to individual dataframes. 

    """

    if not os.path.exists(f'{save_path}/{subj_id}/scratch/eBOSC/{event_name}/dfs'):
        os.makedirs(f'{save_path}/{subj_id}/scratch/eBOSC/{event_name}/dfs')

    data_df = MNE_object.copy().pick_channels([chan_name]).to_data_frame(time_format=None)

    # parameters for eBOSC
    cfg_eBOSC = kwargs
    cfg_eBOSC['channel'] = [chan_name]

    # Compute BOSC: 
    [eBOSC, cfg] = eBOSC_wrapper(cfg_eBOSC, data_df)

    # Cut off buffer time
    if ev_dict[event_name][0] < 0:
        eBOSC['detected'] = eBOSC['detected'].query(f'(time>=0) & (time<={ev_dict[event_name][1]})')

    eBOSC['detected'] = eBOSC['detected'].reset_index().rename(columns={0:'prop_detect'})

    # Update: Let's actually do this AFTER loading the saved BOSC results so we are not being redundant. 
    # # Add events to the BOSC data:  
    # event_df['trial'] = eBOSC['detected']['trial'].unique()
    # eBOSC['detected'] = eBOSC['detected'].merge(event_df, on=['trial'])

    # identify frequency bands 
    eBOSC['detected']['fband'] = eBOSC['detected'].frequency.apply(lambda x: 'theta' if x<10 else 'alpha' if (x>=10) & (x<14) else 'beta' if (x>=14) & (x<30) else 'slowgamma' if (x>=30) & (x<55) else 'hfa')

    # # get rid of all the annoying line messages
    # clear_output(wait=True)

    # Dataframe for saving
    if both_dfs:
        time_averaged_df = pd.DataFrame(eBOSC['detected'].groupby(['trial', 'frequency']).mean()).reset_index().drop(columns=['time'])
        time_averaged_df.insert(0,'channel', chan_name)
        time_averaged_df.insert(0, 'region', elec_df[elec_df.label==chan_name].salman_region.values[0])
        time_averaged_df.insert(0,'subj', subj_id)
        time_averaged_df['event'] = event_name    
        if do_save: 
            time_averaged_df.to_csv(f'{save_path}/{subj_id}/scratch/eBOSC/{event_name}/dfs/{chan_name}_time_averaged_df.csv', index=False)

        time_resolved_df = eBOSC['detected'].groupby(['trial', 'fband', 'time']).mean().reset_index().drop(columns=['frequency'])
        if do_save: 
            time_resolved_df.to_csv(f'{save_path}/{subj_id}/scratch/eBOSC/{event_name}/dfs/{chan_name}_time_resolved_df.csv', index=False)
    
    if mean_across_time:
        time_averaged_df = pd.DataFrame(eBOSC['detected'].groupby(['trial', 'frequency']).mean()).reset_index().drop(columns=['time'])
        time_averaged_df.insert(0,'channel', chan_name)
        time_averaged_df.insert(0, 'region', elec_df[elec_df.label==chan_name].salman_region.values[0])
        time_averaged_df.insert(0,'subj', subj_id)
        time_averaged_df['event'] = event_name
        if do_save: 
            time_averaged_df.to_csv(f'{save_path}/{subj_id}/scratch/eBOSC/{event_name}/dfs/{chan_name}_time_averaged_df.csv', index=False)
    elif mean_across_freqs: 
        # Average across frequencies within a band, rename some columns 
        time_resolved_df = eBOSC['detected'].groupby(['trial', 'fband', 'time']).mean().reset_index().drop(columns=['frequency'])
        if do_save: 
            time_resolved_df.to_csv(f'{save_path}/{subj_id}/scratch/eBOSC/{event_name}/dfs/{chan_name}_time_resolved_df.csv', index=False)
    
#     THE FOLLOWING CODE APPLIE TO TFR parallelized code as well!! 
#     if do_plot:
#         # If we want to plot we need the event data and the condition information
#         fig, ax = plt.subplots(nrows=1, ncols=2, figsize=[10,5], sharex=True, sharey=True)
#         for ix, cond in enumerate(conditions): 
#             # Plot: 
#             detected_avg = pd.DataFrame(eBOSC['detected'].query(cond).groupby(['frequency', 'time']).mean().drop(columns=['trial'])['prop_detect'])

#             # eBOSC['detected'].groupby(level=['frequency', 'time']).mean()
#             detected_avg = detected_avg.pivot_table(index=['frequency'], columns='time')
#             cur_multiindex = eBOSC['detected'].index
#             cur_time = eBOSC['detected']z.time.unique()
#             # cur_multiindex.get_level_values('time').unique()
#             cur_freq = eBOSC['detected'].frequency.unique()
#             # cur_multiindex.get_level_values('frequency').unique()

            
# #                 ax.vlines(250, 0, len(cfg_eBOSC['F']), 'white')
#             im = ax[ix].imshow(detected_avg, aspect = 'auto', interpolation='bicubic', cmap='rocket', vmin=0, vmax=.4)

#             [x0, x1] = ax[ix].get_xlim()
#             [y0, y1] = ax[ix].get_ylim()
#             xticks_loc = np.linspace(0,750, 4)
#             # [t for t in ax.get_xticks() if t>=x0 and t<=x1]
#             yticks_loc = [t for t in ax[ix].get_yticks() if t>=y1 and t<=y0]
#             x_label_list = np.round(cur_time[np.int_(xticks_loc)],1).tolist()
#             y_label_list = np.round(cur_freq[np.int_(yticks_loc)],1).tolist()
#             ax[ix].set_xticks(xticks_loc)
#             ax[ix].set_xticklabels(x_label_list)
#             ax[ix].set_yticks(yticks_loc)
#             ax[ix].set_yticklabels(y_label_list)
#             ax[ix].invert_yaxis()
#             ax[ix].set_xlabel('Time [s]')
#             ax[ix].set_ylabel('Frequency [Hz]') 
#             ax[ix].set_title(f'{cond}')
#             fig.colorbar(im, ax=ax[ix])
#         plt.suptitle('Avg. detected rhythms across trials', fontsize=12)
#         plt.tight_layout()
#         plt.savefig(f'{save_path}/{subj_id}/scratch/eBOSC/{event_name}/plots/{chan_name}_eBOSC.pdf', dpi=100)
#         plt.close()

# # USAGE example from: https://github.com/jkosciessa/eBOSC_py/blob/main/examples/eBOSC_example_empirical.ipynb
# pn = dict()
# pn['root']  = os.path.join(os.getcwd(),'..')
# pn['examplefile'] = os.path.join(pn['root'],'data','1160_rest_EEG_Rlm_Fhl_rdSeg_Art_EC.csv')
# pn['outfile'] = os.path.join(pn['root'],'data','example_out.npy')

# cfg_eBOSC = dict()
# cfg_eBOSC['F'] = 2 ** np.arange(1,6,.125)   # frequency sampling
# cfg_eBOSC['wavenumber'] = 6                 # wavelet parameter (time-frequency tradeoff)
# cfg_eBOSC['fsample'] = 500                  # current sampling frequency of EEG data
# cfg_eBOSC['pad.tfr_s'] = 1                  # padding following wavelet transform to avoid edge artifacts in seconds (bi-lateral)
# cfg_eBOSC['pad.detection_s'] = .5           # padding following rhythm detection in seconds (bi-lateral); 'shoulder' for BOSC eBOSC.detected matrix to account for duration threshold
# cfg_eBOSC['pad.background_s'] = 1           # padding of segments for BG (only avoiding edge artifacts)

# cfg_eBOSC['threshold.excludePeak'] = np.array([[8,15]])   # lower and upper bound of frequencies to be excluded during background fit (Hz) (previously: LowFreqExcludeBG HighFreqExcludeBG)
# cfg_eBOSC['threshold.duration'] = np.kron(np.ones((1,len(cfg_eBOSC['F']))),3) # vector of duration thresholds at each frequency (previously: ncyc)
# cfg_eBOSC['threshold.percentile'] = .95    # percentile of background fit for power threshold

# cfg_eBOSC['postproc.use'] = 'yes'           # Post-processing of rhythmic eBOSC.episodes, i.e., wavelet 'deconvolution' (default = 'no')
# cfg_eBOSC['postproc.method'] = 'FWHM'       # Deconvolution method (default = 'MaxBias', FWHM: 'FWHM')
# cfg_eBOSC['postproc.edgeOnly'] = 'yes'      # Deconvolution only at on- and offsets of eBOSC.episodes? (default = 'yes')
# cfg_eBOSC['postproc.effSignal'] = 'PT'      # Power deconvolution on whole signal or signal above power threshold

# cfg_eBOSC['channel'] = ['Oz']            # select posterior channels (default: all)
# cfg_eBOSC['trial'] = []                  # select trials (default: all, indicate in natural trial number (not zero-starting))
# cfg_eBOSC['trial_background'] = []       # select trials for background (default: all, indicate in natural trial

# # Either concatenate all epochs or use with continuous data: 
# [eBOSC, cfg] = eBOSC_wrapper(cfg_eBOSC, data)

# # Plot: 
# detected_avg = eBOSC['detected'].mean(level=['frequency', 'time'])
# detected_avg = detected_avg.pivot_table(index=['frequency'], columns='time')
# cur_multiindex = eBOSC['detected'].index
# cur_time = cur_multiindex.get_level_values('time').unique()
# cur_freq = cur_multiindex.get_level_values('frequency').unique()

# fig, ax = plt.subplots(nrows=1, ncols=1)
# im = ax.imshow(detected_avg, aspect = 'auto')
# [x0, x1] = ax.get_xlim()
# [y0, y1] = ax.get_ylim()
# xticks_loc = [t for t in ax.get_xticks() if t>=x0 and t<=x1]
# yticks_loc = [t for t in ax.get_yticks() if t>=y1 and t<=y0]
# x_label_list = np.round(cur_time[np.int_(xticks_loc)],1).tolist()
# y_label_list = np.round(cur_freq[np.int_(yticks_loc)],1).tolist()
# ax.set_xticks(xticks_loc)
# ax.set_xticklabels(x_label_list)
# ax.set_yticks(yticks_loc)
# ax.set_yticklabels(y_label_list)
# plt.colorbar(im, label='Proportion detected across trials')
# plt.xlabel('Time [s]')
# plt.ylabel('Frequency [Hz]')
# plt.title('Avg. detected rhythms across trials', fontsize=12)
# plt.show()