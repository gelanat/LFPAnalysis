import pandas as pd 
import numpy as np
import scipy
import matplotlib.pyplot as plt
from LFPAnalysis_GT import sync_utils

def parse_logfile(logfile_path):
    """
    parses a psychopy .log file 
    args:
    log_file_path (str): path to the .log file.

    returns:
    pd.dataframe
    """
    # initialize lists to hold parsed data
    decision_trial_start, non_decision_trial_start, choice_start, rt, key_pressed, photodiode_status, space_press_time = [], [], [], [], [], [], []
    
    # placeholders for tracking events
    last_decision_start = None
    current_trial_type = None

    # read and parse the log file line-by-line
    with open(logfile_path, 'r') as file:
        for line in file:
            line = line.strip()
            tokens = line.split('\t')
            if len(tokens) < 2:
                continue  # skip lines that are too short

            timestamp = float(tokens[0])
            event_type = tokens[1]

            # check for photodiode events
            if event_type == 'EXP ' and tokens[2].startswith('photodiode'):
                if 'white.png' in tokens[2]:
                    # decision trial start
                    current_trial_type = 'decision'
                    last_decision_start = timestamp
                    decision_trial_start.append(timestamp)
                    non_decision_trial_start.append(None)
                    choice_start.append(None)  # placeholder
                    rt.append(0)  # initialize with 0 for no response case
                    key_pressed.append(None)
                    photodiode_status.append('white.png')
                    space_press_time.append(None)
                elif 'blank.png' in tokens[2]:
                    # non-decision trial start
                    current_trial_type = 'narration'
                    decision_trial_start.append(None)
                    non_decision_trial_start.append(timestamp)
                    choice_start.append(None)
                    rt.append(None)
                    key_pressed.append(None)
                    photodiode_status.append('blank.png')
                    space_press_time.append(None)

            # check for the first valid key press (1 or 2) after a decision trial start to calculate reaction time
            elif event_type == 'DATA ' and 'Keypress: ' in tokens[2]:
                key_info = tokens[2].split('Keypress: ')[1]

                # if this is a decision trial and the key is valid (1 or 2), calculate rt and choice start
                if current_trial_type == 'decision' and last_decision_start is not None and key_info in ['1', '2']:
                    reaction_time = timestamp - last_decision_start
                    rt[-1] = reaction_time  # update reaction time for this trial
                    choice_start[-1] = last_decision_start + reaction_time  # calculate choice start
                    key_pressed[-1] = key_info  # capture key pressed
                    last_decision_start = None  # reset after capturing first valid response

                # if it's a non-decision trial and the key is "space," record the timestamp
                elif current_trial_type == 'narration' and key_info == 'space':
                    space_press_time[-1] = timestamp

    # build the dataframe with each row aligned to one trial (either decision or narration)
    parsed_data = pd.DataFrame({
        'trial': range(1, len(decision_trial_start) + 1),
        'non_decision_trial_start': non_decision_trial_start,
        'space_press_time': space_press_time,
        'decision_trial_start': decision_trial_start,
        'choice_start': choice_start,
        'rt': rt,
        'key_pressed': key_pressed,
        'photodiode': photodiode_status
    })

    # rounding?
    parsed_data = parsed_data.apply(lambda x: x.round(4) if x.dtype.kind in 'fc' else x)

    return parsed_data


def synchronize(beh_ts, photodiode_data, subj_id, smoothSize=15, windSize=15, height=0.5, plot_alignment=True, plot_segment=500):
    preproc = '/sc/arion/projects/OlfMem/tostag01/SocialNav/preproc/'
    preproc_dir = f'{preproc}{subj_id}'
    sig = np.squeeze(sync_utils.moving_average(photodiode_data._data, n=smoothSize))
    timestamp = np.squeeze(np.arange(len(sig))/photodiode_data.info['sfreq'])
    sig = scipy.stats.zscore(sig)
    trig_ix = np.where((sig[:-1]<=height)*(sig[1:]>height))[0] # rising edge of trigger

    neural_ts = timestamp[trig_ix]
    neural_ts = np.array(neural_ts)
    print(f'There are {len(neural_ts)} neural syncs detected')
    
    nwin = len(neural_ts) - len(beh_ts)
    rvals = []
    slopes = [] 
    offsets = []
    for i in range(nwin+1):
        slope, offset, rval = sync_utils.sync_matched_pulses(np.array(beh_ts), neural_ts[i:len(beh_ts)+i])
        rvals.append(rval)
        slopes.append(slope)
        offsets.append(offset)
    rvals = np.array(rvals)
    offsets = np.array(offsets)
    slopes = np.array(slopes)
    
    slope=slopes[np.argmax(rvals)]
    offset=offsets[np.argmax(rvals)]
    print(f'Max rval with slope of {slope} and offset of {offset}')

    if plot_alignment:
        sig_indices = [index for index,value in enumerate(timestamp) if value > 0 and value < plot_segment]
        neu_indices = [index for index,value in enumerate(neural_ts) if value > 0 and value < plot_segment]
        plt.figure()
        plt.plot(timestamp[sig_indices], sig[sig_indices])
        plt.plot(neural_ts[neu_indices], np.ones_like(neural_ts[neu_indices]), 'o', markersize=3)
        plt.title("Photodiode " + subj_id)
        plt.savefig(f'{preproc_dir}/{subj_id}_photodiode_alignment.png', dpi=300)

    return slope, offset


def visualize_photodiode_and_behavior(
    photodiode_data, 
    time_df, 
    subj_id, 
    slope, 
    offset, 
    smoothSize=15, 
    height=0.5, 
    plot_save_path=None, 
    segment=None
):
    """
    visualizes the photodiode signal alongside behavioral timestamps with slope and offset correction.

    args:
        photodiode_data (raw): mne raw object containing the photodiode signal.
        time_df (dataframe): parsed logfile containing behavioral timestamps.
        subj_id (str): subject identifier.
        slope (float): slope from synchronization.
        offset (float): offset from synchronization.
        smoothSize (int): window size for moving average smoothing.
        height (float): threshold for detecting rising edges.
        plot_save_path (str): path to save the plot (optional).
        segment (tuple): start and end times in seconds (e.g., (0, 100)). visualizes the full signal if none.
    """
    # get photodiode signal and timestamps
    sfreq = photodiode_data.info['sfreq']
    raw_signal = photodiode_data.get_data().squeeze()
    smoothed_signal = sync_utils.moving_average(raw_signal, n=smoothSize)
    smoothed_signal = scipy.stats.zscore(smoothed_signal)
    timestamps = np.arange(len(smoothed_signal)) / sfreq

    # apply segment filtering if specified
    if segment:
        start_idx = int(segment[0] * sfreq)
        end_idx = int(segment[1] * sfreq)
        timestamps = timestamps[start_idx:end_idx]
        smoothed_signal = smoothed_signal[start_idx:end_idx]

    # correct behavioral timestamps with slope and offset
    time_df['corrected_decision_trial_start'] = time_df['decision_trial_start'] * slope + offset
    time_df['corrected_choice_start'] = time_df['choice_start'] * slope + offset

    # detect rising edges in the specified segment
    rising_edges = np.where((smoothed_signal[:-1] <= height) & (smoothed_signal[1:] > height))[0]
    photodiode_events = timestamps[rising_edges]

    # plot photodiode signal with annotations
    plt.figure(figsize=(15, 6))
    plt.plot(timestamps, smoothed_signal, label='photodiode signal', alpha=0.8)
    plt.scatter(photodiode_events, [height] * len(photodiode_events), color='r', label='rising edges', zorder=5)
    for _, row in time_df.iterrows():
        if row['corrected_decision_trial_start'] is not None and segment and segment[0] <= row['corrected_decision_trial_start'] <= segment[1]:
            plt.axvline(row['corrected_decision_trial_start'], color='g', linestyle='--', label='corrected decision start' if 'corrected decision start' not in plt.gca().get_legend_handles_labels()[1] else "")
        if row['corrected_choice_start'] is not None and segment and segment[0] <= row['corrected_choice_start'] <= segment[1]:
            plt.axvline(row['corrected_choice_start'], color='b', linestyle='-.', label='corrected button press' if 'corrected button press' not in plt.gca().get_legend_handles_labels()[1] else "")

    plt.xlabel('time (s)')
    plt.ylabel('z-scored signal')
    plt.title(f'photodiode signal and corrected behavioral events ({subj_id}) - segment {segment if segment else "full"}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # save or show the plot
    if plot_save_path:
        plt.savefig(plot_save_path, dpi=300)
    plt.show()
