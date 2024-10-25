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
    
    decision_trial_start, non_decision_trial_start, choice_start, rt, key_pressed, photodiode = [], [], [], [], [], []
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
                    photodiode.append('white.png')
                elif 'blank.png' in tokens[2]:
                    # non-decision trial start
                    current_trial_type = 'narration'
                    decision_trial_start.append(None)
                    non_decision_trial_start.append(timestamp)
                    choice_start.append(None)
                    rt.append(None)
                    key_pressed.append(None)
                    photodiode.append('blank.png')

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

                # if not a decision trial, record the key pressed for non-decision trials
                elif current_trial_type == 'narration':
                    key_pressed[-1] = key_info

    parsed_data = pd.DataFrame({
        'trial': range(1, len(decision_trial_start) + 1),
        'non_decision_trial_start': non_decision_trial_start,
        'decision_trial_start': decision_trial_start,
        'choice_start': choice_start,
        'rt': rt,
        'key_pressed': key_pressed,
        'photodiode': photodiode
    })

    # rounding?
    parsed_data = parsed_data.apply(lambda x: x.round(4) if x.dtype.kind in 'fc' else x)

    return parsed_data

def synchronize(beh_ts, photodiode_data, subj_id, smoothSize=15, windSize=15, height=0.5, plot_alignment=True):
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
        sig_indices = [index for index,value in enumerate(timestamp) if value > 0 and value < 2000]
        neu_indices = [index for index,value in enumerate(neural_ts) if value > 0 and value < 2000]

        plt.plot(timestamp[sig_indices], sig[sig_indices])
        plt.plot(neural_ts[neu_indices], np.ones_like(neural_ts[neu_indices])+1, 'o', markersize=3)
        plt.title("Photodiode " + subj_id)

    return slope, offset