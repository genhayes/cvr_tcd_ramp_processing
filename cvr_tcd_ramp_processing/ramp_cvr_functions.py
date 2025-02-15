# G. Hayes 2024
# This file contains the functions used to process the data for the ramp cerebrovascular reactivity (CVR) analysis in:
# G. Hayes, S. Sparks, J. Pinto, and D. P. Bulte, “Ramp protocol for non-linear cerebrovascular reactivity with transcranial doppler ultrasound,” Journal of Neuroscience Methods, vol. 416, p. 110381, Apr. 2025, doi: 10.1016/j.jneumeth.2025.110381.

import numpy as np
import pandas as pd
import os,sys
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.signal import find_peaks
import copy
from numpy.lib.stride_tricks import sliding_window_view as swv
from scipy.stats import zscore

def get_endtidal_peaks(data, data_init_index, search_window, sample_rate=200, height=0, prominence=3):
    '''peaks_inds, peaks_vals, peaks_df = get_endtidal_peaks(data, data_init_index, search_window=6, sample_rate=200, height=0, prominence=3)
    where data is the data to be searched for peaks (pandas Dataframe), data_init_index is the index of the first data point in the data (df.index[0]), 
    search_window is the window in seconds to search for peaks (s), sample_rate is the sample rate of the data (Hz), height is the minimum height of the 
    peak (default 0), and prominence is the minimum prominence of the peak (default 3).'''
    # Get the end tidal peaks from the data
    peaks_inds, peaks_vals = find_peaks(data, height=height, prominence=prominence, distance=search_window*sample_rate)
    # Shift CO2peaks_inds by the starting index of df_baseline1
    peaks_inds = peaks_inds + data_init_index
    # Create pandas dataframe of the peaks
    peaks_vals = peaks_vals['peak_heights']
    peaks_df = pd.DataFrame([{'index': peaks_inds[i], 'val': peaks_vals[i]} for i in range(len(peaks_vals))])
    return peaks_inds, peaks_vals, peaks_df

def get_endtidal_valleys(data, data_init_index, search_window=6, sample_rate=200, height=-20, prominence=2):
    '''valleys_inds, valleys_vals, valleys_df = get_endtidal_valleys(data, data_init_index, search_window=6, sample_rate=200, height=0, prominence=3)
    where data is the data to be searched for valleys (pandas Dataframe), data_init_index is the index of the first data point in the data (df.index[0]), 
    search_window is the window in seconds to search for valleys (s), sample_rate is the sample rate of the data (Hz), height is the minimum height of the 
    valley (default -20), and prominence is the minimum prominence of the valley (default 2).'''
    # Get the end tidal valleys from the data
    valleys_inds, valleys_vals = find_peaks(-data, height=height, prominence=prominence, distance=search_window*sample_rate)
    # Shift valleys_inds by the starting index of df_baseline1
    valleys_inds = valleys_inds + data_init_index
    # Create pandas dataframe of the valleys
    valleys_vals = -valleys_vals['peak_heights']
    valleys_df = pd.DataFrame([{'index': valleys_inds[i], 'val': valleys_vals[i]} for i in range(len(valleys_vals))])
    return valleys_inds, valleys_vals, valleys_df

def get_average_breathing_rate(data, sample_rate):
    '''br_avg, f_oneside, X, n_oneside = get_average_breathing_rate(data, sample_rate)
    where data is the data to be analyzed (pandas Dataframe), and sample_rate is the sample rate of the data (Hz).'''
    X = np.fft.fft(data)
    N = len(X)
    n = np.arange(N)
    T = N/sample_rate
    freq = n/T
    # Get the one-sided specturm
    n_oneside = N//2
    # Get the one side frequency
    f_oneside = freq[:n_oneside]
    # Find index of X closest to 0.05 Hz
    idx = (np.abs(f_oneside - 0.05)).argmin()
    # Find the index of the max amplitude above 0.05 Hz 
    idx_max = idx + (np.abs(X[idx:n_oneside])).argmax()
    br_avg = f_oneside[idx_max]
    return br_avg, f_oneside, X, n_oneside

def plot_breathing_rate_fft(data, sample_rate):
    '''plot_breathing_rate_fft(data, sample_rate)
    where data is the data to be analyzed (pandas Dataframe), and sample_rate is the sample rate of the data (Hz).'''
    br_avg, f_oneside, X, n_oneside = get_average_breathing_rate(data, sample_rate)
    print('The average baseline breathing rate is:', br_avg, 'Hz', 1/br_avg, 's')
    print('Breaths per minute:', (br_avg*60), 'BPM')

    plt.figure(figsize = (6, 3))
    plt.xlim(0, 0.5)
    plt.plot(f_oneside, np.abs(X[:n_oneside]), 'b')
    plt.xlabel('Freq (Hz)')
    plt.ylabel('FFT Amplitude |X(freq)|')
    plt.title('Breathing Rate FFT Spectrum')
    plt.show()

def get_average_tcd_rate(data, sample_rate):
    '''br_avg, f_oneside, X, n_oneside = get_average_breathing_rate(data, sample_rate)
    where data is the data to be analyzed (pandas Dataframe), and sample_rate is the sample rate of the data (Hz).'''
    X = np.fft.fft(data)
    N = len(X)
    n = np.arange(N)
    T = N/sample_rate
    freq = n/T
    # Get the one-sided specturm
    n_oneside = N//2
    # Get the one side frequency
    f_oneside = freq[:n_oneside]
    # Find index of X closest to 0.4 Hz
    idx = (np.abs(f_oneside - 0.4)).argmin()
    # Find the index of the max amplitude above 0.4 Hz 
    idx_max = idx + (np.abs(X[idx:n_oneside])).argmax()
    tcd_avg = f_oneside[idx_max]
    return tcd_avg, f_oneside, X, n_oneside

def plot_tcd_fft(data, sample_rate):
    '''plot_breathing_rate_fft(data, sample_rate)
    where data is the data to be analyzed (pandas Dataframe), and sample_rate is the sample rate of the data (Hz).'''
    tcd_avg, f_oneside, X, n_oneside = get_average_tcd_rate(data, sample_rate)
    print('The average tcd rate is:', tcd_avg, 'Hz', 1/tcd_avg, 's')
    print('Pulses per minute:', (tcd_avg*60), 'BPM')

    plt.figure(figsize = (6, 3))
    plt.xlim(0, 4)
    plt.plot(f_oneside, np.abs(X[:n_oneside]), 'b')
    plt.xlabel('Freq (Hz)')
    plt.ylabel('FFT Amplitude |X(freq)|')
    plt.title('TCD Rate FFT Spectrum')
    plt.show()

def get_data_between_comments(dataframe, comment1_index, comment2_index, crop_start, crop_end):
    '''df_5co2 = get_data_between_comments(dataframe, comment1, comment2)
    where dataframe is the dataframe to be analyzed, comment1 is the first comment to start the new dataframe, and comment2 is the comment index to end the new dataframe.'''
    sample_rate = 1/(float(dataframe.iloc[2][0]) - float(dataframe.iloc[1][0]))
    # Get the rows of the comments
    rows_comments = dataframe[dataframe['Comments'].notnull()].index
    # Crop the DataFrame to the rows after comment1_index and before comment2_index
    df_between_comments = dataframe.iloc[rows_comments[comment1_index]:rows_comments[comment2_index]]
    return df_between_comments.iloc[
        (int(crop_start * sample_rate)) : -(int(crop_end * sample_rate))
    ]

def get_tcd_peaks(data, data_init_index, search_window=0.5, sample_rate=200, height=0, prominence=15):
    '''peaks_inds, peaks_vals, peaks_df = get_tcd_peaks(data, data_init_index, search_window=6, sample_rate=200, height=0, prominence=3)
    where data is the data to be searched for peaks (pandas Dataframe), data_init_index is the index of the first data point in the data (df.index[0]), 
    search_window is the window in seconds to search for peaks (s), sample_rate is the sample rate of the data (Hz), height is the minimum height of the 
    peak (default 0), and prominence is the minimum prominence of the peak (default 15).'''
    # Get the end tidal peaks from the data
    peaks_inds, peaks_vals = find_peaks(data, height=height, prominence=prominence, distance=search_window*sample_rate)
    # Shift CO2peaks_inds by the starting index of df_baseline1
    peaks_inds = peaks_inds + data_init_index
    # Create pandas dataframe of the peaks
    peaks_vals = peaks_vals['peak_heights']
    peaks_df = pd.DataFrame([{'index': peaks_inds[i], 'val': peaks_vals[i]} for i in range(len(peaks_vals))])
    return peaks_inds, peaks_vals, peaks_df

def get_tcd_valleys(data, data_init_index, search_window=0.5, sample_rate=200, height=-100, prominence=15):
    '''valleys_inds, valleys_vals, valleys_df = get_tcd_valleys(data, data_init_index, search_window=6, sample_rate=200, height=0, prominence=3)
    where data is the data to be searched for valleys (pandas Dataframe), data_init_index is the index of the first data point in the data (df.index[0]), 
    search_window is the window in seconds to search for valleys (s), sample_rate is the sample rate of the data (Hz), height is the minimum height of the 
    valley (default -100), and prominence is the minimum prominence of the valley (default 15).'''
    # Get the end tidal valleys from the data
    valleys_inds, valleys_vals = find_peaks(-data, height=height, prominence=prominence, distance=search_window*sample_rate)
    # Shift valleys_inds by the starting index of df_baseline1
    valleys_inds = valleys_inds + data_init_index
    # Create pandas dataframe of the valleys
    valleys_vals = -valleys_vals['peak_heights']
    valleys_df = pd.DataFrame([{'index': valleys_inds[i], 'val': valleys_vals[i]} for i in range(len(valleys_vals))])
    return valleys_inds, valleys_vals, valleys_df

def high_pass_filter(data, sample_rate, cutoff_freq):
    '''filtered_data = high_pass_filter(data, sample_rate, cutoff_freq)
    where data is the data to be filtered (pandas Dataframe), sample_rate is the sample rate of the data (Hz), and cutoff_freq is the cutoff frequency (Hz).'''
    # Get the FFT of the data
    X = np.fft.fft(data)
    # Get the number of data points
    N = len(X)
    # Get the frequency
    n = np.arange(N)
    T = N/sample_rate
    freq = n/T
    # Get the one-sided specturm
    n_oneside = N//2
    # Get the one side frequency
    f_oneside = freq[:n_oneside]
    # Find index of X closest to cutoff_freq
    idx = (np.abs(f_oneside - cutoff_freq)).argmin()
    # Filter the data
    X[idx:n_oneside] = 0
    # Get the filtered data
    filtered_data = np.real(np.fft.ifft(X))
    # replace values in dataframe with filtered data
    filtered_df = data
    filtered_df[:] = filtered_data
    return filtered_df

def low_pass_filter(data, sample_rate, cutoff_freq):
    '''filtered_data = low_pass_filter(data, sample_rate, cutoff_freq)
    where data is the data to be filtered (pandas Dataframe), sample_rate is the sample rate of the data (Hz), and cutoff_freq is the cutoff frequency (Hz).'''
    # Get the FFT of the data
    X = np.fft.fft(data)
    # Get the number of data points
    N = len(X)
    # Get the frequency
    n = np.arange(N)
    T = N/sample_rate
    freq = n/T
    # Get the one-sided specturm
    n_oneside = N//2
    # Get the one side frequency
    f_oneside = freq[:n_oneside]
    # Find index of X closest to cutoff_freq
    idx = (np.abs(f_oneside - cutoff_freq)).argmin()
    # Filter the data
    X[idx:] = 0
    # Get the filtered data
    filtered_data = np.real(np.fft.ifft(X))
    # replace values in dataframe with filtered data
    filtered_df = copy.deepcopy(data)
    filtered_df[:] = filtered_data
    return filtered_df

def align_peaks_and_valleys(peaks_inds, peaks_vals, valleys_inds, valleys_vals, TCD_data):
    '''MCAvmean_inds, MCAvmean_vals, MCAvmean_df = align_peaks_and_valleys(peaks_inds, peaks_vals, valleys_inds, valleys_vals, TCD_data)
    where peaks_inds is the indices of the peaks (pandas Dataframe), peaks_vals is the values of the peaks (pandas Dataframe), valleys_inds is the indices of the valleys (pandas Dataframe), valleys_vals is the values of the valleys (pandas Dataframe), and TCD_data is the TCD data (pandas Dataframe).'''
    
    #drop to last peak or trough if necessary to make the number of peaks and troughs the same
    if len(valleys_vals)>len(peaks_vals):
        valleys_vals = valleys_vals[:len(peaks_vals)]
        valleys_inds = valleys_inds[:len(peaks_inds)]
    if len(valleys_vals)<len(peaks_vals):
        peaks_vals = peaks_vals[:len(valleys_vals)]
        peaks_inds = peaks_inds[:len(valleys_inds)]

    #find the mean of the peaks and troughs
    # MCAvmean_inds = (
    #     (peaks_inds - valleys_inds) + peaks_inds
    #     if peaks_inds[0] > valleys_inds[0]
    #     else (peaks_inds - valleys_inds) + valleys_inds
    # )
    # MCAvmean_vals = [
    #     TCD_data[MCAvmean_inds[i]] for i in range(len(MCAvmean_inds))
    # ]
    # MCAvmean_df = pd.DataFrame([{'index': MCAvmean_inds[i], 'val': MCAvmean_vals[i]} for i in range(len(MCAvmean_vals))])

    if peaks_inds[0]>valleys_inds[0]:
        MCAvmean_inds = (peaks_inds-valleys_inds)+peaks_inds
    else:
        MCAvmean_inds = (peaks_inds-valleys_inds)+valleys_inds

        MCAvmean_vals = ((peaks_vals-valleys_vals)/2)+valleys_vals

    MCAvmean_vals = ((peaks_vals-valleys_vals)/2)+valleys_vals
    MCAvmean_df = pd.DataFrame([{'index': MCAvmean_inds[i], 'val': MCAvmean_vals[i]} for i in range(len(MCAvmean_vals))])


    return MCAvmean_inds, MCAvmean_vals, MCAvmean_df

def remove_data_between_comments(df, comment1_dfindex, comment2_dfindex):
    '''df_cleaned = remove_data_between_comments(df, comment1_index, comment2_index)
    where df is the dataframe to be cleaned, comment1_index is the index of the first comment to remove data from, comment2_index is the index of the second comment to end the removal, crop_start is the time in seconds to crop the beginning of the data, and crop_end is the time in seconds to crop the end of the data.'''
    sample_rate = 1/(float(df.iloc[2][0]) - float(df.iloc[1][0]))

    # Crop the DataFrame to the rows before comment1_index and after comment2_index and concatenate the two DataFrames

    df_cleaned = pd.concat([df.iloc[:comment1_dfindex], df.iloc[comment2_dfindex:]])
   
    print('removed data between comments:', comment1_dfindex, comment2_dfindex)
    print('the length of the initial dataframe is:', len(df))
    print('length of first segment', len(df.iloc[:comment1_dfindex]))
    print('length of second segment', len(df.iloc[comment2_dfindex:]))
    print('before and after lengths:',len(df), len(df_cleaned))

    # Reset the index of the DataFrame so that the data is continuous
    df_cleaned = df_cleaned.reset_index(drop=True)

    return df_cleaned


def x_corr(func, co2, n_shifts=None, offset=0, abs_xcorr=False):
    """
    Cross correlation between `func` and `co2`.

    Parameters
    ----------
    func : np.ndarray
        Timeseries, must be SHORTER (or of equal length) than `co2`
    co2 : np.ndarray
        Second timeseries, can be LONGER than `func`
    n_shifts : int or None, optional
        Number of shifts to consider when cross-correlating func and co2.
        When None (default), consider all possible shifts.
        Each shift consists of one sample.
    offset : int, optional
        Optional amount of offset desired for `func`, i.e. the amount of samples
        of `co2` to exclude from the cross correlation.
    abs_xcorr : bool, optional
        If True, x_corr will find the maximum absolute correlation,
        i.e. max(|corr(func, co2)|), rather than the maximum positive correlation.

    Returns
    -------
    float :
        Highest correlation
    int :
        Index of higher correlation
    xcorr : np.ndarray
        Full Xcorr

    Raises
    ------
    ValueError
        If `offset` is higher than the difference between the length of `co2` and `func`.
    NotImplementedError
        If `offset` < 0
        If `co2` length is smaller than `func` length.
    """
    if offset < 0:
        raise NotImplementedError("Negative offsets are not supported yet.")

    if func.shape[0] + offset > co2.shape[0]:
        if offset > 0:
            raise ValueError(
                f"The specified offset of {offset} is too high to "
                f"compare func of length {func.shape[0]} with co2 of "
                f"length {co2.shape[0]}"
            )
        else:
            raise NotImplementedError(
                f"The timeseries has length of {func.shape[0]}, more than the "
                f"length of the given regressor ({co2.shape[0]}). This case "
                "is not supported."
            )

    if n_shifts is None:
        n_shifts = co2.shape[0] - (func.shape[0] + offset) + 1
        print(
            f"Considering all possible shifts of regressor for Xcorr, i.e. {n_shifts}"
        )
    else:
        if n_shifts + offset + func.shape[0] > co2.shape[0]:
            print(
                f"The specified amount of shifts ({n_shifts}) is too high for the "
                f"length of the regressor ({co2.shape[0]})."
            )
            n_shifts = co2.shape[0] - (func.shape[0] + offset) + 1
            print(f"Considering {n_shifts} shifts instead.")

    sco2 = swv(co2, func.shape[0], axis=-1)[offset : n_shifts + offset]

    xcorr = np.dot(zscore(sco2, axis=-1), zscore(func)) / func.shape[0]

    if abs_xcorr:
        return np.abs(xcorr).max(), np.abs(xcorr).argmax() + offset, xcorr
    else:
        return xcorr.max(), xcorr.argmax() + offset, xcorr

def ramp_identifier(df):
    '''df_max_idx, df_min_idx = ramp_identifier(df)
    where df is the MCA data to be analyzed (pandas Dataframe).
    This function finds the indexes of the highest 3 peaks and the lowest 3 valleys in the data.'''
    # Find the indexes of the highest 3 peaks in the DF data
    # find peaks
    df_max_idx = find_peaks(df.iloc[:,1], height=1.1, distance=19)[0]

    # if there are on 2 peaks, add the last index of the data to the end of mca_max_ind
    if len(df_max_idx) == 2:
        print('Only 2 peaks found, adding last index as 3rd peak')
        df_max_idx = np.append(df_max_idx, len(df)-1)

    # find the lowest value before the first peak
    df_min_idx0 = df.iloc[0:df_max_idx[0],1].idxmin()

    # find the lowest value between the first and second peak
    df_min_idx1 = df.iloc[df_max_idx[0]:df_max_idx[1],1].idxmin()

    # find the lowest value between the second and third peak
    df_min_idx2 = df.iloc[df_max_idx[1]:df_max_idx[2],1].idxmin()

    df_min_idx = [df_min_idx0, df_min_idx1, df_min_idx2]

    return df_max_idx, df_min_idx

def ramp_identifier_adv(df, peak_min_distance=19, peak_prominence=1.1, start_search=0):
    '''df_max_idx, df_min_idx = ramp_identifier(df)
    where df is the MCA data to be analyzed (pandas Dataframe).
    This function finds the indexes of the highest 3 peaks and the lowest 3 valleys in the data.'''
    # Find the indexes of the highest 3 peaks in the DF data
    # flatten the data
    df_flat = df.to_numpy().flatten()

    # only consider the data after the start_search index
    df_flat = df_flat[start_search:]

    # find peaks
    df_max_idx = find_peaks(df_flat, height=peak_prominence, distance=peak_min_distance)[0]

    # if there are on 2 peaks, add the last index of the data to the end of mca_max_ind
    if len(df_max_idx) == 2:
        print('Only 2 peaks found, adding last index as 3rd peak')
        df_max_idx = np.append(df_max_idx, len(df)-1)

    # find the lowest value before the first peak
    df_min_idx0 = np.argmin(df_flat[:df_max_idx[0]])

    #print(df_flat[df_max_idx[0]:df_max_idx[1]])
    # find the lowest value between the first and second peak
    print(np.argmin(df_flat[df_max_idx[0]:df_max_idx[1]]))
    print(start_search)
    print(df_max_idx[0])
    #print(df_flat[df_max_idx[1]:df_max_idx[2]])

    # find the lowest value between the first and second peak
    df_min_idx1 = np.argmin(df_flat[df_max_idx[0]:df_max_idx[1]])
    # find the lowest value between the second and third peak
    df_min_idx2 = np.argmin(df_flat[df_max_idx[1]:df_max_idx[2]])
    #df_min_idx = [df_min_idx0, df_min_idx1, df_min_idx2]

    # add the start_search index to the max indexes
    df_max_idx_adjusted = [df_max_idx[0]+start_search, df_max_idx[1]+start_search, df_max_idx[2]+start_search]
    # add the start_search index to the min indexes
    df_min_idx_adjusted = [df_min_idx0+start_search, df_min_idx1+df_max_idx[0]+start_search, df_min_idx2+df_max_idx[1]+start_search]

    return df_max_idx_adjusted, df_min_idx_adjusted

def ramp_up_segmentor(df, df2, df_max_idx, df_min_idx):
    '''df_ramp_up, df2_ramp_up = ramp_up_segmentor(df, df2, df_max_idx, df_min_idx)
    where df is the MCA data to be analyzed (pandas Dataframe), df2 is the PETCO2 data
    to be analyzed (pandas Dataframe), df_max_idx is the indexes of the highest 3 peaks
    in the MCA data, and df_min_idx is the indexes of the lowest 3 valleys in the blood
    flow data.
    This function segments the data into the ramp up segments.'''

    print(df_max_idx)
    print(df_min_idx)

    # remove ramp downs from the mca data
    # remove values from the mca data that are after the first peak and before min_idx1
    df_ramp_up1 = df.iloc[df_min_idx[0]:df_max_idx[0],:]
    # remove values from the mca data that are after the second peak and before min_idx2
    df_ramp_up2 = df.iloc[df_min_idx[1]:df_max_idx[1],:]
    # remove values after the third peak
    df_ramp_up3 = df.iloc[df_min_idx[2]:df_max_idx[2],:]

    # concatenate the ramp up data
    df_ramp_up = pd.concat([df_ramp_up1,df_ramp_up2,df_ramp_up3])

    # remove ramp downs from the petco2 data
    # remove values from the mca data that are after the first peak and before min_idx1
    df2_ramp_up1 = df2.iloc[df_min_idx[0]:df_max_idx[0],:]
    # remove values from the mca data that are after the second peak and before min_idx2
    df2_ramp_up2 = df2.iloc[df_min_idx[1]:df_max_idx[1],:]
    # remove values after the third peak
    df2_ramp_up3 = df2.iloc[df_min_idx[2]:df_max_idx[2],:]

    # concatenate the ramp up data
    df2_ramp_up = pd.concat([df2_ramp_up1,df2_ramp_up2,df2_ramp_up3])




    # #calculate the derivative of df2_ramp_up and remove values of df2_ramp_up that are less than 0
    # df2_ramp_up_derivative = df2_ramp_up.diff()
    # # remove the values of df_ramp_up where the derivative of df2_ramp_up is less than 0
    # df_ramp_up = df_ramp_up[df2_ramp_up_derivative.iloc[:,0]>0.1]
    # df2_ramp_up = df2_ramp_up[df2_ramp_up_derivative.iloc[:,0]>0.1]

    


    return df_ramp_up, df2_ramp_up

# def get_baseline_values(df, min_idx0, num_values):
#     '''base = get_baseline_values(df, min_idx0, num_values)
#     where df is the data to be analyzed (pandas Dataframe), min_idx0 is the index of the first valley in the data, and num_values is the number of values to use for the baseline calculation.
#     This function calculates the baseline of the data.'''

#     len_deep_breaths = 5
#     pre_ramp = df.iloc[0:min_idx0,1]

#     if num_values <= (len(pre_ramp)-len_deep_breaths):
#         print('Selecting first', num_values, 'values for baseline calculation')
#         base = df.iloc[0:num_values,1].mean()
#     else:
#         print('Not enough values to get baseline, using all values before ramp')
#         num_values = len(pre_ramp-5)
#         base = df.iloc[0:num_values,1].mean()

#     return base

def get_baseline_values(df, min_idx0, num_values):
    '''base = get_baseline_values(df, min_idx0, num_values)
    where df is the data to be analyzed (pandas Dataframe), min_idx0 is the index of the first valley in the data, and num_values is the number of values to use for the baseline calculation.
    This function calculates the baseline of the data.'''

    #remove the nan values from the df
    df = df.dropna()
    df = df.reset_index(drop=True)

    len_deep_breaths = 5
    pre_ramp = df.iloc[0:min_idx0,0]

    if num_values <= (len(pre_ramp)-len_deep_breaths):
        print('Selecting first', num_values, 'values for baseline calculation')
        base = df.iloc[0:num_values,0].mean()
    else:
        print('Not enough values to get baseline, using all values before ramp')
        num_values = len(pre_ramp-5)
        base = df.iloc[0:num_values,0].mean()

    return base

def remove_nans(mca, petco2):
    '''mca, petco2 = remove_nans(mca, petco2)
    where mca is the MCA data to be analyzed (pandas Dataframe), and petco2 is the PETCO2 data to be analyzed (pandas Dataframe).
    This function removes the nans from both datasets and the corresponding value in the other dataset and then resets indices.'''

    #get the indices of all nans in the data and remove them
    nan_idx_mca = np.argwhere(np.isnan(mca.iloc[:,1]))
    nan_idx_petco2 = np.argwhere(np.isnan(petco2.iloc[:,1]))
    # combine the indices of the nans
    nan_idx = np.concatenate((nan_idx_mca,nan_idx_petco2),axis=0)
    print('Removing', len(nan_idx), 'nan values from both datasets')
    # remove the nans from the mca and petco2 data
    mca = mca.drop(nan_idx[:,0])
    petco2 = petco2.drop(nan_idx[:,0])

    # reset the index of the mca and petco2 data after removing the nans
    mca = mca.reset_index(drop=True)
    petco2 = petco2.reset_index(drop=True)

    return mca, petco2

def get_data_between_triggers(df_raw, trig_col_name = 'Trig', trigger_thresh=2, num_ind_before_trig=1, num_ind_afer_trig=1, num_triggers=769):
    '''df = get_data_between_triggers(df_raw, trig_col_name = 'Trig', trigger_thresh=2, num_ind_before_trig=1, num_ind_afer_trig=1)
    where df_raw is the data to be analyzed (pandas Dataframe), trig_col_name is the name of the column with the triggers, trigger_thresh 
    is the threshold for the triggers, num_ind_before_trig is the number of indices before the trigger to include in the data, and 
    num_ind_afer_trig is the number of indices after the trigger to include in the data.
    This function crops the data to the time between the first and last trigger.'''

    # get the indices of the triggers (channel 'Trig') above the threshold given
    trigger_idx = df_raw[df_raw[trig_col_name]>3].index

    print('number of triggers found:', len(trigger_idx))
    print('extracting the last', num_triggers ,'triggers')

    # get the indices of the first and last trigger based on the number of triggers
    trigger_start = trigger_idx[-num_triggers]

    print('index of the first trigger:', trigger_start)
    trigger_end = trigger_idx[-1]
    print('index of the last trigger:', trigger_end)

    df_start = trigger_start - int(num_ind_before_trig)
    print('index to start crop:', df_start)
    df_end = trigger_end + int(num_ind_afer_trig)
    print('index to end crop:', df_end)

  

    # crop the data to the time between the first and last trigger
    df = df_raw.iloc[df_start:df_end,:]



    return df