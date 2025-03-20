import stumpy
import numpy as np
import matplotlib.pyplot as plt

def find_motif(ts, m, plot=True):
    """
    Find motifs and plot themin a time series.
    
    Args:
        ts (pd.DataFrame): Time series data
        m (int): Window size for motif detection
        plot (bool): Whether to plot the motifs
        
    Returns:
        tuple: (mps, motifs_idx) where mps is a dictionary of matrix profiles and motifs_idx is a dictionary of motif index locations
    """
    mps = {}  # Store the 1-dimensional matrix profiles
    motifs_idx = {}  # Store the index locations for each pair of 1-dimensional motifs (i.e., the index location of two smallest matrix profile values within each dimension)
    for dim_name in ts.columns:
        mps[dim_name] = stumpy.stump(ts[dim_name], m)
        motif_distance = np.round(mps[dim_name][:, 0].astype(float).min(), 1)
        print(f"The motif pair matrix profile value in {dim_name} is {motif_distance}")
        motifs_idx[dim_name] = np.argsort(mps[dim_name][:, 0])[:2]

    if plot:
        fig, axs = plt.subplots(len(mps), sharex=True, gridspec_kw={'hspace': 0})
        for i, dim_name in enumerate(list(mps.keys())):
            axs[i].set_ylabel(dim_name, fontsize='8')
            axs[i].plot(ts[dim_name])
            axs[i].set_xlabel('Step', fontsize ='20')
            for idx in motifs_idx[dim_name]:
                axs[i].plot(ts[dim_name].iloc[idx:idx+m], c='red', linewidth=4)
                axs[i].axvline(x=idx, linestyle="dashed", c='black')
            
        plt.show()
    return mps, motifs_idx
