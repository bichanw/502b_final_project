import numpy as np
import pandas as pd
from fmritools.hrf import spm_hrf
from scipy.special import legendre

def kernel_bold(n_trs,events):
    # create time and supersampling time
    tr = 3

    sfreq = .1
    sst = np.arange(0, n_trs * tr, sfreq)

    boxcar = np.zeros_like(sst)
    for ievent in range(len(events)):
        boxcar[np.logical_and(sst>=events.onset[ievent], sst<events.onset[ievent]+events.duration[ievent])] = 1


    # Define HRF
    hrf = spm_hrf(sfreq)
    # Convolve events and HRF.
    bold = np.apply_along_axis(np.convolve, 0, boxcar, hrf)[:sst.size]

    # Downsample and plot both time series here:
    times = np.arange(n_trs) * tr
    ix = np.in1d(sst, times)
    
    # plotting
    # _,ax = plt.subplots(1,2,figsize=(8,3))
    # ax[0].plot(hrf); ax[0].set(title='hrf');
    # ax[1].plot(sst[ix],bold[ix]); ax[1].set(title='bold');

    return bold[ix]

def kernel_confounds(base_folder,sub,run,hm_labels=['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']):
    # Load fMRIPrep confound regressors
    confounds_f = f'{base_folder}derivatives/fmriprep/sub-{sub}/ses-postop/func/sub-{sub}_ses-postop_task-es_run-{run:02}_desc-confounds_regressors.tsv'
    confounds = pd.read_csv(confounds_f, sep='\t')

    # Extract motion regressors
    hm = confounds[hm_labels].values

    # _,ax = plt.subplots(2,1,sharex=True);
    # ax[0].plot(hm[:,:3]); ax[0].legend(hm_labels[:3]);
    # ax[1].plot(hm[:,3:]); ax[1].legend(hm_labels[3:]);

    # Extract 5 anatomical compcor signals
    acompcor_n = 5
    acompcor_labels = [f'a_comp_cor_0{n}' for n in range(acompcor_n)]
    acompcor = confounds.filter(acompcor_labels).values

    # Plot aCompCor regressors
    # fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    # ax.plot(acompcor)
    # sns.despine()

    return hm, acompcor

def kernel_poly(n_trs,n_runs,order=2):
    
    # Set the runwise polynomial parameters
    run_trs = n_trs // n_runs

    # Create a confound matrix with a block of polynomials for each run
    detrend_mat = [np.split(irun, n_runs, axis=1) for irun in
                 np.split(np.zeros((n_trs, (order + 1) * n_runs)), n_runs)]


    # create coefficient block
    X_poly = np.empty_like(detrend_mat[0][0])
    x = np.linspace(-1,1,run_trs)
    for n in range(order+1):
        f = legendre(n)
        X_poly[:,n] = f(x)
        
        
    for irun in np.arange(n_runs):
        detrend_mat[irun][irun] = X_poly ### Fill in the Legendre polyomials here!!!

    detrend_mat = np.block(detrend_mat)
    # plt.plot(detrend_mat);
    # plt.title('polyomials');

    return detrend_mat





