import numpy as np
import pandas as pd
import nibabel as nib

def plt_mean_func_in_region(ind,func_filtered,aseg,events,seg_name):
    _,ax = plt.subplots(1)

    # plot filtered bold
    ax.plot(np.arange(func_filtered.shape[3])*3,np.mean(func_filtered[aseg.get_fdata()==ind,:],axis=0));

    # plot stimulation time
    ymin, ymax = ax.get_ylim();
    [ax.fill_between([row.onset,row.onset+row.duration],ymin,ymax, color='0.8', alpha=0.5) for ind,row in events.iterrows()];

    # figure setting
    ax.set(xlabel='time (s)',title=seg_name[seg_name.index==ind].name.to_string());


def get_coord_seg(coords,seg_name,aseg):
    ind = np.round(np.linalg.inv(aseg.affine)@np.append(coords,1))[:3].astype('int')
    return seg_name[seg_name.index==aseg.get_fdata()[tuple(ind)]].name.to_string()


def print_stimiulated(sub,base_folder='/jukebox/scratch/bichanw/502b/ds002799-download/'):
    n_runs = len(glob.glob(f'{base_folder}sub-{sub}/ses-postop/func/*run*_bold.json'))

    stimulated_sub = np.empty((n_runs,2))
    for run in range(n_runs):
        # electrodes stimulated during the run
        stimulated_f = f'{base_folder}sub-{sub}/ses-postop/ieeg/sub-{sub}_ses-postop_task-es_run-{run+1:02}_channels.tsv'
        stimulated = pd.read_csv(stimulated_f,sep='\t')
        stimulated_sub[run] = stimulated[:].name.to_numpy()

    stim,stim_ind= np.unique(stimulated_sub,axis=0,return_index=True)

    print(stim)
    print(stim_ind)


# run subject electrode location
def read_sub(sub,run,base_folder='/jukebox/scratch/bichanw/502b/ds002799-download/'):
    # load segmentation map (pre-subject)
    seg_name = pd.read_csv(f'{base_folder}derivatives/fmriprep/desc-aseg_dseg.tsv',sep='\t')

    # read electrodes
    electrodes_f = f'{base_folder}sub-{sub}/ses-postop/ieeg/sub-{sub}_ses-postop_space-MNI152NLin6Asym_electrodes.tsv'
    electrodes = pd.read_csv(electrodes_f,sep='\t')

    # read segmentation
    aseg1 = nib.load(f'{base_folder}derivatives/fmriprep/sub-{sub}/ses-postop/func/sub-{sub}_ses-postop_task-es_run-{run:02}_space-MNI152NLin2009cAsym_desc-aparcaseg_dseg.nii.gz')
    aseg2 = nib.load(f'{base_folder}derivatives/fmriprep/sub-{sub}/ses-postop/func/sub-{sub}_ses-postop_task-es_run-{run:02}_space-MNI152NLin2009cAsym_desc-aseg_dseg.nii.gz')
    
    # generate output
    regions_seg1 = [get_coord_seg(np.array(electrodes.loc[i,['x','y','z']]).astype('float'),seg_name,aseg1) for i in range(len(electrodes))]
    regions_seg2 = [get_coord_seg(np.array(electrodes.loc[i,['x','y','z']]).astype('float'),seg_name,aseg2) for i in range(len(electrodes))]
    electrodes['seg1'] = regions_seg1
    electrodes['seg2'] = regions_seg2


    # electrodes stimulated during the run
    stimulated_f = f'{base_folder}sub-{sub}/ses-postop/ieeg/sub-{sub}_ses-postop_task-es_run-{run:02}_channels.tsv'
    stimulated = pd.read_csv(stimulated_f,sep='\t')
    stimulated_ind = stimulated[:].name.to_numpy()
    # read events
    events_f = f'{base_folder}sub-{sub}/ses-postop/func/sub-{sub}_ses-postop_task-es_run-{run:02}_events.tsv'
    events   = pd.read_csv(events_f,sep='\t')

    # brain mask
    mask_brain = nib.load(f'{base_folder}derivatives/fmriprep/sub-{sub}/ses-postop/func/sub-{sub}_ses-postop_task-es_run-{run:02}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz').get_fdata()



    return seg_name,electrodes,aseg1,aseg2,stimulated,events,mask_brain