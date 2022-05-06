#!/usr/bin/env python
# coding: utf-8

# In[13]:

print('Start importing')
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
sns.set_style('white')
sns.set_context('notebook', font_scale=1.5)
get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import nibabel as nib

from nilearn.plotting import (plot_anat, plot_epi, plot_roi,
                              plot_glass_brain, plot_stat_map,
                              plot_surf_stat_map, plot_img_on_surf)
from nilearn.image import mean_img,coord_transform,resample_img,smooth_img
from nilearn.signal import clean
from scipy.special import legendre
from scipy.stats import zscore
from util import get_coord_seg,read_sub
from kernel import kernel_bold,kernel_confounds,kernel_poly
from nibabel.nifti1 import Nifti1Image

print('Finish importing')

# In[2]:


base_folder = '/jukebox/scratch/bichanw/502b/ds002799-download/'


# In[3]:


# create a list for viable subjects
sub_num = np.array([folder[-3:] for folder in glob.glob(base_folder+'sub*')],dtype='int')
sub_2_delete = np.array([394,369,339]) 
sub_num = np.delete(sub_num,np.nonzero(sub_2_delete[:,None] == sub_num)[1])


# In[4]:

sub = sub_num[int(sys.argv[1])]
run_num = len(glob.glob(f'{base_folder}sub-{sub}/ses-postop/func/sub-{sub}_ses-postop_task-es_run-*_bold.nii'))




# In[5]:

for run in np.arange(1,run_num+1):
    # read basic info
    seg_name,electrodes,aseg1,aseg2,stimulated,events,mask_brain = read_sub(sub,run)


    # In[6]:


    # data in MNI152 space
    fimage_file = f'{base_folder}derivatives/fmriprep/sub-{sub}/ses-postop/func/sub-{sub}_ses-postop_task-es_run-{run:02}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
    fimage_MNI152 = nib.load(fimage_file)
    smoothed_img = smooth_img(fimage_MNI152,fwhm=3)
    data_MNI152 = smoothed_img.get_fdata()
    print(f'MNI152 func shape is {data_MNI152.shape}; {np.prod(data_MNI152.shape[:3])} voxels in total')


    # In[7]:


    # read kernels
    bold = kernel_bold(data_MNI152.shape[3],events)
    hm_labels = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
    hm,acompcor = kernel_confounds(base_folder,sub,run,hm_labels=hm_labels)
    detrend_mat = kernel_poly(data_MNI152.shape[3],1)


    print(f"{np.sum(mask_brain).astype('int')} / {mask_brain.size} of voxels in total")

    func_masked = data_MNI152[mask_brain.astype('bool'),:]
    print(f'{func_masked.shape}')


    # In[114]:


    # zscore data, which made results worse
    # from scipy.stats import zscore 
    # func_masked = zscore(func_masked,axis=None)


    # In[11]:


    # create design matrix
    XZ = np.hstack((bold[:,np.newaxis],hm,acompcor,detrend_mat))
    XZ_labels = [*['bold'], *hm_labels, *[f'a_comp_cor_0{n}' for n in range(5)],*[f'order_{i}' for i in range(detrend_mat.shape[1])]]


    # In[14]:


    # Plot full design matrix here:
    fig,ax = plt.subplots(1)
    ax.imshow(zscore(XZ,axis=0), aspect='auto',interpolation='none')
    ax.set(ylabel='TR',xticks=np.arange(XZ.shape[1]));
    ax.set_xticklabels(XZ_labels, rotation = 90);
    plt.savefig(f'/jukebox/scratch/bichanw/502b/results/sub{sub}_run{run}_designmat.png',bbox_inches="tight")
    # In[117]:


    # Run regression with basic OLS
    b, _, _, _ = np.linalg.lstsq(XZ, func_masked.T, rcond=-1)
    func_sig = np.zeros(tuple([b.shape[0]]) + data_MNI152.shape[:3])
    func_sig[:,mask_brain.astype('bool')] = b


    # In[118]:
    Y_hat = XZ@b
    data_filtered = (func_masked.T - XZ[:,1:]@b[1:,:]).T

    func_filtered = np.zeros(data_MNI152.shape)
    func_filtered[mask_brain.astype('bool'),:] = data_filtered


    # In[122]:


    plt.subplots(1)
    plt.plot(Y_hat[:,10],label='predicted');
    plt.plot(func_masked[10,:],label='raw');
    plt.plot(data_filtered[10,:],label='filtered');
    plt.legend()
    plt.savefig(f'/jukebox/scratch/bichanw/502b/results/sub{sub}_run{run}_filtered.pdf',bbox_inches="tight")



    # Plot beta coefficients for the face regressor on the brain here:
    stim_nifti = Nifti1Image(func_sig[0,:]/1e3, fimage_MNI152.affine)
    nib.save(stim_nifti,f'/jukebox/scratch/bichanw/502b/results/sub{sub}_run{run}_b.nii')

    plot_stat_map(stim_nifti,mean_img(fimage_file),title=f'sub{sub}_run{run}',output_file=f'/jukebox/scratch/bichanw/502b/results/sub{sub}_run{run}_b.pdf');

    fig,ax = plt.subplots(1)
    ax.hist(func_sig[0,:].flatten());
    plt.savefig(f'/jukebox/scratch/bichanw/502b/results/sub{sub}_run{run}_bhist.pdf',bbox_inches="tight")


    plt.close('all')
    # In[ ]:




