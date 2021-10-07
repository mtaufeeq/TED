#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os 
import glob

import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt 
import seaborn as sns 


# In[5]:


def ted(au_intensities, gaze_0, gaze_1, pose_orientation, pose_rot, landmarks_xy, window_size):
    """
    Return the TED score, static expressiveness and dynamic expressiveness for a given video 
    
    Input: numpy matrix 
    Output: 
    """
    # compute static expressiveness 
    static_exprv = static_expressiveness(au_intensities)
    
    # compute dynamics 
    leng = au_intensities.shape[0]
#     m_dynamics_df = None 
    m_dynamics_lst = []
    for f in range(leng):
        au_inten_sdist = std_euclidean_dist(au_intensities[f, :], get_rep_data(au_intensities)[f, :])
        gaze_0_sdist = std_euclidean_dist(gaze_0[f, :], get_rep_data(gaze_0)[f, :]) 
        gaze_1_sdist = std_euclidean_dist(gaze_1[f, :], get_rep_data(gaze_1)[f, :]) 
        pose_orientation_sdist = std_euclidean_dist(pose_orientation[f, :], get_rep_data(pose_orientation)[f, :]) 
        pose_rot_sdist = std_euclidean_dist(pose_rot[f, :], get_rep_data(pose_rot)[f, :]) 
        landmarks_xy_sdist = std_euclidean_dist(landmarks_xy[f, :], get_rep_data(landmarks_xy)[f, :]) 

        au_inten_diff = np.sum(np.subtract(au_intensities[f, :], get_rep_data(au_intensities)[f, :]))
        gaze_0_diff = np.sum(np.subtract(gaze_0[f, :], get_rep_data(gaze_0)[f, :])) 
        gaze_1_diff = np.sum(np.subtract(gaze_1[f, :], get_rep_data(gaze_1)[f, :])) 
        pose_orientation_diff = np.sum(np.subtract(pose_orientation[f, :], get_rep_data(pose_orientation)[f, :]))
        pose_rot_diff = np.sum(np.subtract(pose_rot[f, :], get_rep_data(pose_rot)[f, :])) 
        landmarks_xy_diff = np.sum(np.subtract(landmarks_xy[f, :], get_rep_data(landmarks_xy)[f, :])) 
        
        sdist_lst = [au_inten_sdist, gaze_0_sdist, gaze_1_sdist, pose_orientation_sdist, pose_rot_sdist, landmarks_xy_sdist]
        diff_lst = [au_inten_diff, gaze_0_diff, gaze_1_diff, pose_orientation_diff, pose_rot_diff, landmarks_xy_diff]
#         print(sdist_lst)
        # TODO - save the distance as well 
        
        # TODO - save the difference as well 
        
        
        # take care of the direction 
        for j in range(len(diff_lst)):
            if diff_lst[j] < 0:
                sdist_lst[j] = sdist_lst[j] * -1
                
        # merge the dynaics to list 
        m_dynamics_lst.append(sdist_lst)                
        
#         print(gaze_rot_sdist, gaze_rot_diff)
        
    # compute the rolling average 
    m_dynamics_df = pd.DataFrame(m_dynamics_lst, columns=["dynm_au_intent", "dynm_gaze_0", "dynm_gaze_1", 
                                                         "dynm_pose_orien", "dynm_pose_rot", "landmarks_xy"])
    # print(m_dynamics_df.head())
                    
    m_lookback = rolling_avg(m_dynamics_df, window_size)  # + 1 # window size 
    # print(m_lookback.head())
    # add all of them up for final results 
#     m_lookback += 1 
#     dynm_exprv = 1 + (m_lookback.prod(axis=1).to_frame() / 6)
    
    dynm_exprv = 1 + m_lookback.sum(axis=1).to_frame() # 0 effect if we multiple 
 
    ted_score = static_exprv * dynm_exprv # TODO - to be updated 
    
    return ted_score

def static_expressiveness(au_inten_df):
    """
    Return compute the exponent of AUs to pay more attention to the highly expressiven AUs
    """
    exp_df = np.sum(np.exp(au_inten_df), axis=1)

    return np.expand_dims(exp_df, axis=1) 


def std_euclidean_dist(x0, x1):  # ref: https://stackoverflow.com/questions/38161071/how-to-calculate-normalized-euclidean-distance-on-two-vectors
    if (np.var(x0) + np.var(x1)) == 0:
        return 0
    else:
        var_ = np.var(x1 - x0) / (np.var(x0) + np.var(x1))
        return 0.5 * var_


def get_rep_data(df):
    vec = np.expand_dims(df[0], axis=0) # row 0
    
    return np.concatenate([vec, df[:df.shape[0]-1, :]], axis=0) # got 2 copies of row 0
    

def diff_with_dir(df1, df0):
    return np.subtract(df1, df0) # important thing is direction (+ or -)


def rolling_avg(df, window_size): 
    init_frame_vals = df.iloc[:window_size, :]
    rolled_df = df.rolling(window=window_size).mean().iloc[window_size:, :]
    
    return init_frame_vals.append(rolled_df, ignore_index=True)


# In[6]:


def get_target_mat(df, target_AUs):
    """
    Return processed OpenFace file 
    """
    col_names = df.columns.tolist()
    
    # create a new dataframe 
    # TODO - take care of the naming issue 
    meta = df.iloc[:, :col_names.index(' success')+1]
    gaze_loc = df.iloc[:, col_names.index(' gaze_0_x'):col_names.index(' gaze_1_z')+1] 
    gaze_rot = df.iloc[:, col_names.index(' gaze_angle_x'):col_names.index(' gaze_angle_y')+1]
    pose_loc = df.iloc[:, col_names.index(' pose_Tx'):col_names.index(' pose_Tz')+1]
    pose_rot = df.iloc[:, col_names.index(' pose_Rx'):col_names.index(' pose_Rz')+1]
    landmarks = df.iloc[:, col_names.index(' x_0'):col_names.index(' y_67')+1] 
    # au_inten_OF = df.iloc[:, col_names.index(' AU01_r'):col_names.index(' AU45_r')+1].values
    # au_inten_FACS = df.iloc[:, col_names.index('au1'):col_names.index('au9')+1].values # generic

#     au_involv_pain = [' AU04_r', ' AU06_r', ' AU09_r', ' AU10_r', ' AU25_r']  #  ["au4", "au6", "au9", "au10", "au25"] 
#     au_inten_FACS = df[au_involv_pain]
    au_inten_OFPAU = df[target_AUs]

    return meta, gaze_loc, gaze_rot, pose_loc, pose_rot, landmarks, au_inten_OFPAU


# ## Load file 

# In[7]:


dir_ = r"D:\rdevs\datasets\processed\painful_2011\dummy_OpenFace" 
fname = "p042_vid01.csv"

dir_w_fname = os.path.join(dir_, fname)


# In[8]:


dir_w_fname


# In[9]:


df = pd.read_csv(dir_w_fname)


# In[10]:


df.head()


# ## Run TED algorithm 
# 
# **Paper link**: https://arxiv.org/abs/2010.14705

# In[19]:


target_AUs = [' AU01_r', ' AU02_r', ' AU04_r', ' AU05_r', ' AU06_r', ' AU07_r', ' AU09_r', ' AU10_r',
         ' AU12_r', ' AU14_r', ' AU15_r', ' AU17_r', ' AU20_r', ' AU23_r', ' AU25_r', ' AU26_r', 
         ' AU45_r']

# get target input data for TED 
meta, gaze_loc, gaze_rot, pose_loc, pose_rot, landmarks, au_inten_OFPAU = get_target_mat(df, target_AUs) 

# get the TED score for a given video sequence 
window_size = 10 
ted_score = ted(au_inten_OFPAU.values, 
                gaze_loc.values, gaze_rot.values, 
                pose_loc.values, pose_rot.values, 
                landmarks.values, 
                window_size)


# ## Plot the figures 

# In[28]:


plt.rcParams["figure.figsize"] = [20, 5] # [10, 5] looks better though
fig, ax = plt.subplots()

sns.set_context("paper")
sns.set(font_scale=1.2)
sns.set_style("whitegrid")


plt.plot(ted_score)

plt.xlabel('Video frames')
plt.ylabel('TED score')


# In[ ]:




