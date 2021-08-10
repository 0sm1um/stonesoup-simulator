#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 14:43:19 2021

@author: Sean O'Rourke
"""
import pickle 
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as mplt

#%% File Setup
import os
import glob 

root_path = 'C:/Users/0sm1um' 
sims_path = 'simulation/'
linear_sim_path = '2D_Linear_Tx_Meas_Model'
nonlinear_sim_path = '50_Runs_Nonlinear_Measurement_Model'
paper_path = 'EnKFDocs/figures/'

linear_inpath = os.path.join(root_path, sims_path, linear_sim_path, '')
nonlinear_inpath = os.path.join(root_path, sims_path, nonlinear_sim_path, '')
paper_outpath = os.path.join(root_path, paper_path, '')

# This technique gives you a list of files that match a "glob" regular expression pattern.
linear_rmse_files = glob.glob(linear_inpath + 'rmse_*.txt')
nonlinear_rmse_files = glob.glob(nonlinear_inpath + 'rmse_*.txt')

#%% Initial setup & import 
import itertools

position_mapping = [0, 2]
velocity_mapping = [1, 3]

nonlin_EnKF_idx = 0
nonlin_SREnKF_idx = 1

lin_KF_idx = 0
lin_EnKF_idx = 1
lin_SREnKF_idx = 2

# Figure width of an IEEE column is ~3.39 inches. "Golden Ratio" is most aesthetically pleasing alignment, ergo:
width = 3.39
golden_ratio = (np.sqrt(5)-1.0)/2.0
height = width*golden_ratio

# # Setup parameters borrowed from https://www.bastibl.net/publication-quality-plots/
# plt.rc('font', family='serif', serif='Times')
# plt.rc('text', usetex=True)
# plt.rc('xtick', labelsize=8)
# plt.rc('ytick', labelsize=8)
# plt.rc('axes', labelsize=8)

# Now trying "SciencePlots" https://github.com/garrettj403/SciencePlots/
# Note that this has to be installed in your Python environment before you use it. 
# This automatically sizes the figure, adds the color/linestyle cycle, formats axes, etc.
# to make the figure compliant with IEEE's recommended formats. It's neat!
plt.style.use(['science','ieee'])


# Basic labels we'll reuse:
x_label = 'Timestep'
y_labels = [r'${RMSE}$ (m)', r'${RMSE}$ (m/s)', r'${RMSE}$ (m)', r'${RMSE}$ (m/s)']
titles = [r'$x$ - position', r'$v_x$ - velocity', r'$y$ - position', r'$v_y$  - velocity']


#%% Nonlinear example data import & munging
with open(nonlinear_rmse_files[0], 'rb') as f:
    nonlinear_RMSE_data = list(itertools.chain.from_iterable(pickle.load(f)))

#%% Linear example data import & munging
linear_RMSE_data = []
for file in linear_rmse_files: 
    with open(file, 'rb') as f:
        linear_RMSE_data.append(list(itertools.chain.from_iterable(pickle.load(f))))


#%% Nonlinear test plotting:

nonlin_fig, nonlin_axes = plt.subplots(2,2, constrained_layout=True)

#nonlin_fig.subplots_adjust(hspace=1)

for idx, axis in enumerate(nonlin_axes.flatten()):
    title = titles[idx]
    
    axis.set(title=title, xlabel=x_label, ylabel=y_labels[idx])
    axis.tick_params(length=1)
    EKF_Line = axis.plot(nonlinear_RMSE_data[0][idx], label='EKF')
    EnKF_line = axis.plot(nonlinear_RMSE_data[1][idx], 'b', label='EnKF')
    SREnKF_line = axis.plot(nonlinear_RMSE_data[2][idx], 'r--', label='SREnKF')
    
    # Force similar datatypes to share y-axes -- thus, if you want to make the 
    # y-axes independent, comment this block out. OR if you want new limits for
    # the y-axes, add axis.set_ylim(arguments) *after* the call to sharey. 
    if idx in position_mapping:
        axis.sharey(nonlin_axes[0, position_mapping[0]])
    elif idx in velocity_mapping:
        axis.sharey(nonlin_axes[0, velocity_mapping[0]])
        

# Get the end handles/labels for the figure-wide legend from the last set of axes
# Borrowed from https://stackoverflow.com/a/46921590
handles, labels = axis.get_legend_handles_labels()

# Plot the legend, but ensure that it's below all of my subplot labels
# This partially continues from https://stackoverflow.com/a/46921590, but also
# adds the bounding box trick from https://stackoverflow.com/a/17328230
nonlin_fig.legend(handles, labels, frameon=True, loc='lower center', ncol = 3,  
                  bbox_to_anchor = (0,-0.1,1,1), bbox_transform = plt.gcf().transFigure)
    
#nonlin_fig.supxlabel(x_label)
#nonlin_fig.supylabel(r'${RMSE}$')
nonlin_fig.align_labels()
nonlin_fig.align_ylabels()
plt.savefig(os.path.join(paper_outpath,'nonlinear_EnSRF_vs_EnKF.pdf'))
plt.show()

#%% Linear plotting: EnKF

linear_fig, linear_axes = plt.subplots(2,2, constrained_layout=True)


for idx, axis in enumerate(linear_axes.flatten()):
    title = titles[idx]
    
    axis.set(title=title, xlabel=x_label, ylabel=y_labels[idx])
    axis.tick_params(length=1)
    KF_line = axis.plot(linear_RMSE_data[0][lin_KF_idx][idx], label='KF')
    EnKF_line = axis.plot(linear_RMSE_data[2][lin_EnKF_idx][idx], label='EnKF, $M = 5$')
    EnKF50_line = axis.plot(linear_RMSE_data[1][lin_EnKF_idx][idx], label='EnKF, $M = 10$')
    EnKF500_line = axis.plot(linear_RMSE_data[0][lin_EnKF_idx][idx], label='EnKF, $M = 15$')
    
    
    # this example has too many ticks for the position axis, let's force that down.
    if idx == 0:
        axis.yaxis.set_major_locator(mplt.MultipleLocator(1))
    elif idx == 1: 
        axis.yaxis.set_major_locator(mplt.MultipleLocator(0.5))
        
    # Force similar datatypes to share axes. 
    if idx in position_mapping:
        axis.sharey(linear_axes[0, position_mapping[0]])
    elif idx in velocity_mapping:
        axis.sharey(linear_axes[0, velocity_mapping[0]])
        
# Get the end handles/labels for the figure-wide legend from the last set of axes
# Borrowed from https://stackoverflow.com/a/46921590
handles, labels = axis.get_legend_handles_labels()

# Plot the legend, but ensure that it's below all of my subplot labels
# This partially continues from https://stackoverflow.com/a/46921590, but also
# adds the bounding box trick from https://stackoverflow.com/a/17328230
linear_fig.legend(handles, labels, frameon=True, loc='lower center', ncol = 2,  bbox_to_anchor = (0,-0.2,1,1),
            bbox_transform = plt.gcf().transFigure)

#nonlin_fig.supxlabel(x_label)
#nonlin_fig.supylabel(r'${RMSE}$')
linear_fig.align_labels()
linear_fig.align_ylabels()
plt.savefig(os.path.join(paper_outpath,'linear_EnKF_vs_KF.pdf'))
plt.show()

#%% Linear plotting: EnSRF
linear_fig, linear_axes = plt.subplots(2,2, constrained_layout=True)

#nonlin_fig.subplots_adjust(hspace=1)

for idx, axis in enumerate(linear_axes.flatten()):
    title = titles[idx]
    
    axis.set(title=title, xlabel=x_label, ylabel=y_labels[idx])
    axis.tick_params(length=1)
    KF_line = axis.plot(linear_RMSE_data[0][lin_KF_idx][idx], label='KF')
    SREnKF_line = axis.plot(linear_RMSE_data[2][lin_SREnKF_idx][idx], label=r'EnSRF, $M = 5$')
    SREnKF50_line = axis.plot(linear_RMSE_data[1][lin_SREnKF_idx][idx], label=r'EnSRF, $M = 10$')
    SREnKF500_line = axis.plot(linear_RMSE_data[0][lin_SREnKF_idx][idx], label=r'EnSRF, $M = 15$')
    
    # this example has too many ticks for the position axis, let's force that down.
    if idx == 0:
        axis.yaxis.set_major_locator(mplt.MultipleLocator(5))
    elif idx == 1: 
        axis.yaxis.set_major_locator(mplt.MultipleLocator(1))
    
    # Force similar datatypes to share axes. 
    if idx in position_mapping:
        axis.sharey(linear_axes[0, position_mapping[0]])
    elif idx in velocity_mapping:
        axis.sharey(linear_axes[0, velocity_mapping[0]])
        
# Get the end handles/labels for the figure-wide legend from the last set of axes
# Borrowed from https://stackoverflow.com/a/46921590
handles, labels = axis.get_legend_handles_labels()

# Plot the legend, but ensure that it's below all of my subplot labels
# This partially continues from https://stackoverflow.com/a/46921590, but also
# adds the bounding box trick from https://stackoverflow.com/a/17328230
linear_fig.legend(handles, labels, frameon=True, loc='lower center', ncol = 2,  bbox_to_anchor = (0,-0.2,1,1),
            bbox_transform = plt.gcf().transFigure)

#nonlin_fig.supxlabel(x_label)
#nonlin_fig.supylabel(r'${RMSE}$')
linear_fig.align_labels()
linear_fig.align_ylabels()
plt.savefig(os.path.join(paper_outpath, 'linear_EnSRF_vs_KF.pdf'))
plt.show()