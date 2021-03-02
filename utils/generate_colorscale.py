#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate color scale for DEMs
"""
import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as plticker
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime
from matplotlib.ticker import FormatStrFormatter

def save_fig2png(fig, size=[8, 6.7], folder=None, fname=None):
    if size is None:
        fig.set_size_inches([8, 6.7])
    else:
        fig.set_size_inches(size)
    plt.pause(.1)
    if fname is None:
        if fig._suptitle is None:
            fname = 'figure_{:d}'.format(fig.number)
        else:
            ttl = fig._suptitle.get_text()
            ttl = ttl.replace('$','').replace('\n','_').replace(' ','_')
            fname = re.sub(r"\_\_+", "_", ttl)
    if folder:
        plt.savefig(os.path.join(folder, fname +'_'+datetime.now().strftime("%Y%m%d%H%M%S") +'.pdf'),format='pdf', dpi=1200,  orientation='landscape', papertype='letter')
    else:
        plt.savefig(fname +'_'+datetime.now().strftime("%Y%m%d%H%M%S") +'.png',format='png', dpi=300)

'''
# Get colormap from opencv
full_map_BGR = cv2.applyColorMap(np.linspace(0,255,256).astype(np.uint8),cv2.COLORMAP_JET)
full_map_RGB = cv2.cvtColor(full_map_BGR, cv2.COLOR_BGR2RGB)
full_map = full_map_RGB[:,0,:] /255
cmap= mpl.colors.ListedColormap(full_map)
'''

# LINEAT BLUE - GREEN - YELLO - RED

cmap = LinearSegmentedColormap.from_list('cloudcompare_blue_green_yellow_red_cmap',
                                         ['blue', (0.0, 1.0, 0.0), 'yellow','red'])
norm= mpl.colors.Normalize(vmin=0,vmax=65)

fig1, fig1_axes = plt.subplots(2,1)
cb_ax = fig1.add_axes([0.125, 0.08, 0.125, 0.9-0.08])

fig1_axes[0].set_axis_off()
fig1_axes[1].set_axis_off()

cb = mpl.colorbar.ColorbarBase(cb_ax, cmap=cmap,
                                       norm=norm, orientation='vertical')
loc = plticker.MultipleLocator(base=5.0) # this locator puts ticks at regular intervals
cb_ax.yaxis.set_major_locator(loc)

fig2, fig2_axes = plt.subplots(figsize=(6, 1))
fig2.subplots_adjust(bottom=0.5)

cb = fig2.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                  cax=fig2_axes, orientation='horizontal', label='Some Units')


#cb = mpl.colorbar.ColorbarBase(cb_ax, cmap=cmap,
#                                       norm=norm, orientation='horizontal')
loc = plticker.MultipleLocator(base=5.0) # this locator puts ticks at regular intervals
cb.ax.xaxis.set_major_locator(loc)
cb.ax.tick_params(labelsize=10, width=2, length=6)
cb.set_label('Height (m)')
plt.tight_layout()
save_fig2png(fig2, fname='cloudcompare_bgyr_colorscale_horizontal', size=[8, 1])


# LOG BLUE - GREEN - YELLO - RED

cmap = LinearSegmentedColormap.from_list('cloudcompare_blue_green_yellow_red_cmap',
                                         ['blue', (0.0, 1.0, 0.0), 'yellow','red'])

norm= mpl.colors.LogNorm(vmin=0.01,vmax=40)

fig3, fig3_axes = plt.subplots(figsize=(4, 1))
fig3.subplots_adjust(bottom=0.5)

cb = fig3.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                  cax=fig3_axes, orientation='horizontal', label='Some Units')

tick_locs = np.array([0.01, 0.1, 1, 5, 10, 20, 40])
#tick_values = np.log10(tick_locs)
tick_labels = ["{:.2g}".format(t) for t in tick_locs]
cb.set_ticks(tick_locs)
cb.set_ticklabels(tick_labels)
cb.ax.tick_params(labelsize=10, width=2, length=6)
cb.ax.tick_params(which = 'minor', length=4)
cb.set_label('Log Height Difference (m)')
plt.tight_layout()
save_fig2png(fig2, fname='cloudcompare_bgyr_log_colorscale_horizontal', size=[4, 1])
#cb.set_major_formatter(FormatStrFormatter('%.2f'))
