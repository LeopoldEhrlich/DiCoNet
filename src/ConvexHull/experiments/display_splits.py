import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np

import colorsys 

import matplotlib
matplotlib.use('TkAgg')

class SplitVisualizer:
    def __init__(self, pair, plots_per_page=3):
        """
        Initialize the visualizer with pagination controls for comparing two sets of splits
        
        Args:
            pair: pair of (input, e, scales) tuples
            plots_per_page: Number of subplots per page (per dataset)
        """
        out1, self.input1, self.scales1 = pair[0]
        out2, self.input2, self.scales2 = pair[1]

        cvt_out = lambda out, idx, batch : [x[batch].data.cpu().numpy() for x in out[idx]][1:]

        self.phi1 = cvt_out(out1,0,0)
        self.phi2 = cvt_out(out2,0,0)

        self.input1 = self.input1[0,:,:]
        self.input2 = self.input2[0,:,:]

        print(self.input1.shape)

        self.plots_per_page = plots_per_page
        self.current_page = 0
        self.total_pages = (len(self.phi1) + plots_per_page - 1) // plots_per_page
        self.fig = None
        
    def plot_page(self, page):
        """Plot a specific page of results with both datasets"""
        if self.fig:
            plt.close(self.fig)

        print(page,len(self.phi1))

        n_cols = min(3, len(self.phi1) - 3*page)
        n_rows = 2  # Two rows (one for each dataset)

        fig_width = n_cols * 4  
        fig_height = n_rows * 4
        self.fig, self.axes = plt.subplots(n_rows, n_cols, 
                                         figsize=(fig_width, fig_height),squeeze=False)
        
        plt.subplots_adjust(
            left=0.1, right=0.9,  
            bottom=0.15, top=0.85,
            wspace=0.4, hspace=0.4  
        )

        start_idx = page * self.plots_per_page
        end_idx = min((page + 1) * self.plots_per_page, len(self.phi1))

        for i in range(start_idx,end_idx):
            ax = self.axes[0, i%n_cols]
            self._plot_single_split(ax, self.input1, self.phi1[i], i, end_idx)

            ax = self.axes[1, i%n_cols]
            self._plot_single_split(ax, self.input2, self.phi2[i], i, end_idx)
            
        
        # Add page navigation buttons
        ax_prev = plt.axes([0.3, 0.05, 0.1, 0.05])
        ax_next = plt.axes([0.6, 0.05, 0.1, 0.05])
        self.btn_prev = Button(ax_prev, 'Previous', color='lightgray')
        self.btn_next = Button(ax_next, 'Next', color='lightgray')
        
        self.btn_prev.on_clicked(self.prev_page)
        self.btn_next.on_clicked(self.next_page)
        
        self.fig.suptitle(f'Split Visualizations (Page {page + 1}/{self.total_pages})', fontsize=14, y=0.98)
        plt.show()
    
    def _plot_single_split(self, ax, input_data, phi, idx, end_idx):
        """Helper method to plot a single split on a given axis"""
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # Add minor ticks for better grid
        ax.set_xticks(np.linspace(0, 1, 11), minor=True)
        ax.set_yticks(np.linspace(0, 1, 11), minor=True)
        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.5)

        if idx >= end_idx:
            ax.axis('off')
            return
        
        groups = np.transpose(np.unique(phi,axis=1)).astype(bool)

        # Remove the group of uncounted padding points
        groups = groups[np.any(groups,1),:][-2:]

        colors = self.generate_tree_colormap(groups.shape[0])


        for i, grp in enumerate(groups):
            pts = input_data[grp,:]
            ax.scatter(pts[:, 0], pts[:, 1], color=colors[i], 
                alpha=0.7, edgecolor='w', linewidth=0.5, label=f'Split {i}')     
        
        ax.set_title(f'Split {idx + 1}', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_facecolor('#f5f5f5')
        ax.set_aspect('equal')
    
    def next_page(self, event):
        """Navigate to next page"""
        if self.current_page < self.total_pages - 1:
            self.current_page += 1
            self.plot_page(self.current_page)
    
    def prev_page(self, event):
        """Navigate to previous page"""
        if self.current_page > 0:
            self.current_page -= 1
            self.plot_page(self.current_page)
    
    def show(self):
        """Start the visualization"""
        self.plot_page(0)

    def generate_tree_colormap(self, n, base_hue=0.0, base_sat=0.6, base_light=0.6):
        """
        Generate a colormap with n colors in a tree-structured hue division, 
        so that leaves are colored similarly based on how far apart they diverged.

        Returns: List of RGB tuples in [0,1] format.
        """
        def recurse(depth, max_depth, hue_start, hue_end, sat, light):
            if depth == max_depth:
                hue = (hue_start + hue_end) / 2
                return [colorsys.hls_to_rgb(hue % 1.0, light, sat)]
            
            else:
                mid = (hue_start + hue_end) / 2
                # Slightly vary saturation and lightness at each level
                sat_delta = 0.05
                light_delta = 0.05
                left = recurse(depth + 1, max_depth, hue_start, mid,
                            min(1.0, sat + sat_delta), min(1.0, light + light_delta))
                right = recurse(depth + 1, max_depth, mid, hue_end,
                                max(0.0, sat - sat_delta), max(0.0, light - light_delta))
                return left + right

        max_depth = np.ceil(np.log2(n))
        return recurse(0, max_depth, base_hue, base_hue + 1.0, base_sat, base_light)

