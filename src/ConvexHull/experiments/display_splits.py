import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('TkAgg')

from matplotlib.widgets import Button
import numpy as np
import torch

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
import torch

class SplitVisualizer:
    def __init__(self, input, e, scales, plots_per_page=6):
        """
        Initialize the visualizer with pagination controls
        
        Args:
            Inputs_N: Input data
            all_e: List of edge index arrays
            all_Perms: List of permutation arrays
            scales: Scale factors
            plots_per_page: Number of subplots per page
        """
        self.input = input
        self.e = e
        self.scales = scales
        self.plots_per_page = plots_per_page
        self.current_page = 0
        self.total_pages = (len(input) + plots_per_page - 1) // plots_per_page
        self.fig = None
        
    def plot_page(self, page):
        """Plot a specific page of results"""
        if self.fig:
            plt.close(self.fig)

        n_cols, n_rows = 3, 2

        fig_width = n_cols * 4  
        fig_height = n_rows * 4
        self.fig, self.axes = plt.subplots(n_rows, n_cols, 
                                         figsize=(fig_width, fig_height),
                                         squeeze=False)
        self.axes = self.axes.ravel()

        plt.subplots_adjust(
            left=0.1, right=0.9,  
            bottom=0.15, top=0.85,
            wspace=0.4, hspace=0.4  
        )

        start_idx = page * self.plots_per_page
        input = self.input
        end_idx = min((page + 1) * self.plots_per_page, len(input))
        
        colors = ('#1f77b4', '#ff7f0e')
        
        self.e = torch.sort(self.e, 1)[0]

        for i, ax in enumerate(self.axes):
            ax.set_xlim(0, 1)
            ax.set_xlim(0, 1)
            
            # Add minor ticks for better grid
            ax.set_xticks(np.linspace(0, 1, 11), minor=True)
            ax.set_yticks(np.linspace(0, 1, 11), minor=True)
            ax.grid(which='minor', alpha=0.2)
            ax.grid(which='major', alpha=0.5)

            idx = start_idx + i

            if idx >= end_idx:
                ax.axis('off')
                continue
            
            e = self.e[i].data.cpu().numpy()
            
            for node in (0, 1):
                ind = np.where(e == node)[0]
                pts = input[ind]
                ax.scatter(pts[:, 0], pts[:, 1], c=colors[node], 
                            alpha=0.7, edgecolor='w', linewidth=0.5, label=f'Class {node}')
            
            ax.set_title(f'Split {idx + 1}', fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.set_facecolor('#f5f5f5')
            
            ax.set_aspect('equal')
        
        # Add page navigation buttons
        ax_prev = plt.axes([0.3, 0.05, 0.1, 0.05])
        ax_next = plt.axes([0.6, 0.05, 0.1, 0.05])
        self.btn_prev = Button(ax_prev, 'Previous', color='lightgray')
        self.btn_next = Button(ax_next, 'Next', color='lightgray')
        
        self.btn_prev.on_clicked(self.prev_page)
        self.btn_next.on_clicked(self.next_page)
        
        self.fig.suptitle(f'Split Visualizations (Page {page + 1}/{self.total_pages})', fontsize=14, y=0.98)
        plt.show()
    
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
