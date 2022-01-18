from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

viridis = cm.get_cmap('viridis', 256)
newcolors = viridis(np.linspace(0, 1, 256))
pink = np.array([248/256, 24/256, 148/256, 1])
newcolors[:25, :] = pink
newcmp = ListedColormap(newcolors)

def ydata_colormap(n: int = None):
    """Returns a colormap with the YData colors and a discrete boundary norm.
    Pass n to define a truncated color map (use less colors)"""
    colors = ["#830000", "#040404", "#FFFFFF", "#E32212"]
    if n and n>len(colors):
        n=len(colors)
    return ListedColormap(colors[:n])

if __name__ == '__main__':
    def plot_examples(cms):
        """
        helper function to plot colormaps
        """
        np.random.seed(19680801)
        data = np.random.randn(30, 30)

        fig, axs = plt.subplots(1, len(cms), figsize=(6, 3), constrained_layout=True)
        for [ax, cmap] in zip(axs, cms):
            psm = ax.pcolormesh(data, cmap=cmap, rasterized=True, vmin=-4, vmax=4)
            fig.colorbar(psm, ax=ax)
        plt.show()

    plot_examples([viridis, ydata_colormap()])
