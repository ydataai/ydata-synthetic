from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

def create_custom_colormap(cmap: cm.ScalarMappable, replacement_colors: np.ndarray) -> ListedColormap:
    """
    Creates a custom colormap by replacing a specific range of colors in the given colormap.

    :param cmap: The colormap to modify.
    :param replacement_colors: A numpy array of shape (n, 4) representing the new colors to use.
    :return: A new ListedColormap with the modified colors.
    """
    new_colors = np.concatenate([cmap(np.linspace(0, 0.24, 25)), replacement_colors, cmap(np.linspace(0.76, 1, 25))])
    return ListedColormap(new_colors)

def ydata_colormap(n: int = None) -> ListedColormap:
    """
    Returns a colormap with the YData colors and a discrete boundary norm.
    Pass n to define a truncated color map (use less colors)

    :param n: Number of colors to use in the colormap. If None, all colors will be used.
    :return: A new ListedColormap with the YData colors.
    """
    if n is not None and not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")

    colors = ["#830000", "#040404", "#FFFFFF", "#E32212"]
    colors = [np.array(color) / 255 for color in colors]

    if n is None or n >= len(colors):
        return ListedColormap(colors)

    return create_custom_colormap(cm.colors.LinearSegmentedColormap.from_list('YData', colors), np.array([[1, 1, 1, 1]] * n))

def plot_examples(cms: list[ListedColormap]) -> None:
    """
    Plots colormaps examples.

    :param cms: List of colormaps to plot.
    """
    np.random.seed(19680801)
    data = np.random.randn(30, 30)

    fig, axs = plt.subplots(1, len(cms), figsize=(6, 3), constrained_layout=True)
    for [ax, cmap] in zip(axs, cms):
        psm = ax.pcolormesh(data, cmap=cmap, rasterized=True)
        fig.colorbar(psm, ax=ax, norm=cm.colors.BoundaryNorm(np.arange(-4, 5, 1), cmap.N))
    plt.show()

if __name__ == '__main__':
    viridis = cm.get_cmap('viridis', 256)
    newcolors = np.array([[248/256, 24/256, 148/256, 1]] * 25)
    custom_viridis = create_custom_colormap(viridis, newcolors)

    plot_examples([viridis, custom_viridis, ydata_colormap(4)])
