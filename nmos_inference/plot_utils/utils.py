import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import numpy as np


def diy_cmap_by_list(self, list_of_colors=[(0, '#F8B9B9')], cmap_name="my_cmap", N=256):
    """
    Create a custom color map.
    """
    cmap = colors.LinearSegmentedColormap.from_list(cmap_name, list_of_colors, N)
    return cmap


def get_cmap_slice(cmap, start, stop, n=256, name='my_slice'):
    '''
    Create a slice of a colormap.
    '''
    return colors.LinearSegmentedColormap.from_list(name, cmap(np.linspace(start, stop, cmap.N)),N=n)


def digits2color(digits):
    d_group = [digits[i:i+3] for i in range(0, len(digits), 3)]
    colors = [tuple((np.array(d)/255).tolist()) for d in d_group]
    return colors
