import numpy as np
from matplotlib import pyplot as plt


def plot_distribution(data):


    plt.figure(figsize=(10, 6))

    # Plot histogram
    n, bins, patches = plt.hist(data, bins=20, edgecolor='black', alpha=0.7, color='steelblue')

    # Add kernel density line
    from scipy import stats
    kde = stats.gaussian_kde(data)
    x_range = np.linspace(0, 1, 200)
    kde_values = kde(x_range) * len(data) * (bins[1] - bins[0])
    plt.plot(x_range, kde_values, 'r-', linewidth=2, label='Kernel Density', alpha=0.7)

    plt.xlabel('Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.xlim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()  # Add legend to show the KDE line
    plt.show()


def plot_distribution_overlayed(data_list, labels=None, colors=None, show_kde=True,name='Dataset'):
    """
    Plot multiple overlayed histograms with optional KDE lines.

    Args:
        data_list: List of arrays/tensors (can have different sizes)
        labels: List of labels for legend (optional)
        colors: List of colors for each dataset (optional)
        show_kde: Whether to show KDE lines (default: True)
    """
    plt.figure(figsize=(10, 6))

    # Default colors if not provided
    if colors is None:
        colors = ['steelblue', 'coral', 'forestgreen', 'purple', 'goldenrod',
                  'darkred', 'darkcyan', 'hotpink', 'navy', 'olive']

    # Default labels if not provided
    if labels is None:
        labels = [f'{name} {i + 1}' for i in range(len(data_list))]

    from scipy import stats

    # Plot each dataset
    for i, data in enumerate(data_list):
        # Convert to numpy if it's a tensor
        if hasattr(data, 'numpy'):
            data = data.numpy()

        # Plot histogram with transparency
        counts, bins, patches = plt.hist(data, bins=20, alpha=0.5,
                                         edgecolor='black', linewidth=0.5,
                                         color=colors[i % len(colors)],
                                         label=labels[i])

        # Add KDE line for each dataset
        if show_kde and len(data) > 1:
            kde = stats.gaussian_kde(data)
            x_range = np.linspace(0, 1, 200)
            kde_values = kde(x_range) * len(data) * (bins[1] - bins[0])
            plt.plot(x_range, kde_values, color=colors[i % len(colors)],
                     linestyle='--', linewidth=2, alpha=0.7)

    # plt.xlabel('Value', fontsize=12)
    # plt.ylabel('Frequency', fontsize=12)
    plt.xlim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
