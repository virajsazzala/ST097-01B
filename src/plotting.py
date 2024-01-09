import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def display_plot(nparray):
    plt.grid(False)
    plt.imshow(nparray)
    plt.show()


def show_computed_path(plt, datahandler):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection="3d")
    ax.view_init(elev=-20, azim=270)
    xs = datahandler.gt[:, 0, 3]
    ys = datahandler.gt[:, 1, 3]
    zs = datahandler.gt[:, 2, 3]
    ax.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))
    ax.plot(xs, ys, zs, c="grey")
