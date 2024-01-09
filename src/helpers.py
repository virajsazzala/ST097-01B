import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def display_plot(nparray):
    plt.grid(False)
    plt.imshow(nparray)
    plt.show()
