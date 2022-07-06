import matplotlib.pyplot as plt
import numpy as np


def plot_data(y_data, x_data=0, x_label="X_Axis", y_label="Y_axis", title="Plot", legend="Legend"):

    if x_data == 0:
        plt.plot(y_data)
    else:
        plt.plot(x_data, y_data)
    plt.title(title)
    plt.legend(legend)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()