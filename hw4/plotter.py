import numpy as np
import matplotlib.pyplot as plt


def plot():
    plt.plot(np.arange(1, 10), '-')
    plt.plot(np.arange(-1, 9), '-')
    plt.show()


if __name__ == '__main__':
    plot()
