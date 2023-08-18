import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

if __name__ == "__main__":
    data = np.loadtxt('test-results.csv', delimiter=',')

    steps = 5
    cols = [0.1 + (i/steps) for i in range(steps)]
    inds = [0.1 + (i/steps) for i in range(steps)]

    df = pd.DataFrame(data, columns=cols, index=inds)

    ax = sns.heatmap(df, square=True, cmap="viridis")
    ax.set(xlabel='feedback strength', ylabel='input strength')
    plt.show()
