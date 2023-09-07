import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

"""A small script made to produce heatmaps from an external file.
"""

if __name__ == "__main__":
    data = np.loadtxt('test-results.csv', delimiter=',')

    steps = 15
    cols = [round(0.1 + (i/steps), 2) for i in range(steps)]
    inds = [round(0.1 + (i/steps), 2) for i in range(steps)]

    print(cols)

    df = pd.DataFrame(data, columns=cols, index=inds)

    ax = sns.heatmap(df, square=True, cmap="viridis")
    ax.set(xlabel='feedback strength', ylabel='input strength')
    plt.show()
