import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

if __name__ == "__main__":
    # data = np.loadtxt('test-results.txt', delimiter=',')
    # zeros = np.zeros((200, 200))
    #
    # for row in range(len(data)):
    #     current_index = zeros[int(round(data[row][0], 2) * 100)][int(data[row][1])]
    #     if current_index != 0:
    #         current_index = (current_index + data[row][2]) / 2
    #     else:
    #         zeros[int(round(data[row][0], 2) * 100)][int(data[row][1])] = data[row][2]
    #
    # zeros = zeros[~np.all(zeros == 0, axis=1)]
    # data = zeros[:, ~np.all(zeros == 0, axis=0)]
    #
    # ind = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
    # cols = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    #
    # plt.figure(figsize=(8, 4))
    #
    # print(data)
    #
    # df = pd.DataFrame(data[:1], columns=cols)
    # dfe = pd.DataFrame(data.T, columns=ind)
    #
    # ax = sns.lineplot(data=df, dashes=False)
    # ax.set(xlabel='feedback strength', ylabel='error')
    # sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    # plt.show()
    #
    #
    # ax = sns.lineplot(data=dfe, dashes=False)
    # ax.set(xlabel='tau', ylabel='error')
    # sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    # plt.show()
    #
    # ax = sns.heatmap(df, square=True, cmap="viridis")
    # ax.set(xlabel='tau', ylabel='feedback strength')
    # plt.show()

    data = np.loadtxt('test-results.csv', delimiter=',')

    ind = [.4,  .5,  .6,  .7,  .8,  .9, 1.0, 1.1,1.2,1.3,]
    cols = [15,16,17,18,19,20,21,22,23,24]
    # cols = [.4,  .5,  .6,  .7,  .8,  .9, 1.0, 1.1,1.2,1.3]

    df = pd.DataFrame(data, columns=cols, index=ind)

    ax = sns.heatmap(df, square=True, cmap="viridis")
    ax.set(xlabel='Number of Virtual Nodess', ylabel='input strength')
    plt.show()
