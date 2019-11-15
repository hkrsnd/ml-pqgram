import seaborn as sns
import sklearn
import sklearn.manifold as skm
import numpy as np
import matplotlib.pyplot as plt


class Visualize():
    """
    Visualize is a class to visualization methods.
    """

    def __init__(self, problem):
        self.problem = problem
        sns.set_style('white')

    def mds_embed(self, trees):
        """
        Return embedded points of trees using the pq-gram distance.
        """
        N = len(trees)
        distance_mat = np.zeros((N, N))
        for i in range(N):
            for j in range(i+1):
                distance_mat[i][j] = self.problem.pq_gram_distance(trees[i], trees[j])

        for j in range(N):
            for i in range(j+1):
                distance_mat[i][j] = distance_mat[j][i]

        mds = skm.MDS(n_components=2, dissimilarity='precomputed')
        XY = mds.fit_transform(distance_mat)
        return XY, distance_mat

    def mds_plot(self, trees, labels, epoch=0):
        """
        Plot the embedding space using  Multi-Dimensional Scaling.
        """
        Xs, distance_mat = self.mds_embed(trees)
        label_set = sorted(list(set(labels)))
        X_dic = {}
        for l in label_set:
            X_dic[l] = []
        for i in range(len(trees)):
            X_dic[labels[i]].append(Xs[i])
        plt.figure(figsize=(6, 6))
        k = 1
        plt.tick_params(labelbottom=False,
                        labelleft=False,
                        labelright=False,
                        labeltop=False)
        color_list = ['orange', 'dogerblue', 'gold', 'deeppink']
        for X in X_dic.values():
            Xp = np.array(X)
            plt.scatter(Xp[:, 0], Xp[:, 1], linewidth=1, edgecolors='black', label='class_'+str(k))
            k += 1
        leg = plt.legend(frameon=True, bbox_to_anchor=(1, 1), loc='upper right')
        leg.get_frame().set_edgecolor('black')
        plt.title('MDS embedding')
        plt.show()
        plt.figure(figsize=(6, 6))
        plt.imshow(distance_mat)
        plt.colorbar()
        plt.show()

    def mds_gif_imgs(self, trees, labels, epoch=0):
        """
        Generate mds embedding images to create gif.
        """
        Xs, distance_mat = self.mds_embed(trees)
        label_set = sorted(list(set(labels)))
        X_dic = {}
        for l in label_set:
            X_dic[l] = []
        for i in range(len(trees)):
            X_dic[labels[i]].append(Xs[i])
        plt.figure(figsize=(6, 6))
        k = 1
        for X in X_dic.values():
            Xp = np.array(X)
            plt.scatter(Xp[:, 0], Xp[:, 1], linewidth=1, edgecolors='black', label='class_'+str(k))
            k += 1
        plt.savefig('img/embedding/mds_epoch_'+str(epoch)+'.png')
        plt.show()
