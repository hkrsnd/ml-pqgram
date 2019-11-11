import seaborn as sns
import sklearn
import sklearn.manifold as skm
import numpy as np
import matplotlib.pyplot as plt


class Visualize():
    def __init__(self, problem):
        self.problem = problem
        sns.set_style('white')

    def mds_embed(self, trees):
        N = len(trees)
        distance_mat = np.zeros((N, N))
        for i in range(N):
            for j in range(i+1):
                distance_mat[i][j] = self.problem.pq_gram_distance(trees[i], trees[j])
            
        for j in range(N):
            for i in range(j+1):
                distance_mat[i][j] = distance_mat[j][i]

        for j in range(N):
            for i in range(j+1):
                if distance_mat[i][j] == 0:
                    distance_mat[i][j] = distance_mat[j][i]
                    mds = skm.MDS(n_components = 2, dissimilarity='precomputed')
                    XY = mds.fit_transform(distance_mat)
        return XY, distance_mat
        #x = XY[:,0]
        #y = XY[:,1]
        #plt.figure(figsize=(16,16))
        #fig, ax = plt.subplots(figsize=(6,6))
        #plt.scatter(x,y,s=100, c=labels, alpha=0.8, cmap='viridis')
        #for i, txt in enumerate(labels):
        #    ax.annotate(str(txt), (x[i], y[i]))
        #plt.show()

    def mds_plot(self, trees, labels, epoch=0):
        Xs, distance_mat = self.mds_embed(trees)
        X_p = np.array([Xs[i] for i in range(len(trees)) if labels[i] > 0])
        X_n = np.array([Xs[i] for i in range(len(trees)) if labels[i] <= 0])
        
        plt.figure(figsize=(6, 6))
        #sns.set_style('white')
        plt.tick_params(labelbottom=False,
                        labelleft=False,
                        labelright=False,
                        labeltop=False)
        plt.scatter(X_p[:, 0], X_p[:, 1], c='orange', linewidth=1, edgecolors='black', label='class 1')
        plt.scatter(X_n[:, 0], X_n[:, 1], c='dodgerblue', linewidth=1, edgecolors='black', label='class 2')
        leg = plt.legend(frameon=True, bbox_to_anchor=(1, 1), loc='upper right')
        leg.get_frame().set_edgecolor('black')
        plt.title('MDS embedding')
        plt.show()

        plt.figure(figsize=(6,6))
        plt.imshow(distance_mat)
        plt.colorbar()
        plt.show()

 
    def mds_gif_imgs(self, trees, labels, epoch=0):
        Xs, distance_mat = self.mds_embed(trees)
        X_p = np.array([Xs[i] for i in range(len(trees)) if labels[i] > 0])
        X_n = np.array([Xs[i] for i in range(len(trees)) if labels[i] <= 0])
        
        plt.figure(figsize=(6, 6))
        #sns.set_style('white')
        plt.tick_params(labelbottom=False,
                        labelleft=False,
                        labelright=False,
                        labeltop=False)
        plt.scatter(X_p[:, 0], X_p[:, 1], c='orange', linewidth=1, edgecolors='black', label='class 1')
        plt.scatter(X_n[:, 0], X_n[:, 1], c='dodgerblue', linewidth=1, edgecolors='black', label='class 2')
        #leg = plt.legend(frameon=True, bbox_to_anchor=(1, 1), loc='upper right')
        #leg.get_frame().set_edgecolor('black')
        #plt.title('MDS embedding')
        plt.savefig('img/embedding/mds_epoch_'+str(epoch)+'.png')
        plt.show()        


    def plot_params(self):
        params = np.expand_dims(ops.soft_plus(self.params).detach().numpy(), axis=-1)
        width = len(params) // 5
        rep = np.transpose(np.repeat(params, width, axis=0).reshape(len(params), width))
        plt.figure(figsize=(16,5))
        plt.tick_params(bottom=True,
                        labelleft=False,
                        right=True,
                        top=False)
        plt.imshow(rep,  cmap='viridis')
        plt.colorbar()
        plt.show()
