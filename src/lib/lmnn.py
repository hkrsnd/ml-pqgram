import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import operator
import torch
import random
from tqdm import tqdm_notebook as tqdm
import heapq
import torch

from visualize import Visualize


class LMNN():
    """
    LMNN represents Large Margin Nearest Neighbor metric learning scheme.
    """

    def __init__(self, problem, k=3, target_k=3, lr=1e-2, wd=1e-3, epoch=500, b=50, margin=5.0, push_margin=5.0):
        """
        problem: the Problem class
        k: the number of neighbors of k-nearest neighbor classification
        target_k: the number of neighbor for the Large Margin Nrearest Neighbor learning
        lr: learning rate for gradient descent using Adam optimizer
        wd: the weight decay parameter for gradient descent
        epoch: the number of epochs
        b: interval for executing test
        margin: pull margin for positive pairs
        push_margin: push margin for negative pairs
        """
        self.problem = problem
        self.X_train = problem.X_train
        self.y_train = problem.y_train
        self.X_test = problem.X_test
        self.y_test = problem.y_test
        self.k = k
        self.target_k = target_k
        self.lr = lr
        self.wd = wd
        self.epoch = epoch
        self.b = b
        self.margin = torch.tensor([margin], dtype=torch.float64)
        self.push_margin = torch.tensor([push_margin], dtype=torch.float64)
        self.target_dic = {}
        self.imposter_dic = {}
        self.set_optimizer(self.problem.get_params())
        self.create_pairs()
        self.vis = Visualize(problem)

    def set_optimizer(self, params):
        """
        Set weight parameters as pytorch parameters.
        """
        self.optimizer = torch.optim.Adam(
            params, lr=self.lr, weight_decay=self.wd)

    def create_pairs(self):
        """
        Create positive and negative pairs from neighbors.
        """
        print('set targets and imposters')
        for i in tqdm(range(len(self.X_train))):
            self.set_targets_imposters(
                self.X_train[i], self.y_train[i], self.X_train, self.y_train, n=self.target_k)
        for i in tqdm(range(len(self.X_test))):
            self.set_targets_imposters(
                self.X_test[i], self.y_test[i], self.X_test, self.y_test, n=self.target_k)

    def get_neigbors(self, dist_list, n):
        """
        Get neighbors using min heap.
        """
        heap_elts = [(item[1], item) for item in dist_list]
        heapq.heapify(heap_elts)
        neighbors = []
        for i in range(n):
            try:
                neighbors.append(heapq.heappop(heap_elts)[1])
            except IndexError:
                break
        return neighbors

    def set_targets_imposters(self, tree, label, X_train, y_train, n):
        """
        Set targets and imposters for input tree.
        The list of targets and neighbors are saved as dictionary from a tree to a list.
        """
        same_dist_list = [(i, self.problem.calc_distance(tree, X_train[i])) for i in range(
            len(X_train)) if y_train[i] == label and tree != X_train[i]]
        diff_dist_list = [(i, self.problem.calc_distance(tree, X_train[i]))
                          for i in range(len(X_train)) if y_train[i] != label]
        target_dist_list = self.get_neigbors(same_dist_list, n)
        target_indexes = [x[0] for x in target_dist_list]
        target_trees = [X_train[i] for i in target_indexes]
        if len(target_dist_list) > 0:
            max_dist = target_dist_list[-1][1]
        else:
            max_dist = 0.0
        imposter_indexes = [x[0]
                            for x in diff_dist_list if x[1] < max_dist + self.margin]
        self.target_dic[tree] = target_indexes
        self.imposter_dic[tree] = imposter_indexes

    def set_imposters(self, tree, label, X_train, y_train, n):
        """
        Set imposters for input tree to update training pairs.
        """
        same_dist_list = [(i, self.problem.calc_distance(tree, X_train[i])) for i in range(
            len(X_train)) if y_train[i] == label and tree != X_train[i]]
        diff_dist_list = [(i, self.problem.calc_distance(tree, X_train[i]))
                          for i in range(len(X_train)) if y_train[i] != label]
        target_dist_list = self.get_neigbors(same_dist_list, n)
        target_indexes = [x[0] for x in target_dist_list]
        target_trees = [X_train[i] for i in target_indexes]
        if len(target_dist_list) > 0:
            max_dist = target_dist_list[-1][1]
        else:
            max_dist = 0.0
        imposter_indexes = [x[0]
                            for x in diff_dist_list if x[1] < max_dist + self.margin]
        self.imposter_dic[tree] = imposter_indexes

    def update_imposters(self):
        """
        Update imposters.
        Call set_impostors function for every training point.
        """
        for i in range(len(self.X_train)):
            tree = self.X_train[i]
            label = self.y_train[i]
            self.set_imposters(tree, label, self.X_train, self.y_train, self.k)

    def compute_loss(self, X, y):
        """
        Returns a loss for lmnn: sum of push and pull loss.
        """
        loss = torch.tensor([0.0], dtype=torch.float64)
        for i in range(len(X)):
            loss += self.margin_loss(X[i], y[i], X, y, n=self.k)
        return loss

    def margin_loss(self, tree, label, X_train, y_train, n):
        """
        Compute the push loss and pull loss for lmnnn.
        """
        loss = torch.tensor([0.0], dtype=torch.float64)
        target_trees = [X_train[i] for i in self.target_dic[tree]]
        imposters = [X_train[i] for i in self.imposter_dic[tree]]
        target_dist_dic = {}
        zero = torch.tensor([0.0], dtype=torch.float64)
        diff = zero

        target_loss = torch.tensor([0.0], dtype=torch.float64)
        imposter_loss = torch.tensor([0.0], dtype=torch.float64)

        for target_tree in target_trees:
            diff = self.problem.calc_distance(tree, target_tree)
            target_dist_dic[target_tree] = diff
            target_loss += torch.max(zero,  diff-self.margin)

        for imposter in imposters:
            push = self.push_margin - \
                self.problem.calc_distance(tree, imposter)
            imposter_loss += torch.max(zero, push)
        return target_loss + imposter_loss

    def learn(self):
        """
        Learn the pq-gram distance from examples.
        """
        train_acc_list = []
        test_acc_list = []
        best_acc = 0.0

        print('BEFORE METRIC LEARNING')
        predicted = self.k_nearest_neighbor(
            self.X_train, self.y_train, self.X_test)
        test_acc = accuracy_score(predicted, self.y_test)
        print('Initial Test Accuracy: ', test_acc)
        test_acc_list.append(test_acc)
        self.vis.mds_plot(self.X_train, self.y_train)

        for e in tqdm(range(self.epoch)):
            loss = self.compute_loss(self.X_train, self.y_train)
            # loss_list.append(loss.item())
            if e % self.b == 0 and e > 0:
                # UPDATE imposrters
                print("UPDATE IMPOSTERS")
                self.update_imposters()
            # UPDATE PARAMS
            loss.backward(retain_graph=True)
            self.optimizer.step()

        # TEST acc
        predicted = self.k_nearest_neighbor(
            self.X_train, self.y_train, self.X_test)
        test_acc = accuracy_score(predicted, self.y_test)
        print('Test Accuracy: ', test_acc)
        # PLOT DISTANCE SPACE
        self.vis.mds_plot(self.X_train, self.y_train)

    def k_nearest_neighbor(self, train_X, train_y, test_X):
        """
        Execute the k-nearest neighbor classification.
        """
        predicted = []
        for i, tx in enumerate(test_X):
            dist_list = [(i, self.problem.calc_distance(tx, train_X[i]))
                         for i in range(len(train_X))]
            close_list = self.get_neigbors(dist_list, self.k)
            class_count = {}
            for label in self.problem.labels:
                class_count[label] = 0
            for c in close_list:
                x_index = c[0]
                class_count[train_y[x_index]] += 1
            predicted.append(
                max(class_count.items(), key=operator.itemgetter(1))[0])
        return predicted

    def get_gif_imgs(self):
        """
        Learn and test for visualization.
        """
        loss_list = []
        train_acc_list = []
        test_acc_list = []
        best_acc = 0.0
        best_params = self.problem.params

        print('BEFORE METRIC LEARNING')
        predicted = self.k_nearest_neighbor(
            self.X_train, self.y_train, self.X_test)
        test_acc = accuracy_score(predicted, self.y_test)
        print('test_acc (init): ', test_acc)
        test_acc_list.append(test_acc)
        self.vis.mds_gif_imgs(self.X_train, self.y_train, epoch=0)

        for e in tqdm(range(self.epoch)):
            loss = self.compute_loss(self.X_train, self.y_train)
            loss_list.append(loss.item())
            if e % self.b == 0 and e > 0:
                # TEST acc
                predicted = self.k_nearest_neighbor(
                    self.X_train, self.y_train, self.X_test)
                test_acc = accuracy_score(predicted, self.y_test)
                print('test_acc: ', test_acc)
                predicted_ = self.k_nearest_neighbor(
                    self.X_train, self.y_train, self.X_train)
                train_acc = accuracy_score(predicted_, self.y_train)
                print('train_acc: ', train_acc)
                if test_acc > max(test_acc_list) and len(test_acc_list) > 0:
                    best_params = self.problem.params
                    best_acc = test_acc
                test_acc_list.append(test_acc)

                self.vis.mds_gif_imgs(self.X_train, self.y_train, epoch=e)

                # UPDATE imposrters
                print("UPDATE IMPOSTERS")
                self.update_imposters()
            loss.backward(retain_graph=True)
            self.optimizer.step()
