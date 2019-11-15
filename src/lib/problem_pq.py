import torch
import tree
import numpy as np
from tqdm import tqdm
from pq_profile import Profile
from sklearn.metrics import accuracy_score, precision_score, recall_score


class ProblemPQ():
    """
    ProblemPQ represents a metric learning problem, contains distance methods.
    """

    def __init__(self, X_train, y_train, X_test=[], y_test=[], p=2, q=2):
        """
        X_train, X_test: trees for training and test
        y_train, y_test: labels for training and test        
        p, q: parameters for the size of pq-grams
        """
        self.p = p
        self.q = q
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.trees = X_train + X_test
        self.labels = y_train + y_test
        self.str_grams = []
        self.profiles = {}
        self.vecs = {}
        self.class_num = 0
        self.gram_index_dic = {}
        self.node_labels = set()
        self.gram_list = []  # corresponds to param vector

    def summary(self):
        """
        Print the summary of input dataset.
        """
        size = 0
        for t in self.trees:
            size += t.size()
        self.node_labels = self.all_nodelabels()
        print('tensor dimension: ' + str(self.dim))
        print('params: ' + str(len(self.params)))
        print('trees: ' + str(len(self.trees)))
        print('node labels: ' + str(len(self.node_labels)))
        print('average size: ' + str(size / len(self.trees)))

    def compile(self):
        """
        Compile into vector representations.
        1: extract pq-grams from input trees as profiles
        2: encode extracted profiles into pq-gram vector representations
        3: set initial parameters
        """
        print('Encoding trees into profiles ...')
        for tree in tqdm(self.trees):
            profile = Profile(tree, self.p, self.q)
            self.profiles[tree] = profile
            self.str_grams = list(set(self.str_grams).union(set(profile.list)))

        self.dim = len(self.str_grams)
        for gram in self.str_grams:
            self.gram_index_dic[gram] = 0.0
        self.gram_list = list(self.gram_index_dic.keys())

        print('Encoding trees into tensors ...')

        for tree in tqdm(self.trees):
            profile = self.profiles[tree]
            vec = self.profile_to_vector(profile)
            self.vecs[tree] = vec
        self.params = torch.tensor(np.full(self.dim, 0.5), requires_grad=True, dtype=torch.float64)
        self.best_params = self.params
        self.class_num = len(list(set(self.y_train + self.y_test)))
        self.summary()

    def get_params(self):
        """
        Return parameters.
        """
        return [self.params]

    def refresh_parameters(self):
        """
        Refresh parameters to be default.
        """
        self.params = torch.tensor(np.full(self.dim, 0.5), requires_grad=True, dtype=torch.float64)
        self.best_params = self.params

    def profile_to_vector(self, profile):
        """
        Encode pq-gram profile into pq-gram vector representation.
        """
        dic = self.gram_index_dic.copy()
        grams = profile.list
        for gram in grams:
            dic[gram] += 1.0
        tensor = torch.DoubleTensor(list(dic.values()))
        return tensor

    def calc_distance(self, tree1, tree2):
        """
        Wrapper function for computing the pq-gram distance between trees.
        """
        return self.pq_gram_distance(tree1, tree2)

    def sym_diff_vec(self, vec1, vec2):
        """
        Return the pq-gram symmetric vector.
        """
        return vec1 + vec2 - 2 * torch.min(vec1, vec2)

    def softplus_vec(self, x):
        """
        The softplus function.
        """
        return torch.log(torch.exp(x) + torch.tensor(1.0))

    def pq_gram_distance(self, tree1, tree2, normalize=False):
        """
        Compute the pq-gram distance from trees using the pq-gram vector representations.
        """
        a_w = self.softplus_vec(self.params)
        d = self.sym_diff_vec(self.vecs[tree1], self.vecs[tree2])
        return torch.dot(a_w, d)

    def all_nodelabels(self):
        """
        Return the set of node labels of input trees.
        """
        node_labels = set()
        for t in self.trees:
            s = t.labels_set()
            node_labels = node_labels | s
        return list(node_labels)
