import random
import sys
sys.path.append('../src/lib/')
from tree import Node

class DataGen():

    def __init__(self, train_N=100, test_N=100):
        self.train_N = train_N
        self.test_N = test_N
    
    def gen_small_block(self):
        if random.random() < 0.5:
            node = Node('A')
        else:
            node = Node('B')
        
        if random.random() < 0.5:
            node.addkid(Node('C'))
        else:
            node.addkid(Node('D'))
        
        if random.random() < 0.5:
            node.children[-1].addkid(Node('A'))
        else:
            node.children[-1].addkid(Node('B'))
        return node

    def random_node(self):
        r = random.random()
        if r < 0.25:
            node = Node('A')
        elif r < 0.5:
            node = Node('B')
        elif r < 0.75:
            node = Node('C')
        else:
            node = Node('D')
        return node

    def gen_class1(self):
        b1 = self.gen_small_block()
        b2 = self.gen_small_block()
        b3 = self.gen_small_block()
        b1.addkid_leaf(b2).addkid_leaf(b3)
        return b1
   
    def gen_class2(self):
        root = self.random_node()
        root.addkid(self.random_node())
        for i in range(7):
            root.addkid_leaf(self.random_node())
        return root

    def generate(self):
        n_train = int(self.train_N / 2)
        n_test = int(self.train_N / 2)
        X_train = [self.gen_class1() for i in range(n_train)] + [self.gen_class2() for i in range(n_train)]
        y_train = [1 for i in range(n_train)] + [-1 for i in range(n_train)]
        X_test = [self.gen_class1() for i in range(n_test)] + [self.gen_class2() for i in range(n_test)]
        y_test = [1 for i in range(n_test)] + [-1 for i in range(n_test)]

        return X_train, y_train, X_test, y_test
