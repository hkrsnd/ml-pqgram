import random
import functools as func


class Node(object):
    """
    Node represents a node of trees.
    """

    def __init__(self, label):
        """
        Create a node with the given label.
        """
        self.label = label.replace('\n', '')
        self.children = list()

    def addkid(self, node, before=False):
        """
        Add a child node to a given node.
        """
        if before:
            self.children.insert(0, node)
        else:
            self.children.append(node)
        return self

    def addkid_leaf(self, node):
        """
        Add a child node of the leaf node.
        """
        if self.is_leaf():
            self.addkid(node)
        else:
            self.children[-1].addkid_leaf(node)
        return self

    def is_leaf(self):
        """
        Return if a given node is the leaf node or not.
        """
        if len(self.children) == 0:
            return True
        else:
            return False

    def size(self):
        """
        Return the size of tree.
        """
        if len(self.children) == 0:
            return 1
        else:
            return 1 + sum([x.size() for x in self.children])

    def to_list(self):
        """
        Encode the tree representation into a list representation.
        """
        root = self.label
        if len(self.children) > 0:
            children = [c.to_list() for c in self.children]
        else:
            children = []
        return [root, [children]]

    def max_depth(self):
        """
        Return the maximum depth of the tree.
        """
        if len(self.children) == 0:
            return 1
        else:
            child_depths = [c.max_depth() for c in self.children]
            return 1 + max(child_depths)

    def labels_set(self):
        """
        Return the set of labels in the tree.
        """
        if len(self.children) == 0:
            return {self.label}
        else:
            children_labels = set()
            for c in self.children:
                children_labels = children_labels | c.labels_set()
            return set([self.label]) | children_labels
