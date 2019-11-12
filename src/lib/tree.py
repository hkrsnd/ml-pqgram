import random
import functools as func

class Node(object):
    """
        A generic representation of a tree node. Includes a string label and a list of a children.
    """

    def __init__(self, label):
        """
            Creates a node with the given label. The label must be a string for use with the PQ-Gram
            algorithm.
        """
        self.label = label.replace('\n', '')
        self.children = list()

    def addkid(self, node, before=False):
        """
            Adds a child node. When the before flag is true, the child node will be inserted at the
            beginning of the list of children, otherwise the child node is appended.
        """
        if before:  self.children.insert(0, node)
        else:   self.children.append(node)
        return self

    def addkid_leaf(self, node):
        if self.is_leaf():
            self.addkid(node)
        else:
            self.children[-1].addkid_leaf(node)

            return self

    def is_leaf(self):
        if len(self.children) == 0:
            return True
        else:
            return False

    def size(self):
        if len(self.children) == 0:
            return 1
        else:
            return 1 + sum([x.size() for x in self.children])

    def to_list(self):
        root = self.label
        if len(self.children) > 0:
            children = [c.to_list()  for c in self.children]
        else:
            children = []
        return [root, [children]]

    def max_depth(self):
        if len(self.children) == 0:
            return 1
        else:
            child_depths = [c.max_depth() for c in self.children]
            return 1 + max(child_depths)
     
    def labels_set(self):
        if len(self.children) == 0:
            return {self.label}
        else:
            children_labels = set()
            for c in self.children:
                children_labels = children_labels | c.labels_set()
            return set([self.label]) | children_labels
        
##### Helper Methods #####
        
def split_tree(root, delimiter=""):
    """
        Traverses a tree and explodes it based on the given delimiter. Each node is split into a null
        node with each substring as a separate child. For example, if a node had the label "A:B:C" and
        was split using the delimiter ":" then the resulting node would have "*" as a parent with the
        children "A", "B", and "C". By default, this explodes each character in the label as a separate
        child node. Relies on split_node.
    """
    if(delimiter == ''):
        sub_labels = [x for x in root.label]
    else:
        sub_labels = root.label.rsplit(delimiter)
    if len(sub_labels) > 1: # need to create a new root
        new_root = Node("*", 0)
        for label in sub_labels:
            new_root.children.append(Node(label, 0))
        heir = new_root.children[0]
    else: # root wasn't split, use it as the new root
        new_root = Node(root.label, 0)
        heir = new_root
    for child in root.children:
        heir.children.extend(split_node(child, delimiter))
    return new_root

def split_node(node, delimiter):
    """
        Splits a single node into children nodes based on the delimiter specified.
    """
    if(delimiter == ''):
        sub_labels = [x for x in node.label]
    else:
        sub_labels = node.label.rsplit(delimiter)
    sub_nodes = list()
    for label in sub_labels:
        sub_nodes.append(Node(label, 0))
    if len(sub_nodes) > 0:
        for child in node.children:
            sub_nodes[0].children.extend(split_node(child, delimiter))
    return sub_nodes


def random_tree(labels=list('a'), n=10):
    root = Node(random.choice(labels))
    nodes = [root]
    for i in range(n):
        parent_index = random.randint(0,i)
        parent_node = nodes[parent_index]
        new_kid = Node(random.choice(labels))
        parent_node.addkid(new_kid)
        nodes.append(new_kid)
    return root


def filter_dict(f, d):
    return {k:v for k,v in d.items() if f(k,v)}

def tree2text(self, tree):
    # Example: "a(b(c,d)e)"
        if len(tree.children) == 0:
            return tree.label
        else:
            result = ""
            result += tree.label
            result += "("
            for c in tree.children:
                result += tree2text(c)
                if result[-1] != ')':
                    result += ","
            # delete last ','
            if result[-1] == ',':
                ls = list(result)
                ls[-1] = ')'
                result = ''.join(ls)
            else:
                result += ")"
            return result


def random_tree_(n=8, max_depth=20, max_width=20, labels=list('a')):
    root = Node(random.choice(labels))
    nodes = [root]
    depth_map = {root: 1}
    
    for i in range(n):
        node_candidates = list(filter_dict(lambda k,v: v<max_depth, depth_map).keys())
        parent_node = random.choice(node_candidates)
        # repeat until satisfying the conditions
        while len(parent_node.children) >= max_width:
            parent_node = random.choice(node_candidates)
        new_kid = Node(random.choice(labels))
        parent_node.addkid(new_kid)
        nodes.append(new_kid)
        depth_map[new_kid] = depth_map[parent_node] + 1
    return root


def generate_test_trees(N=30):
    ab = list('ab')
    cd = list('cd')

    trees1 = []
    trees2 = []

    for i in range(N):
        root1 = Node(random.choice(ab))
        nodes = [root1]
        for i in range(5):
            nodes[-1].addkid(Node(random.choice(ab)))
        nodes[-1].addkid(Node(random.choice(cd)))
        for i in range(5):
            nodes[-1].addkid(Node(random.choice(ab)))

        root2 = Node(random.choice(ab))
        nodes_ = [root2]
        for i in range(4):
            nodes_[-1].addkid(Node(random.choice(ab)))
	        
        nodes_[-1].addkid(Node(random.choice(cd)))
	    
        for i in range(6):
            nodes_[-1].addkid(Node(random.choice(ab)))

        trees1.append(root1)
        trees2.append(root2)
	
    return trees1 + trees2
