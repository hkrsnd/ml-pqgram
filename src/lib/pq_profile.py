import copy


class Profile(object):
    """
    Profile represents the pq-gram profiles.
    """

    def __init__(self, root, p=2, q=2):
        """
        Create pq-gram profile from given p and q values.
        """
        super(Profile, self).__init__()
        ancestors = ShiftRegister(p)
        self.list = list()

        self.build_profile(root, p, q, ancestors)

    def build_profile(self, root, p, q, ancestors):
        """
        Build pq-gram profile.
        """
        ancestors.shift(root.label)
        siblings = ShiftRegister(q)

        if(len(root.children) == 0):
            self.append(ancestors.concatenate(siblings))
        else:
            for child in root.children:
                siblings.shift(child.label)
                self.append(ancestors.concatenate(siblings))
                self.build_profile(child, p, q, copy.deepcopy(ancestors))
            for i in range(q-1):
                siblings.shift("*")
                self.append(ancestors.concatenate(siblings))
        self.list = [str(x) for x in self.list]

    def append(self, value):
        self.list.append(value)

    def __len__(self):
        return len(self.list)

    def __repr__(self):
        return str(self.list)

    def __str__(self):
        return str(self.list)

    def __getitem__(self, key):
        return self.list[key]

    def __iter__(self):
        for x in self.list:
            yield x


class ShiftRegister(object):
    """
    ShiftRegister represents the register to put node labels, which is used in the algorithm of pq-gram distance.
    """

    def __init__(self, size):
        """
        Initialize with dummy nodes.
        """
        self.register = list()
        for i in range(size):
            self.register.append("*")

    def concatenate(self, reg):
        """
            Concatenate registers.
        """
        temp = list(self.register)
        temp.extend(reg.register)
        return temp

    def shift(self, el):
        """
        The shift operation of the register.
        """
        self.register.pop(0)
        self.register.append(el)
