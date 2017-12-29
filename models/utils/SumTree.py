import numpy
from time import time

class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = numpy.zeros( 2*capacity - 1 )

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total(self):
        return self.tree[0]

    def max(self, idx=0):
        return max(self.tree[self.capacity-1:])

    def set(self, idx, p):
        self.tree[self.capacity + idx - 1] = p

    def add(self, p):
        idx = self.write + self.capacity - 1

        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update_tree(self, idx=0):
        left = 2 * idx + 1
        right = left + 1

        if (not left >= len(self.tree)):
            self.update_tree(left)
            self.update_tree(right)
            self.tree[idx] = self.tree[left] + self.tree[right]

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return dataIdx, self.tree[idx]
