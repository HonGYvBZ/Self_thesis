# https://github.com/rlcode/per
import random
import numpy as np


# SumTree
# a binary tree data structure where the parent’s value is the sum of its children
class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # check if data exista
    def existornot(self, instance, nextinstance):
        collectall = []
        # print("self.n_entries",self.n_entries,self.data )
        if self.n_entries == 0:
            return [], []
        sz = self.n_entries
        for i in range(sz):
            collectall.append(self.data[i])
        # print("collectall",collectall)
        d = np.array(collectall, dtype=object).transpose()
        # print("d",d)
        thestate = np.vstack(d[0])
        theNstate = np.vstack(d[2])
        # print("thestate",thestate)
        indexofin = np.where((thestate == instance).all(axis=1))
        indexofNin = np.where((theNstate == nextinstance).all(axis=1))
        # if indexofin[0].size>1:
        # print("indexofin",indexofin, thestate)
        # quit()
        return indexofin, indexofNin

    # store priority and sample
    def add(self, p, data, dataindex, update):
        if update == 0:
            idx = self.write + self.capacity - 1

            self.data[self.write] = data
            self.update(idx, p)

            self.write += 1
            if self.write >= self.capacity:
                self.write = 0

            if self.n_entries < self.capacity:
                self.n_entries += 1
        else:
            # dataindx = idx - self.capacity + 1
            dcheck = self.data[dataindex]
            dcheck = np.array(dcheck, dtype=object).transpose()
            # print("dd1", data[1])
            # print("dd2", dcheck[0][1])
            if data[1] > dcheck[0][1]:  # reward update
                # print("dd", dcheck[0][1],dataindex[0][0],self.write)
                self.data[dataindex] = data  # data  [0][0]
                dcheck = self.data[dataindex]
                dcheck = np.array(dcheck, dtype=object).transpose()
                # print("dd2after", dcheck[0][1])
                idx = dataindex + self.capacity - 1
                self.tree[idx] = 0  # reset priority
                self.update(idx, p)
            '''   
            #elif data[1] == dcheck[0][1]:# if they have the same reward, then check if their next state is different
                #print("reward", data[1], dcheck[0][1])
            if not (data[3] == dcheck[0][3]).all():# save it if their next states are different
                #print("different next items/ states", data[3], dcheck[0][3])
                idx = self.write + self.capacity - 1

                self.data[self.write] = data
                self.update(idx, p)

                self.write += 1
                if self.write >= self.capacity:
                    self.write = 0

                if self.n_entries < self.capacity:
                    self.n_entries += 1
            '''
            # update priority

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])

class Memory:  # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def addorupdate(self, error, sample):
        dsearch = []
        update = 0
        self.add(error, sample, dsearch, update)  # add as new entry

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def add(self, error, sample, dindex, update):
        p = self._get_priority(error)
        self.tree.add(p, sample, dindex, update)

    def sample(self, n):
        batch = []
        idxs = []
        is_weight = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)
        # print("priorities =",priorities)
        # sampling_probabilities = priorities/ self.tree.total()
        priorities_arr = np.array(priorities)
        sampling_probabilities = (priorities_arr + 1e-9) / (self.tree.total() + 1e-9)
        # print("sampling_probabilities =",sampling_probabilities)
        # zero=0
        # if self.tree.total()!=0 and min(priorities)!=0:
        # zero=1
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()
        # print("is_weight = ",is_weight)

        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)
