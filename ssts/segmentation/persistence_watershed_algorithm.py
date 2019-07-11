import numpy as np


class PersistenceWatershed(object):
    def __init__(self, arr):
        self.WSHED = 0
        self.INIT = -1
        self.arr = arr
        self.dim = self.arr.shape
        self.lab = np.empty(self.dim, dtype=int)
        self.lab.fill(self.INIT)
        self.dual = {}
        self.current_label = 1
        self.maxes = {}
        self.mt = None

    def train(self, pers_thresh):
        rank = np.argsort(self.arr.flatten())
        for r in rank[::-1]:
            p = np.unravel_index(r, self.dim)
            L = self.get_neighbor_labels(p)

            if len(L) == 0:
                # assign new label
                l = self.current_label
                self.current_label += 1

                self.lab[p] = l
                self.dual[l] = {}
                self.maxes[l] = self.arr[p]

            elif len(L) == 1:
                # assign existing label
                self.lab[p] = list(L)[0]

            else:
                # Watershed - Connect adjacent labels
                w = self.arr[p]
                self.lab[p] = self.WSHED

                for l0 in L:
                    for l1 in L:
                        if l0 == l1:
                            continue

                        n0, n1 = [min(l0, l1), max(l0, l1)]
                        if n1 not in self.dual[n0]:
                            self.dual[n0][n1] = -1 * np.inf

                        self.dual[n0][n1] = max(self.dual[n0][n1], w)

        # Create edge list
        edges = []
        for n0 in self.dual:
            for n1 in self.dual[n0]:
                edges.append((n0, n1, self.dual[n0][n1]))

        self.mt = PersistenceWatershed.merge_tree(self.maxes, edges, pers_thresh)

    def apply_threshold(self, t):
        # Build relabel dictionary
        relabel = {i: i for i in range(self.current_label)}
        for e in self.mt:
            if e[2] < t:
                relabel[e[0]] = e[1]

        # Perform relabeling
        rlab = np.empty(self.dim, dtype=int)
        wvox = []
        for i in range(self.lab.size):
            p = np.unravel_index(i, self.dim)
            l = self.lab[p]
            if l == self.WSHED:
                rlab[p] = 0
                wvox.append(p)
                continue
            rlab[p] = PersistenceWatershed.find(relabel, l)

        # Clean up false watersheds
        for p in wvox:
            L = self.get_neighbor_labels(p, rlab)
            if len(L) == 1:
                rlab[p] = list(L)[0]

        # Clean labels to avoid skips
        ol2cl = {k: i + 1 for i, k in enumerate(set(rlab.flatten())) if k != 0}
        ol2cl[0] = 0
        for i in range(rlab.shape[0]):
            for j in range(rlab.shape[1]):
                rlab[i, j] = ol2cl[rlab[i, j]]

        return rlab

    def get_neighbor_labels(self, p, source=None):
        N = PersistenceWatershed.get_neighbors(p, self.arr.shape)
        L = set()
        for n in N:
            l = source[n] if source is not None else self.lab[n]
            if l != self.INIT and l != self.WSHED:
                L.add(l)
        return L

    @staticmethod
    def get_neighbors(p, bounds, k=1):
        N = [(p[0] + i, p[1] + j) for i in range(-k, k + 1) for j in range(-k, k + 1)]
        N = [n for n in N if (PersistenceWatershed.inbounds(n, bounds) and not (n[0] == p[0] and n[1] == p[1]))]
        return N

    @staticmethod
    def inbounds(p, bounds):
        return np.prod([0 <= p[i] < bounds[i] for i in range(2)])

    @staticmethod
    def find(components, u):
        while components[u] != u:
            u = components[u]
        return u

    @staticmethod
    def merge_tree(maxima, edges, pers_thresh):
        # maxima=(label, max value of label in data)
        # edges=(label1, label2, max value of boundary in data)
        pairs = []
        values = {}
        components = {}

        for (u, val) in maxima.items():
            values[u] = val
            components[u] = u

        # Sort by edge height
        edges.sort(key=lambda x: x[2], reverse=True)

        for e in edges:
            u, v, val = e
            uc = PersistenceWatershed.find(components, u)
            vc = PersistenceWatershed.find(components, v)
            if uc == vc:
                continue

            if values[vc] < values[uc]:  # if later label has lower max
                uc, vc = vc, uc

            pairs.append((uc, vc, values[uc] - val)) # source, target, persistence (size of lower label hill)

            # NOTE without this constraint, artifacts appear in the output plot
            if (values[uc] - val) < pers_thresh:
                components[uc] = components[vc]  # lower label maps to higher label

        return pairs

