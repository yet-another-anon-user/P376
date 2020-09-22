from partitioner import EqualWidthPartitioner
import pandas as pd
from utils import *
from collections import defaultdict
import sklearn, math
from sklearn.utils import shuffle

class SampleMap(object):
    def __init__(self, hpa):
        self.map = {} #a dict w/ PA's id as key.
        self.data = hpa.data
        self.hpa = hpa
        self.seed = 1234
        self.layer_partitions = {}

    def split_chunks(self, l, n):
        for i in range(0, len(l), n):
            yield l[i:i + n]

    def split_dataset(self, n_pa):
        raise NotImplementedError

    def assign_partitions_to_layers(self):
        n_pa = self.hpa.n_pa
        n_layers = len(self.hpa.layers)
        data_partitions = self.split_dataset(n_layers)
        idx = 0

        n_tuples = 0
        for layer, partitions in self.hpa.layers.items():
            self.layer_partitions[layer] = data_partitions[idx]
            idx += 1
            n_tuples += self.layer_partitions[layer].shape[0]

        assert(n_tuples == self.data.df.shape[0])
        print("SampleMap verified: assigned %s partitions (%s tuples) to %s layers.\n" % (idx, n_tuples, len(self.layer_partitions.keys())))

    def build_sample_map(self):
        self.assign_partitions_to_layers()

        for layer, nodes in self.hpa.layers.items():
            for node in nodes:
                pa = node.pa
                df = self.layer_partitions[layer]
                for attr, [l, r] in pa.predicates.items():
                    if df.shape[0] == 0:
                        continue
                    df = df[(df[attr] >= l) & (df[attr] <= r)]
                self.map[pa.id] = df

class RandomEqualSizeSampleMap(SampleMap):
    def split_dataset(self, n_pa):
        shuffled = sklearn.utils.shuffle(self.data.df, random_state = self.seed)
        n_tuples = shuffled.shape[0]
        partition_size = math.ceil(n_tuples/n_pa)
        chunks = self.split_chunks(range(0, n_tuples), partition_size)
        last_chunk = None
        ret = []
        for chunk in chunks:
            ret.append(shuffled[chunk[0]: chunk[-1]+1])
        return ret

class LeafOnlySampleMap(SampleMap):

    def build_sample_map(self):

        for node in self.hpa.leaves:

            pa = node.pa
            df = self.data.df
            for attr, [l, r] in pa.predicates.items():
                if df.shape[0] == 0:
                    continue
                df = df[(df[attr] >= l) & (df[attr] <= r)]

            self.map[pa.id] = df.sample(frac=1, random_state = self.seed)

