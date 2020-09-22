from hpa import *
from sample_map import *
from dataset import Dataset

class VarAcc(object):
    def __init__(self, dataset, partitioner, max_depth, p_attrs, t_attrs, max_leaf = 0, va_cls = HPA):
        self.dataset = dataset
        self.partitioner = partitioner
        self.sample_tree = None
        self.max_depth = max_depth
        self.p_attrs = p_attrs
        self.t_attrs = t_attrs
        self.hpa = va_cls(self.dataset, self.partitioner, self.max_depth, self.p_attrs, self.t_attrs, max_leaf = max_leaf)

    def initialize(self):
        self.hpa.build_pa_tree()
        # self.hpa.plot_tree(verbose=False)
        self.hpa.verify_tree(self.t_attrs[0])
        self.sample_map = LeafOnlySampleMap(self.hpa)
        self.sample_map.build_sample_map()
