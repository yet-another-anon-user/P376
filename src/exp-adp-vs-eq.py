from varacc import VarAcc
from hpa import *
import numpy as np
import sys
from solver import *
from dataset import *
from partitioner import *
from query_gen import *
from tqdm import tqdm
from utils import *
from analyzer import *
from collections import defaultdict
from exputils import *

def build_sp_pair(partitioner_sr, eps = 0.1):
    # solver and partitioners we want to test in this experiment.
    sp_pair = [
        [{'ADP': OptVAST}, DP_AMAX(sr = partitioner_sr, eps=eps)],
        [{'EQ': OptVAST}, EqualDepthPartitioner(sr = partitioner_sr, eps=eps)],
    ]
    return sp_pair

class ExpConfig():
    def __init__(self):
        self.config = {
            'intel':{
                'p_attrs': ['itime'],
                't_attrs': ['light'],
                'ds_cls': IntelWireless,
                'partitioner_sr': 0.0004,
                'challenging_range': {'min_p': {'itime':120600}, 'max_p': {'itime':120700}},
                'ds': None,
            },
            'insta':{
                'p_attrs': ['product_id'],
                't_attrs': ['reordered'],
                'ds_cls': Instacart,
                'partitioner_sr': 0.0005,
                'challenging_range': {'min_p': {'product_id':7000}, 'max_p': {'product_id':7200}},
                'ds': None,
            },
            'taxi':{
                'p_attrs': ['pickup_datetime'],
                't_attrs': ['trip_distance'],
                'ds_cls': Taxi,
                'partitioner_sr': 0.0001,
                'challenging_range': {'min_p': {'pickup_datetime':981147307}, 'max_p': {'pickup_datetime':1546353568}},
                'ds': None,
            },
            'data1m':{
                'p_attrs': ['C'],
                't_attrs': ['A'],
                'ds_cls': Data1M,
                'partitioner_sr': 0.001,
                'challenging_range': {'min_p': {'C':875000}, 'max_p': {'C':1000000}},
                'ds': None,
            }
        }
    def get_config(self, name):
        if self.config[name]['ds'] is None:
            self.config[name]['ds'] = self.config[name]['ds_cls']()
        return self.config[name]


def run_one(config, n_queries, sample_rate, max_depth, max_leaf):
    dataset = config['ds']
    p_attrs = config['p_attrs']
    t_attrs = config['t_attrs']
    partitioner_sr = config['partitioner_sr']
    challenging_range = config['challenging_range']

    sp_pair = build_sp_pair(partitioner_sr)
    challenging = IntervalRangeSum(challenging_range['min_p'], challenging_range['max_p']).generate_queries(dataset, p_attrs,
                                                                                    t_attrs[0], n_queries)
    random = RangeSum().generate_queries(dataset, p_attrs, t_attrs[0], n_queries)

    compare_solver_partitioner_pairs(dataset, p_attrs, t_attrs,
                                    max_depth, max_leaf, sp_pair,
                                    {
                                        'challenging': challenging,
                                        'random': random
                                    },
                                    sample_rate)

if __name__ == '__main__':
    max_leaf = int(sys.argv[2])
    max_depth = int(math.log(max_leaf, 2)+1)
    assert(math.pow(2, max_depth-1) == max_leaf)
    dsname = sys.argv[1]
    n_queries = 2000
    sr = 0.005 #float(sys.argv[3]) fix the sample rate, no need to change

    print(dsname, "Sample Rate:", sr, "Max Leaf:", max_leaf, "Depth (including root):", max_depth)
    config = ExpConfig()
    run_one(config.get_config(dsname), n_queries, sr, max_depth, max_leaf)

