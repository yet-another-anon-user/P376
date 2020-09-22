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
        [
            {
                'PASS': OptVAST
            }, DP_AMAX(sr = partitioner_sr, eps = eps)
        ],
        [
            {
                'US': US,
                'ST': Stratified
            }, EqualDepthPartitioner(sr = partitioner_sr)
        ],
        [
            {
                'AQP++': AQPPP
            },  HillClimbing(sr = partitioner_sr)
        ]
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
                'ds': None,
            },
            'insta':{
                'p_attrs': ['product_id'],
                't_attrs': ['reordered'],
                'ds_cls': Instacart,
                'partitioner_sr': 0.0005,
                'ds': None,
            },
            'taxi':{
                'p_attrs': ['pickup_datetime'],
                't_attrs': ['trip_distance'],
                'ds_cls': Taxi,
                'partitioner_sr': 0.00005, #0.0001 is used for challenging ones.
                'ds': None,
            }
        }
    def get_config(self, name):
        if self.config[name]['ds'] is None:
            self.config[name]['ds'] = self.config[name]['ds_cls']()
        return self.config[name]


def run_one_macro(config, n_queries, sample_rate, max_depth, max_leaf):
    dataset = config['ds']
    p_attrs = config['p_attrs']
    t_attrs = config['t_attrs']
    partitioner_sr = config['partitioner_sr']

    sp_pair = build_sp_pair(partitioner_sr)
    random_queries = RangeSum().generate_queries(dataset, p_attrs, t_attrs[0], n_queries)

    compare_solver_partitioner_pairs(dataset, p_attrs, t_attrs,
                                    max_depth, max_leaf, sp_pair,
                                    {'random': random_queries}, sample_rate)

if __name__ == '__main__':
    sr = float(sys.argv[2])
    max_leaf = int(sys.argv[1])
    max_depth = int(math.log(max_leaf, 2)+1)
    assert(math.pow(2, max_depth-1) == max_leaf)
    dsname = sys.argv[3]
    n_queries = 2000

    print(dsname, "Sample Rate:", sr, "Max Leaf:", max_leaf, "Depth (including root):", max_depth)
    config = ExpConfig()
    run_one_macro(config.get_config(dsname), n_queries, sr, max_depth, max_leaf)

