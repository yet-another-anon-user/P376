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
                'KDPASS': OptVAST
            }, MedianPartitioner()
            , NKDTree
        ],
        [
            {
                'AQP++': AQPPP
            }, MedianPartitioner()
            , NaiveKDTree
        ]
    ]
    return sp_pair

class ExpConfig():
    def __init__(self):
        self.config = {
            'taxi1':{
                'p_attrs': ['pickup_time'],
                't_attrs': ['trip_distance'],
                'ds_cls': Taxi1D,
                'partitioner_sr': 0.0001,
                'ds': None,
            },
            'taxi2':{
                'p_attrs': ['pickup_date', 'pickup_time'],
                't_attrs': ['trip_distance'],
                'ds_cls': Taxi2D,
                'partitioner_sr': 0.0001,
                'ds': None,
            },
            'taxi3':{
                'p_attrs': ['pickup_date', 'pickup_time','PULocationID'],
                't_attrs': ['trip_distance'],
                'ds_cls': Taxi3D,
                'partitioner_sr': 0.0001,
                'ds': None,
            },
            'taxi4':{
                'p_attrs': ['pickup_date', 'pickup_time','dropoff_date', 'PULocationID'],
                't_attrs': ['trip_distance'],
                'ds_cls': Taxi4D,
                'partitioner_sr': 0.0001,
                'ds': None,
            },
            'taxi5':{
                'p_attrs': ['pickup_date', 'pickup_time','dropoff_date', 'dropoff_time','PULocationID'],
                't_attrs': ['trip_distance'],
                'ds_cls': Taxi5D,
                'partitioner_sr': 0.0001,
                'ds': None,
            },
            'taxi6':{
                'p_attrs': ['pickup_date', 'pickup_time','dropoff_date', 'dropoff_time','PULocationID','DOLocationID'],
                't_attrs': ['trip_distance'],
                'ds_cls': Taxi6D,
                'partitioner_sr': 0.0001,
                'ds': None,
            },
        }
    def get_config(self, name):
        if self.config[name]['ds'] is None:
            self.config[name]['ds'] = self.config[name]['ds_cls']()
        return self.config[name]


def run_one(config, n_queries, sample_rate, max_leaf):
    dataset = config['ds']
    p_attrs = config['p_attrs']
    t_attrs = config['t_attrs']
    partitioner_sr = config['partitioner_sr']
    max_depth = math.ceil(math.log(max_leaf, pow(2, len(p_attrs)))+1)
    print("MaxDepth for %sD %s = %s" % (len(p_attrs), max_leaf, max_depth))
    sp_pair = build_sp_pair(partitioner_sr)
    random = RangeSum().generate_queries(dataset, p_attrs, t_attrs[0], n_queries)

    compare_solver_partitioner_pairs(dataset, p_attrs, t_attrs,
                                    max_depth, max_leaf, sp_pair,
                                    {
                                        'random': random
                                    },
                                    sample_rate
                                    )

if __name__ == '__main__':
    max_leaf = int(sys.argv[2])

    dsname = sys.argv[1]

    n_queries = 1000
    sr = 0.005 #float(sys.argv[3]) fix the sample rate, no need to change

    print(dsname, "Sample Rate:", sr, "Max Leaf:", max_leaf)
    config = ExpConfig()
    run_one(config.get_config(dsname), n_queries, sr, max_leaf)

