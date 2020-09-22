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
            }, MedianPartitioner() #tested this one for multiple dimension let's just use it.
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
            'taxi2':{
                'p_attrs': ['pickup_date', 'pickup_time'],
                't_attrs': ['trip_distance'],
                'ds_cls': Taxi2D,
                'partitioner_sr': 0.0001,
                'ds': None,
            }
        }
    def get_config(self, name):
        if self.config[name]['ds'] is None:
            self.config[name]['ds'] = self.config[name]['ds_cls']()
        return self.config[name]


def run_one(config, n_queries, sample_rate, max_leaf, workload_name):
    dataset = config['ds']
    p_attrs = config['p_attrs']
    t_attrs = config['t_attrs']
    partitioner_sr = config['partitioner_sr']
    max_depth = math.ceil(math.log(max_leaf, pow(2, len(p_attrs)))+1)
    print("MaxDepth for %sD %s = %s" % (len(p_attrs), max_leaf, max_depth))
    sp_pair = build_sp_pair(partitioner_sr)
    print("Loading workloads")
    workload = {
            '1D': restore_obj('Taxi1D-RangeSumQuery-pickup_time-trip_distance-1000.jpk')[:n_queries],
            '2D': restore_obj('Taxi2D-RangeSumQuery-pickup_date_pickup_time-trip_distance-1000.jpk')[:n_queries],
            '3D': restore_obj('Taxi3D-RangeSumQuery-PULocationID_pickup_date_pickup_time-trip_distance-1000.jpk')[:n_queries],
            '4D': restore_obj('Taxi4D-RangeSumQuery-PULocationID_dropoff_date_pickup_date_pickup_time-trip_distance-1000.jpk')[:n_queries],
            '5D': restore_obj('Taxi5D-RangeSumQuery-PULocationID_dropoff_date_dropoff_time_pickup_date_pickup_time-trip_distance-1000.jpk')[:n_queries]
    }
    compare_solver_partitioner_pairs(dataset, p_attrs, t_attrs,
                                    max_depth, max_leaf, sp_pair,
                                    {workload_name: workload[workload_name]},
                                    sample_rate
                                    )

if __name__ == '__main__':
    max_leaf = 1024

    dsname = 'taxi2' #we use taxi2 partition to solve queries of other dimensions

    n_queries = 1000
    sr = 0.005 #float(sys.argv[3]) fix the sample rate, no need to change

    workload_name = sys.argv[1]

    print(dsname, "Sample Rate:", sr, "Max Leaf:", max_leaf)
    config = ExpConfig()
    run_one(config.get_config(dsname), n_queries, sr, max_leaf, workload_name)

