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

def compare_solver_partitioner_pairs(dataset, p_attrs, t_attrs,
            max_depth, max_leaf, solver_partitioner,
            queriesets, sample_rate, var_cls = BUBT,
            ):

    for qname, queries in queriesets.items():
        print("Working on %s %s queries" % (len(queries), qname))
        for sp in solver_partitioner:
            solvers = sp[0]
            partitioner = sp[1]
            if len(sp) >= 3:
                var_cls = sp[2]
            run_exp(dataset, p_attrs, t_attrs,
                        max_depth, max_leaf, qname,
                        queries = queries, va_cls = var_cls,
                        solvers = solvers,
                        sample_rate = sample_rate, partitioner = partitioner)

def run_exp(data, p_attrs, t_attrs, tree_depth, max_leaf,
            qname, queries=[], va_cls = HPA,
            solvers = {},
            sample_rate = 0.001,
            partitioner = None):

    p = partitioner
    print("======New Experiment======\nMax Depth: %s, Max Leaf: %s, Sample Rate:%s, Pred Attr: %s, Target Attr: %s"
            % (tree_depth, max_leaf, sample_rate, ','.join(p_attrs), ','.join(t_attrs)))

    if p is None:
        p = EqualWidthPartitioner()

    va = VarAcc(data, p, tree_depth, p_attrs, t_attrs, va_cls=va_cls, max_leaf = max_leaf)

    va.initialize()

    print("\nSolving %s %s queries..." % (len(queries), qname))
    results = defaultdict(list)

    n_queries = len(queries)
    # sample_rate = 0.001
    for name, solver_cls in solvers.items():
        solver = solver_cls()
        print("Solving by %s" % name)

        for q in tqdm(queries):
            solver.reset_seed(int(q.id[:6], 16))
            r = solver.solve(q, va.hpa, 10000000, va.sample_map, sample_rate)
            if len(r) == 10:
                c, s, m, c_ci, s_ci, m_ci, min_est, max_est, sr_c, sr_r = r
                results[name].append([float(c), float(s), float(m), float(c_ci), float(s_ci), float(m_ci), min_est, max_est, float(sr_c), float(sr_r)])
            else:
                c, s, m, c_ci, s_ci, m_ci, min_est, max_est = r
                results[name].append([float(c), float(s), float(m), float(c_ci), float(s_ci), float(m_ci), min_est, max_est])

        save_to = "%s-%s-%s-%s-%s-%s-%s-%s.json" % (data.__class__.__name__,
                                            p.__class__.__name__, name,
                                            tree_depth, max_leaf, sample_rate, n_queries, qname)

        dump_json(results, save_to)


        print("Analyzing RelativeError")
        analyzer = RelativeError(queries, results)
        analyzer.analyze(True)

        print("Analyzing FailureRate")
        analyzer = FailureRate(queries, results)
        analyzer.analyze(True)

        print("Analyzing CI Ratio")
        analyzer = CIRatio(queries, results)
        analyzer.analyze(True)

        print("Analyzing Skip Rate")
        analyzer = SkipRate(queries, results)
        analyzer.analyze(True)