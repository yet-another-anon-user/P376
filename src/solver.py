import numpy as np
from pre_agg import PreAgg, PANode
from hpa import HPA
from collections import defaultdict
import copy, math, itertools, time
import pandas as pd

class Solver(object):
    def __init__(self):
        self.seed = None
        self.frontier = None
        # self.ci_lambda = 1.96 # 95% CI
        self.ci_lambda = 2.576 # 99% CI

    def reset_seed(self, n):
        self.seed = n

    def solve(self, query, tree, budget, sample_map, sample_ratio):
        raise NotImplementedError

class PASolver(Solver):
    def traverse(self, query, tree, budget):
        '''
        A best-first-search algo
        We expand the nodes of the best values (partial overlapping)
        Return the frontier of traverse
        '''
        frontier = [tree.root]
        budget -= 1

        while True:
            to_expand = []
            best = [float('-inf'), None]
            candidates = []
            for node in frontier.copy():
                value = self.expand_value(query, node)
                if value > 0:
                    if (value > best[0]
                            and budget > node.cost() and node.cost() > 0):
                        best = [value, node]
                    candidates.append(node)
                elif value == -2:
                    frontier.remove(node)

            if best[1] is not None:
                frontier.remove(best[1])
                frontier.extend(best[1].children)
                budget -= best[1].cost()
            else:
                break
        return frontier

    def show_frontier(self, frontier):
        print("======= Frontier: %s nodes =======" % len(frontier))
        for node in frontier:
            print(node)
        print("=================================\n")

    def frontier_solve(self, query, frontier):
        sum_bound = cnt_bound = 0
        for node in frontier:
            if query.intersect(node):
                sum_bound += node.pa.summaries[query.pa.t_attrs[0]]['sum']
                cnt_bound += node.pa.cnt
        return cnt_bound, sum_bound

    def expand_value(self, query, node):
        '''
        '''
        if node.intersect(query) == False:
            return -2
        if query.cover(node):
            return -1 #no value to further expand a covered node
        if node.cover(query):
            return 1
        # does not cover, only intersect.
        return 2

    def find_delta_leaf_nodes(self, query, leaf_nodes):
        # intersecting only, but not covered by query. (nodes that introduce delta)
        nodes = []
        for node in leaf_nodes:
            if query.intersect(node) and query.cover(node) == False:
                nodes.append(node)
        return nodes

    def find_covering_query(self, query, frontiers):
        if type(query) != PANode:
            query = PANode(PreAgg.from_query(query))
        covering = set()
        new_query = copy.deepcopy(query)
        sum_bound = cnt_bound = 0
        for node in frontiers:
            if query.intersect(node):
                covering.add(node)

        if len(covering) == 0:
            return new_query

        for attr in new_query.pa.predicates.keys():
            left = float('inf')
            right = float('-inf')
            for node in covering:
                left = min(node.pa.predicates[attr][0], left)
                right = max(node.pa.predicates[attr][1], right)
            new_query.pa.predicates[attr] = [left, right]
        return new_query

    def find_covered_query(self, query, relevant_pas):
        if type(query) != PANode:
            query = PANode(PreAgg.from_query(query))
        sum_bound = cnt_bound = 0
        covered = set()
        new_query = copy.deepcopy(query)

        for node in relevant_pas:
            if query.cover(node):
                covered.add(node)
                sum_bound += node.pa.summaries[query.pa.t_attrs[0]]['sum']
                cnt_bound += node.pa.cnt

        return covered, (cnt_bound, sum_bound)

class BFS(PASolver):
    def solve(self, query, tree, budget, sample_map = None, sample_ratio = None):
        if type(query) != PANode:
            query = PANode(PreAgg.from_query(query))
        self.frontier = self.traverse(query, tree, budget)
        return self.frontier_solve(query, self.frontier)

class LeafPA(PASolver):
    def solve(self, query, tree, budget, sample_map = None, sample_ratio = None):
        if type(query) != PANode:
            query = PANode(PreAgg.from_query(query))
        self.frontier = tree.leaves
        return self.frontier_solve(query, self.frontier)

class SampleSolver(Solver):
    def gen_samples_from_map(self, map, budget):
        raise NotImplementedError

    def gen_samples_from_map_partitions(self, partitions, sample_ratio):
        samples = None
        for partition in partitions:
            sample = partition.sample(frac = sample_ratio, replace = False, random_state = self.seed)
            if samples is not None:
                samples = pd.concat([samples, sample], ignore_index=True)
            else:
                samples = sample
        return samples

    def solve(self, query, tree, budget, sample_map, sample_ratio, strata_nodes=None):
        if type(query) != PANode:
            query = PANode(PreAgg.from_query(query))
        t1 = time.time()
        samples = self.gen_samples_from_map(sample_map, sample_ratio)
        t2 = time.time()
        return self.solve_by_samples(query, samples, sample_ratio)

    def var_to_ci(self, var):
        return self.ci_lambda*math.sqrt(var)

    def solve_by_samples(self, query, samples, ratio):
        data = samples
        orig_size = data.shape[0]

        t_attr = query.pa.t_attrs[0]

        for attr in query.pa.predicates.keys():
            pred = query.pa.predicates[attr]
            data = data[(data[attr] >= pred[0]) & (data[attr] <= pred[1])]

        csum, ssum = data.shape[0]/ratio, sum(data[query.pa.t_attrs[0]])/(ratio)
        avg = data[query.pa.t_attrs[0]].mean() if data.shape[0] > 0 else 0

        min_est = min(data[t_attr]) if data.shape[0] > 0 else float('inf')
        max_est = max(data[t_attr]) if data.shape[0] > 0 else float('-inf')

        paddings = [0]*(orig_size - data.shape[0])

        scale = orig_size/data.shape[0] if data.shape[0] > 0 else 0
        target = [x for x in data[query.pa.t_attrs[0]]]
        target.extend(paddings)
        mean_var = scale*scale*np.var(target)/orig_size if data.shape[0] > 0 else 0

        scale = orig_size/ratio
        target = [x for x in data[query.pa.t_attrs[0]]]
        target.extend(paddings)
        sum_var = scale*scale*np.var(target)/orig_size if data.shape[0] > 0 else 0

        target = [1]*data.shape[0]
        target.extend(paddings)
        cnt_var = scale*scale*np.var(target)/orig_size if data.shape[0] > 0 else 0
        return csum, ssum, avg, self.var_to_ci(cnt_var), self.var_to_ci(sum_var), self.var_to_ci(mean_var), min_est, max_est

class US(SampleSolver):
    # uniform sampling
    def gen_samples_from_map(self, sample_map, sample_ratio):
        samples = sample_map.data.df.sample(frac=sample_ratio, replace=False, random_state = self.seed)
        return samples

    def batch_solve(self, queries, tree, budget, sample_map, sample_ratio, strata_nodes):
        sample_size = int(sample_map.data.df.shape[0]*sample_ratio)
        samples = self.gen_samples_from_map(sample_map, sample_ratio)

        return self.solve_queries_by_uniform_samples(samples, sample_size, queries, sample_ratio)

    def solve_queries_by_uniform_samples(self, samples, total_sample_size, queries, ratio):
        orig_size = samples.shape[0]
        filtered = None

        t_attr = queries[0].pa.t_attrs[0]

        for query in queries:
            data = samples
            for attr in query.pa.predicates.keys():
                pred = query.pa.predicates[attr]
                data = data[(data[attr] >= pred[0]) & (data[attr] <= pred[1])]
            if filtered is not None:
                filtered = pd.concat([filtered, data], ignore_index=True)
            else:
                filtered = data

        data = filtered


        csum, ssum = data.shape[0]/ratio, sum(data[query.pa.t_attrs[0]])/(ratio)
        avg = data[query.pa.t_attrs[0]].mean() if data.shape[0] > 0 else 0

        min_est = min(data[t_attr])  if data.shape[0] > 0 else float('inf')
        max_est = max(data[t_attr])  if data.shape[0] > 0 else float('-inf')

        paddings = [0]*(orig_size - data.shape[0])

        scale = orig_size/data.shape[0] if data.shape[0] > 0 else 0
        target = [x for x in data[query.pa.t_attrs[0]]]
        target.extend(paddings)
        mean_var = scale*scale*np.var(target)/orig_size if data.shape[0] > 0 else 0

        scale = orig_size/ratio
        target = [x for x in data[query.pa.t_attrs[0]]]
        target.extend(paddings)
        sum_var = scale*scale*np.var(target)/orig_size if data.shape[0] > 0 else 0

        target = [1]*data.shape[0]
        target.extend(paddings)
        cnt_var = scale*scale*np.var(target)/orig_size if data.shape[0] > 0 else 0

        return csum, ssum, avg, self.var_to_ci(cnt_var), self.var_to_ci(sum_var), self.var_to_ci(mean_var), min_est, max_est

class Stratified(SampleSolver):

    def batch_solve(self, queries, tree, budget, sample_map, sample_ratio, strata_nodes):
        sample_size = int(sample_map.data.df.shape[0]*sample_ratio)

        strata = []

        attr = queries[0].pa.t_attrs[0]
        strata_nodes = tree.leaf_nodes #all of them, no skipping.
        strata_size = 0
        for node in strata_nodes:
            strata.append(sample_map.map[node.id])
            strata_size += sample_map.map[node.id].shape[0]

        sample_size = min(sample_size, strata_size)

        return self.solve_queries_by_stratified_samples(strata, sample_size, queries)

    def opt_stratum_allocation(self, ss, attr, total_sample_size):
        allocation = []
        variances = [s[attr].var() if s.shape[0] > 0 else 0 for s in ss]
        stratum_sizes = [s.shape[0] for s in ss]
        weighted_var_sum = sum([i[0]*i[1] for i in zip(variances, stratum_sizes)])
        for idx, s in enumerate(ss):
            allocation.append( int(total_sample_size*s.shape[0]*variances[idx]/weighted_var_sum) )

        return allocation

    def proportion_allocation(self, ss, total_sample_size):
        allocation = []
        stratum_sizes = [s.shape[0] for s in ss]
        strata_size = sum(stratum_sizes)
        for idx, s in enumerate(ss):
            allocation.append( int(total_sample_size*stratum_sizes[idx]/strata_size) )
        return allocation

    def solve_by_stratified_samples(self, strata, total_sample_size, query):
        '''
        given a strata and sample size, calculate the allocation then solve cnt, sum, avg queries.
        return cnt, sum, avg results using the layer as population. (scale later for entire dataset)
        '''
        allocation = self.proportion_allocation(strata, total_sample_size)

        stratum_size = [s.shape[0] for s in strata]
        target_attr = query.pa.t_attrs[0]
        stratum_mean = []
        stratum_cnt = []
        stratum_mean_var = []
        stratum_cnt_var = []

        min_est = float('inf')
        max_est = float('-inf')
        for idx, stratum in enumerate(strata):

            samples = stratum.sample(n = allocation[idx], replace = False, random_state = self.seed)

            mean, cnt, mi, ma = self.stratum_mean(query, samples)
            min_est = min(min_est, mi)
            max_est = max(max_est, ma)

            m_var, c_var = self.stratum_variance(query, samples, stratum.shape[0])
            stratum_mean.append(mean)
            stratum_cnt.append(cnt)
            stratum_mean_var.append(m_var)
            stratum_cnt_var.append(c_var)

        N_i_pr = [i[0]*i[1]/i[2] if i[2] > 0 else 0 for i in zip(stratum_size, stratum_cnt, allocation)]

        total = sum(i[0] * i[1] for i in zip(stratum_mean, N_i_pr))
        avg = total/sum(N_i_pr) if sum(N_i_pr) > 0 else 0
        cnt = sum(N_i_pr)

        total_var = sum(i[0] * i[1] * i[1] for i in zip(stratum_mean_var, N_i_pr))
        avg_var = total_var/(sum(N_i_pr)**2) if sum(N_i_pr) > 0 else 0
        cnt_var = sum(i[0] * i[1] * i[1] for i in zip(stratum_cnt_var, N_i_pr)) #CNT is a special SUM

        return cnt, total, avg, self.var_to_ci(cnt_var), self.var_to_ci(total_var), self.var_to_ci(avg_var), min_est, max_est

    def stratum_mean(self, query, samples):

        data = samples

        for attr in query.pa.predicates.keys():
            pred = query.pa.predicates[attr]
            data = data[(data[attr] >= pred[0]) & (data[attr] <= pred[1])]

        mean = data[query.pa.t_attrs[0]].mean() if data.shape[0] > 0 else 0
        cnt = data.shape[0]
        min_est = min(data[query.pa.t_attrs[0]]) if data.shape[0] > 0 else float('inf')
        max_est = max(data[query.pa.t_attrs[0]]) if data.shape[0] > 0 else float('-inf')
        return mean, cnt, min_est, max_est

    def stratum_variance(self, query, samples, stratum_size):
        data = samples
        orig_size = data.shape[0]

        for attr in query.pa.predicates.keys():
            pred = query.pa.predicates[attr]
            data = data[(data[attr] >= pred[0]) & (data[attr] <= pred[1])]
        paddings = [0]*(orig_size - data.shape[0])

        target = [x for x in data[query.pa.t_attrs[0]]]
        target.extend(paddings)
        scale = orig_size/data.shape[0] if data.shape[0] > 0 else 0 #K/K_pred
        mean_var = scale*scale*np.var(target)/orig_size if data.shape[0] > 0 else 0

        target = [1]*data.shape[0]
        target.extend(paddings)
        cnt_var = scale*scale*np.var(target)/orig_size if data.shape[0] > 0 else 0
        return mean_var, cnt_var

    def solve_by_leaf_layer(self, query, tree, budget, sample_map, sample_size):
        if type(query) != PANode:
            query = PANode(PreAgg.from_query(query))

        leaf_nodes = tree.leaves
        strata = []

        attr = query.pa.t_attrs[0]

        for node in leaf_nodes:
            strata.append(sample_map.map[node.id])

        return self.solve_by_stratified_samples(strata, sample_size, query)

    def solve(self, query, tree, budget, sample_map, sample_ratio, strata_nodes=None):
        sample_size = int(sample_map.data.df.shape[0]*sample_ratio)
        return self.solve_by_leaf_layer(query, tree, budget, sample_map, sample_size)

    def solve_queries_by_stratified_samples(self, strata, total_sample_size, queries):
        allocation = self.proportion_allocation(strata, total_sample_size)

        stratum_size = [s.shape[0] for s in strata]
        target_attr = queries[0].pa.t_attrs[0]
        stratum_mean = []
        stratum_cnt = []
        stratum_mean_var = []
        stratum_cnt_var = []

        min_est = float('inf')
        max_est = float('-inf')

        for idx, stratum in enumerate(strata):
            samples = stratum.sample(n = allocation[idx], replace = False, random_state = self.seed)

            mean, cnt, mi, ma = self.batch_stratum_mean(queries, samples)
            m_var, c_var = self.batch_stratum_variance(queries, samples)
            min_est = min(min_est, mi)
            max_est = max(max_est, ma)
            stratum_mean.append(mean)
            stratum_cnt.append(cnt)
            stratum_mean_var.append(m_var)
            stratum_cnt_var.append(c_var)

        N_i_pr = [i[0]*i[1]/i[2] if i[2] > 0 else 0 for i in zip(stratum_size, stratum_cnt, allocation)]
        total = sum(i[0] * i[1] for i in zip(stratum_mean, N_i_pr))
        avg = total/sum(N_i_pr) if sum(N_i_pr) > 0 else 0
        cnt = sum(N_i_pr)

        total_var = sum(i[0] * i[1] * i[1] for i in zip(stratum_mean_var, N_i_pr))
        avg_var = total_var/(sum(N_i_pr)**2) if sum(N_i_pr) > 0 else 0
        cnt_var = sum(i[0] * i[1] * i[1] for i in zip(stratum_cnt_var, N_i_pr))

        return cnt, total, avg, self.var_to_ci(cnt_var), self.var_to_ci(total_var), self.var_to_ci(avg_var), min_est, max_est

    def batch_stratum_mean(self, queries, samples):

        filtered = None
        for query in queries:
            data = samples
            for attr in query.pa.predicates.keys():
                pred = query.pa.predicates[attr]
                data = data[(data[attr] >= pred[0]) & (data[attr] <= pred[1])]
            if filtered is not None:
                filtered = pd.concat([filtered, data], ignore_index=True)
            else:
                filtered = data

        data = filtered
        mean = data[queries[0].pa.t_attrs[0]].mean() if data.shape[0] > 0 else 0
        mi = min(data[queries[0].pa.t_attrs[0]]) if data.shape[0] > 0 else float('inf')
        ma = min(data[queries[0].pa.t_attrs[0]]) if data.shape[0] > 0 else float('-inf')

        cnt = data.shape[0]
        return mean, cnt, mi, ma

    def batch_stratum_variance(self, queries, samples):
        filtered = None
        orig_size = samples.shape[0]

        for query in queries:
            data = samples
            for attr in query.pa.predicates.keys():
                pred = query.pa.predicates[attr]
                data = data[(data[attr] >= pred[0]) & (data[attr] <= pred[1])]
            if filtered is not None:
                filtered = pd.concat([filtered, data], ignore_index=True)
            else:
                filtered = data

        data = filtered

        paddings = [0]*(orig_size - data.shape[0])

        target = list(data[queries[0].pa.t_attrs[0]])
        target.extend(paddings)

        scale = orig_size/data.shape[0] if data.shape[0] > 0 else 0 #K/K_pred

        mean_var = scale*scale*np.var(target)/orig_size if data.shape[0] > 0 else 0 #for avg and sum queries

        target = [1]*data.shape[0]
        target.extend(paddings)

        cnt_var = scale*scale*np.var(target)/orig_size if data.shape[0] > 0 else 0
        return mean_var, cnt_var

class OptStratified(Stratified):
    def solve(self, query, tree, budget, sample_map, sample_ratio, strata_nodes=None):

        sample_size = int(sample_map.data.df.shape[0]*sample_ratio)

        strata = []

        attr = query.pa.t_attrs[0]

        strata_size = 0
        for node in strata_nodes:
            strata.append(sample_map.map[node.id])
            strata_size += sample_map.map[node.id].shape[0]

        sample_size = min(sample_size, strata_size)

        return self.solve_by_stratified_samples(strata, sample_size, query)

class BatchOptStratified(OptStratified):
    def batch_solve(self, queries, tree, budget, sample_map, sample_ratio, strata_nodes):
        sample_size = int(sample_map.data.df.shape[0]*sample_ratio)

        strata = []

        attr = queries[0].pa.t_attrs[0]

        strata_size = 0
        for node in strata_nodes:
            strata.append(sample_map.map[node.id])
            strata_size += sample_map.map[node.id].shape[0]

        sample_size = min(sample_size, strata_size)

        return self.solve_queries_by_stratified_samples(strata, sample_size, queries)


class CrossLayerVarianceWeighted(Stratified):
    def variance_weighted_avg(self, values, variances):
        weights = [1/v if v > 0 else 0 for v in variances]
        weighted = [i[0]*i[1] for i in zip(values, weights)]
        return sum(weighted)/sum(weights) if sum(weights) > 0 else 0

    def solve(self, query, tree, budget, sample_map, sample_ratio):
        cnt_values = []
        cnt_variances = []
        total_values = []
        total_variances = []
        avg_values = []
        avg_variances = []
        for nth_layer in range(len(tree.layers)):
            layer_partition = sample_map.layer_partitions[nth_layer]
            layer_partition_ratio = layer_partition.shape[0] / sample_map.data.df.shape[0]
            sample_size = min(layer_partition.shape[0], int(sample_map.data.df.shape[0]*sample_ratio))
            sample_size = round(sample_size/len(tree.layers))
            cnt, total, avg, cnt_var, total_var, avg_var = self.solve_by_nth_layer(query, tree, budget, sample_map, sample_size, nth_layer)

            cnt_values.append(cnt)
            cnt_variances.append(cnt_var)
            total_values.append(total)
            total_variances.append(total_var)
            avg_values.append(avg)
            avg_variances.append(avg_var)

        cnt = self.variance_weighted_avg(cnt_values, cnt_variances)
        total = self.variance_weighted_avg(total_values, total_variances)
        avg = self.variance_weighted_avg(avg_values, avg_variances)

        return cnt/layer_partition_ratio, total/layer_partition_ratio, avg

class HybridCovered(object):
    def __init__(self, pa_solver = None, sample_solver = None):
        self.pa_solver = pa_solver
        self.sample_solver = sample_solver

    def reset_seed(self, n):
        self.sample_solver.reset_seed(n)

    def merge_query(self, hr1, hr2):
        extend_attr = None
        for idx, r in enumerate(hr1):
            if hr2[idx] is None or r is None:
                return None
            if hr2[idx] != r:
                if extend_attr != None:
                    return None
                extend_attr = idx

        r1 = hr1[extend_attr]
        r2 = hr2[extend_attr]

        if max(r1[0], r2[0]) == min(r1[1], r2[1]) + 1:
            new_hr = list(copy.deepcopy(hr1))
            new_hr[extend_attr] = [min(r1[0], r2[0]), max(r1[1], r2[1])]
            return new_hr
        return None

    def get_delta(self, q, q_prime):

        for attr, pred in q_prime.pa.predicates.items():
            if pred[0] > pred[1]:
                return None

        ranges = defaultdict(list)
        qp_range = []

        for attr, pred in q.pa.predicates.items():
            a = pred[0]
            d = pred[1]
            b = q_prime.pa.predicates[attr][0]
            c = q_prime.pa.predicates[attr][1]
            qp_range.append([b,c])
            if a <= b-1:
                ranges[attr].append([a, b - 1])
            ranges[attr].append([b, c])
            if c+1 <= d:
                ranges[attr].append([c+1, d])

        hrs = list(itertools.product(*list(ranges.values())))
        hrs.remove(tuple(qp_range))

        merged = True
        while merged == True:
            merged = False

            for idx, hr in enumerate(copy.deepcopy(hrs)):
                if idx+1 == len(hrs):
                    break
                if hr not in hrs:
                    continue

                for idx2, hr1 in enumerate(copy.deepcopy(hrs)[idx+1:]):
                    if hr1 not in hrs:
                        continue
                    new_hr = self.merge_query(hr, hr1)
                    if new_hr is None:
                        continue
                    else:
                        merged = True
                        hrs.remove(hr)
                        hrs.remove(hr1)
                        hrs.insert(0, new_hr)
                        break
                if merged:
                    break
            if merged is False:
                break
        return hrs

    def get_delta_query_by_intersection(self, q, delta_node):
        delta_query = copy.deepcopy(q)

        for attr, pred in delta_node.pa.predicates.items():
            left = max(q.pa.predicates[attr][0], delta_node.pa.predicates[attr][0])
            right = min(q.pa.predicates[attr][1], delta_node.pa.predicates[attr][1])
            delta_query.pa.predicates[attr] = [left, right]

        return delta_query

    def solve(self, query, tree, budget, sample_map, sample_ratio):
        '''
        combine PA and Sampling in a AQP++ style.
        we skip sampling from irrelevant strata that are either irrelevant or covered by PA.
        our advantages:
        1. hyrarchical partition tree provides opportunity of early circut break.
        2. hyrarchical partition tree provides better expansion on inaccurate PAs, thus provides finer-grain delta.
        3. stratified PA from sample map provides better accuracy than uniform sampling.
        '''
        if type(query) != PANode:
            query = PANode(PreAgg.from_query(query))

        _ = self.pa_solver.solve(query, tree, budget, sample_map, sample_ratio)
        covered_nodes, q_bfs = self.pa_solver.find_covered_query(query, self.pa_solver.frontier)

        leaf_nodes = tree.leaves

        delta_nodes = self.pa_solver.find_delta_leaf_nodes(query, leaf_nodes)

        delta_results = []

        min_est = float('inf')
        max_est = float('-inf')

        t_attr = query.pa.t_attrs[0]
        for node in list(covered_nodes) + delta_nodes:
            min_est = min(min_est, node.pa.summaries[t_attr]['min'])
            max_est = max(max_est, node.pa.summaries[t_attr]['max'])

        if len(covered_nodes) == 0:
            r = self.sample_solver.solve(query, tree, budget, sample_map, sample_ratio, delta_nodes)
            c, s, m, c_ci, s_ci, m_ci = r[:6]
            return c, s, m, c_ci, s_ci, m_ci, min_est, max_est, len(delta_nodes)/len(leaf_nodes), sum([node.pa.cnt for node in delta_nodes])/sample_map.data.df.shape[0]

        if len(delta_nodes) == 0:

            return q_bfs[0], q_bfs[1], q_bfs[1]/q_bfs[0], 0, 0, 0, min_est, max_est, 0, 0

        queries = []
        for node in delta_nodes:
            q_delta = self.get_delta_query_by_intersection(query, node)
            queries.append(q_delta)

        delta = self.sample_solver.batch_solve(queries, tree, budget, sample_map, sample_ratio, delta_nodes)

        c = q_bfs[0] + delta[0]
        s = q_bfs[1] + delta[1]
        a = s/c if c > 0 else 0
        return c, s, a, delta[3], delta[4], delta[5], min_est, max_est, len(delta_nodes)/len(leaf_nodes), sum([node.pa.cnt for node in delta_nodes])/sample_map.data.df.shape[0]

class OptVAST(HybridCovered):
    def __init__(self):
        super().__init__(BFS(), BatchOptStratified())

class AQPPP(HybridCovered):
    def __init__(self):
        super().__init__(LeafPA(), US())
