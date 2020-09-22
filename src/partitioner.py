import itertools, math
import copy
from tqdm import tqdm
import random
import pandas as pd
import numpy as np
import sklearn.utils
from utils import *
from solver import SampleSolver
from collections import defaultdict

class Partitioner(object):
    def __init__(self):
        self.partitions = []

    def partition_to(self, data, n_partitions, p_attrs, t_attrs):
        raise NotImplementedError

    def get_partitions(self):
        return self.partitions

    def partition_to_sample_index(self, partition, sample, p_attr):
        ret = []
        ppoints = set()

        for p in partition:
            left, right = p.pa.predicates[p_attr]
            ppoints.add(left)
            ppoints.add(right)

        ppoints = sorted(list(ppoints))

        idx = 0

        for i, row in sample.iterrows():
            for pidx, p in enumerate(ppoints):
                if row[p_attr] == p:
                    ret.append(idx)
                    pidx+=1

            idx += 1

        return ret

    def sample_index_to_partition(self, sample, p_attr, indices):
        ret = []
        left = None

        indices.append(-1)
        for idx in indices:
            if left is None:
                left = sample.iloc[0][p_attr]
            right = sample.iloc[idx][p_attr]
            ret.append([left, right-1])
            left = right
        return ret

    def sample_index_to_sample_partition(self, sample, indices):
        ret = []
        left = None

        for idx in indices+[sample.shape[0]]:
            if left is None:
                left = 0
            right = idx
            ret.append([left, right-1])
            left = right
            if right == sample.shape[0]: break

        return ret


    def sample_index_to_partition_points(self, sample, p_attr, indices):
        ret = []
        for idx in indices:
            right = sample.iloc[idx][p_attr]
            ret.append(right)
        return ret

    def partition_points_to_partition(self, p_min, p_max, ppoints):
        ret = []
        left = None
        ppoints.append(p_max)
        for idx in ppoints:
            if left is None:
                left = p_min
            right = idx
            ret.append([left, right-1])
            left = right
        return ret

class EqualWidthPartitioner(Partitioner):
    def partition_to(self, data, n_partitions, p_attrs, t_attrs):
        return self.grid_partition_with_func(self.range_partition, data,
                                        p_attrs, t_attrs[0], n_partitions)

    def split_chunks(self, l, n):
        for i in range(0, len(l), n):
            yield l[i:i + n]

    def grid_partition_with_func(self, partition_func, df, attrs, sum_attr, n_partitions):
        partitions = []

        for pair in zip(attrs, n_partitions):
            attr = pair[0]
            n_part = pair[1]
            partitions.append(partition_func(df, attr, n_part))

        cells = itertools.product(*partitions)
        output = []
        stratify_samples = None
        cells = list(cells)
        for cell in cells:#tqdm(cells,
                        # total = np.prod([len(partition) for partition in partitions])):
            data = df
            for idx, pred in enumerate(cell):
                by_attr = attrs[idx]
                l = pred[0]
                r = pred[1]
                data = data[(data[by_attr] >= l) & (data[by_attr] <= r)]
                if data.shape[0] == 0:
                    break
            if data.shape[0] > 0:
                output.append(data)
        return output

    def range_partition(self, df, by_attr, n_partitions):
        ret = []
        max_a = int(max(df[by_attr])) + 1
        min_a = int(min(df[by_attr])) - 1
        step = math.ceil((max_a - min_a + 1)/n_partitions)
        if step == 0:
            return [[min_a+1, max_a]]
        chunks = self.split_chunks(range(min_a-1, max_a + 1), step)
        last_chunk = None
        for chunk in chunks:
            if last_chunk:
                l = last_chunk[-1] + 1
            else:
                l = chunk[0]
            r = chunk[-1]
            ret.append([l, r])
            last_chunk = chunk
        return ret

class MedianPartitioner(EqualWidthPartitioner):
    def partition_to(self, data, n_partitions, p_attrs, t_attrs):
        # del t_attrs #unused by equal width partitioning
        return self.grid_partition_with_func(self.median_partition, data,
                                        p_attrs, t_attrs[0], n_partitions)

    def split_chunks(self, l, n):
        for i in range(0, len(l), n):
            yield l[i:i + n]


    def median_partition(self, df, by_attr, n_partitions):
        assert(n_partitions == 2)
        ret = []

        m = int(df[by_attr].median())

        max_a = int(max(df[by_attr])) + 1
        min_a = int(min(df[by_attr])) - 1
        ret.append([min_a, m])
        ret.append([m+1, max_a])
        return ret


class RandomPartitioner(EqualWidthPartitioner):
    # randomly choose one
    def partition_to(self, data, n_partitions, p_attrs, t_attrs):

        nth = random.randint(0, len(p_attrs) - 1)
        return super().partition_to(data, n_partitions, [p_attrs[nth]], t_attrs[0])

class DPPartitioner(Partitioner):
    # this is for 1D
    def __init__(self, sr=0.001, eps=0.1):
        self.partitions = []
        self.min_samples_per_bucket = 3
        self.seed = 12345
        self.max_range = [0,0]
        self.preprocess = None
        self.sample = None
        self.epsilon = eps
        self.sr = sr
        self.amax_cache = {}
        self.max_cache = {} #in practice we don't need it, but this is to speed up the exps.
        random.seed(self.seed)

    def show_leaf_errors(self, leaf_layer, data, p_attrs, t_attrs):
        #! this might not work if random seed changes and samples are different.
        sample = self.draw_samples(data, p_attrs, t_attrs, show=False)

        p_attr = p_attrs[0]
        preprocess = self.preprocessing(data, sample, p_attr, t_attrs[0])
        print("Restoring sample partition index")
        si = self.partition_to_sample_index(leaf_layer, sample, p_attr)
        if len(si) == 0: return

        print("Sample Index: %s %s" % (len(si), si))
        print("Leaf Errors")
        self.show_partition_max_var(sample, si, preprocess)

    def draw_samples(self, data, p_attrs, t_attrs, n = None, show=False):
        if n is None:
            n = int(data.shape[0]*self.sr)

        print("DPPartitioning by %s n=%s sr=%s epsilon=%s" %(self.__class__.__name__, n, self.sr, self.epsilon))

        p_attr = p_attrs[0]
        t_attr = t_attrs[0]

        data = data[[p_attrs[0], t_attrs[0]]]
        data = data.sort_values(by = [p_attr], ignore_index = True)

        sample = data.sample(n = n, random_state = self.seed)
        sample = sample.sort_values(by = [p_attr]) #now the sample index are also
        sample = pd.concat([data[0:1], sample, data[-1:]]) #make sure the first and the last tuple is always sampled
        sample = sample.drop_duplicates(subset=p_attrs[0], keep="last") # dropped ~1%, so it is safe to assume keys are unique.
        self.sample = sample

        print("Sample Description")
        print(sample.describe())

        if show is True:
            print("#\tidx\tp_attr\tt_attr")
            idx = 0
            for i, row in sample.iterrows():
                print("%s\t%s\t%s\t%s"% (idx, i, row[p_attr], row[t_attr]))
                idx+=1

        return sample


    def get_interval_variance(self, i, j, preprocess = None):
        # * return the max variance of subintervals in the bucket defined by i-th and j-th sample, i, j inclusive
        raise NotImplementedError

    def preprocessing(self, data, sample, p_attr, t_attr):
        raise not NotImplementedError

    def partition_to(self, data, k, p_attrs, t_attrs):
        # *1. sample
        sample = self.draw_samples(data, p_attrs, t_attrs)

        m = sample.shape[0]
        print("DP Partitioner w/ SR:", m/data.shape[0], m, k, self.min_samples_per_bucket)

        # *2. build oracle or preprocess.
        self.preprocess = self.preprocessing(data, sample, p_attrs[0], t_attrs[0])

        # *3. do partitioning with DP.
        partition = self.dp_partition(sample, k, p_attrs, t_attrs, self.preprocess)
        return partition

    def dp_partition(self, sample, k, p_attrs, t_attrs, preprocess):
        m = sample.shape[0]
        min_step = self.min_samples_per_bucket
        if k*min_step > m:
            print("Error: need %s samples for %s partitions with %s samples/partition" % (k*min_step, k, min_step))
            return None
        # * A[m][k] is defined to contain the solution of partitioning the first m samples to k partitions.
        # * our goal is to find the k-1 partition points (samples) that generate the minMaxError.
        # * A[m][k]['p'] = [2,4,6] means [:2],[2:4],[4:6],[6:] half open like

        A = defaultdict(lambda: defaultdict(lambda: {'v':-1, 'p':[]}))

        print("Initializing... %s x %s" % (m, k))

        for i in tqdm(range(min_step, m+1)):
            A[i][1]['p'] = []
            v = self.get_interval_variance(0, i-1, preprocess)
            A[i][1]['v'] = v

        max_var = -1
        for i in tqdm(range(1, k+1)):
            A[min_step * i][i]['p'] = list(range(min_step*i))[min_step:min_step*i:min_step]

            max_var = A[min_step * i][i]['v'] = max(max_var,
                                                    self.get_interval_variance(min_step*(i-1), min_step*i - 1, preprocess))

        print("Partitioning...")
        for n in range(2, k+1):
            print("Column %s/%s" % (n, k))
            for i in tqdm(range(min_step*n+1, m+1)):

                min_max_var = float('inf')
                best_idx = 0

                for j in range(min_step*(n-1), i):

                    if i-j-1 < min_step:

                        break

                    new_partition_cost = self.get_interval_variance(j+1, i-1, preprocess)
                    cur_max_var =  max(A[j][n-1]['v'], new_partition_cost)
                    if cur_max_var < min_max_var:
                        min_max_var = cur_max_var

                        best_idx = j

                A[i][n]['v'] = min_max_var
                A[i][n]['p'] = copy.deepcopy(A[best_idx][n-1]['p'])
                A[i][n]['p'].append(best_idx)

        partition = self.sample_index_to_partition(sample, p_attrs[0], A[m][k]['p'])

        return partition

class DP_OnTheFlyMax(DPPartitioner):
    def preprocessing(self, data, sample, p_attr, t_attr):
        print("Preprocessing")
        segment_size = defaultdict(int) # * the #tuples on the left of the ith sample
        prefix_sum = defaultdict(float) # * the prefix sum of the first i samples
        prefix_squaredsum = defaultdict(float) # * the prefix squared sum of the first i samples

        for idx, si in enumerate(sample.index):
            segment_size[idx] = si

        for idx, t in enumerate(sample[t_attr].tolist()):
            prefix_sum[idx] = t + prefix_sum[idx-1]
            prefix_squaredsum[idx] = t*t + prefix_squaredsum[idx-1]

        return segment_size, prefix_sum, prefix_squaredsum

    def MAX(self, left, right, segment_size, prefix_sum, prefix_squaredsum, verbose=False):
        # * return the max variance of subintervals in the bucket defined by left-th and right-th sample,
        # * including the left and right sample.
        cache = {}
        assert(left <= right)
        if (left, right) in self.max_cache:
            return self.max_cache[(left, right)]
        max_var = 0
        max_range = [float('-inf'),0,0]
        min_step = self.min_samples_per_bucket
        N_i = segment_size[right] - segment_size[left]
        n_i = right - left + 1
        count = 0
        for i in range(left, right - min_step + 2):
            for j in range(i + min_step - 1, right + 1):
                count += 1
                n_i_q_square = (j - i + 1)*(j - i + 1)
                sum_square = (prefix_squaredsum[j] - prefix_squaredsum[i - 1])
                squared_sum = (prefix_sum[j] - prefix_sum[i-1])*(prefix_sum[j] - prefix_sum[i-1])
                var = (1/n_i)*(1 - n_i/N_i)*((n_i/n_i_q_square)*sum_square - (1/n_i_q_square)*squared_sum)
                max_var = max(max_var, var)
                cache[(i,j)] = var

                if max_range[0] < max_var and verbose is True:
                    max_range = [max_var, i, j]
                    if self.max_range[0] < max_range[0]:
                        self.max_range = max_range + [left, right]

        if verbose is True:
            print("Bucket: %s %s, SubInterval: %s %s, EstVar: %s, HalfCI for AVG: %s" % (
                    left, right,
                    max_range[1], max_range[2], max_range[0],
                    SampleSolver().var_to_ci(max_range[0]) if max_range[0]>0 else 'NaN'))
        self.max_cache[(left, right)] = max_var
        return max_var

    def show_partition_max_var(self, sample, points, preprocess):
        last = None
        if points[0] != 0:
            points = [0] + points
        points = points + [len(sample) - 1]
        print("Max Var of %s partition points: %s" % (len(points), points))
        for i in points:
            if last is not None and last != i:
                self.get_interval_variance_with_OptMAX(last, i, preprocess, verbose=True)
            last = i

        if len(self.max_range) >= 5:
            print("Max of all partitions:")
            print("Bucket: %s %s, SubInterval: %s %s, EstVar: %s, HalfCI for AVG: %s" % (
                self.max_range[3], self.max_range[4],
                self.max_range[1], self.max_range[2],
                self.max_range[0], SampleSolver().var_to_ci(self.max_range[0])))
            print(self.max_range[3], sample[self.max_range[3]: self.max_range[3]+1])
            print(self.max_range[4], sample[self.max_range[4]: self.max_range[4]+1])
        else:
            print("Fail to show max var for partitions", points, self.max_range)


    def get_interval_variance_with_OptMAX(self, i, j, preprocess = None, verbose=False):
        if preprocess is None: preprocess = self.preprocess
        segment_size, prefix_sum, prefix_squaredsum = preprocess
        return self.MAX(i, j, segment_size, prefix_sum, prefix_squaredsum, verbose)


    def get_interval_variance(self, i, j, preprocess = None, verbose=False):

        if preprocess is None: preprocess = self.preprocess
        segment_size, prefix_sum, prefix_squaredsum = preprocess
        return self.MAX(i, j, segment_size, prefix_sum, prefix_squaredsum, verbose)


class EqualDepthPartitioner(DP_OnTheFlyMax):

    def partition_to(self, data, k, p_attrs, t_attrs):
        return self.eq_partition_by_sample(data, k, p_attrs, t_attrs)

    def eq_partition_by_sample(self, data, k, p_attrs, t_attrs, return_index_and_sample=False):
        # *1. sample
        n_per_bucket = int((self.sr*data.shape[0])/k)

        sample = self.draw_samples(data, p_attrs, t_attrs, n = n_per_bucket*k)

        m = sample.shape[0]
        n_per_bucket = int(m/k)

        points = []
        last = None
        for i in list(range(m))[:-1:n_per_bucket]:

            if last is not None and sample[p_attrs[0]].iloc[last] == sample[p_attrs[0]].iloc[i]:
                print("Skip")
                last = i
                continue
            points.append(i)
            last = i

        print("All points, " ,points, len(points), sample.shape, int((self.sr*data.shape[0])), n_per_bucket, k)
        points = points[1:k]

        preprocess = self.preprocessing(data, sample, p_attrs[0], t_attrs[0])
        self.show_partition_max_var(sample, points, preprocess)

        if return_index_and_sample:
            return points, sample


        partition = self.sample_index_to_partition(sample, p_attrs[0], points)

        return partition


class HillClimbing(EqualDepthPartitioner):
    '''
    from AQP++
    start from an EQ partition.
    in each bucket, find a mid point that introduce the largest error. bookkeeping it for later use.
    select the largest two mid points i_1, i_2 that cause the largest error.
    this error is for any query in the template because these mid points are like the endpoint of predicate that cause delta. Figure 5 of the paper.

    now we need to decide which existing partition points to move to i1 or i2. We select the one that introduce the least error, point p.

    we then move p to i1 or i2, whichever yilds a smaller upper bound error.
    if error_up does not change, we stop.

    HC_local only look at the 4 partition points around i1, i2.
    HC_global look at all ppoints, we will just implement this one.
    '''
    def __init__(self, sr=0.001, eps=0.1):
        super().__init__(sr, eps)
        self.error_cache = {}

    def error_i(self, sample, left, right, t_attr):
        # * find the max variance of subintervals split by a mid-point in the bucket defined by left-th and right-th sample,
        # * return the mid point and the error.
        # * This is a simplified MAX, because we only consider interval starting at left and ending at right
        cache = {}
        max_var = 0
        min_step = self.min_samples_per_bucket
        mid_point = -1
        for mid in range(left + min_step, right - min_step + 2):

            sl = sample[left:mid]
            sr = sample[mid:right]

            lvar = sl[t_attr].var()
            rvar = sr[t_attr].var()
            var = max(lvar, rvar)

            if max_var < var:
                max_var = max(max_var, var)
                mid_point = mid

        return mid_point, max_var

    def error_up(self, sample, ppoints, t_attr):
        sample_partition = self.sample_index_to_sample_partition(sample, ppoints)
        max_error = None
        max_2_error = None
        for p in sample_partition:
            str_p = str(p[0]) + ',' + str(p[1])
            if str_p not in self.error_cache:
                self.error_cache[str_p] = self.error_i(sample, p[0], p[1], t_attr)
            if max_error is None or max_error[1] < self.error_cache[str_p][1]:
                max_error = self.error_cache[str_p]
            elif ((max_error is not None and max_2_error is None)
                        or max_2_error[1] < self.error_cache[str_p][1]):
                max_2_error = self.error_cache[str_p]

        print("Max", max_error, max_2_error)
        return max_error, max_2_error, max_error[1]+max_2_error[1]

    def partition_to(self, data, k, p_attrs, t_attrs):
        p_attr = p_attrs[0]
        t_attr = t_attrs[0]
        ppoints, sample = super().eq_partition_by_sample(data, k, p_attrs, t_attrs, True)
        print(ppoints, sample.shape)
        sample_partition = self.sample_index_to_sample_partition(sample, ppoints)
        print(sample_partition)
        max_errors = None
        it = 1
        while True:

            if max_errors is None:
                max_errors = self.error_up(sample, ppoints, t_attr)
            max_error = max_errors[0][1] + max_errors[1][1]

            dest_candidate = []

            if max_errors[0][0] > 0:
                dest_candidate.append(max_errors[0][0])

            if max_errors[1][0] > 0:
                dest_candidate.append(max_errors[1][0])


            max_source_error = None

            print("Current ErrorUp", max_errors[2])
            for p in range(0, len(ppoints)):
                left = 0
                right = sample.shape[0]-1
                if p > 0:
                    left = ppoints[p-1]
                if p < len(ppoints) - 1:
                    right = ppoints[p+1]
                str_p = str(left) + ',' + str(right)
                if str_p not in self.error_cache:
                    self.error_cache[str_p] = self.error_i(sample, left, right, t_attr)

                if (max_source_error is None
                    or max_source_error[1] < self.error_cache[str_p][1]):
                    max_source_error = ppoints[p], self.error_cache[str_p][1]

            print("it#%s moving %s -> %s" % (it, max_source_error[0], max_errors))


            new_p2 = copy.deepcopy(ppoints)
            new_p1 = copy.deepcopy(ppoints)
            new_p1.remove(max_source_error[0])
            new_p2.remove(max_source_error[0])

            # todo: here we definitely can use some refactorying for duplicating code
            assert(dest_candidate[0] > 0)
            print(len(ppoints), len(new_p1), dest_candidate)
            new_p1.append(dest_candidate[0])
            new_p1 = sorted(new_p1)


            error1 = self.error_up(sample, new_p1, t_attr)

            error2 = max_errors

            if(len(dest_candidate) > 1):
                new_p2.append(dest_candidate[1])
                new_p2 = sorted(new_p2)


                error2 = self.error_up(sample, new_p2, t_attr)


            if error1[2] >= max_error and error2[2] >= max_error:
                print("ErrorUp Not improving, stop.")
                break

            if error1[2] < error2[2]:
                ppoints = new_p1
                max_errors = error1
            else:
                ppoints = new_p2
                max_errors = error2

            it += 1

        partition = self.sample_index_to_partition(sample, p_attrs[0], ppoints)
        return partition

class DP_NLgN(DP_OnTheFlyMax):
    '''
    the improvement is to utilize a fact that under some assumption. larger ranges will have larger error.
    this can help us increase the complexity of DP (w/o considering MAX function) from n2 to nlgn.
    min(max(A, M)) is a unimodal function of h' because: A monotonically increase w/ h while
    M monotonically decrease w/ h. therefore min(max(A, M)) have only one minimum value which makes it unimodal and can be solved by tenary search.

    '''
    def tenary_search(self, a, b, i, j, A):
        h = None
        max_var = float('inf')
        if b-a <= 3:
            for k in range(a, b+1):
                var = max(A[k][j-1]['v'], self.get_interval_variance(k+1, i-1)) #get
                if max_var > var:
                    h = k
                    max_var = var
            return h, max_var

        left3 = int(a + (b-a)/3)
        right3 = int(b - (b-a)/3)

        lvar =  max(A[left3][j-1]['v'], self.get_interval_variance(left3+1, i-1))
        rvar =  max(A[right3][j-1]['v'], self.get_interval_variance(right3+1, i-1))

        if lvar <= rvar:
            return self.tenary_search(a, right3, i, j, A)
        else:
            return self.tenary_search(left3, b, i, j, A)

    def dp_partition(self, sample, k, p_attrs, t_attrs, preprocess):
        m = sample.shape[0]
        min_step = self.min_samples_per_bucket
        if k*min_step > m:
            print("Error: need %s samples for %s partitions with %s samples/partition" % (k*min_step, k, min_step))
            return None

        A = defaultdict(lambda: defaultdict(lambda: {'v':-1, 'p':[]}))

        print("Initializing... %s x %s" % (m, k))
        for i in tqdm(range(min_step, m+1)):
            A[i][1]['p'] = []
            A[i][1]['v'] = self.get_interval_variance(0, i-1, preprocess)

        max_var = -1
        for i in tqdm(range(1, k+1)):
            A[min_step * i][i]['p'] = list(range(min_step*i))[min_step:min_step*i:min_step]
            # get_interval_variance is inclusive on both sides.
            max_var = A[min_step * i][i]['v'] = max(max_var,
                                                    self.get_interval_variance(min_step*(i-1), min_step*i - 1, preprocess))

        print("Partitioning...")

        for n in range(2, k+1):
            for i in tqdm(range(min_step*n+1, m+1)):
                min_max_var = float('inf')
                best_idx = 0
                best_idx, min_max_var = self.tenary_search(min_step*(n-1), i, i, n, A)
                A[i][n]['v'] = min_max_var
                A[i][n]['p'] = copy.deepcopy(A[best_idx][n-1]['p'])
                A[i][n]['p'].append(best_idx)

        partition = self.sample_index_to_partition(sample, p_attrs[0], A[m][k]['p'])
        return partition

class DP_AMAX(DP_OnTheFlyMax):
    def AMAX(self, left, right, segment_size, prefix_sum, prefix_squaredsum):
        # * return the max variance of subintervals in the bucket defined by left-th and right-th sample,
        # * including the left and right sample.
        cache = {}
        if (left, right) in self.amax_cache:
            return self.amax_cache[(left, right)]
        max_var = 0
        min_step = self.min_samples_per_bucket
        count = 0
        all_count = 0
        max_range = [float('-inf'),0,0]
        N_i = segment_size[right] - segment_size[left]
        n_i = right - left +1
        for i in range(left, right - min_step + 2): #because both are inclusive, for [0, 1] we already have 2 samples.
            for j in range(i + min_step - 1, right + 1):
                all_count += 1
                if not (i == left and j == right) and random.random() > self.epsilon:
                    continue
                count += 1
                n_i_q_square = (j - i + 1)*(j - i + 1)
                sum_square = (prefix_squaredsum[j] - prefix_squaredsum[i - 1])
                squared_sum = (prefix_sum[j] - prefix_sum[i-1])*(prefix_sum[j] - prefix_sum[i-1])
                var = (1/n_i)*(1 - n_i/N_i)*((n_i/n_i_q_square)*sum_square - (1/n_i_q_square)*squared_sum)
                max_var = max(max_var, var)
                cache[(i,j)] = var

        self.amax_cache[(left, right)] = max_var
        return max_var

    def get_interval_variance(self, i, j, preprocess = None):
        if preprocess is None: preprocess = self.preprocess
        segment_size, prefix_sum, prefix_squaredsum = preprocess
        return self.AMAX(i, j, segment_size, prefix_sum, prefix_squaredsum)
