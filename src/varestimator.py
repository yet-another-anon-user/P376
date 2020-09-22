import itertools, math
import copy
from tqdm import tqdm
import random
import pandas as pd
import numpy as np
import sklearn.utils
from collections import defaultdict


class AMAX1D(object):
    '''
    TODO: duplicate code from partitioner, should refactor partitioner to use an estimator class.
    '''
    def __init__(self, data, n, p_attr, t_attr, eps=0.1):
        self.seed = 12345
        self.p_attr = p_attr
        self.t_attr = t_attr
        self.data = data
        self.sample = None
        self.epsilon = eps
        self.n_sample = n
        self.amax_cache = {}
        self.min_samples_per_bucket = 3
        self.segment_size = defaultdict(int)
        self.prefix_sum = defaultdict(float)
        self.prefix_squaredsum = defaultdict(float)
        random.seed(self.seed)

    def draw_samples(self):
        n = self.n_sample
        print("AMAX1D sampling n=%s %s" %(n, self.p_attr))
        data = self.data

        data = data.sort_values(by = [self.p_attr], ignore_index = True)

        sample = data.sample(n = n, random_state = self.seed)
        sample = sample[[self.p_attr, self.t_attr]]
        sample = sample.sort_values(by = [self.p_attr])
        sample = pd.concat([data[0:1], sample, data[-1:]])
        self.sample = sample
        self.idxsample = sample[[self.p_attr]].reset_index(-1)

        print("Sample Description", sample.shape)
        print(sample.describe())


    def preprocessing(self):
        self.draw_samples()
        sample = self.sample
        print("AMAX1D Preprocessing", self.p_attr)

        for idx, si in enumerate(sample.index):
            self.segment_size[idx] = si

        for idx, t in enumerate(sample[self.t_attr].tolist()):
            self.prefix_sum[idx] = t + self.prefix_sum[idx-1]
            self.prefix_squaredsum[idx] = t*t + self.prefix_squaredsum[idx-1]

    def AMAX(self, min_p, max_p):
        sample = copy.deepcopy(self.idxsample)
        sample = sample[(sample[self.p_attr] >= min_p) & (sample[self.p_attr] <= max_p)]
        left = min(sample.index)
        right = max(sample.index)
        cache = {}
        if (left, right) in self.amax_cache:
            return self.amax_cache[(left, right)]
        segment_size = self.segment_size
        prefix_sum = self.prefix_sum
        prefix_squaredsum = self.prefix_squaredsum
        max_var = 0
        min_step = self.min_samples_per_bucket
        count = 0
        all_count = 0
        max_range = [float('-inf'),0,0]
        N_i = segment_size[right] - segment_size[left]
        n_i = right - left +1

        for i in range(left, right - min_step + 2):
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

class AMAXND(object):

    def __init__(self, max_k, eps=0.1):
        self.seed = 12345
        self.preprocess = None
        self.sample = None
        self.epsilon = eps
        self.amax_cache = {}
        self.min_samples_per_bucket = 10
        self.max_k = max_k
        self.estimators = {}
        self.p_attrs = None
        random.seed(self.seed)

    def initialize(self, data, p_attrs, t_attr):
        n = min(self.max_k * self.min_samples_per_bucket * 10, 10000)
        self.p_attrs = p_attrs
        for p_attr in p_attrs:
            est = AMAX1D(data, n, p_attr, t_attr)
            est.preprocessing()
            self.estimators[p_attr] = est

    def get_error(self, partition):
        errors = []
        for p_attr in self.p_attrs:
            r = partition[p_attr]
            error = self.estimators[p_attr].AMAX(min(r), max(r))
            errors.append(error)

        return np.mean(errors) + random.random()/100


