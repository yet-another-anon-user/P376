import math
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Analyzer(object):
    def __init__(self, queries = None, baseline_performances = None, baseline=None):
        self.queries = queries
        self.bp = baseline_performances
        self.baseline = baseline
        self.percentiles = [95, 99, 100]

    def initialize(self, queries, baseline_performances):
        self.queries = queries
        self.bp = baseline_performances

    def metric(self):
        raise NotImplemented

    def analyze(self,  verbose = False, percentile = 50):
        cnt_results = defaultdict(list)
        sum_results = defaultdict(list)
        avg_results = defaultdict(list)
        for baseline, perf in self.bp.items():
            for idx, result in enumerate(perf):
                cnt_results[baseline].append(self.metric(self.queries[idx].cnt, result[0]))
                sum_results[baseline].append(self.metric(self.queries[idx].sum, result[1]))
                avg_results[baseline].append(self.metric(self.queries[idx].avg, result[2]))
            if verbose is True:
                print("%s %s P%s %s %s %s" % (self.__class__.__name__, baseline, percentile,
                                "Cnt: %s" % (np.percentile(cnt_results[baseline], percentile)),
                                "Sum: %s" % (np.percentile(sum_results[baseline], percentile)),
                                "Avg: %s" % (np.percentile(avg_results[baseline], percentile)))
                                )
        return cnt_results, sum_results, avg_results

    def analyze_and_save(self, name):
        c, s, m = self.analyze()
        dump_obj([c, s, m], name)

class RelativeError(Analyzer):
    def metric(self, gt, result):
        if gt == 0: return abs(result)
        return abs(float(gt - result)/float(gt))

class RE_w_MinMax(RelativeError):
    def analyze(self,  verbose = False, percentile = 99):
        cnt_results = defaultdict(list)
        sum_results = defaultdict(list)
        avg_results = defaultdict(list)
        min_results = defaultdict(list)
        max_results = defaultdict(list)

        for baseline, perf in self.bp.items():
            for idx, result in enumerate(perf):

                cnt_results[baseline].append(self.metric(self.queries[idx].cnt, result[0]))
                sum_results[baseline].append(self.metric(self.queries[idx].sum, result[1]))
                avg_results[baseline].append(self.metric(self.queries[idx].avg, result[2]))

                min_results[baseline].append(self.metric(self.queries[idx].min, result[6]))
                max_results[baseline].append(self.metric(self.queries[idx].max, result[7]))

            if verbose is True:
                print("%s %s P%s %s %s %s" % (self.__class__.__name__, baseline, percentile,
                                "Cnt: %s" % (np.percentile(cnt_results[baseline], percentile)),
                                "Sum: %s" % (np.percentile(sum_results[baseline], percentile)),
                                "Avg: %s" % (np.percentile(avg_results[baseline], percentile)))
                                )
        return cnt_results, sum_results, avg_results, min_results, max_results

    def analyze_and_save(self, name):
        c, s, m = self.analyze()
        dump_obj([c, s, m], name)

class SkipRate(Analyzer):
    def analyze(self,  verbose = False, percentile = 99):
        results = defaultdict(list)
        for baseline, perf in self.bp.items():
            if 'PASS' not in baseline: continue
            for idx, result in enumerate(perf):
                results[baseline].append(1-result[7] if len(result) > 7 else 0)
            if verbose is True:
                print("%s %s P%s %s " % (self.__class__.__name__, baseline, percentile,
                                "%s" % (np.percentile(results[baseline], percentile)),
                                ))
        return results

class CIRatio(Analyzer):
    def analyze(self,  verbose = False, percentile = 99):
        cnt_results = defaultdict(list)
        sum_results = defaultdict(list)
        avg_results = defaultdict(list)

        for baseline, perf in self.bp.items():
            for idx, result in enumerate(perf):
                cnt_results[baseline].append(self.metric(self.queries[idx].cnt, result[3]))
                sum_results[baseline].append(self.metric(self.queries[idx].sum, result[4]))
                avg_results[baseline].append(self.metric(self.queries[idx].avg, result[5]))
            if verbose is True:
                print("%s %s P%s %s %s %s" % (self.__class__.__name__, baseline, percentile,
                                "Cnt: %s" % (np.percentile(cnt_results[baseline], percentile)),
                                "Sum: %s" % (np.percentile(sum_results[baseline], percentile)),
                                "Avg: %s" % (np.percentile(avg_results[baseline], percentile)))
                                )


        return cnt_results, sum_results, avg_results

    def metric(self, gt, result):
        if gt == 0: gt=1
        return float(result)/float(gt)

class FailureRate(Analyzer):
    def analyze(self, verbose= False):
        cnt_results = defaultdict(list)
        sum_results = defaultdict(list)
        avg_results = defaultdict(list)


        for baseline, perf in self.bp.items():
            for idx, result in enumerate(perf):
                cnt_results[baseline].append(self.metric(self.queries[idx].cnt, result[0], result[3]))
                sum_results[baseline].append(self.metric(self.queries[idx].sum, result[1], result[4]))
                avg_results[baseline].append(self.metric(self.queries[idx].avg, result[2], result[5]))
            if len(perf) > 0 and verbose is True:
                print("%s %s %s %s %s" % (
                                    self.__class__.__name__, baseline,
                                    ("Cnt: %.4f" % (sum(cnt_results[baseline])/len(perf))),
                                    ("Sum: %.4f" % (sum(sum_results[baseline])/len(perf))),
                                    ("Avg: %.4f" % (sum(avg_results[baseline])/len(perf)))
                                )
                            )
        return cnt_results, sum_results, avg_results

    def metric(self, gt, est, half_ci):
        return 1 if gt > est+half_ci or gt < est-half_ci else 0

class PERCENTILE_CI(Analyzer):
    def analyze(self,  verbose = False, percentile = 99):

        cnt_results = defaultdict(list)
        sum_results = defaultdict(list)
        avg_results = defaultdict(list)

        for baseline, perf in self.bp.items():
            for idx, result in enumerate(perf):

                cnt_results[baseline].append(self.metric(result[3]))
                sum_results[baseline].append(self.metric(result[4]))
                avg_results[baseline].append(self.metric(result[5]))
            if verbose is True:
                print("%s %s P%s %s %s %s" % (self.__class__.__name__, baseline, percentile,
                                "Cnt: %s" % (np.percentile(cnt_results[baseline], percentile)),
                                "Sum: %s" % (np.percentile(sum_results[baseline], percentile)),
                                "Avg: %s" % (np.percentile(avg_results[baseline], percentile)))
                                )
        return cnt_results, sum_results, avg_results

    def metric(self, half_ci):
        return float(half_ci)

class New_CIRatio(Analyzer):
    def analyze(self,  verbose = False, percentile = 50):

        cnt_results = defaultdict(list)
        sum_results = defaultdict(list)
        avg_results = defaultdict(list)

        for baseline, perf in self.bp.items():
            for idx, result in enumerate(perf):
                cnt_results[baseline].append(self.metric(self.queries[idx].cnt, result[0], result[3]))
                sum_results[baseline].append(self.metric(self.queries[idx].sum, result[1], result[4]))
                avg_results[baseline].append(self.metric(self.queries[idx].avg, result[2], result[5]))

            if verbose is True:
                print("%s %s P%s %s %s %s" % (self.__class__.__name__, baseline, percentile,
                                "Cnt: %s" % (np.percentile(cnt_results[baseline], percentile)),
                                "Sum: %s" % (np.percentile(sum_results[baseline], percentile)),
                                "Avg: %s" % (np.percentile(avg_results[baseline], percentile)))
                                )
        return cnt_results, sum_results, avg_results

    def metric(self, gt, est, half_ci):
        return float(half_ci)/abs(gt - est) if gt - est != 0 else float(half_ci)

class PERCENTILE_CI_Ratio(Analyzer):
    def analyze(self, verbose = False):
        percentile = 50
        cnt_results = defaultdict(list)
        sum_results = defaultdict(list)
        avg_results = defaultdict(list)
        baseline_perf = self.bp[self.baseline]

        for baseline, perf in self.bp.items():
            for idx, result in enumerate(perf):
                baseline_result = baseline_perf[idx]
                cnt_results[baseline].append(self.metric(result[3], baseline_result[3]))
                sum_results[baseline].append(self.metric(result[4], baseline_result[4]))
                avg_results[baseline].append(self.metric(result[5], baseline_result[5]))
            if verbose is True:
                print("%s %s P%s %s %s %s" % (self.__class__.__name__, baseline, percentile,
                                "Cnt: %s" % (np.percentile(cnt_results[baseline], percentile)),
                                "Sum: %s" % (np.percentile(sum_results[baseline], percentile)),
                                "Avg: %s" % (np.percentile(avg_results[baseline], percentile)))
                                )
        return cnt_results, sum_results, avg_results

    def metric(self, half_ci, baseline_ci):
        return float(half_ci)/float(baseline_ci) if baseline_ci > 0 else 10
