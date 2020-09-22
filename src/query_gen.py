from utils import *
from tqdm import tqdm
import random
import hashlib

class Query(object):
    def __init__(self, predicates, s, c, a, mi, ma, t):
        self.predicates = predicates
        self.target_attr = t

        #ground truth: sum and cnt
        self.sum = s
        self.cnt = c
        self.avg = a
        self.min = mi
        self.max = ma

        s = ''
        for attr, value in predicates.items():
            s += attr + str(predicates[attr])
        self.id = hashlib.sha1(s.encode()).hexdigest()[:10]

    def update(self, df):
        # update the ground truth when predicates is changed
        data = df

        for attr in self.predicates.keys():
            pred = self.predicates[attr]
            data = data[(data[attr] >= pred[0]) & (data[attr] <= pred[1])]

        suma = sum(data[self.target_attr])

        self.sum = suma
        self.cnt = data.shape[0]
        self.avg = data[self.target_attr].mean()

        s = ''
        for attr, value in self.predicates.items():
            s += attr + str(predicates[attr])
        self.id = hashlib.sha1(s.encode()).hexdigest()[:10]

    def __str__(self):
        s = 'Q:'
        for k, v in self.predicates.items():
            s += k + '[' + ','.join([str(i) for i in v]) + '];'
        s += ';cnt: ' + str(self.cnt) + '; sum: ' + str(self.sum) + '; avg: ' + str(self.avg)
        return s

class QueryGenerator(object):
    def __init__(self, min_p=None, max_p=None):
        self.min_p = min_p
        self.max_p = max_p
        if min_p is None:
            self.min_max_p_str = ""
        else:
            attr = list(self.min_p.keys())[0]
            self.min_max_p_str = "-" + attr +"-"+str(self.min_p[attr])+"-"+str(self.max_p[attr])

    def generate_special_query(self, df, attr, sum_attr):
        preds = {}
        data = df
        min_a, max_a = 6597, 6619
        pred = [min_a, max_a]
        data = data[(data[attr] >= pred[0]) & (data[attr] <= pred[1])]
        preds[attr] = pred
        suma = sum(data[sum_attr])
        avg = data[sum_attr].mean() if data.shape[0] > 0 else 0
        q = Query(preds, c = data.shape[0], s = suma, a = avg, t = sum_attr)
        print('Special query:', q)
        return q

    def preprocessing(self, df):
        return df

    def append_min_max(self, df, q):
        preds = q.predicates
        data = df
        sum_attr = q.target_attr
        for attr in preds.keys():
            pred = preds[attr]
            data = data[(data[attr] >= pred[0]) & (data[attr] <= pred[1])]
        suma = sum(data[sum_attr])
        avg = data[sum_attr].mean() if data.shape[0] > 0 else 0
        assert(suma == q.sum and avg == q.avg)

        return Query(preds, c = data.shape[0], s = suma, a = avg, mi=min(data[sum_attr]), ma=max(data[sum_attr]), t = sum_attr)


    def generate_queries(self, dataset, pred_attrs, target_attr, n_query):
        ret = []

        pred_attrs = sorted(pred_attrs)

        target_name = ("%s-%sQuery-%s-%s-%s%s.jpk" %
                        (dataset.__class__.__name__,
                        self.__class__.__name__,
                        '_'.join(pred_attrs),
                        target_attr, n_query, self.min_max_p_str))

        queries = restore_obj(target_name)
        if queries is not None and n_query > 0:
            added_min_max = False
            for idx, q in enumerate(queries):
                if hasattr(q, 'avg') == False:
                    q.avg = q.sum/q.cnt
                    queries[idx] = q
                if hasattr(q, 'min') == False or hasattr(q, 'max') == False:
                    added_min_max = True
                    queries[idx] = self.append_min_max(dataset.df, q)
            if added_min_max is True:
                dump_obj(queries, target_name)
            return queries

        print("Generating queries")
        if n_query == 0:
            ret.append(self.generate_special_query(dataset.df, pred_attrs[0], target_attr))
        else:
            df = self.preprocessing(dataset.df)
            for n in tqdm(range(n_query)):
                ret.append(self.generate_query(df, pred_attrs, target_attr))

        dump_obj(ret, target_name)
        return ret

    def generate_query(self, dataset, p_attrs, t_attr):
        raise NotImplementedError

class RangeSum(QueryGenerator):
    def generate_query(self, df, attrs, sum_attr):
        preds = {}
        data = df
        try:
            samples = data.sample(n=2)
            for attr in attrs:

                max_a = int(max(samples[attr]))
                min_a = int(min(samples[attr]))

                if min_a == max_a:
                    raise Exception("Too Small")
                pred = [min_a, max_a] #self.random_pred_gen(min_a, max_a)
                data = data[(data[attr] >= pred[0]) & (data[attr] <= pred[1])]
                if data.shape[0] < 20:
                    raise Exception("Too Small")
                preds[attr] = pred
            suma = sum(data[sum_attr])
            avg = data[sum_attr].mean() if data.shape[0] > 0 else 0
            # return Query(preds, c = data.shape[0], s = suma, a = avg, t = sum_attr)
            return Query(preds, c = data.shape[0], s = suma, a = avg, mi=min(data[sum_attr]), ma=max(data[sum_attr]), t = sum_attr)
        except Exception as e:
            # print(e)
            return self.generate_query(df, attrs, sum_attr)

    def random_pred_gen(self, range_min, range_max):
        r1 = random.randint(range_min, range_max)
        r2 = random.randint(range_min, range_max)
        while r1 == r2:
            r2 = random.randint(range_min, range_max)
        return [min(r1, r2), max(r1, r2)]

class IntervalRangeSum(RangeSum):
    def preprocessing(self, df):
        attr = list(self.min_p.keys())[0]
        print("IntervalRangeSum preprocessing", self.min_p, self.max_p, df.shape)
        pred = [self.min_p[attr], self.max_p[attr]]
        df = df[(df[attr] >= pred[0]) & (df[attr] <= pred[1])]
        print("preprocessed", df.shape)
        return df

class BFRangeSum(RangeSum):
    #brute forcely generate every query in the range N^2
    def generate_queries(self, dataset, attr, sum_attr, min_a, max_a):
        data = dataset.df
        queries = []
        pred_attrs = [attr]
        target_name = ("%s-%sQuery-%s-%s-%s-%s.jpk" %
                        (dataset.__class__.__name__,
                        self.__class__.__name__,
                        '_'.join(pred_attrs),
                        sum_attr, min_a, max_a))

        queries = restore_obj(target_name)
        if queries is not None:
            for idx, q in enumerate(queries):
                if hasattr(q, 'avg') == False:
                    q.avg = q.sum/q.cnt
                    queries[idx] = q
            return queries
        print("Generating queries: %s" % ((max_a - min_a) * (max_a - min_a)))
        idx = 0
        queries = []
        existed = set()
        for l in range(min_a-1, max_a-1):
            for r in range(l+2, max_a+1):
                df = data
                preds = {}
                print("%s" % idx, end="\r", flush = True)
                idx += 1
                pred = [l, r]
                if ",".join([str(i) for i in pred]) in existed:
                    print(pred, "exists", existed)
                    assert(False)
                existed.add(",".join([str(i) for i in pred]))
                df = df[(df[attr] >= pred[0]) & (df[attr] <= pred[1])]
                preds[attr] = pred
                if df.shape[0] < 5:
                    continue
                suma = sum(df[sum_attr])
                avg = df[sum_attr].mean() if df.shape[0] > 0 else 0
                queries.append(Query(preds, c = df.shape[0], s = suma, a = avg, t = sum_attr))
        print("Generated %s queries" % len(queries))
        dump_obj(queries, target_name)
        return queries


