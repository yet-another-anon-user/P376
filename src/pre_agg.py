import numpy as np
from query_gen import Query
from collections import defaultdict
import hashlib

class PreAgg(object):

    def __init__(self, data, p_attrs, t_attrs, depth = None):
        self.predicates = defaultdict(list)
        self.summaries = defaultdict(dict)
        self.p_attrs = p_attrs
        self.t_attrs = t_attrs
        self.cnt = len(data)
        s = ''

        if len(data) == 0:
            return

        for attr in p_attrs:
            self.predicates[attr] = [min(data[attr]), max(data[attr])]
            s += attr + str(self.predicates[attr])

        for attr in t_attrs:
            self.summaries[attr] = {'min':min(data[attr]), 'max':max(data[attr]),
                                    'avg': np.mean(data[attr]), 'sum':np.sum(data[attr])}

        self.id = hashlib.sha1(s.encode()).hexdigest()[:10]

    @classmethod
    def from_query(cls, query):
        obj = cls([], [], [])
        obj.predicates = query.predicates
        obj.id = query.id
        obj.p_attrs = list(query.predicates.keys())
        obj.t_attrs = [query.target_attr]
        obj.cnt = query.cnt
        obj.summaries = {query.target_attr:{'sum': query.sum, 'avg': query.avg}}
        return obj

    def __str__(self):
        s = ''

        for attr in self.predicates.keys():
            s += attr + str(self.predicates[attr])
        s += 'c=' + str(self.cnt) +', s='
        s += "%.2f" % (self.summaries[self.t_attrs[0]]['sum']) + ', a='
        s += "%.2f" % (self.summaries[self.t_attrs[0]]['avg']) + ';'
        return s

class PANode(object):
    '''
    Just a wrapper of PA
    '''
    def __init__(self, pa, depth = None):
        self.children = []
        self.pa = pa
        self.depth = depth #depth in pctree
        self.id = pa.id

    def cost(self):
        return len(self.children)

    def _intersect(self, pred, target):
        '''
        Return true if target intersect with pred
        '''
        for attr in pred.predicates.keys():
            if attr in target.predicates and self.dimension_intersect(pred.predicates[attr], target.predicates[attr]) == False:
                return False
        return True

    def _cover(self, pred, target, verbose=False):
        '''
        Return true if target is covered by pred
        '''

        if set(list(pred.predicates.keys())) != set(list(target.predicates.keys())):
            return False

        for attr in pred.predicates.keys():
            if attr in target.predicates and self.dimension_cover(pred.predicates[attr], target.predicates[attr]) == False:
                return False

        return True

    def dimension_cover(self, p1, p2):
        '''
        each partition is from a different partition, only intersect with target on one dimension
        '''
        if min(p1[1], p2[1]) < max(p1[0], p2[0]):
            return False
        if (p1[0] <= p2[0]) and (p1[1] >= p2[1]):
            return True
        return False

    def dimension_intersect(self, p1, p2):
        '''
        each partition is from a different partition, only intersect with target on one dimension
        '''
        if min(p1[1], p2[1]) < max(p1[0], p2[0]):
            return False

        return True

    def cover(self, node, verbose=False):
        assert(type(node) == PANode)
        return self._cover(self.pa, node.pa, verbose)

    def intersect(self, node):
        assert(type(node) == PANode)
        return self._intersect(self.pa, node.pa)

    def __str__(self):
        s = str(self.pa)
        if self.depth is not None:
            s += "@L"+str(self.depth)
        return s