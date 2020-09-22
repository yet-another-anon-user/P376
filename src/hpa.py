from collections import defaultdict
from pre_agg import PreAgg, PANode
from utils import *
from varestimator import *
import jsonpickle
import heapq, math
from tqdm import tqdm
import numpy as np
import time

class HPA(object):
    def __init__(self, data, partitioner, max_depth, p_attrs, t_attrs, fan_out=2, max_leaf = 0):
        self.data = data
        self.partitioner = partitioner
        self.root = None
        self.n_pa = 0
        self.fan_out = fan_out
        self.max_depth = max_depth
        self.max_leaf = max_leaf
        self.p_attrs = p_attrs
        self.t_attrs = t_attrs
        self.leaves = [] #this is a max heap
        self.layers = defaultdict(list) #the partitions in each layer
        self.leaf_count = 0
        self.queue = []

    def load_layers(self, name):
        layers = restore_obj(name)
        if layers is None:
            return None
        self.n_pa = 0
        for k, v in layers.items():
            for _ in v:
                self.n_pa += 1
        return layers

    def build_pa_tree(self):
        name = ("%s-%s-%s-%s-HPA-%s-%s.jpk" %
                        (self.data.__class__.__name__, self.max_depth,
                        self.partitioner.__class__.__name__,
                        self.fan_out,
                        '_'.join(self.p_attrs),
                        '_'.join(self.t_attrs)))

        self.layers = self.load_layers(name)
        if self.layers is not None:
            self.root = self.layers[0][0]
            self.leaves = self.layers[len(self.layers)-1]
            return
        else:
            self.layers = defaultdict(list)
        leaf_count = 1
        print("Building PA Tree...", self.max_depth, self.max_leaf)
        self.root = self._build_pa_tree(0, self.max_depth, self.data.df, self.p_attrs, self.t_attrs)
        self.layers[0].append(self.root)
        print("Depth:", len(self.layers))
        self.leaves = self.layers[len(self.layers)-1]
        # print("%s leaf nodes" % len(self.leaves))
        dump_obj(self.layers, name)

    def _build_pa_tree(self, cur_depth, max_depth, cur_partition, p_attrs, t_attrs):

        if cur_depth == max_depth or len(cur_partition) == 0:
            return None

        self.n_pa += 1

        node = PANode(PreAgg(cur_partition, p_attrs, t_attrs), depth = cur_depth)
        current_node = node

        partitions = self.partitioner.partition_to(cur_partition, [self.fan_out]*len(p_attrs), p_attrs, t_attrs)

        for partition in partitions:
            child = self._build_pa_tree(cur_depth+1, max_depth, partition, p_attrs, t_attrs)
            if child is not None:
                node.children.append(child)
                self.layers[cur_depth+1].append(child)

        return node

    def dump_tree(self, path):
        dump_obj(self.root, path)

    def restore_tree(self, path):
        self.root = restore_obj(path)

    def plot_tree(self, layers = -1, verbose = False):
        # plot the first K layers top down.
        print("\n======= PA Tree =======")
        queue = [self.root]
        s = ''
        idx = 0
        cnt = 0
        while len(queue) > 0 or idx < layers:
            layer = []
            s = ''
            s = "L%s: %s nodes" % (idx, len(queue))
            cnt += len(queue)
            if verbose:
                s += '\n'
            while len(queue) > 0:
                if verbose:
                    s += str(queue[0]) + '\n'
                for c in queue[0].children:
                    layer.append(c)
                queue.pop(0)

            print(s)
            queue = layer
            idx += 1
        print("==== %s nodes in tree. =====" % cnt)

    def verify_tree(self, t_attr):
        true_cnt = self.root.pa.cnt
        true_sum = self.root.pa.summaries[t_attr]['sum']
        #verify each layer to see if things add up
        print("\nVerifying PA Tree...")
        queue = [self.root]
        s = ''
        idx = 0
        while len(queue) > 0:
            layer = []
            cnt = 0
            sum = 0
            while len(queue) > 0:
                sum += queue[0].pa.summaries[t_attr]['sum']
                cnt += queue[0].pa.cnt
                for c in queue[0].children:
                    layer.append(c)
                queue.pop(0)
            print('L'+str(idx), cnt, true_cnt, sum, true_sum)
            assert(is_close(cnt, true_cnt) == True and is_close(sum, true_sum) == True)
            queue = layer
            idx += 1
        print("Tree verified.\n")

class KDTree(HPA):
    def build_pa_tree(self):
        name = ("%s-%s-%s-%s-%s-KD-%s-%s.jpk" %
                        (self.data.__class__.__name__, self.max_depth, self.max_leaf,
                        self.partitioner.__class__.__name__,
                        self.fan_out,
                        '_'.join(self.p_attrs),
                        '_'.join(self.t_attrs)))

        self.layers = self.load_layers(name)
        if self.layers is not None:
            self.root = self.layers[0][0]
            self.leaves = self.layers[-1]
            return
        else:
            self.layers = defaultdict(list)

        print("Building KD-PA Tree...")
        self.root = self._build_pa_tree(0, self.max_leaf, self.data.df, self.p_attrs, self.t_attrs)
        self.layers[0].append(self.root)

        dump_obj(self.layers, name)

    def verify_tree(self, t_attr):
        #verify the tree: see if leaves add up to the entire dataset.
        true_sum = self.root.pa.summaries[t_attr]['sum']
        true_cnt = self.root.pa.cnt
        l_sum = 0
        l_cnt = 0

        for leaf in self.leaves:
            l_cnt += leaf.pa.cnt
            l_sum += leaf.pa.summaries[t_attr]['sum']
        print(l_sum, true_sum, l_cnt, true_cnt)
        assert(is_close(l_cnt, true_cnt) == True and is_close(l_sum, true_sum) == True)
        print("Verified KD-PA-Tree, %s leaf nodes form a partition of the dataset." % len(self.leaves))

    def _build_pa_tree(self, cur_depth, max_leaf, cur_partition, p_attrs, t_attrs):
        if len(cur_partition) == 0:
            return None

        node = PANode(PreAgg(cur_partition, p_attrs, t_attrs), depth = cur_depth)
        root = node
        self.leaves = [(1, (cur_partition, 0, node))]
        var_0_offset = 0
        self.n_pa = 1
        while len(self.leaves) < max_leaf:
            attr_idx = cur_depth%len(p_attrs)

            v, (cur_partition, cur_depth, node) = heapq.heappop(self.leaves)

            partitions = self.partitioner.partition_to(cur_partition, 2, [p_attrs[attr_idx]], t_attrs)

            if len(partitions) == 1:
                heapq.heappush(self.leaves, (v+100, (cur_partition, cur_depth, node)))

            elif len(partitions) > 1:
                for p in partitions:
                    self.n_pa += 1
                    child = PANode(PreAgg(p, p_attrs, t_attrs), depth = cur_depth+1)
                    node.children.append(child)
                    self.layers[cur_depth+1].append(child)
                    v = (-1*(p[t_attrs[0]].var())) if p.shape[0] > 10 else 0+var_0_offset
                    if p.shape[0] <= 10: var_0_offset += 0.001

                    heapq.heappush(self.leaves, (v, (p, cur_depth+1, child)))

        # TODO: duplicate code here after adding NKD, refactoring needed.

        min_depth = 99999
        min_node_shape = None
        min_var = 99999

        max_depth = 0
        max_depth_node_shape = None
        max_depth_var = 999999

        max_var = 99999
        max_var_shape = None
        max_var_depth = None
        for idx, leaf in enumerate(self.leaves):
            if leaf[0] < max_var:
                max_var = leaf[0]
                max_var_shape = leaf[1][0].shape
                max_var_depth = leaf[1][1]
            if leaf[1][1] >= max_depth:
                max_depth = leaf[1][1]
                max_depth_node_shape = leaf[1][0].shape
                max_depth_var = leaf[0]
            if leaf[1][1] < min_depth:
                min_depth = leaf[1][1]
                min_node_shape = leaf[1][0].shape
                min_var = leaf[0]
            self.leaves[idx] = leaf[1][2]
            self.layers[-1].append(leaf[1][2])
        print("PA-Tree Summary\nMin Depth Leaf Node: \n\tdepth: %s, variance: %s, partition shape: %s"
              "\nMax Var Leaf Node (next to split): \n\tdepth: %s, variance: %s, partition shape: %s "
              "\nTotal Leaf Nodes: %s, Total Nodes in Tree: %s" %
                (min_depth, abs(min_var), min_node_shape,
                max_var_depth, abs(max_var), max_var_shape,
                len(self.leaves),
                self.n_pa))
        return root


class NKDTree(KDTree):
    '''
    We use equal partitioner, but AMAX to choose which node to split.
    And restrain the max depth diff. if exceeded, then we will split the lest split node instead (i.e. the node w/ largest sample).
    '''
    def __init__(self, data, partitioner, max_depth, p_attrs, t_attrs, fan_out=2, max_leaf = 0):
        super().__init__(data, partitioner, max_depth, p_attrs, t_attrs, fan_out, max_leaf)
        self.max_depth_diff = 2
        self.var_est = None

    def build_pa_tree(self):
        name = ("%s-%s-%s-%s-%s-NKD-%s-%s.jpk" %
                        (self.data.__class__.__name__, self.max_depth, self.max_leaf,
                        self.partitioner.__class__.__name__,
                        self.fan_out,
                        '_'.join(self.p_attrs),
                        '_'.join(self.t_attrs)))

        self.layers = self.load_layers(name)
        if self.layers is not None:
            self.root = self.layers[0][0]
            self.leaves = self.layers[-1]
            return
        else:
            self.layers = defaultdict(list)

        print("\nInitializing estimators...")

        self.var_est = AMAXND(self.max_leaf)
        self.var_est.initialize(self.data.df, self.p_attrs, self.t_attrs[0])

        print("\nBuilding KD-PA Tree...")

        self.root = self._build_pa_tree(0, self.max_leaf, self.data.df, self.p_attrs, self.t_attrs)
        self.layers[0].append(self.root)

        dump_obj(self.layers, name)

    def _build_pa_tree(self, cur_depth, max_leaf, cur_partition, p_attrs, t_attrs):
        if len(cur_partition) == 0:
            return None

        node = PANode(PreAgg(cur_partition, p_attrs, t_attrs), depth = cur_depth)
        root = node
        self.leaves = [(1, (cur_partition, 0, node))]
        var_0_offset = 0
        self.n_pa = 1
        max_depth = 0
        last_len = 0
        pbar = tqdm(total = max_leaf)
        while len(self.leaves) < max_leaf :
            if len(self.leaves) > last_len:
                pbar.update(len(self.leaves) - last_len)
                last_len = len(self.leaves)

            v, (cur_partition, cur_depth, node) = heapq.heappop(self.leaves)

            partitions = self.partitioner.partition_to(cur_partition, [2]*len(p_attrs), p_attrs, t_attrs)
            print("Depth %s/%s, %s partitions splitted, Total Nodes: %s, Leaf: %s" % (cur_depth, self.max_depth, len(partitions), self.n_pa, len(self.leaves)))

            if len(partitions) <= 1:
                heapq.heappush(self.leaves, (v+100, (cur_partition, cur_depth, node)))

            elif len(partitions) > 1:
                for p in partitions:
                    assert(p.shape[0] > 0)
                    self.n_pa += 1
                    child = PANode(PreAgg(p, p_attrs, t_attrs), depth = cur_depth+1)
                    node.children.append(child)
                    self.layers[cur_depth+1].append(child)

                    v = -1*(self.var_est.get_error(p)) if p.shape[0] > 10 else 0+var_0_offset

                    if p.shape[0] <= 10: var_0_offset += 0.001
                    heapq.heappush(self.leaves, (v, (p, cur_depth+1, child)))

                    max_depth = max(cur_depth + 1, max_depth)

                min_depth_leaf = (0, (0, 99999))
                min_idx = -1
                for idx, l in enumerate(self.leaves):

                    depth = l[1][1]
                    if depth < min_depth_leaf[1][1]:
                        min_depth_leaf = l
                        min_idx = idx
                    elif depth == min_depth_leaf[1][1]:
                        top_node = min_depth_leaf[1][2]
                        cur_node = l[1][2]
                        if top_node.pa.cnt < cur_node.pa.cnt:
                            min_depth_leaf = l
                            min_idx = idx

                if min_idx > 0 and max_depth - min_depth_leaf[1][1] > self.max_depth_diff:
                    new_pq = []
                    while len(self.leaves) > 0:
                        p, q = heapq.heappop(self.leaves)

                        if min_depth_leaf[1][2] == q[2]:
                            p = float("-inf")

                        new_pq.append((p, q))
                    for entry in new_pq:
                        heapq.heappush(self.leaves, (entry[0], entry[1]))
        pbar.close()

        min_depth = 99999
        min_node_shape = None
        min_var = 99999

        max_depth = 0
        max_depth_node_shape = None
        max_depth_var = 999999

        max_var = 99999
        max_var_shape = None
        max_var_depth = None
        for idx, leaf in enumerate(self.leaves):
            if leaf[0] < max_var:
                max_var = leaf[0]
                max_var_shape = leaf[1][0].shape
                max_var_depth = leaf[1][1]
            if leaf[1][1] >= max_depth:
                max_depth = leaf[1][1]
                max_depth_node_shape = leaf[1][0].shape
                max_depth_var = leaf[0]
            if leaf[1][1] < min_depth:
                min_depth = leaf[1][1]
                min_node_shape = leaf[1][0].shape
                min_var = leaf[0]
            self.leaves[idx] = leaf[1][2]
            self.layers[-1].append(leaf[1][2])
        print("PA-Tree Summary\nMin Depth Leaf Node: \n\tdepth: %s, variance: %s, partition shape: %s"
              "\nMax Var Leaf Node (next to split): \n\tdepth: %s, variance: %s, partition shape: %s "
              "\nTotal Leaf Nodes: %s, Total Nodes in Tree: %s" %
                (min_depth, abs(min_var), min_node_shape,
                max_var_depth, abs(max_var), max_var_shape,
                len(self.leaves),
                self.n_pa))
        return root


class NaiveKDTree(KDTree):
    '''
    We use equal partitioner, but AMAX to choose which node to split.
    And restrain the max depth diff. if exceeded, then we will split the lest split node instead (i.e. the node w/ largest sample).
    '''
    def __init__(self, data, partitioner, max_depth, p_attrs, t_attrs, fan_out=2, max_leaf = 0):
        super().__init__(data, partitioner, max_depth, p_attrs, t_attrs, fan_out, max_leaf)
        self.max_depth_diff = 2
        self.var_est = None

    def build_pa_tree(self):
        name = ("%s-%s-%s-%s-%s-NaiveKD-%s-%s.jpk" %
                        (self.data.__class__.__name__, self.max_depth, self.max_leaf,
                        self.partitioner.__class__.__name__,
                        self.fan_out,
                        '_'.join(self.p_attrs),
                        '_'.join(self.t_attrs)))

        self.layers = self.load_layers(name)
        if self.layers is not None:
            self.root = self.layers[0][0]
            self.leaves = self.layers[-1]
            return
        else:
            self.layers = defaultdict(list)

        print("\nBuilding KD-PA Tree...")

        self.root = self._build_pa_tree(0, self.max_leaf, self.data.df, self.p_attrs, self.t_attrs)
        self.layers[0].append(self.root)

        dump_obj(self.layers, name)

    def _build_pa_tree(self, cur_depth, max_leaf, cur_partition, p_attrs, t_attrs):
        if len(cur_partition) == 0:
            return None

        node = PANode(PreAgg(cur_partition, p_attrs, t_attrs), depth = cur_depth)
        root = node
        self.leaves = [(1, (cur_partition, 0, node))]
        var_0_offset = 0
        self.n_pa = 1
        max_depth = 0
        last_len = 0
        pbar = tqdm(total = max_leaf)
        while len(self.leaves) < max_leaf :
            if len(self.leaves) > last_len:
                pbar.update(len(self.leaves) - last_len)
                last_len = len(self.leaves)

            v, (cur_partition, cur_depth, node) = heapq.heappop(self.leaves)

            partitions = self.partitioner.partition_to(cur_partition, [2]*len(p_attrs), p_attrs, t_attrs)

            if len(partitions) <= 1:
                heapq.heappush(self.leaves, (v+100, (cur_partition, cur_depth, node)))
            elif len(partitions) > 1:
                for p in partitions:
                    assert(p.shape[0] > 0)
                    self.n_pa += 1
                    child = PANode(PreAgg(p, p_attrs, t_attrs), depth = cur_depth+1)
                    node.children.append(child)

                    self.layers[cur_depth+1].append(child)

                    v = -1*((100 - cur_depth) + random.random())

                    if p.shape[0] <= 10: var_0_offset += 0.001

                    heapq.heappush(self.leaves, (v, (p, cur_depth+1, child)))

                    max_depth = max(cur_depth + 1, max_depth)

        pbar.close()

        min_depth = 99999
        min_node_shape = None
        min_var = 99999

        max_depth = 0
        max_depth_node_shape = None
        max_depth_var = 999999

        max_var = 99999
        max_var_shape = None
        max_var_depth = None
        for idx, leaf in enumerate(self.leaves):
            if leaf[0] < max_var:
                max_var = leaf[0]
                max_var_shape = leaf[1][0].shape
                max_var_depth = leaf[1][1]
            if leaf[1][1] >= max_depth:
                max_depth = leaf[1][1]
                max_depth_node_shape = leaf[1][0].shape
                max_depth_var = leaf[0]
            if leaf[1][1] < min_depth:
                min_depth = leaf[1][1]
                min_node_shape = leaf[1][0].shape
                min_var = leaf[0]
            self.leaves[idx] = leaf[1][2]
            self.layers[-1].append(leaf[1][2])
        print("PA-Tree Summary\nMin Depth Leaf Node: \n\tdepth: %s, variance: %s, partition shape: %s"
              "\nMax Var Leaf Node (next to split): \n\tdepth: %s, variance: %s, partition shape: %s "
              "\nTotal Leaf Nodes: %s, Total Nodes in Tree: %s" %
                (min_depth, abs(min_var), min_node_shape,
                max_var_depth, abs(max_var), max_var_shape,
                len(self.leaves),
                self.n_pa))
        return root


#BottomUpBinaryTree
class BUBT(HPA):
    # 1D only
    def build_pa_tree(self, force=False):
        name = ("%s-%s-%s-%s-%s-%s-BUBT-%s-%s.jpk" %
                        (self.data.__class__.__name__,
                        self.max_depth, self.max_leaf,
                        self.partitioner.__class__.__name__,
                        self.partitioner.sr,
                        self.partitioner.epsilon,
                        '_'.join(self.p_attrs),
                        '_'.join(self.t_attrs)))

        self.layers = self.load_layers(name)
        if self.layers is not None and force is False:
            self.root = self.layers[0][0]
            self.leaves = self.layers[-1]
            self.partitioner.show_leaf_errors(self.leaves, self.data.df,
                                             self.p_attrs, self.t_attrs)
            return
        else:
            self.layers = defaultdict(list)
        leaves = self.partitioner.partition_to(self.data.df, self.max_leaf, self.p_attrs, self.t_attrs)
        print("Building Binary Tree from Bottom Up\n>>>>>\tLeaf layer %s" % len(leaves))
        self.build_tree_from_leaves(self.data.df, leaves, self.p_attrs, self.t_attrs)
        self.root = self.layers[0][0]
        dump_obj(self.layers, name)

    def build_tree_from_leaves(self, data, leaves, p_attrs, t_attrs):

        leaf_depth = int(math.log(len(leaves), 2))
        p_attr = p_attrs[0]
        for leaf in leaves:
            chunk = data[(data[p_attr] >= leaf[0]) & (data[p_attr] <= leaf[1])]
            node = PANode(PreAgg(chunk, p_attrs, t_attrs), depth = leaf_depth)
            self.layers[leaf_depth].append(node)
            self.layers[-1].append(node)

        layers = defaultdict(list)
        layers[leaf_depth] = leaves

        depths = sorted(list(range(leaf_depth)), reverse = True)
        for depth in depths:

            child_predicates = layers[depth+1]
            child_nodes = self.layers[depth+1]

            for idx in list(range(len(child_predicates)))[0::2]:
                l = child_predicates[idx][0]
                r = child_predicates[idx+1][1]
                left_child = child_nodes[idx]
                right_child = child_nodes[idx+1]

                chunk = data[(data[p_attr] >= l) & (data[p_attr] <= r)]

                node = PANode(PreAgg(chunk, p_attrs, t_attrs), depth = depth)
                node.children = [left_child, right_child]

                self.layers[depth].append(node)

                assert(child_predicates[idx][1]+1 == child_predicates[idx+1][0])

                layers[depth].append([l, r])
            print(">>>>>\tLayer ", depth, len(layers[depth]))
        self.leaves = self.layers[-1]

    def verify_tree(self, t_attr):
        #verify the tree: see if leaves add up to the entire dataset.
        true_sum = self.root.pa.summaries[t_attr]['sum']
        true_cnt = self.root.pa.cnt
        l_sum = 0
        l_cnt = 0

        print("Verifying PA-Tree...")
        for leaf in self.leaves:
            l_cnt += leaf.pa.cnt
            l_sum += leaf.pa.summaries[t_attr]['sum']

        assert(is_close(l_cnt, true_cnt) == True and is_close(l_sum, true_sum) == True)
        print("Verified BUBT-PA-Tree, %s leaf nodes form a partition of the dataset." % len(self.leaves))

