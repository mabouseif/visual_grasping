import operator
import numpy as np

class SegmentTree(object):
    def __init__(self, capacity, operation, neutral_element):
        """Build a Segment Tree data structure.
        https://en.wikipedia.org/wiki/Segment_tree
        Can be used as regular array, but with two
        important differences:
            a) setting item's value is slightly slower.
               It is O(lg capacity) instead of O(1).
            b) user has access to an efficient ( O(log segment size) )
               `reduce` operation which reduces `operation` over
               a contiguous subsequence of items in the array.
        Paramters
        ---------
        capacity: int
            Total size of the array - must be a power of two.
        operation: lambda obj, obj -> obj
            and operation for combining elements (eg. sum, max)
            must form a mathematical group together with the set of
            possible values for array elements (i.e. be associative)
        neutral_element: obj
            neutral element for the operation above. eg. float('-inf')
            for max and 0 for sum.
        """
        assert capacity > 0 and capacity & (capacity - 1) == 0, "capacity must be positive and a power of 2."
        self._capacity = capacity
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self._operation(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
                )

    def reduce(self, start=0, end=None):
        """Returns result of applying `self.operation`
        to a contiguous subsequence of the array.
            self.operation(arr[start], operation(arr[start+1], operation(... arr[end])))
        Parameters
        ----------
        start: int
            beginning of the subsequence
        end: int
            end of the subsequences
        Returns
        -------
        reduced: obj
            result of reducing self.operation over the specified range of array elements.
        """
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        # index of the leaf
        idx += self._capacity
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(
                self._value[2 * idx],
                self._value[2 * idx + 1]
            )
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(
            capacity=capacity,
            operation=operator.add,
            neutral_element=0.0
        )

    def sum(self, start=0, end=None):
        """Returns arr[start] + ... + arr[end]"""
        return super(SumSegmentTree, self).reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        """Find the highest index `i` in the array such that
            sum(arr[0] + arr[1] + ... + arr[i - i]) <= prefixsum
        if array values are probabilities, this function
        allows to sample indexes according to the discrete
        probability efficiently.
        Parameters
        ----------
        perfixsum: float
            upperbound on the sum of array prefix
        Returns
        -------
        idx: int
            highest index satisfying the prefixsum constraint
        """
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1
        while idx < self._capacity:  # while non-leaf
            if self._value[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1
        return idx - self._capacity


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(MinSegmentTree, self).__init__(
            capacity=capacity,
            operation=min,
            neutral_element=float('inf')
        )

    def min(self, start=0, end=None):
        """Returns min(arr[start], ...,  arr[end])"""

        return super(MinSegmentTree, self).reduce(start, end)
    
    
# from OpenAI's baselines https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
class ExperienceReplay():
    def __init__(self, size):
        self.buffer = []
        self.maxsize = size
        self.next_idx = 0
        self.np_random = np.random.RandomState()

    def __len__(self):
        return len(self.buffer)
    
    def add(self, experience):
        if self.next_idx >= len(self.buffer):   # increase size of buffer if there's still room
            self.buffer.append(experience)
        else:                                   # overwrite old experience
            self.buffer[self.next_idx] = experience
        self.next_idx = (self.next_idx + 1)%self.maxsize

    def sample(self, batch_size):
        # sample indices into buffer
        idxs = self.np_random.randint(0,len(self.buffer),size=(batch_size,))    # randint samples ints from [low,high)
        return self.encode_samples(idxs)
        
    def encode_samples(self, idxs, ranked_priority=False):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for idx in idxs:    # extract experience at given indices
            if ranked_priority:
                state, action, reward, next_state, done = self.buffer[idx][0]
            else:
                state, action, reward, next_state, done = self.buffer[idx]
            
            states.append(state)    # list of int arrays
            actions.append(action)  # list of ints
            rewards.append(reward)  # list of ints
            next_states.append(next_state)  # list of int arrays
            dones.append(done)  # list of bools
        return (np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones))
    
    
# proportional sampling as implemented by OpenAI
# https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
class ProportionalReplay(ExperienceReplay):
    def __init__(self, size, alpha):
        super(ProportionalReplay, self).__init__(size)
        assert alpha >= 0
        self.alpha = alpha

        self.tree_size = 1
        while self.tree_size < self.maxsize:
            self.tree_size *= 2

        self.min_tree = MinSegmentTree(self.tree_size)    # for calculating maximum IS weight
        self.sum_tree = SumSegmentTree(self.tree_size)    # for proportional sampling
        self.max_priority = 1.0   # maximum priority we've seen so far. will be updated

    def add(self, experience):
        idx = self.next_idx     # save idx before it's changed in super call
        super().add(experience) # put experience data (s,a,r,s',done) in buffer

        # give new experience max priority to ensure it's replayed at least once
        self.min_tree[idx] = self.max_priority ** self.alpha 
        self.sum_tree[idx] = self.max_priority ** self.alpha

    # To sample a minibatch of size k, the range [0, p_total] is divided equally into k ranges. 
    # Next, a value is uniformly sampled from each range.
    def sample_proportional(self, batch_size):
        idxs = []
        p_total = self.sum_tree.sum(0, len(self.buffer)-1) # sum of the priorities of all experience in the buffer
        every_range_len = p_total / batch_size  # length of every range over [0,p_total] (batch_size = k)
        for i in range(batch_size): # for each range
            mass = self.np_random.uniform()*every_range_len + i*every_range_len  # uniformly sampling a probability mass from this range
            idx = self.sum_tree.find_prefixsum_idx(mass) # get smallest experience index s.t. cumulative dist F(idx) >= mass
            idxs.append(idx)
        return idxs

    # sample batch of experiences along with their weights and indices
    def sample(self, batch_size, beta):
        assert beta > 0
        idxs = self.sample_proportional(batch_size)    # sampled experience indices

        weights = []
        p_min = self.min_tree.min() / self.sum_tree.sum() # minimum possible priority for a transition
        max_weight = (p_min * len(self.buffer)) ** (-beta)    # (p_uniform/p_min)^beta is maximum possible IS weight

        # get IS weights for sampled experience
        for idx in idxs:
            p_sample = self.sum_tree[idx] / self.sum_tree.sum()   # normalize sampled priority
            weight = (p_sample * len(self.buffer)) ** (-beta) # (p_uniform/p_sample)^beta. IS weight
            weights.append(weight / max_weight) # weights normalized by max so that they only scale the update downwards
        weights = np.array(weights)

        encoded_sample = self.encode_samples(idxs) # collect experience at given indices 
        return tuple(list(encoded_sample) + [weights, idxs])

    # set the priorities of experiences at given indices 
    def update_priorities(self, idxs, priorities):
        assert len(idxs) == len(priorities)
        for idx, priority in zip(idxs, priorities):
            assert priority > 0
            assert 0 <= idx < len(self.buffer)
            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)