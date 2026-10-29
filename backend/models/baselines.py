import numpy as np

class StrictPriorityScheduler:
    """Always serves the highest priority queue with packets available on the best link."""
    def select_action(self, state):
        # state: [q0, q1, q2, hol0, hol1, hol2, arr0, arr1, arr2, c0, c1, c2]
        queues = state[0:3]
        capacities = state[9:12]
        best_link = int(np.argmax(capacities))
        
        for q_idx in range(3):
            if queues[q_idx] > 0:
                return q_idx * 3 + best_link
                
        return 0 * 3 + best_link


class RoundRobinScheduler:
    """Rotates queue servicing across time steps."""
    def __init__(self):
        self.current_q = 0

    def select_action(self, state):
        capacities = state[9:12]
        best_link = int(np.argmax(capacities))
        
        q_selected = self.current_q
        self.current_q = (self.current_q + 1) % 3
        
        return q_selected * 3 + best_link


class WeightedFairQueuingScheduler:
    """Serves queues proportional to weights: Voice(0.5), Video(0.3), Best Effort(0.2)."""
    def __init__(self):
        self.weights = [0.5, 0.3, 0.2]

    def select_action(self, state):
        queues = state[0:3]
        capacities = state[9:12]
        best_link = int(np.argmax(capacities))
        
        # Weight choice based on queue backlogs
        weighted_backlogs = [queues[i] * self.weights[i] for i in range(3)]
        
        if sum(weighted_backlogs) == 0:
            q_selected = 0
        else:
            q_selected = int(np.argmax(weighted_backlogs))
            
        return q_selected * 3 + best_link