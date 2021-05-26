import numpy as np
from copy import copy

def simulate_episode(init_prob_matrix, n_steps_max):
    prob_matrix = init_prob_matrix.copy()
    n_nodes = prob_matrix.shape[0]
    initial_active_nodes = np.random.binomial(1, 0.1, size=(n_nodes))
    history = np.array([initial_active_nodes])
    active_nodes = initial_active_nodes
    newly_active_nodes = active_nodes
    t = 0
    while t < n_steps_max and np.sum(newly_active_nodes)>0:
        p = (prob_matrix.T * active_nodes).T
        activated_edges = p > np.random.rand(p.shape[0], p.shape[1])

        # remove from the probability matrix all the probabilities related to the previously
        # activated nodes
        prob_matrix = prob_matrix * ( (p!=0) == activated_edges )

        # compute the value of the newly activated nodes
        newly_active_nodes = (np.sum(activated_edges, axis=0) >0 ) * (1-active_nodes)

        active_nodes = np.array( active_nodes + newly_active_nodes)

        history = np.concatenate((history, [newly_active_nodes]), axis=0)

        t += 1

    return history


"""
we will estimate the probability using a credit assignment approach.

p_v,w = sum(credit_u,v) / A_v

credit_u,v = 1 / ( sum{w in S} I(t_w = t_v - 1) )

at each episode in which the node has been active, we can assign credit to its neighbors 
depending whether these nodes have been active at the previous episode (partitioning the credit 
among them)

"""


def estimate_probabilities(dataset, node_index, n_nodes):
    estimated_probabilities = np.ones(n_nodes)*1.0 / (n_nodes -1 )
    credits = np.zeros(n_nodes)
    occur_v_active = np.zeros(n_nodes)
    n_episodes = len(dataset)
    for episode in dataset:
        # in which row the target node has been activated
        idx_w_active = np.argwhere(episode[:, node_index] == 1).reshape(-1)
        if len(idx_w_active) > 0 and idx_w_active > 0:
            active_nodes_in_prev_step = episode[idx_w_active-1,:].reshape(-1)
            credits += active_nodes_in_prev_step / np.sum(active_nodes_in_prev_step)

        #check occurrences of each node in each step
        for v in range(n_nodes):
            if v!=node_index:
                idx_v_active = np.argwhere(episode[:, v] == 1).reshape(-1)
                if len(idx_v_active)>0 and (idx_v_active<idx_w_active or len(idx_w_active)==0):
                    occur_v_active[v] += 1

    estimated_probabilities = credits/occur_v_active
    estimated_probabilities = np.nan_to_num(estimated_probabilities)
    return estimated_probabilities



# example

if __name__ == '__main__':
    n_nodes = 5
    n_episodes = 1000
    prob_matrix = np.random.uniform(0.0, 0.1, (n_nodes, n_nodes))
    node_index = 4
    dataset = []
    for e in range(n_episodes):
        dataset.append(simulate_episode(prob_matrix, n_steps_max=10))

    estimated_probabilities = estimate_probabilities(dataset=dataset,
                                                     node_index=node_index,
                                                     n_nodes=n_nodes)

    print(" True matrix : {}".format(prob_matrix[:,4]))
    print(" Estimated matrix : {}".format(estimated_probabilities))
