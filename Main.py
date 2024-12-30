# action as weight
import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from functools import partial
from sklearn.preprocessing import normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm, trange
# import plotly.express as px
import pandas as pd
import pygal
from bisect import bisect
from collections import defaultdict
import math


def generate_SF_network(N, gamma, L):
    alpha = 1 / (gamma - 1)
    n = np.linspace(1, N, N)
    eta = n ** (-alpha)
    nom_eta = eta / np.sum(eta)
    random.shuffle(nom_eta)
    cum_eta = np.array([np.sum(nom_eta[:i]) for i in range(N)])
    edges = []

    c = 0
    while c < L:
        # i = bisection_search(cum_eta, np.random.rand(2)[0])
        # j = bisection_search(cum_eta, np.random.rand(2)[1])
        i = bisect(cum_eta, np.random.rand(2)[0])
        j = bisect(cum_eta, np.random.rand(2)[1])
        if i == j:
            continue
        e1 = (i, j)
        e2 = (j, i)
        if e1 not in edges and e2 not in edges:
            edges.append(e1)
            c += 1

    G = nx.Graph()
    G.add_edges_from(edges)

    return G


def runge_kutta(y, t, T, dx, f):
    """ y is the initial value for y
        x is the initial value for x
        dx is the time step in x
        f is derivative of function y(t)
    """
    iter = int((T-t)/dx)
    u = np.array((iter + 1) * [y])
    time_sequence_aux = np.linspace(t, T, iter+1)
    time_sequence_aux2 = np.expand_dims(time_sequence_aux, 1).repeat(np.size(y, 0) * np.size(y, 1), axis=1)
    time_sequence = time_sequence_aux2.reshape((iter + 1), np.size(y, 0), np.size(y, 1))
    for cnt_iter in trange(iter):
    #  龙格库塔标准公式
    #  k1 = dx * f(u[cnt_iter], time_sequence[cnt_iter])
    #  k2 = dx * f(u[cnt_iter] + 0.5 * k1, time_sequence[cnt_iter] + 0.5 * dx)
    #  k3 = dx * f(u[cnt_iter] + 0.5 * k2, time_sequence[cnt_iter] + 0.5 * dx)
    #  k4 = dx * f(u[cnt_iter] + k3, time_sequence[cnt_iter] + dx)
        k1 = np.dot(dx, f(u[cnt_iter], time_sequence[cnt_iter]))
        k2 = np.dot(dx, f(u[cnt_iter] + np.dot(0.5, k1), time_sequence[cnt_iter] + np.dot(0.5 * dx, np.ones_like(time_sequence[cnt_iter]))))
        k3 = np.dot(dx, f(u[cnt_iter] + np.dot(0.5, k2), time_sequence[cnt_iter] + np.dot(0.5 * dx, np.ones_like(time_sequence[cnt_iter]))))
        k4 = np.dot(dx, f(u[cnt_iter] + k3, time_sequence[cnt_iter] + np.dot(dx, np.ones_like(time_sequence[cnt_iter]))))
    #  k1 = [dx * value for value in f(u[cnt_iter], time_sequence[cnt_iter])]
    #  k2 = [dx * value for value in f(list(u[cnt_iter]) + [0.5 * value2 for value2 in k1], list(time_sequence[cnt_iter] + 0.5 * dx * np.ones_like(time_sequence[cnt_iter])))]
        u[cnt_iter + 1] = u[cnt_iter] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return u, time_sequence

def sat_function(x,UB,LB):
    # upper_matrix = UB*np.ones_like(x)
    # lower_matrix = LB*np.ones_like(x)
    # x[x>upper_matrix] = UB
    # x[x<lower_matrix] = LB
    ## Sigmoid
    # y = 1. / (1 + np.exp(-0.1*x))
    # Tanh
    y = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    return y


## The opinion similarity between any two agents
## Return 'state_full_cosine_binary', 'clusters', and 'Shannon_value'
def similarity_function(x):
    # cosine similarity
    x_r = np.repeat(x,x.shape[0],axis=0)
    x_t = np.tile(x, (x.shape[0],1))
    cosine = np.sum(x_r * x_t, axis=1)\
        /(np.linalg.norm(x_r, axis=1)\
        *np.linalg.norm(x_t, axis=1))
    x_cosineMatrix = np.reshape(cosine, (x.shape[0],-1))

    return x_cosineMatrix

def shannon_of_clusters(state_discrete_full, action_discrete_full):
    t_seq_len = state_discrete_full.shape[0]
    clusters=defaultdict(dict)
    state_full_cosine_binary = np.zeros((t_seq_len,Num_agent,Num_agent))
    Shannon_value = np.zeros((t_seq_len,1))
    Shannon_action_value = np.zeros((t_seq_len,1))
    for cnt_time in range(t_seq_len):
        state_opinion_k = state_discrete_full[cnt_time,:,:]
        state_similarity = similarity_function(state_opinion_k)

        # clusters based on cosine similarity
        state_cosine_round = np.round(state_similarity)
        state_cosine_binary = (state_cosine_round > 0) + 0
        state_full_cosine_binary[cnt_time,:,:] = state_cosine_binary

        index_seq = np.arange(Num_agent)
        cnt_cluster = 0
        
        while len(index_seq)!=0:
            state_full_cosine_binary_sample = state_cosine_binary[index_seq[0],:]
            flag_equal = (state_cosine_binary \
                        == np.tile(state_full_cosine_binary_sample,(Num_agent,1))).all(axis=1)
            agent_index = np.argwhere((flag_equal + 0) == 1)[:,0]
            clusters[cnt_time][cnt_cluster] = agent_index
            cnt_cluster = cnt_cluster+1
            index_seq = np.setdiff1d(index_seq, agent_index)
        
        # The Shannon information entropy of opinion clusters
        Num_clusters = len(clusters[cnt_time])
        Shannon_iteration = 0
        for cnt_cluster in range(Num_clusters):
            Sizes_clusters = len(clusters[cnt_time][cnt_cluster])/Num_agent
            Shannon_iteration = Shannon_iteration + Sizes_clusters * math.log(Sizes_clusters, 2)
        Shannon_value[cnt_time] = -Shannon_iteration

        # The Shannon information entropy of action clusters
        action_discrete_k = action_discrete_full[cnt_time,:,:]
        clusters_on_action = np.sum(action_discrete_k, axis=0)
        Shannon_action_iteration = 0
        for cnt_action_cluster in range(Num_task):
            Sizes_action_clusters = clusters_on_action[cnt_action_cluster]/Num_agent
            if (np.abs(Sizes_action_clusters-0)>1e-6):
                Shannon_action_iteration = Shannon_action_iteration + Sizes_action_clusters * math.log(Sizes_action_clusters, 2)
        Shannon_action_value[cnt_time] = -Shannon_action_iteration

    return Shannon_value, Shannon_action_value



def model_with_action(state_full, action_full, AdjacencyMatrix, t, coefficient_affect_bifurcation_max, lambda_self, rho):

    # action

    # lambda_self = 1
    lambda_others = 1
    reward_positive = 1
    agent_ration = 0.1
    a = 1
    # rho = 20
    mu = 0.5

    prob_select = np.zeros((Num_agent,Num_task))
    type_start = 0
    for type_num in same_type_num:
        type_end = type_start+int(type_num)
        prob_select[:,type_start:type_end] =\
             np.exp(agent_ration*state_full[:,type_start:type_end])\
                /np.sum(np.exp(agent_ration*state_full[:,type_start:type_end]),axis=1)[:,np.newaxis]
        type_start = type_start+type_end
    
    
    part0 = np.ones((Num_agent,Num_task)) - prob_select
    # partI = prob_select
    partI = np.ones((Num_agent,Num_task)) - prob_select

    action_task_similarity = np.sum(np.repeat(action_full, Num_agent, axis=0) * \
           np.tile(action_full, (Num_agent,1)), axis=1)
    task_similarity_num = np.reshape(action_task_similarity, (Num_agent,Num_agent))

    # WeightedMatrix = task_similarity_num/task_type * AdjacencyMatrix
    # WeightedMatrix = task_similarity_num/task_type * (np.ones((Num_agent,Num_agent))-np.eye(Num_agent))
    WeightedMatrix = task_similarity_num/task_type * AdjacencyMatrix + AdjacencyMatrix
    # WeightedMatrix = task_similarity_num/task_type + AdjacencyMatrix

    revised_Action = np.sum(action_full, axis=0)
    # revised_Action[revised_Action==0] = 100000000
    # payoff_action = np.tile(rho/revised_Action \
    #                         * np.sum(partI * action_full, axis=0)[np.newaxis,:],\
    #                         (Num_agent,1)) - partI * action_full + lambda_self * state_full * action_full
    
    total_network_paroff =  rho/np.sum(AdjacencyMatrix,axis=1)[:,np.newaxis] * \
        np.dot(AdjacencyMatrix,(partI * action_full))
    payoff_action = total_network_paroff +np.ones((Num_agent,Num_task))- partI \
                + lambda_self * np.sign(state_full * action_full) 
    # total_network_paroff =  rho/Num_agent * np.sum(partI, axis=0)[np.newaxis,:]
    
    # payoff_action = np.tile(total_network_paroff,\
    #           (Num_agent,1))+np.ones((Num_agent,Num_task))- partI \
    #             + lambda_self * np.sign(state_full * action_full) 

    action_full_next = np.zeros((Num_agent,Num_task))

    opinion_current_temp = np.copy(state_full)
    opinion_cluster = np.sign(opinion_current_temp)
    opinion_cluster[opinion_cluster<0] = 0
    

    

    type_start = 0
    for type_num in same_type_num:
        type_end = type_start+int(type_num)

        opinion_task_similarity = np.linalg.norm(np.repeat(opinion_cluster[:,type_start:type_end], Num_agent, axis=0) - \
            np.tile(opinion_cluster[:,type_start:type_end], (Num_agent,1)), axis=1)
        opinion_similarity_temp = np.reshape(opinion_task_similarity, (Num_agent,Num_agent))
        opinion_similarity = np.copy(opinion_similarity_temp)
        opinion_similarity[opinion_similarity_temp==0] = 1
        opinion_similarity[opinion_similarity_temp!=0] = 0
        majority_conditionI = opinion_similarity * AdjacencyMatrix
        majority_conditionII = state_full * action_full

        majority_conditionII_flag = np.max(majority_conditionII[:,type_start:type_end], axis=1)[:,np.newaxis]
        action_majority = np.dot(majority_conditionI, majority_conditionII_flag)
        action_random_index = np.argwhere(action_majority==0)[:,0]
        action_index = type_start + np.argmax(prob_select[:,type_start:type_end], axis=1)
        action_full_next[action_random_index,action_index[action_random_index]] = 1
        action_imitate_index = np.argwhere(action_majority!=0)[:,0]
        action_imitate_majority = np.dot(majority_conditionI, action_full[:,type_start:type_end])
        action_imitate = type_start + np.argmax(action_imitate_majority, axis=1)
        action_full_next[action_imitate_index,action_imitate[action_imitate_index]] = 1
        
        type_start = type_end
    # prob_select_action = np.exp(agent_ration*payoff_action)/np.sum(np.exp(agent_ration*payoff_action),axis=1)[:,np.newaxis]
    

    
    coefficient_affect = coefficient_affect_bifurcation_max-0.05+0.1*t/100
    state_self = -coefficient_confidence * state_full
    
    # affect dynamics
    state_affect_cooperation = state_full + coefficient_cooperation * np.dot(WeightedMatrix, state_full)
    state_affect_competition = coefficient_competition_self * np.dot(state_full, Tendency_Adj_Matrix)\
    + coefficient_competition_others * np.dot(np.dot(WeightedMatrix, state_full), Tendency_Adj_Matrix)
    
    # state_affect = (state_affect_cooperation + state_affect_competition +task_priority)
    state_affect = (state_affect_cooperation + state_affect_competition)
    
    
    # agent_dynamics_full = payoff_action +\
    # state_self + coefficient_affect * sat_function(state_affect,upper_bound,lower_bound)
    ttt = (coefficient_affect *  state_affect).astype(np.float128)
    agent_dynamics_full = (state_self + \
        sat_function(ttt,upper_bound,lower_bound))
    

    task_type_scale_array = np.sum((Tendency_Adj_Matrix + np.eye(Num_task)), axis=1)
    task_type_scale_matrix = np.diag(1/task_type_scale_array)   
    agent_dynamics_full_average = np.dot((Tendency_Adj_Matrix + np.eye(Num_task)), task_type_scale_matrix)
    agent_dynamics_full_with_exclusion =  agent_dynamics_full \
        - np.dot(agent_dynamics_full, agent_dynamics_full_average)
    

    agent_full_with_input = state_full + dt * agent_dynamics_full_with_exclusion
    return agent_full_with_input, action_full_next, WeightedMatrix




Num_task = 7
Num_agent = 5000

G = nx.barabasi_albert_graph(Num_agent, 2)
# G = nx.random_graphs.erdos_renyi_graph(Num_agent, 0.05)
# G = nx.watts_strogatz_graph(Num_agent, 5, 0.3)  # results in Supplementary Information


# Initiate the tendency matrix
task_type = 3
task_relation_ori = np.arange(Num_task)
task_relation = np.copy(task_relation_ori)
task_relation_splits = np.array_split(task_relation, task_type) 
same_type_num = np.zeros(task_type)
TendencyMatrix = np.zeros((task_type, len(task_relation)))
for cnt_splits in range(task_type):
    same_type_num[cnt_splits] = len(task_relation_splits[cnt_splits])
Tendency_Adj_Matrix = np.zeros((Num_task,Num_task))
Tendency_Matrix_change_for_solution = np.zeros((task_type,Num_task))
Agents_tendency_initial_all_final = np.zeros((Num_agent,Num_task))
same_type_num_start = 0
cnt_type = 0
for same_type_num_index in same_type_num:
    Tendency_Adj_Matrix[same_type_num_start:same_type_num_start+int(same_type_num_index),\
                        same_type_num_start:same_type_num_start+int(same_type_num_index)]\
    = np.ones((int(same_type_num_index), int(same_type_num_index)))-np.eye(int(same_type_num_index))
    
    Tendency_Matrix_change_for_solution[cnt_type,\
        same_type_num_start:same_type_num_start+int(same_type_num_index)]\
             = np.ones((1, int(same_type_num_index)))
    
    same_type_num_start = same_type_num_start + int(same_type_num_index)
    cnt_type = cnt_type + 1


# initial states setting

Agents_tendency_initial = np.zeros((Num_agent,Num_task))
mu, sigma = 0, 0.0005      
Agents_tendency_initial = np.random.normal(mu, sigma, (Num_agent,Num_task))


Tendencyeigenvalue, Tendencyeigenvector = np.linalg.eig(Tendency_Adj_Matrix)

degrees_all = dict(G.degree())
degrees_lst = list(degrees_all.values())
degrees_array = np.array(degrees_lst)
degrees_sorted = np.array(sorted(list(degrees_array))) # ascending sort
degrees_array_normalized = degrees_sorted/np.max(degrees_array)
degrees_sorted_index = np.argsort(degrees_array)
AdjacencyMatrix = nx.to_numpy_array(G)
eigenvalue_Adj, eigenvector_Adj = np.linalg.eig(AdjacencyMatrix)




# Dynamics model
## Parameter settings

T = 300
t = 0
dt = 0.5  
coefficient_confidence = 1
coefficient_cooperation = 1  # when competition, coefficient_cooperation = -1 and coefficient_competition_others = 1
coefficient_competition_others = -1  
coefficient_competition_self = -1


# for coefficient_cooperation > coefficient_competition_others
coefficient_affect_bifurcation_max =\
    coefficient_confidence/(-coefficient_competition_self\
                            +np.max(eigenvalue_Adj.real)*(coefficient_cooperation-coefficient_competition_others))


# for coefficient_cooperation < coefficient_competition_others
coefficient_affect_bifurcation_min =\
    coefficient_confidence/(-coefficient_competition_self\
                            +np.min(eigenvalue_Adj.real)*(coefficient_cooperation-coefficient_competition_others))


upper_bound = 1
lower_bound = -1

t_seq = np.linspace(0,T,int(T/dt+1))

cases_type = 3

state_discrete_full = np.zeros((len(t_seq)+1,Num_agent,Num_task))
action_discrete_full = np.zeros((len(t_seq)+1,Num_agent,Num_task))
WeightedMatrix_full = np.zeros((len(t_seq),Num_agent,Num_agent))

state_discrete_full[0,:,:]= Agents_tendency_initial
action_discrete_full[0,:,:]= np.zeros((Num_agent,Num_task))


Shannon_value = np.zeros((1,len(t_seq)))
Shannon_action_value = np.zeros((1,len(t_seq)))

state_discrete_full_all_cases =[]
state_discrete_full_degree_all_cases=[]
state_discrete_full_random_all_cases=[]

degrees_reversesorted_index = np.argsort(-degrees_array)
AdjacencyMatrix_changed = np.copy(AdjacencyMatrix)



for cnt_time in range(len(t_seq)):
    state_instant, action_instant, WeightedMatrix= model_with_action(\
        state_discrete_full[cnt_time,:,:], \
            action_discrete_full[cnt_time,:,:], AdjacencyMatrix_changed,\
                t_seq[cnt_time], coefficient_affect_bifurcation_max, lambda_self=15, rho=2)
                
    state_instant = np.nan_to_num(state_instant, nan=0)
    

    zeros_row_index = np.where(~state_instant.any(axis=1))[0]
    # state_instant[zeros_row_index,:] = state_instant[0,:]
    state_instant[zeros_row_index,:] = state_discrete_full[cnt_time,zeros_row_index,:]

    state_discrete_full[cnt_time+1,:,:]= state_instant
    action_discrete_full[cnt_time+1,:,:]= action_instant
    WeightedMatrix_full[cnt_time,:,:]= WeightedMatrix
    

    # The Shannon information entropy of opinion clusters
    opinion_current_temp = np.copy(state_discrete_full[cnt_time,:,:])
    opinion_cluster = np.sign(opinion_current_temp)
    opinion_cluster[opinion_cluster<0] = 0
    uniques_cluster = np.unique(opinion_cluster, axis=0)
    Num_clusters = len(uniques_cluster)
    Shannon_iteration = 0
    for cnt_cluster in range(Num_clusters):
        Sizes_clusters = np.sum((opinion_cluster==uniques_cluster[cnt_cluster,:]).all(axis=1) + 0)/Num_agent
        Shannon_iteration = Shannon_iteration + Sizes_clusters * math.log(Sizes_clusters, 2)
    Shannon_value[cnt_time] = -Shannon_iteration

    # The Shannon information entropy of action clusters
    action_current_temp = np.copy(action_discrete_full[cnt_time,:,:])
    uniques_action_cluster = np.unique(action_current_temp, axis=0)
    Num_action_clusters = len(uniques_action_cluster)
    Shannon_action_iteration = 0
    for cnt_cluster in range(Num_action_clusters):
        Sizes_action_clusters = np.sum((action_current_temp==uniques_action_cluster[cnt_cluster,:]).all(axis=1) + 0)/Num_agent
        Shannon_action_iteration = Shannon_action_iteration + Sizes_action_clusters * math.log(Sizes_action_clusters, 2)
    Shannon_action_value[cnt_time] = -Shannon_action_iteration




np.save('../HDD/ResultI_Shannon_action_value_WS.npy',Shannon_action_value)
np.save('../HDD/ResultI_Shannon_value_WS.npy',Shannon_value)
np.save('../HDD/ResultI_opinion_state_WS.npy',state_discrete_full)
np.save('../HDD/ResultI_action_state_WS.npy',action_discrete_full)
np.save('../HDD/ResultI_weight_state_WS.npy',WeightedMatrix_full)
# np.save('../HDD/ResultI_Shannon_action_value_SF_c.npy',Shannon_action_value)
# np.save('../HDD/ResultI_Shannon_value_SF_c.npy',Shannon_value)
# np.save('../HDD/ResultI_opinion_state_SF_c.npy',state_discrete_full)
# np.save('../HDD/ResultI_action_state_SF_c.npy',action_discrete_full)
# np.save('../HDD/ResultI_weight_state_SF_c.npy',WeightedMatrix_full)
