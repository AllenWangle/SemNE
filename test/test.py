from __future__ import division
import numpy as np
import networkx as nx
import os
from tqdm import tqdm
import argparse
import utils
import prettytable as pt
from sklearn import metrics
import random
from sklearn.metrics import accuracy_score


class AUC_MR(object):

    # AUC calculation function
    @staticmethod
    def euclidean_distance(start_node, node_vector_dict):
        distance_list = []
        start_node_vector = np.asarray(node_vector_dict[start_node])
        for i in node_vector_dict.keys():
            if i != start_node:
                ith_node_vector = np.asarray(node_vector_dict[i])
                distance_value = np.linalg.norm(start_node_vector - ith_node_vector)
                distance_list.append([i, float(distance_value)])

        distance_list_sort = sorted(distance_list, key=lambda data: data[1])
        return distance_list_sort

    @staticmethod
    def softmax(x):
        x_max = np.max(x)
        x = x - x_max
        numerator = np.exp(x)
        denominator = 1.0 / np.sum(numerator)
        x = numerator.dot(denominator)
        return x

    @staticmethod
    def compute(test, start_node, vector_dict):
        AUC_prob = []
        euc_dist = AUC_MR.euclidean_distance(start_node, vector_dict)
        E_D_node = [int(float(euc_n[0])) for euc_n in euc_dist]
        E_D_prob = [float(euc_v[1]) for euc_v in euc_dist]
        prob = list(AUC_MR.softmax(E_D_prob))
        for i in range(0, len(E_D_node)):
            if E_D_node[i] in list(test.neighbors(start_node)):
                AUC_prob.append([prob[i], 1])
            else:
                AUC_prob.append([prob[i], 0])
        probability = [1/p[0] for p in AUC_prob]  
        label = [int(l[1]) for l in AUC_prob]
        probability_np = np.array(probability)
        label_np = np.array(label)
        fpr, tpr, thresholds = metrics.roc_curve(label_np, probability_np, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        return auc

    # MR calculation function

    @staticmethod
    def result_rank(test_graph, vector_list):
        result_dict = {}
        for node in tqdm(test_graph.nodes()):
            r_list = AUC_MR.euclidean_distance(node, vector_list)
            result_dict[node] = [int(float(r_node[0])) for r_node in r_list]

        return result_dict

    @staticmethod
    def mean_rank(test_graph, start_node, euc_dict):
        m_r = [] 
        for i in range(len(euc_dict[start_node])):
            if (start_node, euc_dict[start_node][i]) in test_graph.edges():
                m_r.append(i)
            else:
                continue 
                
        mean_rank = sum(m_r) / len(m_r)

        return mean_rank


class ACC(object):

    @staticmethod
    def generate_neg_link(gtest, number, start_node):
        neg_node = []
        node_index = list(gtest.nodes())
        while len(neg_node) < number:
            sample = random.randint(0, len(node_index)-1)
            if node_index[sample] != start_node and node_index[sample] not in gtest.neighbors(start_node):
                neg_node.append(node_index[sample])
        return neg_node[0]  

    @staticmethod
    def eval_link_prediction(test_edges, emd, EmbMap):

        score_res = []
        for i in range(len(test_edges)):
            score_res.append(np.dot(emd[EmbMap[str(float(test_edges[i][0]))]],
                                    emd[EmbMap[str(float(test_edges[i][1]))]]))
        test_label = np.array(score_res)
        median = np.median(test_label)
        index_pos = test_label >= median
        index_neg = test_label < median
        test_label[index_pos] = 1
        test_label[index_neg] = 0
        true_label = np.zeros(test_label.shape)
        true_label[0: len(true_label) // 2] = 1
        accuracy = accuracy_score(true_label, test_label)
        return accuracy


def parse_args():
    """
    Parses the SemNE arguments.
    """
    parser = argparse.ArgumentParser(description="Run SemNE.")
    
    parser.add_argument('--train', nargs='?', default='../graph/ECCI.txt',
                        help='Input train graph path')

    parser.add_argument('--test', nargs='?', default='../graph/ECCI.test.txt',
                        help='Input test graph path')
    
    parser.add_argument('--embedding', nargs='?', default='pre-trained/ECCI_embedding.txt',
                        help='Input the embedding path')
    
    parser.add_argument('--dimensions', type=int, default=100,
                        help='Number of dimensions. Default is 100.')
    
    parser.add_argument('--negative_num', type=int, default=1,
                        help='Ratio of negative links for ACC. Default is 1.')

    return parser.parse_args()


def read_node_vectors(emb_path, gtest):
    node_vector = {}
    with open(emb_path, "r") as file:
        for line in file.readlines():
            if int(float(line.strip().split(" ")[0])) in gtest.nodes():
                node_vector[int(float(line.strip().split(" ")[0]))] = [float(i) for i in line.strip().split(" ")[1:]]
    return node_vector
                
                
def main(args):
    G_train = nx.read_weighted_edgelist(args.train, nodetype=int, create_using=nx.Graph())
    G_test = nx.read_weighted_edgelist(args.test, nodetype=int, create_using=nx.Graph())
    vector = read_node_vectors(args.embedding, G_test)       
    
    print("=====Compute AUC====")
    auc = []
    for node in tqdm(list(G_test.nodes())):
        try:
            auc.append(AUC_MR.compute(G_test, node, vector))
        except ValueError:
            continue
    auc_mean = float(sum(auc) / len(auc))
    
    print("=====Compute MR====")
    sequence_order = AUC_MR.result_rank(G_test, vector)
    mr = []
    for node in tqdm(G_test.nodes()):
        try:
            mr.append(AUC_MR.mean_rank(G_test, node, sequence_order))
        except ValueError:
            continue
    Mean_Rank = sum(mr) / len(mr)
    
    print("=====Compute ACC====")
    n_node = len(G_train.nodes())
    
    neg_sample_link = []
    for edge in tqdm(G_test.edges()):
        neg_sample_link.append([edge[0], ACC.generate_neg_link(G_test, args.negative_num, edge[0])])
    np.savetxt("temp/negtive_link.txt", np.asarray(neg_sample_link), fmt="%s", newline="\n", delimiter="\t")
    
    test_edge = utils.read_edges_from_file(args.test)
    test_edge_neg = utils.read_edges_from_file("temp/negtive_link.txt")
    test_edge.extend(test_edge_neg)
    EMB, EMBMAP = utils.read_embeddings(args.embedding, n_node, args.dimensions)
    acc = ACC.eval_link_prediction(test_edge, EMB, EMBMAP)
    
    print("=====Show Results====")
    dataset_name = args.train.split("/")[-1].split(".")[0]
    tb = pt.PrettyTable()
    tb.field_names = ["dataset", "AUC", "MR", "ACC"]
    tb.add_row([dataset_name, auc_mean, Mean_Rank, acc])
    print(tb)

    
if __name__ == "__main__":
    tmp_path = "temp/"
    if not os.path.exists(tmp_path):
        os.mkdir(tmp_path)
    args = parse_args()
    main(args)
