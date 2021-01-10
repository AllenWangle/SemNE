from __future__ import division
import numpy as np
import networkx as nx
import node2vec
import Self_organization as S_O
import JointlyME as J_ME
import argparse
import os
from sklearn.decomposition import PCA
from tqdm import tqdm


def read_embeddings(emb_path):
    """
    Import the necessary embeddings
    :param emb_path: the file path of embeddings
    :return: the dict of embeddings
    """
    vec = {}
    with open(emb_path, "r") as emb:
        for line in emb.readlines():
            vec[int(line.strip().split(" ")[0])] = [float(i) for i in line.strip().split(" ")[1:]]
    return vec


def read_index(ind_path):
    """
    Read the semantics of node from the file

    :param ind_path: the file path of node's semantic
    :return: the dict of semantic
    """
    index_dict = {}
    with open(ind_path, "r") as index:
        for line in index.readlines():
            index_dict[int(line.strip().split("\t")[0])] = str(line.strip().split("\t")[1])
    return index_dict


def read_graph():
    """
    Reads the input network in networkx.

    :return: graph G
    """
    if args.weighted:
        G = nx.read_edgelist(args.input, nodetype=int, data=(('weight',float),), create_using=nx.Graph())
    else:
        G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.Graph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

    if not args.directed:
        G = G.to_undirected()

    return G


def sentenceemb_with_wordemb(wordvec, sentvec):
    """
    Combine the word-level embeddings and fact-level embeddings to obtain the final fact vectors

    :param wordvec: word-level embeddings
    :param sentvec: fact-level embeddings
    :return: final fact vectors
    """
    con_ID = []
    con_vec = []
    
    for word_i in tqdm(wordvec.keys()):
        wordvec[word_i].extend(sent_j for sent_j in sentvec[word_i])
        
    for k in wordvec.keys():
        con_ID.append([k])
        con_vec.append(wordvec[k])
    pca = PCA(n_components=args.dimensions)
    pca.fit(con_vec)
    emb_pca = pca.transform(con_vec)
    for j in tqdm(range(len(emb_pca))):
        con_ID[j].extend(float(k) for k in emb_pca[j])
    np.savetxt(args.output, np.array(con_ID), fmt="%s", newline = "\n")
    
    return


def parse_args():
    """
    Parses the SemNE arguments.
    """
    parser = argparse.ArgumentParser(description="Run SemNE with python 3.6.")

    parser.add_argument('--input', nargs='?', default='graph/ECCI.txt',
                        help='Input graph path. Default is ECCI graph')
    
    parser.add_argument('--semantic', nargs='?', default='graph/ECCI.index.txt',
                        help='Input the semantic path of nodes. Default is semantic in ECCI graph')

    parser.add_argument('--output', nargs='?', default='emb/ECCI_embedding.txt',
                        help='Embeddings path to save. Default is embeddings of ECCI graph')
                        
    parser.add_argument('--mode', nargs='?', default='skipgram', 
                        help='The mode of jointly learning word-level and fact-level embeddings')

    parser.add_argument('--dimensions', type=int, default=100,
                        help='Number of dimensions. Default is 100.')

    parser.add_argument('--walk_length', type=int, default=80,
                        help='Length of sampling per source. Default is 80.')

    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of sampling per source. Default is 10.')

    parser.add_argument('--window_size', type=int, default=5,
                        help='Context size for optimization. Default is 5.')

    parser.add_argument('--iter', default=10, type=int,
                        help='Number of epochs in SGD')

    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')

    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default=1,
                        help='Inout hyperparameter. Default is 1.')

    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')
    
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    
    parser.set_defaults(weighted=False)

    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    
    parser.set_defaults(directed=False)

    return parser.parse_args()


def main(args):
    """
    Pipeline for representational learning for all nodes in a graph.
    """
    print("==========Read Network and Semantic!===========")
    nx_G = read_graph()
    print("The number of nodes in network is {}".format(len(nx_G.nodes())))
    index = read_index(args.semantic)
    
    print("==========Sampling for Fact!===========")
    G = node2vec.Graph(nx_G, args.directed, args.p, args.q)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(args.epochs, args.walk_length)
   
    print("==========Semantic Alignment!===========")
    for walks_length in tqdm(range(len(walks))):
        for walk_len in range(len(walks[walks_length])):
            walks[walks_length][walk_len] = index[walks[walks_length][walk_len]]
            
    print("==========Jointly Learning Fact Embeddings in Word-Level and Fact-Level!===========")
    word_model = J_ME.learn_embeddings(args.mode, walks, args.dimensions, args.window_size, args.workers, args.iter, index)
    
    print("==========Combine Two Embeddings to Obtain the Final Fact Embeddings!===========")
    fact_vec_from_word = S_O.self_organization(word_model, nx_G, index, args.dimensions)
    fact_vec_from_node = read_embeddings("temp/fact_embeddings.txt")
    sentenceemb_with_wordemb(fact_vec_from_word, fact_vec_from_node)
    
    
if __name__ == "__main__":

    tmp_path = "temp/"
    if not os.path.exists(tmp_path):
        os.mkdir(tmp_path)
    args = parse_args()
    main(args)
