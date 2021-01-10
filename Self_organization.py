import numpy as np
import gensim
from tqdm import tqdm


def sigmoid(inx):
    """
    sigmoid function
    :param inx: x parameter for sigmoid
    :return: a sigmoid value
    """
    if inx >= 0:
        return 1.0 / (1 + np.exp(-inx))
    else:
        return np.exp(inx) / (1 + np.exp(inx))


def word_vec_first_con(sourceveclist, targetveclist, dim):
    """
    Compute the first-order proximity between two adjacency words in a semantic fact.

    :param sourceveclist: the first word embeddings
    :param targetveclist: the second word embeddings
    :param dim: the number of dimensions
    :return: the merge embeddings for two word embeddings
    """
    vec_error = [0] * dim
    sum_dim = 0
    for ii in range(len(sourceveclist)):
        sum_dim = sum_dim + (float(sourceveclist[ii]) * float(targetveclist[ii]))
    g = sigmoid(-sum_dim)
    for jj in range(len(targetveclist)):
        vec_error[jj] = vec_error[jj] + g * float(targetveclist[jj])
    for kk in range(len(sourceveclist)):
        sourceveclist[kk] = float(sourceveclist[kk]) + vec_error[kk]

    return sourceveclist


def self_organization(word_model, g_train, node_sem, dim):
    """
    main function of self-organization.

    :param word_model: the fasttext model of word embeddings
    :param g_train: the graph used to obtain the fact embeddings
    :param node_sem: the semantic of nodes (dictionary)
    :param dim: the number of dimensions
    :return: the dict of fact embeddings obtained by the word embeddings
    """

    node_emb_from_word = {}
    for node in tqdm(g_train.nodes()):
        W = []
        for word in node_sem[node].split(" "):
            W.append(word)
        W.append("</s>")
        source_vector = word_model[W[0]].tolist()
        if len(W) == 1:
            target_vector = source_vector
            target_vector = word_vec_first_con(source_vector, target_vector, dim)
        else:
            for w_len in range(1, len(W)):
                try:
                    target_vector = word_model[W[w_len]].tolist()
                    target_vector = word_vec_first_con(source_vector, target_vector, dim)
                except KeyError:
                    continue
        node_emb_from_word[node] = target_vector

    return node_emb_from_word
