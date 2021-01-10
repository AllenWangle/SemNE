# coding=utf-8
from gensim.models import Word2Vec
import numpy as np
import fasttext as ft
from tqdm import tqdm

sen_file_path = "temp/sentence.txt"
word_model_path = "temp/WordModel"
fact_model_path = "temp/FactModel"
fact_embedding_path = "temp/fact_embeddings.txt"


def learn_embeddings(mode, sentences, dimensions, window_size, workers, iter, ind):
    """
    Jointly Learn word-level and fact-level embeddings by optimizing the Language Model.

    :param ind: the index for each fact
    :param mode: the chosen language model
    :param sentences: the sequence sampled by node2vec
    :param dimensions: the number of dimensions
    :param window_size: the size of window in language model
    :param workers: the number of parnell threads
    :param iter: the number of epochs in SGD.
    :return: the word-level (model_W) and the fact-level (model_S) model
    """
    np.savetxt(sen_file_path, np.array(sentences), fmt="%s", newline="\n")

    if mode == "skipgram":
        print("                    +++Learning Word-level Embeddings++++")
        wm = ft.train_unsupervised(sen_file_path, model=mode, dim=dimensions)
        wm.save_model(word_model_path + "_" + mode + ".bin")

        print("                    +++Learning Fact-level Embeddings++++")
        sent = list(list(map(str, s)) for s in sentences)
        fm = Word2Vec(sent, size=dimensions, window=window_size, min_count=0, sg=1, workers=workers, iter=iter)
        fm.wv.save_word2vec_format(fact_embedding_path, binary=False)
        fm.save(fact_model_path + "_" + mode + ".bin")

        # Turn fact into corresponding nodes
        semantic_to_fact(ind, dimensions)

        return wm

    if mode == "cbow":
        print("                    +++Learning Word-level Embeddings++++")
        wm = ft.train_unsupervised(sen_file_path, model=mode, dim=dimensions)
        wm.save_model(word_model_path + "_" + mode + ".bin")

        print("                    +++Learning Fact-level Embeddings++++")
        sent = list(list(map(str, s)) for s in sentences)
        fm = Word2Vec(sent, size=dimensions, window=window_size, min_count=0, sg=0, workers=workers, iter=iter)
        fm.wv.save_word2vec_format(fact_embedding_path, binary=False)
        fm.save(fact_model_path + "_" + mode + ".bin")

        # Turn fact into corresponding nodes
        semantic_to_fact(ind, dimensions)

        return wm


def semantic_to_fact(ind, dim):
    """
    Align the fact-level(string) embeddings into the fact(node,int) embeddings.

    :param ind: a dict of semantic
    :param dim: the number of dimensions
    :return: the fact embeddings in "node_ID + embeddings" format
    """

    ind_new = {v: k for k, v in ind.items()}

    sem_to_node = []
    fact_emb = []

    with open(fact_embedding_path, "r") as emb_file:
        content = [i.strip().split() for i in emb_file.readlines()]
        embedding = content[1:]
        vectors = [i[len(i) - dim:] for i in embedding]
        words = [i[:len(i) - dim] for i in embedding]

    for num, i in enumerate(vectors):
        sem_to_node.append('{0}\t{1}\n'.format(' '.join(words[num]), ' '.join(i)))

    for p_len in range(len(sem_to_node)):
        fact_emb.append(sem_to_node[p_len].strip().split("\t"))

    for i in tqdm(range(len(fact_emb))):
        fact_emb[i][0] = ind_new[fact_emb[i][0]]

    np.savetxt(fact_embedding_path, np.asarray(fact_emb), fmt="%s", newline="\n")

    return
