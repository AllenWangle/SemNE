# Requirements

The code base is implemented in Python 3.6.10. package versions used for development are just below:

| Package      | Version |
| ------------ | ------- |
| networkx     | 2.4.0   |
| numpy        | 1.18.4  |
| tqdm         | 4.46.1  |
| scikit-learn | 0.19.1  |
| gensim       | 3.6.0   |
| argparse     | 1.1.0   |
| prettytable  | 0.7.2   |
| fasttext     | 0.9.2   |

# Datasets

The code takes the **edge list** of the network in a `.txt` file. Each row indicates an edge between two nodes separated by a tab. Nodes can be indexed starting with any ID. The network also can be a weighted network. Also, there should be a txt file that contains the semantic of each node. Three network used in our paper are provide in `graph/`.

# Download pretrained embedding files

Download pretrained embedding files from [Google Drive](https://drive.google.com/drive/folders/12AWQFPkCmF2RMflOOvYW_dU0DXVGNcmt?usp=sharing) and save `CN_embedding.txt`, `ECCI_embedding.txt` and `PB_embedding.txt` at `test/pre-trained` folder.

# Training the model

Training the model is handled by the `SemNE.py` script which provides the following command line arguments.

## input and output options

| Command    | Type | Description             | Default                              |
| ---------- | ---- | ----------------------- | ------------------------------------ |
| --input    | STR  | Input graph path.       | Default is "graph/ECCI.txt"          |
| --semantic | STR  | Input semantic path.    | Default is "graph/ECCI.index.txt"    |
| --output   | STR  | Output embeddings path. | Default is "emb/ECCI_embeddings.emb" |

## Random walk options in node2vec

| Command       | Type  | Description                    | Default        |
| ------------- | ----- | ------------------------------ | -------------- |
| --walk_length | INT   | Number of nodes in each epoch. | Default is 80  |
| --epochs      | INT   | Number of epochs per source.   | Default is 10  |
| --window_size | INT   | Context size for optimization. | Default is 5   |
| --p           | FLOAT | Return parameter.              | Default is 1.0 |
| --q           | FLOAT | Inout parameter.               | Default is 1.0 |

## Language model options

| Command      | Type | Description                    | Default               |
| ------------ | ---- | ------------------------------ | --------------------- |
| --mode       | STR  | Model for learning embeddings. | Default is "skipgram" |
| --dimensions | INT  | Number of dimensions.          | Default is 100        |
| --iter       | INT  | Number of epochs in SGD.       | Default is 10         |
| --workers    | INT  | Number of parallel threads.    | Default is 8          |

## Other options

| Command      | Type | Description          |
| ------------ | ---- | -------------------- |
| --directed   | NULL | Graph is directed.   |
| --undirected | NULL | Graph is undirected. |

`parse.set_defaults(directed=False)`  indicates the graph is undirected
`parse.set_defaults(directed=True)`  indicates the graph is directed

| Command      | Type | Description         |
| ------------ | ---- | ------------------- |
| --weighted   | NULL | Graph is weighted   |
| --unweighted | NULL | Graph is unweighted |

`parse.set_defaults(weighted=False)`  indicates the graph is unweighted
`parse.set_defaults(weighted=True)`  indicates the graph is weighted

# Testing the model

Testing the model is handled by the `test/test.py` script which provides the following command line arguments. 
The embeddings of three network are provided in `test/pre-trained/`. It can be used to reproduce the results in the paper.

| Command        | Type | Description                 | Default                                              |
| -------------- | ---- | --------------------------- | ---------------------------------------------------- |
| --train        | STR  | Input training graph path.  | Default is "../graph/ECCI.txt"                       |
| --test         | STR  | Input testing graph path.   | Default is "../graph/ECCI.test.txt"                  |
| --embedding    | STR  | Input embedding file path.  | Default is "pre-trained/ECCI_embeddings.txt"         |
| --dimensions   | INT  | Number of dimensions.       | Default is 100                                       |
| --negative_num | INT  | Ratio of neg-links for ACC. | Default is 1 (equal to the number of positive links) |

# Examples

The following commands learn a SemNE embeddings.

standard hyperparameter setting for training the embeddings on the default dataset.

```
python SemNE.py
```

using `cbow` model to train the embeddings on the default dataset.

```
python SemNE.py --mode cbow [skipgram(default), fasttext]
```

using a custom dimension for the embedding.

```
python SemNE.py --dimensions 64
```

using DFS (or BFS) to sample fact.

```
python SemNE.py --p 4 (or 0.5) --q 0.5 (or 4)
```

test the embedding.

```
cd test
python test.py
```

# Acknowledge

Our code is based on the [aditya-grover/node2vec](https://github.com/aditya-grover/node2vec) algorithm. We appreciate their contribution on the technique development and released codes.
