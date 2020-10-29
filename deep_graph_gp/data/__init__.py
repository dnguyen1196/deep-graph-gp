import os
import numpy as np
import dgl
from scipy.io import loadmat
import torch

dataset_folder = os.path.dirname(os.path.realpath(__file__)) + "/datasets"

def cerebellum():
    """
    Load the cerebellum data set into a graph


    Return:
        a DGLGraph
    """
    data_folder = os.path.join(dataset_folder, "cerebellum")
    fmri_adj = loadmat(f"{data_folder}/A_cerebellum.mat")["A"].tocoo()
    fmri_signal = loadmat(f"{data_folder}/signal_set_cerebellum.mat")["F2"]

    (n, m) = fmri_adj.shape
    assert(n == m)
    (nprime, d) = fmri_signal.shape
    assert(n == nprime)
    g = dgl.DGLGraph()
    g.add_nodes(n, {"h" : torch.FloatTensor(fmri_signal)})
    g.add_edges(fmri_adj.row, fmri_adj.col)
    return g

def weather():
    """

    Return:
        a DGLGraph
    """
    data_folder = os.path.join(dataset_folder, "weather")
    cities_adj = loadmat(f"{data_folder}/city45data.mat")["A45"]
    temperature_data = np.transpose(loadmat(f"{data_folder}/smhi_temp_17.mat")["temp_17"])

    (n, m) = cities_adj.shape
    assert(n == m)
    (nprime, d) = temperature_data.shape
    assert(n == nprime)
    g = dgl.DGLGraph()
    g.add_nodes(n, {"h" : torch.FloatTensor(temperature_data)})
    for i in range(n):
        for j in range(n):
            if (i != j):
                g.add_edge(i, j, {"h": torch.FloatTensor([cities_adj[i,j]])})

    return g

def etex():
    """

    Return:
        a DGLGraph
    """
    data_folder = os.path.join(dataset_folder, "etex")
    location_adj = loadmat(f"{data_folder}/A_etex.mat")["A"]
    etex_data_1 = loadmat(f"{data_folder}/etex_1.mat")["pmch"]
    etex_data_2 = loadmat(f"{data_folder}/etex_2.mat")["pmcp"]

    (n, m) = location_adj.shape
    assert(n == m)

    etex_data = np.hstack((etex_data_1, etex_data_2))
    (nprime, d) = etex_data.shape
    assert(n == nprime)
    
    g = dgl.DGLGraph()
    g.add_nodes(n, {"h" : torch.FloatTensor(etex_data)})

    for i in range(n):
        for j in range(n):
            if (i != j):
                g.add_edge(i, j, {"h": torch.FloatTensor([location_adj[i,j]])})

    return g


def cora(directed=True):
    """

    Return:
        a DGLGraph

    """
    data_folder = os.path.join(dataset_folder, "cora")

    pub_ids = dict()

    with open(os.path.join(data_folder, "cora.content"), "r") as f:
        for line in f:
            s = line.rstrip().split()
            # Get node data
            node_id = int(s[0])
            features = [float(s[i] )for i in range(1, len(s)-1)]
            label = s[-1]
            pub_ids[node_id] = (features, label)

    node_ids_to_pub_ids = dict([(i, id) for i, id in enumerate(pub_ids.keys())])
    pub_ids_to_node_ids = dict([(value, key) for (key, value) in node_ids_to_pub_ids.items()])

    all_features = torch.FloatTensor([pub_ids[node_ids_to_pub_ids[i]][0] for i in range(len(node_ids_to_pub_ids))])
    all_labels   = [pub_ids[node_ids_to_pub_ids[i]][1] for i in range(len(node_ids_to_pub_ids))]
    all_labels_set = set()
    for item in all_labels:
        all_labels_set.add(item)

    labels_to_class_id = {
        'Neural_Networks':0,
        'Genetic_Algorithms':1,
        'Case_Based':2,
        'Reinforcement_Learning':3,
        'Probabilistic_Methods':4,
        'Rule_Learning':5,
        'Theory':6
    }
    all_labels = torch.LongTensor([labels_to_class_id[label] for label in all_labels])

    g = dgl.DGLGraph()
    g.add_nodes(all_features.shape[0], {"h" : all_features, "y" : all_labels})

    with open(os.path.join(data_folder, "cora.cites"), "r") as f:
        for line in f:
            s = line.rstrip().split()
            cited, citing = int(s[0]), int(s[1])
            cited = pub_ids_to_node_ids[cited]
            citing = pub_ids_to_node_ids[citing]
            g.add_edge(cited, citing)
            if not directed:
                g.add_edge(citing, cited)

    return g


def citeseer(directed=True):
    data_folder = os.path.join(dataset_folder, "citeseer")

    g = dgl.DGLGraph()
    node_labels = []
    with open(os.path.join(data_folder, "citeseer.node_labels"), "r") as f:
        for line in f:
            s = line.rstrip().split(",")
            node_id = int(s[0])-1
            class_label = int(s[1])-1
            node_labels.append(class_label)

    g.add_nodes(len(node_labels), {"y": torch.LongTensor(node_labels)})

    with open(os.path.join(data_folder, "citeseer.edges"), "r") as f:
        for line in f:
            s = line.rstrip().split(",")
            citing = int(s[0])-1
            cited = int(s[1])-1
            g.add_edge(cited, citing)
            if not directed:
                g.add_edge(citing, cited)

    return g


def pubmed():
    data_folder = os.path.join(dataset_folder, "pubmed_diabetes/data")

    g = dgl.DGLGraph()

    all_keys = dict()
    pub_ids_to_node_ids = dict()

    with open(os.path.join(data_folder, "Pubmed-Diabetes.NODE.paper.tab"), "r") as f:
        f.readline() # Ignore the first line
        header = f.readline()
        s = header.rstrip().split()[1:] # Label is not part of features
        keys = [s[i].split(":")[1] for i in range(len(s)-1)]

        for (i, key) in enumerate(keys):
            all_keys[key] = i
        
        all_features = []
        all_labels   = []

        node_id = 0
        for line in f:
            node_features = [np.nan for i in range(len(all_keys))]
            s = line.rstrip().split()

            # Get the publication id and node_label
            pub_id = int(s[0])
            _, label = s[1].split("=")
            node_label = int(label)-1

            pub_ids_to_node_ids[pub_id] = node_label
            # Extract the node features
            for feat in s[2:]:
                key, val = feat.split("=")

                if key in all_keys:
                    val = float(val)
                    key_index = all_keys[key]
                    node_features[key_index] = val

            all_features.append(node_features)
            all_labels.append(node_label)

            node_id += 1
        
        g.add_nodes(node_id, {"h" : torch.FloatTensor(all_features), "y" : torch.LongTensor(all_labels)})
    
    with open(os.path.join(data_folder, "Pubmed-Diabetes.DIRECTED.cites.tab"), "r") as f:
        f.readline() # Skip first two lines
        f.readline()

        for line in f:
            s = line.rstrip().split()
            citing_pub_id = int(s[1][6:])
            cited_pub_id  = int(s[3][6:])
            citing_node_id = pub_ids_to_node_ids[citing_pub_id]
            cited_node_id = pub_ids_to_node_ids[cited_pub_id]
            g.add_edge(cited_node_id, citing_node_id)

    return g