import pandas as pd
from graph_tool import all as gt
from graph_tool import GraphView
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import linkage, dendrogram
from graph_tool.topology import label_components
from . import graph_fit


def load_network(fname, remove_sources=False, remove_sinks=True, remove_selfloops=True, add_selfloops_to_sources=True, header=None):
    G = gt.Graph()
    infile = pd.read_csv(fname, header=header, dtype="str")
    vertex_dict = dict()
    vertex_names = G.new_vertex_property('string')
    vertex_source = G.new_vertex_property('bool')
    vertex_sink = G.new_vertex_property('bool')
    for tf in set(list(infile[0]) + list(infile[1])):
        v = G.add_vertex()
        vertex_dict[tf] = v
        vertex_names[v] = tf
        
    for i in infile.index:
        if (not remove_selfloops or infile.loc[i,0] != infile.loc[i,1]):
            v1 = vertex_dict[infile.loc[i,0]]
            v2 = vertex_dict[infile.loc[i,1]]
            if v2 not in v1.out_neighbors(): G.add_edge(v1, v2)

    G.vertex_properties["name"]=vertex_names
    
    if (remove_sources or remove_sinks):
        G = prune_network(G, remove_sources=remove_sources, remove_sinks=remove_sinks)
        vertex_dict = dict()
        for v in G.vertices():
            vertex_dict[vertex_names[v]] = v
            
    
    for v in G.vertices():
        if v.in_degree() == 0:
            if add_selfloops_to_sources: G.add_edge(v, v)
            vertex_source[v] = True
        else: vertex_source[v] = False
        if v.out_degree() == 0: vertex_sink[v] = True
        else: vertex_sink[v] = False
        
    G.vertex_properties['sink']=vertex_sink
    G.vertex_properties['source']=vertex_source
    
    return G, vertex_dict


def prune_network(G, remove_sources=True, remove_sinks=False):
    oneStripped = True
    while (oneStripped):

        vfilt = G.new_vertex_property('bool');
        oneStripped = False
        for v in G.vertices():
            if (remove_sources and v.in_degree() == 0) or (remove_sinks and v.out_degree() == 0):
                vfilt[v] = False
                oneStripped = True
            else: vfilt[v] = True
        
        G = GraphView(G, vfilt)
    return G
    
    
# Reads dataframe. File must have rows=genes, cols=samples. Returned dataframe is transposed.
# If norm is one of "gmm" (data are normalized via 2-component gaussian mixture model),
# "minmax" (data are linearly normalized from 0(min) to 1(max)) or no normalization is done
def load_data(filename, nodes, log=False, log1p=False, sample_order = None, delimiter=",", norm="gmm", index_col=0):
    data = pd.read_csv(filename,index_col=index_col, delimiter=delimiter, na_values=['null','NULL'])
    if index_col > 0: data = data[data.columns[index_col:]]
    data.index = [str(i) for i in data.index]
    missing_nodes = [i for i in nodes if not i in data.index]
    if len(missing_nodes) > 0: raise ValueError("Missing nodes: %s"%repr(missing_nodes))
    
    data = data.loc[nodes]

    if log1p: data = np.log(data+1)
    elif log: data = np.log(data)
    
    df = data.transpose() # Now: rows=samples, columns=genes
    data = pd.DataFrame(index=df.index, columns=nodes)
    for node in nodes:
        if type(df[node])==pd.Series: data[node]=df[node]
        else: data[node] = df[node].mean(axis=1)

    if type(norm)==str:
        if norm.lower()=="gmm":
            gm = GaussianMixture(n_components=2)
            for gene in data.columns:
                d = data[gene].values.reshape(data.shape[0],1)
                gm.fit(d)
                
                # Figure out which cluster is ON
                idx = 0
                if gm.means_[0][0] < gm.means_[1][0]: idx = 1
                
                data[gene] = gm.predict_proba(d)[:,idx]
        elif norm.lower()=="minmax":
            data = (data - data.min()) / (data.max() - data.min())
    elif type(norm)==float:
        if norm > 0 and norm < 1:
            lq = data.quantile(q=norm)
            uq = data.quantile(q=1-norm)
            data = (data - lq) / (uq - lq)
            data[data<0] = 0
            data[data>1] = 1
            
    

    if sample_order is None:
        data.dropna(inplace=True)
        cluster_linkage = linkage(data)
        cluster_dendro = dendrogram(cluster_linkage, no_plot=True)
        cluster_leaves = [data.index[i] for i in cluster_dendro['leaves']]
        data = data.loc[cluster_leaves]
    elif type(sample_order) != bool: # If sample_order=False, don't sort at all
        data = data.loc[sample_order]
    
    return data

# filenames is a list of filenames, nodes gives the only genes we are reading, log is True/False, or list of [True, False, True...], delimiter is string, or list of strings
def load_data_multiple(filenames, nodes, log=False, delimiter=",", norm="gmm"):
    datasets = []
    for i, filename in enumerate(filenames):
        if type(log)==list: log_i = log[i]
        else: log_i = log
        if type(delimiter)==list: delimiter_i = delimiter[i]
        else: delimiter_i = delimiter
        
        datasets.append(load_data(filename, nodes, log=log_i, sample_order=False, delimiter=delimiter_i, norm=norm))
    
    data = pd.concat(datasets)    

    cluster_linkage = linkage(data)
    cluster_dendro = dendrogram(cluster_linkage, no_plot=True)
    cluster_leaves = [data.index[i] for i in cluster_dendro['leaves']]
    data = data.loc[cluster_leaves]

    return data
    
def binarize_data(data, phenotype_labels=None, threshold=0.5):
    if phenotype_labels is None: binaries = set()
    else:
        binaries = dict()
        for c in phenotype_labels['class'].unique(): binaries[c]=set()

    f = np.vectorize(lambda x: '0' if x < threshold else '1')
    for sample in data.index:
        b = state2idx(''.join(f(data.loc[sample])))

        if phenotype_labels is None: binaries.add(b)
        else: binaries[phenotype_labels.loc[sample,'class']].add(b)
    return binaries
        
    

def idx2binary(idx, n):
    binary = "{0:b}".format(idx)
    return "0"*(n-len(binary))+binary
    
def state2idx(state):
    return int(state,2)
    
# Returns 0 if state is []
def state_bool2idx(state):
    n = len(state)-1
    d = dict({True:1, False:0})
    idx = 0
    for s in state:
        idx += d[s]*2**n
        n -= 1
    return idx
    
# Hamming distance between 2 states
def hamming(x,y):
    s = 0
    for i,j in zip(x,y):
        if i!=j: s+= 1
    return s

# Hamming distance between 2 states, where binary states are given by decimal code
def hamming_idx(x,y,n):
    return hamming(idx2binary(x,n), idx2binary(y,n))
    
# Given a graph calculate the graph condensation (all nodes are reduced to strongly
# connected components). Returns the condensation graph, a dictionary mapping
# SCC->[nodes in G], as well as the output of graph_tool's label_components.
# I often use this on the output of graph_sim.prune_stg_edges, or a deterministic stg
def condense(G, directed=True, attractors=True):
    components = label_components(G, directed=directed, attractors=attractors)
    c_G = gt.Graph()
    c_G.add_vertex(n=len(components[1]))

    vertex_dict = dict()
    for v in c_G.vertices(): vertex_dict[int(v)] = []
    component = components[0]

    for v in G.vertices():
        c = component[v]
        vertex_dict[c].append(v)
        for w in v.out_neighbors():
            cw = component[w]
            if cw==c: continue
            if c_G.edge(c,cw) is None:
                edge = c_G.add_edge(c,cw)
    return c_G, vertex_dict, components
    
def average_state(idx_list, n):
    av = np.zeros(n)
    for idx in idx_list:
        av = av + np.asarray([float(i) for i in idx2binary(idx, n)])/(1.*len(idx_list))
    return av
    
def draw_grn(G, gene2vertex, rules, regulators_dict, fname, gene2group=None, gene2color=None):
    vertex2gene = G.vertex_properties['name']
    
    vertex_group = None
    if gene2group is not None:
        vertex_group = G.new_vertex_property("int")
        for gene in gene2group.keys():
            vertex_group[gene2vertex[gene]]=gene2group[gene]
    
    vertex_colors = [0.4,0.2,0.4,1]
    if gene2color is not None:
        vertex_colors = G.new_vertex_property("vector<float>")
        for gene in gene2color.keys():
            vertex_colors[gene2vertex[gene]]=gene2color[gene]
    
    
    edge_markers = G.new_edge_property("string")
    edge_weights = G.new_edge_property("float")
    edge_colors = G.new_edge_property("vector<float>")
    for edge in G.edges():
        edge_colors[edge] = [0., 0., 0., 0.3]
        edge_markers[edge] = "arrow"
        edge_weights[edge] = 0.2
    
    for edge in G.edges():
        vs, vt = edge.source(), edge.target()
        source = vertex2gene[vs]
        target = vertex2gene[vt]
        regulators = regulators_dict[target]
        if source in regulators:
            i = regulators.index(source)
            n = 2**len(regulators)
            
            rule = rules[target]
            off_leaves, on_leaves = graph_fit.get_leaves_of_regulator(n, i)
            if rule[off_leaves].mean() < rule[on_leaves].mean(): # The regulator is an activator
                edge_colors[edge] = [0., 0.3, 0., 0.8]
            else:
                edge_markers[edge] = "bar"
                edge_colors[edge] = [0.88, 0., 0., 0.5]
            edge_weights[edge] = rule[on_leaves].mean() - rule[off_leaves].mean()+0.2
            
    pos = gt.sfdp_layout(G, groups=vertex_group, mu=1, eweight=edge_weights)
    eprops = {"color":edge_colors, "pen_width":2, "marker_size":15, "end_marker":edge_markers}
    vprops = {"text":vertex2gene, "shape":"octagon", "size":80, "pen_width":1, 'fill_color':vertex_colors}
    gt.graph_draw(G, pos=pos, output=fname, vprops=vprops, eprops=eprops, output_size=(1000,1000))

def compare_rules(regs0, rule0, regs1, rule1):
    present_in_0_missing_in_1 = [i for i in regs0 if not i in regs1]
    present_in_1_missing_in_0 = [i for i in regs1 if not i in regs0]
    for i in present_in_0_missing_in_1:
        regs1 = [i,] + regs1
        rule1 = np.concatenate([rule1, rule1])
    for i in present_in_1_missing_in_0:
        regs0 = [i,] + regs0
        rule0 = np.concatenate([rule0, rule0])

    # To facilitate comparison, regulators must be in the same order
    # Here we make regs1 match regs0
    neworder = graph_fit.reorder_binary_decision_tree(regs1, regs0)
    rule1 = rule1[neworder]

    return np.linalg.norm(rule0-rule1)