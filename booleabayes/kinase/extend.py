import pandas as pd
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from scipy.optimize import fsolve

from sympy import Symbol
#from sympy.solvers import nsolve, solve
from sympy.functions.elementary.miscellaneous import Max, Min

def get_kinase_activators_inhibitors(node, G, kinases):
    """Given a regulatory network consisting of kinases and other types of regulation, we assume kinases follow inhibitory dominant rules. This function seprates out which kinases are activators, and which are inhibitors, based on the "weight" edge property in G.

    :param str node: Name of grpah node, should be a node that is regulated by kinases.
    :param networkx.DiGraph: Directed graph
    :param iterable kinases: list of nodes that are kinases
    """
    activators = set()
    inhibitors = set()
    for edge in G.in_edges(node):
        source = edge[0]
        if source not in kinases: continue
        weight = G.edges[edge]['weight']
        if weight > 0: activators.add(source)
        elif weight < 0: inhibitors.add(source)
    return activators, inhibitors

def create_new_dataset(df, _activated, _activators, _inhibitors):
    """This takes the normalized (0 to 1) gene expression data, and creates "node_active" columns for each gene that has a distinct "activated" state. It fills these in using fuzzy logic algebra.

    :param pandas.DataFrame df: Normalized (0 to 1) transcription values from (e.g.,) rnaseq data.
    :param iterable _activated: List of nodes that should have an "_active" state
    :param dict[str->set] _activators: dict mapping each activatable node to its activator kinases
    :param dict[str->set] _inhibitors: dict mapping each activatable node to its inhibitor kinases

    :return pd.DataFrame: Dataframe equivalent to df, but with additional rows for _active nodes (_T nodes match the original transcription value)
    """
    dfnew = pd.DataFrame(index = df.columns)
    for node in df.index:
        if node in _activated:
            dfnew['%s_T'%node] = 0
            dfnew['%s_active'%node] = 0
        else:
            dfnew[node] = 0
    dfnew = dfnew.transpose()

    for col in df.columns:
        print(col)
        colvals = df[col]
        newvals = solve_system(colvals, _activated, _activators, _inhibitors)
        dfnew[col] = newvals

    return dfnew

def solve_system(colvals, _activated, _activators, _inhibitors, inhibitor_strength=1/3.):
    """Solves a system of equations given by fuzzy logic inhibitory dominant activation for _activated nodes. One of three equations is chosen depending on the types of regulators:

        If no activators (only inhibitors):
            
            node_active = node_T * max(1 - sum(inhibitors), 0)
            
            (As fuzzy logic, this is "Transcribed and not inhibited")

        If no inhibitors (only activators):
            
            node_active = node_T * min(sum(activators), 1)
            
            (As fuzzy logic, this is "Transcribed and activated")
        
        If both inhibitors and activators:
            
            node_active = node_T * min(sum(activators),1) * max(1-sum(inhibtors),0)

            (As fuzzy logic, this is "Transcribed, activated, and not inhibited")

    The min's and max's prevent weird behavior if activators or inhibitors sum to be higher than 1.

    :param pandas.Series colvals: Normalized (0 to 1) probabilities of each node being ON (1=100% sure on, 0=100% sure off)
    :param iterable _activated: List of nodes that should have an "_active" state
    :param dict[str->set] _activators: dict mapping each activatable node to its activator kinases
    :param dict[str->set] _inhibitors: dict mapping each activatable node to its inhibitor kinases
    :param float inhibitor_strength: Relative strength of inhibitors VS activators

    :return pandas.series: System solution with both _T and _active values for each node
    """

    # Build the system of equations
    constants = dict() # This is the transcription information measured experimentally
    for node in df.index:
        if not node in _activated:
            constants[node] = colvals[node]
        else:
            constants["%s_T"%node] = colvals[node]

    symbols = dict() # string to sympy symbols
    eqns = []

    for node in _activated:
        symbols["%s_active"%node] = Symbol("%s_active"%node)

    for node in _activated:
        target = "%s_active"%node

        lhs = symbols[target] # left hand side
        rhs_T = constants["%s_T"%node] # right hand side
        rhs_activators = 0 # right hand side
        rhs_inhibitors = 0 # right hand side

        activators = _activators[target]
        inhibitors = _inhibitors[target]
        
        for edge in G_extended.in_edges("%s_active"%node):
            source = edge[0]
            if source in constants:
                val = constants[source]
            else:
                val = symbols[source]
            if source in activators:
                rhs_activators += val
            elif source in inhibitors:
                rhs_inhibitors += val*inhibitor_strength

        if len(activators) == 0:
            eqn = lhs - rhs_T*Max(1-rhs_inhibitors,0)
        elif len(inhibitors) == 0:
            eqn = lhs-rhs_T*Min(rhs_activators,1)
        else:
            eqn = lhs-rhs_T*Min(rhs_activators,1)*Max(1-rhs_inhibitors,0)

        eqns.append(eqn)

    vars_to_solve = list(symbols.keys())
    syms_to_solve = [symbols[i] for i in vars_to_solve]

    # Start with an initial guess being each _active value is equal to that node's transcription.
    x0_guess = [constants[i.replace("_active","_T")] for i in vars_to_solve]
    
    soln = fsolve(scipy_f, x0_guess, (eqns, syms_to_solve))

    soln = dict(zip(vars_to_solve, soln))
    for c in constants.keys():
        soln[c] = constants[c]
    return pd.Series(soln)

def scipy_f(x, eqns, syms_to_solve):
    """System of equations to be solved in fsolve"""
    ret = []
    values = dict()
    for value, symbol in zip(x, syms_to_solve):
        values[symbol] = value
    
    for eqn in eqns:
        ret.append(eqn.subs(values))
    return ret



################### READ THE INPUTS ################

G = nx.DiGraph()
# network.csv should have "source,target,weight"
df = pd.read_csv('network.csv', header=None).fillna(0)
for i in df.index:
    G.add_edge(df.loc[i,0], df.loc[i,1], weight=df.loc[i,2])

tfs = []
kinases = []
# kinaess.txt should have a single kinase on each line, plain text
with open("kinases.txt","r") as infile:
    for line in infile:
        line = line.strip()
        kinases.append(line)

# tfs.txt should have a single TF on each line, plain text
with open("tfs.txt","r") as infile:
    for line in infile:
        line = line.strip()
        tfs.append(line)

# data.csv should have columns=samples, rows=gene, values are normalized (0 to 1) probability of gene transcription
df = pd.read_csv("data.csv", index_col=0)




################### DO THE WORK ################


# Get list of _active nodes and _T nodes
_activated = set() # nodes regulated by a kinase
_T = set() # nodes regulated by a TF
for kinase in kinases:
    for e in G.out_edges(kinase):
        _activated.add(e[1])
for tf in tfs:
    for e in G.out_edges(tf):
        _T.add(e[1])

for node in _activated:
    if node in tfs: tfs.append("%s_active"%node)
    if node in kinases: kinases.append("%s_active"%node)

# Add distinct _T and _active nodes to G where relevant
G_extended = nx.DiGraph()
for edge in G.edges():
    source = edge[0]
    target = edge[1]
    if target in _activated and source in tfs:
        target = "%s_T"%target
    elif target in _activated and source in kinases:
        target = "%s_active"%target
    if source in _activated:
        source = "%s_active"%source
    G_extended.add_edge(source, target, weight=G.edges[edge]['weight'])


# For nodes that have separate _T and _activated states, we assume activation follows inhibitory doinant rules based on its kinase regulators. Set up dicts to store these signs in memory.
_activators = dict()
_inhibitors = dict()
for node in _activated:
    for suffix in ["_active",]:
        activators, inhibitors = get_kinase_activators_inhibitors("%s%s"%(node, suffix), G_extended, kinases)
        _activators["%s%s"%(node, suffix)] = activators
        _inhibitors["%s%s"%(node, suffix)] = inhibitors

newdf = create_new_dataset(df, _activated, _activators, _inhibitors)

new_edges = []
for edge in G_extended.edges():
    new_edges.append("%s,%s"%(edge[0],edge[1]))
outfile = open("network_extended.csv","w")
outfile.write("\n".join(new_edges))
outfile.close()

newdf.to_csv("extended_df.csv")
newdf.transpose().to_csv("data_ready_for_bb.csv")