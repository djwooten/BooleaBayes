from booleabayes import graph_fit, graph_sim, graph_utils
from booleabayes import truth_tables as tt


graph, vertex_dict = graph_utils.load_network('network.txt', remove_selfloops=True, remove_sinks=False)
v_names = graph.vertex_properties['name']

# vertext_dict maps name -> vertex
# v_names maps vertex -> name (basically the inverse of vertex_dict)

nodes = sorted(vertex_dict.keys())

data = graph_utils.load_data("data.csv", nodes, norm=0.2)

rules, regulators_dict = graph_fit.get_rules(data, vertex_dict, plot=True, threshold=0, directory="output")

graph_fit.save_rules(rules,regulators_dict,fname='output/rules.txt')


# This rounds BooleaBayes outputs to 0's and 1's to get the "best fit". These rules can be run through StableMotifs to get attractors
print(tt.get_most_likely_booleabayes_rules(rules, regulators_dict, fname=None))