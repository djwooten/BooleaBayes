import numpy as np
from sympy import symbols, evaluate, substitution
from sympy.logic import SOPform
from sympy.logic.boolalg import simplify_logic
from sympy.parsing.sympy_parser import parse_expr

def get_regulator_signs(rule):
    n_regulators = int(np.log2(len(rule)))
    signs = dict()
    for idx in range(n_regulators):
        signs[idx] = {-1:0, 0:0, 1:0}
        off_leaves, on_leaves = get_leaves_of_regulator(len(rule), idx)
        for a,b in zip(off_leaves, on_leaves):
            # These are for continuous rules, like booleabayes, but should use get_continuous_regulator_signs instead
            #if rule[b] > rule[a]: signs[idx][1] += 1
            #elif rule[b] < rule[a]: signs[idx][-1] += 1
            #else: signs[idx][0] += 1
            sign = int(round(rule[b])) - int(round(rule[a]))
            signs[idx][sign] += 1
    return signs

def get_regulator_signs_continuous(rule):
    n_regulators = int(np.log2(len(rule)))
    signs = dict()
    for idx in range(n_regulators):
        signs[idx] = 0
        off_leaves, on_leaves = get_leaves_of_regulator(len(rule), idx)
        for a,b in zip(off_leaves, on_leaves):
            signs[idx] += rule[b]-rule[a]
    return signs

def get_max_regulatory_weight(rule):
    n_regulators = int(np.log2(len(rule)))
    signs = dict()
    
    for idx in range(n_regulators):
        absmax = -1
        trumax = 0
        off_leaves, on_leaves = get_leaves_of_regulator(len(rule), idx)
        for a,b in zip(off_leaves, on_leaves):
            diff = rule[b]-rule[a]
            absdiff = abs(diff)
            if absdiff > absmax:
                absmax = absdiff
                trumax = diff
        signs[idx] = trumax
    return signs

def get_average_on_off(rule): # This is 0 if a node is necessary
    n_regulators = int(np.log2(len(rule)))
    on = dict()
    off = dict()
    for idx in range(n_regulators):
        on[idx] = 0
        off[idx] = 0
        off_leaves, on_leaves = get_leaves_of_regulator(len(rule), idx)
        for a,b in zip(off_leaves, on_leaves):
            on[idx] += 2*rule[b]/len(rule)
            off[idx] += 2*rule[a]/len(rule)
    return on, off

def detect_irrelevant_regulators(rule):
    if type(rule)==str: rule, tokens = get_array(rule)
    irrelevant = []
    if len(rule)==0: return irrelevant
    n_regulators = int(np.round(np.log2(len(rule))))
    for i in range(n_regulators):
        relevant = False
        off_leaves, on_leaves = get_leaves_of_regulator(len(rule), i)
        for off, on in zip(off_leaves, on_leaves):
            if rule[off] != rule[on]:
                relevant = True
                break
        if not relevant: irrelevant.append(i)
    return irrelevant

def remove_irrelevant_regulators(rule, regulators=None):
    if type(rule)==str: rule, regulators = get_array(rule)
    if len(rule)==0: return rule, []
    n_regulators = int(np.round(np.log2(len(rule))))
    if regulators is None: regulators = list(range(n_regulators))
    
    irrelevant = detect_irrelevant_regulators(rule)
    leaves_to_remove = set()
    for regulator in irrelevant:
        off_leaves, on_leaves = get_leaves_of_regulator(len(rule), regulator)
        leaves_to_remove = leaves_to_remove.union(set(on_leaves))
    return np.asarray([rule[i] for i in range(len(rule)) if not i in leaves_to_remove]), [regulators[i] for i in range(n_regulators) if not i in irrelevant]


def get_leaves_of_regulator(n, index):
    step_size = int(np.round(n/(2**(index+1))))
    num_steps = int(np.round(n/step_size/2))
    off_leaves = []
    on_leaves = []

    base = 0
    for step in range(num_steps):
        for i in range(step_size): off_leaves.append(base+i)
        base += step_size
        for i in range(step_size): on_leaves.append(base+i)
        base += step_size
    return off_leaves, on_leaves

def binary_array_to_int(a):
    if len(a) < 64: return np.asarray(a).dot(1 << np.arange(len(a)-1,-1,-1))
    return int(''.join([str(int(i)) for i in a]),2)

def get_bit(a, n):
    return (a>>n)&1

def get_rule_string(rule, regulators=None):
    n_inputs = int(np.log2(len(rule)))
    if (n_inputs==0):
        if rule[0] < 0.5: return "False"
        return "True"

    if regulators is None:
        regulators = [chr(i+66) for i in range(n_inputs)]
    syms = symbols(' '.join(regulators))
    if not hasattr(syms, "__iter__"): syms = [syms,]
        
    minterms = []
    for i, leaf in enumerate(rule):
        if leaf=="1" or leaf==1:
            minterm = "{0:b}".format(i)
            minterm = (n_inputs-len(minterm))*"0" + minterm
            minterms.append([int(i) for i in minterm])
    j = repr(SOPform(syms, minterms))
    return j.replace('~', 'not ').replace('|',"or").replace("&","and")

def get_SOP(rule, deep=True, force=False):
    rule = rule.replace('and','&').replace('or','|').replace('not ','~').replace("E","__E__")
    rule = str(simplify_logic(rule, form='dnf', deep=deep, force=force))
    return rule.replace('&','and').replace('|','or').replace('~','not ').replace("__E__","E")

def get_tokens(rule_string):
    tokens = set()
    for t in set(rule_string.split(' ')):
        tstrip = t.replace('(','').replace(')','').replace('~','').strip()
        if tstrip.lower() not in ['not', 'and', 'or', '&', '|', '~', 'true', 'false', '0', '1', '']: tokens.add(tstrip)#.replace("__E__","E"))
    return sorted(list(tokens))

def get_array(rule):
    if rule in ['0', 'False', 0, 'false']: return np.asarray([0]), []
    elif rule in ['1', 'True', 1, 'true']: return np.asarray([1]), []
    rule = rule.replace(' and ',' & ').replace(' or ',' | ').replace('not ','~').replace("E","__E__")
    tokens = get_tokens(rule)
    if len(tokens)==0: return np.asarray([]), []
    syms = symbols(' '.join(tokens))
    if not hasattr(syms, "__iter__"): syms = [syms,]
    substitution_dict = dict()

    f = parse_expr(rule)
    
    n = len(tokens)
    n_leaves = 2**len(tokens)
    rule_array = np.zeros(n_leaves)
    for i in range(n_leaves):
        binary = idx2binary(i, n)
        for sym, val in zip(syms, binary):
            substitution_dict[sym]=val
        if f.subs(substitution_dict): rule_array[i] = 1
        else: rule_array[i] = 0
    for i in range(len(tokens)): tokens[i] = tokens[i].replace("__E__","E")
    return rule_array, tokens
    
def idx2binary(idx, n):
    binary = "{0:b}".format(idx)
    return "0"*(n-len(binary))+binary

def idx2binary_array(idx, n):
    return np.asarray([int(i) for i in idx2binary(idx,n)], dtype=np.uint8)

def get_weighted_rules(weights, nodes=None, inhibitory_dominant=False, get_string=True, fname=None):
    '''
    weights[target,regulator] is the influence of node regulator on node target
    '''
    n = weights.shape[0]
    if nodes is None:
        nodes = [chr(i+66) for i in range(n)]
    
    if inhibitory_dominant:
        weights = weights.copy()
        weights[weights<0] *= n

    rules = []
    for ni in range(n):
        w = weights[ni,:]
        regulator_indices = np.where(w!=0)[0]
        w = w[regulator_indices]
        regulators = [nodes[i] for i in regulator_indices]
        n_regulators = len(regulators)
        n_leaves = 2**n_regulators
        rule = np.zeros(n_leaves)
        for leaf_i in range(n_leaves):
            regulator_state = idx2binary_array(leaf_i, n_regulators)
            x = np.dot(regulator_state, w)
            if (x > 0): rule[leaf_i] = 1
        if get_string:
            rules.append("%s *= %s"%(nodes[ni],get_rule_string(rule, regulators=regulators)))
        else: rules.append(rule)
    
    if (fname is not None and get_string):
        outfile = open(fname, "w")
        outfile.write("#BOOLEAN RULES\n")
        outfile.write("\n".join(rules))
        outfile.close()
    return rules

def get_most_likely_booleabayes_rules(rules, regulators_dict, fname=None):
    if fname is not None:
        outfile = open(fname, "w")
        outfile.write("#BOOLEAN RULES")

    rule_strings = []
    for node in rules:
        rule = np.round(rules[node]) # Rounding makes it deterministic
        regulators = regulators_dict[node]
        if node in regulators:
            rule_string = node
        else:
            rule_string = get_rule_string(rule, regulators=regulators)
        rule_strings.append(rule_string)
        if fname is not None:
            outfile.write("\n%s *= %s"%(node,rule_string))
    return rule_strings

if __name__=="__main__":

    if (False):
        n_inputs = 3
        rule_size = 2**n_inputs
        good_rules = []
        for r in range(2**rule_size):
            good=True
            rule = "{0:b}".format(r)
            rule = (rule_size - len(rule)) * '0' + rule
            signs = get_regulator_signs([int(i) for i in rule])
            for i in range(n_inputs):
                if signs[i][-1]*signs[i][1] != 0:
                    print("Conflicting rule: %s"%rule)
                    good=False
                    break
                if signs[i][0] == rule_size / 2:
                    print("Overspecified rule: %s"%rule)
                    good=False
                    break
            if good:
                good_rules.append(rule)

        syms = symbols(' '.join([chr(i+66) for i in range(n_inputs)]))
        for rule in good_rules:
            minterms = []
            for i, leaf in enumerate(rule):
                if leaf=="1":
                    minterm = "{0:b}".format(i)
                    minterm = (n_inputs-len(minterm))*"0" + minterm
                    minterms.append([int(i) for i in minterm])
            j = repr(SOPform(syms, minterms))
            print(rule)
            print(j.replace('~', 'not ').replace('|',"or").replace("&","and"))

    rule = "NtSyp121 and GHR1 and MRP5 or not ABH1 or not ERA1 or Actin"
    ra, regulators = get_array(rule)