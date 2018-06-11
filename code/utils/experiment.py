from scipy.spatial.distance import cosine
from gensim.models import KeyedVectors
from collections import defaultdict
from sklearn.preprocessing import Imputer
import operator
from nltk import ngrams
from numpy import average
from nltk.tree import ParentedTree
import pandas as pd
# contains adapted code from: https://stackoverflow.com/questions/28681741/find-a-path-in-an-nltk-tree-tree

#############################
#                           #
#        BASELINE           #
#                           #
#############################

def get_dictionary_mean(df_train, pred_label="label"):
    roles = df_train.groupby(['role_label'])[pred_label]
    summary = roles.describe()
    dict_mean = summary["mean"].to_dict()
    return dict_mean

def get_dictionary_mf(df_train, pred_label="label"):
    roles = df_train.groupby(['role_label', pred_label]).size().reset_index(name='size')
    dict_roles = roles.to_dict("records")
    freq_dict = defaultdict(dict)
    for group in dict_roles:
        role_label = group["role_label"]
        label = group[pred_label]
        size = group["size"]
        freq_dict[role_label][label] = size
    for role in freq_dict:
        most_frequent = max(freq_dict[role].items(), key=operator.itemgetter(1))[0]
        freq_dict[role] = most_frequent
    return freq_dict


def apply_baseline(df_test, df_train, approach="mean", pred_label="label"):
    
    # Get dictionary with mean or most frequent sense of each label
    if approach == "mean":
        dict_roles = get_dictionary_mean(df_train, pred_label=pred_label)
    elif approach == "mf":
        dict_roles = get_dictionary_mf(df_train, pred_label=pred_label)
        
    # Get role labels of test set and use corresponding mean/mf score as prediction
    roles = df_test.role_label
    predictions = [dict_roles[role] for role in roles]
    
    return predictions 


#############################
#                           #
#     SENTENCE FEATURES     #
#                           #
#############################


def get_ptree(sent_df):
    sentence = sent_df.to_dict('records')
    # create tree with token_identifiers as leaves
    tree_string = "".join([token["parse"].replace("*", f" {token['token_id']} ") for token in sentence])
    ptree = ParentedTree.fromstring(tree_string)
    return ptree

def get_semroles(sent_df, semrole_col):
    sent = sent_df.to_dict('records')
    semroles = list()
    # iterate over tokens to find first tokens of semroles
    for index, token in enumerate(sent):
        if token[semrole_col].startswith("("):
            role = [token]
            
            # deal with multiple role labels for one token 
            if token[semrole_col].count("ARG") == 2 and token[semrole_col].endswith("*))"):
                token[semrole_col] = token[semrole_col].split("(")[1] + ")"
            if token[semrole_col].count("ARG") == 2 and token[semrole_col].endswith("*)"):
                token[semrole_col] = token[semrole_col].split("(")[1]
            
            # iterate over next tokens to find full span
            if not token[semrole_col].endswith(")"):
                for index2, token2 in enumerate(sent[index+1:]):
                    role.append(token2)
                    if token2[semrole_col].endswith(")"):
                        break
            
            # get characteristics of role and add dictionary to list
            role_span = (role[0]["token_id"], role[-1]["token_id"])
            role_tokens = [token["word_form"] for token in role]
            role_label = token[semrole_col].split("*")[0].replace("(", "").replace(")", "")
            role_head_wf = role[-1]["word_form"]
            role_head_pos = role[-1]["POS"]
            dict_semrole = {"role_span": role_span,
                            "role_label": role_label,
                            "role_text": " ".join(role_tokens),
                            "role_tokens": role_tokens,
                            "role_num_tokens": len(role),
                            "role_head_wf": role_head_wf,
                            "role_head_pos": role_head_pos}
            semroles.append(dict_semrole)
    #print(semroles)
    return semroles

def get_predicate_features(predicate_df):
    predicate_wf = list(predicate_df["word_form"])[0]
    predicate_pos = list(predicate_df["POS"])[0]      
    features = {"predicate_wf": predicate_wf,
                "predicate_pos": predicate_pos}
    return features
       
    
def get_semrole_features(semroles, role_span):
    try:
        semrole = [semrole for semrole in semroles if semrole["role_span"] == role_span][0]
    except:
        # accounts for phrasal verbs which include following particle: only check for first part
        try:
            semrole = [semrole for semrole in semroles if semrole["role_span"][0] == role_span[0]][0]
        except:
            # accounts for semroles starting with preposition (also include preceding preposition)
            first_token, last_token = role_span
            new_role_span = (first_token-1,last_token)
            semrole = [semrole for semrole in semroles if semrole["role_span"] == new_role_span][0]
    semrole_tokens = semrole["role_tokens"]
    features = {"semrole_label": semrole["role_label"],
                "semrole_num_tokens": semrole["role_num_tokens"],
                "semrole_head_wf": semrole["role_head_wf"],
                "semrole_head_pos": semrole["role_head_pos"]}
    return features, semrole_tokens

def get_lca_length(location1, location2):
    i = 0
    while i < len(location1) and i < len(location2) and location1[i] == location2[i]:
        i+=1
    return i

def get_labels_from_lca(ptree, lca_len, location):
    labels = []
    for i in range(lca_len, len(location)):
        labels.append(ptree[location[:i]].label())
    return labels

def find_path(ptree, text1, text2):
    leaf_values = ptree.leaves()
    leaf_index1 = leaf_values.index(text1)
    leaf_index2 = leaf_values.index(text2)

    location1 = ptree.leaf_treeposition(leaf_index1)
    location2 = ptree.leaf_treeposition(leaf_index2)
    
    if location1 == location2:
        top_label_lca = ptree[location1[:-1]].label()
        path = top_label_lca
        return path, top_label_lca

    #find length of least common ancestor (lca)
    lca_len = get_lca_length(location1, location2)
    #find path from the node1 to lca

    labels1 = get_labels_from_lca(ptree, lca_len, location1)
    top_label_lca = labels1[0]
    #ignore the first element, because it will be counted in the second part of the path
    result1 = labels1[1:]
    #inverse, because we want to go from the node to least common ancestor
    result1 = result1[::-1]
    result1 = "+".join(result1)
    #add path from lca to node2
    result2 = get_labels_from_lca(ptree, lca_len, location2)
    result2 = "-".join(result2)
    if result1 and result2:
        path = f"{result1}+{result2}" # only add plus if necessary
    else:
        path = f"{result1}{result2}"
    return path, top_label_lca
                
def get_syntactic_features_semrole(ptree, role_start):
    leaf_values = ptree.leaves()
    leaf_index = leaf_values.index(str(role_start))
    location = ptree.leaf_treeposition(leaf_index)

    # get syntactic node
    synt_node = location[:-1]
    synt_node_label = ptree[synt_node].label()

    # get parent
    parent_node = location[:-2]
    parent_node_label = ptree[parent_node].label()

    # get right sibling
    right_node = list(location)[:-1]
    right_node[-1] += 1
    try:
        right_node_label = tree[right_node].label()
    except:
        right_node_label = None

    # get left sibling
    left_node = list(location)[:-1]
    left_node[-1] -= 1
    try:
        left_node_label = tree[left_node].label()
    except:
        left_node_label = None
            
    features = {"semrole_synt_node": synt_node_label,
                "semrole_synt_node_parent": parent_node_label,
                "semrole_synt_node_left": left_node_label,
                "semrole_synt_node_right": right_node_label}
    return features

def get_verb_semrole_features(ptree, role_start, pred_start):
    features = {}
    # determine direction
    if role_start < pred_start:
        features["verb-semrole_direction"] = "before"
    elif role_start > pred_start:
        features["verb-semrole_direction"] = "after" 
    
    # find lowest ancestor and syntactic path from predicate to role
    path, top_label_lca = find_path(ptree, str(pred_start), str(role_start))
    features["verb-semrole_lowest_ancestor"] = top_label_lca
    features["verb-semrole_synt_path"] = path
    
    return features

def get_verbarg_struct_features(semroles):
    features = {}
    
    # determine presence of all role labels and, if present, get their head_wf and head_pos
    for role in semroles:
        label = role["role_label"]
        role_features = {f"verbarg-struct_{label}_present": 1,
                         f"verbarg-struct_{label}_head_wf": role["role_head_wf"],
                         f"verbarg-struct_{label}_head_pos": role["role_head_pos"]}
        features.update(role_features)
    
    # determine last and first role
    features["verbarg-struct_first_role"] = semroles[0]["role_label"]
    features["verbarg-struct_last_role"] = semroles[-1]["role_label"]
    
    return features

#############################
#                           #
#         EXPERIMENT        #
#                           #
#############################


def encode_features(df, feature_categories, impute_prefixes=None, continuous_prefixes=()):
    # Fill missing values with mean for sim_ features
    if impute_prefixes:
        imp = Imputer(missing_values="NaN", strategy="mean")
        feat_to_impute = [feat for feat in df.columns if feat.startswith(impute_prefixes)]
        for feat in feat_to_impute:
            df[feat]=imp.fit_transform(df[[feat]]).ravel()

    # Distinguish between continuous and categorical features
    feature_names = [feat for feat in df.columns if feat.startswith(feature_categories)]
    continuous_features = [feat for feat in df.columns if feat.startswith(continuous_prefixes)]
    categorical_features = [feat for feat in feature_names if feat not in continuous_features]

    # One Hot Encoding of categorical features
    df = pd.get_dummies(df, prefix_sep='#', columns=categorical_features)
    
    return df

def mark_significance(row, column, p_column):
    if row[p_column] <= 0.001:
        return str(round(row[column], 3)) + "**"
    if row[p_column] <= 0.01:
        return str(round(row[column], 3)) + "*"
    if row[p_column] >= 0.01:
        return str(round(row[column], 3))