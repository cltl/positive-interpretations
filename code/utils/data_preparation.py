import csv
import collections
import pandas as pd
from random import shuffle
from tqdm import tqdm


def get_all_tokens_conll(conll_file):
    """
    Reads a CoNLL-2011 file and returns all tokens with their annotations in a dataframe including the original 
    sentence identifiers from OntoNotes
    """
    all_tokens = list()
    most_semroles = 0
    with open(conll_file, "r") as infile:
        for line in infile:
            # Get sentence identifiers: distinguish between sentence count per file and per file part
            # (some files are divided into multiple parts numbered as 000, 001, 002, ... etc.)
            if line.startswith("#begin document"):
                sent_id_part = 0
                part_id = line.split("; part ")[1].rstrip("\n")
                if part_id == "000":
                    sent_id_file = 0
                else:
                    sent_id_file += 1
            elif line.startswith("#end document"):
                sent_id_file -= 1 # prevent counting too much (empty line followed by end document)
            elif line == "\n":
                sent_id_part += 1
                sent_id_file += 1
            else:
                columns = line.split()
                dict_token = {"file_id": columns[0],
                              "part_id": int(columns[1]), 
                              "sent_id_part": int(sent_id_part),
                              "sent_id_file": int(sent_id_file),
                              "token_id": columns[2],
                              "word_form": columns[3],
                              "POS": columns[4],
                              "parse": columns[5],
                              "pred_lemma": columns[6],
                              "pred_frameset": columns[7],
                              "word_sense": columns[8],
                              "speaker": columns[9],
                              "NE": columns[10],
                              "coref": columns[-1].rstrip("\n")
                             }
                semroles = {f"APRED{i}": role for i, role in enumerate(columns[11:-1], 1)}
                dict_token.update(semroles)
                all_tokens.append(dict_token)
                if len(semroles) > most_semroles:
                    most_semroles = len(semroles)  
                    cols = list(dict_token.keys())
    df_tokens = pd.DataFrame(all_tokens, columns=cols)
    return df_tokens

def find_original_sent_ids(df_instances, df_conll):
    """
    Takes the file_id, part_id and sent_id indicating a specific sentence in the CoNLL-2011 data (where file is split
    into smaller parts and sent_id restarts for each part) and finds the corresponding 'original' sentence identifier
    """
    print("Finding original sentence identifiers")
    for index, row in tqdm(df_instances.iterrows(), total=len(df_instances)):
        # For each instance in the set, find the corresponding sent_id_file in the annotations of CoNLL-2011
        file_id = row["file_id"]
        part_id = row["part_id"]
        sent_id_part = row["sent_id_part"]
        matching_rows = df_conll.loc[(df_conll["file_id"] == file_id) & (df_conll["part_id"] == part_id) &
                                   (df_conll["sent_id_part"] == sent_id_part)]
        sent_id_file = matching_rows.iloc[0]["sent_id_file"]
        df_instances.set_value(index, "sent_id_file", sent_id_file)
    return df_instances

def get_role_features_from_annotations(role_annotations):
    """Splits the verb and role information (in original annotations file) to separate values"""
    head, role = role_annotations.split(")] ")
    head_pos, head_wf = head.lstrip("[(").split()
    span, tokens = role.split(maxsplit=1)
    span, label = span.rstrip(":").split(":")
    role_features = (head_wf, head_pos, span, label, tokens)
    return role_features

def rewrite_verb_and_role_features(df):
    """Rewrites the verb and role information in the original annotations file to separate columns"""
    instances = df.to_dict("records")
    for index, inst in enumerate(instances):

        # Get verb features
        verb = inst["verb"]
        verb_features = get_role_features_from_annotations(verb)
        verb_wf, verb_pos, verb_span, verb_label, verb_tokens = verb_features

        # Get role features
        role = inst["role"]
        role_features = get_role_features_from_annotations(role)
        role_head_wf, role_head_pos, role_span, role_label, role_tokens = role_features

        new_dict = {"verb_wf": verb_wf,
                    "verb_pos": verb_pos,
                    "verb_span": verb_span, 
                    "verb_label": verb_label, 
                    "verb_tokens": verb_tokens,
                    "role_head_wf": role_head_wf,
                    "role_head_pos": role_head_pos,
                    "role_span": role_span,
                    "role_label": role_label,
                    "role_tokens": role_tokens,
                    "role_tokens": role_tokens}
        inst.update(new_dict)
        del inst["verb"]
        del inst["role"]
        instances[index] = inst
    columns = list(instances[0].keys())
    df = pd.DataFrame(instances, columns=columns)
    return df

def transform_labels_three(row):
    """Takes original score (label) and converts to tertiary classes"""
    label = int(row['label'])
    if label <= 1:
        return 0
    if 1 < label <= 3:
        return 1    
    if label >= 4:
        return 2 

def transform_labels_two(row):
    """Takes original score (label) and converts to binary classes"""
    label = int(row['label'])
    if label <= 2:
        return 0
    else:
        return 1    

def categorize_scores(df):
    """Takes original score (label) and converts to tertiary/binary classes"""
    df["class_tertiary"] = df.apply(lambda row: transform_labels_three(row),axis=1)
    df["class_binary"] = df.apply(lambda row: transform_labels_two(row),axis=1)
    return df

def split_train_test(df_instances, test_ratio=0.2, to_shuffle=True):
    """Splits the instances into train and test sets. Each negation is either assigned to the train or test set."""
    instances = df_instances.to_dict("records")
    neg_ids = list({(inst["file_id"], inst["sent_id_file"], inst["verb_span"]) for inst in instances})
    if to_shuffle:
        shuffle(neg_ids)
    test_size = int(len(neg_ids) * test_ratio)
    test_ids = neg_ids[0:test_size]
    test_instances = [inst for inst in instances if (inst["file_id"], 
                                                     inst["sent_id_file"], 
                                                     inst["verb_span"]) in test_ids]
    train_instances = [inst for inst in instances if (inst["file_id"], 
                                                      inst["sent_id_file"], 
                                                      inst["verb_span"]) not in test_ids]
    columns = list(train_instances[0].keys())
    df_train = pd.DataFrame(train_instances, columns=columns)
    df_test = pd.DataFrame(test_instances, columns=columns)
    return df_train, df_test

def k_fold(df_instances, k=10):
    """Divides all the samples in k groups of samples. Each negation is either assigned to the train or test set."""
    instances = df_instances.T.to_dict().values()
    neg_ids = list({(inst["file_id"], inst["sent_id_file"], inst["verb_span"]) for inst in instances})
    kf = list()
    test_size = int(len(neg_ids) / k)
    start = 0
    for n in range(0, k):
        test_ids = neg_ids[start:start+test_size]
        test_instances = [inst for inst in instances if (inst["file_id"], 
                                                         inst["sent_id_file"], 
                                                         inst["verb_span"]) in test_ids]
        train_instances = [inst for inst in instances if (inst["file_id"], 
                                                          inst["sent_id_file"], 
                                                          inst["verb_span"]) not in test_ids]
        train_test = (pd.DataFrame(train_instances), pd.DataFrame(test_instances))
        kf.append(train_test)
        start += test_size
        
    return kf
