import numpy as np


def update_data_collected(data_collected:dict, data_new:dict)->dict:
    """ update the collected data with new data

    Args:
        data_collected (dict): the collected data dictionary
        data_new (dict): the new data dictionary

    Returns:
        dict: the updated collected data dictionary
    """

    data_collected["params"].append(data_new["params"])
    data_collected["metrics"].append(data_new["metrics"])
    data_collected["targets"].append(data_new["targets"])
    data_collected["aux_info"].append(data_new["aux_info"])
    data_collected["params_numpy"] = np.vstack((data_collected["params_numpy"], data_new["params_numpy"]))
    return data_collected


def get_example(data_collected:dict, idx:int) -> dict:
    """ get the example from the collected data

    Args:
        data_collected (dict): the collected data dictionary
        idx (int): the index of the example

    Returns:
        dict: the example dictionary
    """
    example = {}
    for key in data_collected.keys():
        example[key] = data_collected[key][idx]
    return example

def get_examples(data_collected:dict, idxs:list) -> dict:
    """ get the examples from the collected data

    Args:
        data_collected (dict): the collected data dictionary
        idxs (list): the list of indices of the examples

    Returns:
        dict: the examples dictionary
    """
    examples = {}
    for key in data_collected.keys():
        examples[key] = [data_collected[key][idx] for idx in idxs]
    return examples

def get_best_example(data_collected:dict) -> dict:
    """ get the best example from the collected data

    Args:
        data_collected (dict): the collected data dictionary

    Returns:
        dict: the best example dictionary
    """
    idx_best = np.argmax(data_collected["targets"])
    best_example = get_example(data_collected, idx_best)
    return best_example

def get_topk_indices(data_collected:dict, k:int) -> list:
    """ get the top-k indices from the collected data

    Args:
        data_collected (dict): the collected data dictionary
        k (int): the number of top examples to return

    Returns:
        dict: the top-k indices list
    """
    idxs_topk = np.argsort(-np.array(data_collected["targets"])).tolist()[:k]
    return idxs_topk

def get_random_indices(data_collected:dict, k:int) -> list:
    """ get the random k indices from the collected data

    Args:
        data_collected (dict): the collected data dictionary
        k (int): the number of random examples to return

    Returns:
        dict: the random k indices list
    """
    idxs_random = np.random.choice(len(data_collected["targets"]), k, replace=False).tolist()
    return idxs_random

def get_topk_examples(data_collected:dict, k:int) -> dict:
    """ get the top-k examples from the collected data

    Args:
        data_collected (dict): the collected data dictionary
        k (int): the number of top examples to return

    Returns:
        dict: the top-k examples dictionary
    """
    idxs_topk = get_topk_indices(data_collected, k)
    topk_examples = get_examples(data_collected, idxs_topk)
    return topk_examples

def get_random_examples(data_collected:dict, k:int) -> dict:
    """ get the random k examples from the collected data

    Args:
        data_collected (dict): the collected data dictionary
        k (int): the number of random examples to return

    Returns:
        dict: the random k examples dictionary
    """
    idxs_random = get_random_indices(data_collected, k)
    random_examples = get_examples(data_collected, idxs_random)
    return random_examples

def get_subset(data_collected:dict, n_topk:int=5, n_random:int=5) -> dict:
    """ get the subset of examples from the collected data

    Args:
        data_collected (dict): the collected data dictionary
        topk (int): the number of top examples to return
        random (int): the number of random examples to return

    Returns:
        dict: the subset examples dictionary, descendingly sorted by the target values
    """
    topk_idx = get_topk_indices(data_collected, n_topk)
    random_idx = get_random_indices(data_collected, n_random+ n_topk)
    subset_idx = list(set(topk_idx + random_idx))[:n_topk+n_random]
    subset_examples = get_examples(data_collected, subset_idx)
    return get_topk_examples(subset_examples, n_topk+n_random)