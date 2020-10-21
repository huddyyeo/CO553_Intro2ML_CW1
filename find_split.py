import numpy as np


def find_split(data):

    if len(np.unique(data[:,7])) == 1:
        return "Something"
    
    ret = {"feature": None, "split_point": None, "entropy": 0}

    # Create array of unique datapoints
    split_points_list = list()

    for i in range(data.shape[1] - 1):
        unique_signal = np.unique(data[:, i])
        split_points = list()
        for j in range(len(unique_signal) - 1):
            split_points.append((unique_signal[j]+unique_signal[j+1])/2)
        split_points_list.append(split_points)

    # Split data by split points and calculate entropy
    for feature in range(len(split_points_list)):
        for i in range(len(split_points_list[feature])):
            data_l, data_r = split_data(data, feature + 1, split_points_list[feature][i])
            entropy = get_entropy(data_l[:, 7], data_r[:, 7])
            if entropy > ret["entropy"]:
                ret["entropy"] = entropy
                ret["feature"] = feature + 1
                ret["split_point"] = split_points_list[feature][i]

    return ret["feature"], ret["split_point"]


def split_data(data, feature, split_point):

    data_l = data[data[:, feature - 1] < split_point]
    data_r = data[data[:, feature - 1] >= split_point]

    return data_l, data_r


def get_entropy(*labels):
    '''
    Calculates the entropy for the given set (or subsets)
    -------
    *labels: list(s), list(s) of labels
    '''
    entropies = [] #list of entropy values from each subset
    total = 0      #total number of datapoints
    for subset in labels:
        n = len(subset)
        total += n
        counts = np.unique(subset, return_counts=True)[1]         #frequency of unique values
        entropy = np.sum([-(i/n) * np.log2(i/n) for i in counts]) #subset entropy calcuation
        entropies.append((entropy, n))
    return np.sum([(n/total) * ent for n, ent in iter(entropies)])
