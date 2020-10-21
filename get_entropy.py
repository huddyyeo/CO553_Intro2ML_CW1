import numpy as np

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
