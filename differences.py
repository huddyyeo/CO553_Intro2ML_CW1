import numpy as np
import pandas as pd
import evaluation as ev
import trees
from collections import Counter

clean_data = np.loadtxt('clean_dataset.txt')
noisy_data = np.loadtxt('noisy_dataset.txt')


# all data counts are the same for each of the 7 emitters
def check_data_count(data1, data2):
    for i in range(7):
        count = Counter(data1[:, i])
        count2 = Counter(data2[:, i])
        if count2 == count:
            continue
        print("Data count not the same for all 7 emitters")
    print("Data count is the same for all 7 emitters")


# labels are different
def labels(data1, data2):
    print(np.bincount([int(i) for i in data1[:, -1]]))
    print(np.bincount([int(i) for i in data2[:, -1]]))


def get_different_rows(source_df, new_df, which=None):
    """
    The above showed that the both datasets has the same router values for each emitter,
    even though the count for each room is different.

    This function returns just the rows from the new dataframe
    that differ from the source dataframe
    """
    merged_df = source_df.merge(new_df, indicator=True, how='outer')
    if which is None:
        diff_df = merged_df[merged_df['_merge'] != 'both']
    else:
        diff_df = merged_df[merged_df['_merge'] == which]

    return diff_df.drop('_merge', axis=1)


# This takes only the signal values, and returns an empty dataframe,
# which implies that the observations are exactly the same across both datasets.

def check_observations_only(data1, data2):
    df1 = pd.DataFrame(data1[:, :-1])
    df2 = pd.DataFrame(data2[:, :-1])
    return get_different_rows(df1, df2)


# This returns the observations that have the same signal values but of different rooms.
def check_signals_only(data1, data2):
    df1 = pd.DataFrame(data1)
    df2 = pd.DataFrame(data2)
    diff_obs = get_different_rows(df1, df2, which=None)
    diff_obs = diff_obs.sort_values(by=[0, 1, 2, 3, 4, 5, 6])
    return diff_obs


# Training and testing a tree on the noisy dataset without those observations above
# Results in a 97% accuracy
def new_model(data1, data2):
    diff_obs = check_signals_only(data1, data2)
    df2 = pd.DataFrame(data2)
    clean_removed = pd.concat([df2, diff_obs, diff_obs]).drop_duplicates(keep=False)
    clean_removed_dataset = clean_removed.to_numpy()

    np.random.shuffle(clean_removed_dataset)
    split = 0.7
    train = clean_removed_dataset[:int(len(clean_removed_dataset) * split)]
    test = clean_removed_dataset[int(len(clean_removed_dataset) * split):]

    model = trees.binarySearchTree(train)
    print('Max depth is', model.get_max_depth())
    y_pred = model.predict(test[:, :-1])
    cm = ev.confusion_matrix(test[:, -1], y_pred)
    i = ev.get_metrics(cm, printout=True)
    ev.plot_conf_matrix(cm)
