import numpy as np
import pandas as pd
import find_split as fs
import evaluation as ev
import matplotlib.pyplot as plt
import trees
from collections import Counter

data = np.loadtxt('clean_dataset.txt')
data2 = np.loadtxt('noisy_dataset.txt')

# all data counts are the same for each of the 7 emitters
for i in range(7):
    count = Counter(data[:, i])
    count2 = Counter(data[:, i])
    print(count2 == count)

# labels are different
print(np.bincount([int(i) for i in data[:, -1]]))
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
df1 = pd.DataFrame(data[:, :-1])
df2 = pd.DataFrame(data2[:, :-1])
get_different_rows(df1, df2)

# This returns the observations that have the same signal values but of different rooms.
df1 = pd.DataFrame(data)
df2 = pd.DataFrame(data2)
diff_obs = get_different_rows(df1, df2, which=None)
diff_obs = diff_obs.sort_values(by=[0, 1, 2, 3, 4, 5, 6])

# Training and testing a tree on the noisy dataset without those observations above
# Results in around a 97% accuracy
clean_removed = pd.concat([df2, diff_obs, diff_obs]).drop_duplicates(keep=False)
clean_removed_dataset = clean_removed.to_numpy()

np.random.shuffle(clean_removed_dataset)
split = 0.7
train = clean_removed_dataset[:int(len(clean_removed_dataset) * split)]
test = clean_removed_dataset[int(len(clean_removed_dataset) * split):]

model = trees.binarySearchTree(train)
print('Max depth is', model.get_max_depth())
y_pred = model.predict(test[:, :-1])
cm = ev.confusion_matrix(test[:, -1], y_pred, plot=True)
i = ev.get_metrics(test[:, -1], y_pred, printout=True)
