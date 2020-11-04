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
diff_obs = get_different_rows(df1, df2)
column_names = {x: "Emitter " + str(x) for x in range(7)}

diff_obs.rename(columns={**column_names, **{7: 'Room'}})


# Training and testing a tree on the noisy dataset without those observations above
pd.concat([df2, diff_obs, diff_obs]).drop_duplicates(keep=False)
diff_obs = diff_obs.to_numpy()
