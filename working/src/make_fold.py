SEED = 42

import pandas as pd
from sklearn.model_selection import KFold
df = pd.read_csv('../../input/squad1/squad1.2_ta_formatted.csv')
kf = KFold(n_splits=5, random_state=SEED, shuffle=True)
for i, (train_index, test_index) in enumerate(kf.split(df)):
    df.loc[test_index, 'fold'] = i
df['fold'] = df['fold'].astype('int')
df.to_csv('../../input/squad1/squad1.2_ta_formatted.csv', index=False)
print(df.head())