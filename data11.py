import pickle
import pandas as pd

data = pd.read_pickle("merged_training".pkl)
print(data)

with open('/Users/kylakim/Desktop/Thingiverse/Coding/CMU/data/serialized.pkl', 'rb') as f:
    data = pickle.load(f)