import pickle
import pandas as pd

data = pd.read_pickle("/Users/kylakim/Desktop/Thingiverse/Coding/CMU/data/merged_training.pkl")
print(data)

with open('/Users/kylakim/Desktop/Thingiverse/Coding/CMU/data/merged_training.pkl', 'rb') as f:
    data = pickle.load(f)