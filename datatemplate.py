#Template for how to read pkl files

import pickle
import pandas as pd

data = pd.read_pickle("data/EmoryNLP.pkl")
print(data)

with open('data/Emor.pkl', 'rb') as f:
    data = pickle.load(f)