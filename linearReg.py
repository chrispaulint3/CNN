import pandas as pd
import matplotlib.pyplot as plt

path = "./IrisDataset/Iris.csv"
irisDf = pd.read_csv(path)
fig, ax = plt.subplots(figsize=(4, 5))
