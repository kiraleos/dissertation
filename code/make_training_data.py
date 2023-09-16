import pandas as pd

df = pd.read_csv('TESTclean.csv')

threshold = 0.35
df = df[df['z'] < threshold]

df.to_csv('TESTclean_filtered.csv', index=False)
