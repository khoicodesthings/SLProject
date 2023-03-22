import pandas as pd

data = pd.read_csv('final.csv',header=0)

print(data)

data = data.drop_duplicates()

print(data)

print('Number of dislikes:',data['preference'].value_counts()[0])

print('Number of likes:',data['preference'].value_counts()[1])