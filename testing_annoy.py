import pandas as pd

'''
from annoy import AnnoyIndex
import random

f = 40
t = AnnoyIndex(f, 'angular')  # Length of item vector that will be indexed
for i in range(1000):
    v = [random.gauss(0, 1) for z in range(f)]
    t.add_item(i, v)

t.build(10) # 10 trees
t.save('test.ann')

# ...

u = AnnoyIndex(f, 'angular')
u.load('test.ann') # super fast, will just mmap the file
print(u.get_nns_by_item(0, 5, include_distances=True)) # will find the 1000 nearest neighbors
'''

df1 = pd.DataFrame(columns = ['a','b'])
df1 = df1.append({'a':1, 'b':0}, ignore_index=True)

df2 = pd.DataFrame(columns = ['a','c'])
df2 = df2.append({'a':1, 'c':0}, ignore_index=True)

print(pd.concat([df1,df2], ignore_index=True))