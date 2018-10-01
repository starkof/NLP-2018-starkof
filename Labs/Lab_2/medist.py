#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import sys

def medistance(source, target):
    n = len(source)
    m = len(target)
    
    table = np.zeros((n+1, m+1))
    
    for i in range(1, n + 1):
        table[i][0] = table[i-1][0] + del_cost(source[i-1])
    
    for j in range(1, m + 1):
        table[0][j] = table[0][j-1] + ins_cost(target[j-1])
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            table[i][j] = min(table[i-1][j] + del_cost(source[i-1]),
                              table[i-1][j-1] + sub_cost(source[i-1], target[j-1]),
                              table[i][j-1] + ins_cost(target[j-1]))
    return int(table[i][j])


# In[2]:


def del_cost(s):
    return len(s)

def ins_cost(s):
    return len(s)

def sub_cost(source, target):
    if source == target:
        return 0
    else:
        return 2


# In[3]:


def main(argv):
    print(medistance(argv[1], argv[2]))

if __name__ == '__main__':
    main(sys.argv)
