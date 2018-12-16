
# coding: utf-8

# In[ ]:


'''
Function to calculate for the Minimum Edit Distance between two words
'''
import numpy as np
import sys

 #Initialising cost variables
def del_cost(source):
    return 1
def ins_cost(target):
    return 1
def sub_cost(source,target):
    if source==target:
        return 0
    else:
        return 2
        
def MEDistance(source,target):
    #Creating distance matrix
    row = len(source)+1
    column = len(target)+1
    #matrix = [[0 for x in range (row)] for y in range (column)]
    matrix = np.zeros(shape=[row,column])
   
    #Row value initialization
    for i in range(1,row):
        matrix[i][0] = (matrix[i-1][0])+del_cost(source[i-1])
        
    #Column value initialization
    for j in range(1,column):
        matrix[0][j] = (matrix[0][j-1])+ins_cost(target[j-1])

    #Recurrence relation
    for i in range(1,row):
         for j in range(1,column):
                matrix[i][j] = min((matrix[i-1][j])+del_cost(source[i-1]), 
                                   (matrix[i-1][j-1])+sub_cost(source[i-1],target[j-1]), 
                                   (matrix[i][j-1])+ins_cost(target[j-1]))
        
    return("Minimum edit distance between", source, "and", target, "is", str(matrix[row-1][column-1]))
print(MEDistance(str(sys.argv[1]),str(sys.argv[2])))

