
# coding: utf-8

# In[ ]:


#Importing necessary classifiers and libraries
import sys
import naive_bayes
import normalized_naive_bayes
import logistic_regression
import normalized_logistic_regression


# In[ ]:


#Checking the argument variables and decides which classifier to run.
def main(argv):
    if str(argv[1])=='nb' and str(argv[2])=='u':
        naive_bayes.main(argv)
        
    elif str(argv[1])=='nb' and str(argv[2])=='n':
        normalized_naive_bayes.main(argv)
        
    elif str(argv[1])=='lr' and str(argv[2])=='u':
        logistic_regression.main(argv)
        
    elif str(argv[1])=='lr' and str(argv[2])=='n':
        normalized_logistic_regression.main(argv)
        
main(sys.argv)

