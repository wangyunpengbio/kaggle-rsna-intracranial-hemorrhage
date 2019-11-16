
# coding: utf-8

# In[1]:


import pandas as pd
from glob import glob
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

datadir = "./data/submission"


# In[2]:


def filetoDataframe(submitFiles):
    filedf = pd.read_csv(submitFiles[0])
    for file in submitFiles[1:]:
        filedf2 = pd.read_csv(file)
        filedf = pd.merge(filedf, filedf2, how='inner', on=['ID', 'ID'])

    filedf["avg"] = filedf.mean(axis=1)
    dataframe = filedf.loc[:,["ID","avg"]]
    dataframe.rename(columns={'avg':'Label'}, inplace = True)
    return dataframe


# In[ ]:


submitFiles = glob(join(datadir,"*resnet34*_ep3_test_tta5.csv"))
avgdfresnet34 = filetoDataframe(submitFiles)


# In[ ]:


submitFiles = glob(join(datadir,"*_ep2_test_tta5.csv"))
submitFiles = [x for x in submitFiles if "efficientnet" not in x]
avgdfresnext50 = filetoDataframe(submitFiles)


# In[ ]:


submitFiles = glob(join(datadir,"*efficientnet*_ep*_test_tta5.csv"))
avgdfefficientnet = filetoDataframe(submitFiles)


# In[ ]:


avgdfresnet34.head()


# In[ ]:


avgdfresnext50.head()


# In[ ]:


avgdfefficientnet.head()


# In[ ]:


avgdfresnet34.to_csv("1-test-avgresnet34.csv",index=False)
avgdfresnext50.to_csv("1-test-avgresnext50.csv",index=False)
avgdfefficientnet.to_csv("1-test-avgefficientnet.csv",index=False)


# In[ ]:


alldata = pd.merge(avgdfresnet34, avgdfresnext50, how='inner', on=['ID', 'ID'])
alldata = pd.merge(alldata, avgdfefficientnet, how='inner', on=['ID', 'ID'])


# In[ ]:


alldata.rename(columns={'Label_x':'resnet34Ep3',
                       'Label_y':'resnext50Ep2',
                       'Label':'efficientnetb3Ep2'}, inplace = True)


# In[ ]:


alldata.head()


# In[ ]:


alldata["avg"] = alldata.mean(axis=1)
avgdf = alldata.loc[:,["ID","avg"]]
avgdf.rename(columns={'avg':'Label'}, inplace = True)
avgdf.to_csv("1-test-avg3model.csv",index=False)


# In[ ]:


def weightAverage(x,weight=[0.1,0.6,0.3]):
    return np.average(np.array(x,dtype=np.double),weights = weight) 


# In[ ]:


alldata["weightavg"] = alldata.apply(lambda row: weightAverage(row[1:4], weight = [0.15,0.5,0.35]), axis=1)


# In[ ]:


avgweightdf = alldata.loc[:,["ID","weightavg"]]
avgweightdf.rename(columns={'weightavg':'Label'}, inplace = True)


# In[ ]:


avgweightdf.to_csv("1-best-test-weightavg3model.csv",index=False)

# In[ ]:


alldata["weightavg"] = alldata.apply(lambda row: weightAverage(row[1:4], weight = [0.3,0.4,0.3]), axis=1)


# In[ ]:


avgweightdf = alldata.loc[:,["ID","weightavg"]]
avgweightdf.rename(columns={'weightavg':'Label'}, inplace = True)


# In[ ]:


avgweightdf.to_csv("1-test-weightavg3model.csv",index=False)