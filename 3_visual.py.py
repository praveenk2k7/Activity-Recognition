
import pandas as pd
import numpy as np


# In[8]:


data=pd.read_csv('Data.csv')


# In[9]:


data.head()


# In[10]:


df=data.drop(['attitude_sum_roll','attitude_sum_pitch','attitude_sum_yaw','gravity_sum_x','gravity_sum_y', 'gravity_sum_z', 'rotationRate_sum_x','rotationRate_sum_y', 'rotationRate_sum_z', 'userAcceleration_sum_x',
       'userAcceleration_sum_y','attitude_sumSS_roll', 'attitude_sumSS_pitch','attitude_sumSS_yaw', 'gravity_sumSS_x', 'gravity_sumSS_y',
       'gravity_sumSS_z', 'rotationRate_sumSS_x', 'rotationRate_sumSS_y','rotationRate_sumSS_z', 'userAcceleration_sumSS_x',
       'userAcceleration_sumSS_y', 'userAcceleration_sumSS_z','Activities_Types'],inplace=False,axis=1)


# In[11]:


labels=data['Activities_Types'].as_matrix()
labels


# In[12]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(df,labels,test_size=0.2)    


# In[13]:


y = np.zeros((y_train.shape[0], 6))
y[np.arange(y_train.shape[0]), y_train-1] = 1
y1 = np.zeros((y_test.shape[0], 6))
y1[np.arange(y_test.shape[0]), y_test-1] = 1


# In[23]:


A=x_test


# In[24]:


c=(A-np.mean(A,axis=0)).T


# In[25]:





# In[26]:


cov=np.cov(c)


# In[27]:


cov.shape


# In[28]:


val,eigv=np.linalg.eig(cov)


# In[30]:


P=eigv[0:2].dot(c)



# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(val[i]), eigv[:,i]) for i in range(len(val))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
#for i in eig_pairs:
#    print(i[0])


# In[34]:


matrix=np.hstack((eig_pairs[0][1].reshape(45,1),eig_pairs[1][1].reshape(45,1)))


# In[35]:





# In[36]:


p=matrix.T.dot(c).T
p.shape


# In[37]:


df1=pd.DataFrame(data=p,columns=['pc1','pc2'])
df1['cat']=y_test


# In[61]:


import matplotlib.pyplot as plt



#ax2 = df1.plot.scatter(x='pc1',y='pc2',c='cat',colormap='viridis')


# In[43]:


import sklearn.manifold
tsne=sklearn.manifold.TSNE(n_components=2,random_state=0)


# In[56]:


compressed = tsne.fit_transform(x_test)


# In[57]:





df2=pd.DataFrame(data=compressed,columns=['pc1','pc2'])
df2['cat']=y_test


# In[59]:


import matplotlib.pyplot as plt


#plt.scatter(np.arange(p.shape[0]),p[:,0])
plt.scatter(compressed[:,0],compressed[:,1],c=y_test)#, cmap=matplotlib.colors.ListedColormap(colors))
plt.show()






