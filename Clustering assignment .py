#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns


# In[17]:


df = pd.read_excel(r"C:\Users\J G TECH\AppData\Local\Temp\ca5e0f05-be3b-4f8f-8da3-5715a4bcd9e4_Clustering (3).zip.9e4\Clustering\EastWestAirlines.xlsx")
print(df)


# In[18]:


df = df.select_dtypes(include=[np.number])


# In[19]:


df.fillna(df.mean(), inplace=True)


# In[20]:


scaler = StandardScaler()
scaled_df = scaler.fit_transform(df)


# In[ ]:





# In[16]:


sns.pairplot(pd.DataFrame(scaled_df, columns=df.columns))
plt.show()


# In[ ]:


kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(scaled_df)


# In[ ]:


kmeans_silhouette = silhouette_score(scaled_df, kmeans_labels)
print(f"K-Means Silhouette Score: {kmeans_silhouette}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




