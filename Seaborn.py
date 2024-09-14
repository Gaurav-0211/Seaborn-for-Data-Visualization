#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set_style('whitegrid')
sns.set_context('notebook')
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")


# In[5]:


from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected = True)
import cufflinks as cf
cf.go_offline()


# In[6]:


df = pd.read_csv(r"tips.csv")
df.head()


# In[7]:


df.plot()


# In[8]:


df.iplot()


# In[9]:


df.plot.area(alpha=0.5)


# In[12]:


sns.countplot(x=df.time, order = df.time.value_counts().index, hue = df.sex)
sns.despine()


# In[13]:


data = df.sex.value_counts()
data


# In[14]:


plt.bar(data.index, data, width=0.5, color='red')


# In[15]:


fig = px.histogram(df.time, color= df.sex, barmode = 'group')
fig.update_layout(title_text = 'CountPlot', width=400, height = 300)
fig.update_xaxes(title_text = 'Time')
fig.update_yaxes(title_text = 'Count')


# In[16]:


plt.figure(figsize=(12,8))
sns.distplot(df.total_bill, hist = True, kde =True,rug=True,bins=30)
sns.despine()


# In[18]:


df.total_bill.hist(bins=30, alpha = 0.7)
df.total_bill.plot.kde(lw=3, ls='--')
df.plot.kde()


# In[19]:


print(plt.hist(df['total_bill'], bins=30)[0])  #Counts
print(plt.hist(df['total_bill'], bins =30)[1]) #bins range


# In[20]:


df.total_bill.iplot(kind='hist', bins=30)


# In[21]:


fig = px.histogram(df.total_bill, marginal="box", title = "Year Count Plot")
fig.update_layout(width=800, height = 550)
fig.update_xaxes(title_text = 'Total_bill')
fig.update_yaxes(title_text = 'Count')


# In[22]:


data = df.sex.value_counts()
data


# In[23]:


plt.pie(data, labels= data.index, startangle = 90, autopct = '%1.2f%%')
plt.show()


# In[24]:


fig = px.pie(values=data, names=data.index)
fig.update_layout(title_text = 'Gender Distribution', width=500, height = 400)


# In[25]:


df = pd.DataFrame({'x':np.linspace(0,5,11), 'y': np.linspace(0,5,11)**2})
df.head()


# In[28]:


sns.lineplot(x='x', y='y', data=df, marker='o', label='y')
plt.show()


# In[31]:


plt.plot(df['x'], df['y'], 'r-o', lw=2, alpha=0.9, label='y')
plt.legend()
plt.show()


# # Jointplot : {scatter,hex,reg}

# In[34]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set_style('whitegrid')
sns.set_context('notebook')
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")


# In[36]:


df = pd.read_csv(r"tips.csv")
df.head()


# In[38]:


sns.jointplot(x='total_bill', y='tip', data=df, kind='scatter', hue='sex', height=10, ratio=5, joint_kws={'s': 200})


# In[46]:


plt.figure(figsize=(10,8),dpi =70)
sns.scatterplot(x='total_bill', y='tip', data=df, color='b', hue='sex', size='size', sizes=(100, 500))


# In[48]:


plt.figure(figsize = (8,6), dpi=70)
plt.subplot(1,2,1)
sns.scatterplot(x='total_bill', y='tip', data=df, s=100, color='b')
plt.subplot(1,2,2)
sns.scatterplot(x='tip', y='total_bill',data = df, s= 100, color ='r')


# In[49]:


df.iplot(kind='scatter', x= 'total_bill', y='tip', mode = 'markers', size=10)


# In[50]:


fig = px.scatter(x=df['total_bill'], y=df['tip'], title="Total bill vs Tip", color=df['sex'], size_max=20)

fig.update_layout(width=1200, height=500)
fig.update_xaxes(title_text='Total_bill')
fig.update_yaxes(title_text='Tip')

fig.show()


# In[51]:


plt.figure(figsize=(8,6), dpi=70)
df.plot.scatter(x='total_bill',y='tip', c= 'size',s = df.size/20, cmap='coolwarm')
df.plot.scatter(x = 'total_bill',y = 'tip', c='red',s =50, figsize=(6,6))


# In[52]:


df.plot.hexbin(x='total_bill', y='tip', gridsize=20)


# In[53]:


df.iplot(kind = 'bubble',x='total_bill',y='tip', size='size')


# # Regression Plot

# In[54]:


sns.lmplot(data = df, x = 'total_bill', y= 'tip', hue='sex',markers = ['o','+'],height=8, aspect = 2)


# In[55]:


sns.lmplot(data=df, x='total_bill', y='tip', hue='sex',col='sex', height=8)


# In[57]:


sns.lmplot(data=df, x ='total_bill', y='tip', hue='sex', col='sex',row='time',height=8)


# # Categorical vs Numerical

# In[61]:


sns.barplot(x='sex', y='total_bill', data=df, estimator=np.sum, ci=False, color='blue')
plt.xticks(rotation=45, fontsize=14)


# In[62]:


df.iplot(kind ='bar', x = 'sex', y='total_bill')


# In[63]:


df.mean().iplot(kind='bar')


# # Box & Wiskers plot (Describe plot)

# In[64]:


sns.boxplot(x='day',y='total_bill',data=df, hue='smoker')
sns.despine()


# In[68]:


fig = px.box(x=df.day, y=df.total_bill, color=df.smoker)
fig.update_layout(title_text = 'BoxPlot',height = 500)
fig.update_xaxes(title_text='Day')
fig.update_yaxes(title_text='Total Bill')


# In[69]:


pd.DataFrame(df.describe()).T


# In[70]:


df.total_bill.plot.box()
df.plot.box()


# In[71]:


df.iplot(kind='box')


# In[ ]:


# Violine Plot + Swarm Plot


# In[73]:


sns.violinplot(x='day', y='total_bill', hue='smoker', data=df, split=False)


# In[77]:


sns.violinplot(x=df.day,y=df.total_bill)
sns.swarmplot(x= df.day, y = df.total_bill,color='black',size = 3)


# In[78]:


# Heatmap


# In[79]:


sns.heatmap(df.corr(), annot=True, cmap= 'coolwarm')


# In[80]:


df2 = pd.read_csv(r"flights.csv")
df2.head(3)


# In[83]:


row=['month']
col = ['year']
value = ['passengers']
aggfun = ['sum']
sns.heatmap(df2.pivot_table(value, row, col, aggfun), annot = False, cmap = 'viridis', lw=1, linecolor = 'white')


# In[84]:


row=['month']
col = ['year']
value = ['passengers']
aggfun = ['sum']
sns.clustermap(df2.pivot_table(value, row, col, aggfun), annot = False, cmap = 'Greys', lw=1, linecolor = 'white',standard_scale=1)


# In[85]:


#Pairplot (DatFrame plot)


# In[87]:


sns.pairplot(df, hue='sex', height=2.4, aspect=2)


# In[88]:


df[['total_bill', 'tip', 'size']].scatter_matrix(size=3)

