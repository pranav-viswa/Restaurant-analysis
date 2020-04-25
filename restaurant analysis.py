#!/usr/bin/env python
# coding: utf-8

# <h1>Zomato Restaurants Analysis in Banglore</h1>
# <h4>The basic idea of analysing the Zomato dataset is to get a fair idea about the factors affecting the aggregate rating of each restaurant, establishment of different types of restaurant at different places, Bengaluru being one such city has more than 12,000 restaurants with restaurants serving dishes from all over the world. With each day new restaurants opening the industry hasn’t been saturated yet and the demand is increasing day by day. In spite of increasing demand it however has become difficult for new restaurants to compete with established restaurants. Most of them serving the same food. Bengaluru being an IT capital of India. Most of the people here are dependent mainly on the restaurant food as they don't have time to cook for themselves. With such an overwhelming demand of restaurants it has therefore become important to study the demography of a location </h4>

# <h5>IMPORTING LIBRARIES</h5>

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# <h5>IMPORTING DATASET</h5>

# In[2]:


source='C:\\Users\\user\\Desktop\\zomato.xlsx'


# In[3]:


df=pd.read_excel(source)


# In[4]:


df.head()


# In[5]:


df.tail()


# <h5>Data Information and Data Description</h5>

# In[6]:


df.info()


# There are 41373 observation in 17 columns.

# In[7]:


df.columns


# This shows the list of columns in the given dataset

# In[8]:


df.shape


# it has 41373 rows and 17 columns

# In[9]:


df.describe()


# Note:-
# 
# 1)as min value is 0, this means that we have restaurent with no votes.
# 2)we have a restaurent with highest votes count is 16832.

# <h3>MISSING VALUES </h3>

# In[10]:


df.isna().sum()


# Note:-
# 
# The Column that has the most number of missing values is the Dish Liked column followed by rates
# After Studying the data we can clearly delete the following columns as the make are not useful for our analysis "url", 'address','phone','listed_in(city)'

# In[11]:


df.drop(columns=["url", 'address','phone','listed_in(city)'], inplace  =True)


# <h3>Renaming the Approximate cost for two column for easier access</h3>

# In[12]:


df.rename(columns={'approx_cost(for two people)': 'average_cost'}, inplace=True)


# <h3>Preprocessing and visualizations</h3>

# <h3>Name</h3>

# In[13]:


df.name.value_counts().head()


# In[14]:


plt.figure(figsize = (12,6))
ax =df.name.value_counts()[:20].plot(kind = 'bar')
ax.legend(['* Restaurants'])
plt.xlabel("Name of Restaurant")
plt.ylabel("Count of Restaurants")
plt.title("Name vs Number of Restaurant",fontsize =20, weight = 'bold')


# Note:-
# Cafe Coffee Day has more number of Restaurents.

# <h3>ONLINE ORDERS</h3>

# <h4>Restaurants accepting online orders</h4>

# In[15]:


df.online_order.value_counts()


# In[16]:


ax= sns.countplot(df['online_order'])
plt.title('Number of Restaurants accepting online orders', weight='bold')
plt.xlabel('online orders')


# note :
# most of the orders are online

# <h4>Restaurants having the option of booking table</h4>
# 

# In[17]:


df['book_table'].value_counts()


# In[18]:


sns.countplot(df['book_table'], palette= "Set1")
plt.title("No of Restaurant with Book Table Facility", weight = 'bold')
plt.xlabel('Book table facility')
plt.ylabel('No of restaurants')


# note: 36231 restaurents have No book table feature.

# <h4>RESTAURANTS BASED LOCATION </h4>

# In[19]:


df['location'].value_counts()


# In[20]:


plt.figure(figsize=(12,6)) 
df['location'].value_counts()[:10].plot(kind = 'pie')
plt.title('Location', weight = 'bold')


# In[21]:


plt.figure(figsize = (12,6))
names = df['location'].value_counts()[:10].index
values = df['location'].value_counts()[:10].values
colors = ['gold', 'red', 'lightcoral', 'lightskyblue','blue','green','silver']
explode = (0.1, 0, 0, 0,0,0,0,0,0,0)  # explode 1st slice

plt.pie(values, explode=explode, labels=names, colors=colors,autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.title("Percentage of restaurants present in that location", weight = 'bold')
plt.show()


# <h4>Location Vs Count
# 
# </h4>
# 

# In[22]:


plt.figure(figsize = (12,6))
df['location'].value_counts()[:10].plot(kind = 'bar', color = 'g')
plt.title("Location vs Count", weight = 'bold')


# In[23]:


df['location'].nunique()


# Note:-
# So we have 93 Neighbourhoods in Bangalore.
# We have the highest no of restaurants in BTM.
# We have the least is in Electronic city.

# <h4>Restaurant type
# 
# </h4>
# 

# In[24]:


df['rest_type'].value_counts().head(10)


# In[25]:


plt.figure(figsize = (14,8))
df.rest_type.value_counts()[:15].plot(kind = 'pie')
plt.title('Restaurent Type', weight = 'bold')
plt.show()


# In[26]:


colors = ['#800080','red','#00FFFF','#FFFF00','#00FF00','#FF00FF']


# In[27]:


plt.figure(figsize = (12,6))
names = df['rest_type'].value_counts()[:6].index
values = df['rest_type'].value_counts()[:6].values
explode = (0.1, 0.1, 0.1, 0.1,0.1,0.1)  # explode 1st slice

plt.title('Type of restaurant in percentage', weight = 'bold')
plt.pie(values, explode=explode, labels=names, colors=colors,autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.show()


# Note:-
# 
# We have the most frequently visited Restaurant type - Quickbites

# <h4>Note:-
# 
# We have the most frequently visited Restaurant type - Quickbites</h4>

# <h4>Average Cost</h4>

# In[28]:


df['average_cost'].value_counts()[:20]


# In[29]:


plt.figure(figsize = (12,8))
df['average_cost'].value_counts()[:20].plot(kind = 'pie')
plt.title('Avg cost in Restaurent for 2 people', weight = 'bold')
plt.show()


# In[30]:


colors  = ("red", "green", "orange", "cyan", "brown", "grey", "blue", "indigo", "beige", "yellow")


# In[31]:


fig= plt.figure(figsize=(18, 9))
explode = (0.1, 0, 0, 0,0,0,0,0,0,0) 

delplot = df['average_cost'].value_counts()[:10].plot(kind = 'pie',autopct='%1.1f%%',fontsize=20,shadow=True,explode = explode,colors = colors)

#draw circle
centre_circle = plt.Circle((0,0),0.80,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

plt.title("Average cost for 2 people in Rupees",fontsize = 15,weight = 'bold')


# Note:- The Average cost for two persons in Banglore is 300rupees¶

# In[32]:


#lets delete the nulll values
dishes_data = df[df.dish_liked.notnull()]
dishes_data.dish_liked = dishes_data.dish_liked.apply(lambda x:x.lower().strip())


# In[33]:


dishes_data.isnull().sum()


# In[34]:


# count each dish to see how many times each dish repeated
dish_count = []
for i in dishes_data.dish_liked:
    for t in i.split(','):
        t = t.strip() # remove the white spaces to get accurate results
        dish_count.append(t)


# In[35]:


plt.figure(figsize=(12,6)) 
pd.Series(dish_count).value_counts()[:10].plot(kind='bar',color= 'c')
plt.title('Top 10 dished_liked in Bangalore',weight='bold')
plt.xlabel('Dish')
plt.ylabel('Count')


# Note:-
# The Most liked dish in Banglore is Pasta

# <h4>Rate</h4>

# In[36]:


df['rates'] = df['rates'].replace('NEW',np.NaN)
df['rates'] = df['rates'].replace('-',np.NaN)
df.dropna(how = 'any', inplace = True)


# In[37]:


df['rates'] = df.loc[:,'rates'].replace('[ ]','',regex = True)
df['rates'] = df['rates'].astype(str)
df['rates'] = df['rates'].apply(lambda r: r.replace('/5',''))
df['rates'] = df['rates'].apply(lambda r: float(r))


# In[38]:


df.rates.hist(color='grey')
plt.axvline(x= df['rates'].mean(),ls='--',color='yellow')
plt.title('Average Rating for Bangalore Restaurants',weight='bold')
plt.xlabel('Rating')
plt.ylabel('No of Restaurants')
print(df['rates'].mean())


# <h4>The Average rating per restaurant in Banglore is found to be 3.9</h4>

# <h4>Cuisines</h4>

# In[39]:


#lets delete the nulll values
cuisines_data = df[df.cuisines.notnull()]
cuisines_data.cuisines = cuisines_data.cuisines.apply(lambda x:x.lower().strip())


# In[40]:


cuisines_count= []
for i in cuisines_data.cuisines:
    for j in i.split(','):
        j = j.strip()
        cuisines_count.append(j)


# In[41]:


plt.figure(figsize=(12,6)) 
pd.Series(cuisines_count).value_counts()[:10].plot(kind='bar',color= 'r')
plt.title('Top 10 cuisines in Bangalore',weight='bold')
plt.xlabel('cuisines type')
plt.ylabel('No of restaurants')


# <h4>The Most liked Cuisine in Banglore is North Indian</h4>

# <h4>Rate vs Online Order</h4>

# In[42]:


plt.figure(figsize = (12,6))
sns.countplot(x=df['rates'], hue = df['online_order'])
plt.ylabel("Restaurants that Accept/Not Accepting online orders")
plt.title("rate vs oline order",weight = 'bold')


# <h1>INFERENCES</h1>

# <li>24330 restaurants are accepting online orders
# <li>36231 restaurants do not have the book table feature. #### Restaurants:-
# <li>So we have 93 locations where the restaurants can be accessed through zomato in Bangalore.
# <li>We have highest number of restaurants in BTM.
# <li>The average cost is rs 300

# </h3> The most preferred cuisine is <b>north indian<b></h3>

# </h3> The most preferred dish is <b>Pasta<b></h3>

# <h1>INFERENCES</h1>

# In[43]:


df['online_order']= pd.get_dummies(df.online_order, drop_first=True)
df['book_table']= pd.get_dummies(df.book_table, drop_first=True)
df.head()


# In[44]:


df.drop(columns=['dish_liked','reviews_list','menu_item','listed_in(type)'], inplace  =True)


# In[45]:


df['rest_type'] = df['rest_type'].str.replace(',' , '') 
df['rest_type'] = df['rest_type'].astype(str).apply(lambda x: ' '.join(sorted(x.split())))
df['rest_type'].value_counts().head()


# In[46]:


df['cuisines'] = df['cuisines'].str.replace(',' , '') 
df['cuisines'] = df['cuisines'].astype(str).apply(lambda x: ' '.join(sorted(x.split())))
df['cuisines'].value_counts().head()


# <h1>Label Encoding</h1>

# In[47]:


from sklearn.preprocessing import LabelEncoder
T = LabelEncoder()                 
df['location'] = T.fit_transform(df['location'])
df['rest_type'] = T.fit_transform(df['rest_type'])
df['cuisines'] = T.fit_transform(df['cuisines'])


# In[48]:


df["average_cost"] = df["average_cost"].str.replace(',' , '') 
df["average_cost"] = df["average_cost"].astype('float')
df.head()


# In[49]:


x = df.drop(['rates','name'],axis = 1)
y = df['rates']


# In[50]:


x.shape


# In[51]:


y.shape


# <h1>Splitting the data for Model Building</h1>

# In[52]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 33)


# <h1>Standardization of values</h1>

# In[53]:


#standarizing
#taking numeric values
from sklearn.preprocessing import StandardScaler
num_values1=df.select_dtypes(['float64','int64']).columns
scaler = StandardScaler()
scaler.fit(df[num_values1])
df[num_values1]=scaler.transform(df[num_values1])


# In[54]:


df.head()


# <h1>MODEL 1-LINEAR REGRESSION</h1>

# In[63]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)
y_pred_lr = lr.predict(X_test)


# In[64]:


lr.score(X_test, y_test)*100


# <h1>MODEL 2-RANDOM FOREST</h1>

# In[65]:


from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()
rfr.fit(X_train,y_train)
y_pred_rfr = rfr.predict(X_test)


# In[66]:


rfr.score(X_test,y_test)*100


# <h1>MODEL 3-RIDGE REGRESSION</h1>

# In[67]:


from sklearn.linear_model import Ridge
rdg = Ridge()
rdg.fit(X_train,y_train)
y_pred_rdg = rdg.predict(X_test)


# In[68]:


rdg.score(X_test,y_test)*100


# <h1>INSIGHTS FROM ABOVE MODEL</h1>
# <table>
#     <tr>
#     <th>MODEL</th>
#     <th>ACCURACY</th>
#    </tr>
#     <tr>
#     <td>LINEAR REGRESSION</td>
#     <td>20.36 %</td>
#    </tr>
#     <tr>
#     <td>RANDOM FOREST</td>
#     <td>87.01 %</td>
#    </tr>
#     <tr>
#     <td>RIDGE REGRESSION</td>
#     <td>20.36 %</td>
#    </tr>

# <h1>From the above table we see that the Random forest is performing better </h1>
# <h1>Prediction for Random forest Regressor</h1>

# In[71]:


Randpred = pd.DataFrame({ "actual": y_test, "pred": y_pred_rfr })
Randpred


# In[ ]:




