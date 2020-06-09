#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np # library to handle data in a vectorized manner

import pandas as pd # library for data analsysis
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import json # library to handle JSON files

get_ipython().system("conda install -c conda-forge geopy --yes # uncomment this line if you haven't completed the Foursquare API lab")
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values

import requests # library to handle requests
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

# import k-means from clustering stage
from sklearn.cluster import KMeans

#!conda install -c conda-forge folium=0.5.0 --yes # uncomment this line if you haven't completed the Foursquare API lab
import folium # map rendering library
print('Libraries imported.')


# # Capstone Project - The Battle of the Neighborhoods 
# ### Applied Data Science Capstone by IBM/Coursera
# #### *M.HARINI*
# 
# #### Table of contents
# * Introduction
# * Data
# * Methodology
# * Analysis
# * Results and Discussion
# * Conclusion
# 
# ### *  Introduction
# This report analyses to find the best location and features for starting a Chinese Restaurant in Chennai,a city in Southern State of India. we will try finding neighborhood which is not crowded with many restaurants and also analyse features that are missing in other restaurants like analysing the number of restaurants with online delivery and neighborhoods with less delivery rating rate and  less average rating rate.
# #### * Data
# we will be using Foursquare location data to map the clustering of the restaurants in a neighborhood and also to get the top venues in each neighborhood like what is the top visited venues in each locality.
# we will also be using the Zomato Restaurant data to get the details of the restaurants in each neighborhood their average rating, price, latitude,longitude,location and etc. we will refine the dataset so that it contains only relevant information.

# In[4]:


df = pd.read_csv('https://raw.githubusercontent.com/haanjiankur/Capstone-Project---The-Battle-of-Neighborhoods/master/zomato.csv',encoding='ISO-8859-1')
df.head()


# In[5]:


df_india = df[df['Country Code'] == 1]
df_tn = df_india[df_india['City'] == 'Chennai']
df_tn.reset_index(drop=True, inplace=True)
df_tn.head()


# In[6]:


df_tn_1= df_tn[df_tn.Longitude !=0.000000][['Restaurant Name','Locality','Longitude','Latitude','Cuisines','Aggregate rating','Rating text','Votes','Average Cost for two','Has Online delivery']]


# In[7]:


df_tn_1.head()


# ### Methodology section
# In this report we have  compared the top restaurants in the city based on the locality and had found the locality with the least and highest number of restaurants and also those with the highest and least rating.we had also found the proportion of restaurants that has online delivery. and those with that the least and highest average rating.
# 

# ### visualise the neighborhoods with Chennai map

# In[8]:


address = 'Chennai'

geolocator = Nominatim(user_agent="tn_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Chennai are {}, {}.'.format(latitude, longitude))


# In[9]:


map_chennai = folium.Map(location=[latitude, longitude], zoom_start=11)

# add markers to map
for lat, lng, label in zip(df_tn_1['Latitude'], df_tn_1['Longitude'], df_tn_1['Restaurant Name']):
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_chennai)  
    
map_chennai


# Now lets group our data based on locality

# In[10]:


df_Loc =  df_tn_1.groupby('Locality').count()['Restaurant Name'].to_frame()
df_aggrating= df_tn_1.groupby('Locality')['Aggregate rating'].mean().to_frame()
df_Cuisines = df_tn_1.groupby(['Locality'])['Cuisines'].agg(', '.join).reset_index()
df_rating = df_tn_1.groupby(['Locality'])['Rating text'].unique().agg(', '.join).reset_index()
df_votes = df_tn_1.groupby(['Locality'])['Votes'].sum().to_frame()
df_Lat = df_tn_1.groupby('Locality').mean()['Latitude'].to_frame()
df_Lng = df_tn_1.groupby('Locality').mean()['Longitude'].to_frame()
df_final = pd.merge(df_Lat,df_Lng,on='Locality').merge(df_Loc, on='Locality').merge(df_Cuisines, on='Locality').merge(df_aggrating,on ='Locality').merge(df_rating, on ='Locality').merge(df_votes, on ='Locality')


# In[11]:


df_final.rename(columns = {'Restaurant Name':'No.of.Restaurants'},inplace = True)
df_final.head()


# ### Foursquare location API

# In[12]:


CLIENT_ID = 'PWHATOAYCCHTD4T2DDO3230HXN4BX30OJX5UVZE1JOJD3FJH' 
CLIENT_SECRET = 'MHZUDHKDFWPB3ZPRVF30MGLUNQZY2ALDZVYTJQDOYY2XGMMJ'
VERSION = '20200609' 

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)


# Now lets get top venues around each locality

# In[13]:


def getNearbyVenues(names, latitudes, longitudes, radius=500,LIMIT = 100):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng,
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


# In[14]:


chennai_venues = getNearbyVenues(names=df_final['Locality'],
                                   latitudes=df_final['Latitude'],
                                   longitudes=df_final['Longitude']
                                  )


# In[15]:


print(chennai_venues.shape)
chennai_venues.head()


# In[16]:


chennai_venues.groupby('Neighborhood').count()


# In[17]:


print('There are {} uniques categories.'.format(len(chennai_venues['Venue Category'].unique())))


# #### one hot encoding

# In[18]:


# one hot encodin
chennai_onehot = pd.get_dummies(chennai_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
chennai_onehot['Neighborhood'] = chennai_venues['Neighborhood'] 

# move neighborhood column to the first column
fixed_columns = [chennai_onehot.columns[-1]] + list(chennai_onehot.columns[:-1])
chennai_onehot = chennai_onehot[fixed_columns]

chennai_onehot.head()


# In[19]:


chennai_grouped = chennai_onehot.groupby('Neighborhood').mean().reset_index()
chennai_grouped


# ##### frequency of top venues

# In[20]:


num_top_venues = 5

for hood in chennai_grouped['Neighborhood']:
    print("----"+hood+"----")
    temp = chennai_grouped[chennai_grouped['Neighborhood'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')


# #### define a function to return the most common venues

# In[21]:


def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# In[22]:


num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = chennai_grouped['Neighborhood']

for ind in np.arange(chennai_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(chennai_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()


# ### Analysis

# #### analyse each neighborhood by forming clusters

# In[23]:


kclusters = 5
chennai_grouped_clustering = chennai_grouped.drop('Neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(chennai_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10] 


# In[25]:


df_tn_1 =df_tn_1.rename(columns = {'Locality':'Neighborhood'})


# In[26]:


df_tn_1.head()


# In[29]:


neighborhoods_venues_sorted.insert(0, 'Cluster Labels',kmeans.labels_)
chennai_merged = df_tn_1

# # merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
chennai_merged = chennai_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighborhood')

chennai_merged.dropna(inplace=True)
chennai_merged.head()


# #### mapping the venues

# In[30]:


map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(chennai_merged['Latitude'], chennai_merged['Longitude'], chennai_merged['Neighborhood'], chennai_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# #### examining clusters

# In[31]:


cluster_1 = chennai_merged.loc[chennai_merged['Cluster Labels'] == 0, chennai_merged.columns[[1] + list(range(5, chennai_merged.shape[1]))]]
cluster_1


# In[32]:


cluster_2 = chennai_merged.loc[chennai_merged['Cluster Labels'] == 1, chennai_merged.columns[[1] + list(range(5, chennai_merged.shape[1]))]]
cluster_2


# In[33]:


cluster_3 = chennai_merged.loc[chennai_merged['Cluster Labels'] == 2, chennai_merged.columns[[1] + list(range(5, chennai_merged.shape[1]))]]
cluster_3


# In[34]:


cluster_4 = chennai_merged.loc[chennai_merged['Cluster Labels'] == 3, chennai_merged.columns[[1] + list(range(5, chennai_merged.shape[1]))]]
cluster_4


# In[35]:


cluster_5 = chennai_merged.loc[chennai_merged['Cluster Labels'] == 4, chennai_merged.columns[[1] + list(range(5, chennai_merged.shape[1]))]]
cluster_5


# #### Analyse by plotting 

# In[36]:


import matplotlib.pyplot as plt
plt.figure(figsize=(9,5), dpi = 100)
# title
plt.title('The highest number of Restaurant available in Chennai')
#On x-axis

#giving a bar plot
df_tn_1.groupby('Neighborhood')['Restaurant Name'].count().nlargest().plot(kind='bar')

plt.xlabel('Resturant Locality in Chennai')
#On y-axis
plt.ylabel('Number of Restaurant')

#displays the plot
plt.show()


# In[44]:


plt.figure(figsize=(9,5), dpi = 100)
# title
plt.title('The lowest number of Restaurant available in Locality of Chennai')
#On x-axis

#giving a bar plot
df_tn_1.groupby('Neighborhood')['Restaurant Name'].count().nsmallest().plot(kind='bar')

plt.xlabel('Resturant Locality in Chennai')
#On y-axis
plt.ylabel('Number of Restaurant')

#displays the plot
plt.show()


# In[38]:


plt.figure(figsize=(9,5), dpi = 100)
# title
plt.title('The lowest number of Chinese Restaurant available in Locality of Chennai')
#On x-axis

#giving a bar plot
chennai_onehot.groupby('Neighborhood')['Chinese Restaurant'].count().nsmallest().plot(kind='bar')

plt.xlabel('Resturant Locality in Chennai')
#On y-axis
plt.ylabel('Number of Chinese Restaurant')

#displays the plot
plt.show()


# In[46]:


plt.figure(figsize=(9,5), dpi = 100)
# title
plt.title('The total number of Chinese Restaurant available in Locality of Chennai')
#On x-axis

#giving a bar plot
chennai_onehot.groupby('Neighborhood')['Chinese Restaurant'].count().plot(kind='bar')

plt.xlabel('Resturant Locality in Chennai')
#On y-axis
plt.ylabel('Number of Chinese Restaurant')
plt.ylim(0,50)

#displays the plot
plt.show()


# In[53]:


plt.figure(figsize=(9,5), dpi = 100)
# title
plt.title('The  rating of restaurants available in Locality of Chennai')
#On x-axis

#giving a bar plot
df_tn_1.groupby('Neighborhood')['Aggregate rating'].count().plot(kind='bar')

plt.xlabel('Resturant Locality in Chennai')
#On y-axis
plt.ylabel('Number of  Restaurant')
plt.ylim(0,5)

#displays the plot
plt.show()


# In[57]:


plt.figure(figsize=(9,5), dpi = 100)
# title
plt.title('The total number of  Restaurant with Online delivery available in Locality of Chennai')
#On x-axis

#giving a bar plot
df_tn_1.groupby('Neighborhood')['Has Online delivery'].count().plot(kind='bar')

plt.xlabel('Resturant Locality in Chennai')
#On y-axis
plt.ylabel('Number of  Restaurant')
plt.ylim(0,10)

#displays the plot
plt.show()


# #### Conclusion
# If the restaurant is located in Adyar,Ashok Nagar ,Perungudi and Santhome,the owner need to take a chance. But if it is located in Kotturpuram the risk is less.since,the competition is less and also rating of surrounding restaurant is also less.
# 
# 

# In[ ]:




