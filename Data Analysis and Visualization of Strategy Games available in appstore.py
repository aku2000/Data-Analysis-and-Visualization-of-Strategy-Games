#!/usr/bin/env python
# coding: utf-8

# -Developed By: Akansha Agarwal
# <br>
# akurox123@gmail.com

# # **<u>Introduction:</u>**
# 
# <font color=red>**The global mobile gaming market could grow from 56 dollars billion in revenue in 2017 to 106.4 dollars billion in 2021, according to a recent report by Newzoo and Arm**</font>
# 
# 
# 
# ![1.png](attachment:1.png)
# 
# 
# 
# **The mobile games industry is worth billions of dollars, with companies spending vast amounts of money on the development and marketing of these games to an equally large market. Using this data set, insights can be gained into a sub-market of this market, strategy games. This sub-market includes titles such as Clash of Clans, Plants vs Zombies and Pokemon GO.**
# 

# # <u>Objective:</u>
# 
# ###  **To expose the best combination for strategy games available in the appstore in order to get a good user rating (4.0/5.0 and above).**
# 
# ***
# 
# ***

# # <u><font color= deeppink > Importing packages and collecting data </font></u> 

# In[26]:


'''Ignore deprecation and future, and user warnings.'''
import warnings as wrn
wrn.filterwarnings('ignore', category = DeprecationWarning) 
wrn.filterwarnings('ignore', category = FutureWarning) 
wrn.filterwarnings('ignore', category = UserWarning) 

'''Import basic modules.'''
import pandas as pd
import numpy as np
from scipy import stats

'''Customize visualization
Seaborn and matplotlib visualization.'''
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
get_ipython().run_line_magic('matplotlib', 'inline')

'''Special Visualization'''
import missingno as msno

'''Plotly visualization .'''
import plotly.offline as py
from plotly.offline import iplot, init_notebook_mode
import plotly.graph_objs as go
py.init_notebook_mode(connected = True) # Required to use plotly offline in jupyter notebook

import cufflinks as cf #importing plotly and cufflinks in offline mode  
import plotly.offline  
cf.go_offline()  
cf.set_config_file(offline=False, world_readable=True)

'''Display markdown formatted output like bold, italic bold etc.'''
from IPython.display import Markdown
def bold(string):
    display(Markdown(string))

'''Reading the data from csv files'''
data = pd.read_csv(r'C:\Users\Lenovo\Desktop\ML\minor project-strategy games analysis/appstore_games.csv', header = None)

'''Replacing header with top row'''
data.columns = data.iloc[0]
data = data[1:]

display(data.head(5))
print('Dimension of data:', data.shape)


# In[27]:


'''Determing the total count of null values, of each column of the dataset.'''

data.isnull().sum()


# In[28]:


'''Data Visualization of missing values in dataset'''
msno.bar(data) 


# ### Above bar chart depicts that <font color=green>Subtitle, Average User Rating, User Rating Count and In-app Purchases</font> contain the most number of missing values.
# <br>

# # <u><font color= deeppink > DATA CLEANING:</font></u>
# 
# ###  -> Games without User Rating are dropped.
# ###  -> Games with less than 200 user rating and days since last update date is less than6month are dropped to prevent biased ratings from the developer.
# ###  -> Genre tags "Entertainment" and "Games" are removed from the Genre string as it does not provide meaningful insight because every game has these tags.
# ###  -> The remaining string is checked and grouped as follows:
# #### 1)Puzzle= Puzzle/Board
# #### 2)Adventure= Adventure/Role/Role Playing
# #### 3)Action = Action
# #### 4)Family = Family/Education

# In[29]:


dataf = pd.read_csv(r'C:\Users\Lenovo\Desktop\ML\minor project-strategy games analysis/appstore_games.csv')
#dataf=data.copy()

# Drop unused rows and columns
dataf = dataf.drop(columns="URL")

dataf.columns= ['ID', 'Name','Subtitle','Icon URL','User_Rating','User_Rating_Count','Price','In_App_Purchase','Desc','Dev','Age_Rating','Languages','Size','Primary_Genre','Genre','Release_Date','Current_Version_Date']


#Drop games that has no rating
dataf = dataf[pd.notnull(dataf['User_Rating'])]

# Converting Date strings to datetime objects
import datetime

dataf['Release_Date'] = pd.to_datetime(dataf['Release_Date'])
dataf['Current_Version_Date'] = pd.to_datetime(dataf['Current_Version_Date'])

#New column for time gap between release & update
dataf['Update_Gap']=dataf.Current_Version_Date-dataf.Release_Date
# To prevent biased ratings,Drop games that has rating less than 200 user rating count AND 
#(Release Date to Update date) less than 6 months

Low_User_Count=(dataf[dataf.User_Rating_Count < 200].index) &dataf[dataf.Update_Gap < datetime.timedelta(days=175)].index

dataf.drop(Low_User_Count , inplace=True)


# In[30]:


datafg = dataf.copy()
datafg['Genre'] = datafg['Genre'].str.replace(',', '').str.replace('Games', '').str.replace('Entertainment', '').str.replace('Strategy', '') 
datafg['Genre'] = datafg['Genre'].str.split(' ').map(lambda x: ' '.join(sorted(x)))
datafg['Genre']= datafg['Genre'].str.strip()
Non_Main_Genre=datafg[~datafg.Genre.str.contains('Puzzle') &                            ~datafg.Genre.str.contains('Action') &                            ~datafg.Genre.str.contains('Family')&                            ~datafg.Genre.str.contains('Education')&                            ~datafg.Genre.str.contains('Family')&                            ~datafg.Genre.str.contains('Adventure')&                           ~datafg.Genre.str.contains('Board')&                           ~datafg.Genre.str.contains('Role')].index
datafg.drop(Non_Main_Genre , inplace=True)
datafg.loc[datafg['Genre'].str.contains('Puzzle'),'Genre'] = 'Puzzle'
datafg.loc[datafg['Genre'].str.contains('Board'),'Genre'] = 'Puzzle'
datafg.loc[datafg['Genre'].str.contains('Action'),'Genre'] = 'Action'
datafg.loc[datafg['Genre'].str.contains('Adventure'),'Genre'] = 'Adventure'
datafg.loc[datafg['Genre'].str.contains('Role'),'Genre'] = 'Adventure'
datafg.loc[datafg['Genre'].str.contains('Family'),'Genre'] = 'Family'
datafg.loc[datafg['Genre'].str.contains('Education'),'Genre'] = 'Family'


# In[31]:


sns.set_style('darkgrid')
f, axes = plt.subplots (2,1, figsize=(8,8))

#Histogram
x=['Puzzle','Action','Adventure','Family']
y = [datafg.Genre[(datafg['Genre']=='Puzzle')].count(),datafg.Genre[(datafg['Genre']=='Action')].count(),     datafg.Genre[(datafg['Genre']=='Adventure')].count(),datafg.Genre[(datafg['Genre']=='Family')].count()]

vis1= sns.barplot(x,y,palette='Accent',ax=axes[0])
vis1.set(xlabel='Genre',ylabel='Number of Games')
for p in vis1.patches:
             vis1.annotate("%.f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                 ha='center', va='center', fontsize=11, color='gray', xytext=(0, 20),
                 textcoords='offset points')


#Pie Chart
NG = [datafg.Genre[(datafg['Genre']=='Puzzle')].count(),datafg.Genre[(datafg['Genre']=='Action')].count(),     datafg.Genre[(datafg['Genre']=='Adventure')].count(),datafg.Genre[(datafg['Genre']=='Family')].count()]
G = ['Puzzle','Action','Adventure','Family']

plt.pie(NG, labels=G, startangle=90, autopct='%.1f%%')
plt.show()


plt.ioff()


# # Simple Analysis on Genre distribution, we can see that number of games follows Puzzle > Adventure > Action > Family
# *** 
# ***
# 
# <br>
# 
# # <u>Identifying the <font color=lime>genres</font> which are most significant</u>

# In[32]:


bold('**MOST POPULAR GENRE**')
import squarify

data['Genreslist'] = data['Genres'].str.extract('([A-Z]\w{5,})', expand=True)
temp_df = data['Genreslist'].value_counts().reset_index()

sizes=np.array(temp_df['Genreslist'])
labels=temp_df['index']
colors = [plt.cm.Spectral(i/float(len(labels))) for i in range(len(labels))]
plt.figure(figsize=(12,8), dpi= 100)
squarify.plot(sizes=sizes, label=labels, color = colors, alpha=.5, edgecolor="yellow", linewidth=3, text_kwargs={'fontsize':12})
plt.title('Treemap of Strategy Game Genres', fontsize = 18)
plt.axis('on')
plt.show()


# ### Above treemap shows, <font color=orange>Strategy</font> is the most popular genre, followed by  <font color=orange>Entertainment and Simulation.</font>
# 
# ***
# 
# <br>
# 
# #  _<u><font color=red>Analysis of DISTRIBUTION according to GENRES</font></u>_
# 
# <BR>
#     
# ## <font color=cornflowerblue>1.MOST REVIEWED GENRE [Higher user rating]</font>
# 

# In[33]:


review = data.sort_values(by='User Rating Count', ascending=False)[['Name', 'Average User Rating','Genres', 'User Rating Count', ]].head(8)
review.iloc[:, 0:-1]


# ### <font color=darklimegreen>We see that in above analysis, the most reviewed games are: <font color=orange> *Epic War - Castle Alliance,"Papa's Hot Doggeria To Go!" , Cooking Madness-Kitchen Frenzy*.</font> Each of them have <u>Average User Rating of 4.5</u> and all of them have common genre - <font color=orange> STRATEGY.</font> </font>

# ## <font color=cornflowerblue>2.GENRE vs In-app Purchases and Price </font>

# In[34]:


review = data.sort_values(by=['Price','In-app Purchases'], ascending=False)[['Name','Genres','In-app Purchases','Price',]].head(8)
review.iloc[:, 0:-1]


# ### <font color=darklimegreen>We see that in the above analysis, <font color=orange>Ticket to Ride , Blood & Honor No Ads </font> has the most number of In-App Purchases made.The game falls under the following genres: <font color=orange>Board, Entertainment and Strategy</font>.</font>

# ## <font color=cornflowerblue>3.Genre vs Size </font>

# In[35]:


review = data.sort_values(by='Size', ascending=True)[['Name','Genres','Size','Genres']].head(8)
review.iloc[:, 0:-1]


# ***
# 
# ***
# 
# <br>
# 
# #  _<u><font color=red>*Identifying trends of user rating based on pricing.*</font></u>_
# 

# # <font color=cornflowerblue>1.Price</font>

# In[36]:


bold('****MOST OF THE APPS PRICES BETWEEN 0 TO 10 DOLLARS****')
plt.rcParams['figure.figsize'] = (18, 10)
ax = sns.kdeplot(data['Price'], shade = True, linewidth = 5, color = 'k')
ax.set_ylabel('Count', fontsize = 20)
ax.set_xlabel('Price', fontsize = 20)
plt.show()


# ### <font color=darklimegreen>MOST OF THE APPS PRICES BETWEEN<font color=orange> 0 TO 10 DOLLARS </font></font>
# 
# 
# # <font color=cornflowerblue>2.Price vs Average User Rating</font>

# In[37]:


import plotly.graph_objects as go
fig = go.Figure(data=go.Scatter(                
                x=data['Average User Rating'],
                y=data['Price'],
                mode='markers'))

fig.update_yaxes(range=[0, 20])

fig.update_layout(title_text='Price vs Average User Rating',
    xaxis_title_text='Average User Rating',
    yaxis_title_text='Price in $',)
fig.show()


# ### <font color=darklimegreen>We see that in the above analysis,upto <font color=orange> $8.99 </font> is spent by users giving an Average User Rating of <font color=orange>4.5</font></font>
# 
# ### Most of the apps are rated good with rating 3.5 - 4.5
# 
# *** 
# 
# <br>
# 
# # _<font color=darkblue><u>*Some more TRENDS:*</u></font>_
# 
# 
# # <font color=cornflowerblue>1.Developers</font>

# In[48]:


plt.rcParams['figure.figsize'] = (12, 5)
data.Developer.value_counts()[:20].plot(kind='bar',color = 'lime', alpha =1, linewidth=4, edgecolor='darkblue')
plt.xlabel("Developers", fontsize=18)
plt.ylabel("Count", fontsize=18)
plt.title("TOP 20 Most Commmon Developers ", fontsize=30)
plt.xticks(rotation=90, fontsize = 13) 
plt.show()


# ### <font color=darklimegreen>We see that in the above analysis <font color=orange>DETENTION APPS</font> has developed the most no. of apps.</font>
# 
# 
# ***
# 
# <br>
# 
# 
# 
# # <font color=cornflowerblue>2.Age Rating</font>

# In[49]:


'''A Function To Plot Pie Plot using Plotly'''
'''Takes time to load, click on run twice'''
def pie_plot(cnt_srs, colors, title):
    labels=cnt_srs.index
    values=cnt_srs.values
    trace = go.Pie(labels=labels, 
                   values=values, 
                   title=title, 
                   hoverinfo='percent+value', 
                   textinfo='percent',
                   textposition='inside',
                   hole=0.5,
                   showlegend=True,
                   marker=dict(colors=colors,
                               line=dict(color='#000000',
                                         width=2),
                              )
                  )
    return trace

bold('**MOST OF THE APPS HAVE 4+ AGE RATING**')
py.iplot([pie_plot(data['Age Rating'].value_counts(), ['lightlime', 'gold', 'red'], 'Age Rating')])


# ### <font color=darklimegreen>We see that in the above analysis, <font color=orange>50.3%</font> of apps have age rating of <font color=orange>4+ (4 and above)</font>.</font>
# 
# 
# ***
# 
# <br>
# 
# 
# # <font color=cornflowerblue>3.Correlation:</font>
# 
# 
# 
# ### <font color=chocolate> ~ Correlation is about the <u>_relationship between variables_.</u><br><br> ~ Correlation values range between -1 and 1.<br> ~ There are two key components of a correlation value:<br><br>1]<font color=red> _magnitude_ </font>  – The larger the magnitude (closer to 1 or -1), the stronger the correlation.<br>2]<font color=red> _sign_ </font>  – If negative, there is an inverse correlation. If positive, there is a regular correlation.</font>
# <br>

# In[50]:


data['GenreList'] = data['Genres'].apply(lambda s : s.replace('Games','').replace('&',' ').replace(',', ' ').split()) 
data['GenreList'].head()


# In[51]:


from sklearn.preprocessing import MultiLabelBinarizer # Similar to One-Hot Encoding

test = data['GenreList']
mlb = MultiLabelBinarizer()
res = pd.DataFrame(mlb.fit_transform(test), columns=mlb.classes_, index=test.index)


# In[52]:


corr = res.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(18, 19))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5})


#   
# ### <font color=darklimegreen>Here,the colour<font color=red> _red_ </font>indicates a stronger relationship b/w two values.<br><br>For eg:<br>     1] <font color=orange>Word and Books </font>are related strongly (magnitude=0.3).<br>     2] <font color=orange>Role and puzzle</font> are not that related (magnitude= -0.2).<br>     3] <font color=orange>Video and Photo</font> are strongly related (magnitude= 0.3).</font></font>
# 
# 
# ***
# 
# <br>
# 
# # <font color=cornflowerblue>4.Most EXPENSIVE Games:</font>
# <br>

# In[53]:


data.dropna(inplace = True)
price = data.sort_values(by='Price', ascending=False)[['Name', 'Price', 'Average User Rating','Genres', 'Size', 'Icon URL']].head(8)
price.iloc[:, 0:-1]


# In[54]:


import urllib.request
from PIL import Image

plt.figure(figsize=(6,3))
plt.subplot(121)
image = Image.open(urllib.request.urlopen(price.iloc[0,-1]))
plt.title('1.Tarot - Single and Multiplayer', fontsize=15)
plt.imshow(image)
plt.axis('off')

plt.subplot(122)
image = Image.open(urllib.request.urlopen(price.iloc[1,-1]))
plt.title('2.Ticket to Ride', fontsize=15)
plt.imshow(image)
plt.axis('off')

plt.show()


# ### <font color=darklimegreen>We see that in the above analysis, <font color=orange>Tarot - Single and Multiplayer</font> and <font color=orange>Ticket to Ride </font>are the two most expensive apps in the appstore.<br><br><font color=orange>Carrier Battles 4 Guadalcanal</font> and <font color=orange>Realpolitiks Mobile</font> are the 3rd and 4th most expensive apps as shown below.</font>

# In[55]:


import urllib.request
from PIL import Image

plt.figure(figsize=(6,3))
plt.subplot(121)
image = Image.open(urllib.request.urlopen(price.iloc[2,-1]))
plt.title('3.Carrier Battles 4 Guadalcanal', fontsize=15)
plt.imshow(image)
plt.axis('off')

plt.subplot(122)
image = Image.open(urllib.request.urlopen(price.iloc[3,-1]))
plt.title('4.Realpolitiks Mobile', fontsize=15)
plt.imshow(image)
plt.axis('off')

plt.show()


# ***
# 
# <br>
# 
# # <u>INFERENCES</u>
# 
# # <font color='aqua'>Best Overall Game in App Store</font>
# 
# 
# 

# In[56]:


best = data.sort_values(by=['Average User Rating', 'User Rating Count'], ascending=False)[['Name', 'Average User Rating', 'User Rating Count', 'Size', 
                                                                                         'Price', 'Developer',  'Icon URL',]].head(10)
best.iloc[:, 0:-1]


# In[57]:


bold('**Bloons TD 5 Game Develop by Ninja Kiwi**')
plt.figure(figsize=(5,5))
image = Image.open(urllib.request.urlopen(best.iloc[0, -1]))
plt.axis('off')
plt.title('Bloons TD 5')
plt.imshow(image)
plt.show()


# ### <font color=darklimegreen><font color=orange>Bloons TD 5</font> turns out to be best overall game with <font color=orange>5.0 rating</font> and <font color=orange>97776</font> reviews.<br>There are also several other Games with 4.0+ rating and healthy number of reviews</font>
# 
# 
# ***
# 
# <br>
# 
# 

# ## To conclude,<br><br>1.Games under <u>genre</u>: <font color='aqua'>Strategy,Entertainment</font> and <font color='aqua'>Simulation</font>,<br>2.<u>Priced</u> at not more than <font color='aqua'> $ 10 </font> and <br>3.With <u>Age Rating</u> <font color='aqua'> 4+ </font><br><br> tend to get a good user rating  (4.0/5.0 and above).
