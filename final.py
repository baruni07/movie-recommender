#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

path = "./IMDb movies.csv"
movies_data = pd.read_csv(path)

similar_movie_ip = "Cinderella and the Secret Prince"    #find movies similar to this

#final script
num_col = ['year','duration','votes','avg_vote']

#changing dtype so it can be used for comparison later
movies_data[num_col] = movies_data[num_col].astype('float64', errors = 'ignore')  
final_cols = ['imdb_title_id','title','year','genre','duration','country','language','director','writer','actors','description',
              'votes','avg_vote']
final_movies_data = movies_data[final_cols]

#cleaning data
final_movies_data.fillna("", inplace = True)
final_movies_data['language'].replace('None',"", regex = True, inplace = True)

pd.options.mode.chained_assignment = None  # default='warn'

#getting data of inputted movie
similar_movie_dat = final_movies_data[final_movies_data['title']==similar_movie_ip]

similar_movie_dat.head()

#get list of different types of chosen features ( sinc the data is stored and seperated by commas)
def get_appr_list(feat,df):
    t = []
    for gen in df[feat].iteritems():
        x = str(gen[1]).split(", ")
        t = t + x
    return list(dict.fromkeys(t))


feat_to_check = ['genre','language','country','director']
feat_data_set = []

for i in feat_to_check:
    feat_data_set.append(get_appr_list(i,similar_movie_dat))

#drop the given similar movie
check_data_sim = final_movies_data
check_data_sim = check_data_sim[check_data_sim['title'] != similar_movie_ip]

check_data_sim.head()


genre_main_data = check_data_sim
user_genre_sim = check_data_sim

#filtering data according to priority(order of features) in 'feat_to_check' consecutively
# after filtering similar genre, using this filtered data to filter next feature and so on

for i in range(0,len(feat_to_check)):
    for g in feat_data_set[i]:
        if g==None: continue

        t1 = user_genre_sim[feat_to_check[i]].str.contains(g, na=False)
        
        '''
        combination doesnt exist or less rows . 
        then take rows from original dataset (for genre bczo it has highest priority) 
        for other features take from data set after filtering data
        '''       
        if (~t1).all() or t1.values.sum()<1000:  
            t2 = genre_main_data[genre_main_data[feat_to_check[i]].str.contains(g, na=False)]
            user_genre_sim = pd.concat([user_genre_sim,t2]).drop_duplicates().reset_index(drop=True)
            
        else:
            user_genre_sim = user_genre_sim[user_genre_sim[feat_to_check[i]].str.contains(g, na=False)] 
    
    if feat_to_check[i] == 'genre':
        genre_main_data = user_genre_sim

        
# check = user_genre_sim



#choosing most poupular movie when > 1 similar movies(input)
if len(similar_movie_dat) > 1:   
    num_similar_movie_dat = pd.DataFrame(similar_movie_dat.iloc[similar_movie_dat['avg_vote'].idxmax()]).transpose().reset_index(drop = True)
else:
    num_similar_movie_dat = similar_movie_dat

num_similar_movie_dat.reset_index(inplace=True,drop = True)


#priority of numeric based factors
feat_to_check_num = ['year','votes','duration']
feat_to_check_num_var = []

#getting values of similar data of given numeric features
for i in range(0,len(feat_to_check_num)):       
    feat_to_check_num_var.append((float)(num_similar_movie_dat[feat_to_check_num[i]]))   

# range within which similar movies can lie
feat_to_check_num_range = [5.0,0.1*feat_to_check_num_var[1],20.0]


res_step2_data = user_genre_sim
c = 20

#filtering data from previos filter data(on categorical columns) based on numerical columns
for i in range(0,len(feat_to_check_num)):
    actual_val = feat_to_check_num_var[i]
    r = feat_to_check_num_range[i]
  
    u1 = res_step2_data[feat_to_check_num[i]].between(actual_val - r, actual_val + r)
    if (~u1).all() or u1.values.sum() < c:  #combination doesnt exist or less rows
        continue

    else:   
        res_step2_data = res_step2_data[res_step2_data[feat_to_check_num[i]].between(actual_val - r, actual_val + r)]

res_data = res_step2_data

# changing to lower case so it can be compared without bias
res_data['description'] = res_data['description'].str.lower()
similar_movie_dat['description'] = similar_movie_dat['description'].str.lower()

#create df with combined genre and description
def get_comp_data(df):
    df_comb = pd.DataFrame()
    df_comb['comb']= df['genre'] + " " + df['description'].fillna("")
    df_comb[['title','imdb_title_id','avg_vote']] = df[['title','imdb_title_id','avg_vote']]
    
    return df_comb

comb_genre_desc = (get_comp_data(res_data)).reset_index(drop=True)
sim_comb_genre_desc = get_comp_data(similar_movie_dat)

#getting words in description and genre for each movie in vector format to later compare using TD-IDF
vectorizer = TfidfVectorizer(stop_words="english")

temp = pd.DataFrame()
temp = pd.concat([comb_genre_desc,sim_comb_genre_desc]).reset_index(drop=True) 
vectorizer.fit(temp['comb'])
description_matrix = vectorizer.transform(comb_genre_desc['comb'])
sim_description_matrix = vectorizer.transform(sim_comb_genre_desc['comb'])

#finding cosine simlarity of the matrices b/w so far filtered movies and actual movie inputted
from sklearn.metrics.pairwise import cosine_similarity
cos_sim = cosine_similarity(description_matrix, sim_description_matrix)

#finding max value of similarity for each movie incase there r multiple movies of same title of inputted movie
cos_sim_max = np.amax(cos_sim, axis=1)
cos_sim_max = np.reshape(cos_sim_max, (-1,1))

cos_sim_max_final = np.append(cos_sim_max, comb_genre_desc[['avg_vote']], axis=1)

# finding weighted average of similarity and avg rating
w = [0.6, 0.4]
final_comp = np.average(cos_sim_max_final, weights=w, axis=1)

#get top k movies
k = 10
indices = (-final_comp).argsort()[:k]
movies = comb_genre_desc['title'].iloc[indices]
movies


# In[ ]:




