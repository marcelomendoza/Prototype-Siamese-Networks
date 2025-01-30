import numpy as np
import os
import operator
import json
import matplotlib.pyplot as plt
import time
import datetime


class User(object):
    def __init__(self, path):
        users = {}
        for file in os.listdir(path+'/users/'):
            user = json.loads("".join(open(path+'/users/'+file).readlines()))
            users[int(file.split(".")[0])] = self.__try(user)
        self.user = users
            
    def __try(self, user):
        try:
            ers = user['followers_count']
        except:
            ers = None
        try:
            ings = user['friends_count']
        except:
            ings = None
        try:
            statuses = user['statuses_count']
        except:
            statuses = None
        try:
            listed = user['listed_count']
        except:
            listed = None
        try:
            fav = user['favourites_count']
        except:
            fav = None
        time_now = datetime.datetime.now().strftime("%a %b %d %H:%M:%S +0000 %Y")
        ts_now = time.mktime(time.strptime(time_now,"%a %b %d %H:%M:%S +0000 %Y")) 
        try:
            ts_user = time.mktime(time.strptime(user['created_at'],"%a %b %d %H:%M:%S +0000 %Y"))  
            age = int((ts_now - ts_user) /(3600*24))
        except:
            age = None
        return {'followers': ers, 'followings': ings, 'statuses': statuses, 'listed': listed, 'favourites': fav, 'age': age}
    
class Post(object):
    def __init__(self, path):
        self.post = {}
        for file in os.listdir(path+'/post/'):
            self.post[int(file.split(".")[0])] = json.loads("".join(open(path+'/post/'+file).readlines()))
    
class Node(object):
    def __init__(self, tripla, retweet):
        '''
        tripla: id_user, id_post, time to root
        retweet: Si el post es retweet o no
        '''
        self.user_id = tripla[0]
        self.post_id = tripla[1]
        self.timestamp = tripla[2]
        self.retweet = retweet
        
class Tree(object):
    def __init__(self, id_news, tree_str):  
        '''
        id_news: Id de la noticia
        nodes_order: Lista de posts ordenados por tiempo
        '''
        self.id = id_news
        nodes_order = []
        for line in tree_str:
            parent, child = line.strip().split("->")
            parent = self.__node_to_list(parent)
            child = self.__node_to_list(child)     
            if parent == (None, None, 0.0):
                nodes_order.append(Node(parent, False)) #uid, post id, root time 
            else: 
                if parent[2] <= child[2]: 
                    retweet = True if parent[1] == child[1] else False
                    nodes_order.append(Node(child,retweet))     
        self.nodes_order = sorted(nodes_order, key = operator.attrgetter('timestamp'))
        self.first_time = self.nodes_order[0].timestamp
        self.last_time = self.nodes_order[-1].timestamp
        
    def __node_to_list(self, node_str):
        num = lambda s: eval(s) if not set(s).difference('0123456789. *+-/e') else None
        array_str = [num(i.strip()[1:-1]) for i in node_str[1:-1].split(",")]
        return tuple(array_str)
    
class News:
    def __init__(self, id_news, tree_str, label):
        '''
        id_news: Id de la noticia/claim
        tree: Lista de posts ordenados por tiempo del árbol de la noticia 
        label: Etiqueta de la noticia 
        '''
        self.id = int(id_news)
        self.label = label
        self.tree = Tree(self.id, tree_str)
        self.lifespan = self.tree.last_time - self.tree.first_time
        

def load_data(path='.'):
    labels = {}
    news = {}
    for label in open(path+'/label.txt').readlines():
        labels[int(label.split(":")[1][:-1])] = label.split(":")[0]    
    for file in os.listdir(path+'/tree/'):
        id_news = file.split(".")[0]
        news[int(id_news)] = News(id_news, open(path+'/tree/'+file).readlines(), labels[int(id_news)])        
    return news

def load_data_users(path='.'):
    return User(path)

def load_data_posts(path='.'):
    return Post(path)

class data_process:
    def __init__(self, path = './'):
        self.news = load_data(path = path)
        self.posts = load_data_posts(path = path)
        self.users = load_data_users(path = path)


def ts_div_time_interactions(news, time, window, sliding_window):
    nodes_order = news.tree.nodes_order
    freq_ret, freq_rep, tic = ([] for i in range(3))
    if sliding_window:
        step = 1            #avanza de a un minuto, overlap
    else: 
        step = window       #avanza en una ventana de tiempo, disjuntos
    for i in np.arange(news.tree.first_time, news.tree.first_time + time, step): # interval lim inf 
            f_ret = 0
            f_rep = 0
            j = 0
            while j<len(nodes_order):
                if round(i,2) <= nodes_order[j].timestamp and nodes_order[j].timestamp <= round(i+window,2):
                    if nodes_order[j].retweet:
                        f_ret += 1       
                    else:
                        f_rep += 1
                j +=1      
            freq_ret.append(f_ret)
            freq_rep.append(f_rep)
            tic.append(i)
    return np.array(tic), np.array(freq_ret), np.array(freq_rep)

def get_interactions(news_dict, window, sliding_window, time):
    label_to_idx = {'true': 0, 'non-rumor': 1, 'unverified': 2, 'false': 3}
    retweets, posts, labels = ([] for i in range(3))
    for id_ in news_dict.keys():
        x, y_ret, y_rep = ts_div_time_interactions(news_dict[id_], time, window, sliding_window)
        labels.append(label_to_idx[news_dict[id_].label]) 
        retweets.append(y_ret)
        posts.append(y_rep)
    return np.array(labels), np.array(retweets), np.array(posts)

def create_fingerprints(news, users, posts, count_type, time, window, sliding_window): 
    #count_type options: 'followers', 'followings', 'new_users', 'listed', 'favourites', 'age', 'statuses'
    nodes_order = news.tree.nodes_order
    freq_count, tic = ([] for i in range(2))
    users_exists = set()
    if sliding_window:
        step = 1            #avanza de a un minuto, overlap
    else: 
        step = window       #avanza en una ventana de tiempo, disjuntos
    for i in np.arange(news.tree.first_time, news.tree.first_time + time, step): # interval lim inf 
            f_cont = 0
            j = 0
            while j<len(nodes_order):
                if round(i,2) <= nodes_order[j].timestamp and nodes_order[j].timestamp <= round(i+window,2):
                    if not nodes_order[j].user_id in users_exists:
                        if count_type == 'new_users':
                            f_cont += 1
                        else:
                            try:
                                f_cont += users.user[nodes_order[j].user_id][count_type] if users.user[nodes_order[j].user_id][count_type] else 0
                                users_exists.add(nodes_order[j].user_id)
                            except KeyError:
                                try:
                                    f_cont += users.user[int(posts.post[nodes_order[j].post_id]['user']['id_str'])][count_type]
                                    users_exists.add(int(posts.post[nodes_order[j].post_id]['user']['id_str']))
                                except:
                                    pass
                j +=1      
            freq_count.append(f_cont)
            tic.append(i)
    return np.array(tic), np.array(freq_count)

def get_fingerprints(news_dict, users, posts, count_type, window, sliding_window, time):
    label_to_idx = {'true': 0, 'non-rumor': 1, 'unverified': 2, 'false': 3}
    fingerprints, labels = ([] for i in range(2))
    for id_ in news_dict.keys():
        x, y = create_fingerprints(news_dict[id_], users, posts, count_type, time, window, sliding_window)
        labels.append(label_to_idx[news_dict[id_].label]) 
        fingerprints.append(y)
    return np.array(labels), np.array(fingerprints)

def get_multivariate_series(time_series, labels, scale = 1): 
    paa = PiecewiseAggregateApproximation(n_segments=140) # equals the number of segments and samples!!!
    digits = []
    for i in range(72):
        digits.append(i*10)
    digits = np.array(digits)
    
    if scale == 0:
        time_series_scaled = []
        for ts in time_series:
            scaler = MinMaxScaler()
            # scaler = StandardScaler()
            scaler.fit(ts)
            ts = scaler.transform(ts)
            ts = ts.reshape(ts.shape[0],ts.shape[1])
            time_series_scaled.append(ts)
    elif scale == 1:
        time_series_scaled = []
        for ts in time_series:
            scaler = MinMaxScaler()
            scaler.fit(ts)
            ts = scaler.transform(ts)
            # print('1 ', ts.shape)
            ts = ts.reshape(ts.shape[0], ts.shape[1], 1)
            ts = paa.inverse_transform(paa.fit_transform(ts))
            # print(ts.shape)
            ts = ts[:,digits,:]
            # print(ts.shape)
            ts = ts.reshape(ts.shape[0],ts.shape[1])
            # print(ts.shape)
            time_series_scaled.append(ts)                
    time_series_multi = np.stack(time_series_scaled, axis=-1)
    return time_series_multi
