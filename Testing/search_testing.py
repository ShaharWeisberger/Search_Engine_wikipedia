import itertools
import math
import pickle
from _csv import reader
from collections import Counter
from pathlib import Path
from time import time

import wikipedia
from contextlib import closing
import hashing


from flask import Flask, request, jsonify
import requests
import bs4
from bs4 import BeautifulSoup
from contextlib import closing
import inverted_index_colab
import operator
import gzip
import csv
import json
import pandas as pd


def search(query):
    ''' Returns up to a 100 of your best search results for the query. This is
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use
        PageRank, query expansion, etc.
        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    res = []
    if len(query) == 0:
        return res
    # BEGIN SOLUTION

    # END SOLUTION
    return res


def search_body(query):
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the
        staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords.
        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    res = []
    N = 6348910 # numbers of pages
    sim = {}
    if len(query) == 0:
        return res
    # BEGIN SOLUTION
    name = 'drive/MyDrive/Test_data/len/'
    inverted_index_body = inverted_index_colab.InvertedIndex.read_index('drive/MyDrive/Test_data/body_index', 'index_text')
    with open(Path('drive/MyDrive/Test_data/id_title_dict.pickle'),  'rb')as  f:
        id_title_dic = pickle.loads(f.read())
    # print('trying to open json file of id_len')
    # t1_start = time()
    #
    # with open('drive/MyDrive/Test_data/doc_info_index/id_title_len_dict.json') as f:
    #     dict = json.load(f)
    #     duration = time() - t1_start
    #     print(duration)
    din = {}
    word_count_for_queary = Counter(query.split())
    for word in query.split():
        #calaulting idf for each word(term)
        df = inverted_index_body.df[word]
        idf = math.log10(N/df) ##### might give a condition that if the idf is smaller then a size we continue without checking
        # print('idf')
        # print(idf)
        posting_lst = read_posting_list(inverted_index_body,word, 'drive/MyDrive/Test_data/body_index')
        access_denied = [False for i in range(11)]
        print('trying to open pickle file of id_len')
        t1_start = time()

        for id_tf in posting_lst:
            # res.append((id,wid2pv[id]))
            if not access_denied[hashing.index_hash(id_tf[0])]:
                din.update(hashing.get_dic(name, 'len.pkl', id_tf[0]))
                access_denied[hashing.index_hash(id_tf[0])] = True
            # print(type(id_tf))
            # print(id_tf)
            # print(type(id_tf[0]))
            # print(id_tf[0])
            # print(type(id_tf[1]))
            # print(id_tf[1])
            # print(type(dict[str(id_tf[0])][1]))
            # print(dict[str(id_tf[0])])
            # print(dict[str(id_tf[0])][0])
            # print(dict[str(id_tf[0])][1])
            # print(dict[str(id_tf[0])][2])
            tfij = id_tf[1]/din[id_tf[0]] ######look out in here
            wij = tfij * math.log10(N/df)
            if id_tf[0] not in sim:
                sim[id_tf[0]] = wij*word_count_for_queary[word]
                # res.append((id_tf[0],))
            else:
                sim[id_tf[0]] += wij*word_count_for_queary[word]
            # print(tfij)
            # break
    # print(type(sim))
    # print(sim)
        duration = time() - t1_start
        print(duration)
    sim = sorted(sim.items(), key=operator.itemgetter(1), reverse=True)

    # print(sim)
    # sim = sim[:100]
    # END SOLUTION
    print('starting to append')
    t1_start = time()
    for id in sim[:100]:
        res.append((id[0],id_title_dic[id[0]]))
    duration = time() - t1_start
    print(duration)
    print(sim[:100])
    return res
    # return res


def search_title(query):
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF
        QUERY WORDS that appear in the title. For example, a document with a
        title that matches two of the query words will be ranked before a
        document with a title that matches only one query term.
        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    if len(query) == 0:
        return res
    # BEGIN SOLUTION

    post_list = get_posting_list(query, 'index_title', dir='drive/MyDrive/Test_data/title_index')
    # each element is (id,tf) and we want it to be --> (id,title)
    id_title_dic = {}
    with open(Path('drive/MyDrive/Test_data/id_title_dict.pickle'),  'rb')as  f:
        id_title_dic = pickle.loads(f.read())
    # res = list(map(lambda x: tuple((x[0], wikipedia.page(pageid=x[0], auto_suggest=True, redirect=True).title)),post_list))
    res = map(lambda x: tuple((x[0], id_title_dic[x[0]])), post_list)
    res = list(res)

    # END SOLUTION
    return res

def search_anchor(query):
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE ANCHOR TEXT of articles, ordered in descending order of the
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page.
        For example, a document with a anchor text that matches two of the
        query words will be ranked before a document with anchor text that
        matches only one query term.
        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    if len(query) == 0:
        return res

    # BEGIN SOLUTION

    post_list = get_posting_list(query, 'index_anchor', dir='drive/MyDrive/Test_data/anchor_index')
    # each element is (id,tf) and we want it to be --> (id,title)
    # res = list(map(lambda x: tuple((x[0], wikipedia.page(pageid=x[0], auto_suggest=True, redirect=True).title)),post_list))
    #res = list(map(lambda x: tuple((x[0], wikiId2title_name(x[0]))),post_list))
    id_title_dic = {}
    with open(Path('drive/MyDrive/Test_data/id_title_dict.pickle'),'rb')as f:
        id_title_dic = pickle.loads(f.read())
    # res = list(map(lambda x: tuple((x[0], wikipedia.page(pageid=x[0], auto_suggest=True, redirect=True).title)),post_list))
    # res = list(map(lambda x: tuple((x[0], id_title_dic[x[0]])), post_list))
    res = map(lambda x: tuple((x[0], id_title_dic[x[0]])), post_list)
    res = list(res)
    # END SOLUTION
    return res
    # END SOLUTION

    return res

def get_pagerank(wiki_ids):
    ''' Returns PageRank values for a list of provided wiki article IDs.
        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    rank ={}
    if len(wiki_ids) == 0:
        return res
    # name = 'drive/MyDrive/Test_data/id_rank_dict.pickle'
    name = 'drive/MyDrive/Test_data/pr/' ####need to get this shit smaller
    access_denied = [False for i in range(11)]
    t1_start = time()
    print('trying to open')
    # with open('drive/MyDrive/Test_data/pageviews-202108-user.pkl', 'rb') as f:
    #     wid2pv = pickle.loads(f.read())
    #     all_keys = wid2pv.values()
    #     print('max view')
    #     max_val = max(all_keys)
    #     print(max_val)
    #     # print(wid2pv[34258])
    #     duration = time() - t1_start
    #     print(duration)

    for id in wiki_ids:
        # res.append((id,wid2pv[id]))
        if not access_denied[hashing.index_hash(id)]:
            rank.update(hashing.get_dic(name , 'pr.pkl',id))
            access_denied[hashing.index_hash(id)] = True
    duration = time() - t1_start
    print(duration)
    t1_start = time()
    print('trying to append')
    for id in wiki_ids:
        res.append((id,rank[id])) ###### need to change it only to the int without the id
        #print(wid2pv[34258])
    duration = time() - t1_start
    print(duration)
    # res = sorted(list(map(lambda x: (x,view[x]),wiki_ids)),key=lambda x:x[1] ,reverse=True)
    # res = list(map(lambda x: (x,view[x]),wiki_ids))
    # END SOLUTION
    # return (id, page view)
    return res


def get_pageview(wiki_ids):
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.
        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the
          provided list article IDs.
    '''
    res = []
    if len(wiki_ids) == 0:
        return res
    # BEGIN SOLUTION
    view = {}
    name = 'drive/MyDrive/Test_data/pv/' ####need to get this shit smaller
    access_denied = [False for i in range(11)]
    t1_start = time()
    print('trying to open')
    # with open('drive/MyDrive/Test_data/pageviews-202108-user.pkl', 'rb') as f:
    #     wid2pv = pickle.loads(f.read())
    #     all_keys = wid2pv.values()
    #     print('max view')
    #     max_val = max(all_keys)
    #     print(max_val)
    #     # print(wid2pv[34258])
    #     duration = time() - t1_start
    #     print(duration)

    for id in wiki_ids:
        # res.append((id,wid2pv[id]))
        if not access_denied[hashing.index_hash(id)]:
            view.update(hashing.get_dic(name , 'pv.pkl',id))
            access_denied[hashing.index_hash(id)] = True
    duration = time() - t1_start
    print(duration)
    t1_start = time()
    print('trying to append')
    for id in wiki_ids:
        res.append((id,view[id])) ###### need to change it only to the int without the id
        #print(wid2pv[34258])
    duration = time() - t1_start
    print(duration)
    # res = sorted(list(map(lambda x: (x,view[x]),wiki_ids)),key=lambda x:x[1] ,reverse=True)
    # res = list(map(lambda x: (x,view[x]),wiki_ids))
    # END SOLUTION
    # return (id, page view)
    return res


def get_biggest_id():
    name = 'drive/MyDrive/Test_data/pageviews-202108-user.pkl'
    x = []
    with open('drive/MyDrive/Test_data/doc_info_index/id_title_len_dict.json') as f:
        dict = json.load(f)

        #print(wid2pv)
        for id in dict:
            # print(id)
            x.append(id)
    return x[:10]



def wikiId2title_name(w_id):
    '''
    this function get an wiki page id and returns a string with the title name
    of the page.
  	'''
    URL = "https://en.wikipedia.org/?curid=" + str(w_id)
    html_content = requests.get(URL).text
    soup = BeautifulSoup(html_content, "lxml")
    title = soup.select("#firstHeading")[0].text
    return title


def get_posting_list(q, name, dir=''):
    inverted = inverted_index_colab.InvertedIndex.read_index(dir, name)

    dic = {}
    p_lists = []
    # we want to get posting list for every word in the query
    split_query = q.split()
    for i in split_query:
        posting_list = read_posting_list(inverted, i, dir)
        p_lists += posting_list

    # putting it in a dict for fast access and mearge results so we wont have duplicates.
    for j in p_lists:
        if j[0] not in dic:
            dic[j[0]] = j[1]
        else:
            dic[j[0]] += j[1]

    # res = Counter()
    # for pair in p_lists:
    #     res[pair[0]] += pair[1]

    dic = dict(sorted(dic.items(), key=operator.itemgetter(1), reverse=True))
    #print('sorted dic')
    #print(dic)
    #print('counter')
    lst = list(dic.items())
    #print(res.most_common())
    return lst
    #return res.most_common()


TUPLE_SIZE = 6
TF_MASK = 2 ** 16 - 1  # Masking the 16 low bits of an integer


def read_posting_list(inverted, w, base_dir=''):
    with closing(inverted_index_colab.MultiFileReader()) as reader:
        try:
            locs = inverted.posting_locs[w]
            real_locs = [tuple((base_dir + '/' + locs[0][0], locs[0][1]))]
            b = reader.read(real_locs, inverted.df[w] * TUPLE_SIZE)
            posting_list = []
            for i in range(inverted.df[w]):
                doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
                tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
                posting_list.append((doc_id, tf))

            return posting_list
        except:
            return []

