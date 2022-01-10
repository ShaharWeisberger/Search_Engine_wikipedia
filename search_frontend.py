import nltk as nltk
# from flask import Flask, request, jsonify
# import requests
# import bs4
# from bs4 import BeautifulSoup
# from contextlib import closing
# import inverted_index_colab
# import operator
# import itertools
import math
import pickle
from _csv import reader
from collections import Counter
from pathlib import Path
# from time import time

# from contextlib import closing
# import os
import re

from flask import Flask, request, jsonify
# import requests
from contextlib import closing
#
#
import inverted_index_gcp
import operator
# import gzip
# import csv
# import json
# import pandas as pd
from nltk.corpus import stopwords
nltk.download('stopwords')



class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):


        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


global index_title
index_title= inverted_index_gcp.InvertedIndex.read_index('title', 'title_index')
global index_anchor
index_anchor= inverted_index_gcp.InvertedIndex.read_index('anchor', 'anchor_index')
global index_body
index_body= inverted_index_gcp.InvertedIndex.read_index('postings_gcp', 'index')

with open(Path('id_title_dict.pickle'), 'rb') as f:
    global id_title
    id_title = pickle.loads(f.read())


with open(Path('id_len_dict.pickle'), 'rb') as f:
    global id_len
    id_len = pickle.loads(f.read())

with open(Path('pageviews-202108-user.pkl'), 'rb') as f:
    global id_view
    id_view = pickle.loads(f.read())

with open(Path('id_rank_dict.pickle'), 'rb') as f:
    global id_rank
    id_rank = pickle.loads(f.read())


@app.route("/search")
def search():
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
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION


    query = tokenize(query)
    title_weight = 2;
    anchor_weight = 1.2;
    body_weight = 0.6;
    view_weight = 1.3;
    rank_weight = 1.8;

    title = search_title_not_for_real(query)
    norm_title = normaliziation_func(title, query, title_weight)
    anchor = search_anchor_not_for_real(query)
    norm_anchor = normaliziation_func(anchor, query, anchor_weight)

    body_sim = search_body_not_for_real(query,body_weight)
    body_25 = search_body_25(query, 1.4*body_weight)


    ids = []
    if (len(norm_title)!= 0):
        for id in norm_title:
            ids.append(id[0])

    view_lst = pv_for_life(ids,view_weight)

    rank_lst = pr_for_life(ids,rank_weight)



    final = {}
    final = update_final_search_dic(final, norm_title)
    final = update_final_search_dic(final, body_sim)
    final = update_final_search_dic(final, body_25)
    final = update_final_search_dic(final, norm_anchor)
    final = update_final_search_dic(final, view_lst)
    final = update_final_search_dic(final, rank_lst)

    final = sorted(final.items(), key=operator.itemgetter(1), reverse=True)
    not_res = final[:100]
    for id in not_res:
        try:
            res.append((id[0], id_title[id[0]]))
        except:
            pass


    # END SOLUTIONS
    # app.id_title.clear()
    return jsonify(res)


@app.route("/search_body")
def search_body():
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
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    query = tokenize(query)
    N = 6348910 # numbers of pages
    sim = {}



    # din = {}
    word_count_for_queary = Counter(query)
    for word in query:
        #calaulting idf for each word(term)
        # df = app.index_body.df[word]
        df = index_body.df[word]
        idf = math.log10(N/df) ##### might give a condition that if the idf is smaller then a size we continue without checking

        posting_lst = read_posting_list(index_body, word, 'postings_gcp')
        access_denied = [False for i in range(11)]

        for id_tf in posting_lst:

            try:
                tfij = id_tf[1]/id_len[id_tf[0]] ######look out in here
                wij = tfij * math.log10(N/df)
                if id_tf[0] not in sim:
                    sim[id_tf[0]] = wij*word_count_for_queary[word]
                else:
                    sim[id_tf[0]] += wij*word_count_for_queary[word]
            except:
                pass

    sim = sorted(sim.items(), key=operator.itemgetter(1), reverse=True)

    for id in sim[:100]:
        try:
            res.append((id[0], id_title[id[0]]))
        except:
            continue
    # END SOLUTION

    return jsonify(res)


@app.route("/search_title")
def search_title():
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
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)

        # return jsonify(res)



    # BEGIN SOLUTION
    query = tokenize(query)

    post_list = get_posting_list(query, index_title, 'title')

    try:
        res = map(lambda x: tuple((x[0], id_title[x[0]])), post_list)
        res = list(res)
    except:
        diffrent_res=[]
        for id in post_list:
            try:
                diffrent_res.append((id[0],id_title[id[0]]))
            except:
                pass
        res = diffrent_res

    return jsonify(res)


@app.route("/search_anchor")
def search_anchor():
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
    query = request.args.get('query', '')
    if len(query) == 0:
            return jsonify(res)

    # BEGIN SOLUTION

    query = tokenize(query)
    # post_list = get_posting_list(query, app.index_anchor, dir='anchor')
    post_list = get_posting_list(query, index_anchor, dir='anchor')



    try:
        res = map(lambda x: tuple((x[0], id_title[x[0]])), post_list)
        res = list(res)
    except:
        diffrent_res=[]
        for id in post_list:
            try:
                diffrent_res.append((id[0],id_title[id[0]]))
            except:
                pass
        res = diffrent_res

    # END SOLUTION

    return jsonify(res)


@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
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
    # print('page rankingggggggg')
    wiki_ids = request.get_json() ######
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION


    for id in wiki_ids:
        try:
            res.append(id_rank[id])  ###### need to change it only to the int without the id
        except:
            pass
    if len(res!=0):
        res.sort(key=lambda x: x[1], reverse=True)

    # END SOLUTION

    return jsonify(res)



@app.route("/get_pageview", methods=['POST'])
def get_pageview():
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
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION




    for id in wiki_ids:
        try:
            res.append(id_view[id]) ###### need to change it only to the int without the id
        except:
            pass


    # END SOLUTION
        return jsonify(res)





def get_posting_list(q, inverted, dir=''): ####WOWOWOWOWOWOWOWOWOWOWOWOWOWOWOWOWOW

    dic = {}
    p_lists = []

    split_query = q
    for i in split_query:

        posting_list = read_posting_list(inverted, i, dir)

        p_lists += posting_list

    # putting it in a dict for fast access and mearge results so we wont have duplicates.
    for j in p_lists:
        if j[0] not in dic:
            dic[j[0]] = j[1]
        else:
            dic[j[0]] += j[1]



    dic = dict(sorted(dic.items(), key=operator.itemgetter(1), reverse=True))

    lst = list(dic.items())
    return lst

TUPLE_SIZE = 6
TF_MASK = 2 ** 16 - 1  # Masking the 16 low bits of an integer


def read_posting_list(inverted, w, dir=''):
    with closing(inverted_index_gcp.MultiFileReader()) as reader:
        try:
            locs = inverted.posting_locs[w]
            real_locs = [tuple((dir + '/' + locs[0][0], locs[0][1]))]
            b = reader.read(real_locs, inverted.df[w] * TUPLE_SIZE)
            posting_list = []
            for i in range(inverted.df[w]):
                doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
                tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
                posting_list.append((doc_id, tf))
            return posting_list
        except:
            return []

def tokenize(query):
    english_stopwords = frozenset(stopwords.words('english'))
    corpus_stopwords = ["category", "references", "also", "external", "links",
                        "may", "first", "see", "history", "people", "one", "two",
                        "part", "thumb", "including", "second", "following",
                        "many", "however", "would", "became"]

    all_stopwords = english_stopwords.union(corpus_stopwords)
    RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

    tokens = [token.group() for token in RE_WORD.finditer(query.lower())]
    return [token for token in tokens if token not in all_stopwords]

#####

def normaliziation_func(lst_to_norm , query , weight):
    """
    normaliziation func for anchor and title.
    :param lst_to_norm:
    :param query:
    :param weight:
    :return: a list of tuple (id,number) the number is the rank this function gives to the id
    based on the weight and the function.
    """
    res = []
    for id in lst_to_norm:
        num = 0
        for word in id[1].split():
            for q in query:
                if q == word.lower():
                    num += 1
        res.append((id[0], weight*(num/len(id[1].split()))))

    return res


def search_body_not_for_real(query, body_weight):
    """
    does the same as search_body but every page is between 1-0 based on the page with the biggest
    comsim.
    :param query:
    :param body_weight:
    :return:
    """
    res = []
    N = 6348910  # numbers of pages
    sim = {}
    if len(query) == 0:
        return res
    # BEGIN SOLUTION


    word_count_for_queary = Counter(query)
    for word in query:

        df = index_body.df[word]
        idf = math.log10(N / df)

        posting_lst = read_posting_list(index_body, word, 'postings_gcp')

        for id_tf in posting_lst:

            try:
                tfij = id_tf[1] / id_len[id_tf[0]]  ######look out in here
                wij = tfij * math.log10(N / df)
                if id_tf[0] not in sim:
                    sim[id_tf[0]] = wij * word_count_for_queary[word]
                else:
                    sim[id_tf[0]] += wij * word_count_for_queary[word]
            except:
                pass

    sim = sorted(sim.items(), key=operator.itemgetter(1), reverse=True)
    sim = sim[:100]
    biggest_sim = sim[0][1]
    for id in sim:
        res.append((id[0], body_weight*(id[1]/biggest_sim)))
    return res


def search_body_25(query, body_weight_25):
    b=0.75
    k1=1.5
    N=6348910
    dic={}
    res = []
    for word in query:
        posting_list=read_posting_list(index_body,word,'postings_gcp')
        if len(posting_list)==0:
            continue
        df = index_body.df[word]
        idf = math.log(1 + (N - df + 0.5) / (df + 0.5)+1)
        for id_tf in posting_list:
            try:
                numerator = idf * (id_tf[1] * (k1 + 1))
                denominator = id_tf[1] + k1 * (1 - b + (b * id_len[id_tf[0]] / 320))
                score = numerator/denominator
                if id_tf[0] in dic:
                    dic[id_tf[0]] += score
                else:
                    dic[id_tf[0]] = score
            except:
                pass
    dic = sorted(dic.items(), key=operator.itemgetter(1), reverse=True)
    dic = dic[:100]
    biggest_sim = dic[0][1]
    for id in dic:
        res.append((id[0], body_weight_25*(id[1]/biggest_sim)))

    return res

def search_title_not_for_real(query):

    """
    same as search_title but we can send the query
    :param query:
    :return:
    """
    res = []
    if len(query) == 0:
        return res
    # BEGIN SOLUTION

    post_list = get_posting_list(query, index_title, dir='title')


    try:
        res = map(lambda x: tuple((x[0], id_title[x[0]])), post_list)
        res = list(res)
    except:
        diffrent_res=[]
        for id in post_list:
            try:
                diffrent_res.append((id[0],id_title[id[0]]))
            except:
                pass
        res = diffrent_res

    return res


def search_anchor_not_for_real(query):
    """
    same as search_anchor but we can send the query
    :param query:
    :return:
    """
    res = []
    if len(query) == 0:
        return res

    # BEGIN SOLUTION

    post_list = get_posting_list(query, index_anchor, dir='anchor')



    try:
        res = map(lambda x: tuple((x[0], id_title[x[0]])), post_list)
        res = list(res)
    except:
        diffrent_res=[]
        for id in post_list:
            try:
                diffrent_res.append((id[0],id_title[id[0]]))
            except:
                pass
        res = diffrent_res
    # END SOLUTION

    return res


def pv_for_life(wiki_ids, view_weight):
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
        list of tuples: (id , view_numbers)
          list of page view numbers from August 2021 that correrspond to the
    '''
    res = []
    if len(wiki_ids) == 0:
        return res
    # BEGIN SOLUTION



    new_res = res
    for id in wiki_ids:
        try:
            res.append((id,id_view[id])) ###### need to change it only to the int without the id
        except:
            pass

    if(len(res)!=0):
        res.sort(key=lambda x: x[1], reverse=True)
        biggest_view = res[0][1]
        new_res = []
        for fix in res:
            new_res.append((fix[0], view_weight*(fix[1]/biggest_view)))
    return new_res


def pr_for_life(wiki_ids, rank_weight):
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
        list of tuples: (id , rank)
          list of page view numbers from August 2021 that correrspond to the
    '''
    res = []
    if len(wiki_ids) == 0:
        return res

    new_res = res



    for id in wiki_ids:
        try:
            res.append((id, id_rank[id])) ###### need to change it only to the int without the id
        except:
            pass

    if(len(res)!=0):
        res.sort(key=lambda x: x[1], reverse=True)

        biggest_rank = res[0][1]
        new_res = []
        for fix in res:
            new_res.append((fix[0], rank_weight*(fix[1]/biggest_rank)))

    return new_res


def update_final_search_dic(final, megazord):
    for page in megazord:
        if page[0] not in final:
            final[page[0]] = page[1]
        else:
            final[page[0]] += page[1]
    return final


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=False)
