import collections
import pandas as pd
import nltk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from string import punctuation
from sklearn.metrics.pairwise import cosine_similarity
import itertools
import string
import math

stop_words = set(stopwords.words('english'))
PUNC_REMOVAL_RE = re.compile('[%s]' % re.escape(punctuation))
NEWLINE_REMOVAL_RE = re.compile('[%s]' % '\n')
# Get max number of repetitions for n-grams in range start to end for each summary
# Returns list of list of scores ordered from start to end
def max_ngram_repetitions(summaries, start, end):
    
    vectorizers = [CountVectorizer(ngram_range=(x,x)) for x in range(start, end + 1)]
    scores_lists = [[] for x in range(end-start + 1)]
    for summary in summaries:
        summary = PUNC_REMOVAL_RE.sub('', NEWLINE_REMOVAL_RE.sub(' ', summary))
        word_tokens = word_tokenize(summary)
        filtered_summary = [w for w in word_tokens if not w in stop_words]
        summary_string = " ".join(filtered_summary)
        
        ngram_vecs = [vectorizer.fit_transform([summary_string]).toarray()[0] for vectorizer in vectorizers]
        
        ngram_reps = [[x for x in vector if x > 1] + [1] for vector in ngram_vecs]
        
        ngram_scores = [max(reps) for reps in ngram_reps]

        for i in range(len(ngram_scores)):
            scores_lists[i].append(ngram_scores[i])
    return scores_lists

# Taken from http://pythonexample.com/code/longest-repeated-substring-python/

#ideas from https://stackoverflow.com/questions/10355103/finding-the-longest-repeated-substring         

def lrs_helper(sentence, return_len=True):
    class Node:
        def __init__(self, val):
            self.val = val
            self.children = {}
            self.indexes = []
            self.count = 1
    
    def insert(root, sentence, index, original_suffix, level=0):
        root.indexes.append(index)
        # update the result if find more than two child in a 
        # node and longer than the result now
        if(len(root.indexes) > 1 and maxLen[0] < level):
            maxLen[0] = level
            maxStr[0] = original_suffix[0:level]
        if not sentence:
            return None
 
        if(sentence[0] not in root.children):
            child = Node(sentence[0])
            root.children[sentence[0]] = child
        else:
            child = root.children[sentence[0]]
            child.count += 1
        insert(child, sentence[1:], index, original_suffix, level+1)
        return None
    maxLen = [0]
    maxStr = ['']
    root = Node('@')
    for i in range(len(sentence)):
        s = sentence[i:]
        insert(root, s, i, s)
    return maxLen[0] if return_len else root

def lrs(summaries):
    scores = []
    for summary in summaries:
        scores.append(lrs_helper(PUNC_REMOVAL_RE.sub('', NEWLINE_REMOVAL_RE.sub(' ', summary)).lower().strip().split(" ")))
    return scores



def mean_sent_redundancy(sentences):
    if len(sentences) == 1:
        return 1
    sentence_combinations = itertools.combinations(sentences, 2)
    similarity_scores = []
    for pair in sentence_combinations:
        similarity_scores.append(cosine_similarity([pair[0]], [pair[1]])[0][0]**2)
    return 1 - (sum(similarity_scores) / len(similarity_scores))

def max_sent_redundancy(sentences):
    if len(sentences) == 1:
        return 1
    sentence_combinations = itertools.combinations(sentences, 2)
    similarity_scores = []
    for pair in sentence_combinations:
        similarity_scores.append(cosine_similarity([pair[0]], [pair[1]])[0][0]**2)
    return 1 - max(similarity_scores)

#taken from https://stackoverflow.com/questions/2158395/flatten-an-irregular-list-of-lists
def flatten(l):
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el

class Node:
    def __init__(self, val):
        self.val = val
        self.children = {}
        self.indexes = []
        self.count = 1
         
            
def traverse_node(node, level, prefix="", min_length=3):
    if not node.children:
        if level >= min_length:
            return [prefix + " " +node.val, node.count]
        else:
            return []
    else:
        new_prefix = prefix + " " + node.val if prefix else node.val
        new_prefix = "" if new_prefix == "@" and level == 0 else new_prefix
        searchable_children = [child_key for child_key in node.children if node.children[child_key].count > 1]
        if not searchable_children:
            if level >= min_length:
                return (new_prefix, node.count)
            else:
                return []
        else:
            return [traverse_node(node.children[child_key], level+1, new_prefix) for child_key in searchable_children]

def get_scores(sentence):
    node = lrs_helper(sentence, return_len=False)
    flattened_scores = list(flatten(traverse_node(node, 0)))
    scores = [flattened_scores[i*2:(i+1) * 2] for i in range(math.ceil(len(flattened_scores) / 2))]
    sorted_scores = sorted(scores, reverse=True, key=lambda x: len(x[0].split(" ")))
    index = 0
    while len(sorted_scores) > index + 1:
        words = sorted_scores[index][0]
        removal_indices = []
        for j in range(index + 1,len(sorted_scores)):
                if sorted_scores[j][0] in words:
                    removal_indices.append(j)
        removal_indices.reverse()
        for j in removal_indices:
            del sorted_scores[j]
        index += 1
    return sorted_scores


def calc_score(scores, summary):
    return sum([len(score[0].split(" ")) * score[1]**2 for score in scores]) / len(summary)