#!/usr/bin/env python
# coding: utf-8

# In[5]:


import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import csv
import numpy as np
import nltk.data
import sqlite3 as sqlite

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
summaries_location = ""
sentence_embeddings_location = "./"


conn = sqlite.connect("stanford_system_summaries.db")
df = pd.read_sql("select * from system_summaries", conn)



sentences = []
sentence_counts = []
index = 0
for _i, row in df.iterrows():
    tokenized_sentences = tokenizer.tokenize(row.system_summary)
    if tokenized_sentences[-1] == '':
        del tokenized_sentences[-1]
    sentence_counts.append(len(tokenized_sentences))
    sentences.extend(tokenized_sentences)


BATCH_SIZE = 200
embed_batches = list(range(0, len(sentences), BATCH_SIZE))
use_sentence_embeddings = []

with tf.Graph().as_default():
    embed = hub.Module("https://tfhub.dev/google/universal-sentence-encoder-large/3")
    tf.logging.set_verbosity(tf.logging.ERROR)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        for batch_index in embed_batches:
            print(batch_index)
            embeddings = embed(sentences[batch_index:batch_index+BATCH_SIZE])
            use_sentence_embeddings.extend(sess.run(embeddings))


print("finished USE")


BATCH_SIZE = 200
elmo_batches = list(range(0, len(sentences), BATCH_SIZE))
elmo_sentence_embeddings = []
with tf.Graph().as_default():
    embed = hub.Module("https://tfhub.dev/google/elmo/2", name="elmo")
    tf.logging.set_verbosity(tf.logging.ERROR)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        for batch_index in elmo_batches:
            print(batch_index)
            embeddings = embed(sentences[batch_index:batch_index+BATCH_SIZE])
            elmo_sentence_embeddings.extend(sess.run(embeddings))


print("finished ELMO")

np.set_printoptions(suppress=True)
                                          
with open(sentence_embeddings_location + "stanford_use_embeddings.csv", "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(use_sentence_embeddings)
        


# In[ ]:


np.set_printoptions(suppress=True)
                                          
with open(sentence_embeddings_location + "stanford_elmo_embeddings.csv", "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(elmo_sentence_embeddings)

