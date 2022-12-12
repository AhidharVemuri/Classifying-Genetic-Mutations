# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 17:31:25 2022

@author: ahish
"""

import pandas as pd


with open("training_text", errors = 'replace') as f:
  lines = f.readlines()


ids = []
data = []

for i in range(1,len(lines)):
  ids.append(lines[i].split("||")[0])
  data.append(lines[i].split("||")[1].strip())

text_df = pd.DataFrame({"ID":id,"Text":data})

variants_df = pd.read_csv('training_variants')

#%%

def make_lower(input):
  return input.lower()

import re
re1 = "[(].*?[)]"
re2 = "[[].*?[]]"

def remove_bracketwords(input):
  new = re.sub(re1, "", input)
  new = re.sub(re2, "", new)
  return new

import string

def remove_punctuations(input):
  newstr = input.translate(str.maketrans('','',string.punctuation))
  return newstr

text_df.Text = text_df.Text.apply(make_lower)
text_df.Text = text_df.Text.apply(remove_bracketwords)
text_df.Text = text_df.Text.apply(remove_punctuations)

#%%
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


df = pd.read_csv('final_df.csv')
data_words1 = df["tokens"].tolist()
lst = data_words1[1:len(data_words1) - 1]

data_words = []

for data in lst:
    new = remove_punctuations(data)
    toks = word_tokenize(new)
    data_words.append(toks)

#%%

from sklearn.datasets import fetch_20newsgroups, fetch_openml
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import NMF, LatentDirichletAllocation
import pyLDAvis
import gensim 
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import spacy 
from nltk.corpus import stopwords 
import pyLDAvis
import pyLDAvis.gensim_models
from nltk.tokenize import word_tokenize, RegexpTokenizer
from os import listdir
from os.path import isfile, join
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
#%%

id2word = corpora.Dictionary(data_words)

corpus = [id2word.doc2bow(text) for text in data_words]

lda_model_10 = gensim.models.LdaMulticore(corpus = corpus, id2word = id2word, num_topics = 10)
lda_model_25 = gensim.models.LdaMulticore(corpus = corpus, id2word = id2word, num_topics = 25)
lda_model_50 = gensim.models.LdaMulticore(corpus = corpus, id2word = id2word, num_topics = 50)



from pprint import pprint
pprint(lda_model_10.print_topics())




import pyLDAvis
LDAvis_prepped_10 = pyLDAvis.gensim_models.prepare(lda_model_10, corpus, id2word)
LDAvis_prepped_25 = pyLDAvis.gensim_models.prepare(lda_model_25, corpus, id2word)
LDAvis_prepped_50 = pyLDAvis.gensim_models.prepare(lda_model_50, corpus, id2word)


pyLDAvis.enable_notebook()

LDAvis_prepped_10

LDAvis_prepped_25

LDAvis_prepped_50





text_df['Text'][7]



if('deletion' in text_df['Text'][7]):
    print('yes')

labels = np.array(variants_df['Class'])
np.where(labels == 4)[0]




text = text_df['Text'][np.where(labels == 4)[0]]
text2 = text_df['Text'][np.where(labels == 5)[0]]

variants1 = pd.DataFrame({"Gene":variants_df['Gene'][np.where(labels == 4)[0]], "Variation":variants_df['Variation'][np.where(labels == 4)[0]],"Text":text})
variants2 = pd.DataFrame({"Gene":variants_df['Gene'][np.where(labels == 5)[0]], "Variation":variants_df['Variation'][np.where(labels == 5)[0]],"Text":text2})


len(np.unique(variants2.Text))

text = np.array(text)
text2 = np.array(text2)

common_texts = np.unique(np.intersect1d(text, text2))

common_texts[7]



len(np.unique(text_df["Text"]))

text_dict = {}
num = 0
for i in np.unique(text_df["Text"]):
    text_dict[i] = num
    num+=1



text_labels = [text_dict[i] for i in np.array(text_df['Text'])]
text_df['Doc_Label'] = text_labels

variants_df["Doc_label"] = text_labels



df = pd.crosstab(index = variants_df.Doc_label, columns = variants_df.Class)

df['Doc_label'] = df.index



#%%


'''Train BERT on the texts. Get the Contexts in the outputs and then use that as your new feature'''

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text


encoder_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4"
preprocess_url = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"


bert_preprocess = hub.KerasLayer(preprocess_url)
bert_encoder = hub.KerasLayer(encoder_url)

'''Bert Layers'''

text_input = tf.keras.layers.Input(shape=(1,), dtype = tf.string, name = 'text')
preprocessed_text = bert_preprocess(text_input)
outputs = bert_encoder(preprocessed_text)

'''Neural Networks Layer'''

l = tf.keras.layers.Dropout(0.1, name = 'dropout')(outputs['pooled_output'])
l = tf.keras.layers.Dense(9, activation = 'sigmoid', name = 'output')(1)

'''Model'''
model = tf.keras.Model(inputs=[text_input], outputs = [l])

model.summary()


op = bert_preprocess(text_df['Text'][0])
bert_encoder(op)['pooled_output']

#%%

text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
preprocessor = hub.KerasLayer(
    "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
encoder_inputs = preprocessor(text_input)
encoder = hub.KerasLayer(
    "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4",
    trainable=True)
outputs = encoder(encoder_inputs)
pooled_output = outputs["pooled_output"]      # [batch_size, 768].
sequence_output = outputs["sequence_output"]

embedding_model = tf.keras.Model(text_input, pooled_output)
sentences = tf.constant(edited_text)
print(embedding_model(sentences))



#%%

ne_df = pd.DataFrame({"ID":variants_df["ID"], "Entities" : all_orgs})
ne_df.to_csv('name_Entity_df.csv', index = False)
ne_df['class'] = variants_df["Class"]


#%%

sent = [" ".join(i) for i in all_orgs]

ne_df["sents"] = sent

variants_df["Class"].value_counts()

train_x, test_x,train_y, test_y = train_test_split(ne_df["sents"], ne_df["class"], stratify=ne_df["class"])



from tensorflow.keras.utils import to_categorical


train_y = to_categorical(train_y)
test_y = to_categorical(test_y)


import transformers
from transformers import AutoTokenizer,TFBertModel
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
bert = TFBertModel.from_pretrained('bert-base-cased')

train_x = tokenizer(
    text=train_x.tolist(),
    add_special_tokens=True,
    max_length=70,
    truncation=True,
    padding=True, 
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = True,
    verbose = True)

test_x = tokenizer(
    text=test_x.tolist(),
    add_special_tokens=True,
    max_length=70,
    truncation=True,
    padding=True, 
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = True,
    verbose = True)


input_ids = train_x['input_ids']
attention_mask = train_x['attention_mask']


#%%


import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense


max_len = 70

input_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
input_mask = Input(shape=(max_len,), dtype=tf.int32, name="attention_mask")
embeddings = bert(input_ids,attention_mask = input_mask)[0] 
out = tf.keras.layers.GlobalMaxPool1D()(embeddings)
out = Dense(128, activation='relu')(out)
out = tf.keras.layers.Dropout(0.1)(out)
out = Dense(32,activation = 'relu')(out)
y = Dense(10,activation = 'sigmoid')(out)
model = tf.keras.Model(inputs=[input_ids, input_mask], outputs=y)
model.layers[2].trainable = True



optimizer = Adam(
    learning_rate=5e-05, # this learning rate is for bert model , taken from huggingface website 
    epsilon=1e-08,
    decay=0.01,
    clipnorm=1.0)
# Set loss and metrics
loss =CategoricalCrossentropy(from_logits = True)
metric = CategoricalAccuracy('balanced_accuracy'),
# Compile the model
model.compile(
    optimizer = optimizer,
    loss = loss, 
    metrics = metric)



train_history = model.fit(
    x ={'input_ids':train_x['input_ids'],'attention_mask':train_x['attention_mask']} ,
    y = train_y,
    validation_data = (
    {'input_ids':test_x['input_ids'],'attention_mask':test_x['attention_mask']}, test_y
    ),
  epochs=1,
    batch_size=36
)

predicted_raw = model.predict({'input_ids':test_x['input_ids'],'attention_mask':test_x['attention_mask']})

y_predicted = np.argmax(predicted_raw, axis = 1)


y_true = [np.argmax(i) for i in test_y]

from sklearn.metrics import classification_report
print(classification_report(y_true, y_predicted))


len(np.unique(df["tokens"]))

#%%


'''Getting an idea of which records have the information about the Gene and the Variant from the Variants_df'''

count = 0
index = []
index1 = []
index2 = []


'''This for loop logic was updated accordingly for all the indices'''
'''
index contains the indices for which text contains the gene in the text.
index1 contains the indices for which text contains the variation in the text
index2 contains the indices for which there is no mention of the gene or the 
variation in the text
'''


for i in range(len(variants_df)):
    if((variants_df["Gene"][i].lower() in text_df["Text"][i].lower())):
        index.append(i)

for i in range(len(variants_df)):
    if((variants_df["Variation"][i].lower() in text_df["Text"][i].lower())):
        index1.append(i)

for i in range(len(variants_df)):
    if((variants_df["Variation"][i].lower() not in text_df["Text"][i].lower()) and (variants_df["Gene"][i].lower() not in text_df["Text"][i].lower())):
        index2.append(i)

'''
Number of samples having Texts with the Gene and the Variation in the text - 2284
Number of samples having Texts with just the Gene in the text - 3081
Number of samples having texts with no mention of the gene or variation - 240
'''


articles = np.unique(text_df["Text"][index])


classes = np.unique(variants_df["Class"][index])

variants_df.loc[index, :]

variants_df["Class"][index].value_counts()


variants_df["Gene_in_text"] = np.zeros(len(variants_df), dtype = int)
variants_df["Variation_in_text"] = np.zeros(len(variants_df), dtype = int)
variants_df["no_mention"] = np.zeros(len(variants_df), dtype = int)

variants_df.loc[index, "Gene_in_text"] = 1
variants_df.loc[index1, "Variation_in_text"] = 1
variants_df.loc[index2, "no_mention"] = 1

#%%

'''Getting sentences with the Gene and Varations mentioned in them for the samples which have them'''

text_df['Text'][0]


'''Index of samples which have either gene or variant or both in the text'''
index = variants_df.loc[((variants_df["Gene_in_text"] == 1) | (variants_df["Variation_in_text"] == 1)),:].index
'''Index of samples which have no mention of the gene or variant in the text'''
index2 = variants_df.loc[(variants_df["no_mention"] == 1),:].index


from nltk.tokenize import sent_tokenize

sent_tokenize(text_df['Text'][0])

sent_list = []
summary_list = []

variants_df["summary"] = " "

'''For the samples which have the gene or/and variation in the text'''

for i in index:
    temp = [sents for sents in sent_tokenize(text_df['Text'][i]) if(variants_df["Gene"][i].lower() in sents.lower() or variants_df["Variation"][i].lower() in sents.lower())]
    sent_list.append(''.join(temp))

'''Summarizing these sentences to get the 2 most important ones to feed to the BERT model'''


from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.kl import KLSummarizer

summarizer = KLSummarizer()

for i in sent_list:
    parser = PlaintextParser.from_string(i, Tokenizer('english'))
    summary = summarizer(parser.document, 2)
    summary_list.append(str(summary[0]))


variants_df.loc[index,"summary"] = summary_list

'''For the samples with no mention of the gene or the variation'''

sent_list = []

summary_list = []

missing = []

for i in index2:
    temp = [sents for sents in sent_tokenize(text_df['Text'][i]) if("mutation" in sents.lower() or "cancer" in sents.lower() or "tumor" in sents.lower() or "gene" in sents.lower() or "protein" in sents.lower())]
    sent_list.append(''.join(temp))
    if(len(temp) == 0):
        missing.append(i)
        


for i in sent_list:
    parser = PlaintextParser.from_string(i, Tokenizer('english'))
    summary = summarizer(parser.document, 2)
    if(len(summary)>0):
        summary_list.append(str(summary[0]))
    else:
        summary_list.append(' ')

missing_summaries = []

for i in missing:
    parser = PlaintextParser.from_string(text_df["Text"][i], Tokenizer('english'))
    summary = summarizer(parser.document, 2)
    if(len(summary)>0):
        missing_summaries.append(str(summary[0]))
    else:
        missing_summaries.append(' ')


variants_df.loc[index2,"summary"] = summary_list
variants_df.loc[missing,"summary"] = missing_summaries

variants_df.to_csv('updated_variants.csv', index = False)

#%%

'''Dropping Duplicates and NA values'''

variants_df.loc[variants_df.summary == "null", "Class"].value_counts()

variants_df=variants_df.replace('null', np.nan)

variants_df = variants_df.dropna()

variants_df = variants_df.drop_duplicates(keep = 'first')

variants_df["Full_Sample"] = variants_df["Gene"].astype(str) + ' ' + variants_df["Variation"]

variants_df.to_csv('updated_variants.csv', index = False)


#%%

'''Training BioBert Model'''


summary = list(variants_df.summary)

columns = ["ID","Gene","Variation","Full_Sample","summary","Gene_in_text","Variation_in_text","no_mention","Class"]

final_df = variants_df[columns]


from sentence_transformers import SentenceTransformer
sentences = list(final_df.summary)
samples = list(final_df.Full_Sample)

model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')
embeddings = model.encode(sentences)

embeddings2 = model.encode(samples)

import numpy as np
from numpy.linalg import norm

cosine_similarity = np.sum(embeddings*embeddings2, axis=1)/(norm(embeddings, axis=1)*norm(embeddings2, axis=1))

variants_df["Relavance"] = cosine_similarity


columns = ["ID","Gene","Variation","Full_Sample","summary","Relavance","Gene_in_text","Variation_in_text","no_mention","Class"]

final_df = variants_df[columns]

final_df.to_csv('Final_df.csv',index = False)


cancer = "cancer"
benign = "benign"
unknown_significance ="unknown"

cancer_embed = model.encode(cancer)
benign_embed = model.encode(benign)
unknown_embed = model.encode(unknown_significance)


cancer_vector = []
benign_vector = []
unknown_vector = []

for i in range(len(variants_df)):
    cancer_vector.append(cancer_embed)
    benign_vector.append(benign_embed)
    unknown_vector.append(unknown_embed)

cancer_vector = np.array(cancer_vector)
benign_vector = np.array(benign_vector)
unknown_vector = np.array(unknown_vector)

cosine_sim_cancer = np.sum(embeddings*cancer_vector, axis=1)/(norm(embeddings, axis=1)*norm(cancer_vector, axis=1))
cosine_sim_benign = np.sum(embeddings*benign_vector, axis=1)/(norm(embeddings, axis=1)*norm(benign_vector, axis=1))
cosine_sim_unknown = np.sum(embeddings*unknown_vector, axis=1)/(norm(embeddings, axis=1)*norm(unknown_vector, axis=1))



variants_df["cancer_relavance"] = cosine_sim_cancer
variants_df["benign_relavance"] = cosine_sim_benign
variants_df["unknown_relavance"] = cosine_sim_unknown

matrix = variants_df[["Relavance","cancer_relavance","benign_relavance","unknown_relavance","Gene_in_text","Variation_in_text","no_mention","Class"]].corr()

final_df = variants_df[["ID","Gene","Variation","Full_Sample","summary","Relavance","cancer_relavance","benign_relavance","unknown_relavance","Gene_in_text","Variation_in_text","no_mention","Class"]]


final_df.to_csv('Final_df.csv',index = False)



