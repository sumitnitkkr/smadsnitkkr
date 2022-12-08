import numpy as np
import pandas as pd
import re
from ast import literal_eval

from gensim.models import Word2Vec
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import sklearn
from sklearn.model_selection import train_test_split 
from tensorflow.keras.layers import Embedding, Input, Dense, LSTM, Bidirectional, Attention
from tensorflow.keras.layers import Layer, Flatten, LayerNormalization, Concatenate, MultiHeadAttention
from keras import backend as K
from keras.models import Model
from matplotlib import pyplot as plt
import seaborn as sns

df_ontology = pd.read_csv("D:\\My\\Implementation\\Dataset Creation\\Phrases Topics\\KeyBERT_POS\\Depression Ontology1.csv")  # Diagnosed users are appended in last.
df_ontology.drop(['Unnamed: 0'], inplace = True, axis=1)

df = pd.read_csv("D:\\My\\Implementation\\Dataset Creation\\CLEF_Complete_n-grams (4000 keyphrases).csv")  # Diagnosed users are appended in last.
df.drop(['Unnamed: 0'], inplace = True, axis=1)
df.drop(['Subject_id','Text'], inplace = True, axis=1)
df.info()
# literal_eval picks each cell and convert it into a list. Then it will join all its elements with space and convert to string.
df.loc[:,'Text_Phrases'] = df.loc[:,'Text_Phrases'].apply(lambda x : " ".join(literal_eval(x)))    # apply(literal_eval) (Saravanan\\Word Frequency)
x = df['Text_Phrases'].values
y = df['Y'].values

categories  = df_ontology.columns
sentence_len = 1000   # sequence length or time steps
query_len = 245
embedding_dim = 50
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, stratify=y)  
# prepare tokenizer
to_exclude = '!"#$%&()*+-/:;<=>@[\\]^`{|}~\t\n'
t = Tokenizer(filters=to_exclude)
t.fit_on_texts(x)
vocab_size = len(t.word_index) + 1
x_train_indices = t.texts_to_sequences(x_train)
x_test_indices = t.texts_to_sequences(x_test)
x_train_indices_padded = np.asarray(x_train_indices)
x_test_indices_padded = np.asarray(x_test_indices)
# pad documents to a max length of 4 words

x_train_indices_padded = pad_sequences(x_train_indices_padded, maxlen=sentence_len, padding='post')
x_test_indices_padded = pad_sequences(x_test_indices_padded, maxlen=sentence_len, padding='post')
#x_encoded = x_encoded.reshape((x_encoded.shape[0],x_encoded.shape[1],n_features))
#print(x_train.shape)

# # check number of keyphrases with more than 1 word.
# count=0
# for word, i in t.word_index.items():
#     if len(word.split('_'))>1: count = count+1

ontology = []
for i,category in enumerate(categories):
    ontology.append( " ".join(df_ontology.loc[:,f'{category}'].dropna()))

concept_indices = t.texts_to_sequences(ontology)
concept_indices_padded = pad_sequences(concept_indices, maxlen=query_len, padding='post')

'''''''''''''''' Reading Vectors from Custom-trained .bin embedding file '''''''''''''''''''''''

# Reading from custom trained file Word2Vec_phrase50.bin so binary=True.
model = Word2Vec.load('D:\\My\\Implementation\\Dataset Creation\\Word Vectors\\Trained\\NN_Keyphrase_Embedding\\KeyBERT_n-grams_Word2Vec_skg50.bin')
vocab = list(model.wv.key_to_index)
from numpy import zeros
embedding_matrix = zeros((vocab_size, embedding_dim))
for word, i in t.word_index.items():
    if word in vocab:
        embedding_vector = model.wv[word]  # Slightly different from getting vectors of pre-trained binary file GoogleNews-vectors-negative300.bin
    else:
        print(word)
        embedding_vector = np.zeros(embedding_dim, dtype = int)
    embedding_matrix[i] = embedding_vector


'''''''''''''''' Bahdanau Attention '''''''''''''''''''''''
# simple function is in n-grams Based Explainability.py
# Bahdanau attention adds the query and value/key vectors while simple attention takes only single vector (key/query/value) and multiple with weight matrix and add bias matrix.
# Self-attention multiplies query and key/value vectors.
# https://medium.com/analytics-vidhya/neural-machine-translation-using-bahdanau-attention-mechanism-d496c9be30c3


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)   # initiate W1
        self.W2 = tf.keras.layers.Dense(units)   # initiate W2
        self.V = tf.keras.layers.Dense(1)        # initiate V
        #self.l1=1000
        #self.l2=1000
    def call(self, query, values): #query = Prev output, values = current input   
        # query and value matrix shapes are (None, 4000, 100) 
        query_with_time_axis = tf.expand_dims(query, 1)   # increase the query matrix by 1 dimension (None, 1, 4000, 100)
        # (None, None, 4000, 50) = (None, 1, 4000, 50) + (None, 4000, 50) if W1 and W2 units=50
        # (None, None, 4000, 50) converted to (None, None, 4000, 1) if V units=1
        score = self.V(tf.nn.tanh( self.W1(query_with_time_axis) + self.W2(values)))    # applying the score function proposed by bahdhanau
        #a = score[0][0][:self.l1]
        attention_weights = tf.nn.softmax(score, axis=1)   # apply softmax  (None, None, 4000, 1)
        #b = attention_weights[0][0][:self.l2]
        context_vector = attention_weights * values      # (None, None, 4000, 100)
        context_vector = tf.reduce_sum(context_vector, axis=1)   # (None, 4000, 100)  attention_output
        #return  context_vector, attention_weights, a, b
        return  context_vector, score

'''''''''''''''' Embedding Layer '''''''''''''''''''''''

word_input = Input(shape=(sentence_len), dtype='float32')   # Input layer with shape = 400 (400 = length of 1 new article)
print("word_input" , word_input.shape)
# Embedding layer = total no. of words, characteristics of each word, embedding matrix, maximum sentence length
embedding_layer = Embedding(input_dim = vocab_size, output_dim=embedding_dim, weights=[embedding_matrix], trainable=False)  # 
word_sequences = embedding_layer(word_input)   # embedding layer
print("word_sequences",word_sequences.shape)



# Embedding layer = total no. of words, characteristics of each word, embedding matrix, maximum sentence length
embedding_layer1 = Embedding(input_dim = vocab_size, output_dim=embedding_dim, weights=[embedding_matrix], input_length= query_len, trainable=False)  # 
concept_seq0 = embedding_layer1(concept_indices_padded[0])   # embedding layer
concept_seq1 = embedding_layer1(concept_indices_padded[1])   # embedding layer
concept_seq2 = embedding_layer1(concept_indices_padded[2])   # embedding layer
concept_seq3 = embedding_layer1(concept_indices_padded[3])   # embedding layer
concept_seq4 = embedding_layer1(concept_indices_padded[4])   # embedding layer
concept_seq5 = embedding_layer1(concept_indices_padded[5])   # embedding layer
concept_seq6 = embedding_layer1(concept_indices_padded[6])   # embedding layer
concept_seq7 = embedding_layer1(concept_indices_padded[7])   # embedding layer
concept_seq8 = embedding_layer1(concept_indices_padded[8])   # embedding layer
concept_seq9 = embedding_layer1(concept_indices_padded[9])   # embedding layer
concept_seq10 = embedding_layer1(concept_indices_padded[10])   # embedding layer
concept_seq11 = embedding_layer1(concept_indices_padded[11])   # embedding layer
concept_seq12 = embedding_layer1(concept_indices_padded[12])   # embedding layer
concept_seq13= embedding_layer1(concept_indices_padded[13])   # embedding layer

print("word_sequences",concept_seq0.shape)


''''''''''''''''''''''' Self-Attention '''''''''''''''''''''''

post_self_attention_op, post_self_attention_wts = BahdanauAttention(50) (word_sequences,word_sequences) 
print("word_attention_op" , post_self_attention_op.shape)

addition = tf.add(post_self_attention_op, word_sequences)
print("addition" , addition.shape)
normalized_post = LayerNormalization(axis=1) (addition)
print("normalized_post" , normalized_post.shape)

'''''''''''''''''''''' Cross Attention  '''''''''''''''''''''

cross_attention_output0, cross_attention_score0 = Attention(name="cross_attention0")([normalized_post,concept_seq0],  return_attention_scores=True, training = True)
print("cross_attention_output0", cross_attention_output0.shape, "cross_attention_seq0", cross_attention_score0.shape)

cross_attention_output1, cross_attention_score1 = Attention(name="cross_attention1")([normalized_post,concept_seq1],  return_attention_scores=True, training = True)
cross_attention_output2, cross_attention_score2 = Attention(name="cross_attention2")([normalized_post,concept_seq2],  return_attention_scores=True, training = True)
cross_attention_output3, cross_attention_score3 = Attention(name="cross_attention3")([normalized_post,concept_seq3],  return_attention_scores=True, training = True)
cross_attention_output4, cross_attention_score4 = Attention(name="cross_attention4")([normalized_post,concept_seq4],  return_attention_scores=True, training = True)
cross_attention_output5, cross_attention_score5 = Attention(name="cross_attention5")([normalized_post,concept_seq5],  return_attention_scores=True, training = True)
cross_attention_output6, cross_attention_score6 = Attention(name="cross_attention6")([normalized_post,concept_seq6],  return_attention_scores=True, training = True)
cross_attention_output7, cross_attention_score7 = Attention(name="cross_attention7")([normalized_post,concept_seq7],  return_attention_scores=True, training = True)
cross_attention_output8, cross_attention_score8 = Attention(name="cross_attention8")([normalized_post,concept_seq8],  return_attention_scores=True, training = True)
cross_attention_output9, cross_attention_score9 = Attention(name="cross_attention9")([normalized_post,concept_seq9],  return_attention_scores=True, training = True)
cross_attention_output10, cross_attention_score10 = Attention(name="cross_attention10")([normalized_post,concept_seq10],  return_attention_scores=True, training = True)
cross_attention_output11, cross_attention_score11 = Attention(name="cross_attention11")([normalized_post,concept_seq11],  return_attention_scores=True, training = True)
cross_attention_output12, cross_attention_score12 = Attention(name="cross_attention12")([normalized_post,concept_seq12],  return_attention_scores=True, training = True)
cross_attention_output13, cross_attention_score13 = Attention(name="cross_attention13")([normalized_post,concept_seq13],  return_attention_scores=True, training = True)

# addition1 = tf.add(cross_attention_attention_output0, word_sequences)
# normalized1 = LayerNormalization(axis=1) (addition1)


'''''''''''''''''''''' Multi Head Cross Attention  '''''''''''''''''''''

''''''''''''''''''''''''' Model Training '''''''''''''''''''''''''
concate = Concatenate()([cross_attention_output0, cross_attention_output1, cross_attention_output2, 
                         cross_attention_output3, cross_attention_output4, cross_attention_output5, cross_attention_output6,
                         cross_attention_output7, cross_attention_output8, cross_attention_output9, cross_attention_output10,
                         cross_attention_output11, cross_attention_output12, cross_attention_output13])

print('concate',concate.shape)
flatten = Flatten()(concate)   # Flatten layer  word_attention_op
predictions = Dense(1, activation='sigmoid')(flatten)   # output layer 
print('predictions2',predictions.shape)
model = Model(word_input, predictions)
model.summary()
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
history=model.fit(x_train_indices_padded, y_train, validation_split= 0.2, epochs=10)

_, test_acc = model.evaluate(x_test_indices_padded, y_test )
print('\nPREDICTION ACCURACY (%):')
print( 'Test: %.3f' % ( test_acc*100))

y_pred = model.predict(x_test_indices_padded)
#conf_matrix = tf.math.confusion_matrix(labels=y_test, predictions=y_pred)
conf_matrix = sklearn.metrics.confusion_matrix(y_test, np.rint(y_pred))
print(conf_matrix)







'''''''''''''''' Attention Visualizaation '''''''''''''''''''''''



def predict_sentence_attention( X):
        """
        For a given set of texts predict the attention
        weights for each sentence.
        :param X: 3d-tensor, similar to the input for predict
        :return: 2d array (num_obs, max_sentences) containing
            the attention weights for each sentence
        """
        att_layer = model.get_layer('bahdanau_attention')
        prev_tensor = att_layer.input

        # Create a temporary dummy layer to hold the
        # attention weights tensor
        dummy_layer = tf.keras.layers.Lambda(lambda x: att_layer(x,x)) (prev_tensor)
        #dummy_layer = tf.keras.layers.Lambda(lambda x: att_layer(x)) (prev_tensor)

        return Model(model.input, dummy_layer).predict(X)


def cross_attention( query, value):
        """
        For a given set of texts predict the attention
        weights for each sentence.
        :param X: 3d-tensor, similar to the input for predict
        :return: 2d array (num_obs, max_sentences) containing
            the attention weights for each sentence
        """
        att_layer = model.get_layer('cross_attention0')
        prev_tensor = att_layer.input

        # Create a temporary dummy layer to hold the
        # attention weights tensor
        dummy_layer = tf.keras.layers.Lambda(lambda x: att_layer([query,value])) (prev_tensor)
        #dummy_layer = tf.keras.layers.Lambda(lambda x: att_layer(x)) (prev_tensor)

        return Model(model.input, dummy_layer).predict(query)
    
res = cross_attention(embedding_layer(x_train_indices_padded[180:181]), embedding_layer1(queries_indices_padded[0]))
back2text = t.sequences_to_texts(x_train_indices_padded[180:181], queries_indices_padded[0])
[normalized,class_seq]

max_wts = []
for i in res[0][0]:
  max = np.argmax(i)
  max_wts.append(i[max])

print(len(max_wts))
max_wts
hundred = max_wts[:50]
trans_10 = np.array([hundred])
a_10 = trans_10.T
print(trans_10.shape , a_10.shape)
box_10 = np.matmul( a_10 ,trans_10  ) 
print(box_10.shape)
back2text[0]
wordList = re.sub("[^\w]", " ",  back2text[0]).split()
wordList
axis_words = wordList[:50]
axis_words
fig, ax = plt.subplots(figsize=(15, 15))
heat = sns.heatmap(box_10, fmt='', cmap='Blues',   xticklabels=axis_words,  yticklabels=axis_words, ax=ax)

figure = heat.get_figure()    
figure.savefig('D:\\My\\Implementation\\Dataset Creation\\Heatmap5.png', dpi=800)



from IPython.core.display import display, HTML
def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb
    
def attention2color(attention_score):
    #print(attention_score)
    r = 255 - int(attention_score * 255)
    #print(r)
    color = rgb_to_hex((255, r, r))
    return str(color)

max_wts = []
def max_weights(arr):
    print("arr len", len(arr))
    for i in arr:
        # print('i',len(i))
         max = np.argmax(i)
         #print('max', max)
         max_wts.append(i[max]) 
    return max_wts

def visualize_attention(idx):
    # Make new model for output predictions and attentions
    model_att = Model(inputs=model.input, outputs=[model.output, model.get_layer('cross_attention0').output])
    #idx = np.random.randint(low = 0, high=X_indices_padded.shape[0]) # Get a random test
    idx =16
    print('idx :', idx)
    tokenized_sample = np.trim_zeros(x_train_indices_padded[idx]) # Get the tokenized text
    label_probs, attentions = model_att.predict(x_train_indices_padded[idx:idx+1]) # Perform the prediction
    #print(len(attentions[1][0]) )
    print("labels",(label_probs[0][0]))
    # Get decoded text and labels
    id2word = dict(map(reversed, t.word_index.items()))
    print("id2word type", type(id2word))
    decoded_text = [id2word[word] for word in tokenized_sample] 
    
    # Get classification
    #label = np.argmax((np.array(label_probs[0])>0.5).astype(int).squeeze()) # Only one
    #label2id = ['Not Fake', 'Fake']

    # Get word attentions using attenion vector
    token_attention_dic = {}
    max_score = 0.0
    min_score = 0.0
    max_wts = max_weights(attentions[0][0][:len(tokenized_sample)])
    print(len(max_wts))      
    for token, attention_score in zip(decoded_text, max_wts):
        #print(token, attention_score)
        token_attention_dic[token] = attention_score
    a = sorted(token_attention_dic.items(), key=lambda x: x[1],reverse=True) 
    #print(a) 
    print(a[:20])    
    #print(token_attention_dic)
    # Build HTML String to viualize attentions
    html_text = "<hr><p style='font-size: large'><b>Text:  </b>"
    for token, attention in token_attention_dic.items():
        #print(attention)
        html_text += "<span style='background-color:{};'>{} <span> ".format(attention2color(attention), token)
    #html_text += "</p><br>"
    #html_text += "<p style='font-size: large'><b>Classified as:</b> "
    #html_text += label2id[label] 
    #html_text += "</p>"
    # Display text enriched with attention scores 
    display(HTML(html_text))
    Func = open("D:\\My\\Implementation\\Dataset Creation\\Phrases Topics\\KeyBERT_POS\\GFG-1.html","w")
    Func.write(html_text)
    Func.close()

visualize_attention(2)
