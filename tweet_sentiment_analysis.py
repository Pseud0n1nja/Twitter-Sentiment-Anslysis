import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from textblob import TextBlob
from nltk.stem import PorterStemmer
from textblob import Word
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

#Reading Data 
train = pd.read_csv("train.csv")

#sneak Peak into data 
train.head()
train.info()

train.isnull().sum()
train.isnull().values.any()
train.isnull().values.sum()

train.shape

#Word _counter
train['word_count']  = train['tweet'].apply(lambda x: len(str(x).split(" "))).head()
train[['tweet','word_count']].head()

#char_counter (including spaces)
train['char_count'] = train['tweet'].apply(lambda x: len(str(x)))
train[['tweet','word_count','char_count']].head()

#Word_length
def avg_word(sentence):
    words  = sentence.split()
    return (sum(len(word)for word in words)/len(words))


train['avg_word'] = train['tweet'].apply(lambda x: avg_word(x))
train[['tweet','word_count','char_count','avg_word']].head()

##################  EDA ###########################
#Number of Stopwords

stop = stopwords.words('english')
train['stopwords'] = train['tweet'].apply(lambda x: len([x for x in x.split() if x in stop]))
train[['tweet','stopwords']].head()

# Number of hashtags
train['hashtags'] = train['tweet'].apply(lambda x: len([x for x in x.split() if x.startswith('#')]))
train[['tweet','hashtags']].head()


#Number of Numerics
train['numerics'] = train['tweet'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
train[['tweet','numerics']].head()

#Number of Uppercases words
train['upper'] = train['tweet'].apply(lambda x: len([x for x in x.split() if x.isupper()]))
train[['tweet','upper']].head()

############################## Basic Preprocessing

#making the lowercase
train['tweet'] = train['tweet'].apply(lambda x: " ".join(x.lower() for x in x.split()))
train['tweet'].head()

#Removal of Punctuatuion
train['tweet'] = train['tweet'].str.replace('[^\w\s]','')
train['tweet'].head()

#Removal of Stopwords
train['tweet'] = train['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
train['tweet'].head()

# High frequency words
hi_freq = pd.Series(' '.join(train['tweet']).split()).value_counts()[:10]

#Removing high frequency words
hi_freq = list(hi_freq.index)
train['tweet'] = train['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in hi_freq))
train['tweet'].head()

# Low frequency words
lo_frew = pd.Series(' '.join(train['tweet']).split()).value_counts()[-10:]

#Removing low frequency words
lo_frew = list(lo_frew.index)
train['tweet'] = train['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in lo_frew))
train['tweet'].head()

#Special Casee
#Spelling correction
#train['tweet'][:5].apply(lambda x: str(TextBlob(x).correct()))

#Tokenization
TextBlob(train['tweet']).words

#Stemming
st = PorterStemmer()
train['tweet'][:5].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))
#dysfunctional has been transformed into dysfunct

#lemmatization
train['tweet'] = train['tweet'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
train['tweet'].head()

#Advance Text Processing
TextBlob(train['tweet'][0]).ngrams(3)

#Term frequency
tf1 = (train['tweet'][1:2]).apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0).reset_index()
tf1.columns = ['words','tf']

#Inverse Document Frequency
for i,word in enumerate(tf1['words']):
  tf1.loc[i, 'idf'] = np.log(train.shape[0]/(len(train[train['tweet'].str.contains(word)])))
  
tf1

#Term Frequency – Inverse Document Frequency (TF-IDF)
tf1['tfidf'] = tf1['tf'] * tf1['idf']
tf1


tfidf = TfidfVectorizer(max_features=1000, lowercase=True, analyzer='word',stop_words= 'english',ngram_range=(1,1))
train_vect = tfidf.fit_transform(train['tweet'])


#Bag of Words

bow = CountVectorizer(max_features=1000, lowercase=True, ngram_range=(1,1),analyzer = "word")
train_bow = bow.fit_transform(train['tweet'])
train_bow

#Checking Sentiments
train['sentiment'] = train['tweet'].apply(lambda x: TextBlob(x).sentiment[0] )
train[['tweet','sentiment']].head()

#Word Embeddings
glove_input_file = 'glove.6B.100d.txt'
word2vec_output_file = 'glove.6B.100d.txt.word2vec'
glove2word2vec(glove_input_file, word2vec_output_file)

#load the above word2vec file as a model.
filename = 'glove.6B.100d.txt.word2vec'
model = KeyedVectors.load_word2vec_format(filename, binary=False)


