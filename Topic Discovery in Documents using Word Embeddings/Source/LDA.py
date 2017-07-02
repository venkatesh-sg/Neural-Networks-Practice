import nltk
import pickle
import os
import string
import numpy as np
import tensorflow as tf
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

Dirr="Data/" # Directory of Text files

# Tokenizer and Stemmer for words
stemmer = PorterStemmer()
def tokenizer(text):
    words = nltk.word_tokenize(text)
    stems = [i for i in words if i not in string.punctuation]
    stemmed_words = [stemmer.stem(i) for i in stems]
    return stemmed_words

# Cosine similarity between two tensors
def CosineSim(x,y):
    x=tf.cast(x,tf.float64)
    y = tf.cast(y, tf.float64)
    # Normalizing the tensors
    Normalised_x=tf.nn.l2_normalize(x, dim=0, epsilon=1e-12)
    Normalised_y = tf.nn.l2_normalize(y, dim=0, epsilon=1e-12)
    cos_sim=tf.reduce_sum(tf.multiply(Normalised_x, Normalised_y))
    return cos_sim

# Load the pre-trained embeddings
vocabulary_size=300000   # Vocab size of pre-trained embeddings
embedding_size=128       # embeddings size
with tf.Session() as session:
    print("Loading pre-trained embeddings")
    embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    embedding_saver = tf.train.Saver({embeddings})

    # Variable initialization
    init = tf.global_variables_initializer()
    # Loading embeddings to variable
    embedding_saver.restore(session, "Output/logs/model.ckpt")

    # Normalizing the embeddings
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm


    print("Loading Dictionary of vocabulary")
    # Loading Dictionary of vocabulary
    with open('Output/Dictionary.pickle', 'rb') as handle:
        dictionary = pickle.load(handle)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))


    # Query or X
    Query = "cancer"
    QueryInd = dictionary[Query]
    # Embeddings lookup for the Query
    print("Embeddings lookup for Query")
    Query_embeddings = tf.nn.embedding_lookup(normalized_embeddings, [QueryInd])


    for subdir, dirs, files in os.walk(Dirr):
        for file in files:
            vectorizer = CountVectorizer(tokenizer=tokenizer, stop_words='english', binary=False, dtype=np.float64)
            lines = (open(os.path.join(subdir, file)).read()).split(".")
            Tf = vectorizer.fit_transform(lines)
            feature_names = vectorizer.get_feature_names()

            # Latent Dirichlet Allocation
            LDA = LatentDirichletAllocation(n_topics=5, max_iter=100,learning_method="batch")
            LDA.fit(Tf)

            # Get top 10 words for each topic using LDA
            Topic_words = []
            for topic_id, topic in enumerate(LDA.components_):
                Topic_words.append([feature_names[i] for i in topic.argsort()[:-10 - 1:-1]])

            Topic_similarity=[]
            for topic in Topic_words:
                Word_similarity=[]
                for word in topic:
                    try:
                        ind=dictionary[word]
                    except:
                        continue

                    # Get embeddings for each word
                    word_embeddings = tf.nn.embedding_lookup(normalized_embeddings, [ind])

                    # Cosine Similarity between two words using embeddings
                    Word_CosinSim=CosineSim(Query_embeddings,word_embeddings)
                    Word_similarity.append(Word_CosinSim)

                # Mean over all the words_similarity for the topic
                Mean_sim = tf.reduce_mean(Word_similarity)
                Topic_similarity.append(Mean_sim)

            # Get the highest similar topic among all Topics
            Best_topic = tf.reduce_max(tf.cast(Topic_similarity,tf.float64))
            print("Document ",file," will explain about '",Query,"' with topic similarity of ",Best_topic.eval()," percent")
