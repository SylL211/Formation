import numpy as np
import pickle
import nltk
nltk.download('brown')
nltk.download('stopwords')
from nltk.corpus import brown, stopwords
from scipy.cluster.vq import kmeans2
from sklearn.decomposition import PCA

for i in range(50):
    print (brown.words()[i],)
    
my_stopwords = set(stopwords.words('english'))
word_stream = [str(w).lower() for w in brown.words() if w.lower() not in my_stopwords]
my_word_stream = [w for w in word_stream if (len(w) > 1 and w.isalnum())]

my_word_stream[:20]

# Step 1: Get a list of words and their frequencies.

N = len(my_word_stream)
words = []
totals = {}
for i in range(1, N-1):
    w = my_word_stream[i]
    if w not in words:
        words.append(w)
        totals[w] = 0
    totals[w] = totals[w] + 1
    
# Step 2: Decide on the vocabulary.

vocab_words = [w for w in words if totals[w] > 19]
context_words = [w for w in words if totals[w] > 99]

# Step 3: Get co-occurrence counts.

def get_counts(window_size=2):
    counts = {}
    for w0 in vocab_words:
        counts[w0] = {}
    for i in range(window_size, N-window_size):
        w0 = my_word_stream[i]
        if w0 in vocab_words:
            for j in (list(range(-window_size,0)) + list(range(1,window_size+1))):
                w = my_word_stream[i+j]
                if w in context_words:
                    if w not in counts[w0].keys():
                        counts[w0][w] = 1
                    else:
                        counts[w0][w] = counts[w0][w] + 1
    return counts

counts = get_counts(window_size=2)

def get_co_occurrence_dictionary(counts):
    probs = {}
    for w0 in counts.keys():
        sum = 0
        for w in counts[w0].keys():
            sum = sum + counts[w0][w]
        if sum > 0:
            probs[w0] = {}
            for w in counts[w0].keys():
                probs[w0][w] = float(counts[w0][w])/float(sum)
    return probs

def get_context_word_distribution(counts):
    counts_context = {}
    sum_context = 0
    context_frequency = {}
    for w in context_words:
        counts_context[w] = 0
    for w0 in counts.keys():
        for w in counts[w0].keys():
            counts_context[w] = counts_context[w] + counts[w0][w]
            sum_context = sum_context + counts[w0][w]
    for w in context_words:
        context_frequency[w] = float(counts_context[w])/float(sum_context)
    return context_frequency

print ("Computing counts and distributions")
counts = get_counts(2)
probs = get_co_occurrence_dictionary(counts)
context_frequency = get_context_word_distribution(counts)
#
print ("Computing pointwise mutual information")
n_vocab = len(vocab_words)
n_context = len(context_words)
pmi = np.zeros((n_vocab, n_context))
for i in range(0, n_vocab):
    w0 = vocab_words[i]
    for w in probs[w0].keys():
        j = context_words.index(w)
        pmi[i,j] = max(0.0, np.log(probs[w0][w]) - np.log(context_frequency[w]))
        
pca = PCA(n_components=100)
vecs = pca.fit_transform(pmi)
for i in range(0,n_vocab):
    vecs[i] = vecs[i]/np.linalg.norm(vecs[i])
        
fd = open("embedding.pickle", "wb")
pickle.dump(vocab_words, fd)
pickle.dump(context_words, fd)
pickle.dump(vecs, fd)
fd.close()

def word_NN(w):
    if not(w in vocab_words):
        print ("Unknown word")
        return
    v = vecs[vocab_words.index(w)]
    neighbor = 0
    curr_dist = np.linalg.norm(v - vecs[0])
    for i in range(1, n_vocab):
        dist = np.linalg.norm(v - vecs[i])
        if (dist < curr_dist) and (dist > 0.0):
            neighbor = i
            curr_dist = dist
    return vocab_words[neighbor]