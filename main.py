
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import CubicSpline
import statsmodels.api as sm
import random

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from collections import defaultdict, Counter

import time
from datetime import datetime, timedelta

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)
random.seed(123)

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')


### hyperparameters/tuning parameters
min_word_count = 5  # minimum appearances of word
K = 10              # number of topics
alpha = 0.5         # document-topic distribution hyperparameter
beta = 0.5          # topic-word distribution hyperparameter
vector_size = 5     # embedding vector size
L = 10              # window size


### preprocess function
def preprocess_text(text):
    # tokenize the text
    tokens = nltk.word_tokenize(text) 

    # remove stopwords from ALL languages
    stop_words = set(stopwords.words(stopwords.fileids())) 
    tokens = [word for word in tokens if word.lower() not in stop_words]

    # remove punctuation and symbols
    tokens = [word for word in tokens if word.isalpha()] 

    # lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # remove stopwords
    tokens = [word for word in tokens if word.lower() not in stop_words]

    return tokens


### print topic distribution for each document and top 10 keywords for each topic
def print_topic_distributions(doc_topic_distribution, topic_word_distribution, vocab, global_lyrics):
    for d, dist in enumerate(doc_topic_distribution):
        print(global_lyrics[d])
        print(f"Document {d+1} Topic Distribution:")
        for topic_idx, prob in enumerate(dist):
            print(f"Topic {topic_idx}: {prob:.4f}")
        print("\n")

    for t, dist in enumerate(topic_word_distribution):
        top_indices = dist.argsort()[-10:][::-1]  # Indices of top 10 words for this topic
        top_words = [vocab[i] for i in top_indices]
        print(f"Top 10 Keywords for Topic {t}: {', '.join(top_words)}")
        print("\n")


### latent dirichlet allocation function
def lda(df):
    # remove duplicates, initial spaces, to list
    global_lyrics = df['lyrics'].drop_duplicates()
    global_lyrics = global_lyrics.apply(lambda x: x.lstrip())
    global_lyrics = global_lyrics.tolist()

    # preprocess the texts
    docs = []
    for t in global_lyrics:
        processed_t = preprocess_text(t)
        docs.append(processed_t)

    text = []
    for t in docs:
        text.append(t)

    words = [word.lower() for sentence in text for word in sentence]

    # count the occurrences of each word
    word_counts = Counter(words)

    # filter words that appear more than min value
    filtered_words = [word for word in words if word_counts[word] > min_word_count]

    # join filtered words back into a string
    vocab = list(set(filtered_words))

    # vocabulary build
    V = len(vocab)

    # initialize count matrices and topic assignments with Dirichlet Priors
    doc_topic_count = np.zeros((len(docs), K)) + alpha
    topic_word_count = np.zeros((K, V)) + beta
    topic_count = np.zeros(K) + V * beta
    word_topic_assignment = [[0 for _ in doc] for doc in docs]

    # randomly assign initial topics to each word -> reduce dimensions to filtered words, otherwise too long
    for d, doc in enumerate(docs):
        for w, word in enumerate(set(vocab)&set(doc)):
            initial_topic = random.randint(0, K-1)
            word_topic_assignment[d][w] = initial_topic
            doc_topic_count[d][initial_topic] += 1
            topic_word_count[initial_topic][vocab.index(word)] += 1
            topic_count[initial_topic] += 1

    # gibbs sampling
    for it in range(100):
        start = time.time()
        for d, doc in enumerate(docs):
            for w, word in enumerate(set(vocab)&set(doc)):
                # get current word's topic
                current_topic = word_topic_assignment[d][w]

                # decrement counts for current word's topic
                doc_topic_count[d][current_topic] -= 1
                topic_word_count[current_topic][vocab.index(word)] -= 1
                topic_count[current_topic] -= 1

                # calculate probabilities for topics
                topic_probs = (doc_topic_count[d] * topic_word_count[:, vocab.index(word)]) / topic_count
                topic_probs = np.divide(topic_probs, np.sum(topic_probs))

                # sample a new topic based on the probabilities
                new_topic = np.random.choice(np.arange(K), p=topic_probs)

                # assign new topic to the word
                word_topic_assignment[d][w] = new_topic

                # increment counts for new topic
                doc_topic_count[d][new_topic] += 1
                topic_word_count[new_topic][vocab.index(word)] += 1
                topic_count[new_topic] += 1
        end = time.time()
        secs = end - start
        print(f"Iteration {it+1} completed: {secs/60:.4f} minutes elapsed.")

    # normalization
    topic_word_distribution = topic_word_count / np.sum(topic_word_count, axis=1, keepdims=True)

    # adjust normalization for doc_topic_distribution
    doc_topic_distribution = (doc_topic_count - alpha) / np.sum(doc_topic_count - alpha, axis=1, keepdims=True)

    # assuming 'doc_topic_distribution' and 'topic_word_distribution' are the final distributions from the LDA code
    print_topic_distributions(doc_topic_distribution, topic_word_distribution, vocab, global_lyrics)

    return global_lyrics, doc_topic_distribution, topic_word_distribution, vocab


### load datasets
# usa
df_us = pd.read_csv(r'./data/by_country/us')

# southern cone countries
df_ar = pd.read_csv(r'./data/by_country/ar')
df_cl = pd.read_csv(r'./data/by_country/cl')
df_uy = pd.read_csv(r'./data/by_country/uy')
df_py = pd.read_csv(r'./data/by_country/py')
df_sc = pd.concat([df_ar, df_cl, df_uy, df_py], ignore_index=True)

# france
df_fr = pd.read_csv(r'./data/by_country/fr')

### clean some (very frequent) stuff
df_us['lyrics'] = df_us['lyrics'].str.replace('\n', ' ')
df_us['lyrics'] = df_us['lyrics'].str.replace('\'', "'")
df_us['lyrics'] = df_us['lyrics'].str.replace('\u2005', ' ')
df_us['lyrics'] = df_us['lyrics'].str.replace(r'\[.*?\]', '', regex=True)

df_sc['lyrics'] = df_sc['lyrics'].str.replace('\n', ' ')
df_sc['lyrics'] = df_sc['lyrics'].str.replace('\'', "'")
df_sc['lyrics'] = df_sc['lyrics'].str.replace('\u2005', ' ')
df_sc['lyrics'] = df_sc['lyrics'].str.replace(r'\[.*?\]', '', regex=True)

df_fr['lyrics'] = df_fr['lyrics'].str.replace('\n', ' ')
df_fr['lyrics'] = df_fr['lyrics'].str.replace('\'', "'")
df_fr['lyrics'] = df_fr['lyrics'].str.replace('\u2005', ' ')
df_fr['lyrics'] = df_fr['lyrics'].str.replace(r'\[.*?\]', '', regex=True) #!

### run the lda model
global_lyrics_sc, doc_topic_distribution_sc, topic_word_distribution_sc, vocab_sc = lda(df_sc)
global_lyrics_fr, doc_topic_distribution_fr, topic_word_distribution_fr, vocab_fr = lda(df_fr)
global_lyrics_us, doc_topic_distribution_us, topic_word_distribution_us, vocab_us = lda(df_us)

### save the results
np.savetxt('global_lyrics_us.txt', doc_topic_distribution_fr, encoding='utf-8')
np.savetxt('doc_topic_distribution_us.txt', doc_topic_distribution_fr, encoding='utf-8')
np.savetxt('topic_word_distribution_us.txt', topic_word_distribution_fr, encoding='utf-8')
np.savetxt('vocab_us.txt', np.array(vocab_fr, dtype=str), fmt='%s', encoding='utf-8')

np.savetxt('global_lyrics_sc.txt', doc_topic_distribution_sc, encoding='utf-8')
np.savetxt('doc_topic_distribution_sc.txt', doc_topic_distribution_sc, encoding='utf-8')
np.savetxt('topic_word_distribution_sc.txt', topic_word_distribution_sc, encoding='utf-8')
np.savetxt('vocab_sc.txt', np.array(vocab_sc, dtype=str), fmt='%s', encoding='utf-8')

np.savetxt('global_lyrics_fr.txt', doc_topic_distribution_fr, encoding='utf-8')
np.savetxt('doc_topic_distribution_fr.txt', doc_topic_distribution_fr, encoding='utf-8')
np.savetxt('topic_word_distribution_fr.txt', topic_word_distribution_fr, encoding='utf-8')
np.savetxt('vocab_fr.txt', np.array(vocab_fr, dtype=str), fmt='%s', encoding='utf-8')

### merge lyrics with topic distribution
df_merged_us = pd.DataFrame({'lyrics': global_lyrics_us})
df_merged_sc = pd.DataFrame({'lyrics': global_lyrics_sc})
df_merged_fr = pd.DataFrame({'lyrics': global_lyrics_fr})

for i in range(K):
    df_merged_us[f'topic_{i}'] = doc_topic_distribution_us[:, i]
    df_merged_sc[f'topic_{i}'] = doc_topic_distribution_sc[:, i]
    df_merged_fr[f'topic_{i}'] = doc_topic_distribution_fr[:, i]

df_merged_us = pd.merge(df_us, df_merged_us, on='lyrics', how='inner')
df_merged_sc = pd.merge(df_sc, df_merged_sc, on='lyrics', how='inner')
df_merged_fr = pd.merge(df_fr, df_merged_fr, on='lyrics', how='inner')

### create dates
df_merged_us['date'] = pd.to_datetime(df_merged_us[['day', 'month', 'year']])
df_merged_sc['date'] = pd.to_datetime(df_merged_sc[['day', 'month', 'year']])
df_merged_fr['date'] = pd.to_datetime(df_merged_fr[['day', 'month', 'year']])

### construct a database with average topic values for each unique date
date_topic_avg_us = df_merged_us.groupby('date')[[f'topic_{i}' for i in range(K)]].mean().reset_index()
date_topic_avg_sc = df_merged_sc.groupby('date')[[f'topic_{i}' for i in range(K)]].mean().reset_index()
date_topic_avg_fr = df_merged_fr.groupby('date')[[f'topic_{i}' for i in range(K)]].mean().reset_index()

### plot the values of each topic across time
[plt.plot(date_topic_avg_us['date'], date_topic_avg_us[f'topic_{i}'], label=f'Topic {i+1}') for i in range(K)]
plt.xlabel('Date')
plt.ylabel('Topic Weights')
plt.title('Topic Weights Across Time, Southern Cone')
plt.legend()
plt.savefig('us_topics.png')
plt.show()

[plt.plot(date_topic_avg_sc['date'], date_topic_avg_sc[f'topic_{i}'], label=f'Topic {i+1}') for i in range(K)]
plt.xlabel('Date')
plt.ylabel('Topic Weights')
plt.title('Topic Weights Across Time, Southern Cone')
plt.legend()
plt.savefig('sc_topics.png')
plt.show()

[plt.plot(date_topic_avg_fr['date'], date_topic_avg_fr[f'topic_{i}'], label=f'Topic {i+1}') for i in range(K)]
plt.xlabel('Date')
plt.ylabel('Topic Weights')
plt.title('Topic Weights Across Time, France')
plt.legend()
plt.savefig('fr_topics.png')
plt.show()

### create romantic and conflicting topics (after looking at the keywords)
date_topic_avg_us['romantic'] = date_topic_avg_us['topic_0'] + date_topic_avg_us['topic_1'] + date_topic_avg_us['topic_5'] + date_topic_avg_us['topic_7']
date_topic_avg_us['conflicting'] = date_topic_avg_us['topic_2'] + date_topic_avg_us['topic_8'] + date_topic_avg_us['topic_9'] + date_topic_avg_us['topic_3']

date_topic_avg_sc['romantic'] = date_topic_avg_sc['topic_0'] + date_topic_avg_sc['topic_1'] + date_topic_avg_sc['topic_5'] + date_topic_avg_sc['topic_6'] + date_topic_avg_sc['topic_8']  
date_topic_avg_sc['conflicting'] = date_topic_avg_sc['topic_4'] + date_topic_avg_sc['topic_7'] 

date_topic_avg_fr['romantic'] = date_topic_avg_fr['topic_1'] + date_topic_avg_fr['topic_3'] + date_topic_avg_fr['topic_8'] 
date_topic_avg_fr['conflicting'] = date_topic_avg_fr['topic_4'] + date_topic_avg_fr['topic_7'] 

### interpolate observations for all days
date_range_us = pd.date_range(date_topic_avg_us['date'].min(), date_topic_avg_us['date'].max(), freq='D')
interpolated_topic_avg_us = date_topic_avg_us.set_index('date').reindex(date_range_us).interpolate().reset_index()

date_range_sc = pd.date_range(date_topic_avg_sc['date'].min(), date_topic_avg_sc['date'].max(), freq='D')
interpolated_topic_avg_sc = date_topic_avg_sc.set_index('date').reindex(date_range_sc).interpolate().reset_index()

date_range_fr = pd.date_range(date_topic_avg_fr['date'].min(), date_topic_avg_fr['date'].max(), freq='D')
interpolated_topic_avg_fr = date_topic_avg_fr.set_index('date').reindex(date_range_fr).interpolate().reset_index()

### dates
trump_election = datetime.strptime('2016-11-08', '%Y-%m-%d')
first_lockdown = datetime.strptime('2020-03-19', '%Y-%m-%d')
floyd_death = datetime.strptime('2020-05-25', '%Y-%m-%d')
capitol_storm = datetime.strptime('2021-01-06', '%Y-%m-%d')

hebdo_attack = datetime.strptime('2015-01-07','%Y-%m-%d')
bataclan_attack = datetime.strptime('2015-11-13','%Y-%m-%d')
nice_attack = datetime.strptime('2016-07-14','%Y-%m-%d')

''' 
# for some extension, I'm running out of time
ramblas_attack = datetime.strptime('2017-08-17','%Y-%m-%d')
manchester_attack = datetime.strptime('2017-05-22','%Y-%m-%d')
berlin_attack = datetime.strptime('2016-12-19','%Y-%m-%d')
brussels_attack = datetime.strptime('2016-03-22','%Y-%m-%d')
munich_attack = datetime.strptime('2016-07-22','%Y-%m-%d')
'''

### pooled topics plot
plt.plot(interpolated_topic_avg_us['index'], interpolated_topic_avg_us['romantic'], label='Romantic', color='green')
plt.plot(interpolated_topic_avg_us['index'], interpolated_topic_avg_us['conflicting'], label='Conflicting', color='red')
plt.axvline(x=trump_election, color='black', linestyle='--')
plt.axvline(x=first_lockdown, color='black', linestyle='--')
plt.axvline(x=floyd_death, color='black', linestyle='--')
plt.axvline(x=capitol_storm, color='black', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Topic Relative Weights')
plt.legend()
plt.savefig('us_topics_pn.png')
plt.show()

plt.plot(interpolated_topic_avg_sc['index'], interpolated_topic_avg_sc['romantic'], label='Romantic', color='green')
plt.plot(interpolated_topic_avg_sc['index'], interpolated_topic_avg_sc['conflicting'], label='Conflicting', color='red')
plt.axvline(x=hebdo_attack, color='black', linestyle='--')
plt.axvline(x=bataclan_attack, color='black', linestyle='--')
plt.axvline(x=nice_attack, color='black', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Topic Relative Weights')
plt.legend()
plt.savefig('sc_topics_pn.png')
plt.show()

plt.plot(interpolated_topic_avg_fr['index'], interpolated_topic_avg_fr['romantic'], label='Romantic', color='green')
plt.plot(interpolated_topic_avg_fr['index'], interpolated_topic_avg_fr['conflicting'], label='Conflicting', color='red')
plt.axvline(x=hebdo_attack, color='black', linestyle='--')
plt.axvline(x=bataclan_attack, color='black', linestyle='--')
plt.axvline(x=nice_attack, color='black', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Topic Relative Weights')
plt.legend()
plt.savefig('fr_topics_pn.png')
plt.show()

### splines
cs_sc_rom = CubicSpline(date_topic_avg_sc['date'], date_topic_avg_sc['romantic'])
cs_sc_con = CubicSpline(date_topic_avg_sc['date'], date_topic_avg_sc['conflicting'])

cs_fr_rom = CubicSpline(date_topic_avg_fr['date'], date_topic_avg_fr['romantic'])
cs_fr_con = CubicSpline(date_topic_avg_fr['date'], date_topic_avg_fr['conflicting'])

### plot the smoothed versions
plt.plot(interpolated_topic_avg_sc['index'], cs_sc_rom(interpolated_topic_avg_sc['index']), label='Romantic', color='green')
plt.plot(interpolated_topic_avg_sc['index'], cs_sc_con(interpolated_topic_avg_sc['index']), label='Conflicting', color='red')
plt.axvline(x=hebdo_attack, color='black', linestyle='--')
plt.axvline(x=bataclan_attack, color='black', linestyle='--')
plt.axvline(x=nice_attack, color='black', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Topic Weight')
plt.savefig('sc_topics_pn_spline.png')
plt.show()

plt.plot(interpolated_topic_avg_fr['index'], cs_fr_rom(interpolated_topic_avg_fr['index']), label='Romantic', color='green')
plt.plot(interpolated_topic_avg_fr['index'], cs_fr_con(interpolated_topic_avg_fr['index']), label='Conflicting', color='red')
plt.axvline(x=hebdo_attack, color='black', linestyle='--')
plt.axvline(x=bataclan_attack, color='black', linestyle='--')
plt.axvline(x=nice_attack, color='black', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Topic Weight')
plt.savefig('fr_topics_pn_spline.png')
plt.show()

### prepare data for regression analysis
# add the spline values
interpolated_topic_avg_sc['romantic_spline'] = cs_sc_rom(interpolated_topic_avg_sc['index'])
interpolated_topic_avg_sc['conflicting_spline'] = cs_sc_con(interpolated_topic_avg_sc['index'])

interpolated_topic_avg_fr['romantic_spline'] = cs_fr_rom(interpolated_topic_avg_fr['index'])
interpolated_topic_avg_fr['conflicting_spline'] = cs_fr_con(interpolated_topic_avg_fr['index'])

# add the country indicator
interpolated_topic_avg_sc['france'] = 0
interpolated_topic_avg_fr['france'] = 1

# merge the datasets
merged_inter_data = pd.concat([interpolated_topic_avg_fr, interpolated_topic_avg_sc], ignore_index=True)

# add the attack indicators
merged_inter_data['hebdo_indicator'] = np.where(merged_inter_data['index'] > hebdo_attack, 1, 0)
merged_inter_data['bataclan_indicator'] = np.where(merged_inter_data['index'] > bataclan_attack, 1, 0)
merged_inter_data['nice_indicator'] = np.where(merged_inter_data['index'] > nice_attack, 1, 0)

# construct design matrix
X_hebdo = np.column_stack((merged_inter_data['hebdo_indicator'], merged_inter_data['france'], merged_inter_data['hebdo_indicator']*merged_inter_data['france']))
X_bataclan = np.column_stack((merged_inter_data['bataclan_indicator'], merged_inter_data['france'], merged_inter_data['bataclan_indicator']*merged_inter_data['france']))
X_nice = np.column_stack((merged_inter_data['nice_indicator'], merged_inter_data['france'], merged_inter_data['nice_indicator']*merged_inter_data['france']))

# add constant
X_hebdo = sm.add_constant(X_hebdo)
X_bataclan = sm.add_constant(X_bataclan)
X_nice = sm.add_constant(X_nice)

### diff-in-diff -- full sample
model_hebdo_rom = sm.OLS(merged_inter_data['romantic'],X_hebdo)
model_bataclan_rom = sm.OLS(merged_inter_data['romantic'],X_bataclan)
model_nice_rom = sm.OLS(merged_inter_data['romantic'],X_nice)

model_hebdo_con = sm.OLS(merged_inter_data['conflicting'],X_hebdo)
model_bataclan_con = sm.OLS(merged_inter_data['conflicting'],X_bataclan)
model_nice_con = sm.OLS(merged_inter_data['conflicting'],X_nice)

model_hebdo_rom_spline = sm.OLS(merged_inter_data['romantic_spline'],X_hebdo)
model_bataclan_rom_spline = sm.OLS(merged_inter_data['romantic_spline'],X_bataclan)
model_nice_rom_spline = sm.OLS(merged_inter_data['romantic_spline'],X_nice)

model_hebdo_con_spline = sm.OLS(merged_inter_data['conflicting_spline'],X_hebdo)
model_bataclan_con_spline = sm.OLS(merged_inter_data['conflicting_spline'],X_bataclan)
model_nice_con_spline = sm.OLS(merged_inter_data['conflicting_spline'],X_nice)

model_hebdo_rom.fit(cov_type='HC1').summary()
model_bataclan_rom.fit(cov_type='HC1').summary()
model_nice_rom.fit(cov_type='HC1').summary()

model_hebdo_con.fit(cov_type='HC1').summary()
model_bataclan_con.fit(cov_type='HC1').summary()
model_nice_con.fit(cov_type='HC1').summary()

model_hebdo_rom_spline.fit(cov_type='HC1').summary()
model_bataclan_rom_spline.fit(cov_type='HC1').summary()
model_nice_rom_spline.fit(cov_type='HC1').summary()

model_hebdo_con_spline.fit(cov_type='HC1').summary()
model_bataclan_con_spline.fit(cov_type='HC1').summary()
model_nice_con_spline.fit(cov_type='HC1').summary()

### diff-in-diff -- restricted sample

merged_inter_data_h = merged_inter_data[merged_inter_data['index'] < bataclan_attack]
merged_inter_data_b = merged_inter_data[merged_inter_data['index'] < nice_attack]
days_bn  = divmod((nice_attack - bataclan_attack).total_seconds() , 86400)[0] 
merged_inter_data_n = merged_inter_data[merged_inter_data['index'] < nice_attack + timedelta(days=days_bn)]

X_hebdo_nn = np.column_stack((merged_inter_data_h['hebdo_indicator'], merged_inter_data_h['france'], merged_inter_data_h['hebdo_indicator']*merged_inter_data_h['france']))
X_bataclan_nn = np.column_stack((merged_inter_data_b['hebdo_indicator'], merged_inter_data_b['france'], merged_inter_data_b['hebdo_indicator']*merged_inter_data_b['france']))
X_nice_nn = np.column_stack((merged_inter_data_n['hebdo_indicator'], merged_inter_data_n['france'], merged_inter_data_n['hebdo_indicator']*merged_inter_data_n['france']))

X_hebdo_nn = sm.add_constant(X_hebdo_nn)
X_bataclan_nn = sm.add_constant(X_bataclan_nn)
X_nice_nn = sm.add_constant(X_nice_nn)

model_hebdo_rom_nn= sm.OLS(merged_inter_data_h['romantic'],X_hebdo_nn)
model_bataclan_rom_nn = sm.OLS(merged_inter_data_b['romantic'],X_bataclan_nn)
model_nice_rom_nn = sm.OLS(merged_inter_data_n['romantic'],X_nice_nn)

model_hebdo_con_nn= sm.OLS(merged_inter_data_h['conflicting'],X_hebdo_nn)
model_bataclan_con_nn = sm.OLS(merged_inter_data_b['conflicting'],X_bataclan_nn)
model_nice_con_nn = sm.OLS(merged_inter_data_n['conflicting'],X_nice_nn)

model_hebdo_rom_nn.fit(cov_type='HC1').summary()
model_bataclan_rom_nn.fit(cov_type='HC1').summary()
model_nice_rom_nn.fit(cov_type='HC1').summary()

model_hebdo_con_nn.fit(cov_type='HC1').summary()
model_bataclan_con_nn.fit(cov_type='HC1').summary()
model_nice_con_nn.fit(cov_type='HC1').summary()

model_hebdo_rom_nn_spline = sm.OLS(merged_inter_data_h['romantic_spline'],X_hebdo_nn)
model_bataclan_rom_nn_spline  = sm.OLS(merged_inter_data_b['romantic_spline'],X_bataclan_nn)
model_nice_rom_nn_spline  = sm.OLS(merged_inter_data_n['romantic_spline'],X_nice_nn)

model_hebdo_con_nn_spline  = sm.OLS(merged_inter_data_h['conflicting_spline'],X_hebdo_nn)
model_bataclan_con_nn_spline  = sm.OLS(merged_inter_data_b['conflicting_spline'],X_bataclan_nn)
model_nice_con_nn_spline  = sm.OLS(merged_inter_data_n['conflicting_spline'],X_nice_nn)

model_hebdo_rom_nn_spline.fit(cov_type='HC1').summary()
model_bataclan_rom_nn_spline.fit(cov_type='HC1').summary()
model_nice_rom_nn_spline.fit(cov_type='HC1').summary()

model_hebdo_con_nn_spline.fit(cov_type='HC1').summary()
model_bataclan_con_nn_spline.fit(cov_type='HC1').summary()
model_nice_con_nn_spline.fit(cov_type='HC1').summary()