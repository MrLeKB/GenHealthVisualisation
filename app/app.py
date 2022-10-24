from flask import Flask, render_template
import threading
import sqlalchemy
from sqlalchemy import create_engine
import psycopg2
from psycopg2 import Error
import pandas as pd
from datetime import datetime


#imports
import pandas as pd
import json

import nltk
#nltk.download("all")
#Sentiment Analysis
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import FreqDist
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
from nltk.corpus import wordnet
import matplotlib.pyplot as plt
import re
import string
import math
# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
# spacy for lemmatization
import spacy
# Plotting tools
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis # don't skip this
import matplotlib.pyplot as plt
import sqlalchemy
from sqlalchemy import create_engine

# from signal import signal, SIGPIPE, SIG_DFL  
# signal(SIGPIPE,SIG_DFL) 

app = Flask(__name__)

engine = create_engine('postgresql+psycopg2://pvahbfuxwqhvpq:3837ad2efc075df162ec73cc54d80e55b1aff7a1098b0eb5916502107f4b97bb@ec2-34-194-40-194.compute-1.amazonaws.com/dcrh9u79n7rtsa')


@app.route("/")
def index():
    for thread in threading.enumerate(): 
        print(thread.name)
    return render_template("FinalVisualisation.html")

# def checkScheduler():
#      for thread in threading.enumerate(): 
#           print(thread.name)


#      x=0
#      while True:
#           if x>10000000:
#               break
#           if x% 100000 == 0:
#                print("Log--------------%s" % (x))   
#           x+=1

@app.route("/request_html/<date>")
def request_html(date):
    print("Retrieving html file from {}".format(date))
    #dbConnection = engine.connect()
    # Read data from PostgreSQL database table and load into a DataFrame instance
    # dataFrame = pd.read_sql("select * from \"html_table\"", dbConnection)
    #dbConnection.close()
    # pd.set_option('display.expand_frame_repr', False)
    # html_str = dataFrame.iloc[0,0]

    html_str="<div>helloe</div>"
    print("requested_html")
    return html_str

@app.route("/initialise_scraper/<date>")
def initialise_scraper(date):
    print("test")
    return None

def initialise_analysis(date):
    print("Log---Initiated analysis {}".format(date))
    currentYearMonth="2022-09-01"
    #currentYearMonth=datetime.today().replace(day=1).strftime('%Y-%m-%d')
    print("Scheduler:Analysis Request Input {}".format(currentYearMonth))
    #initialise_analysis(currentYearMonth)
    analysisThread = threading.Thread(target=analysis,args=(currentYearMonth,))
    try:
        analysisThread.start()
        print("Log---Analysis completed{}".format(date))
    except:
        print("Log---Analysis failed{}".format(date))
    return None
def analysis(date):    # Read data from PostgreSQL database table and load into a DataFrame instance
    dbConnection = engine.connect()
    dataFrame = pd.read_sql("select * from \"json_table\" where DATE(timestamp) = '{}'".format(date), dbConnection)
    dbConnection.close()
    print("Log---Retrieved Data from json_table for {}".format(date))
    #Sentiment Analysis

    #Converting json str to df
    json_str = dataFrame.iloc[0,0]
    json_df = pd.read_json(json_str, orient ='index')
    df = json_df.copy()

    #analysis
    print("Log---Conducting Sentiment Analysis")
    sid = SentimentIntensityAnalyzer()
    #score
    df['score'] = df['Content'].apply(lambda text:sid.polarity_scores(str(text)))

    #compound score
    df['compound']  = df['score'].apply(lambda score_dict: score_dict['compound'])

    #compound label
    df['comp_score'] = df['compound'].apply(lambda c: 'pos' if 0.3<c<=1 else("neu" if -0.3<=c<=0.3 else "neg"))

    df.dropna(inplace=True)
    
    #Topic Modelling
    print("Log---Preprocessing data for topic modelling")
    #cleaning post sentiment analysis
    def clean_text(text):
        #convert to lowercasing, remove non words and remove digits
        text = text.lower()
        text = re.sub(r'\W', ' ', text)
        text = re.sub(r'\d', ' ', text)

        return text

    # Dictionary of English Contractions
    contractions_dict = { "ain't": "are not","'s":" is","aren't": "are not",
                        "can't": "cannot","can't've": "cannot have",
                        "'cause": "because","could've": "could have","couldn't": "could not",
                        "couldn't've": "could not have", "didn't": "did not","doesn't": "does not",
                        "don't": "do not","hadn't": "had not","hadn't've": "had not have",
                        "hasn't": "has not","haven't": "have not","he'd": "he would",
                        "he'd've": "he would have","he'll": "he will", "he'll've": "he will have",
                        "how'd": "how did","how'd'y": "how do you","how'll": "how will",
                        "i'd": "i would", "i'd've": "i would have","i'll": "i will",
                        "i'll've": "i will have","i'm": "i am","i've": "i have", "isn't": "is not",
                        "it'd": "it would","it'd've": "it would have","it'll": "it will",
                        "it'll've": "it will have", "let's": "let us","ma'am": "madam",
                        "mayn't": "may not","might've": "might have","mightn't": "might not", 
                        "mightn't've": "might not have","must've": "must have","mustn't": "must not",
                        "mustn't've": "must not have", "needn't": "need not",
                        "needn't've": "need not have","o'clock": "of the clock","oughtn't": "ought not",
                        "oughtn't've": "ought not have","shan't": "shall not","sha'n't": "shall not",
                        "shan't've": "shall not have","she'd": "she would","she'd've": "she would have",
                        "she'll": "she will", "she'll've": "she will have","should've": "should have",
                        "shouldn't": "should not", "shouldn't've": "should not have","so've": "so have",
                        "that'd": "that would","that'd've": "that would have", "there'd": "there would",
                        "there'd've": "there would have", "they'd": "they would",
                        "they'd've": "they would have","they'll": "they will",
                        "they'll've": "they will have", "they're": "they are","they've": "they have",
                        "to've": "to have","wasn't": "was not","we'd": "we would",
                        "we'd've": "we would have","we'll": "we will","we'll've": "we will have",
                        "we're": "we are","we've": "we have", "weren't": "were not","what'll": "what will",
                        "what'll've": "what will have","what're": "what are", "what've": "what have",
                        "when've": "when have","where'd": "where did", "where've": "where have",
                        "who'll": "who will","who'll've": "who will have","who've": "who have",
                        "why've": "why have","will've": "will have","won't": "will not",
                        "won't've": "will not have", "would've": "would have","wouldn't": "would not",
                        "wouldn't've": "would not have","y'all": "you all", "y'all'd": "you all would",
                        "y'all'd've": "you all would have","y'all're": "you all are",
                        "y'all've": "you all have", "you'd": "you would","you'd've": "you would have",
                        "you'll": "you will","you'll've": "you will have", "you're": "you are",
                        "you've": "you have"}

    # Regular expression for finding contractions
    contractions_re=re.compile('(%s)' % '|'.join(contractions_dict.keys()))

    # Function for expanding contractions
    def expand_contractions(text,contractions_dict=contractions_dict):
        def replace(match):
            return contractions_dict[match.group(0)]
        return contractions_re.sub(replace, text)

    #Pre-Processing - Expand contractions, Lowercase and removal of punctuations
    #Expand Contractions
    df['Content'] = df['Content'].apply(lambda x:expand_contractions(x))

    #remove punctuations, numbers,lowercase
    df['Content'] = df['Content'].apply(clean_text)

    #Modelling
    #Convert List into bag of words
    def sent_to_words(sentences):
        for sentence in sentences:
            yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

    data_words = list(sent_to_words(df['Content']))

    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=30) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=30)  

    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    # Define functions for stopwords, bigrams, trigrams and lemmatization
    def remove_stopwords(texts):
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]

    def make_trigrams(texts):
        return [trigram_mod[bigram_mod[doc]] for doc in texts]

    def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent)) 
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out

    # Remove Stop Words
    data_words_nostops = remove_stopwords(data_words)

    # Form Bigrams
    data_words_bigrams = make_bigrams(data_words_nostops)

    # Initiate spacy for lemmatization
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    # Do lemmatization keeping only noun, adj, vb, adv
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    print("Log---Data cleaning phase 2 completed")
    # Create Dictionary
    # id2word = corpora.Dictionary(data_lemmatized)
    id2word = corpora.Dictionary(data_words_bigrams)
    # Create Corpus
    texts = data_words_bigrams

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]    
    
    def compute_coherence_values(dictionary, corpus, texts, limit, start=4, step=2):
        print("Log---Caculating Optimal number of topics")
        coherence_values = []
        model_list = []
        for num_topics in range(start, limit, step):
            # model = gensim.models.wrappers.LdaMallet(mallet_path, num_topics=num_topics, id2word=id2word)
            model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=num_topics, 
                                            random_state=100)
            model_list.append(model)
            coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
            coherence_values.append(coherencemodel.get_coherence())
        return model_list, coherence_values

    limit=16; start=4; step=2
    # Can take a long time to run.
    model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=texts, limit=limit)

    # Print the coherence scores
    coherence_score_dict = {}
    x = range(start, limit, step)
    for m, cv in zip(x, coherence_values):
        coherence_score_dict[m] = round(cv, 4)

    def calculate_optimal_coherence(dict_cv):
        cv = 0
        current_cv = dict_cv[4]
        count = 0
        for key, val in dict_cv.items():
            if val<current_cv:
                return count-1
            else:
                current_cv = val
                count+=1
        return count-1

    optimal = calculate_optimal_coherence(coherence_score_dict)

    # Select the model and print the topics
    optimal_model = model_list[optimal]
    model_topics = optimal_model.show_topics(formatted=False)
    print("Log---Selected optimal topics")
    #Merging Sentiment Analysis with Topic Modelling
    def format_topics_sentences(ldamodel=optimal_model, corpus=corpus, texts=df):
        # Init output
        sent_topics_df = pd.DataFrame()

        # Get main topic in each document
        for i, row in enumerate(ldamodel[corpus]):
            row = sorted(row, key=lambda x: (x[1]), reverse=True)
            # Get the Dominant topic, Perc Contribution and Keywords for each document
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:  # => dominant topic
                    wp = ldamodel.show_topic(topic_num)
                    topic_keywords = ", ".join([word for word, prop in wp])
                    sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
                else:
                    break
        sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

        # Add original text to the end of the output
        contents = texts.squeeze()
        sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
        return(sent_topics_df)

    #Dominant Topic
    print("Log---Preparing Dominant Topics")
    df_topic_sents_keywords = format_topics_sentences(ldamodel=optimal_model, corpus=corpus, texts=df)
    # Format
    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Content', 'Datetime', 'orginal_text','score','compound','comp_score']

    #Dominant Topic Docs
    df_dominant_docs = df_dominant_topic.copy()
    top_dominant_topic_docs = {}
    no_of_topics = list(df_dominant_docs.Dominant_Topic.unique())

    for topic_no in no_of_topics:
        if not math.isnan(topic_no):
            top_dominant_topic_docs[topic_no] = {}
            #Overall
            df_topic = df_dominant_docs.copy()
            df_topic = df_topic[df_topic["Dominant_Topic"]==topic_no]
            df_topic.sort_values(by='Topic_Perc_Contrib', ascending=False, inplace=True)
            top_dominant_topic_docs[topic_no]['mean'] = list(df_topic["orginal_text"][:3])
            
            #Positive
            df_pos = df_dominant_docs.copy()
            df_pos = df_pos[df_pos["Dominant_Topic"]==topic_no]
            df_pos = df_pos[df_pos["comp_score"]=="pos"]
            df_pos.sort_values(by='Topic_Perc_Contrib', ascending=False, inplace=True)
            top_dominant_topic_docs[topic_no]['pos'] = list(df_pos["orginal_text"][:3])
                                                                    
            #Neutral
            df_neu = df_dominant_docs.copy()
            df_neu = df_neu[df_neu["Dominant_Topic"]==topic_no]
            df_neu = df_neu[df_neu["comp_score"]=="neu"]
            df_neu.sort_values(by='Topic_Perc_Contrib', ascending=False, inplace=True)
            top_dominant_topic_docs[topic_no]['neu'] = list(df_neu["orginal_text"][:3])
            
            #Negative
            df_neg = df_dominant_docs.copy()
            df_neg = df_neg[df_neg["Dominant_Topic"]==topic_no]
            df_neg = df_neg[df_neg["comp_score"]=="neg"]
            df_neg.sort_values(by='Topic_Perc_Contrib', ascending=False, inplace=True)
            top_dominant_topic_docs[topic_no]['neg'] = list(df_neg["orginal_text"][:3])

    #Topic Sentiments
    print("Log---Preparing Topic Sentiment")
    topic_sentiment={}
    for i in range(len(df_dominant_topic.groupby(['Dominant_Topic']).mean())):
        topic_df=df_dominant_topic[df_dominant_topic['Dominant_Topic']==i]
        mean=topic_df['compound'].mean()
        sd=topic_df['compound'].std()
        size= len(topic_df)
        pos= round(len(topic_df[(topic_df['compound']>0.3)&(topic_df['compound']<=1)])/size,2)
        neg= round(len(topic_df[(topic_df['compound']>=-1)&(topic_df['compound']<-0.3)]) /size,2)
        neu= round(len(topic_df[(topic_df['compound']>=-0.3)&(topic_df['compound']<=0.3)]) /size,2)

        topic_sentiment[i] = {}
        topic_sentiment[i]['mean'] = [mean, top_dominant_topic_docs[i]['mean']]
        topic_sentiment[i]['pos'] = [pos, top_dominant_topic_docs[i]['pos']]
        topic_sentiment[i]['neg'] = [neg, top_dominant_topic_docs[i]['neg']]
        topic_sentiment[i]['neu'] = [neu, top_dominant_topic_docs[i]['neu']]

    #Create Visualisation
    sentiment = json.dumps(topic_sentiment).replace("https://", "")
    # Visualize the topics
    vis = gensimvis.prepare(topic_model=optimal_model, corpus = corpus, dictionary = id2word, sentiment=sentiment)
    vis_html = pyLDAvis.prepared_data_to_html(vis)
    print("Log---prepared html-------")
    #Send HTML to database
    currDate = "test1"

    try:
        # Connect to an existing database
        connection = psycopg2.connect(user="pvahbfuxwqhvpq",
                                    password="3837ad2efc075df162ec73cc54d80e55b1aff7a1098b0eb5916502107f4b97bb",
                                    host="ec2-34-194-40-194.compute-1.amazonaws.com",
                                    port="5432",
                                    database="dcrh9u79n7rtsa")

        # Create a cursor to perform database operations
        cursor = connection.cursor()
        cursor.execute("INSERT INTO html_table (html_string, timestamp) VALUES (%s, %s)", (vis_html, currDate))
        connection.commit()
        print("1 item inserted successfully")

    except (Exception, Error) as error:
        print("Error while connecting to PostgreSQL", error)
    finally:
        if (connection):
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")
     
if __name__ == '__main__':

    app.run(host='0.0.0.0', port=5000, debug=True)
