from flask import Flask, render_template
import threading
import sqlalchemy
from sqlalchemy import create_engine
import psycopg2
from psycopg2 import Error
import pandas as pd
import datetime 
import json

import nltk
#Sentiment Analysis
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import re
import math
# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
# spacy for lemmatization
import spacy
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis # don't skip this
import snscrape.modules.twitter as sntwitter
import praw

import wordninja


import logging
logging.disable(logging.INFO)


app = Flask(__name__)

engine = create_engine('postgresql+psycopg2://pvahbfuxwqhvpq:3837ad2efc075df162ec73cc54d80e55b1aff7a1098b0eb5916502107f4b97bb@ec2-34-194-40-194.compute-1.amazonaws.com/dcrh9u79n7rtsa')


@app.route("/")
def index():
    return render_template("FinalVisualisation.html")

@app.route("/request_html/<date>")
def request_html(date):
    print("Log---Retrieving html file from {}".format(date))
    dbConnection = engine.connect()
    # Read data from PostgreSQL database table and load into a DataFrame instance
    dataFrame = pd.read_sql("select * from \"html_table\" WHERE timestamp = '{}'".format(date), dbConnection)
    dbConnection.close()
    if len(dataFrame)==0:
        return "<div>No Data Available</div>"
    pd.set_option('display.expand_frame_repr', False)
    html_str = dataFrame.iloc[0,0]

    print("Log---requested_html")
    return html_str

def checkDataExist(typeReq,date):
    table=""
    if typeReq=="analysis":
        table= "html_table"
    else:
        table= "json_table"
    
    dbConnection = engine.connect()
    query='''SELECT timestamp FROM {}
                        WHERE timestamp = '{}'
                        '''.format(table,date)
    df = pd.read_sql(query, dbConnection)
    if len(df)>0:
        print("Log---{} data for {} already exists".format(typeReq.date))
        return True
    print("Log---{} data for {} do not exist".format(typeReq,date))
    return False

def clearJsonTable():
    dbConnection = engine.connect()
    common_date_query='''SELECT DISTINCT html_table.timestamp
                        FROM html_table
                        INNER JOIN json_table
                        ON html_table.timestamp = json_table.timestamp
                        '''
    common_date_df = pd.read_sql(common_date_query, dbConnection)
    common_date = common_date_df['timestamp'].to_list()
    for i in common_date:
        d= '''DELETE FROM json_table
                WHERE timestamp = '{}'
                        '''.format(i)        
        dbConnection.execute(d)
    dbConnection.close()
    print('Log---cleared json data for {}'.format(common_date))
    return 'Log---cleared json data for {}'.format(common_date)

@app.route("/backend/scraper/<user>")    
def initialise_scraper(user="Scheduler"):
    json_req=checkDate("scraper", user)
    if json_req == False:
        return "Log---:Request rejected as scraped data already exists"
    else:
        scraperThread = threading.Thread(target=scraper, args=(json_req,))
        try:
            scraperThread.start()
        except:
            print("Log---Scraper failed to initiate due to server error")
            return "Log---Scraper failed to initiate due to server error"
        return "Log---Scraper initiated"

@app.route("/backend/analysis/<user>") 
def initialise_analysis(user="Scheduler"):
    json_req=checkDate("analysis", user)
    if json_req == False:
        return "Log---:Request rejected as analysed data already exists"
    scraped_data_exist = checkDate("scraper", user)
    if scraped_data_exist !=False:
        return "Log---:Request rejected as scraped data do not exists"
    else:
        analysisThread = threading.Thread(target=analysis,args=(json_req,))
        try:
            analysisThread.start()
        except:
            print("Log---Analysis failed to initiate due to server error")
            return "Log---Analysis failed to initiate due to server error"
    return "Log---Analysis initiated"

def scraper(json_req):
    start_date=json_req[0]
    end_date=json_req[1]
    print("Log---Clearing json table to free up storage space")
    clearJsonTable()
    print("Log---Initiated scraper {} to {}".format(start_date,end_date))
    #initialisation
    keywords =  ['nutrition','health', 'wellness','longevity']
    #Twitter Scrape
    twitter_dict = []
    print("Log---start of twitter scrapper!")
    for each_keyword in keywords:      
        print("Log---twitter scraping for {}".format(each_keyword))
        start = datetime.datetime.now()      
        for i,tweet in enumerate(sntwitter.TwitterSearchScraper(each_keyword,'since:%s until:%s lang:en'%(start_date, end_date)).get_items()):
            if i>8000:
                break
            dtime = tweet.date
            new_datetime = datetime.datetime.strftime(datetime.datetime.strptime(str(dtime), '%Y-%m-%d %H:%M:%S+00:00'), '%Y-%m-%d %H:%M:%S')
            twitter_dict.append([tweet.content, new_datetime])
        
        print("Log---time taken:", datetime.datetime.now()-start)

    print("Log---length of twitter_dict before slicing:", len(twitter_dict))
    twitter_dict.sort(key=lambda row: (row[1]), reverse=True)

    #Reddit Scrape
    reddit_read_only = praw.Reddit( client_id = 'X51vAo_gxeYLE_4l3IGKIg',
                                    client_secret = '8fVY5UM-zLjRAam06evgexOzY0QwIg',
                                    user_agent = 'FYP WebScraping', check_for_async=False)

    redditposts_dict = []
    print("Log---start of reddit scrapper!")

    for i in keywords: 
        
        print("Log---reddit scraping for {}".format(i))
        start = datetime.datetime.now()

        redditposts = reddit_read_only.subreddit(i)
        posts = redditposts.top(time_filter="month")

        for post in posts: 
            redditposts_dict.append([])
            redditposts_dict[-1].append(post.title + " -- " + post.selftext)
            
            post_parsed_date = datetime.datetime.utcfromtimestamp(post.created_utc)
            redditposts_dict[-1].append(post_parsed_date)

            if not post.stickied:
                post.comments.replace_more(limit=0)
                for comment in post.comments.list():
                    if comment.author == "AutoModerator": 
                        pass
                    else: 
                        redditposts_dict.append([])
                        redditposts_dict[-1].append(post.title + "--" + comment.body)
                        
                        comment_parsed_date = datetime.datetime.utcfromtimestamp(comment.created_utc)
                        redditposts_dict[-1].append(comment_parsed_date)
        
        print("Log---time taken:", datetime.datetime.now()-start)

    print("Log---length of reddit_dict:", len(redditposts_dict))

    #Combine
    print("Log---Start preprocessing phase 1 of scraped data")
    combined_dict = twitter_dict[:10000] + redditposts_dict
    final_df = pd.DataFrame(combined_dict, columns=["Content", "Datetime"])
    
    final_df.drop_duplicates(subset=['Content'],inplace=True)
    #Data Pre-processing
    final_df['original_text'] = final_df.loc[:, 'Content']

    #Functions for pre-processing
    def remove_urls (text):
        text = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', text, flags=re.MULTILINE)
        return(text)

    def clean_text_sentiment(text):

        text = re.sub(r"(&amp;)",' ',text)
        text = re.sub(r"@[\w]+",' ',text)
        text = re.sub(r"\n",' ',text)
        text = re.sub(r"#",' ',text)
        text = re.sub(r"[^a-zA-Z0-9]+",' ',text)

        return text

    def small_words_removal(paragraph):
        result = []
        tokens = paragraph.split(" ")
        for word in tokens:
            if len(word) >= 3:
                result.append(word)

        return " ".join(result)

    def bigwords_advanced_cleaning(paragraph):

        result = []
        
        tokens = paragraph.split(" ")
        for outer_idx, word in enumerate(tokens):
            if len(word) > 12:
                # ['r', 'take'...]
                split_words = wordninja.split(word.lower())

                # The result for a nonsencial string is '' for e.g. 'aaaaaa'
                if split_words == '':
                    continue

                # cases like Gastroenterology (Corner cases)
                if type(split_words) != list:
                    result.append(split_words)
                    continue 

                for idx, split_word in enumerate(split_words):
                    
                    # remove super small split_word
                    if (len(split_word) < 3 or split_word == ''):
                        split_words.pop(idx)  

                for split_word in split_words:
                    result.append(split_word)

            else:
                result.append(word)

        return " ".join(result)

    #Remove URLs
    final_df['Content'] = final_df['Content'].apply(lambda x:remove_urls(x))
    #remove /n, &amp, @usernames, non english characters
    final_df['Content'] = final_df['Content'].apply(lambda x:clean_text_sentiment(x))
    #remove small words
    final_df['Content'] = final_df['Content'].apply(small_words_removal)
    #remove big words
    final_df['Content'] = final_df['Content'].apply(bigwords_advanced_cleaning)
    #Final JSON Output
    data = final_df.to_json(orient="index")
    #Ingestion to Database
    currDate = start_date

    try:
        # Connect to an existing database
        connection = psycopg2.connect(user="pvahbfuxwqhvpq",
                                    password="3837ad2efc075df162ec73cc54d80e55b1aff7a1098b0eb5916502107f4b97bb",
                                    host="ec2-34-194-40-194.compute-1.amazonaws.com",
                                    port="5432",
                                    database="dcrh9u79n7rtsa")

        # Create a cursor to perform database operations
        cursor = connection.cursor()
        
        # Executing a SQL query to insert datetime into table
        
    #     read_file = open('pre_processed_data.json')
    #     data = json.load(read_file)
        
        cursor.execute("INSERT INTO json_table (json_string, timestamp) VALUES (%s, %s)", (data, currDate))
        connection.commit()
        print("1 item inserted successfully")

    #     # Executing a SQL query
    #     cursor.execute("SELECT json_string FROM json_table;")
    #     # Fetch result
    #     record = cursor.fetchone()
    #     print("You are connected to - ", record, "\n")

    except (Exception, Error) as error:
        print("Error while connecting to PostgreSQL", error)
    finally:
        if (connection):
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")
    return "Log---Scraper completed data collection"
def checkDate(typeReq,user):
    today = datetime.date.today()
    print("Log---{} :Request to initiate on {} by {}".format(typeReq,today,user))
    last_month_last = today.replace(day=1) - datetime.timedelta(days=1)
    last_month_first = last_month_last.replace(day=1)
    if today.strftime("%d") != "01" or user !="Scheduler":
        if checkDataExist(typeReq,last_month_first):           
            print("Log---{} :Request rejected as data already exists".format(typeReq))
            return False
        else:
            print("Log---{} :Request accepted as data do not exists".format(typeReq))
    start_date= last_month_first.strftime("%Y-%m-%d")
    if typeReq =="analysis":        
        request=start_date
        print("Log---{} :Request Input {}".format(typeReq,request))
    else:        
        end_date= last_month_last.strftime("%Y-%m-%d")
        request= [start_date, end_date]
        print("Log---{}: Request Input {} to {}".format(typeReq,request[0],request[1]))

    return request

def analysis(jsonReq):    # Read data from PostgreSQL database table and load into a DataFrame instance
    print("Log---Initiated analysis")
    jsonData= request_json(jsonReq)
    if jsonData ==False:
        return "Log---Analysis failed no data collected for {}".format(jsonReq)
    json_df = pd.read_json(jsonData, orient ='index')
    json_df=sentiment_analysis(json_df)  
    print("Log---Preprocessing data for topic modelling") 
    json_df = contractions(json_df)
    data_words= get_data_words(json_df)
    data_words_bigrams = make_bigrams(remove_stopwords(data_words),data_words)
    print("Log---Data cleaning phase 2 completed")
    # Create Dictionary
    id2word = corpora.Dictionary(data_words_bigrams)
    # Create Corpus
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in data_words_bigrams]    
    limit=20; start=6; step=2
    # Can take a long time to run.
    coherence_values=[]
    coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_words_bigrams, limit=limit)
    optimal_model=getOptimalModel(coherence_values,corpus,id2word)
    # Select the model and print the topics
    print("Log---Selected optimal topics")
    
    #Merging Sentiment Analysis with Topic Modelling
    sentiment = sentimentInfo(optimal_model,corpus,json_df)
    vis_html= prepareHTML(optimal_model,corpus,id2word,sentiment)
    print("Log---prepared html-------")
    
    #Send HTML to database
    try:
        # Connect to an existing database
        connection = psycopg2.connect(user="pvahbfuxwqhvpq",
                                    password="3837ad2efc075df162ec73cc54d80e55b1aff7a1098b0eb5916502107f4b97bb",
                                    host="ec2-34-194-40-194.compute-1.amazonaws.com",
                                    port="5432",
                                    database="dcrh9u79n7rtsa")

        # Create a cursor to perform database operations
        cursor = connection.cursor()
        cursor.execute("INSERT INTO html_table (html_string, timestamp) VALUES (%s, %s)", (vis_html, jsonReq))
        connection.commit()
        print("Log---1 item inserted successfully")

    except (Exception, Error) as error:
        print("Log---Error while connecting to PostgreSQL", error)
    finally:
        if (connection):
            cursor.close()
            connection.close()
            print("Log---PostgreSQL connection is closed")
    return "Log---Analysis completed"
def expand_contractions(text):
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
    contractions_re=re.compile('(%s)' % '|'.join(contractions_dict.keys()))

    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, text)

def request_json(json_req):
    dbConnection = engine.connect()
    dataFrame = pd.read_sql("select * from \"json_table\" where timestamp = '{}'".format(json_req), dbConnection)
    dbConnection.close()
    if len(dataFrame)==0:
        print("Log---No scraped data from json_table for {}".format(json_req))
        return False
    print("Log---Retrieved scraped data from json_table for {}".format(json_req))

    return dataFrame.iloc[0,0]
def sentiment_analysis(json_df):
    print("Log---Conducting Sentiment Analysis")
    
    sid = SentimentIntensityAnalyzer()
    #score
    json_df['score'] = json_df['Content'].apply(lambda text:sid.polarity_scores(str(text)))
    #compound score
    json_df['compound']  = json_df['score'].apply(lambda score_dict: score_dict['compound'])

    #compound label
    json_df['comp_score'] = json_df['compound'].apply(lambda c: 'pos' if 0.3<c<=1 else("neu" if -0.3<=c<=0.3 else "neg"))
    json_df.dropna(inplace=True)
    return json_df
def clean_text(text):
        #convert to lowercasing, remove non words and remove digits
        text = text.lower()
        text = re.sub(r'\W', ' ', text)
        text = re.sub(r'\d', ' ', text)
        return text
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
        # Initiate spacy for lemmatization
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])  
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

def compute_coherence_values(dictionary, corpus, texts, limit, start=6, step=2):
    print("Log---Computing coherence scores")
    coherence_values = []
    count=0
    #model_list = []
    for num_topics in range(start, limit, step):
        # model = gensim.models.wrappers.LdaMallet(mallet_path, num_topics=num_topics, id2word=id2word)
        model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                        id2word=dictionary,
                                        num_topics=num_topics, 
                                        random_state=100)
        #model_list.append(model)
        
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v',processes=1)
        coherence_values.append(coherencemodel.get_coherence())
        print("Log---Coherence Score of {} topics is {}".format(count*2+6,coherence_values[count]))
        count+=1

        
    #return model_list, 
    return coherence_values

def calculate_optimal_coherence(coherence_values):
    print("Log---Caculating Optimal number of topics")
    current_cv = 0
    count=0
    for i in range(len(coherence_values)):
        if coherence_values[i]<current_cv:
            return i-1
        else:
            current_cv = coherence_values[i]
            count=i
    return count
def getOptimalModel(coherence_values,corpus,dictionary):
    index = calculate_optimal_coherence(coherence_values)
    optimalTopics= 6+index*2
    print("Log---Optimal Number of Topics as {}".format(optimalTopics))
    model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                    id2word=dictionary,
                                    num_topics=optimalTopics, 
                                    random_state=100)
    return model
def dominantTopicDoc(df_dominant_topic):
    top_dominant_topic_docs = {}
    no_of_topics = list(df_dominant_topic.Dominant_Topic.unique())
    
    for topic_no in no_of_topics:
        if not math.isnan(topic_no):
            top_dominant_topic_docs[topic_no] = {}
            #Overall
            df_topic = df_dominant_topic[df_dominant_topic["Dominant_Topic"]==topic_no]
            df_topic.sort_values(by='Topic_Perc_Contrib', ascending=False, inplace=True)
            top_dominant_topic_docs[topic_no]['mean'] = list(df_topic["orginal_text"][:3])
            
            #Positive
            df_pos = df_dominant_topic[df_dominant_topic["Dominant_Topic"]==topic_no]
            df_pos = df_pos[df_pos["comp_score"]=="pos"]
            df_pos.sort_values(by='Topic_Perc_Contrib', ascending=False, inplace=True)
            top_dominant_topic_docs[topic_no]['pos'] = list(df_pos["orginal_text"][:3])
                                                                    
            #Neutral
            df_neu = df_dominant_topic[df_dominant_topic["Dominant_Topic"]==topic_no]
            df_neu = df_neu[df_neu["comp_score"]=="neu"]
            df_neu.sort_values(by='Topic_Perc_Contrib', ascending=False, inplace=True)
            top_dominant_topic_docs[topic_no]['neu'] = list(df_neu["orginal_text"][:3])
            
            #Negative
            df_neg = df_dominant_topic[df_dominant_topic["Dominant_Topic"]==topic_no]
            df_neg = df_neg[df_neg["comp_score"]=="neg"]
            df_neg.sort_values(by='Topic_Perc_Contrib', ascending=False, inplace=True)
            top_dominant_topic_docs[topic_no]['neg'] = list(df_neg["orginal_text"][:3])
    return top_dominant_topic_docs
def topicSentiments(df_dominant_topic):
    print("Log---Preparing Topic Sentiment")
    top_dominant_topic_docs = dominantTopicDoc(df_dominant_topic)
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
    return topic_sentiment
def sentimentInfo(optimal_model,corpus,json_df):
    #Dominant Topic
    print("Log---Preparing Dominant Topics")
    
    df_topic_sents_keywords = format_topics_sentences(ldamodel=optimal_model, corpus=corpus, texts=json_df)
    # Format
    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Content', 'Datetime', 'orginal_text','score','compound','comp_score']

    #Dominant Topic Docs
    #Topic Sentiments
    #Create Visualisation
    sentiment = json.dumps(topicSentiments(df_dominant_topic)).replace("https://", "")
    return sentiment

def prepareHTML(optimal_model,corpus,id2word,sentiment):
    vis = gensimvis.prepare(topic_model=optimal_model, corpus = corpus, dictionary = id2word, sentiment=sentiment)
    vis_html = pyLDAvis.prepared_data_to_html(vis)
    return vis_html
def format_topics_sentences(ldamodel, corpus, texts):
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
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stopwords.words('english')] for doc in texts]
def get_data_words(json_df):
    #Pre-Processing - Expand contractions, Lowercase and removal of punctuations

    data_words = list(sent_to_words(json_df['Content']))
    return data_words
def contractions(json_df):
    #Expand Contractions
    json_df['Content'] = json_df['Content'].apply(lambda x:expand_contractions(x))
    #remove punctuations, numbers,lowercase
    json_df['Content'] = json_df['Content'].apply(clean_text)
    return json_df

def make_bigrams(texts,data_words):
    bigram_mod = gensim.models.phrases.Phraser(gensim.models.Phrases(data_words, min_count=5, threshold=30))
    return [bigram_mod[doc] for doc in texts]

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
