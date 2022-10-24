#imports
import snscrape.modules.twitter as sntwitter
import pandas as pd
from datetime import datetime
import praw
import numpy as np
import re
import splitter
import json
import psycopg2
from psycopg2 import Error
from datetime import datetime as dt

#initialisation
keywords =  ['nutrition','health', 'wellness','longevity']
start_date = '2022-08-01'
end_date = '2022-09-01'

#Twitter Scrape
twitter_dict = []
print("start of twitter scrapper!")

for each_keyword in keywords:
    
    print("keyword start:", each_keyword)
    start = datetime.now()
    
    for i,tweet in enumerate(sntwitter.TwitterSearchScraper(each_keyword,'since:%s until:%s lang:en'%(start_date, end_date)).get_items()):
        if i>8000:
            break

        dtime = tweet.date
        new_datetime = datetime.strftime(datetime.strptime(str(dtime), '%Y-%m-%d %H:%M:%S+00:00'), '%Y-%m-%d %H:%M:%S')
        twitter_dict.append([tweet.content, new_datetime])
    
    print("time taken:", datetime.now()-start)

print("length of twitter_dict before slicing:", len(twitter_dict))
twitter_dict.sort(key=lambda row: (row[1]), reverse=True)

#Reddit Scrape
reddit_read_only = praw.Reddit( client_id = 'X51vAo_gxeYLE_4l3IGKIg',
                                client_secret = '8fVY5UM-zLjRAam06evgexOzY0QwIg',
                                user_agent = 'FYP WebScraping', check_for_async=False)

redditposts_dict = []
print("start of reddit scrapper!")

for i in keywords: 
    
    print("keyword start:", i)
    start = datetime.now()

    redditposts = reddit_read_only.subreddit(i)
    posts = redditposts.top(time_filter="month")

    for post in posts: 
        redditposts_dict.append([])
        redditposts_dict[-1].append(post.title + " -- " + post.selftext)
        
        post_parsed_date = datetime.utcfromtimestamp(post.created_utc)
        redditposts_dict[-1].append(post_parsed_date)

        if not post.stickied:
            post.comments.replace_more(limit=0)
            for comment in post.comments.list():
                if comment.author == "AutoModerator": 
                    pass
                else: 
                    redditposts_dict.append([])
                    redditposts_dict[-1].append(post.title + "--" + comment.body)
                    
                    comment_parsed_date = datetime.utcfromtimestamp(comment.created_utc)
                    redditposts_dict[-1].append(comment_parsed_date)
    
    print("time taken:", datetime.now()-start)

print("length of reddit_dict:", len(redditposts_dict))

#Combine
combined_dict = twitter_dict[:10000] + redditposts_dict
final_df = pd.DataFrame(combined_dict, columns=["Content", "Datetime"])
print("done!")

#Data Pre-processing
df = final_df.copy()
df['original_text'] = df.loc[:, 'Content']

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
            split_words = splitter.split(word.lower())

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
df['Content'] = df['Content'].apply(lambda x:remove_urls(x))
print("removed URL")

#remove /n, &amp, @usernames, non english characters
df['Content'] = df['Content'].apply(lambda x:clean_text_sentiment(x))
print("removed HTML")

#remove small words
df['Content'] = df['Content'].apply(small_words_removal)
print("removed small words")

#remove big words
df['Content'] = df['Content'].apply(bigwords_advanced_cleaning)
print("removed big words")

#Final JSON Output
data = df.to_json(orient="index")

#Ingestion to Database
currDate = dt.now()

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