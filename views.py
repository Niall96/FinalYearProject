import csv
import os
from operator import itemgetter

import nltk
import tweepy
from twitter import *
import fileinput
import sys
import os
from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.template import loader
import re
import pandas as pd
from nltk import word_tokenize, Counter
import numpy as np
from uploads.core.models import Document
from uploads.core.forms import DocumentForm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# import datashader as ds
import matplotlib.pyplot as plt
import seaborn as sns
from django.shortcuts import render_to_response
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import googlemaps
import gmplot


def remove_pattern(input, pattern):
    r = re.findall(pattern, input)
    for i in r:
        input = re.sub(i, '', input)
    return input

def cal_sent(Clean_Tweet):
    return TextBlob(Clean_Tweet).sentiment
analyse = SentimentIntensityAnalyzer()

def cal_sent_anl(Clean_Tweet):
    return analyse.polarity_scores(Clean_Tweet)

def adminn(request):
    topic = request.POST['topic']
    print(topic)

    consumer_key = "E4AvhUQRBROp0Vwjkjx9l68vE"
    consumer_secret = "Lb23FMXpGWaIvYAjmYzuh55DO3AZTtGV2bBoTaSd6KcVF2WtzN"
    access_key = "1180980930959286272-mProAQ0upfyLRcjMLhKFdLh4kozhaU"
    access_secret = "2uifVW2JQvi3X4INjSgyjBGTYiu4xApMmE8NuXfmSVu0X"

    # API Endpoint
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True)

    ########################################################

    # #Creation of CSV file for tweets to be scraped
    os.unlink('./uploads/static/Test.csv')
    csvFile = open('./uploads/static/Test.csv', 'a')
    csvWriter = csv.writer(csvFile)

    ##################################################


    # Ask the user how many tweets should be extracted
    max_tweets_extract = 1000

    # Ask the user what hashtag you would like to scrape through
    for tweet in tweepy.Cursor(api.search, q=topic + " -rt", lang="en", rpp=100).items(max_tweets_extract):
        csvWriter.writerow([tweet.created_at, tweet.user.screen_name.encode('utf-8'),
                            tweet.text.encode('utf-8')])
        print('Extracted' + str(max_tweets_extract) + 'tweets with the topic ' + topic)
    #
    csvFile.close()

    with open('./uploads/static/Test.csv', newline='') as f:
        r = csv.reader(f)
        data = [line for line in r]
    with open('./uploads/static/Test.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Date', 'Username', 'Tweet'])
        w.writerows(data)


####################################################################################

    ##Sentiment Analysis of Tweets Scraped##
    analyse = SentimentIntensityAnalyzer()
    dataset = pd.read_csv("./uploads/static/Test.csv", encoding='UTF-8')

    dataset['Clean_Tweet'] = np.vectorize(remove_pattern)(dataset['Tweet'], "@[\w]*")
    #
    dataset['Clean_Tweet'] = dataset['Clean_Tweet'].str.replace("[^a-zA-Z#]", " ")
    #
    dataset['Clean_Tweet'] = dataset['Clean_Tweet'].str.replace("https", " ")
    #
    dataset['Clean_Tweet'] = dataset['Clean_Tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 3]))

    from nltk.tokenize import word_tokenize
    nltk.download('punkt')
    #
    tokenize = dataset['Clean_Tweet'].apply(lambda x: x.split())
    tokenize = tokenize.apply(lambda x: [word_tokenize(i) for i in x])
    tokenize.head()
    dataset.loc[:, ('Tweet', 'Clean_Tweet')]

    dataset[dataset.Tweet.isnull()]

    dataset[dataset.Clean_Tweet.isnull()]

    dataset.drop_duplicates(subset=['Clean_Tweet'], keep='first', inplace=True)

    dataset.reset_index(drop=True, inplace=True)


    dataset.info()

    dataset.reset_index(drop=True, inplace=True)
    dataset.info()


    dataset['Sentiment'] = dataset.Clean_Tweet.apply(cal_sent)
    dataset['Sentiment_Analyser'] = dataset.Clean_Tweet.apply(cal_sent_anl)

    s = pd.DataFrame(index=range(0, len(dataset)), columns=['compound_score', 'Sentiment_Type'])

    for i in range(0, len(dataset)):
        s['compound_score'][i] = dataset['Sentiment_Analyser'][i]['compound']

        if (dataset['Sentiment_Analyser'][i]['compound'] <= -0.025):
            s['Sentiment_Type'][i] = 'Negative'
        if (dataset['Sentiment_Analyser'][i]['compound'] >= 0.025):
            s['Sentiment_Type'][i] = 'Positive'
        if ((dataset['Sentiment_Analyser'][i]['compound'] >= -0.025) & (
                dataset['Sentiment_Analyser'][i]['compound'] <= 0.025)):
            s['Sentiment_Type'][i] = 'Neutral'
    dataset['compound_score'] = s['compound_score']
    dataset['Sentiment_Type'] = s['Sentiment_Type']
    keep_col = ['Clean_Tweet', 'Sentiment_Analyser', 'Sentiment_Type']
    ndataset = dataset[keep_col]

    ############################################################################################################

    # ##Sentiment Analysis Accuracy##
    # ##This code below will perform an accuracy check to ensure the sentiment analysis is performing to a high level
    # cleaned = dataset['Clean_Tweet']
    # labels = dataset['Sentiment_Type']
    #
    # from nltk.corpus import stopwords
    # from sklearn.feature_extraction.text import TfidfVectorizer
    #
    # nltk.download('stopwords')
    # vectorizer = TfidfVectorizer(max_features=2500, min_df=3, max_df=0.8, stop_words=stopwords.words('english'))
    # cleaned = vectorizer.fit_transform(cleaned).toarray()
    #
    # from sklearn.model_selection import train_test_split
    #
    # X_train, X_test, y_train, y_test = train_test_split(cleaned, labels, test_size=0.2, random_state=0)
    #
    # from sklearn.ensemble import RandomForestClassifier
    #
    # text_classifier = RandomForestClassifier(n_estimators=200, random_state=0)
    # text_classifier.fit(X_train, y_train)
    #
    # predictions = text_classifier.predict(X_test)
    #
    # from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    #
    # print(confusion_matrix(y_test, predictions))
    # print(classification_report(y_test, predictions))
    # print(accuracy_score(y_test, predictions))
    # accuracy = accuracy_score(y_test, predictions)

   #########################################################################################################


    ##Creation of CSV file to create a result table
    ndataset.to_csv('./media/Test_Sentiment.csv')
    inputFileName = './media/Test_Sentiment.csv'
    with open('./media/Test_Sentiment1.csv', "w") as outfile:
        for line in fileinput.input(
                [inputFileName],
                inplace=False):
            if fileinput.isfirstline():
                outfile.write('Tweet"Sentimental Analysis"Sentiment\n')
            else:
                outfile.write(line)

    mean_pos = 0
    mean_neg = 0
    mean_neutral = 0
    for i in range(0, len(dataset)):
        # s['compound_score'][i] = dataset['Sentiment_Analyser'][i]['compound']
        mean_neg += (dataset['Sentiment_Analyser'][i]['neg'])
        mean_pos += (dataset['Sentiment_Analyser'][i]['pos'])
        mean_neutral += (dataset['Sentiment_Analyser'][i]['neu'])


    ##############################################################
    ##Pie Chart showing the split in sentiment

    negative = (mean_neg / len(dataset)) * 100
    pos = (mean_pos / len(dataset)) * 100
    neutral = (mean_neutral / len(dataset)) * 100
    print(negative, pos, neutral)
    scores = [pos, negative, neutral]
    sentiment_labels = ['Positive', 'Negative', 'Neutral']
    fig = plt.figure()
    plt.pie(scores, labels=sentiment_labels, startangle=90, autopct='%0.2f%%', radius=1)
    plt.title("Pie Chart showing Sentiment Distribution")
    fig.savefig("./media/a.png", transparent = True, dpi=100)

    #########################################################################

    #Bar Chart showing the split in Sentiment
    chart = plt.figure()
    count = [mean_pos,mean_neg, mean_neutral]
    plt.bar(sentiment_labels, count, align ='center')
    plt.title("Bar Chart showing Sentiment")
    chart.savefig("./media/c.png", transparent=True,dpi=100)


    #########################################################################

    #WordCloud showing the most common words appearing with topic
    stopwords = set(STOPWORDS)

    wordcloud = WordCloud(background_color="white", stopwords=stopwords, random_state=2016).generate(
        " ".join([i for i in dataset['Clean_Tweet']]))
    plt.figure(figsize=(5, 5), facecolor='k')
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.title("WordCloud of Common Words")
    plt.savefig("./media/w.png", transparent=True, dpi=100)

    #########################################################################

    #Line Graph showing the distribution amongst compound scores
    line = plt.figure()
    ax = line.add_subplot(111)
    sns.distplot(dataset['compound_score'], bins=15, ax=ax)
    plt.title("Distribution of Compound Score")
    plt.savefig("./media/l.png", transparent = True, dpi=100)

    ##########################################################################

    return render_to_response('result.html')



def adminn_view(request):
    return render(request ,'search.html')

