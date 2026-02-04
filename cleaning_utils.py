from collections import Counter
import re
import pandas as pd

def clean_tweet(tweet: str):
    '''
    Cleans a single tweet of contractions, newlines, numbers, etc.
    Params:
        tweet (str): a single string representing a person's tweet
    Returns:
        tweet (str): a tweet cleaned of contractions, newlines, numbers, etc.
    '''
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    tweet = tweet.lower()
    tweet = emoji_pattern.sub(r'', tweet)   
    tweet=re.sub("isn't",'is not',tweet)
    tweet=re.sub("he's",'he is',tweet)
    tweet=re.sub("she's",'she is',tweet)
    tweet = re.sub("isn't", "is not", tweet)
    tweet = re.sub("aren't", "are not", tweet)
    tweet = re.sub("wasn't", "was not", tweet)
    tweet = re.sub("weren't", "were not", tweet)
    tweet = re.sub("haven't", "have not", tweet)
    tweet = re.sub("hasn't", "has not", tweet)
    tweet = re.sub("hadn't", "had not", tweet)
    tweet = re.sub("won't", "will not", tweet)
    tweet = re.sub("wouldn't", "would not", tweet)
    tweet = re.sub("don't", "do not", tweet)
    tweet = re.sub("doesn't", "does not", tweet)
    tweet = re.sub("didn't", "did not", tweet)
    tweet = re.sub("can't", "cannot", tweet)
    tweet = re.sub("couldn't", "could not", tweet)
    tweet = re.sub("shouldn't", "should not", tweet)
    tweet = re.sub("mightn't", "might not", tweet)
    tweet = re.sub("mustn't", "must not", tweet)
    tweet = re.sub("he's", "he is", tweet)
    tweet = re.sub("she's", "she is", tweet)
    tweet = re.sub("it's", "it is", tweet)
    tweet = re.sub("that's", "that is", tweet)
    tweet = re.sub("there's", "there is", tweet)
    tweet = re.sub("here's", "here is", tweet)
    tweet = re.sub("who's", "who is", tweet)
    tweet = re.sub("what's", "what is", tweet)
    tweet = re.sub("where's", "where is", tweet)
    tweet = re.sub("when's", "when is", tweet)
    tweet = re.sub("why's", "why is", tweet)
    tweet = re.sub("how's", "how is", tweet)
    tweet = re.sub("I'm", "I am", tweet)
    tweet = re.sub("you're", "you are", tweet)
    tweet = re.sub("they're", "they are", tweet)
    tweet = re.sub("we're", "we are", tweet)
    tweet = re.sub("I've", "I have", tweet)
    tweet = re.sub("you've", "you have", tweet)
    tweet = re.sub("they've", "they have", tweet)
    tweet = re.sub("we've", "we have", tweet)
    tweet = re.sub("I'll", "I will", tweet)
    tweet = re.sub("you'll", "you will", tweet)
    tweet = re.sub("he'll", "he will", tweet)
    tweet = re.sub("she'll", "she will", tweet)
    tweet = re.sub("it'll", "it will", tweet)
    tweet = re.sub("they'll", "they will", tweet)
    tweet = re.sub("we'll", "we will", tweet)
    tweet = re.sub("I'd", "I would", tweet)
    tweet = re.sub("you'd", "you would", tweet)
    tweet = re.sub("he'd", "he would", tweet)
    tweet = re.sub("she'd", "she would", tweet)
    tweet = re.sub("it'd", "it would", tweet)
    tweet = re.sub("they'd", "they would", tweet)
    tweet = re.sub("we'd", "we would", tweet)

    #TODO - using the example above, remove characters such as newlines, punctuation, numbers, and other non-alphabetical characters
    tweet = re.sub(r'\n', '', tweet)  # remove newlines
    tweet = re.sub(r'[!,.?";:]', '', tweet)  # remove punctuation
    tweet = re.sub(r'[0-9]+', '', tweet)  # remove numbers
    tweet = re.sub(r'[^a-z\s]', '', tweet)  # remove non-alphabetical characters
    return tweet

def clean_data(data: pd.DataFrame):
    '''
    Applies the clean_tweet function across all tweets in the dataset.
    Params:
        data (pd.DataFrame): the full dataset, including sentiment, id, date, user, tweet, etc.
    Returns:
        cleaned_data (pd.DataFrame): a cleaned, truncated dataset 
    '''

    # clean the 'tweet' column and add it back to the DataFrame
    data['tweet'] = data['tweet'].apply(clean_tweet)

    # we only need the tweet and sentiment
    return data[['tweet', 'sentiment']]

def remove_stopwords(tweet_data: pd.Series, stop_words: list[str]):
    '''
    Removes all the stop words (provided by the NLTK corpus) from the provided series of tweets.
    NOTE: This will take a long time to run, likely at least 15 minutes.
    Params:
        tweet_data (pd.Series): a series with tweets that contain stopwords
        stop_words (list[str]): a list of strings of common stopwords
    Returns:
        cleaned_data (pd.Series): a series with tweets that don't contain stopwords
    '''
    stop_words = set(stop_words)

    return tweet_data.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

def find_freqwords(tweet_data: pd.Series):
    '''
    Finds the 10 most common words amongst all tweets in the dataset.
    Params:
        tweet_data (pd.Series): a series containing all tweets
    Returns:
        freqwords (set): a set of the 10 most common words across all tweets
    '''
    # find value counts of all words in all data points
    counter = Counter()
    for text in tweet_data.values:
        for word in text.split():
            counter[word] += 1

    # creates a set of the 10 most common words amongst all tweets
    return set([w for (w, wc) in counter.most_common(10)])

def remove_freqwords_from_tweet(text: str, freqwords: set):
    '''
    Given a tweet that has been cleaned, returns the tweet without the most frequently appearing words.
    Params: 
        text (str): the actual text of the tweet
        freqwords (set): a set of frequently appearing words across all tweets
    Returns:
        new_text (str): the new text of the tweet, without any frequently appearing words.
    '''
    #TODO - complete this function which removes any of the most frequent words in a given tweet that do not add value
    return " ".join([word for word in str(text).split() if word not in freqwords])
    
    

def remove_all_freqwords_from_all_tweets(tweet_data: pd.Series, freqwords: set):
    '''
    Given a series of tweets, removes all freqwords from all those tweets.
    Params:
        tweet_data (pd.Series): a series containing all tweets
        freqwords (set): a set of frequently appearing words across all tweets
    Returns:
        new_tweets (pd.Series): a series containing all tweets, without the most frequently appearing words
    '''
    #TODO - apply the function you just wrote to remove freqwords across all tweets in your dataframe
    
    new_tweets = tweet_data.apply(lambda x: remove_freqwords_from_tweet(x, freqwords))
    return new_tweets 

def tokenize_data(data: pd.DataFrame):
    '''
    Tokenizes the data by converting the tweet into a series of strings (representing words) rather than a single string
    Params:
        data (pd.DataFrame): tweet and sentiment data
    Returns
        tokenized_data (pd.DataFrame): tokenized tweet and sentiment data
    '''
    tokenized_data = data.drop(columns=['sentiment'])
    #TODO - complete data preprocessing by changing the tweet format from a string of words to a list of strings containing individual words
    # HINT: The dataframe .apply function can be used to apply any function (I.E. str.split()) to all entries in the dataframe
    tokenized_data['tweet'] = tokenized_data['tweet'].apply(lambda x: x.split())
    return tokenized_data