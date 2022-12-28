import nltk
import pandas as pd
import numpy as np
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer

lemmatizer = WordNetLemmatizer()
def lemmatize_words(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

def lower_case(text):
	return text.lower()

def remove_punctuation(text):
	return text.translate(str.maketrans('', '', string.punctuation))

def remove_emoji(string):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

def remove_urls(text):
	text = ' '.join([x for x in text.split() if ('http' not in x and x[0]!='@' and x[0]!='#' and 'www' not in x)])
	return text

def remove_stopwords(text):
	stop_words = set(stopwords.words('english'))
	word_tokens = word_tokenize(text)
	filtered_sentence = [w for w in word_tokens if not w in stop_words]
	return filtered_sentence

def preprocess_post(text):
	return lemmatize_words(remove_urls(remove_emoji(remove_punctuation(lower_case(text)))))