__author__ = 'ramkishore'

import gzip
from collections import defaultdict
from textstat.textstat import textstat
import numpy
from stop_words import get_stop_words
from nltk.corpus import wordnet
import math
import json
from nltk.tokenize import sent_tokenize
from string import punctuation
import pandas
from nltk.stem.porter import *
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

def readGz(f):
  for l in gzip.open(f):
    yield eval(l)

def calculate_reading_ease(review):
    charsByWords = 0.0
    wordsBySentences = 0.0
    if(len(review) > 0):
        noOfSentences = len(sent_tokenize(review))
        noOfWords = len(review.split())
        noOfCharacters = len(review)
        if noOfWords != 0.0:
            charsByWords = float(noOfCharacters) / float(noOfWords)
        if noOfSentences != 0.0:
            wordsBySentences = float(noOfWords) / float(noOfSentences)
        return float((4.71 * (charsByWords))
                 + (0.5 * (wordsBySentences))
                 - 21.43)
    else:
        return 12


def calculate_punctuation(review):
    if len(review) == 0:
        return 1.0
    count = 0
    for punc in punctuation:
        if review.find(punc) != -1:
            count += 1
    return math.sqrt(math.sqrt(count / float(len(review))))


def generate_review_length(review):
    if(len(str(review)) > 0):
        return math.sqrt(math.sqrt(len(str(review))))
    else:
        return 0

def calculate_number_of_lexicons(review):
    if len(review) > 0:
        return math.sqrt(math.sqrt(textstat.lexicon_count(review)))
    else:
        return 0

def calculate_number_of_sentences(review):
    review = str(review)
    if len(review) > 0:
        return math.sqrt(math.sqrt(textstat.sentence_count(review)))
    else:
        return 0

def calculate_number_of_syllables(review):
    review = str(review)
    if len(review) > 0:
        return math.sqrt(math.sqrt(textstat.syllable_count(review)))
    else:
        return 0

def calculate_stop_words_propotion(review):
    stop_words = get_stop_words('english')
    review = str(review)
    count = 0
    words  = review.split()
    for word in words:
        if word in stop_words:
            count += 1
    if len(words) > 0:
        return count / float(len(words))
    else:
        return 1.0


def ratio_of_misspellings(review):
    review = str(review)
    words = review.split()
    count = 0
    for word in words:
        if not wordnet.synsets(word):
            count += 1
    if len(words) > 0:
        return math.sqrt(count/ float(len(words)))
    else:
        return 1.0

def find_words_in_caps(review):
    review = str(review)
    words = review.split()
    count = 0
    for word in words:
        if(word.isupper() == True):
            count += 1
    if len(words) > 0:
        return math.sqrt(math.sqrt(count/ float(len(words))))
    else:
        return 1.0

def calculate_smiley_propotion(review):
    smiley1 = [':)',':(',':D',':d',';)',':o',':O',';(',':|',':*',':P',':p',':$',':&',':@','x(','X(','X-',':S',':s',
               '8|','B|',':x',':?']
    smiley2 = [':=)',':-)',':-)',':=(',':-(',':=D',':-D',':=d','8=)','8-)','B=)','B-)',';-)',';=)',':=o',':=O',':-O',
               ';-(',';=(',':=|',':=*',':=P',':=p',':-p',':-$',':=$',':^)','|-)','I-)','I=)',':-&',':=&',':-@',':=@',
               'x=(',':=S',':=s','8-|','8=|','B=|',':-X',':=x',':=X',':=#',':=?','(y)','(Y)','(n)','(N)']
    count = 0
    review = str(review)
    if len(review) > 0:
        for i in range(len(review)-1):
            char1 = review[i] + review[i+1]
            if char1 in smiley1:
                count += 1
        for i in range(len(review)-2):
            char2 = review[i] + review[i+1] + review[i+2]
            if char2 in smiley2:
                count += 1
        return math.sqrt(math.sqrt(math.sqrt(math.sqrt(count/float(len(review))))))
    else:
        return 0

def calculate_positive_negative_words_propotion(review):
    text_file = open("positive-words.txt", "r")
    positive_words = text_file.read().split('\n')
    text_file = open("negative-words.txt", "r")
    negative_words = text_file.read().split('\n')
    review = str(review)
    words = review.split()
    count = 0
    if len(review) > 0:
        for word in words:
            if ((word in positive_words) or (word in negative_words)):
                count += 1
        return math.sqrt(math.sqrt(count/ float(len(review))))
    else:
        return 0

def calculate_mean_user_ratings_train_data(userID,helpful_votes):
    userRatings = defaultdict(list)
    for i in range(len(userID)):
        userRatings[userID[i]].append(helpful_votes[i])
    return userRatings

def calculate_mean_user_ratings_test_data(userID,userRatings,globalAverage):
    user_helpful_votes = []
    for user in userID:
        if user in userRatings:
            user_helpful_votes.append( sum(userRatings[user])/len(userRatings[user]) )
        else:
            user_helpful_votes.append(globalAverage)
    return user_helpful_votes

def calculate_mean_item_ratings_train_data(itemID,helpful_votes):
    itemRatings = defaultdict(list)
    for i in range(len(itemID)):
        itemRatings[itemID[i]].append(helpful_votes[i])
    return itemRatings

def calculate_mean_item_ratings_test_data(itemID,itemRatings,globalAverage):
    item_helpful_votes = []
    for item in itemID:
        if item in itemRatings:
            item_helpful_votes.append( sum(itemRatings[item])/len(itemRatings[item]) )
        else:
            item_helpful_votes.append(globalAverage)
    return item_helpful_votes

def calculate_tf(review):
    word_frequency = {}
    max_word_frequency = 0
    review = str(review)
    tf_dict = {}
    words  = review.split()
    words = [word.lower() for word in words]
    words = [''.join([c for c in word if c not in ('!', '?','@','#','$','%','^','&','*','(',')','.','-',';',':','\'',',','"')]) for word in words]
    stemmer = PorterStemmer()
    words = [str(stemmer.stem(word)) for word in words]
    for word in words:
        if word in word_frequency:
            word_frequency[word] = word_frequency[word] + 1
            if(max_word_frequency < word_frequency[word]):
                max_word_frequency = word_frequency[word]
        else:
            word_frequency[word] = 1
    for word in words:
        if word not in tf_dict:
            tf_dict[word] = 0.5 + (0.5 * (word_frequency[word] / float(max_word_frequency + 0.0000000001)))
    return tf_dict

def calculate_idf(review_text):
    initial_bag_of_words = []
    review_bag_of_words = []
    idf_dict = {}
    for review in review_text:
        review = str(review)
        words  = review.split()
        words = [word.lower() for word in words]
        words = [''.join([c for c in word if c not in ('!', '?','@','#','$','%','^','&','*','(',')','.','-',';',':','\'',',','"')]) for word in words]
        stemmer = PorterStemmer()
        words = [str(stemmer.stem(word)) for word in words]
        word_frequency = {}
        for word in words:
            initial_bag_of_words.append(word)
            if word not in word_frequency:
                word_frequency[word] = 1
        review_bag_of_words.append(word_frequency)
    final_bag_of_words = list(set(initial_bag_of_words))
    index = 0
    print len(final_bag_of_words)
    for cur_word in final_bag_of_words:
        count = 0
        for i in range(len(review_bag_of_words)):
            if cur_word in review_bag_of_words[i]:
                count += 1
        if cur_word not in idf_dict:
            idf_dict[cur_word] = math.log(len(review_text) / float(count))
        index += 1
    return idf_dict

def calculate_tf_idf(tf_dict,idf_dict):
    tf_idf = []
    for word,val in tf_dict.iteritems():
        tf_idf.append(float(tf_dict[word]) * float(idf_dict[word]))
    if float(len(tf_idf)) > 0:
        return math.sqrt(math.sqrt((float(sum(tf_idf)) / float(len(tf_idf)))))
    else:
        return 0


def normalize(data):
    maxi = max(data)
    mini = min(data)
    for i in range(len(data)):
        data[i] = (data[i] - mini)/ float(maxi - mini)
    return data


allHelpful = []
userHelpful = defaultdict(list)
itemHelpful = defaultdict(list)
rating_list = []
review_list = []
time_list = []
data = []
userID = []
itemID = []
helpful_votes = []
total_votes = []



with open('processed_data.json') as data_file:
    for line in data_file:
        data.append(json.loads(line))

print len(data)

input_data = data
for l in input_data:
    user,item = l['reviewerID'],l['itemID']
    allHelpful.append(l['helpful'])
    userHelpful[user].append(l['helpful'])
    itemHelpful[item].append(l['helpful'])
    rating_list.append(l['rating'])
    review_list.append(l['reviewText'])
    time_list.append(l['unixReviewTime'] ** 5)
    userID.append(l['reviewerID'])
    itemID.append(l['itemID'])
    helpful_votes.append(l['helpful']['nHelpful'])
    total_votes.append(l['helpful']['outOf'])


userRatings = calculate_mean_user_ratings_train_data(userID,helpful_votes)
itemRatings = calculate_mean_item_ratings_train_data(itemID,helpful_votes)

user_sum = 0
user_len = 0
for key, value in userRatings.iteritems():
    user_sum += sum(value)
    user_len += len(value)

user_mean = user_sum / float(user_len)

item_sum = 0
item_len = 0
for key, value in itemRatings.iteritems():
    item_sum += sum(value)
    item_len += len(value)

item_mean = item_sum / float(item_len)


time_list = normalize(time_list)
train_input_data = []
train_output_data = []
all_help = [x['nHelpful'] for x in allHelpful]
all_outof = [x['outOf'] for x in allHelpful]

idf_dict = calculate_idf(review_list)
tf_idf_list = []
for review in review_list:
    tf_dict = calculate_tf(review)
    tf_idf_list.append(calculate_tf_idf(tf_dict,idf_dict))

for i in range(len(input_data)):
    if(all_outof[i] != 0):
        feature_vector = []
        feature_vector.append(calculate_number_of_lexicons(review_list[i]))
        feature_vector.append(calculate_number_of_sentences(review_list[i]))
        feature_vector.append(calculate_number_of_syllables(review_list[i]))
        feature_vector.append(calculate_stop_words_propotion(review_list[i]))
        feature_vector.append(generate_review_length(review_list[i]))
        feature_vector.append(find_words_in_caps(review_list[i]))
        feature_vector.append(calculate_punctuation(review_list[i]))
        feature_vector.append(calculate_reading_ease(review_list[i]))
        feature_vector.append(calculate_smiley_propotion(review_list[i]))
        feature_vector.append(calculate_positive_negative_words_propotion(review_list[i]))
        feature_vector.append(sum(userRatings[userID[i]]) / float(len(userRatings[userID[i]])))
        feature_vector.append(sum(itemRatings[itemID[i]]) / float(len(itemRatings[itemID[i]])))
        feature_vector.append(math.log(math.log(all_outof[i])))
        feature_vector.append(time_list[i])
        feature_vector.append(tf_idf_list[i])
        feature_vector.append(rating_list[i])

        train_input_data.append(feature_vector)
        train_output_data.append(all_help[i]/float(all_outof[i]))

pd = pandas.DataFrame(train_input_data)
pd.to_csv("train_input_data.csv")
pd = pandas.DataFrame(train_output_data)
pd.to_csv("train_output_data.csv")


review_list = []
rating_list = []
time_list = []
outof_list = []
test_userRatings = []
test_itemRatings = []
test_userID = []
test_itemID = []
for l in readGz("helpful.json.gz"):
    review_list.append(l['reviewText'])
    rating_list.append(l['rating'])
    if l['helpful']['outOf'] > 1:
        outof_list.append(math.log(math.log(l['helpful']['outOf'])))
    else:
        outof_list.append(0)
    time_list.append(l['unixReviewTime'] ** 5)
    test_userID.append(l['reviewerID'])
    test_itemID.append(l['itemID'])


test_userRatings = calculate_mean_user_ratings_test_data(test_userID,userRatings,user_mean)
test_itemRatings = calculate_mean_item_ratings_test_data(test_itemID,itemRatings,item_mean)
time_list = normalize(time_list)
test_input_data = []

for i in range(len(rating_list)):
    feature_vector = []
    feature_vector.append(calculate_number_of_lexicons(review_list[i]))
    feature_vector.append(calculate_number_of_sentences(review_list[i]))
    feature_vector.append(calculate_number_of_syllables(review_list[i]))
    feature_vector.append(calculate_stop_words_propotion(review_list[i]))
    feature_vector.append(generate_review_length(review_list[i]))
    feature_vector.append(find_words_in_caps(review_list[i]))
    feature_vector.append(calculate_punctuation(review_list[i]))
    feature_vector.append(calculate_reading_ease(review_list[i]))
    feature_vector.append(calculate_smiley_propotion(review_list[i]))
    feature_vector.append(calculate_positive_negative_words_propotion(review_list[i]))
    feature_vector.append(test_userRatings[i])
    feature_vector.append(test_itemRatings[i])
    feature_vector.append(outof_list[i])
    feature_vector.append(time_list[i])
    feature_vector.append(tf_idf_list[i])
    feature_vector.append(rating_list[i])

    test_input_data.append(feature_vector)

pd = pandas.DataFrame(test_input_data)
pd.to_csv("test_input_data.csv")