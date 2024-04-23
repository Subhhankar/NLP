# Import necessary libraries
import pandas as pd 
import nltk
import string
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from nltk import sent_tokenize
from nltk.corpus import cmudict
import re

# Read data from a CSV file
data=pd.read_csv('output.csv')#read the data
# Convert 'Title' and 'Text' columns to string, handling non-string values
data['Title'] = data['Title'].apply(lambda x: str(x) if pd.notnull(x) else '')
data['Text'] = data['Text'].apply(lambda x: str(x) if pd.notnull(x) else '')

# Convert 'Title' and 'Text' columns to lowercase
data['Title'] = data['Title'].str.lower()
data['Text'] = data['Text'].str.lower()
# Tokenize 'Title' and 'Text' columns using NLTK
data['Title'] = data['Title'].apply(lambda x: word_tokenize(x, language='english'))
data['Text']= data['Text'].apply(lambda x: word_tokenize(x, language='english'))

stop_words=open('stopwords.txt').read() #load stop words from file

# Remove stop words from 'Title' and 'Text' columns
data['Title'] = data['Title'].apply(lambda x: [word for word in x if word not in stop_words])
data['Text'] = data['Text'].apply(lambda x: [word for word in x if word not in stop_words])

# Load positive and negative words from files
positive_words = open('positive-words.txt').read()
negative_words = open('negative-words.txt').read()

# Filter positive and negative words removing stop words
filtered_positive_words = [word for word in positive_words if word not in stop_words]
filtered_negative_words = [word for word in negative_words if word not in stop_words]

# Create a dictionary for sentiment analysis
sentiment_dict = {
    'positive': filtered_positive_words,
    'negative': filtered_negative_words
}

# Calculate positive and negative scores
data['Positive_Score'] = data['Text'].apply(lambda x: sum(1 for word in x if word in positive_words))
data['Negative_Score'] = data['Text'].apply(lambda x: sum(1 for word in x if word in negative_words))
# Calculate polarity and subjectivity scores
data['Polarity_score'] = data['Text'].apply(lambda x: TextBlob(' '.join(x)).sentiment.polarity)
data['Subjectivity_score'] = data['Text'].apply(lambda x: TextBlob(' '.join(x)).sentiment.subjectivity)

# Calculate various text statistics
data['Title_Word_Count'] = data['Title'].apply(lambda x: len(nltk.word_tokenize(' '.join(x))))
data['Text_Word_Count'] = data['Text'].apply(lambda x: len(nltk.word_tokenize(' '.join(x))))

data['Title_Sentence_Count'] = data['Title'].apply(lambda x: len(nltk.sent_tokenize(' '.join(x))))
data['Text_Sentence_Count'] = data['Text'].apply(lambda x: len(nltk.sent_tokenize(' '.join(x))))

# Merge word counts and sentence counts
data['Word_Count'] = data['Title_Word_Count'] + data['Text_Word_Count']
data['Sentence_Count'] = data['Title_Sentence_Count'] + data['Text_Sentence_Count']

data['total_word']=sum(data['Word_Count'])
data['total_sentence']=sum(data['Sentence_Count'])

# Calculate Average Sentence Length
data['Avg_Sentence_Length'] = data['Word_Count'] / data['Sentence_Count']

data['Avg_number_word_per_sentence']=data['total_word']/data['total_sentence']


# Count syllables using CMU Pronouncing Dictionary
syllable_dict = cmudict.dict()

def count_syllables(word):
    if word.lower() not in syllable_dict:    # search for lower case version of the word in dictionary 
        return 0
    return [len(list(y for y in x if y[-1].isdigit())) for x in syllable_dict[word.lower()]][0]
                                               # return number of syllable

# Determine if a word is complex based on syllable count
def is_complex(word):
    syllable_count = count_syllables(word)
    return syllable_count > 2

# Count complex words in text
def count_complex_words(text):
    words = nltk.word_tokenize(text)
    num_complex_words = sum(is_complex(word) for word in words)
    return num_complex_words

# Convert 'Title' and 'Text' columns to string
data['Title'] = data['Title'].apply(lambda x: ' '.join(x))
data['Text'] = data['Text'].apply(lambda x: ' '.join(x))

data['complex_word_count_title'] = data['Title'].apply(count_complex_words)
data['complex_word_count_text'] = data['Text'].apply(count_complex_words)
data['complex_word_count'] = data['complex_word_count_title'] + data['complex_word_count_text']

# Calculate percentage of complex words
data['percentage_complex_word'] = (data['complex_word_count'] / data['Word_Count'])
data['fog_index']= 1.4 * ((data["Avg_Sentence_Length"] + data["percentage_complex_word"])) #fog index

# Count syllables per word in 'Title' and 'Text' columns
def count_syllables(word):
    word = word.lower()
    num_vowels = len([char for char in word if char in 'aeiou'])
    if word.endswith('es') or word.endswith('ed'):
        num_vowels -= 1           # do not include 'es' and 'ed' of found
    return num_vowels

def count_syllables_per_word(text):
    words = nltk.word_tokenize(text)
    syllable_counts = [count_syllables(word) for word in words]
    return syllable_counts

data['syllable_count_per_word_title'] = data['Title'].apply(count_syllables_per_word)
data['syllable_count_per_word_text'] = data['Text'].apply(count_syllables_per_word)
data['syllable_count_per_word']=data['syllable_count_per_word_title'] + data['syllable_count_per_word_text']

# Count personal pronouns in 'Title' and 'Text' columns
def count_personal_pronouns(text):
    pattern = r"\b(I|we|my|ours|us)\b"   # pattern to check if those words exists
    pattern = r"(?<!\bUS\b)" + pattern   # pattern should not include US instead of us
    matches = re.findall(pattern, text, flags=re.IGNORECASE)
    count = len(matches)
    return count
data['personal_pronoun_count_title'] = data['Title'].apply(count_personal_pronouns)
data['personal_pronoun_count_text'] = data['Text'].apply(count_personal_pronouns)
data['personal_pronoun_count'] = data['personal_pronoun_count_title'] + data['personal_pronoun_count_text'] 

# Calculate Average Word Length
data['Total_Character_Count_title'] = data['Title'].apply(lambda x: sum(len(word) for word in x.split()))
data['Total_Character_Count_text'] = data['Text'].apply(lambda x: sum(len(word) for word in x.split()))
data['Total_Character_Count'] = data['Total_Character_Count_title'] + data['Total_Character_Count_text']
data['Avg_Word_Length'] = data['Total_Character_Count'] / data['Word_Count']

# Save the results to a CSV file
result_columns = ['URL', 'Title', 'Positive_Score', 'Negative_Score', 'Polarity_score',
                  'Subjectivity_score', 'Avg_Sentence_Length', 'percentage_complex_word',
                  'fog_index', 'Avg_number_word_per_sentence', 'complex_word_count',
                  'Word_Count', 'syllable_count_per_word', 'personal_pronoun_count',
                  'Avg_Word_Length']

data[result_columns].to_csv('output_results.csv', index=False)



