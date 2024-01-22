#######################
# Introduction to Text Mining and Natural Language Processing
#######################
# End-to-end Text Classification Model
#######################
# Sentiment Analysis and Sentiment Modeling for Amazon Reviews
#######################

#1. Text Preprocessing
#2. Text Visualization
#3. Sentiment Analysis
#4. Feature Engineering
#5. Sentiment Modeling

# !pip install nltk
# !pip install textblob
# !pip install wordcloud


from warnings import filterwarnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate
from sklearn.preprocessing import LabelEncoder
from textblob import Word, TextBlob
from wordcloud import WordCloud

filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

#######################
#1. Text Preprocessing
#######################

df = pd.read_csv("nlp/datasets/amazon_reviews.csv", sep=",")
df.head()


####################
# Normalizing Case Folding
####################

df['reviewText'] = df['reviewText'].str.lower()

####################
#Punctuations
####################
# Let's remove the punctuation marks.
df['reviewText'] = df['reviewText'].str.replace('[^\w\s]', '')

# regular expression--identifying and capturing a specific pattern in textual expressions

####################
#Numbers
####################
# Capture numbers and replace them with spaces.
df['reviewText'] = df['reviewText'].str.replace('\d', '')

####################
#Stopwords
####################
# import nltk
# nltk.download('stopwords')

# We created our list
sw = stopwords.words('english')

# Words in the dictionary will be selected and extracted from the text.
df['reviewText'] = df['reviewText'].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))


####################
#rarewords
####################

# To exclude rarely occurring words;
# Capturing the number of occurrences of each word. Necessary for filtering.
# Let's count all words
temp_df = pd.Series(' '.join(df['reviewText']).split()).value_counts()

# Let's select words that appear once
drops = temp_df[temp_df <= 1]

# Select the ones not in the drops and then merge them.
df['reviewText'] = df['reviewText'].apply(lambda x: " ".join(x for x in x.split() if x not in drops))


####################
#Tokenization
####################

# nltk.download("punkt")

# We will break the sentences into parts programmatically.
df["reviewText"].apply(lambda x: TextBlob(x).words).head()


####################
# Lemmatization
####################
# The process of separating words into their roots
import nltk
nltk.download('omw-1.4')

nltk.download('wordnet')

df['reviewText'] = df['reviewText'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))


#######################
#2. Text Visualization
#######################


####################
# Calculation of Term Frequencies
####################
# We will count the frequencies of the words.
tf = df["reviewText"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()

tf.columns = ["words", "tf"]

# We sort by tf variable in descending order
tf.sort_values("tf", ascending=False)

####################
#Barplot
####################
# Let's take the acceptable frequencies in the bar chart.

tf[tf["tf"] > 500].plot.bar(x="words", y="tf")
plt.show()

####################
#wordcloud
####################
# Creating a cloud-shaped visual. We will express the entire text as a single text.
text = " ".join(i for i in df.reviewText)

# Creating the chart
wordcloud = WordCloud().generate(text)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# Lighter color and larger font size to adjust the word size to be projected
wordcloud = WordCloud(max_font_size=50,
                       max_words=100,
                       background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

wordcloud.to_file("wordcloud.png")

####################
# Wordcloud by Templates
####################

# We want to integrate the word cloud into a picture
tr_mask = np.array(Image.open("nlp/tr.png"))

wc = WordCloud(background_color="white",
                max_words=1000,
                mask=tr_mask,
                contour_width=3,
                contour_color="firebrick")

wc.generate(text)
plt.figure(figsize=[10, 10])
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.show()

wc.to_file("wordcloud2.png")

#######################
#3. Sentiment Analysis
#######################
# Expressing emotional state mathematically (negative/positive)
df["reviewText"].head()

nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()
# Let's look at the scores of the expression below. The important ones are the compound emotion scores (between -1 and 1).
sia.polarity_scores("The movie was awesome")

sia.polarity_scores("I liked this music but it is not good as the other one")

# Sentiment score of each review
df["reviewText"][0:10].apply(lambda x: sia.polarity_scores(x))

# It came in the form of a dictionary structure. Let's look at the scores we are interested in.
df["reviewText"][0:10].apply(lambda x: sia.polarity_scores(x)["compound"])

# Let's send the transaction to df permanently
df["polarity_score"] = df["reviewText"].apply(lambda x: sia.polarity_scores(x)["compound"])


####################
#4. Feature Engineering
####################
# To predict whether a comment is negative or positive. To make a classification.
df["reviewText"][0:10].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")

# Let's add the variable to the data set
df["sentiment_label"] = df["reviewText"].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")

# Let's look at the class distributions
df["sentiment_label"].value_counts()

# Let's examine the average score of those with positive comments
df.groupby("sentiment_label")["overall"].mean()

# Let's put it in a measurable form.
df["sentiment_label"] = LabelEncoder().fit_transform(df["sentiment_label"])

# We must assign the following values and make them measurable.
y = df["sentiment_label"]
X = df["reviewText"]

####################
#CountVectors
####################

# Count Vectors: frequency representations
# TF-IDF Vectors: normalized frequency representations
# Word Embeddings (Word2Vec, GloVe, BERT etc.)


#Words
# Numerical representations of words

# Characters
# Numerical representations of characters

#Ngram
a = """I will show this example in a longer text for clarity.
N-grams represent combinations of words used together and are used to produce features."""

TextBlob(a).ngrams(3)

####################
#CountVectors
####################

from sklearn.feature_extraction.text import CountVectorizer

corpus = ['This is the first document.',
           'This document is the second document.',
           'And this is the third one.',
           'Is this the first document?']

# Countvector method to create futures

# word frequency
vectorizer = CountVectorizer()
X_c = vectorizer.fit_transform(corpus)
feature_names = vectorizer.get_feature_names_out()
X_c.toarray()

# Unique words are the names of variables.
# We obtained the output by counting the frequency of occurrence of words in the relevant documents with the Countvektor method.
# First, we approach all the data as a single text and remove the unique words.
# Then these names become column names. We reflect the frequency of occurrence of these names in the relevant documents.

# N-gram frequency
vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(2, 2))
X_n = vectorizer2.fit_transform(corpus)
feature_names2= vectorizer2.get_feature_names_out()
X_n.toarray()

# We applied the countvector method to our own data.
vectorizer = CountVectorizer()
X_count = vectorizer.fit_transform(X)

feature_names3=vectorizer.get_feature_names_out()[10:15]
X_count.toarray()[10:15]


####################
#TF-IDF
####################
# Standardization is made focusing on the frequency of occurrence of words in documents and the frequency of occurrence of words in the entire corpus.

# TF:Term Frequency
# (frequency of term t in the relevant document/total number of terms in the document)

from sklearn.feature_extraction.text import TfidfVectorizer
tf_idf_word_vectorizer = TfidfVectorizer()
X_tf_idf_word = tf_idf_word_vectorizer.fit_transform(X)

# IDF:Inverse Document Frequency
# 1+ loge((total number of documents+1)/(number of documents containing term t+1)

tf_idf_ngram_vectorizer = TfidfVectorizer(ngram_range=(2, 3))
X_tf_idf_ngram = tf_idf_ngram_vectorizer.fit_transform(X)

# Calculating TF*DF

# Performing L2 Normalization
# Find the square root of the sum of squares of the rows, divide all cells in the relevant row by the value you found


####################
#5. Sentiment Modeling
####################

#1. Text Preprocessing
#2. Text Visualization
#3. Sentiment Analysis
#4. Feature Engineering
#5. Sentiment Modeling

####################
# Logistic Regression
####################
# We will apply the machine learning method. The linear form classification method used for classification problems

# Let's fetch the tf_idf data as an independent variable and assign it to the dependent variable as y.
log_model = LogisticRegression().fit(X_tf_idf_word, y)

cross_val_score(log_model,
                 X_tf_idf_word,
                 y,
                 scoring="accuracy",
                 cv=5).mean()
#83 of our 100 predictions are successful.

# If there is a new comment, let's ask the model.
new_review = pd.Series("this product is great") # Action Sequence 0
new_review = pd.Series("look at that shit very bad")# Action Sequence 0
new_review = pd.Series("it was good but I am sure that it fits me")# Action Sequence 0

# We pass the new comment through tfidvectorizer. Because tfidvectorizer was used when creating data.
# I need to perform this process again in the same way and convert the newly received review.

new_review = TfidfVectorizer().fit(X).transform(new_review) # Process Sequence 1

# Let's call the model and let it predict whether it is positive or negative based on the new comment.
log_model.predict(new_review) # Process Sequence 2

# What if we want to extract comments from the original data set and ask these comments to the model?
random_review = pd.Series(df["reviewText"].sample(1).values)

# let's ask the model now
new_review = TfidfVectorizer().fit(X).transform(random_review)

log_model.predict(new_review)


####################
# Random Forests
####################
#3 Let's create different methods and compare them

#CountVectors
rf_model = RandomForestClassifier().fit(X_count, y)
cross_val_score(rf_model, X_count, y, cv=5, n_jobs=-1).mean()

We classify with #0.83.
# ------------------------------------------------- ---------------

# TF-IDF Word-Level
rf_model = RandomForestClassifier().fit(X_tf_idf_word, y)
cross_val_score(rf_model, X_tf_idf_word, y, cv=5, n_jobs=-1).mean()

# tf_idf scores of word frequencies will be calculated
# We scored with 0.82
# ------------------------------------------------- --------------


#TF-IDF N-GRAM
rf_model = RandomForestClassifier().fit(X_tf_idf_ngram, y)
cross_val_score(rf_model, X_tf_idf_ngram, y, cv=5, n_jobs=-1).mean()

# Let's look at the scores according to word phrases
#0.78

# The x_count method seems like a better method.
# RandomForest has hyperparameters that the developer must optimize externally, which it cannot learn internally.
# These hyperparameters may affect the results a little more.

####################
# Hyperparameter Optimization
####################
# We fetch an empty model object.
rf_model = RandomForestClassifier(random_state=17)

# Let's set the Hyperparameters in the Hyperparameter set
rf_params = {"max_depth": [8, None],
              "max_features": [7, "auto"],
              "min_samples_split": [2, 5, 8],
              "n_estimators": [100, 200]}

# To evaluate the success of RandomForest; Gridsearchcv method
rf_best_grid = GridSearchCV(rf_model,
                             rf_params,
                             cv=5,
                             n_jobs=-1,
                             verbose=1).fit(X_count, y)

rf_best_grid.best_params_

# When it comes with different combinations, it is necessary to create the final model with the resulting value.
rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X_count, y)

# Let's evaluate our mistake
cross_val_score(rf_final, X_count, y, cv=5, n_jobs=-1).mean()

#0.84 is our error.

# As someone who wants to analyze comments about competitors, we want to define whether a comment is negative or positive.
# We want a simple model object to return negative or positive when asked.
# Therefore, let the vectorized relationship for the dependent and independent variables be modeled, and when the comment is asked, the negative or positive state will appear.
# Accordingly, first we used logregression and then we used randomforest. And thus the modeling was completed.
