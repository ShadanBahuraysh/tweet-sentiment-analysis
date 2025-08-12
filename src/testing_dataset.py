import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import naive_bayes
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Load the emoji sentiment data and tweets data
df_emojis = pd.read_csv('15_emoticon_data.csv')
df_tweets = pd.read_csv('1k_data_emoji_tweets_senti_posneg.csv')

# TF-IDF vectorizer
vectorizer = TfidfVectorizer(use_idf=True, lowercase=True,
                              strip_accents='ascii', stop_words='english')
# Preprocess tweets data
X = vectorizer.fit_transform(df_tweets.post)
y = df_tweets.sentiment
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=None)

# Train a naive Bayes classifier
clf = naive_bayes.MultinomialNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
print("Test accuracy: {:.2f}%".format(accuracy*100))




# Visualize distribution of sentiments
fig = plt.figure(figsize=(5,5))
sns.countplot(x='sentiment', data=df_tweets)
plt.xticks(ticks=[0, 1], labels=['Negative', 'Positive'])
plt.show()

# Function to get sentiment from a CSV file
def get_sentiment_from_csv(filename):
    df = pd.read_csv(filename)
    input_array = df['post']
    print(input_array)
    input_vector = vectorizer.transform(input_array)
    pred_senti = clf.predict(input_vector)
    return pred_senti

# Function to classify tweets into positive and negative categories
def classify_tweets(filename, positive_file, negative_file):
    pred_sentiments = get_sentiment_from_csv(filename)
    df = pd.read_csv(filename)
    positive_tweets = df[pred_sentiments == 1]['post']
    negative_tweets = df[pred_sentiments == 0]['post']
    positive_tweets.to_csv(positive_file, index=False)
    negative_tweets.to_csv(negative_file, index=False)

# Call the function to classify tweets
filename = '1k_data_emoji_tweets_senti_posneg.csv'
positive_file = 'positive_tweets.csv'
negative_file = 'negative_tweets.csv'
classify_tweets(filename, positive_file, negative_file)



