import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import naive_bayes
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


df_emoticon = pd.read_csv("Emoji_Sentiment_Data_v1.0.csv", encoding='utf-8')
df_emoticon = df_emoticon[df_emoticon['Unicode block'] == 'Emoticons']
df_emoticon.reset_index(inplace=True, drop=True)




_cols = ['Emoji', 'Negative', 'Neutral', 'Positive', 'Unicode name']
df_emoji = pd.read_csv("Emoji_Sentiment_Data_v1.0.csv", encoding='utf-8', usecols=_cols)
df_emoji




def get_sentiment(p, n, neu):
    if p > n or (p == n and neu % 2 != 0):
        return 1.0  # Positive
    else:
        return 0  # Negative


polarity_ls = []
for index, row in df_emoji.iterrows():
    polarity = get_sentiment(row['Positive'], row['Negative'], row['Neutral'])
    polarity_ls.append(polarity)



df_Emojis_2 = pd.DataFrame(polarity_ls, columns=['sentiment'])
df_Emojis_2['emoji'] = df_emoji['Emoji'].values
df_Emojis_2['name'] = df_emoji['Unicode name'].values
df_Emojis_2

print(df_Emojis_2)


pos_tweets = pd.read_csv("Positive.csv")
neg_tweets = pd.read_csv("Negative.csv")




def get_polarity(df, polarity):
    dpos = {'Text': list(df.columns.values), 'Sentiment': polarity}
    df_twt = pd.DataFrame(data = dpos).iloc[:1000]
    return df_twt

df_pos_tweet = get_polarity(pos_tweets, 1)

df_neg_tweet = get_polarity(neg_tweets, 0)


# corresponding emoticon sysmbols
txt_emoji = [
    ':)', ':P', ':D', ':|', ":'(", ':O', ":*", '<3', ':(', ';)',
    'xD', ':/', '=D']
emoji_pic =[
    '😊', '😛', '😄', '😐', '😢', '😲', '😘', '😍', '😧', '😉',
    '😁', '😒', '😀'
]


#convert text emoji to emoji_pic
def emotion_to_emoji(txt, text_emoji, emoji):
    temp = []
    for text in txt:
        for j in range(len(text_emoji)):
            if text == text_emoji[j]:
                text = emoji[j]
        temp.append(text)
    return ' '.join(temp)


def convert_emoticons(df_data):
    convrted_text = []
    for idx, row in df_data.iterrows():
        txt = [i for i in row['Text'].split()]
        emoji_found = emotion_to_emoji(txt, txt_emoji, emoji_pic)
        convrted_text.append(emoji_found)
    return convrted_text


pos_tweets_with_emojis = convert_emoticons(df_pos_tweet)
neg_tweets_with_emojis = convert_emoticons(df_neg_tweet)





# for additional inputs
add_emoji_txt = ['sad', 'unhappy', 'crying', 'smile', 'happy', 'love']
add_emoji_pic =['😔', '😧', '😆', '😭', '😊', '😍']


def add_emoji_text_data(df_data):
    reform_pos_text = []
    for ct in df_data:
        txt = [i for i in ct.split()]
        emoji_found = emotion_to_emoji(txt, add_emoji_txt, add_emoji_pic)
        reform_pos_text.append(emoji_found)
    return reform_pos_text

pos_conv_text = add_emoji_text_data(pos_tweets_with_emojis)

neg_conv_text = add_emoji_text_data(neg_tweets_with_emojis)


def new_df_emoji_tweet(data, polarity):
    temp = pd.DataFrame(columns=['sentiment', 'post'])
    temp['post'] = data
    temp['sentiment'] = polarity
    return temp


df_pos_tweets = new_df_emoji_tweet(pos_conv_text, 1)
df_neg_tweets = new_df_emoji_tweet(neg_conv_text, 0)


df_neg_tweets.to_csv("1k_data_tweets_emoticon_neg.csv")
df_pos_tweets.to_csv("1k_data_tweets_emoticon_pos.csv")



txt_emoji_pic = [
    '😊', '😛', '😄', '😐', '😢', '😲', '😘', '😍', '😧', '😉',
    '😁', '😒', '😀', '😔', '😧', '😆', '😭'
]


def emoji_checker(data_text):
    has_emoji = False
    for i in txt_emoji_pic:
        if i in data_text:
            has_emoji = True
    return has_emoji


def post_emoji_counter(df_tweets):
    c = 0
    for idx, row in df_tweets.iterrows():
        if emoji_checker(row['post']):
            c += 1
    return f'{c} / 1000'


df_neg_tweets2 = pd.DataFrame(df_neg_tweets)
df_pos_tweets2 = pd.DataFrame(df_pos_tweets)


def idx_with_emoji(df_tweets):
    idx_wemo = []
    c = 500
    for idx, row in df_tweets.iterrows():
        has_emoji = emoji_checker(row['post'])
        if has_emoji: idx_wemo.append(idx)
        if has_emoji and c > 0 : c -= 1
        if c == 0 : break
    return idx_wemo


pos_idxs = idx_with_emoji(df_pos_tweets2)
neg_idxs = idx_with_emoji(df_neg_tweets2)


df_pos_500 = pd.DataFrame(df_pos_tweets2)
df_neg_500 = pd.DataFrame(df_neg_tweets2)



df_pos_500.drop(df_pos_500.index[pos_idxs], inplace=True)
df_pos_500.reset_index(inplace=True, drop=True)




df_neg_500.drop(df_neg_500.index[neg_idxs], inplace=True)
df_neg_500.reset_index(inplace=True, drop=True)



# Concatenate the DataFrames
df_tweet_1000 = pd.concat([df_pos_500,df_neg_500])

# Reset the index
df_tweet_1000.reset_index(drop=True, inplace=True)

# Shuffle the DataFrame
df_tweet_1000 = df_tweet_1000.sample(frac=1).reset_index(drop=True)


df_tweet_1000.to_csv('1k_data_emoji_tweets_senti_posneg.csv')

df_test = pd.read_csv('1k_data_emoji_tweets_senti_posneg.csv')


df_emo_ls = df_emoticon[df_emoticon['Emoji'].isin(txt_emoji_pic)]
df_emo_ls = df_emo_ls.drop(columns=['Occurrences', 'Position', 'Negative', 'Neutral', 'Positive', 'Unicode block'])
df_emo_ls.reset_index(inplace=True, drop=True)

df_emo_ls.to_csv('15_emoticon_data.csv')
