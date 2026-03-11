import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
train = pd.read_csv(r"C:\Users\INDRONIIL\Downloads\twitter_training.csv")
test = pd.read_csv(r"C:\Users\INDRONIIL\Downloads\twitter_validation.csv")
train
test

train = train.dropna()
test = test.dropna()

train = train.drop_duplicates()
test = test.drop_duplicates()

train.info()
test.info()

train.describe()
test.describe()

train.shape
test.shape

def clean_text(text):
    text = str(text).lower()
    # Removes special characters/numbers but keeps spaces
    text = re.sub(r'[^a-z\s]', '', text) 
    # Removes extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

train["Clean_Text"] = train["Text"].apply(clean_text)
test["Clean_Text"] = test["Text"].apply(clean_text)

cols = ["ID", "Topic", "Sentiment", "Extra_Col", "Text"]

train.columns = cols
test.columns = cols

train = train.dropna(subset=['Text']).drop_duplicates()
test = test.dropna(subset=['Text']).drop_duplicates()

plt.figure(figsize=(8, 5))
sns.countplot(x="Sentiment", data=train, order=train["Sentiment"].value_counts().index)
plt.title("Sentiment Distribution")
plt.show()

topic_sentiment = train.groupby(["Topic","Sentiment"]).size().unstack()

topic_sentiment.plot(kind="bar", stacked=True, figsize=(10,6))
plt.title("Sentiment by Topic")
plt.xlabel("Topic")
plt.ylabel("Number of Tweets")
plt.show()

top_topics = train["Topic"].value_counts().head(10)

top_topics.plot(kind="bar")
plt.title("Top 10 Discussed Topics")
plt.xlabel("Topic")
plt.ylabel("Number of Tweets")
plt.show()
