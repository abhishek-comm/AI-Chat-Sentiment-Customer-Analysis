#  AI Chat Sentiment & Customer Analysis


import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter
import nltk
import re
import os

# Download required NLTK data 
nltk.download("vader_lexicon", quiet=True)
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

# Create output folder if it doesn't exist
os.makedirs("output", exist_ok=True)


# Data loading

df = pd.read_csv("customer_chats.csv")
print("Total messages loaded:", len(df))
print()


# Text Cleaning

def clean_text(text):
    text = text.lower()                         # making lowercase
    text = re.sub(r"[^a-z\s]", " ", text)       # removes punctuation
    text = re.sub(r"\s+", " ", text).strip()    # removes extra spaces
    return text

df["clean_message"] = df["message"].apply(clean_text)
print("Text cleaning done.")


# Sentiment Analysis using VADER

sia = SentimentIntensityAnalyzer()

def get_sentiment(text):
    score = sia.polarity_scores(text)["compound"]
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

df["sentiment"] = df["clean_message"].apply(get_sentiment)

print("Sentiment Analysis Results:")
print(df["sentiment"].value_counts())
print()


# Finding most common complaints

STOPWORDS = set(stopwords.words("english"))

# Take only negative messages
negative_messages = df[df["sentiment"] == "Negative"]["clean_message"]

# Get all words from negative messages
all_words = []
for message in negative_messages:
    for word in message.split():
        if word not in STOPWORDS and len(word) > 3:
            all_words.append(word)

# Count word frequency
word_counts = Counter(all_words)
top_complaints = word_counts.most_common(10)

print("Top Complaint Keywords:")
for word, count in top_complaints:
    print(f"  {word}: {count} times")
print()


# Categorize chats into topics

def get_topic(text):
    if any(word in text for word in ["delivery", "shipping", "package", "arrived", "order"]):
        return "Delivery"
    elif any(word in text for word in ["refund", "payment", "charged", "money", "paid"]):
        return "Payment"
    elif any(word in text for word in ["app", "website", "login", "error", "crash"]):
        return "Technical"
    elif any(word in text for word in ["product", "quality", "item", "color", "size"]):
        return "Product"
    elif any(word in text for word in ["support", "agent", "helpline", "response"]):
        return "Support"
    else:
        return "General"

df["topic"] = df["clean_message"].apply(get_topic)

print("Topic Distribution:")
print(df["topic"].value_counts())
print()


# Visualizations

# Chart 1: Sentiment Pie Chart 
sentiment_counts = df["sentiment"].value_counts()
colors = ["#2ecc71", "#e74c3c", "#95a5a6"]

plt.figure(figsize=(6, 5))
plt.pie(sentiment_counts, labels=sentiment_counts.index,
        autopct="%1.1f%%", colors=colors, startangle=140)
plt.title("Sentiment Distribution of Customer Chats")
plt.savefig("output/sentiment_chart.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: output/sentiment_chart.png")


# Chart 2: Topic Bar Chart 
topic_counts = df["topic"].value_counts()

plt.figure(figsize=(7, 4))
plt.bar(topic_counts.index, topic_counts.values, color="#3498db", edgecolor="white")
plt.title("Number of Chats by Topic")
plt.xlabel("Topic")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("output/topic_chart.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: output/topic_chart.png")


# Chart 3: Top Complaint Keywords 
words  = [w for w, c in top_complaints]
counts = [c for w, c in top_complaints]

plt.figure(figsize=(8, 4))
plt.bar(words, counts, color="#e74c3c", edgecolor="white")
plt.title("Top Keywords in Negative Messages")
plt.xlabel("Keyword")
plt.ylabel("Frequency")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("output/complaints_chart.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: output/complaints_chart.png")


# Generate summary report

total       = len(df)
positive    = len(df[df["sentiment"] == "Positive"])
negative    = len(df[df["sentiment"] == "Negative"])
neutral     = len(df[df["sentiment"] == "Neutral"])
top_issue   = df[df["sentiment"] == "Negative"]["topic"].value_counts().idxmax()
top_keyword = top_complaints[0][0]

report = f"""
====================================================
  CUSTOMER CHAT ANALYSIS - SUMMARY REPORT
====================================================

DATASET
  Total messages analysed : {total}

SENTIMENT RESULTS
  Positive : {positive} messages ({positive/total*100:.1f}%)
  Negative : {negative} messages ({negative/total*100:.1f}%)
  Neutral  : {neutral} messages  ({neutral/total*100:.1f}%)

TOP COMPLAINT KEYWORDS (from negative messages)
  {", ".join([w for w, c in top_complaints[:5]])}

TOPIC DISTRIBUTION
{df["topic"].value_counts().to_string()}

KEY INSIGHT
  Most complaints are about: {top_issue}
  Most common complaint word: "{top_keyword}"

APPROACH USED
  - Text cleaning  : lowercase, remove punctuation
  - Sentiment      : VADER (NLTK) - rule-based NLP model
  - Topic tagging  : keyword matching
  - Complaints     : word frequency on negative messages
  - Visuals        : Matplotlib charts

====================================================
"""

with open("output/summary_report.txt", "w") as f:
    f.write(report)

print(report)
print("Saved: output/summary_report.txt")
print()
print("ALL DONE! Check the output/ folder.")
