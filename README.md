# Customer Chat Sentiment Analysis

## What this does
Analyses customer support chat messages to find:
- Sentiment (Positive / Negative / Neutral)
- Most common complaints
- Topic categories (Delivery, Payment, Technical, Product, Support)
- Generates charts and a summary report

## How to run

```bash
pip install pandas matplotlib nltk
python analysis.py
```

## Output
All results saved in the `output/` folder:
- `sentiment_chart.png` — pie chart of sentiments
- `topic_chart.png` — bar chart of topics
- `complaints_chart.png` — top complaint keywords
- `summary_report.txt` — full text summary

## Files
- `analysis.py` — the entire analysis (single file, ~130 lines)
- `customer_chats.csv` — mock dataset (39 messages, 5 categories)

## Tools used
- Python, Pandas, NLTK (VADER), Matplotlib

## Dataset note
Mock dataset created as per task instructions (real data unavailable).
Covers 5 business categories: Delivery, Payment, Technical, Product, Support.
