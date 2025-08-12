# 🐦 Tweet Sentiment Analysis (Text + Emojis)

## Overview
This project analyzes tweets and classifies their sentiment as **Positive**, **Neutral**, or **Negative** using text preprocessing, TF-IDF features, and a Naive Bayes classifier.
It leverages emoji sentiment data (`Emoji_Sentiment_Data_v1.0.csv`) and prepared tweet datasets.

## Features
- Clean & preprocess tweets.
- Convert text to TF-IDF features.
- Train a Naive Bayes classifier.
- Evaluate with accuracy and visualizations.
- Works with emoji sentiment data.

## Project Structure
```
.
├─ src/
│  ├─ Emotions_Detection.py
│  └─ testing_dataset.py
├─ dataset/
│  ├─ Emoji_Sentiment_Data_v1.0.csv
│  ├─ Positive.csv / Negative.csv (if provided)
│  └─ other csvs used by the scripts
├─ requirements.txt
├─ .gitignore
└─ README.md
```

## Quickstart (Windows PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\Activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python src/Emotions_Detection.py
python src/testing_dataset.py
```

> If you see unresolved imports in your IDE, ensure it uses the same interpreter that installed the packages.

## License
Educational use only.
