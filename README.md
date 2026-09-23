\# AI-Powered Interactive Analysis of Cryptocurrency and Stock Markets



An interactive financial market analysis project comparing cryptocurrency and traditional stock markets using data preprocessing, financial feature engineering, machine learning, and visualization.



\## Research Publication



This research was presented at \*\*ISBM 2026\*\* and has been \*\*accepted for publication in Springer Nature, Lecture Notes in Networks and Systems (LNNS), 2026\*\*.



The final citation and publication link will be added once the paper is officially published online.



\## Assets Analyzed



\- Bitcoin (BTC)

\- Ethereum (ETH)

\- NASDAQ

\- S\&P 500



\*\*Study Period:\*\* 2021–2025



\## Project Overview



The project develops an interactive framework for analyzing cryptocurrency and traditional stock markets.



The workflow includes:



\- Financial data preprocessing

\- Time-series alignment

\- Feature engineering

\- Crypto-stock market comparison

\- Risk and volatility analysis

\- Machine learning classification

\- Interactive visualization using Streamlit



\## Final Datasets



Three processed datasets were created:



\- `FINAL\_Daily\_Analysis.csv`

\- `FINAL\_4H\_Aligned.csv`

\- `FINAL\_1H\_Crypto\_Aligned.csv`



\### Daily Dataset

Used for the main long-term cryptocurrency and stock-market comparison.



\### 4-Hour Dataset

Used for medium-term market structure analysis.



\### 1-Hour Dataset

Used for short-term cryptocurrency movement and volatility analysis.



\## Data Preprocessing



The preprocessing workflow includes:



\- Standardizing column names

\- Converting timestamps to datetime format

\- Sorting observations chronologically

\- Removing duplicate timestamps

\- Handling missing values

\- Resampling Ethereum hourly data into daily data

\- Aligning datasets using a common time index

\- Creating final analysis-ready datasets



Extreme observations were retained because crashes and booms represent important real financial market events.



\## Financial Feature Engineering



The project includes financial indicators such as:



\- Daily returns

\- 30-day returns

\- Rolling volatility

\- Drawdown

\- Boom indicators

\- Crash indicators

\- Rolling correlation

\- Convexity proxy

\- Liquidity-cost proxy

\- Risk ratio



\## Machine Learning



The interactive system includes:



\- Random Forest

\- Logistic Regression



The current Streamlit implementation demonstrates classification of high-risk Bitcoin market days using engineered financial indicators.



\## Interactive Dashboard



The Streamlit application contains four main stages:



1\. Input

2\. Preprocessing

3\. Process / Model

4\. Output \& Analysis



The dashboard provides interactive analysis and visualizations including:



\- Indexed price trend comparison

\- Daily return comparison

\- 30-day rolling volatility

\- Bitcoin crash-period detection

\- Bitcoin boom-period identification

\- Crypto vs. NASDAQ rolling correlation

\- Machine-learning performance metrics

\- Confusion matrix

\- Feature importance



\## Technologies Used



\- Python

\- Pandas

\- NumPy

\- Scikit-learn

\- Plotly

\- Streamlit



\## Project Files



```text

crypto-stock-market-analysis/

├── app.py

├── FINAL\_Daily\_Analysis.csv

├── FINAL\_4H\_Aligned.csv

├── FINAL\_1H\_Crypto\_Aligned.csv

├── requirements.txt

├── .gitignore

└── README.md

