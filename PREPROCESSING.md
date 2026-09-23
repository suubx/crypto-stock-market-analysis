\# Data Preprocessing Workflow



This document summarizes the preprocessing workflow used to prepare the financial datasets for the project.



\## Raw Financial Data



The analysis used historical financial data for:



\- Bitcoin (BTC)

\- Ethereum (ETH)

\- NASDAQ

\- S\&P 500



The study period was 2021–2025.



Multiple temporal resolutions were collected, including daily, 4-hour, and 1-hour market data.



\## 1. Data Loading and Standardization



The raw CSV datasets were loaded using Python and Pandas.



The preprocessing process included:



\- Standardizing column names

\- Detecting date/time columns

\- Converting timestamps to Pandas datetime format

\- Sorting observations chronologically

\- Removing duplicate timestamps



\## 2. Time-Period Filtering



The datasets were restricted to the study period:



2021-01-01 to 2025-12-31.



\## 3. Ethereum Daily Resampling



Ethereum daily data was generated from the Ethereum 1-hour dataset.



The hourly observations were resampled using:



\- Open: first value

\- High: maximum value

\- Low: minimum value

\- Close: last value

\- Volume: sum



This allowed Ethereum to be aligned with the daily cryptocurrency and stock-market datasets.



\## 4. Dataset Alignment



The datasets were aligned using a common datetime index.



Three final analysis layers were produced:



\### Daily Analysis Dataset



Contains:



\- Bitcoin

\- Ethereum

\- NASDAQ

\- S\&P 500



This dataset is used for the main long-term cryptocurrency and stock-market comparison.



\### 4-Hour Analysis Dataset



Contains:



\- Bitcoin

\- Ethereum

\- NASDAQ



This dataset supports medium-term market structure analysis.



\### 1-Hour Crypto Dataset



Contains:



\- Bitcoin

\- Ethereum



This dataset supports short-term cryptocurrency movement and volatility analysis.



\## 5. Missing Values



Missing observations occurred mainly because cryptocurrency markets operate continuously while traditional stock markets close during weekends and holidays.



Missing values were handled using time-series-based methods such as:



\- Forward filling

\- Interpolation

\- Limited filling of stock-market gaps



Mean imputation was avoided because it could distort the temporal structure of financial data.



\## 6. Outlier Treatment



Extreme values were retained.



Large positive and negative market movements can represent real boom and crash events, so removing them would reduce the usefulness of volatility and risk analysis.



\## 7. Feature Engineering



Financial indicators were created for Bitcoin, Ethereum, and NASDAQ.



The engineered features included:



\- Daily returns

\- 30-day returns

\- 30-day rolling volatility

\- Drawdown

\- Crash indicators

\- Boom indicators

\- Convexity proxy

\- Liquidity-cost proxy

\- Rolling crypto-stock correlation

\- Risk ratio



\## 8. Final Processed Datasets



The preprocessing workflow produced:



\- `FINAL\_Daily\_Analysis.csv`

\- `FINAL\_4H\_Aligned.csv`

\- `FINAL\_1H\_Crypto\_Aligned.csv`



The final daily dataset contains 1,731 observations and 44 variables.



\## Reproducibility Note



The original preprocessing notebook is not included in this repository.



The processed datasets and a summary of the preprocessing methodology are provided to document the analytical workflow used in the research.

