# Data Sources

This project uses historical financial-market datasets collected from Kaggle and processed into aligned research datasets for the 2021–2025 study period.

## Bitcoin

Historical Bitcoin OHLCV data were obtained from Kaggle-hosted Bitcoin historical datasets.

Examples used during data collection included:

- Bitcoin Historical Data (2014–2025)
- Bitcoin Historical Datasets (2018–2024/2025)

The processed project datasets use Bitcoin daily, 4-hour, and 1-hour observations.

## Ethereum

Ethereum data were obtained from the Kaggle-hosted:

**Bybit ETH/USDT Historical Data (2021–2025)**

The project uses 1-hour and 4-hour ETH data. Daily ETH observations were reconstructed by resampling the 1-hour OHLCV data.

## NASDAQ

NASDAQ-100 historical market data were obtained from the Kaggle-hosted:

**NASDAQ 100 Historical Price Data**

Daily and 4-hour observations were used in the final analysis workflow.

## S&P 500

Bitcoin and S&P 500 daily price data were obtained from the Kaggle-hosted:

**Bitcoin and S&P 500 Historical Prices**

The S&P 500 series was used in the daily cross-market comparison.

## Processed Datasets in This Repository

The original source datasets were cleaned, aligned, filtered, and transformed into:

- `FINAL_Daily_Analysis.csv`
- `FINAL_4H_Aligned.csv`
- `FINAL_1H_Crypto_Aligned.csv`

Processing included timestamp standardization, chronological sorting, duplicate removal, time alignment, missing-value handling, resampling where required, and financial feature engineering.

## Licensing and Attribution

The original datasets remain subject to the licensing and usage terms specified by their respective Kaggle dataset authors and upstream data providers.

Users of this repository should review the original dataset pages before redistributing or reusing source-derived data.
