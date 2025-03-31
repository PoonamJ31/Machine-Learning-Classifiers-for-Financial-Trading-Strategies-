# Machine-Learning-Classifiers-for-Financial-Trading-Strategies-
Project Overview
This project aims to implement and evaluate machine learning classifiers to predict stock trading signals based on historical financial data. By leveraging two distinct trading strategies—Next-Day Price Comparison and Moving Average Crossover—we analyze the performance of various models, including K-Nearest Neighbors (KNN), Random Forest (RF), and Gradient Boosting (GB). Our results provide insights into the effectiveness of machine learning in financial trading.

## Objectives
Develop machine learning models for predicting stock trading signals.

Compare the effectiveness of multiple trading strategies.

Evaluate model performance using key classification metrics.

# Technologies Used
Programming Language: Python

Libraries: NumPy, pandas, scikit-learn, yfinance

Machine Learning Models: KNN, Random Forest, Gradient Boosting

# Dataset
Source: Yahoo Finance (yfinance API)

Period: 2015 - 2023

# Features:

Historical closing prices

Trading volumes

50-day & 200-day moving averages

Daily returns

# Methodology
1. Data Collection & Preprocessing
Downloaded historical stock data from Yahoo Finance.

Handled missing values and outliers.

Engineered features such as moving averages and return rates.

Defined target variables for trading signals.

2. Trading Strategies Implemented
Strategy 1: Next-Day Price Comparison
Logic: Buy signal (+1) if the next day’s close price is higher than today’s close; otherwise, sell (-1).

Implementation:

python
Copy
Edit
df['Signal'] = np.where(df['Close'].shift(-1) > df['Close'], 1, -1)
Strategy 2: Moving Average Crossover
Logic:

Golden Cross (50-day MA > 200-day MA) → Buy signal (+1).

Death Cross (50-day MA < 200-day MA) → Sell signal (-1).

# Implementation:

python
Copy
Edit
df['50_MA'] = df['Close'].rolling(window=50).mean()
df['200_MA'] = df['Close'].rolling(window=200).mean()
df['Signal'] = np.where(df['50_MA'] > df['200_MA'], 1, -1)
3. Model Development & Evaluation
Training & Testing Split: 80% training, 20% testing.

# Models Used:

K-Nearest Neighbors (KNN): Simple and interpretable but sensitive to noise.

Random Forest (RF): Robust ensemble method.

Gradient Boosting (GB): Captures complex patterns effectively.

Performance Metrics: Accuracy, Precision, Recall, F1-score.

# Results & Model Performance
Model	Strategy 1 Accuracy	Strategy 2 Accuracy	Best Performing Strategy
KNN	40%	66.67%	Strategy 2
Random Forest	73.34%	66.67%	Strategy 1
Gradient Boosting	53.73%	78.68%	Strategy 2
# Key Insights
Random Forest performed best for short-term price prediction (Strategy 1).

Gradient Boosting was most effective for trend-based strategy (Strategy 2).

Ensemble methods (RF, GB) significantly outperformed KNN.

Future improvements: Hyperparameter tuning, adding macroeconomic indicators, backtesting with real trading scenarios.

# Challenges Faced
Handling missing data and computational inefficiencies with large datasets.

Avoiding overfitting when using moving averages and limited features.

Balancing sensitivity and specificity in trading signal predictions.

# Future Work
Implement deep learning models (e.g., LSTMs) for sequential data analysis.

Use reinforcement learning for dynamic trading strategies.

Perform real-time backtesting to simulate actual trading performance.

# Conclusion
This project highlights the potential of machine learning in financial trading. While Gradient Boosting demonstrated superior performance in trend-based strategies, Random Forest was more effective for short-term price movements. Future improvements will focus on incorporating real-time data, macroeconomic indicators, and deep learning approaches.

References
Python Libraries: yfinance, scikit-learn, NumPy, pandas.

Machine Learning for Asset Managers – Marcos López de Prado.

Advances in Financial Machine Learning – Marcos López de Prado.
