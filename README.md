# Machine-Learning-Classifiers-for-Financial-Trading-Strategies-
Project Overview
This project aims to implement and evaluate machine learning classifiers to predict stock trading signals based on historical financial data. By leveraging two distinct trading strategies—Next-Day Price Comparison and Moving Average Crossover—we analyze the performance of various models, including K-Nearest Neighbors (KNN), Random Forest (RF) and Gradient Boosting (GB). Our results provide insights into the effectiveness of machine learning in financial trading.

## Objectives
Develop machine learning models for predicting stock trading signals.

Compare the effectiveness of multiple trading strategies.

Evaluate model performance using key classification metrics. 

## Technologies Used
Programming Language: Python

Libraries: NumPy, pandas, scikit-learn, yfinance

Machine Learning Models: KNN, Random Forest, Gradient Boosting

## Dataset
Source: Yahoo Finance (yfinance API)

Period: 2015 - 2023

Features:

Historical closing prices

Trading volumes

50-day & 200-day moving averages

Daily returns

## Methodology
### Data Collection & Preprocessing

Downloaded historical stock data from Yahoo Finance.

Handled missing values and outliers.

Engineered features such as moving averages and return rates.

Defined target variables for trading signals.

### Trading Strategies Implemented

Strategy 1: Next-Day Price Comparison
   
Logic: Buy signal (+1) if the next day’s close price is higher than today’s close; otherwise, 
sell (-1).

Strategy 2: Moving Average Crossover

Logic: Golden Cross (50-day MA > 200-day MA) → Buy signal (+1).
      Death Cross (50-day MA < 200-day MA) → Sell signal (-1).

### Model Development & Evaluation

Training & Testing Split: 80% training, 20% testing.

Models Used:
K-Nearest Neighbors (KNN): Simple and interpretable but sensitive to noise.

Random Forest (RF): Robust ensemble method.

Gradient Boosting (GB): Captures complex patterns effectively.

Performance Metrics: Accuracy, Precision, Recall, F1-score.

## Results & Model Performance
The K-Nearest Neighbors (KNN) model achieved 40% accuracy for next-day price prediction (Strategy 1) and 66.67% accuracy for the moving average crossover strategy (Strategy 2), indicating its limited effectiveness in financial trading.

The Random Forest model performed significantly better, with 73.34% accuracy for next-day price prediction and 66.67% accuracy for the moving average crossover strategy, making it the best performer for short-term price movements.

The Gradient Boosting model showed moderate performance in next-day price prediction, achieving 53.73% accuracy, but excelled in the moving average crossover strategy with 78.68% accuracy, making it the most effective model for trend-based trading signals.

Overall, Random Forest was the best model for short-term predictions, while Gradient Boosting proved to be the most effective in capturing market trends using moving averages.

## Key Insights
Random Forest performed best for short-term price prediction (Strategy 1).

Gradient Boosting was most effective for trend-based strategy (Strategy 2).

Ensemble methods (RF, GB) significantly outperformed KNN.

Future improvements: Hyperparameter tuning, adding macroeconomic indicators, backtesting with real trading scenarios.

## Challenges Faced
Handling missing data and computational inefficiencies with large datasets.
Avoiding overfitting when using moving averages and limited features.
Balancing sensitivity and specificity in trading signal predictions.

## Conclusion
This project highlights the potential of machine learning in financial trading. While Gradient Boosting demonstrated superior performance in trend-based strategies, Random Forest was more effective for short-term price movements. Future improvements will focus on incorporating real-time data, macroeconomic indicators, and deep learning approaches.
