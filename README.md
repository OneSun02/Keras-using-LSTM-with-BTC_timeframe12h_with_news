# Predicting Bitcoin Price Using LSTM and News Sentiment Analysis

## Overview
This project aims to predict Bitcoin price movements based on historical price data and financial news sentiment analysis. The model leverages LSTM (Long Short-Term Memory) networks for time-series forecasting and utilizes TextBlob for sentiment analysis of news articles.

## Technologies Used
- **Deep Learning**: Keras, TensorFlow
- **Data Processing**: Pandas, NumPy
- **Natural Language Processing (NLP)**: TextBlob
- **Visualization**: Matplotlib, Seaborn
- **Programming Language**: Python

## Project Workflow
1. **Data Collection**: Gather historical Bitcoin price data and financial news.
2. **Data Preprocessing**: Clean and prepare the price data and tokenize text data.
3. **Sentiment Analysis**: Use TextBlob to analyze the sentiment of news headlines.
4. **Feature Engineering**: Combine price data with sentiment scores.
5. **Model Training**: Train an LSTM model on the combined dataset.
6. **Prediction & Evaluation**: Predict Bitcoin prices and evaluate the model's performance.

## How to Run the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/OneSun02/Keras-using-LSTM-with-BTC_timeframe12h_with_news.git
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
4. Open the `.ipynb` file and execute the cells in order.

## Results
- The model successfully learns from price history and news sentiment to predict future Bitcoin prices.
- Performance evaluation includes RMSE (Root Mean Squared Error) and visualization of actual vs. predicted prices.

## Future Improvements
- Incorporate more advanced NLP techniques such as transformers (BERT, FinBERT).
- Experiment with different timeframes and alternative data sources.
- Optimize hyperparameters for better prediction accuracy.

## Author
[OneSun02](https://github.com/OneSun02)

