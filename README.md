# LSTM-Stock-Prediction-Model
Personal project of mine

Currently only implemented Apple (APPL) stock to the model, in the near future it will support most stocks on the market and at some point hopefully cryptocurrency.

**Now using multiple models in an ensemble**

**Prediction based on majority vote from models**

*Made Using*
- *Pandas #Data Manipulation*
- *yFinance #Stock Data*
- *TensorFlow #ML*
- *SKLearn #Normalize Data*
- *MatPlotLib #Visualization*
- *Keras Tuner #Hyperparameter tuning*
- *Python #of course...*

**Reduced test RMSE from 14.06 to 7.19 (~49%)!!!**

# Features
- 10 Technical Indicators (SMA, EMA, RSI, MACD, BB Upper & Lower, Stochastic K, ATR)
- Volume Indicator
- Visual plot representation of the model & prediction
- HyperParameter Tuning

Train Price RMSE: 5.44  
Test Price RMSE: 7.19  

## How this script works 
- Retrieves Apple's OHLCV stock data from 2020 to 2025 using yfinance 
- Creates sequences for LSTM input and prepares multi-output targets(price and trend)
- leverages hyperparameter tuning, thanks to Keras Tuner
- Trains the models with early stopping and evalutaes it on a train/test split(80/20)
- Plots results and predicts the next day's price and trend

# How to install & run 
* Clone github in terminal
* CD into newly created github folder
* Install prerequesities in terminal
`pip install yfinance pandas numpy scikit-learn tensorflow matplotlib`
* run the script `python enhanced_lstm_tuned.py`
* wait ~60 minutes for training to complete, a prediction price plot will pop up automatically 




