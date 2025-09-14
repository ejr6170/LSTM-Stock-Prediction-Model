# LSTM-Stock-Prediction-Model
Personal project of mine
Currently only implemented Apple (APPL) stock to the model, in the near future it will support most stocks on the market and at some point hopefully cryptocurrency.

Made Using
- Pandas #Data Manipulation
- yFinance #Stock Data
- TensorFlow #LLM
- SKLearn #Normalize Data
- MatPlotLib #Visualization
- Keras Tuner #Hyperparameter tuning 
- Python #of course...

Reduced test RMSE from 14.06 to 7.19 (~49%)!!!
Trend accuracy still needs a lot of work, same with the entire model as its pretty barebones but its a start. 

Features
- 10 Technical Indicators
- Volume Indicator
- Visual plot representation of the model & prediction
- HyperParameter Tuning

Train Price RMSE: 5.44                                                                                                                                                                               Test Price RMSE: 7.19  

How this script works -
- Retrieves Apple's OHLCV stock data from 2020 to 2025 using yfinance 
- Creates sequences for LSTM input and prepares multi-output targets(price and trend)
- leverages hyperparameter tuning, thanks to Keras Tuner
- Trains the model with early stopping and evalutaes it on a train/test split(80/20)
- Plots results and predicts the next day's price and trend


