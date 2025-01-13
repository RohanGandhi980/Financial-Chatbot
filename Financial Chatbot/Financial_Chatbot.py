#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import yfinance as yf


# In[3]:


def fetch_stock_data(symbol, start_date, end_date):
    
    data = yf.download(symbol, start=start_date, end=end_date)
    data['Return'] = data['Close'].pct_change() 
    data.dropna(inplace=True)
    
    return data


# In[5]:


def preprocess_data(data):
    
    X = data[['Open', 'High', 'Low', 'Volume', 'Return']]
    y = data['Close']
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler


# In[7]:


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    
    print(f"Model Trained. Mean Squared Error: {mse:.2f}")
    
    return model


# In[9]:


def predict_price(model, scaler, stock_data):

    X_latest = stock_data[['Open', 'High', 'Low', 'Volume', 'Return']].iloc[-1].values.reshape(1, -1)
    X_latest_scaled = scaler.transform(X_latest)
    
    predicted_price = model.predict(X_latest_scaled)[0]
    
    return predicted_price


# In[11]:


def financial_chatbot():
    print("Welcome to the Financial Chatbot! Type 'exit' to quit.")
    symbol = input("Enter a stock symbol (e.g., AAPL, TSLA): ").upper()
    while symbol.lower() != 'exit':
        try:
            
            stock_data = fetch_stock_data(symbol, start_date='2020-01-01', end_date='2023-01-01')
            X, y, scaler = preprocess_data(stock_data)
            
            
            model = train_model(X, y)
            
            
            predicted_price = predict_price(model, scaler, stock_data)
            print(f"The predicted current price of {symbol} is ${predicted_price:.2f}.")
            
        except Exception as e:
            print(f"Error: {e}")
        
        
        symbol = input("Enter another stock symbol or type 'exit' to quit: ").upper()


# In[ ]:


financial_chatbot()



# In[ ]:




