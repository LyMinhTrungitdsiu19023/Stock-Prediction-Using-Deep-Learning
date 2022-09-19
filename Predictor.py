import pandas as pd
import numpy as np 
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler 
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM 

import math
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt 

class Predictor: #Create class Predictor using LSTM
  
  def __init__(self,df_ticker:pd.DataFrame):
    self.scaler = MinMaxScaler(feature_range=(0,1))
    self.data=df_ticker.filter(['Close'])            #pick columns Close
    self.dataset = np.array(self.data)               #change 'data' to numpy array

  def create_scaler_trainsize_test_size(self):                          #function to calculate the size of train and test dataset, scaling dataset
    scaled_data =self.scaler.fit_transform(self.dataset.reshape(-1,1)) 
    training_size = int(len(scaled_data)*0.8)
    test_size = len(scaled_data)-training_size  
    return scaled_data, training_size, test_size

  def split_data(self):                                                                                    #function to split dataset into train and test dataset
    scaled_data, training_size, test_size  = self.create_scaler_trainsize_test_size()
    train_data,test_data = scaled_data[0:training_size,:], scaled_data[training_size:len(scaled_data),:1] ##splitting dataset into train and test split
    return train_data, test_data
  
  def create_dataset(self,dataset, time_step=1):   # convert an array of values into a dataset matrix
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
      a = dataset[i:(i+time_step), 0]              ###i=0, X=0,1,2,3-----99   Y=100 
      dataX.append(a)
      dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY) 

  def create_xytrain_xytest(self):                                        #function to create X_train, Y_train, X_test and Y_test (numpy array dataset)
    time_step = 66
    train,test = self.split_data()                                          #train_data and test_data
    x_train, y_train = self.create_dataset(train, time_step)
    x_train = x_train.reshape(x_train.shape[0],x_train.shape[1] , 1)
    x_test, y_test = self.create_dataset(test, time_step) 
    return x_train, y_train, x_test, y_test, time_step

  def create_predictions_datapast(self):                                                                #fundtion to predict the past data(from 2018 to 6/2022)
    time_step = self.create_xytrain_xytest()[4]
    training_size = self.create_scaler_trainsize_test_size()[1]

    train_dataset = self.data[:training_size].iloc[time_step + 1: , :]

    test_dataset = self.data[training_size:].iloc[time_step + 1: , :]  

    data_plotting = pd.concat((train_dataset, test_dataset))                                          #data_plotting is the dataset having trained and tested dataset 

    data_prediction_full = data_plotting['Close']                                                           #data_prediction_full is the dataset with Close column and Prediction(of past dataset) columns

    data_prediction_full = self.scaler.fit_transform(data_prediction_full.to_numpy().reshape(-1,1)) 
    data_plotting_future = data_plotting.reset_index()                                                     #data_plotting future is the dataset having data_plotting and the 66 days predicted future dataset(only 1 model)
    return data_plotting, data_plotting_future, data_prediction_full

  def setParameters(self, start, end, step, epochs, batch_size):  #setter function
    self.start = start
    self.end = end
    self.step = step 
    self.epochs = epochs
    self.batch_size = batch_size

  def getParameters(self):                                                #getter function
    return self.start, self.end, self.step, self.epochs, self.batch_size

  def LTSM_model(self, start, end, step, epochs, batch_size):                               #Build model and allowing the Controller change the Parameters(layer of 1st model, layer of last model, step layer between the next 2 model ) 
    list_dataframe = []
    x_train, y_train, x_test, y_test, time_step = self.create_xytrain_xytest()
    data_plotting_future = self.create_predictions_datapast()[1] 
    data_prediction_full = self.create_predictions_datapast()[2]
    for j in range(start, end, step): 
    
      model = Sequential()
      model.add(LSTM(j,return_sequences=True,input_shape=(x_train.shape[1],1)))
      model.add(LSTM(j,return_sequences=True))
      model.add(LSTM(j))
      model.add(Dense(1))
      model.compile(loss='mean_squared_error',optimizer='adam')
      model.fit(x_train,y_train,epochs=epochs,batch_size = batch_size,verbose=1)

      #Prediction    
      lst_output=[]

      x_input = data_prediction_full[len(data_prediction_full)-time_step:].reshape(1,-1) 

      temp_input=list(x_input)
      temp_input=temp_input[0].tolist() 

      n_steps=66
      i=0

      while(i<66):
      
          if(len(temp_input)>66):
            x_input=np.array(temp_input[1:])
            x_input=x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            lst_output.extend(yhat.tolist())
            i=i+1
          else:
            x_input = x_input.reshape((1, n_steps,1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            i=i+1 
      future_price = self.scaler.inverse_transform(lst_output) 
      data_future_2 = pd.DataFrame(columns = ['Date', f'Model_{j}'])
      data_future_2['Date'] = pd.date_range(start = data_plotting_future['Date'].iloc[-1] + pd.Timedelta(days=1), periods = 66)  
      data_future_2[f'Model_{j}'] = future_price  
      list_dataframe.append(data_future_2.set_index('Date')) 

    new_data_predictions = pd.concat(list_dataframe, axis = 1) 
    return new_data_predictions #The models's dataframe with predictions of 66 days in future

  def result(self):                                                                               #Visualization the data ans show dataframe
    start, end, step, epochs, batch_size = self.getParameters()

    data_plotting = self.create_predictions_datapast()[0]
    data_past =  data_plotting[['Close']] 
    data_past['Lower'] = np.nan
    data_past['Upper'] = np.nan

    data_past['Lower'].iloc[-1] = data_past['Close'].iloc[-1] 
    data_past['Upper'].iloc[-1] = data_past['Close'].iloc[-1] 

    new_data_predictions = self.LTSM_model(start,end, step, epochs, batch_size)
    min_df = new_data_predictions.min(axis = 1).to_frame().rename(columns={0: 'Lower'})

    max_df = new_data_predictions.max(axis = 1).to_frame().rename(columns={0: 'Upper'}) 

    data_forecast = pd.concat((min_df, max_df), axis = 1) 
    df_past_forecast = pd.concat((data_past, data_forecast)).reset_index()
    df_past_forecast['Date'] = pd.to_datetime(df_past_forecast['Date']).dt.date
    df_past_forecast = df_past_forecast.set_index('Date') 
    
    plt.figure(figsize=(30,8))
    plt.title('Stock Price Simiulations')
    plt.xlabel('Date',fontsize=18)
    plt.ylabel('Price',fontsize=18)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m/%Y"))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval = 60)) 
    plt.plot(df_past_forecast['Close']) 
    plt.plot(df_past_forecast['Upper'], color = 'red') 
    plt.plot(df_past_forecast['Lower'], color = 'red') 
    plt.fill_between(df_past_forecast.index, df_past_forecast['Upper'], df_past_forecast['Lower'], color = 'lightgreen') 
    plt.axhline(y = np.min(data_forecast['Lower']), label=f"Lowest price: {np.min(data_forecast['Lower'])}",color='r', linestyle='--', linewidth=1)  
    plt.legend(loc='lower left')


    return data_forecast.to_csv("data_forecast"), new_data_predictions.to_csv("new_data")  #Print final dataframe

    # return plt.show(),  data_forecast, new_data_predictions  #Print final dataframe
