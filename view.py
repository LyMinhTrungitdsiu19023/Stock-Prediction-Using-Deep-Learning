import pandas as pd 
class view: 
  def __init__(self, df:pd.DataFrame): 
    self.df = df
  def enterticker_normolize(self): 
        for i in range(10):
          input_ticker = input("Enter Ticker: ").upper()
 
          #Normolize
          if len(input_ticker) == 3:
              df_sorted_ticker = self.df.loc[self.df["Ticker"] == input_ticker]
                            #Sorting dataframe by Date increasing
              df_sorted_ticker = df_sorted_ticker.sort_values(by="Date") 
              df_sorted_ticker['Date'] = pd.to_datetime(df_sorted_ticker['Date']).dt.date 

              #Set Date into Index
              df_sorted_ticker = df_sorted_ticker.set_index('Date')   
              return df_sorted_ticker
              break 
          else:
              return print("Please enter 3 characters ticker") 