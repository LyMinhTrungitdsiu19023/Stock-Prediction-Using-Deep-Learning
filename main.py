# import Controller
# import view 
from Predictor import * 
from Controller import *
from view import * 
import pandas as pd
class main: 
    df = pd.read_csv("StockPrice.csv") 
    view = view(df).enterticker_normolize()
    a = Controller(view) 
    B = a.showAll()
    print(A)
