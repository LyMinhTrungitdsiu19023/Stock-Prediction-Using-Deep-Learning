from Predictor import *
class Controller(Predictor):
    def __init__(self, df:pd.DataFrame):
        self.pred = Predictor(df)
        self.pred.setParameters(4, 30, 4, 30, 64) #layer of 1st model, layer of last model, step of layer change between 2 model, epochs, batch_size 
        #Ex: 4 is the layer of 1st model, 10 is layer of last model, step is 4 -> we have model with layer 4, model with layer 8   
                                
    def showAll(self):  #function to show all chart and dataframe
        return self.pred.result() 

    def show_minmax(self):
        return self.pred.result()[1] #function to show datafram of boundary

    def show_data_model_predicted(self):    #function to show dataframe of model's prediction
        return self.pred.result()[2] 