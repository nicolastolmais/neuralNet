#Nicolas Tolmais
#Capstone project
#Last Updated: 4-19-16
#Description: A neural network to predict the movement of the stock market.

import tablib
from pybrain.structure import RecurrentNetwork
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import TanhLayer
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet

#values for normalizaiton
close_min_Value = 7.2386
close_max_Value = 133
volume_min_Value = 13046.45
volume_max_Value = 841930.153
opens_min_Value = 7.3879
opens_max_Value = 134.455
high_min_Value = 7.5557
high_max_Value = 134.54
low_min_Value = 7.1657
low_max_Value = 131.4
seven_day_max_Value = 12
seven_day_min_Value = -12.22
thirty_day_max_Value = 23.37
thirty_day_min_Value = -23.03

#importing data
data = open("HistoricalQuotes.csv").read()
dataset = tablib.import_set(data)
header = dataset.headers
rows = dataset._data

#creating dataset
ds = SupervisedDataSet(5, 1)

#building network
network = buildNetwork(5, 10, 1, recurrent=True,hiddenclass=TanhLayer)

print("Network Built")

#retrieving data from excel
for row in rows[:2000]:
    close = float(row[1])
    volume = float(row[2])
    opens = float(row[3])
    high = float(row[4])
    low = float(row[5])
    nextClose = float(row[6])
    move_seven = float(row[8])
    move_thirty = float(row[9])
    #normalization of Data
    close_n_value = ((close - close_min_Value)/(close_max_Value-close_min_Value)-0.5)*2
    volume_n_value = ((volume - volume_min_Value)/(volume_max_Value-volume_min_Value)-0.5)*2
    opens_n_value = ((opens - opens_min_Value)/(opens_max_Value-opens_min_Value)-0.5)*2
    high_n_value = ((high - high_min_Value)/(high_max_Value-high_min_Value)-0.5)*2
    low_n_value = ((low - low_min_Value)/(low_max_Value-low_min_Value)-0.5)*2
    seven_n_value = ((move_seven - seven_day_min_Value)/(seven_day_max_Value-seven_day_min_Value)-0.5)*2
    thirty_n_value = ((move_thirty - thirty_day_min_Value)/(thirty_day_max_Value-thirty_day_min_Value)-0.5)*2
    nextClose_n_value = ((close - close_min_Value)/(close_max_Value-close_min_Value)-0.5)*2
    #adding samples to dataset
    ds.addSample((low_n_value, high_n_value, opens_n_value, close_n_value, volume_n_value), (nextClose_n_value,))
    #ds.addSample((low_n_value, high_n_value, opens_n_value, close_n_value, volume_n_value, seven_n_value, thirty_n_value), (nextClose_n_value,))
    
print("Dataset Built")

#building trainer
trainer1 = BackpropTrainer(network, ds)

#training data
trainer1.trainUntilConvergence( verbose = True,
                               validationProportion = 0.15,
                               maxEpochs = 100,
                               continueEpochs = 10 )

print("Data Trained")

normalizedAfter = 0
closeMoveUp = 0
predMoveUp = 0
totalCorrect = 0
total = 0
#using trainer to see how data forms
for row in rows[2001:2520]:
    date = row[0]
    close = float(row[1])
    volume = float(row[2])
    opens = float(row[3])
    high = float(row[4])
    low = float(row[5])
    nextClose = float(row[6])
    move_seven = float(row[8])
    move_thirty = float(row[9])
#normalizing the data
    close_n_value = ((close - close_min_Value)/(close_max_Value-close_min_Value)-0.5)*2
    volume_n_value = ((volume - volume_min_Value)/(volume_max_Value-volume_min_Value)-0.5)*2
    opens_n_value = ((opens - opens_min_Value)/(opens_max_Value-opens_min_Value)-0.5)*2
    high_n_value = ((high - high_min_Value)/(high_max_Value-high_min_Value)-0.5)*2
    low_n_value = ((low - low_min_Value)/(low_max_Value-low_min_Value)-0.5)*2
    seven_n_value = ((move_seven - seven_day_min_Value)/(seven_day_max_Value-seven_day_min_Value)-0.5)*2
    thirty_n_value = ((move_thirty - thirty_day_min_Value)/(thirty_day_max_Value-thirty_day_min_Value)-0.5)*2
    nextClose_n_value = ((close - close_min_Value)/(close_max_Value-close_min_Value)-0.5)*2
    normalizedPrediction = network.activate([low_n_value, high_n_value, opens_n_value, close_n_value, volume_n_value])
    #normalizedPrediction = network.activate([low_n_value, high_n_value, opens_n_value, close_n_value, volume_n_value, seven_n_value, thirty_n_value])
    predicted_close = (close_min_Value + ((close_max_Value-close_min_Value)*(.5+(normalizedPrediction/2))))
    print("Date: ",date)
    print("Predicted Close: ",predicted_close[0])
    print("Actual Close: ",nextClose)
#making ration of correct stock movement
    if (nextClose_n_value > normalizedPrediction[0]):
        closeMoveUp = 1
    if (normalizedPrediction>=normalizedAfter):
        predMoveUp = 1
    if(closeMoveUp == predMoveUp):
        totalCorrect = totalCorrect+1
    closeMoveUp = 0
    predMoveUp = 0
    total = total+1
    normalizedAfter = network.activate([low_n_value, high_n_value, opens_n_value, close_n_value,volume_n_value])
    print("//")
print(totalCorrect,total)

