import tablib
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import FullConnection
from pybrain.supervised.trainers import BackpropTrainer
from pybrain import *
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.utilities           import percentError
from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer


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

#importing data
data = open("HistoricalQuotes.csv").read()
dataset = tablib.import_set(data)
header = dataset.headers
rows = dataset._data



ds = SupervisedDataSet(4, 1)

n = buildNetwork(4, 20, 20, 1, bias=True, hiddenclass=TanhLayer)



print("Network Built")

#inputs setting up datasets
for row in rows[:2000]:
    close = float(row[1])
    volume = float(row[2])
    opens = float(row[3])
    high = float(row[4])
    low = float(row[5])
    nextClose = float(row[6])
    #ds.addSample((low,high,opens,close),(nextClose),)
    close_n_value = ((close - close_min_Value)/(close_max_Value-close_min_Value)-0.5)*2
    volume_n_value = ((volume - volume_min_Value)/(volume_max_Value-volume_min_Value)-0.5)*2
    opens_n_value = ((opens - opens_min_Value)/(opens_max_Value-opens_min_Value)-0.5)*2
    high_n_value = ((high - high_min_Value)/(high_max_Value-high_min_Value)-0.5)*2
    low_n_value = ((low - low_min_Value)/(low_max_Value-low_min_Value)-0.5)*2
    nextClose_n_value = ((close - close_min_Value)/(close_max_Value-close_min_Value)-0.5)*2
    ds.addSample((low_n_value, high_n_value, opens_n_value, close_n_value), (nextClose_n_value,))

print("Dataset Built")
#print(ds)
#for inpt, target in ds:
#    print (inpt, target)

#normalize data


trainer = BackpropTrainer(n, ds)


#for i in range(100):
#    s = trainer.train()
#trainer.trainUntilConvergence()

trainer.trainUntilConvergence( verbose = True,
                               validationProportion = 0.15,
                               maxEpochs = 100,
                               continueEpochs = 10 )


print("Data Trained")
b = 0
c = 0
d = 0
e = 0
f = 0
for row in rows[2001:2520]:
    close = float(row[1])
    volume = float(row[2])
    opens = float(row[3])
    high = float(row[4])
    low = float(row[5])
    nextClose = float(row[6])
    a = n.activate([low, high, opens, close])
    #print("should be:", nextClose, "/ it is :", a)
    if (nextClose > close):
        print("Up")
        c = 1
    else:
        print("Down")
    if (a>=b):
        print("Up")
        d = 1
    else:
        print("Down")
    if(c == d):
        e = e+1
    print("//")
    c = 0
    d = 0
    f = f+1
    b = n.activate([low, high, opens, close])       

print(e,f)
#p = n.activateOnDataset( ds )
#print(p)
#r = trainer.trainEpochs(epochs = 15)
#print(r)

#t = trainer.trainUntilConvergence()
#print(t)


#result = n.activateOnDataset( ds )
#print (p)
'''
#prints column 2
for row in rows:
    print (row[1])

#prints all
for row in rows:
    close = row[1]
    volume = row[2]
    opens = row[3]
    high = row[4]
    low = row[5]
    
    print (row)
'''
