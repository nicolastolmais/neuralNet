import tablib
from pybrain.structure import LinearLayer, SigmoidLayer
from pybrain.structure import FeedForwardNetwork
from pybrain.structure import FullConnection
from pybrain.supervised.trainers import BackpropTrainer
from pybrain import *
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet
from pybrain.structure import TanhLayer

#importing data
data = open("HistoricalQuotes.csv").read()
dataset = tablib.import_set(data)
header = dataset.headers
rows = dataset._data



ds = SupervisedDataSet(1, 1)
ds.addSample((0), (0,))
ds.addSample((0), (1,))
ds.addSample((1), (0,))
ds.addSample((1), (1,))
ds.addSample((1), (0,))

for inpt, target in ds:
    print (inpt, target)



net = buildNetwork(1, 3, 1, bias=True, hiddenclass=TanhLayer)

trainer = BackpropTrainer(net, ds)

for i in range(10):
    s = trainer.train()
    print (s)

a = net.activate([0])
print("should be 101.41 / it is :", a)
