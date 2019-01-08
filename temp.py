import os
import gc
import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd

DS= [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220,
                   223, 230]

pathDB = os.getcwd()
print pathDB
DB_name = 'data'

# Read files: signal (.csv )  annotations (.txt)
fRecords = list()
fAnnotations = list()

lst = os.listdir(pathDB + "/" + DB_name + "/csv")
lst.sort()
for filename in lst:
    if filename.endswith(".csv"):
        if int(filename[0:3]) in DS:
            fRecords.append(filename)
    elif filename.endswith(".txt"):
        if int(filename[0:3]) in DS:
            fAnnotations.append(filename)


winL=90
winR=90
do_preprocess=True
class_ID = [[] for i in range(len(DS))]
beat = [[] for i in range(len(DS))]  # record, beat, lead
R_poses = [np.array([]) for i in range(len(DS))]
Original_R_poses = [np.array([]) for i in range(len(DS))]
valid_R = [np.array([]) for i in range(len(DS))]

size_RR_max = 20
pathDB = os.getcwd()
print pathDB
DB_name = 'data'

filename = pathDB + "/" + DB_name + "/csv/" + fAnnotations[0]
print filename
# data = pd.read_csv(filename, sep=" ",names=['Time','Sample','#','Type','Sub','Chan','Num','Aux'])
data = pd.read_csv(filename, delimiter="\t")

for i in range(len(data)):
    print i
