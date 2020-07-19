import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import glob
plt.style.use('seaborn')

def createDataFrame(file_name):
    data = []
    headers = None
    with open(file_name) as f:
        line = f.readline()
        headers = line.split(",")

        line = f.readline()
        while line:
            data.append(line.split(","))
            line = f.readline()

    df = pd.DataFrame(data=[sub for sub in data], columns=headers)
    return df


def getTotalTravelTime(df):
    time = 0
    for i,j in zip(df['ActualTravelTime'], df['BestTravelTime']):
       time += float(i)-float(j)
    return time/df.shape[0]

def getTravelTimeShift(df, per):
    totalCount = 0
    outCount = 0
    for i,j in zip(df['ActualTravelTime'], df['BestTravelTime']):
       if (float(i)/float(j)) > per:
           outCount += 1
       totalCount += 1
    return outCount/totalCount

def sort(df, column_name):
    for index, item in enumerate(df[column_name]):
        df[column_name][index] = float(index)

    df.sort_values([column_name])

dir= "C:/Users/pgunarathna/IdeaProjects/Temporary_update_smarts/download/Journal/Lookup/"
freq = [1,2,3,4,5,6,7,8,9]
#freq = [0,8,9,95,99]
frac = [0, 1, 2, 10, 20]
noLA = []
CLLA = []
gain = []

dataFrame = pd.DataFrame(columns=["Travel time", "Algorithm", "Freq"])

for f,fr in zip(freq, freq):
    #file_names = ["VD_noLA_"+f+".txt", "VD_CLLA_"+f+".txt"]
    #for file_name in file_names:

    #path  = dir+"VD_noLA_"+"DIJKSTRA_LPF_true_"+str(f)+"_28"+"_"+"*"
    #file  = glob.glob(path)
    #
    #df = createDataFrame(file[0])
    #print(df.shape)
    #print(getTotalTravelTime(df))
    #noLA.append(getTotalTravelTime(df))
    #print(getTravelTimeShift(df, 6))
    #
    #dataFrame = dataFrame.append({"Travel time": getTotalTravelTime(df), "Algorithm": "noLA", "Freq": (f)}, ignore_index=True)

    path  = dir+"VD_CLLA__"+str(f)+"_DIJKSTRA_LPF_"+"300_28"+"_6000"+"*"
    print(path)
    file  = glob.glob(path)

    df = createDataFrame(file[0])
    print(df.shape)
    print(getTotalTravelTime(df))
    CLLA.append(getTotalTravelTime(df))
    print(getTravelTimeShift(df, 6))


    #x = math.log(1/(1-fr),10)
    dataFrame = dataFrame.append({"Travel time": getTotalTravelTime(df), "Algorithm": "CLLA", "Freq": (fr)}, ignore_index=True)


#for index in range(len(CLLA)):
#    gain.append([(noLA[index]-CLLA[index])/max(noLA[index],CLLA[index]), freq[index]])

#gain = np.array(gain)
#gain_pd = pd.DataFrame({"Travel time gain":gain[:,0], "Freq":gain[:,1]})

#print(gain_pd)
print(dataFrame)
fmri = pd.DataFrame({'timepoint':[0,5,10,15,20,30], 'val':[0,0,0,0,0,0]})

#sns.lineplot(y="Travel time gain", x="Freq", data=gain_pd)
#sns.lineplot(x="timepoint", y="val", data=fmri)

ax = plt.gca()
#ax.fill_between(gain_pd["Freq"], 0, gain_pd["Travel time gain"], where=gain_pd["Travel time gain"]>0, facecolor='green', interpolate=True)
#ax.fill_between(gain_pd["Freq"], 0, gain_pd["Travel time gain"], where=gain_pd["Travel time gain"]<0, facecolor='red', interpolate=True)
sns.lineplot(y="Travel time", x="Freq", hue="Algorithm", data=dataFrame)
#plt.ylim(350, 450)
plt.xlabel("Moving Average Window for PDG (log(x))", fontsize='x-large')
plt.ylabel("Travel time", fontsize='x-large')

plt.show()