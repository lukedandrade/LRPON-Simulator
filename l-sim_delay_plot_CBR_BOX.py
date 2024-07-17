import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, errno
import sys

#create img directory
try:
    os.makedirs('img')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

#settings
IPACT = {}
PD_DBA = {}
MPD_DBA = {}
loads = [768000, 1536000, 3072000]
CPRI_PKT = ['768000', '1536000', '3072000']

seeds = [20]
parameters = [{'w':10,'p':3}, {'w':20, 'p':8}, {'w':15, 'p':3}, {'w':5, 'p':3}]

for param in parameters:
    PD_DBA['{}-{}'.format(param['w'],param['p'])] = {}
    MPD_DBA['{}-{}'.format(param['w'],param['p'])] = {}

#read the ipact delay file for each simulated scenario with different seeds,
# and calculate the mean and std of each seed simulations
# ipact = []

# for payload_size in CPRI_PKT:
    
#     for seed in seeds:
#         IPACT[payload_size] = pd.read_csv("csv/delay/IPACT-3ONUs-1OLTs-CBR_PG-exp0-pkt{}-s{}-delay.csv".format(payload_size, seed))
#     ipact.append(list(IPACT[payload_size]['delay']))

# print(len(ipact))

#ipact_df = pd.DataFrame(ipact)

#read the pd_dba delay file for each simulated scenario with different seeds,
# and calculate the mean and std of each seed simulations
pd_dba = []
for param in parameters :
    for payload_size in CPRI_PKT:
        
        for seed in seeds:
            PD_DBA['{}-{}'.format(param['w'],param['p'])][payload_size] = pd.read_csv(
                "csv/delay/PD_DBA-dist{}-3ONUs-1OLTs-CBR_PG-exp0-pkt{}-w{}-p{}-s{}-delay.csv".format(
                    20, payload_size, param['w'], param['p'], seed)
            )
            pd_dba.append(list(PD_DBA['{}-{}'.format(param['w'],param['p'])][payload_size]['delay']))

#print(ipact_df)

fig, ax = plt.subplots()

title = "Boxplot CBR w=10, p=5 (20km) - 3 ONUS - OLS - IPACT"
ax.set_title(title)
ax.set_xlabel("Payload (bytes)")
ax.set_ylabel("Delay (ms)")

number = 4
cmap = plt.get_cmap('gnuplot')
colors = [cmap(i) for i in np.linspace(0.25, 1, number)]

ax.boxplot(pd_dba[0:3], labels=CPRI_PKT)

plt.show()