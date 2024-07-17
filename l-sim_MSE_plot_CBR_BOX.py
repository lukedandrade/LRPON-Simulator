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

for param in parameters :
    for payload_size in CPRI_PKT:
        pd_dba_msestart = []
        pd_dba_mseend = []
        for seed in seeds:
            df_tmp = pd.read_csv("csv/PD_DBA-dist{}-3ONUs-1OLTs-CBR_PG-exp0-pkt{}-w{}-p{}-s{}-mse.csv".format(
                    20, payload_size, param['w'], param['p'], seed)
            )
            pd_dba_msestart.append(df_tmp['mse_start'])
            pd_dba_mseend.append(df_tmp['mse_end'])

        PD_DBA['{}-{}'.format(param['w'],param['p'])][payload_size] = [np.mean(pd_dba_mseend), np.std(pd_dba_mseend)]

pd_dba_df = pd.DataFrame(PD_DBA)

#creating figure
plt.figure()

title = "STD CBR (20km) - Predições Grant End"#figure title from argument
plt.title(title)
plt.xlabel("Payload (bytes)")
plt.ylabel("Standard Deviation")

number = 4
cmap = plt.get_cmap('gnuplot')
colors = [cmap(i) for i in np.linspace(0.25, 1, number)]

# #Line graph
# for j, param in enumerate(parameters):
#     array = np.array([ i for i in pd_dba_df['{}-{}'.format(param['w'],param['p'])].iloc[:] ])

#     #plt.errorbar(CPRI_PKT, array[:,0],array[:,1], color=colors[j],linestyle='None')
#     plt.plot(CPRI_PKT, array[:,0], '->',color=colors[j] ,label="PD-DBA w={} p={}".format(param['w'],param['p']))
    
#     for i, v in enumerate(array[:,1]):
#         plt.text(i, v, str(v)[:4], ha="center", va="center", fontsize="smaller", family="serif")

# plt.legend(loc='upper center', shadow=True)
# plt.show()


# Bar graph
br = []
for j, param in enumerate(parameters):
    array = np.array([ i for i in pd_dba_df['{}-{}'.format(param['w'],param['p'])].iloc[:] ])
    if len(br) != 0:
        br = [x+0.15 for x in br]
    else:
        br = np.arange(len(array[:,1]))
    plt.bar(br, array[:,1], color=colors[j], edgecolor="grey", width=0.15, label="PD-DBA w={} p={}".format(param['w'],param['p']))

plt.xticks([r+0.30 for r in np.arange(len(array[:,1]))], CPRI_PKT)

plt.legend(loc='upper center', shadow=True)
plt.show()
