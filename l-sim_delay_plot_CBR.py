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
for payload_size in CPRI_PKT:
    ipact = []
    for seed in seeds:
        df_tmp = pd.read_csv("csv/delay/IPACT-dist{}-3ONUs-1OLTs-CBR_PG-exp0-pkt{}-s{}-delay.csv".format(20, payload_size, seed))
        ipact.append(df_tmp['delay'].mean()*1000)
    IPACT[payload_size] = [np.mean(ipact),np.std(ipact)]

print(IPACT)

#read the pd_dba delay file for each simulated scenario with different seeds,
# and calculate the mean and std of each seed simulations
for param in parameters :
    for payload_size in CPRI_PKT:
        pd_dba = []
        for seed in seeds:
            df_tmp = pd.read_csv("csv/delay/PD_DBA-dist{}-3ONUs-1OLTs-CBR_PG-exp0-pkt{}-w{}-p{}-s{}-delay.csv".format(
                    20, payload_size, param['w'], param['p'], seed))
            pd_dba.append(df_tmp['delay'].mean()*1000)
        PD_DBA['{}-{}'.format(param['w'],param['p'])][payload_size] = [np.mean(pd_dba),np.std(pd_dba)]

#read the mpd_dba delay file for each simulated scenario with different seeds,
# and calculate the mean and std of each seed simulations
#for param in parameters :
#    for exp in exponents:
#        pd_dba = []
#        for seed in seeds:
#            df_tmp = pd.read_csv("csv/delay/MPD_DBA-3ONUs-1OLTs-poisson-exp{}-s{}-delay.csv".format(exp, seed))
#            pd_dba.append(df_tmp['delay'].mean())
#        MPD_DBA['{}-{}'.format(param['w'],param['p'])][exp] = [np.mean(pd_dba),np.std(pd_dba)]

#create a data frame
ipact_df = pd.DataFrame(IPACT)
pd_dba_df = pd.DataFrame(PD_DBA)
#mpd_dba_df = pd.DataFrame(MPD_DBA)

print(ipact_df)
print(pd_dba_df)
#creating figure
plt.figure()

title = "Teste CBR (20km) - 3 ONUS - OLS - EXTRA"#figure title from argument
plt.title(title)
plt.xlabel("Payload (bytes)")
plt.ylabel("Delay (ms)")


plt.errorbar(CPRI_PKT, ipact_df.iloc[0].values, ipact_df.iloc[1].values,color="k", linestyle='None')
plt.plot(CPRI_PKT, ipact_df.iloc[0], 'o-', color="k",label="IPACT")
#

number = 4
cmap = plt.get_cmap('gnuplot')
colors = [cmap(i) for i in np.linspace(0.25, 1, number)]

for j, param in enumerate(parameters):
    array = np.array([ i for i in pd_dba_df['{}-{}'.format(param['w'],param['p'])].iloc[:] ])

    plt.errorbar(CPRI_PKT, array[:,0],array[:,1], color=colors[j],linestyle='None')
    plt.plot(CPRI_PKT, array[:,0], '->',color=colors[j] ,label="PD-DBA w={} p={}".format(param['w'],param['p']))

    # array = np.array([ i for i in mpd_dba_df['{}-{}'.format(param['w'],param['p'])].iloc[:] ])
    # plt.errorbar(loads, array[:,0],array[:,1], color=colors[j],linestyle='None')
    # plt.plot(loads, array[:,0], '->',color=colors[j] ,label="MPD-DBA w={} p={}".format(param['w'],param['p']))


plt.legend(loc='upper center', shadow=True)
plt.show()
#plt.savefig("img/Delay-"+title)