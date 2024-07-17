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
loads = [25,31,37,43,50,56,62,68,75,81,87,93]
exponents = [1160, 1450, 1740, 2030, 2320, 2610, 2900, 3190, 3480, 3770, 4060, 4350]
seeds = [20]
parameters = [{'w':10,'p':5}]

for param in parameters:
    PD_DBA['{}-{}'.format(param['w'],param['p'])] = {}
    MPD_DBA['{}-{}'.format(param['w'],param['p'])] = {}

#read the ipact delay file for each simulated scenario with different seeds,
# and calculate the mean and std of each seed simulations
for exp in exponents:
    ipact = []
    for seed in seeds:
        df_tmp = pd.read_csv("csv/delay/Ipact-3ONUs-1OLTs-poisson-exp{}-s{}-delay.csv".format(exp, seed))
        ipact.append(df_tmp['delay'].mean())
    IPACT[exp] = [np.mean(ipact),np.std(ipact)]

#read the pd_dba delay file for each simulated scenario with different seeds,
# and calculate the mean and std of each seed simulations
for param in parameters :
    for exp in exponents:
        pd_dba = []
        for seed in seeds:
            df_tmp = pd.read_csv("csv/delay/PD_DBA-3ONUs-1OLTs-poisson-exp{}-s{}-delay.csv".format(exp, seed))
            pd_dba.append(df_tmp['delay'].mean())
        PD_DBA['{}-{}'.format(param['w'],param['p'])][exp] = [np.mean(pd_dba),np.std(pd_dba)]

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

#creating figure
plt.figure()

title = "Test com janela tamanho 10, 5 predicoes (20km) - 3 ONUS - OLS"#figure title from argument
plt.title(title)
plt.xlabel("load (%)")
plt.ylabel("delay (ms)")


plt.errorbar(loads, ipact_df.iloc[0].values,ipact_df.iloc[1].values,color="k", linestyle='None')
plt.plot(loads, ipact_df.iloc[0], 'o-', color="k",label="IPACT")

number = 4
cmap = plt.get_cmap('gnuplot')
colors = [cmap(i) for i in np.linspace(0.25, 1, number)]

for j, param in enumerate(parameters):
    array = np.array([ i for i in pd_dba_df['{}-{}'.format(param['w'],param['p'])].iloc[:] ])

    plt.errorbar(loads, array[:,0],array[:,1], color=colors[j],linestyle='None')
    plt.plot(loads, array[:,0], '->',color=colors[j] ,label="PD-DBA w={} p={}".format(param['w'],param['p']))

    # array = np.array([ i for i in mpd_dba_df['{}-{}'.format(param['w'],param['p'])].iloc[:] ])
    # plt.errorbar(loads, array[:,0],array[:,1], color=colors[j],linestyle='None')
    # plt.plot(loads, array[:,0], '->',color=colors[j] ,label="MPD-DBA w={} p={}".format(param['w'],param['p']))


plt.legend(loc='upper center', shadow=True)
plt.savefig("img/Delay-"+title)