import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, errno
import sys
from math import sqrt

#Settings
PRED_ALG = ['ols', 'ridge', 'lasso', 'mlp']
NUMBER_OF_OLTs = 1
NUMBER_OF_ONUs_vet = ['5', '10', '15', '20']
DISTANCE = 20 #Distance in kilometers
TRAFFIC = "poisson"
DBA_ALG = "ipact_pred"

arquivo_aux_erros = open("arquivo_aux_erros.txt", "w")

IPACT_Pred_delay_medio = {}
ipact_basic_delay_medio = {}
IPACT_Pred_std = {}
ipact_basic_std = {}
IPACT_Pred_pl_medio = {}
ipact_basic_pl_medio = {}
IPACT_Pred_pl_std = {}
ipact_basic_pl_std = {}
delay_proportion_medio = {}
delay_proportion_std = {}

#load % values which represents each exponent
loads = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]
#pkt arrival distribution exponents
exponents = [0.01, 0.05, 0.1, 0.20, 0.30, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]

seeds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
parameters = [{'w':5, 'p':1}, {'w':15, 'p':1}, {'w':25, 'p':1}, {'w':5, 'p':2}, {'w':15, 'p':2}, {'w':25, 'p':2}, {'w':5, 'p':3}, {'w':15, 'p':3},{'w':25, 'p':3}]
#parameters = [{'w':30, 'p':2}, {'w':20, 'p':2}, {'w':25, 'p':2}, {'w':15, 'p':2}]

#PKT_SIZES = ['768000', '1536000', '3072000', '3840000']
PKT_SIZES = ['3840000']

def calculateT_student(vetor_medias, f_degrees):
    table_const = 2.262
    media_final = np.mean(vetor_medias)
    std_final = np.std(vetor_medias)
    interval_amp = table_const*std_final/(sqrt(f_degrees))
    
    structured_resp = {'t-test amp': interval_amp, 't-test interval': [media_final-interval_amp, media_final+interval_amp]}

    return structured_resp

for model in PRED_ALG:
    for param in parameters:
        try:
            for NUMBER_OF_ONUs in NUMBER_OF_ONUs_vet:
                IPACT_Pred_delay_medio[NUMBER_OF_ONUs] = {}
                ipact_basic_delay_medio[NUMBER_OF_ONUs] = {}
                IPACT_Pred_std[NUMBER_OF_ONUs] = {}
                ipact_basic_std[NUMBER_OF_ONUs] = {}
                IPACT_Pred_pl_medio[NUMBER_OF_ONUs] = {}
                ipact_basic_pl_medio[NUMBER_OF_ONUs] = {}
                IPACT_Pred_pl_std[NUMBER_OF_ONUs] = {}
                ipact_basic_pl_std[NUMBER_OF_ONUs] = {}
                delay_proportion_medio[NUMBER_OF_ONUs] = {}
                delay_proportion_std[NUMBER_OF_ONUs] = {}
            
            #Ipact_pred_delay
            for size in PKT_SIZES:
                for exp in exponents:            
                    for NUMBER_OF_ONUs in NUMBER_OF_ONUs_vet:
                        BASE_DIR = 'test_poissonPG_IPACT_pred_{}'.format(model)
                        ipact_mean_delay = []
                        ipact_std_delay = []
                        mean_pkt_loss = []
                        std_pkt_loss = []
                        delay_mean_proportion = []
                        for seed in seeds:
                            df_tmp = pd.read_csv("{}/{}km/{}/csv/delay/{}-dist{}-{}ONUs-{}OLTs-{}-exp{}-pkt{}-w{}-p{}-s{}-delay.csv".format(BASE_DIR, DISTANCE, size, DBA_ALG, DISTANCE,
                                                                                                    NUMBER_OF_ONUs, NUMBER_OF_OLTs,
                                                                                                    TRAFFIC, exp, size, param['w'], param['p'], seed))
                            df_tmp_ipact_basic = pd.read_csv("test_poissonPG_IPACT/{}km/{}/csv/delay/ipact-dist{}-{}ONUs-{}OLTs-{}-exp{}-pkt{}-s{}-delay.csv".format(DISTANCE, size, DISTANCE,
                                                                                                                                        NUMBER_OF_ONUs, NUMBER_OF_OLTs,
                                                                                                                                        TRAFFIC, exp, size, seed))
                            ipact_mean_delay.append(df_tmp['Mean']*1000)
                            ipact_std_delay.append(df_tmp['STD']*1000)
                            mean_pkt_loss.append(df_tmp['Pkt_loss_mean'])
                            std_pkt_loss.append(df_tmp['Pkt_loss_std'])
                            pred_divided_by_basic = df_tmp['Mean']/df_tmp_ipact_basic['Mean']
                            delay_mean_proportion.append(pred_divided_by_basic)
            
                        #media e std de delays    
                        media = np.mean(ipact_mean_delay)
                        t_test_results = calculateT_student(ipact_mean_delay, 10)
                        std = t_test_results['t-test amp']
                        IPACT_Pred_delay_medio[NUMBER_OF_ONUs][exp] = media
                        IPACT_Pred_std[NUMBER_OF_ONUs][exp] = std

                        #media e std de packet loss
                        media = np.mean(mean_pkt_loss)
                        t_test_results = calculateT_student(mean_pkt_loss, 10)
                        std = t_test_results['t-test amp']
                        IPACT_Pred_pl_medio[NUMBER_OF_ONUs][exp] = media
                        IPACT_Pred_pl_std[NUMBER_OF_ONUs][exp] = std

                        #media e std da proporcao
                        media = np.mean(delay_mean_proportion)
                        t_test_results = calculateT_student(delay_mean_proportion, 10)
                        std = t_test_results['t-test amp']
                        delay_proportion_medio[NUMBER_OF_ONUs][exp] = media
                        delay_proportion_std[NUMBER_OF_ONUs][exp] = std

            ipact_df_pr_medias = pd.DataFrame(IPACT_Pred_delay_medio)
            ipact_df_pr_stds = pd.DataFrame(IPACT_Pred_std)
            ipact_df_pr_pl_medias = pd.DataFrame(IPACT_Pred_pl_medio)
            ipact_df_pr_pl_stds = pd.DataFrame(IPACT_Pred_pl_std)
            proportion_df_medias = pd.DataFrame(delay_proportion_medio)
            proportion_df_stds = pd.DataFrame(delay_proportion_std)
            
            #Ipact_basico
            for size in PKT_SIZES:
                for exp in exponents:
                    for NUMBER_OF_ONUs in NUMBER_OF_ONUs_vet:
                        ipact_mean_delay = []
                        ipact_std_delay = []
                        mean_pkt_loss = []
                        std_pkt_loss = []
                        for seed in seeds:
                            df_tmp = pd.read_csv("test_poissonPG_IPACT/{}km/{}/csv/delay/ipact-dist{}-{}ONUs-{}OLTs-{}-exp{}-pkt{}-s{}-delay.csv".format(DISTANCE, size, DISTANCE,
                                                                                                                                        NUMBER_OF_ONUs, NUMBER_OF_OLTs,
                                                                                                                                        TRAFFIC, exp, size, seed))
                            ipact_mean_delay.append(df_tmp['Mean']*1000)
                            ipact_std_delay.append(df_tmp['STD']*1000)
                            mean_pkt_loss.append(df_tmp['Pkt_loss_mean'])
                            std_pkt_loss.append(df_tmp['Pkt_loss_std'])

                        #media e std de delays    
                        media = np.mean(ipact_mean_delay)
                        t_test_results = calculateT_student(ipact_mean_delay, 10)
                        std = t_test_results['t-test amp']
        
                        ipact_basic_delay_medio[NUMBER_OF_ONUs][exp] = media
                        ipact_basic_std[NUMBER_OF_ONUs][exp] = std

                        #media e std de packet loss
                        media = np.mean(mean_pkt_loss)
                        t_test_results = calculateT_student(mean_pkt_loss, 10)
                        std = t_test_results['t-test amp']
                        ipact_basic_pl_medio[NUMBER_OF_ONUs][exp] = media
                        ipact_basic_pl_std[NUMBER_OF_ONUs][exp] = std

            ipact_b_df_medias = pd.DataFrame(ipact_basic_delay_medio)
            ipact_b_df_stds = pd.DataFrame(ipact_basic_std)
            ipact_b_df_pl_medias = pd.DataFrame(ipact_basic_pl_medio)
            ipact_b_df_pl_std = pd.DataFrame(ipact_basic_pl_std)

            filepath_pasta = "Graficos\Vision by n_onus"

            #Plot Delay
            plt.clf()
            plt.figure(figsize=(14, 10))
            title = "All ONUs - {} OLT ({}km) - Trafego {}, {} - média de Delay em 5s de simulação ({}) w{}-p{}".format(NUMBER_OF_OLTs, DISTANCE, TRAFFIC, DBA_ALG, model, param['w'], param['p'])
            filename = "All_ONUs_{}OLT_{}km-{}-{}-delay-{}_w{}-p{}".format(NUMBER_OF_OLTs, DISTANCE, TRAFFIC, DBA_ALG, model, param['w'], param['p'])
            plt.title(title)
            plt.xlabel("load (%)")
            plt.ylabel("Delay (ms)")
            plt.yscale('linear')
            cmap = plt.get_cmap('gnuplot')
            colors = ['c', 'r', 'g', 'b']
            i = 0
            for NUMBER_OF_ONUs in NUMBER_OF_ONUs_vet:
                aux_df = pd.DataFrame(ipact_df_pr_medias[NUMBER_OF_ONUs])
                aux_label = "Ipact pred, {}, load(MB) = {:.3f}, n_ONUs = {}".format(model, float(PKT_SIZES[0])/(1024**2), NUMBER_OF_ONUs)
                media_delay = np.array(aux_df.iloc[:,0])
                aux_df = pd.DataFrame(ipact_df_pr_stds[NUMBER_OF_ONUs])
                std_delay = np.array(aux_df.iloc[:,0])
                
                plt.plot(loads, media_delay, '->', color=colors[i], label=aux_label)
                plt.errorbar(loads, media_delay, std_delay, color=colors[i], linestyle='None')
                i += 1
            i = 0
            for NUMBER_OF_ONUS in NUMBER_OF_ONUs_vet:
                aux_df_b_ipact = pd.DataFrame(ipact_b_df_medias[NUMBER_OF_ONUS])
                aux_label = "Ipact, load(MB) = {:.3f}, n_ONUs = {}".format(float(PKT_SIZES[0])/(1024**2), NUMBER_OF_ONUs)
                media_delay = np.array(aux_df_b_ipact.iloc[:,0])
                plt.plot(loads, media_delay, '-o', color=colors[i], linestyle='-.', label=aux_label)
                i += 1
                
            filepath_completo = filepath_pasta+"\Delay\{}".format(filename)
            plt.legend(loc='best', shadow=True)
            plt.savefig(filepath_completo)
            plt.close()
            
            #Plot Pkt_loss           
            plt.clf()
            plt.figure(figsize=(14, 10))
            title = "All ONUs - {} OLT ({}km) - Trafego {}, {} - Packet Loss em 5s de simulação ({}) w{}-p{}".format(NUMBER_OF_OLTs, DISTANCE, TRAFFIC, DBA_ALG, model,  param['w'], param['p'])
            filename = "All_ONUs_{}OLT_{}km-{}-{}-pkt_loss-{}_w{}-p{}".format(NUMBER_OF_OLTs, DISTANCE, TRAFFIC, DBA_ALG, model, param['w'], param['p'])
            plt.title(title)
            plt.xlabel("load (%)")
            plt.ylabel("Grants preditos (%)")
            plt.yscale('linear')
            cmap = plt.get_cmap('gnuplot')
            colors = ['c', 'r', 'g', 'b']

            i = 0
            for NUMBER_OF_ONUs in NUMBER_OF_ONUs_vet:
                aux_df = pd.DataFrame(ipact_df_pr_pl_medias[NUMBER_OF_ONUs])
                aux_label = "Ipact pred, {}, load(MB) = {:.3f}, n_ONUs = {}".format(model, float(PKT_SIZES[0])/(1024**2), NUMBER_OF_ONUs)
                media_pl = np.array(aux_df.iloc[:,0])
                aux_df = pd.DataFrame(ipact_df_pr_pl_stds[NUMBER_OF_ONUs])
                std_pl = np.array(aux_df.iloc[:,0])
                
                plt.plot(loads, media_pl, '->', color=colors[i], label=aux_label)
                plt.errorbar(loads, media_pl, std_pl, color=colors[i], linestyle='None')
                i += 1
            i = 0
            for NUMBER_OF_ONUS in NUMBER_OF_ONUs_vet:
                aux_df_b_ipact = pd.DataFrame(ipact_b_df_pl_medias[NUMBER_OF_ONUS])
                aux_label = "Ipact, load(MB) = {:.3f}, n_ONUs = {}".format(float(PKT_SIZES[0])/(1024**2), NUMBER_OF_ONUs)
                media_delay = np.array(aux_df_b_ipact.iloc[:,0])
                plt.plot(loads, media_delay, '-o', color=colors[i], linestyle='-.', label=aux_label)
                i += 1

            filepath_completo = filepath_pasta+"\Packet Loss\{}".format(filename)
            plt.legend(loc='best', shadow=True)
            plt.savefig(filepath_completo)
            plt.close()

            #Plot Delay Proportion           
            plt.clf()
            plt.figure(figsize=(14, 10))
            title = "All ONUs - {} OLT ({}km) - Trafego {}, {} - Proporção de Delays em 5s de simulação ({}) w{}-p{}".format(NUMBER_OF_OLTs, DISTANCE, TRAFFIC, DBA_ALG, model,  param['w'], param['p'])
            filename = "All_ONUs_{}OLT_{}km-{}-{}-delay_proportion-{}_w{}-p{}".format(NUMBER_OF_OLTs, DISTANCE, TRAFFIC, DBA_ALG, model, param['w'], param['p'])
            plt.title(title)
            plt.xlabel("load (%)")
            plt.ylabel("Ipact_pred/Ipact")
            plt.yscale('log')
            cmap = plt.get_cmap('gnuplot')
            colors = ['c', 'r', 'g', 'b']

            i=0
            for NUMBER_OF_ONUs in NUMBER_OF_ONUs_vet:
                aux_df = pd.DataFrame(proportion_df_medias[NUMBER_OF_ONUs])
                aux_label = "Ipact pred, {}, load(MB) = {:.3f}, n_ONUs = {}".format(model, float(PKT_SIZES[0])/(1024**2), NUMBER_OF_ONUs)
                media_pl = np.array(aux_df.iloc[:,0])
                aux_df = pd.DataFrame(proportion_df_stds[NUMBER_OF_ONUs])
                std_pl = np.array(aux_df.iloc[:,0])
                
                plt.plot(loads, media_pl, '->', color=colors[i], label=aux_label)
                plt.errorbar(loads, media_pl, std_pl, color=colors[i], linestyle='None')
                i += 1

            filepath_completo = filepath_pasta+"\Delay\{}".format(filename)
            plt.legend(loc='best', shadow=True)
            plt.savefig(filepath_completo)
            plt.close()

        except Exception as err:
            arquivo_aux_erros.write("Erro: {} \n".format(err))
arquivo_aux_erros.close()