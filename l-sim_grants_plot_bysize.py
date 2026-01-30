import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, errno
import sys
from math import sqrt

#Settings
#PRED_ALG = ['ols', 'ridge', 'lasso', 'mlp']
PRED_ALG = ['mlp']
NUMBER_OF_OLTs = 1
NUMBER_OF_ONUs_vet = [5, 10, 15, 20]
DISTANCE = 20 #Distance in kilometers
TRAFFIC = "poisson"
DBA_ALG = "ipact_pred"

arquivo_aux_erros = open("arquivo_aux_erros.txt", "w")

IPACT_Pred_grant_medio = {}
ipact_basic_grant_medio = {}
IPACT_Pred_std = {}
ipact_basic_std = {}
IPACT_Pred_grant_percent_mean = {}
IPACT_Pred_grant_percent_std = {}

#load % values which represents each exponent
loads = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]
#pkt arrival distribution exponents
exponents = [0.01, 0.05, 0.1, 0.20, 0.30, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]

seeds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
parameters = [{'w':5, 'p':1}, {'w':15, 'p':1}, {'w':25, 'p':1}, {'w':5, 'p':2}, {'w':15, 'p':2}, {'w':25, 'p':2}, {'w':5, 'p':3}, {'w':15, 'p':3},{'w':25, 'p':3}]
#parameters = [{'w':30, 'p':2}, {'w':20, 'p':2}, {'w':25, 'p':2}, {'w':15, 'p':2}]

#PKT_SIZES = ['768000', '1536000', '3072000', '3840000']
PKT_SIZES = ['768000', '1536000', '3072000', '3840000']

def calculateT_student(vetor_medias, f_degrees):
    table_const = 2.262
    media_final = np.mean(vetor_medias)
    std_final = np.std(vetor_medias)
    interval_amp = table_const*std_final/(sqrt(f_degrees))
    
    structured_resp = {'t-test amp': interval_amp, 't-test interval': [media_final-interval_amp, media_final+interval_amp]}

    return structured_resp

for NUMBER_OF_ONUs in NUMBER_OF_ONUs_vet:
    for param in parameters:
        try:
            for size in PKT_SIZES:
                IPACT_Pred_grant_medio[size] = {}
                ipact_basic_grant_medio[size] = {}
                IPACT_Pred_std[size] = {}
                ipact_basic_std[size] = {}
                IPACT_Pred_grant_percent_mean[size] = {}
                IPACT_Pred_grant_percent_std[size] = {}
            
            #Ipact_pred
            for model in PRED_ALG:
                for exp in exponents:
                    for size in PKT_SIZES:
                        BASE_DIR = 'test_poissonPG_IPACT_pred_{}'.format(model)
                        ipact_mean = []
                        ipact_grant_percentages = []
                        for seed in seeds:
                            df_tmp = pd.read_csv("{}/{}km/{}/csv/grant_usage/{}-dist{}-{}ONUs-{}OLTs-{}-exp{}-pkt{}-w{}-p{}-s{}-grant_usage.csv".format(BASE_DIR, DISTANCE, size, DBA_ALG, DISTANCE,
                                                                                                    NUMBER_OF_ONUs, NUMBER_OF_OLTs,
                                                                                                    TRAFFIC, exp, size, param['w'], param['p'], seed))
                            ipact_mean.append(df_tmp['data sent'].mean())
                            n_pred_grants = df_tmp[df_tmp['predicted'] == True].shape[0]
                
                            n_grants = df_tmp.shape[0]
                            percentage = n_pred_grants/n_grants
                            ipact_grant_percentages.append(percentage*100)
            
                        media = np.mean(ipact_mean)
                        t_test_results = calculateT_student(ipact_mean, 10)
                        std = t_test_results['t-test amp']
                        IPACT_Pred_grant_medio[size][exp] = media
                        IPACT_Pred_std[size][exp] = std

                        percentage_media = np.mean(ipact_grant_percentages)
                        t_test_results = calculateT_student(percentage_media, 10)
                        percentage_std = t_test_results['t-test amp']
                        IPACT_Pred_grant_percent_mean[size][exp] = percentage_media
                        IPACT_Pred_grant_percent_std[size][exp] = percentage_std

            ipact_df_pr_medias = pd.DataFrame(IPACT_Pred_grant_medio)
            ipact_df_pr_stds = pd.DataFrame(IPACT_Pred_std)
            ipact_df_percentages_media = pd.DataFrame(IPACT_Pred_grant_percent_mean)
            ipact_df_percentages_std = pd.DataFrame(IPACT_Pred_grant_percent_std)
            
            #Ipact_basico
            for exp in exponents:
                for payload_size in PKT_SIZES:
                    ipact_mean = []
                    ipact_std = []
                    for seed in seeds:
                        df_tmp = pd.read_csv("test_poissonPG_IPACT/{}km/{}/csv/grant_usage/ipact-dist{}-{}ONUs-{}OLTs-{}-exp{}-pkt{}-s{}-grant_usage.csv".format(DISTANCE, payload_size, DISTANCE,
                                                                                                                                    NUMBER_OF_ONUs, NUMBER_OF_OLTs,
                                                                                                                                    TRAFFIC, exp, payload_size, seed))
                        ipact_mean.append(df_tmp['data sent'].mean())

                    media = np.mean(ipact_mean)
                    #std = np.mean(std_pkt_loss)
                    t_test_results = calculateT_student(ipact_mean, 10)
                    std = t_test_results['t-test amp']
                    ipact_basic_grant_medio[payload_size][exp] = media
                    ipact_basic_std[payload_size][exp] = std
            ipact_b_df_medias = pd.DataFrame(ipact_basic_grant_medio)
            ipact_b_df_stds = pd.DataFrame(ipact_basic_std)

            filepath_pasta = "Graficos\Vision by pkt_size"

            #Plot Grant Size
            plt.clf()
            plt.figure(figsize=(14, 10))
            title = "{} ONUs - {} OLT ({}km) - Trafego {}, {} - média de Largura de banda transmitida  5s de simulação ({}) w{}-p{}".format(NUMBER_OF_ONUs, NUMBER_OF_OLTs, DISTANCE, TRAFFIC, DBA_ALG, model, param['w'], param['p'])
            filename = "{}ONUs_{}OLT_{}km-{}-{}-grant_sizes-{}_w{}-p{}".format(NUMBER_OF_ONUs, NUMBER_OF_OLTs, DISTANCE, TRAFFIC, DBA_ALG, model, param['w'], param['p'])
            plt.title(title)
            plt.xlabel("load (%)")
            plt.ylabel("Bandwith (bytes)")
            plt.yscale('linear')
            cmap = plt.get_cmap('gnuplot')
            colors = ['c', 'r', 'g', 'b']
            i=0
            for size in PKT_SIZES:
                aux_df = pd.DataFrame(ipact_df_pr_medias[size])
                aux_label = "Ipact pred, {}, load(MB) = {:.3f}".format(model, float(size)/(1024**2))
                media_delay = np.array(aux_df.iloc[:,0])
                aux_df = pd.DataFrame(ipact_df_pr_stds[size])
                std_delay = np.array(aux_df.iloc[:,0])
                
                plt.plot(loads, media_delay, '->', color=colors[i], label=aux_label)
                plt.errorbar(loads, media_delay, std_delay, color=colors[i], linestyle='None')
                i += 1
            i = 0
            for size in PKT_SIZES:

                aux_df_b_ipact = pd.DataFrame(ipact_b_df_medias[size])
                aux_label = "Ipact, load(MB) = {:.3f}".format(float(size)/(1024**2))
                media_delay = np.array(aux_df_b_ipact.iloc[:,0])
                plt.plot(loads, media_delay, '-o', color=colors[i], linestyle='-.', label=aux_label)
                i += 1

            filepath_completo = filepath_pasta+"\Grant Size\Largura de Banda\{}".format(filename)
            plt.legend(loc='best', shadow=True)
            plt.savefig(filepath_completo)
            plt.close()
            
            #Grants Preditos           
            plt.clf()
            plt.figure(figsize=(14, 10))
            title = "{} ONUs - {} OLT ({}km) - Trafego {}, {} - Grants preditos em 5s de simulação ({}) w{}-p{}".format(NUMBER_OF_ONUs, NUMBER_OF_OLTs, DISTANCE, TRAFFIC, DBA_ALG, model,  param['w'], param['p'])
            filename = "{}ONUs_{}OLT_{}km-{}-{}-n_grants-{}_w{}-p{}".format(NUMBER_OF_ONUs, NUMBER_OF_OLTs, DISTANCE, TRAFFIC, DBA_ALG, model, param['w'], param['p'])
            plt.title(title)
            plt.xlabel("load (%)")
            plt.ylabel("Grants preditos (%)")
            plt.yscale('linear')
            cmap = plt.get_cmap('gnuplot')
            colors = ['c', 'r', 'g', 'b']

            i=0
            for size in PKT_SIZES:
                aux_df = pd.DataFrame(ipact_df_percentages_media[size])
                aux_label = "Ipact pred, {}, load(MB) = {:.3f}".format(model, float(size)/(1024**2))
                media_pl = np.array(aux_df.iloc[:,0])
                aux_df = pd.DataFrame(ipact_df_percentages_std[size])
                std_pl = np.array(aux_df.iloc[:,0])
                
                plt.plot(loads, media_pl, '->', color=colors[i], label=aux_label)
                plt.errorbar(loads, media_pl, std_pl, color=colors[i], linestyle='None')
                i += 1

            filepath_completo = filepath_pasta+"\Grant Size\Grants preditos\{}".format(filename)
            plt.legend(loc='best', shadow=True)
            plt.savefig(filepath_completo)
            plt.close()

        except Exception as err:
            arquivo_aux_erros.write("Erro: {} \n".format(err))
arquivo_aux_erros.close()