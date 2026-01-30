import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, errno
import sys
from math import sqrt

#Settings
PRED_ALG = ['ols', 'ridge', 'lasso', 'mlp']
NUMBER_OF_OLTs = 1
NUMBER_OF_ONUs_vet = [5, 10, 15, 20]
DISTANCE = 20 #Distance in kilometers
TRAFFIC = "poisson"
DBA_ALG = "ipact_pred"

arquivo_aux_erros = open("arquivo_aux_erros.txt", "w")

IPACT_Pred_r2_medio = {}
IPACT_Pred_mse_medio = {}
IPACT_Pred_r2_std = {}
IPACT_Pred_mse_std = {}

#load % values which represents each exponent
loads = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]
#pkt arrival distribution exponents
exponents = [0.01, 0.05, 0.1, 0.20, 0.30, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]

seeds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
parameters = [{'w':5, 'p':1}, {'w':15, 'p':1}, {'w':25, 'p':1}, {'w':5, 'p':2}, {'w':15, 'p':2}, {'w':25, 'p':2}, {'w':5, 'p':3}, {'w':15, 'p':3},{'w':25, 'p':3}]
#parameters = [{'w':30, 'p':2}, {'w':20, 'p':2}, {'w':25, 'p':2}, {'w':15, 'p':2}]

#PKT_SIZES = ['768000', '1536000', '3072000', '3840000']
PKT_SIZES = ['768000']

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
                IPACT_Pred_r2_medio[NUMBER_OF_ONUs] = {}
                IPACT_Pred_mse_medio[NUMBER_OF_ONUs] = {}
                IPACT_Pred_r2_std[NUMBER_OF_ONUs] = {}
                IPACT_Pred_mse_std[NUMBER_OF_ONUs] = {}
            
            #R2 Score
            for size in PKT_SIZES:
                for exp in exponents:
                    for NUMBER_OF_ONUs in NUMBER_OF_ONUs_vet:
                        BASE_DIR = 'test_poissonPG_IPACT_pred_{}'.format(model)
                        ipact_mean = []
                        ipact_std = []
                        for seed in seeds:
                            df_tmp = pd.read_csv("{}/{}km/{}/csv/metrics/{}-dist{}-{}ONUs-{}OLTs-{}-exp{}-pkt{}-w{}-p{}-s{}-metrics.csv".format(BASE_DIR, DISTANCE, size, DBA_ALG, DISTANCE,
                                                                                                                NUMBER_OF_ONUs, NUMBER_OF_OLTs,
                                                                                                                TRAFFIC, exp, size, param['w'], param['p'], seed))
                            ipact_mean.append(df_tmp['r2_score'].mean())
                            ipact_std.append(df_tmp['r2_score'].std())
                        media = np.mean(ipact_mean)
                        t_test_results = calculateT_student(ipact_mean, 10)
                        std = t_test_results['t-test amp']
                        IPACT_Pred_r2_medio[NUMBER_OF_ONUs][exp] = media
                        IPACT_Pred_r2_std[NUMBER_OF_ONUs][exp] = std
            ipact_df_pr_r2medias = pd.DataFrame(IPACT_Pred_r2_medio)
            ipact_df_pr_r2stds = pd.DataFrame(IPACT_Pred_r2_std)

            #MSE
            for size in PKT_SIZES:
                for exp in exponents:
                    for NUMBER_OF_ONUs in NUMBER_OF_ONUs_vet:
                        BASE_DIR = 'test_poissonPG_IPACT_pred_{}'.format(model)
                        ipact_mean = []
                        ipact_std = []
                        for seed in seeds:
                            df_tmp = pd.read_csv("{}/{}km/{}/csv/metrics/{}-dist{}-{}ONUs-{}OLTs-{}-exp{}-pkt{}-w{}-p{}-s{}-metrics.csv".format(BASE_DIR, DISTANCE, size, DBA_ALG, DISTANCE,
                                                                                                    NUMBER_OF_ONUs, NUMBER_OF_OLTs,
                                                                                                    TRAFFIC, exp, size, param['w'], param['p'], seed))
                            ipact_mean.append(df_tmp['mse'].mean())
                            ipact_std.append(df_tmp['mse'].std())
                
                        media = np.mean(ipact_mean)
                        t_test_results = calculateT_student(ipact_mean, 10)
                        std = t_test_results['t-test amp']
                        IPACT_Pred_mse_medio[NUMBER_OF_ONUs][exp] = media
                        IPACT_Pred_mse_std[NUMBER_OF_ONUs][exp] = std
            ipact_df_pr_msemedias = pd.DataFrame(IPACT_Pred_mse_medio)
            ipact_df_pr_msestds = pd.DataFrame(IPACT_Pred_mse_std)

            filepath_pasta = "Graficos\Vision by n_onus"

            #Plot r2
            plt.clf()
            plt.figure(figsize=(14, 10))
            title = "All ONUs - {} OLT ({}km) - Trafego {}, {} - média de R2 Score em 5s de simulação ({}) w{}-p{}".format(NUMBER_OF_OLTs, DISTANCE, TRAFFIC, DBA_ALG, model, param['w'], param['p'])
            filename = "All_ONUs_{}OLT_{}km-{}-{}-r2score-{}_w{}-p{}".format(NUMBER_OF_OLTs, DISTANCE, TRAFFIC, DBA_ALG, model, param['w'], param['p'])
            plt.title(title)
            plt.xlabel("load (%)")
            plt.ylabel("R2 Score")
            plt.yscale('linear')
            cmap = plt.get_cmap('gnuplot')
            colors = ['c', 'r', 'g', 'b']
            i=0
            for NUMBER_OF_ONUs in NUMBER_OF_ONUs_vet:
                aux_df = pd.DataFrame(ipact_df_pr_r2medias[NUMBER_OF_ONUs])
                aux_label = "Ipact pred, {}, load(MB) = {:.3f}, n_ONUs = {}".format(model, float(PKT_SIZES[0])/(1024**2), NUMBER_OF_ONUs)
                media_delay = np.array(aux_df.iloc[:,0])
                aux_df = pd.DataFrame(ipact_df_pr_r2stds[NUMBER_OF_ONUs])
                std_delay = np.array(aux_df.iloc[:,0])
                
                plt.plot(loads, media_delay, '->', color=colors[i], label=aux_label)
                plt.errorbar(loads, media_delay, std_delay, color=colors[i], linestyle='None')
                i += 1

            filepath_completo = filepath_pasta+"\Metricas de IA\R2 Score\{}".format(filename)
            plt.legend(loc='best', shadow=True)
            plt.savefig(filepath_completo)
            plt.close()
            #MSE
            
            plt.clf()
            plt.figure(figsize=(14, 10))
            title = "All ONUs - {} OLT ({}km) - Trafego {}, {} - média de MSE em 5s de simulação ({}) w{}-p{}".format(NUMBER_OF_OLTs, DISTANCE, TRAFFIC, DBA_ALG, model,  param['w'], param['p'])
            filename = "All_ONUs_{}OLT_{}km-{}-{}-mse-{}_w{}-p{}".format(NUMBER_OF_OLTs, DISTANCE, TRAFFIC, DBA_ALG, model, param['w'], param['p'])
            plt.title(title)
            plt.xlabel("load (%)")
            plt.ylabel("Mean Squared Error")
            plt.yscale('log')
            cmap = plt.get_cmap('gnuplot')
            colors = ['c', 'r', 'g', 'b']

            i=0
            for NUMBER_OF_ONUs in NUMBER_OF_ONUs_vet:
                aux_df = pd.DataFrame(ipact_df_pr_msemedias[NUMBER_OF_ONUs])
                aux_label = "Ipact pred, {}, load(MB) = {:.3f}, n_ONUs = {}".format(model, float(PKT_SIZES[0])/(1024**2), NUMBER_OF_ONUs)
                media_pl = np.array(aux_df.iloc[:,0])
                aux_df = pd.DataFrame(ipact_df_pr_msestds[NUMBER_OF_ONUs])
                std_pl = np.array(aux_df.iloc[:,0])
                
                plt.plot(loads, media_pl, '->', color=colors[i], label=aux_label)
                plt.errorbar(loads, media_pl, std_pl, color=colors[i], linestyle='None')
                i += 1

            filepath_completo = filepath_pasta+"\Metricas de IA\MSE\{}".format(filename)
            plt.legend(loc='best', shadow=True)
            plt.savefig(filepath_completo)
            plt.close()


        except Exception as err:
            arquivo_aux_erros.write("Erro: {} \n".format(err))
arquivo_aux_erros.close()