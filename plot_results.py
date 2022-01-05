"""
Plots the performance metrics
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patheffects as path_effects
import seaborn as sns
sns.set_theme(style='ticks')

def plot_human_v_machine():
    #raw performance values in order of 
    #hasSz acc, hasAns F1, noAns F1, overall F1, PQF F1, ELO F1
    clinicalBERT_performance = np.array([
        [0.8200,	66.1647,	91.6185,	80.8431,	81.1750,	80.5111],
        [0.8633,	70.7917,	88.7283,	81.1352,	81.4694,	80.8009],
        [0.8433,	69.4877,	92.4855,	82.7498,	83.7001,	81.7995],
        [0.8367,	69.6999,	91.0405,	82.0063,	83.7489,	80.2637],
        [0.8267,	67.8827,	90.4624,	80.9037,	81.2318,	80.5756]]
        )
    RoBERTa_performance = np.array([
        [0.6600,	74.6710,	91.6185,	84.4441,	83.6059,	85.2823],
        [0.7467,	72.5247,	92.7746,	84.2021,	85.5400,	82.8642],
        [0.8833,	72.4486,	92.7746,	84.1699,	83.8382,	84.5016],
        [0.8467,	73.3264,	91.3295,	83.7082,	84.4509,	82.9655],
        [0.7167,	73.6709,	92.7746,	84.6873,	86.0032,	83.3715]
        ])
    BERT_base_performance = np.array([
        [0.8000,	70.6516,	91.9075,	82.9092,	82.8976,	82.9207],
        [0.8500,	69.5960,	92.4855,	82.7956,	83.5605,	82.0308],
        [0.8367,	67.8618,	93.3526,	82.5615,	83.0517,	82.0713],
        [0.8267,	68.9120,	94.2197,	83.5061,	82.8852,	84.1269],
        [0.8100,	71.1995,	93.0636,	83.8078,	84.3301,	83.2855]
        ])
    human_performance = np.array([
        [0.9231,	0.8769,	0.9385,	0.9825,	0.9677,	0.6538,	0.9180,	0.8852,	0.9608,	0.9216,	0.9412,	0.8730,	0.8231],
        [69.5165,	56.8423,	72.0428,	63.4372,	41.8679,	60.2642,	61.8279,	51.9606,	38.4650,	78.3739,	65.0888,	53.2903,	65.5622],
        [92.4051,	94.9367,	96.2025,	98.4375,	97.3684,	84.6154,	93.5484,	96.7742,	95.0820,	93.4426,	93.4426,	87.5000,	98.6111],
        [83.4257,	79.9920,	86.7245,	83.0865,	75.8843,	72.4398,	77.9482,	74.7347,	71.6730,	87.2123,	81.7194,	72.2372,	83.8662],
        [83.0619,	85.1307,	92.0601,	92.3770,	96.1346,	93.1184,	77.2761,	75.8354,	93.9845,	72.7987,	84.3207,	93.8375,	88.6467],
        [79.3260,	76.7933,	91.6009,	83.4684,	77.8871,	70.7284,	70.2086,	60.4206,	92.6171,	70.5850,	90.3787,	85.6883,	72.9701],
        ]).transpose()
        
    
    fig, axs = plt.subplots(1, 1, dpi=600, figsize=(12,5))
    sns.despine()
    
    #plot results from a question-centric view
    performance = [clinicalBERT_performance[:,0], RoBERTa_performance[:,0], BERT_base_performance[:,0], human_performance[:,0],
                 clinicalBERT_performance[:,-2]/100, RoBERTa_performance[:,-2]/100, BERT_base_performance[:,-2]/100, human_performance[:,-2]/100,
                 clinicalBERT_performance[:,-1]/100, RoBERTa_performance[:,-1]/100, BERT_base_performance[:,-1]/100, human_performance[:,-1]/100]
    
    
    positions = [1, 1.5, 2, 2.5,
                 4, 4.5, 5, 5.5,
                 7, 7.5, 8, 8.5]
    
    axs.boxplot(performance, sym="", positions=positions)
    
    #hasSz
    axs.plot(np.ones(5)*positions[0], clinicalBERT_performance[:,0], '.', color='#1b9e77', alpha=0.75)
    axs.plot(np.ones(5)*positions[1], RoBERTa_performance[:,0], '.', color='#d95f02', alpha=0.75)
    axs.plot(np.ones(5)*positions[2], BERT_base_performance[:,0], '.', color='#7570b3', alpha=0.75)
    axs.plot(np.ones(13)*positions[3] + (np.random.rand(13) - 0.5)*0.15, human_performance[:,0], '.', color='gray', alpha=0.75)
    #PQF
    axs.plot(np.ones(5)*positions[4], clinicalBERT_performance[:,-2]/100, '.', color='#1b9e77', alpha=0.75)
    axs.plot(np.ones(5)*positions[5], RoBERTa_performance[:,-2]/100, '.', color='#d95f02', alpha=0.75)
    axs.plot(np.ones(5)*positions[6], BERT_base_performance[:,-2]/100, '.', color='#7570b3', alpha=0.75)
    axs.plot(np.ones(13)*positions[7] + (np.random.rand(13) - 0.5)*0.15, human_performance[:,-2]/100, '.', color='gray', alpha=0.75)
    #ELO
    axs.plot(np.ones(5)*positions[8], clinicalBERT_performance[:,-1]/100, '.', color='#1b9e77', alpha=0.75)
    axs.plot(np.ones(5)*positions[9], RoBERTa_performance[:,-1]/100, '.', color='#d95f02', alpha=0.75)
    axs.plot(np.ones(5)*positions[10], BERT_base_performance[:,-1]/100, '.', color='#7570b3', alpha=0.75)
    axs.plot(np.ones(13)*positions[11] + (np.random.rand(13) - 0.5)*0.15, human_performance[:,-1]/100, '.', color='gray', alpha=0.75)
    
    axs.set_xticks(ticks=[1.75, 4.75, 7.75])
    axs.set_xticklabels(labels=['Classification (Q1)\nAccuracy', 'Seizure Frequency (Q2)\nF$_1$ score', 'Most Recent Seizure (Q3)\nF$_1$ score'])
    
    #https://matplotlib.org/stable/gallery/text_labels_and_annotations/custom_legends.html
    legend_elements = [Line2D([0], [0], color='k', marker='o', label='Bio_ClinicalBERT$_{\mathrm{FT}}$', markerfacecolor='#1b9e77'),
                       Line2D([0], [0], color='k', marker='o', label='RoBERTa$_{\mathrm{FT}}$', markerfacecolor='#d95f02'),
                       Line2D([0], [0], color='k', marker='o', label='BERT$_{\mathrm{FT}}$', markerfacecolor='#7570b3'),
                       Line2D([0], [0], color='k', marker='o', label='Human', markerfacecolor='grey')]
    
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.04, 0.5), frameon=False)
    axs.set_ylabel('Accuracy or F$_1$ Score')
    axs.set_ylim([0.6, 1])
    axs.set_title('Human vs. Machine Performance on\nExtracting Clinical Information')
    axs.grid(True, alpha=0.25)
    fig.savefig('../Figures/Figure_3.pdf', bbox_inches='tight')
    fig.savefig('../Figures/Figure_3.png', bbox_inches='tight')
    
    
    #plot results from a task-centric view
    fig, axs = plt.subplots(1, 2, dpi=600, figsize=(12,5))
    sns.despine()
    #hasSz
    hasSz_accs = [clinicalBERT_performance[:,0], RoBERTa_performance[:,0],
                  BERT_base_performance[:,0], human_performance[:,0]]
    labels = ['Bio_ClinicalBERT$_{\mathrm{FT}}$', 'RoBERTa$_{\mathrm{FT}}$', 'BERT$_{\mathrm{FT}}$', 'Human']
    axs[0].boxplot(hasSz_accs, sym="")
    axs[0].plot(np.ones(5), clinicalBERT_performance[:,0], '.', color='#1b9e77', alpha=0.75)
    axs[0].plot(np.ones(5)*2, RoBERTa_performance[:,0], '.', color='#d95f02', alpha=0.75)
    axs[0].plot(np.ones(5)*3, BERT_base_performance[:,0], '.', color='#7570b3', alpha=0.75)
    axs[0].plot(np.ones(13)*4 + (np.random.rand(13) - 0.5)*0.15, human_performance[:,0], '.', color='gray', alpha=0.75)
    axs[0].set_xticks(range(1, 5))
    axs[0].set_xticklabels(labels)
    axs[0].set_ylabel('Classification Accuracy')
    axs[0].set_ylim([0.35, 1.0])
    axs[0].set_title('Human vs. Machine Performance on\nClassifying Patients')
    axs[0].annotate('A)', xy=(0., 1.01), xycoords='axes fraction', fontweight='bold')
    #F1 Overall
    extract_f1 = [clinicalBERT_performance[:,3]/100, RoBERTa_performance[:,3]/100,
                  BERT_base_performance[:,3]/100, human_performance[:,3]/100]
    labels = ['Bio_ClinicalBERT$_{\mathrm{FT}}$', 'RoBERTa$_{\mathrm{FT}}$', 'BERT$_{\mathrm{FT}}$', 'Human']
    axs[1].boxplot(extract_f1, sym="")
    axs[1].plot(np.ones(5), clinicalBERT_performance[:,3]/100, '.', color='#1b9e77', alpha=0.75)
    axs[1].plot(np.ones(5)*2, RoBERTa_performance[:,3]/100, '.', color='#d95f02', alpha=0.75)
    axs[1].plot(np.ones(5)*3, BERT_base_performance[:,3]/100, '.', color='#7570b3', alpha=0.75)
    axs[1].plot(np.ones(13)*4 + (np.random.rand(13) - 0.5)*0.15, human_performance[:,3]/100, '.', color='gray', alpha=0.75)
    axs[1].set_xticks(range(1, 5))
    axs[1].set_xticklabels(labels)
    axs[1].set_ylabel('F$_1$ Score')
    axs[1].set_ylim([.35, 1.00])
    axs[1].set_title('Human vs. Machine Performance on\nExtracting Text from Clinic Notes')
    axs[1].annotate('B)', xy=(0., 1.01), xycoords='axes fraction', fontweight='bold')
    fig.savefig('../Figures/HvM_Task_View.pdf', bbox_inches='tight')
    fig.savefig('../Figures/HvM_Task_View.png', bbox_inches='tight')
    
    
    
    
    #plot the F1 scores, split between hasAns, noAns, Overall
    fig, axs = plt.subplots(1, 1, dpi=600, figsize=(12,5))  
    sns.despine()
    szFreq_f1 = [clinicalBERT_performance[:,1]/100, RoBERTa_performance[:,1]/100, BERT_base_performance[:,1]/100, human_performance[:,1]/100,
                 clinicalBERT_performance[:,2]/100, RoBERTa_performance[:,2]/100, BERT_base_performance[:,2]/100, human_performance[:,2]/100,
                 clinicalBERT_performance[:,3]/100, RoBERTa_performance[:,3]/100, BERT_base_performance[:,3]/100, human_performance[:,3]/100]
    
    positions = [1, 1.5, 2, 2.5,
                 4, 4.5, 5, 5.5,
                 7, 7.5, 8, 8.5]
    
    axs.boxplot(szFreq_f1, sym="", positions=positions)
    #hasAns
    axs.plot(np.ones(5)*positions[0], clinicalBERT_performance[:,1]/100, '.', color='#1b9e77', alpha=0.75)
    axs.plot(np.ones(5)*positions[1], RoBERTa_performance[:,1]/100, '.', color='#d95f02', alpha=0.75)
    axs.plot(np.ones(5)*positions[2], BERT_base_performance[:,1]/100, '.', color='#7570b3', alpha=0.75)
    axs.plot(np.ones(13)*positions[3] + (np.random.rand(13) - 0.5)*0.15, human_performance[:,1]/100, '.', color='gray', alpha=0.75)
    #noAns
    axs.plot(np.ones(5)*positions[4], clinicalBERT_performance[:,2]/100, '.', color='#1b9e77', alpha=0.75)
    axs.plot(np.ones(5)*positions[5], RoBERTa_performance[:,2]/100, '.', color='#d95f02', alpha=0.75)
    axs.plot(np.ones(5)*positions[6], BERT_base_performance[:,2]/100, '.', color='#7570b3', alpha=0.75)
    axs.plot(np.ones(13)*positions[7] + (np.random.rand(13) - 0.5)*0.15, human_performance[:,2]/100, '.', color='gray', alpha=0.75)
    #Overall
    axs.plot(np.ones(5)*positions[8], clinicalBERT_performance[:,3]/100, '.', color='#1b9e77', alpha=0.75)
    axs.plot(np.ones(5)*positions[9], RoBERTa_performance[:,3]/100, '.', color='#d95f02', alpha=0.75)
    axs.plot(np.ones(5)*positions[10], BERT_base_performance[:,3]/100, '.', color='#7570b3', alpha=0.75)
    axs.plot(np.ones(13)*positions[11] + (np.random.rand(13) - 0.5)*0.15, human_performance[:,3]/100, '.', color='gray', alpha=0.75)
    
    axs.set_xticks(ticks=[1.75, 4.75, 7.75])
    axs.set_xticklabels(labels=['Answer Exists', 'No Answer Exists', 'Overall'])
    axs.grid(True, alpha=0.25)
    
    #https://matplotlib.org/stable/gallery/text_labels_and_annotations/custom_legends.html
    legend_elements = [Line2D([0], [0], color='k', marker='o', label='Bio_ClinicalBERT$_{\mathrm{FT}}$', markerfacecolor='#1b9e77'),
                       Line2D([0], [0], color='k', marker='o', label='RoBERTa$_{\mathrm{FT}}$', markerfacecolor='#d95f02'),
                       Line2D([0], [0], color='k', marker='o', label='BERT$_{\mathrm{FT}}$', markerfacecolor='#7570b3'),
                       Line2D([0], [0], color='k', marker='o', label='Human', markerfacecolor='grey')]
    
    axs.legend(handles=legend_elements, loc='lower right', frameon=False)
    axs.set_ylabel('F$_1$ Score')
    axs.set_ylim([.35, 1])
    axs.set_title('Human vs. Machine Performance on Extracting\nSeizure Frequencies and Occurrences')
    fig.savefig('../Figures/Supplemental_Figure_2.pdf', bbox_inches='tight')
    fig.savefig('../Figures/Supplemental_Figure_2.png', bbox_inches='tight')

def plot_ablations():
    full_model = np.array([
        [0.8200,	74.6710,	91.6185,	84.4441,	83.6059,	85.2823],
        [0.8633,	72.5247,	92.7746,	84.2021,	85.5400,	82.8642],
        [0.8433,	72.4486,	92.7746,	84.1699,	83.8382,	84.5016],
        [0.8367,	73.3264,	91.3295,	83.7082,	84.4509,	82.9655],
        [0.8267,	73.6709,	92.7746,	84.6873,	86.0032,	83.3715],
        ])
    
    no_MLM = np.array([
        [0.8233,	73.5215,	91.9075,	84.1241,	84.9977,	83.2504],
        [0.8033,	71.1620,	92.1965,	83.2919,	85.3899,	81.1940],
        [0.7967,	72.3573,	91.9075,	83.6312,	84.4229,	82.8396],
        [0.8167,	75.8482,	89.8844,	83.9424,	85.7557,	82.1291],
        [0.8167,	72.0035,	91.6185,	83.3148,	85.1303,	81.4993],
        ])
    
    no_SQB3 = np.array([
        [0.8100,	71.5113,	92.4855,	83.6065,	84.5382,	82.6747],
        [0.8533,	69.6558,	91.0405,	81.9876,	83.4633,	80.5120],
        [0.8567,	72.6761,	93.3526,	84.5996,	85.0099,	84.1892],
        [0.8333,	72.9043,	92.1965,	84.0295,	84.4447,	83.6142],
        [0.8600,	75.0386,	87.8613,	82.4330,	83.8128,	81.0533],
        ])
            
    no_Anno = np.array([
        [0.2100,	33.4519,	48.8439,	42.3280,	45.9732,	38.6828],
        [0.2500,	35.8057,	38.1503,	37.1578,	35.9321,	38.3834],
        [0.2967,	31.7724,	44.5087,	39.1170,	50.6846,	27.5493],
        [0.2867,	37.9556,	51.4451,	45.7345,	56.0042,	35.4649],
        [0.2267,	30.5739,	54.6243,	44.4430,	57.0590,	31.8269],
        ])


    #plot the question-centric view
    fig, axs = plt.subplots(1, 1, figsize=(12,5), dpi=600)
    sns.despine()
    ablation_performance = [full_model[:,0], no_MLM[:,0], no_SQB3[:,0], no_Anno[:,0],
                 full_model[:,-2]/100, no_MLM[:,-2]/100, no_SQB3[:,-2]/100, no_Anno[:,-2]/100,
                 full_model[:,-1]/100, no_MLM[:,-1]/100, no_SQB3[:,-1]/100, no_Anno[:,-1]/100]
    
    
    positions = [1, 1.5, 2, 2.5,
                 4, 4.5, 5, 5.5,
                 7, 7.5, 8, 8.5]
    
    axs.boxplot(ablation_performance, sym="", positions=positions)
    #hasSz
    axs.plot(np.ones(5)*positions[0], full_model[:,0], '.', color='#1b9e77', alpha=0.75)
    axs.plot(np.ones(5)*positions[1], no_MLM[:,0], '.', color='#d95f02', alpha=0.75)
    axs.plot(np.ones(5)*positions[2], no_SQB3[:,0], '.', color='#7570b3', alpha=0.75)
    axs.plot(np.ones(5)*positions[3], no_Anno[:,0], '.', color='gray', alpha=0.75)
    #PQF
    axs.plot(np.ones(5)*positions[4], full_model[:,-2]/100, '.', color='#1b9e77', alpha=0.75)
    axs.plot(np.ones(5)*positions[5], no_MLM[:,-2]/100, '.', color='#d95f02', alpha=0.75)
    axs.plot(np.ones(5)*positions[6], no_SQB3[:,-2]/100, '.', color='#7570b3', alpha=0.75)
    axs.plot(np.ones(5)*positions[7], no_Anno[:,-2]/100, '.', color='gray', alpha=0.75)
    #ELO
    axs.plot(np.ones(5)*positions[8], full_model[:,-1]/100, '.', color='#1b9e77', alpha=0.75)
    axs.plot(np.ones(5)*positions[9], no_MLM[:,-1]/100, '.', color='#d95f02', alpha=0.75)
    axs.plot(np.ones(5)*positions[10], no_SQB3[:,-1]/100, '.', color='#7570b3', alpha=0.75)
    axs.plot(np.ones(5)*positions[11], no_Anno[:,-1]/100, '.', color='gray', alpha=0.75)
    
    axs.set_xticks(ticks=[1.75, 4.75, 7.75])
    axs.set_xticklabels(labels=['Classification (Q1)\nAccuracy', 'Seizure Frequency (Q2)\nF$_1$ score', 'Most Recent Seizure (Q3)\nF$_1$ score'])
    axs.set_ylim([0.2, 1])
    axs.grid(True, alpha=0.25)
    
    #https://matplotlib.org/stable/gallery/text_labels_and_annotations/custom_legends.html
    legend_elements = [Line2D([0], [0], color='k', marker='o', label='Full Model', markerfacecolor='#1b9e77'),
                       Line2D([0], [0], color='k', marker='o', label='-MLM', markerfacecolor='#d95f02'),
                       Line2D([0], [0], color='k', marker='o', label='-BoolQ3L/SQuADv2', markerfacecolor='#7570b3'),
                       Line2D([0], [0], color='k', marker='o', label='-Annotations', markerfacecolor='grey')]
    
    axs.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.04, 0.5), frameon=False)
    axs.set_ylabel('Accuracy or F$_1$ Score')
    axs.set_title('Results of Ablation Studies')
    fig.savefig('../Figures/Figure_4.pdf', bbox_inches='tight')
    fig.savefig('../Figures/Figure_4.png', bbox_inches='tight')
    
    
    
    
    
    #plot the task-centric view
    fig, axs = plt.subplots(1, 2, figsize=(12,5), dpi=600)
    sns.despine()
    #hasSz
    ablation_hasSz = [full_model[:,0], no_MLM[:,0],
                  no_SQB3[:,0], no_Anno[:,0]]
    labels = ['Full Model', '-MLM', '-BoolQ3L', '-Annotations']
    axs[0].boxplot(ablation_hasSz, sym="")
    axs[0].plot(np.ones(5), full_model[:,0], '.', color='#1b9e77', alpha=0.75)
    axs[0].plot(np.ones(5)*2, no_MLM[:,0], '.', color='#d95f02', alpha=0.75)
    axs[0].plot(np.ones(5)*3, no_SQB3[:,0], '.', color='#7570b3', alpha=0.75)
    axs[0].plot(np.ones(5)*4, no_Anno[:,0], '.', color='gray', alpha=0.75)
    axs[0].set_xticks(ticks=range(1, 5))
    axs[0].set_xticklabels(labels=labels)
    axs[0].set_ylabel('Classification Accuracy')
    axs[0].set_title('Classification Accuracy Following Ablation')
    axs[0].set_ylim([0.2, 1.0])
    axs[0].annotate('A)', xy=(0., 1.01), xycoords='axes fraction', fontweight='bold')
    
    #overall F1
    ablation_overall = [full_model[:,3]/100, no_MLM[:,3]/100,
                  no_SQB3[:,3]/100, no_Anno[:,3]/100]
    labels = ['Full Model', '-MLM', '-BoolQ3L', '-Annotations']
    axs[1].boxplot(ablation_overall, sym="")
    axs[1].plot(np.ones(5), full_model[:,3]/100, '.', color='#1b9e77', alpha=0.75)
    axs[1].plot(np.ones(5)*2, no_MLM[:,3]/100, '.', color='#d95f02', alpha=0.75)
    axs[1].plot(np.ones(5)*3, no_SQB3[:,3]/100, '.', color='#7570b3', alpha=0.75)
    axs[1].plot(np.ones(5)*4, no_Anno[:,3]/100, '.', color='gray', alpha=0.75)
    axs[1].set_xticks(ticks=range(1, 5))
    axs[1].set_xticklabels(labels=labels)
    axs[1].set_ylabel('F$_1$ Score')
    axs[1].set_title('Text Extraction F$_1$ Score Following Ablation')
    axs[1].set_ylim([0.2, 1])
    axs[1].annotate('B)', xy=(0., 1.01), xycoords='axes fraction', fontweight='bold')
    fig.savefig('../Figures/ablation_task_view.pdf', bbox_inches='tight')
    fig.savefig('../Figures/ablation_task_view.png', bbox_inches='tight')
    
    
    
    #plot the F1 scores, split between hasAns, noAns, Overall
    fig, axs = plt.subplots(1, 1, figsize=(12,5), dpi=600)
    sns.despine()
    ablation_f1 = [full_model[:,1]/100, no_MLM[:,1]/100, no_SQB3[:,1]/100, no_Anno[:,1]/100,
                 full_model[:,2]/100, no_MLM[:,2]/100, no_SQB3[:,2]/100, no_Anno[:,2]/100,
                 full_model[:,3]/100, no_MLM[:,3]/100, no_SQB3[:,3]/100, no_Anno[:,3]/100]
    
    
    positions = [1, 1.5, 2, 2.5,
                 4, 4.5, 5, 5.5,
                 7, 7.5, 8, 8.5]
    
    axs.boxplot(ablation_f1, sym="", positions=positions)
    #hasAns
    axs.plot(np.ones(5)*positions[0], full_model[:,1]/100, '.', color='#1b9e77', alpha=0.75)
    axs.plot(np.ones(5)*positions[1], no_MLM[:,1]/100, '.', color='#d95f02', alpha=0.75)
    axs.plot(np.ones(5)*positions[2], no_SQB3[:,1]/100, '.', color='#7570b3', alpha=0.75)
    axs.plot(np.ones(5)*positions[3], no_Anno[:,1]/100, '.', color='gray', alpha=0.75)
    #noAns
    axs.plot(np.ones(5)*positions[4], full_model[:,2]/100, '.', color='#1b9e77', alpha=0.75)
    axs.plot(np.ones(5)*positions[5], no_MLM[:,2]/100, '.', color='#d95f02', alpha=0.75)
    axs.plot(np.ones(5)*positions[6], no_SQB3[:,2]/100, '.', color='#7570b3', alpha=0.75)
    axs.plot(np.ones(5)*positions[7], no_Anno[:,2]/100, '.', color='gray', alpha=0.75)
    #overall F1
    axs.plot(np.ones(5)*positions[8], full_model[:,3]/100, '.', color='#1b9e77', alpha=0.75)
    axs.plot(np.ones(5)*positions[9], no_MLM[:,3]/100, '.', color='#d95f02', alpha=0.75)
    axs.plot(np.ones(5)*positions[10], no_SQB3[:,3]/100, '.', color='#7570b3', alpha=0.75)
    axs.plot(np.ones(5)*positions[11], no_Anno[:,3]/100, '.', color='gray', alpha=0.75)
    
    axs.set_xticks(ticks=[1.75, 4.75, 7.75])
    axs.set_xticklabels(labels=['Answer Exists', 'No Answer Exists', 'Overall'])
    axs.set_ylim([.20, 1.00])
    axs.grid(True, alpha=0.25)
    
    #https://matplotlib.org/stable/gallery/text_labels_and_annotations/custom_legends.html
    legend_elements = [Line2D([0], [0], color='k', marker='o', label='Full Model', markerfacecolor='#1b9e77'),
                       Line2D([0], [0], color='k', marker='o', label='-MLM', markerfacecolor='#d95f02'),
                       Line2D([0], [0], color='k', marker='o', label='-SQuADv2', markerfacecolor='#7570b3'),
                       Line2D([0], [0], color='k', marker='o', label='-Annotations', markerfacecolor='grey')]
    
    axs.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.04, 0.5), frameon=False)
    axs.set_ylabel('F$_1$ Score')
    axs.set_title('F$_1$ Score for Extracting Seizure Frequencies\nand Occurrences Following Ablation')
    fig.savefig('../Figures/Supplemental_Figure_3.pdf', bbox_inches='tight')
    fig.savefig('../Figures/Supplemental_Figure_3.png', bbox_inches='tight')


def plot_training_reductions():
    hasSz = np.array([
        [0.8200,	0.8133,	0.8467,	0.8100,	0.8500,	0.7867,	0.7567,	0.7700,	0.7100,	0.6967,	0.2100],
        [0.8633,	0.8333,	0.8267,	0.8567,	0.8200,	0.8233,	0.7900,	0.7467,	0.7300,	0.6900,	0.2500],
        [0.8433,	0.8300,	0.8267,	0.8067,	0.7800,	0.7500,	0.7700,	0.7800,	0.7233,	0.7200,	0.2967],
        [0.8367,	0.8300,	0.8333,	0.7867,	0.8100,	0.7800,	0.7367,	0.7633,	0.7274,	0.6900,	0.2867],
        [0.8267,	0.8400,	0.7833,	0.8167,	0.8033,	0.7500,	0.7400,	0.7267,	0.7267,	0.6700,	0.2267],
        ])
    
    hasAns = np.array([
        [74.6710,	74.0693,	75.6856,	75.8249,	73.7320,	72.3286,	72.0134,	67.0476,	69.1662,	63.4263,	33.4519],
        [72.5247,	75.1534,	68.9275,	71.8366,	69.4055,	72.7037,	71.1831,	71.8380,	65.9726,	59.1252,	35.8057],
        [72.4486,	71.2876,	71.1612,	70.6924,	73.4932,	72.7765,	70.5924,	62.8810,	63.1322,	53.2028,	31.7724],
        [73.3264,	72.9802,	75.9456,	72.7794,	73.4070,	73.5523,	66.9424,	66.5869,	68.0261,	54.5335,	37.9556],
        [73.6709,	72.0830,	72.7322,	74.6153,	72.6455,	72.5765,	72.5914,	66.5440,	68.1386,	55.8477,	30.5739],
        ])/100
    
    noAns = np.array([
        [91.6185,	91.3295,	91.6185,	90.1734,	91.3295,	91.6185,	90.4624,	92.4855,	92.7746,	85.8382,	48.8439],
        [92.7746,	91.6185,	92.7746,	92.1965,	92.7746,	91.9075,	92.7746,	90.4624,	93.3526,	91.6185,	38.1503],
        [92.7746,	92.4855,	91.9075,	92.7746,	91.6185,	93.3526,	91.6185,	94.2197,	94.2197,	90.4624,	44.5087],
        [91.3295,	91.9075,	90.4624,	91.6185,	90.1734,	92.4855,	93.6416,	93.0636,	91.6185,	93.9306,	51.4451],
        [92.7746,	92.1965,	92.1965,	91.3295,	92.1965,	93.6416,	91.0405,	94.5087,	93.3526,	91.3295,	54.6243],
        ])/100
    
    eval_f1 = np.array([
        [84.4441,	84.0227,	84.8736,	84.0992,	83.8799,	83.4524,	82.6523,	81.7168,	82.7804,	76.3504,	42.3280],
        [84.2021,	84.6483,	82.6793,	83.5775,	82.8817,	83.7779,	83.6342,	82.5781,	81.7617,	77.8630,	37.1578],
        [84.1699,	83.5117,	83.1249,	83.4264,	83.9455,	84.6420,	82.7174,	80.9529,	81.0593,	74.6892,	39.1170],
        [83.7082,	83.8949,	84.3170,	83.6433,	83.0756,	84.4705,	82.3389,	81.8551,	81.6311,	77.2525,	45.7345],
        [84.6873,	83.6818,	83.9566,	84.2538,	83.9199,	84.7241,	83.2304,	82.6703,	82.6787,	76.3089,	44.4430],
        ])/100
    
    pqf_f1 = np.array([
        [83.6059,	84.0234,	84.2006,	84.0652,	84.1977,	83.4946,	82.7369,	80.8754,	82.2344,	79.1790,	45.9732],
        [85.5400,	85.5375,	82.9063,	84.5197,	83.2494,	85.3227,	84.0008,	82.8681,	81.1422,	79.1508,	35.9321],
        [83.8382,	83.7656,	83.6721,	83.4417,	85.0842,	85.1101,	84.4540,	78.9690,	80.6713,	74.3280,	50.6846],
        [84.4509,	84.0617,	84.9974,	84.3149,	83.5500,	86.5050,	83.3242,	81.8025,	82.6647,	75.6996,	56.0042],
        [86.0032,	84.1016,	83.9643,	85.9557,	84.6269,	86.3067,	84.6648,	83.2422,	84.1623,	76.9470,	57.0590],
        ])/100
        
    elo_f1 = np.array([
        [85.2823,	84.0220,	85.5465,	84.1332,	83.5621,	83.4103,	82.5677,	82.5583,	83.3264,	73.5219,	38.6828],
        [82.8642,	83.7591,	82.4523,	82.6353,	82.5139,	82.2331,	83.2676,	82.2881,	82.3813,	76.5753,	38.3834],
        [84.5016,	83.2579,	82.5778,	83.4111,	82.8068,	84.1739,	80.9808,	82.9369,	81.4473,	75.0503,	27.5493],
        [82.9655,	83.7282,	83.6366,	82.9716,	82.6013,	82.4360,	81.3536,	81.9078,	80.5974,	78.8055,	35.4649],
        [83.3715,	83.2620,	83.9489,	82.5520,	83.2130,	83.1414,	81.7959,	82.0984,	81.1951,	75.6707,	31.8269],
        ])/100
    
    domain = np.arange(1.0, -0.1, -0.1)
    
    #question-centric view
    #95 confidence interval https://en.wikipedia.org/wiki/Confidence_interval#Basic_steps
    fig, axs = plt.subplots(1,1, figsize=(15,5), dpi=600)
    sns.despine()
    hasSz_mean = np.mean(hasSz, axis=0)
    hasSz_std = np.std(hasSz, axis=0)
    hasSz_95 = 1.96*hasSz_std/np.sqrt(5)
    pqf_f1_mean = np.mean(pqf_f1, axis=0)
    pqf_f1_std = np.std(pqf_f1, axis=0)
    pqf_f1_95 = 1.96*pqf_f1_std/np.sqrt(5)
    elo_f1_mean = np.mean(elo_f1, axis=0)
    elo_f1_std = np.std(elo_f1, axis=0)
    elo_f1_95 = 1.96*elo_f1_std/np.sqrt(5)
    axs.errorbar(domain, hasSz_mean, yerr=hasSz_95, fmt='.:', barsabove=True, capsize=3, color='#1b9e77')
    axs.errorbar(domain, pqf_f1_mean, yerr=pqf_f1_95, fmt='.:', barsabove=True, capsize=3, color='#d95f02')
    axs.errorbar(domain, elo_f1_mean, yerr=elo_f1_95, fmt='.:', barsabove=True, capsize=3, color='#7570b3')
    axs.legend(['Classification (Q1) Accuracy', 'Seizure Frequency (Q2) F$_1$ Score', 'Most Recent Seizure (Q3) F$_1$ Score'], loc='lower right')
    axs.set_xlabel('Fraction of Training Data Used')
    axs.set_ylabel('Accuracy or F$_1$ Score')
    axs.set_title('Accuracy and F$_1$ Score for Extracting Clinical\nInformation vs. Training Set Size (95% CI)')
    axs.set_ylim([0.2,1])
    axs.grid(True, alpha=0.25)
    fig.savefig('../Figures/Figure_5.pdf', bbox_inches='tight')
    fig.savefig('../Figures/Figure_5.png', bbox_inches='tight')

    #task-centric view
    fig, axs = plt.subplots(1,2, figsize=(15,5), dpi=600)
    sns.despine()
    hasSz_mean = np.mean(hasSz, axis=0)
    hasSz_std = np.std(hasSz, axis=0)
    hasSz_95 = 1.96*hasSz_std/np.sqrt(5)
    axs[0].errorbar(domain, hasSz_mean, yerr=hasSz_95, fmt='.:', barsabove=True, capsize=3, color='#1b9e77')
    axs[0].set_xlabel('Fraction of Training Data Used')
    axs[0].set_ylabel('Classification Accuracy')
    axs[0].set_title('Classification Accuracy vs. Training Set Size (95% CI)')
    axs[0].grid(True, alpha=0.25)
    axs[0].set_ylim([0.2, 1.0])
    axs[0].annotate('A)', xy=(0., 1.01), xycoords='axes fraction', fontweight='bold')
    
    eval_f1_mean = np.mean(eval_f1, axis=0)
    eval_f1_std = np.std(eval_f1, axis=0)
    eval_f1_95 = 1.96*eval_f1_std/np.sqrt(5)
    axs[1].errorbar(domain, eval_f1_mean, yerr=eval_f1_95, fmt='.:', barsabove=True, capsize=3, color='#7570b3')
    axs[1].set_xlabel('Fraction of Training Data Used')
    axs[1].set_ylabel('F$_1$ Score')
    axs[1].set_title('F$_1$ Score for Extracting Seizure Frequencies\nand Occurrences vs. Training Set Size (95% CI)')
    axs[1].set_ylim([.20, 1.00])
    axs[1].grid(True, alpha=0.25)
    axs[1].annotate('B)', xy=(0., 1.01), xycoords='axes fraction', fontweight='bold')
    fig.savefig('../Figures/reduction_task_view.pdf', bbox_inches='tight')
    fig.savefig('../Figures/reduction_task_view.png', bbox_inches='tight')
    
    #plot the F1 scores, split between overall, NoAns, HasAns
    fig, axs = plt.subplots(1,1, figsize=(15,5), dpi=600)
    sns.despine()
    hasAns_mean = np.mean(hasAns, axis=0)
    hasAns_std = np.std(hasAns, axis=0)
    hasAns_95 = 1.96*hasAns_std/np.sqrt(5)
    noAns_mean = np.mean(noAns, axis=0)
    noAns_std = np.std(noAns, axis=0)
    noAns_95 = 1.96*noAns_std/np.sqrt(5)
    eval_f1_mean = np.mean(eval_f1, axis=0)
    eval_f1_std = np.std(eval_f1, axis=0)
    eval_f1_95 = 1.96*eval_f1_std/np.sqrt(5)
    axs.errorbar(domain, hasAns_mean, yerr=hasAns_95, fmt='.:', barsabove=True, capsize=3, color='#1b9e77')
    axs.errorbar(domain, noAns_mean, yerr=noAns_95, fmt='.:', barsabove=True, capsize=3, color='#d95f02')
    axs.errorbar(domain, eval_f1_mean, yerr=eval_f1_95, fmt='.:', barsabove=True, capsize=3, color='#7570b3')
    axs.legend(['Answer Exists', 'No Answer Exists', 'Overall'])
    axs.set_xlabel('Fraction of Training Data Used')
    axs.set_ylabel('F$_1$ Score')
    axs.set_title('F$_1$ Score for Extracting Seizure Frequencies\nand Occurrences vs. Training Set Size (95% CI)')
    axs.set_ylim([.20, 1.00])
    axs.grid(True, alpha=0.25)
    fig.savefig('../Figures/Supplemental_Figure_4.pdf', bbox_inches='tight')
    fig.savefig('../Figures/Supplemental_Figure_4.png', bbox_inches='tight')

def triangular(arr, side):
    """ 
    Upper/lower triangularize a matrix, replacing the lower/upper with np.nan
    instead of the default 0s
    Input:
        arr: the np.array to triangularize
        side: 'upper', or 'lower'
    Returns:
        np.array
    """
    if side.lower() == 'upper':
        upper = np.triu(arr)
        upper[np.tril_indices(arr.shape[0])] = np.nan
        return upper
    elif side.lower() == 'lower':
        lower = np.tril(arr)
        lower[np.triu_indices(arr.shape[0])] = np.nan
        return lower
    else:
        raise Exception("Unknown side. Please use upper or lower")
    
def plot_annotator_agreement():
    #cohen's kappa for classification accuracy
    #row1 and col1 are against the ground truth
    #other rows and cols are against fellow annotators
    g1_hasSz = np.array([[np.nan,	0.917,	0.894,	0.950],
        [np.nan, np.nan,	0.856,	0.874],
        [np.nan, np.nan, np.nan,	0.864],
        [np.nan, np.nan, np.nan, np.nan]])
    g2_hasSz = np.array([[np.nan,	0.913,	0.894,	0.932],
        [np.nan,	np.nan,	0.796,	0.890],
        [np.nan,	np.nan,	np.nan,	0.774],
        [np.nan,	np.nan,	np.nan,	np.nan]])
    g3_hasSz = np.array([[np.nan,	0.890,	0.967,	0.882,	0.657],
        [np.nan,	np.nan,	0.865,	0.880,	0.557],
        [np.nan,	np.nan,	np.nan,	0.886,	0.615],
        [np.nan,	np.nan,	np.nan,	np.nan,	np.nan]])
    g4_hasSz = np.array([[np.nan,	0.910,	0.874,	0.922],
        [np.nan,	np.nan,	0.852,	0.842],
        [np.nan,	np.nan,	np.nan,	0.865],
        [np.nan,	np.nan, np.nan, np.nan]])
    g5_hasSz = np.array([[np.nan,	0.947,	0.905,	0.882],
        [np.nan,	np.nan, 0.858,	0.886],
        [np.nan,	np.nan,	np.nan,	0.832],
        [np.nan,	np.nan,	np.nan,	np.nan]])
    
    #F$_1$ Score for span overlap
    #row1 and col1 are against the ground truths
    #other rows and cols are against fellow annotators
    #upper triangle is for overall F1
    #lower triangle is for paired F1
    g1_szFreq = np.array([[np.nan,	0.627,	0.561,	0.862],
        [0.896,		np.nan,	0.473,	0.539],
        [0.768,		0.726,	np.nan,	0.499],
        [0.952,		0.851,	0.759,	np.nan]])
    g2_szFreq = np.array([[np.nan,	0.687,	0.629,	0.642],
        [0.895,		np.nan,	0.507,	0.496],
        [0.916,		0.860,	np.nan,	0.478],
        [0.867,		0.811,	0.816,	np.nan]])
    g3_szFreq = np.array([[np.nan,	0.770,	0.585,	0.405,	0.459],
        [0.938,		np.nan,	0.474,	0.349,	0.410],
        [0.850,		0.786,	np.nan,	0.438,	0.295],
        [0.785,		0.762,	0.836,	np.nan,	np.nan],
        [0.741,		0.718,	0.685,	np.nan,	np.nan]])
    g4_szFreq = np.array([[np.nan,	0.765,	0.416,	0.655],
        [0.936,		np.nan,	0.392,	0.580],
        [0.696,		0.713,	np.nan,	0.369],
        [0.886,		0.861,	0.684,	np.nan]])
    g5_szFreq = np.array([[np.nan,	0.396,	0.743,	0.604],
        [0.806,		np.nan,	0.350,	0.406],
        [0.873,		0.850,	np.nan,	0.477],
        [0.892,		0.734,	0.785,	np.nan]])
    
    #plot the agreement as a bar graph
    fig, axs = plt.subplots(1,1,dpi=600)
    sns.despine()
    x = [1,2,3]
    width = 0.15
    
    #hasSz bars
    plt.bar(x[0]-2*width,   np.nanmean(g1_hasSz[1:,1:]), yerr = 1.96*np.nanstd(g1_hasSz[1:,1:])/np.sqrt(g1_hasSz[1:,1:].shape[0]),
            width=width, color='#d7191c', edgecolor='black', capsize=3)
    plt.bar(x[0]-width,     np.nanmean(g2_hasSz[1:,1:]), yerr = 1.96*np.nanstd(g2_hasSz[1:,1:])/np.sqrt(g2_hasSz[1:,1:].shape[0]),
            width=width, color='#fdae61', edgecolor='black', capsize=3)
    plt.bar(x[0],           np.nanmean(g3_hasSz[1:,1:]), yerr = 1.96*np.nanstd(g3_hasSz[1:,1:])/np.sqrt(g3_hasSz[1:,1:].shape[0]),
            width=width, color='#ffffbf', edgecolor='black', capsize=3)
    plt.bar(x[0]+width,     np.nanmean(g4_hasSz[1:,1:]), yerr = 1.96*np.nanstd(g4_hasSz[1:,1:])/np.sqrt(g4_hasSz[1:,1:].shape[0]),
            width=width, color='#abd9e9', edgecolor='black', capsize=3)
    plt.bar(x[0]+2*width,   np.nanmean(g5_hasSz[1:,1:]), yerr = 1.96*np.nanstd(g5_hasSz[1:,1:])/np.sqrt(g5_hasSz[1:,1:].shape[0]),
            width=width, color='#2c7bb6', edgecolor='black', capsize=3)
    
    #szFreq overall
    plt.bar(x[1]-2*width,   np.nanmean(triangular(g1_szFreq[1:,1:], 'upper')), 
            yerr = 1.96*np.nanstd(triangular(g1_szFreq[1:,1:], 'upper'))/np.sqrt(g1_szFreq[1:,1:].shape[0]),
            width=width, color='#d7191c', edgecolor='black', capsize=3)
    
    plt.bar(x[1]-width,     np.nanmean(triangular(g2_szFreq[1:,1:], 'upper')), 
            yerr = 1.96*np.nanstd(triangular(g2_szFreq[1:,1:], 'upper'))/np.sqrt(g2_szFreq[1:,1:].shape[0]),
            width=width, color='#fdae61', edgecolor='black', capsize=3)
    
    plt.bar(x[1],           np.nanmean(triangular(g3_szFreq[1:,1:], 'upper')), 
            yerr = 1.96*np.nanstd(triangular(g3_szFreq[1:,1:], 'upper'))/np.sqrt(g3_szFreq[1:,1:].shape[0]),
            width=width, color='#ffffbf', edgecolor='black', capsize=3)
   
    plt.bar(x[1]+width,     np.nanmean(triangular(g4_szFreq[1:,1:], 'upper')), 
            yerr = 1.96*np.nanstd(triangular(g4_szFreq[1:,1:], 'upper'))/np.sqrt(g4_szFreq[1:,1:].shape[0]),
            width=width, color='#abd9e9', edgecolor='black', capsize=3)
    
    plt.bar(x[1]+2*width,   np.nanmean(triangular(g5_szFreq[1:,1:], 'upper')), 
            yerr = 1.96*np.nanstd(triangular(g5_szFreq[1:,1:], 'upper'))/np.sqrt(g5_szFreq[1:,1:].shape[0]),
            width=width, color='#2c7bb6', edgecolor='black', capsize=3)
    
    #szFreq paired
    plt.bar(x[2]-2*width,   np.nanmean(triangular(g1_szFreq[1:,1:], 'lower')), 
            yerr = 1.96*np.nanstd(triangular(g1_szFreq[1:,1:], 'lower'))/np.sqrt(g1_szFreq[1:,1:].shape[0]),
            width=width, color='#d7191c', edgecolor='black', capsize=3)
    
    plt.bar(x[2]-width,     np.nanmean(triangular(g2_szFreq[1:,1:], 'lower')), 
            yerr = 1.96*np.nanstd(triangular(g2_szFreq[1:,1:], 'lower'))/np.sqrt(g2_szFreq[1:,1:].shape[0]),
            width=width, color='#fdae61', edgecolor='black', capsize=3)
    
    plt.bar(x[2],           np.nanmean(triangular(g3_szFreq[1:,1:], 'lower')), 
            yerr = 1.96*np.nanstd(triangular(g3_szFreq[1:,1:], 'lower'))/np.sqrt(g3_szFreq[1:,1:].shape[0]),
            width=width, color='#ffffbf', edgecolor='black', capsize=3)
    
    plt.bar(x[2]+width,     np.nanmean(triangular(g4_szFreq[1:,1:], 'lower')), 
            yerr = 1.96*np.nanstd(triangular(g4_szFreq[1:,1:], 'lower'))/np.sqrt(g4_szFreq[1:,1:].shape[0]),
            width=width, color='#abd9e9', edgecolor='black', capsize=3)
    
    plt.bar(x[2]+2*width,   np.nanmean(triangular(g5_szFreq[1:,1:], 'lower')), 
            yerr = 1.96*np.nanstd(triangular(g5_szFreq[1:,1:], 'lower'))/np.sqrt(g5_szFreq[1:,1:].shape[0]),
            width=width, color='#2c7bb6', edgecolor='black', capsize=3)
    
    plt.ylim([0,1])
    plt.xticks(ticks=x, labels=['Seizure Freedom\nClassification $\kappa$', 'Text Extraction\nOverall F$_1$', 'Text Extraction\nPaired F$_1$'])
    plt.ylabel("Cohen's $\kappa$ or F$_1$ Score")
    plt.title('Mean Annotator Agreement (95% CI)')
    
    #https://matplotlib.org/stable/gallery/text_labels_and_annotations/custom_legends.html
    legend_elements = [Line2D([0], [0], color='#d7191c', label='Group 1', path_effects=[path_effects.Stroke(linewidth=2, foreground='k'), path_effects.Normal()]),
                       Line2D([0], [0], color='#fdae61', label='Group 2', path_effects=[path_effects.Stroke(linewidth=2, foreground='k'), path_effects.Normal()]),
                       Line2D([0], [0], color='#ffffbf', label='Group 3', path_effects=[path_effects.Stroke(linewidth=2, foreground='k'), path_effects.Normal()]),
                       Line2D([0], [0], color='#abd9e9', label='Group 4', path_effects=[path_effects.Stroke(linewidth=2, foreground='k'), path_effects.Normal()]),
                       Line2D([0], [0], color='#2c7bb6', label='Group 5', path_effects=[path_effects.Stroke(linewidth=2, foreground='k'), path_effects.Normal()])]
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.04, 0.5), frameon=False)
    
    plt.savefig('../Figures/Figure_2.pdf', bbox_inches='tight')
    plt.savefig('../Figures/Figure_2.png', bbox_inches='tight')
    

plot_human_v_machine()
plot_ablations()
plot_training_reductions()
plot_annotator_agreement()

