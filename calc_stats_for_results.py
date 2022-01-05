"""
Performs 2-sided Mann-Whitney U Tests for performance metrics
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

#HvM performance
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

#p-values for human vs. machine performance
hvc_hasSz_stat, hvc_hasSz_p = stats.mannwhitneyu(
    clinicalBERT_performance[:, 0], human_performance[:, 0])
hvr_hasSz_stat, hvr_hasSz_p = stats.mannwhitneyu(
    RoBERTa_performance[:, 0], human_performance[:, 0])
hvb_hasSz_stat, hvb_hasSz_p = stats.mannwhitneyu(
    BERT_base_performance[:, 0], human_performance[:, 0])
print("Human vs. Machine p-vales for HasSz classification.\nHvC:{hvc}\nHvR:{hvr}\nHvB:{hvb}".format(
    hvc=hvc_hasSz_p, hvr=hvr_hasSz_p, hvb=hvb_hasSz_p))
print("")

hvc_pqf_stat, hvc_pqf_p = stats.mannwhitneyu(
    clinicalBERT_performance[:, -2], human_performance[:, -2])
hvr_pqf_stat, hvr_pqf_p = stats.mannwhitneyu(
    RoBERTa_performance[:, -2], human_performance[:, -2])
hvb_pqf_stat, hvb_pqf_p = stats.mannwhitneyu(
    BERT_base_performance[:, -2], human_performance[:, -2])
print("Human vs. Machine p-vales for PQF extraction.\nHvC:{hvc}\nHvR:{hvr}\nHvB:{hvb}".format(
    hvc=hvc_pqf_p, hvr=hvr_pqf_p, hvb=hvb_pqf_p))
print("")

hvc_elo_stat, hvc_elo_p = stats.mannwhitneyu(
    clinicalBERT_performance[:, -1], human_performance[:, -1])
hvr_elo_stat, hvr_elo_p = stats.mannwhitneyu(
    RoBERTa_performance[:, -1], human_performance[:, -1])
hvb_elo_stat, hvb_elo_p = stats.mannwhitneyu(
    BERT_base_performance[:, -1], human_performance[:, -1])
print("Human vs. Machine p-vales for ELO extraction.\nHvC:{hvc}\nHvR:{hvr}\nHvB:{hvb}".format(
    hvc=hvc_elo_p, hvr=hvr_elo_p, hvb=hvb_elo_p))
print("")

print("")
print("")


#Ablation p-values
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

fvM_hasSz_stat, fvM_hasSz_p = stats.mannwhitneyu(
    full_model[:, 0], no_MLM[:, 0])
fvS_hasSz_stat, fvS_hasSz_p = stats.mannwhitneyu(
    full_model[:, 0], no_SQB3[:, 0])
fvA_hasSz_stat, fvA_hasSz_p = stats.mannwhitneyu(
    full_model[:, 0], no_Anno[:, 0])
print("Full v Ablation p-vales for HasSz classification.\nFvM:{fvm}\nFvS:{fvs}\nFvA:{fva}".format(
    fvm=fvM_hasSz_p, fvs=fvS_hasSz_p, fva=fvA_hasSz_p))
print("")

fvM_pqf_stat, fvM_pqf_p = stats.mannwhitneyu(
    full_model[:, -2], no_MLM[:, -2])
fvS_pqf_stat, fvS_pqf_p = stats.mannwhitneyu(
    full_model[:, -2], no_SQB3[:, -2])
fvA_pqf_stat, fvA_pqf_p = stats.mannwhitneyu(
    full_model[:, -2], no_Anno[:, -2])
print("Full v Ablation p-vales for pqf classification.\nFvM:{fvm}\nFvS:{fvs}\nFvA:{fva}".format(
    fvm=fvM_pqf_p, fvs=fvS_pqf_p, fva=fvA_pqf_p))
print("")

fvM_elo_stat, fvM_elo_p = stats.mannwhitneyu(
    full_model[:, -1], no_MLM[:, -1])
fvS_elo_stat, fvS_elo_p = stats.mannwhitneyu(
    full_model[:, -1], no_SQB3[:, -1])
fvA_elo_stat, fvA_elo_p = stats.mannwhitneyu(
    full_model[:, -1], no_Anno[:, -1])
print("Full v Ablation p-vales for elo classification.\nFvM:{fvm}\nFvS:{fvs}\nFvA:{fva}".format(
    fvm=fvM_elo_p, fvs=fvS_elo_p, fva=fvA_elo_p))
print("")