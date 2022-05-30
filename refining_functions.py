import numpy as np
import pandas as pd
idx = pd.IndexSlice

def cu_growth_cal(TCRC_growth, TCRC_elas, ref_bal = 0, CU_ref_bal_elas = 0):
    if CU_ref_bal_elas == 0:
        growth=TCRC_growth**(TCRC_elas)
    else:
        growth = TCRC_growth**(TCRC_elas)*ref_bal**CU_ref_bal_elas
    return growth

def sec_ratio_growth_cal(TCRC_growth, spread_growth, TCRC_elas, spread_elas):
    growth=TCRC_growth**(TCRC_elas)*spread_growth**(spread_elas)
    return growth

def pri_cap_growth_cal(concentrate_growth):
    growth=concentrate_growth
    return growth


def sec_cap_growth_cal(mine_growth, scrap_growth, sec_coef=0):
    growth=mine_growth*(1-sec_coef)+scrap_growth*sec_coef
    return growth


def simulate_refinery_production_oneyear(year_i, tcrc_series, sp2_series, 
                                         pri_cap_growth_series, sec_cap_growth_series,
                                         ref_stats, ref_hyper_param, 
                                         sec_coef=0, growth_lag=1, ref_bal = 0, 
                                         pri_CU_ref_bal_elas = 0, sec_CU_ref_bal_elas = 0,
                                         ref_cu_pct_change = 0, ref_sr_pct_change = 0):
    
    ref_stats_next=pd.Series(0, index=ref_stats.columns)
    
    pri_CU_TCRC_elas=ref_hyper_param['pri CU TCRC elas']
    sec_CU_TCRC_elas=ref_hyper_param['sec CU TCRC elas']
    sec_ratio_TCRC_elas=ref_hyper_param['sec ratio TCRC elas']
    sec_ratio_SP2_elas=ref_hyper_param['sec ratio scrap spread elas']
    
    t=year_i
    t_lag_1=year_i-1
    t_lag_2=year_i-2
    
    tcrc_growth=tcrc_series.loc[t]/tcrc_series.loc[t_lag_1]
    sp2_growth=sp2_series.loc[t]/sp2_series.loc[t_lag_1]
    ratio_growth=sec_ratio_growth_cal(tcrc_growth, sp2_growth, sec_ratio_TCRC_elas, sec_ratio_SP2_elas)
    if growth_lag == 1:
        pri_growth=pri_cap_growth_series.loc[t_lag_1]/pri_cap_growth_series.loc[t_lag_2]
        sec_growth=sec_cap_growth_series.loc[year_i-1]/sec_cap_growth_series.loc[year_i-2]
    else:
        pri_growth=pri_cap_growth_series.loc[t]/pri_cap_growth_series.loc[t_lag_1]
        sec_growth=sec_cap_growth_series.loc[year_i]/sec_cap_growth_series.loc[year_i-1]
    
    if pri_CU_ref_bal_elas == 0:
        pri_cu_growth=cu_growth_cal(tcrc_growth, pri_CU_TCRC_elas)
    else:
        pri_cu_growth=cu_growth_cal(tcrc_growth, pri_CU_TCRC_elas, ref_bal, pri_CU_ref_bal_elas)
    if sec_CU_ref_bal_elas == 0:
        sec_cu_growth=cu_growth_cal(tcrc_growth, sec_CU_TCRC_elas)
    else: 
        sec_cu_growth=cu_growth_cal(tcrc_growth, sec_CU_TCRC_elas, ref_bal, sec_CU_ref_bal_elas)
    if ref_cu_pct_change != 0:
        pri_cu_growth = 1-ref_cu_pct_change/100
        sec_cu_growth = 1-ref_cu_pct_change/100
        ratio_growth = 1-ref_sr_pct_change/100

    pri_cap_growth=pri_cap_growth_cal(pri_growth)
    sec_cap_growth=sec_cap_growth_cal(pri_growth, sec_growth, sec_coef)

    ref_stats_next.loc['Primary CU']=ref_stats.loc[t_lag_1, 'Primary CU']*pri_cu_growth
    ref_stats_next.loc['Secondary CU']=ref_stats.loc[t_lag_1, 'Secondary CU']*sec_cu_growth
    ref_stats_next.loc['Secondary ratio']=ref_stats.loc[t_lag_1, 'Secondary ratio']*ratio_growth
    ref_stats_next[ref_stats_next<0] = 0
    ref_stats_next[ref_stats_next>1] = 1
    ref_stats_next.loc['Primary capacity']=ref_stats.loc[t_lag_1, 'Primary capacity']*pri_cap_growth
    ref_stats_next.loc['Secondary capacity']=ref_stats.loc[t_lag_1, 'Secondary capacity']*sec_cap_growth
    ref_stats_next.loc['Primary production']=ref_stats_next.loc['Primary capacity']*ref_stats_next.loc['Primary CU']\
    +ref_stats_next.loc['Secondary capacity']*ref_stats_next.loc['Secondary CU']*(1-ref_stats_next.loc['Secondary ratio'])
    ref_stats_next.loc['Secondary production']=\
    ref_stats_next.loc['Secondary capacity']*ref_stats_next.loc['Secondary CU']*ref_stats_next.loc['Secondary ratio']
    
    return ref_stats_next


def ref_stats_init(simulation_time, ref_hyper_param):
    ref_stats=pd.DataFrame(0, index=simulation_time, 
                           columns=['Primary capacity', 'Primary CU', 'Secondary capacity', 'Secondary CU',
                                    'Secondary ratio', 'Primary production', 'Secondary production'])
    pri_cap=ref_hyper_param['pri cap']
    pri_CU=ref_hyper_param['pri CU']
    sec_cap=ref_hyper_param['sec cap']
    sec_CU=ref_hyper_param['sec CU']
    sec_ratio=ref_hyper_param['sec ratio']
    
    ref_stats.loc[simulation_time[0], 'Primary capacity']=pri_cap
    ref_stats.loc[simulation_time[0], 'Primary CU']=pri_CU
    ref_stats.loc[simulation_time[0], 'Secondary capacity']=sec_cap
    ref_stats.loc[simulation_time[0], 'Secondary CU']=sec_CU
    ref_stats.loc[simulation_time[0], 'Secondary ratio']=sec_ratio
    
    ref_stats.loc[simulation_time[0], 'Primary production']=pri_cap*pri_CU+sec_cap*sec_CU*(1-sec_ratio)
    ref_stats.loc[simulation_time[0], 'Secondary production']=sec_cap*sec_CU*sec_ratio
    
    return ref_stats