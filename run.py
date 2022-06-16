import numpy as np
import os
import pandas as pd
idx = pd.IndexSlice
from matplotlib import pyplot as plt
import sys
sys.path.append('generalization')
from integration import Integration
from useful_functions import *
from integration_functions import *
from scipy import stats
from random import seed, sample, shuffle
import warnings

os.chdir('C:\\Users\\ryter\\Dropbox (MIT)\\John MIT\\Research\\generalizationOutside\\generalization')

input_file = pd.read_excel('C:/Users/ryter/Dropbox (MIT)/Group Research Folder_Olivetti/Displacement/08 Generalization/_Python/Data/case study data.xlsx',index_col=0)
commodity_inputs = input_file['Al'].dropna()
commodity_inputs.loc['incentive_require_tune_years'] = 10
commodity_inputs.loc['presimulate_n_years'] = 10
commodity_inputs.loc['end_calibrate_years'] = 10
commodity_inputs.loc['start_calibrate_years'] = 5
commodity_inputs.loc['refinery_follows_concentrate_supply'] = False
commodity_inputs.loc['incentive_opening_probability'] = 0
commodity_inputs.loc['mine_cu_margin_elas'] = 0.8

commodity_inputs.loc['refinery_follows_concentrate_supply'] = False

scenarios = ['sd_nono_5yr_0%tot_0%inc']+\
    ['sd_no_'+str(yr)+'yr_'+str(pct)+'%tot_0%inc' for yr in np.arange(5,21,5) for pct in np.arange(5,21,5)]
#                          'ss_pr_3yr_2%tot_0%inc','ss_no_3yr_2%tot_0%inc',
#                          'sd_pr_3yr_2%tot_0%inc','sd_no_3yr_2%tot_0%inc',
#                          'ss_pr_10yr_2%tot_']
filename = 'data/big_sensitivity_sd_no.pkl'
n_scen = 50
OVERWRITE = True

for i in np.arange(0,n_scen):
    print('Running outside scenario {}/{}, {:.1f}% complete'.format(i+1,n_scen,i/n_scen*100))
    ci = generate_commodity_inputs(commodity_inputs, i)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        OVERWRITE = OVERWRITE if i==0 else False
        s = Sensitivity(filename,ci,OVERWRITE=OVERWRITE,notes='Big sensitivity, scrap demand shock, no collection rate price response',scenarios=scenarios,verbosity=0)
        s.run_monte_carlo(n_scenarios=2,random_state=220615+i)