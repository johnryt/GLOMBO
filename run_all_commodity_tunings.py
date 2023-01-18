import sys
import warnings

from modules.Many import *
import numpy as np

# warnings.filterwarnings('error')
# np.seterr(all='raise')

# ------- Running all mining scenarios, lots of commodities ------- #
to_run = ['Cu', 'Al', 'Au', 'Sn', 'Ni', 'Ag', 'Zn', 'Pb', 'Steel']
# to_run = ['Sn','Ni','Ag','Zn','Pb','Steel']
# to_run = ['Ag','Zn','Pb','Al','Au']
# to_run = ['Ni','Cu','Sn','Steel']

# mod = Many()
# mod.run_all_mining(200,commodities=None, constrain_tuning_to_sign=False, filename_modifier='_unconstrain1_mcpe0',n_parallel=2)
#
# mod = Many()
# mod.run_all_mining(200,commodities=None, constrain_tuning_to_sign=False, filename_modifier='_unconstrain_mcpe0',n_parallel=2)

# need to run mining unconstrain and constrain again, unconstrain3 demand,
#  then can run full integration tune for both.

# ------- Running all demand scenarios, lots of commodities ------- #

# run 1110 10p
# mod = Many()
# mod.run_all_demand(100, constrain_tuning_to_sign=True, filename_modifier='_constrain',commodities=['Al'])

# mod = Many()
# mod.run_all_demand(100, constrain_tuning_to_sign=False, filename_modifier='_unconstrain1_mcpe0',n_parallel=5)
#
# mod = Many()
# mod.run_all_demand(100, constrain_tuning_to_sign=False, filename_modifier='_unconstrain_mcpe0',n_parallel=5)

# mod = Many()
# mod.run_all_demand(100, constrain_tuning_to_sign=True, filename_modifier='_constrain_mcpe0',commodities=['Au'])

# mod = Many()
# mod.run_all_demand(300, constrain_tuning_to_sign=False, filename_modifier='_unconstrain1')


# ------- Running all integrated scenarios, lots of commodities ------- #
# want to run all of these at some point
# mod = Many()
# to_run = ['Al','Au','Sn','Cu','Ni','Ag','Zn','Pb','Steel']
# mod.run_all_integration(200, tuned_rmse_df_out_append='_mcpe0_2016',
#     train_time=np.arange(2001,2016), simulation_time=np.arange(2001,2020),
#     normalize_objectives=True, constrain_previously_tuned=True,
#     commodities=to_run, filename_modifier='_mcpe0_2016',
#     n_parallel=2)
#
# mod = Many()
# mod.run_all_integration(200, tuned_rmse_df_out_append='_mcpe0_2015',
#     train_time=np.arange(2001,2015), simulation_time=np.arange(2001,2020),
#     normalize_objectives=True, constrain_previously_tuned=True,
#     commodities=to_run, filename_modifier='_mcpe0_2015',
#     n_parallel=2)
#
# mod = Many()
# mod.run_all_integration(200, tuned_rmse_df_out_append='_mcpe0_unconstrain',
#     train_time=np.arange(2001,2020), simulation_time=np.arange(2001,2041),
#     normalize_objectives=True, constrain_previously_tuned=True,
#     constrain_tuning_to_sign=False,
#     commodities=to_run, filename_modifier='_mcpe0_unconstrain',
#     n_parallel=2)
#
# mod = Many()
# mod.run_all_integration(n_runs=200, n_params=2, n_jobs=3,
#                         tuned_rmse_df_out_append='_mcpe0_noprice',
#                         train_time=np.arange(2001,2020),
#                         normalize_objectives=True,
#                         constrain_previously_tuned=True,
#                         commodities=to_run,
#                         force_integration_historical_price=False,
#                         filename_modifier='_mcpe0_noprice',
#                         n_parallel=2)
# to_run=['Steel']
# mcpe0 I forgot to include mine_cost_og_elas in the tuning, rerun has that and a method to make sure ore grades don't get too high

#################### Checking how well integ runs
# many=Many()
# many.get_variables('all',filename_modifier='_rerun_overhead_sxew',tuned_rmse_df_out_append='_mcpe0',commodities=['Cu'])
# integ1 = Integration(user_data_folder='generalization/data',
#                      commodity='Cu',
#                      price_to_use='log',
#                      simulation_time=np.arange(2001,2041),
#                      input_hyperparam=many.hyperparam_sorted.loc['copper'][0],
#                      historical_price_rolling_window=5,
#                      use_historical_price_for_mine_initialization=True,
#                      )
# integ1.hyperparam.loc['primary_overhead_regression2use','Value']='None'
# integ1.run()

############# Current main run method
mod = Many()
# to_run = ['Steel']
# to_run = ['Al','Sn','Ni','Ag','Zn','Pb','Steel']
# mod.run_all_integration(n_runs=200, n_params=3, n_jobs=3,
#                         tuned_rmse_df_out_append='_split_grades',
#                         train_time=np.arange(2001,2020),
#                         simulation_time=np.arange(2001,2041),
#                         normalize_objectives=True,
#                         constrain_previously_tuned=False,
#                         commodities=to_run,
#                         force_integration_historical_price=False,
#                         filename_modifier='_split_grades',
#                         save_mining_info=False,
#                         use_historical_price_for_mine_initialization=True,
#                         n_parallel=1,
#                         verbosity=0)
# to_run = ['Cu','Ni']
# mod.run_all_integration(n_runs=5, n_params=3, n_jobs=3,
#                         tuned_rmse_df_out_append='_split_2017_TEST',
#                         train_time=np.arange(2001,2017),
#                         simulation_time=np.arange(2001,2041),
#                         normalize_objectives=True,
#                         constrain_previously_tuned=False,
#                         commodities=to_run,
#                         force_integration_historical_price=False,
#                         filename_modifier='_split_2017_TEST',
#                         save_mining_info=False,
#                         use_historical_price_for_mine_initialization=True,
#                         n_parallel=1,
#                         verbosity=0)

# _rerun_overhead has the updated everything, and uses reduced overhead cost, uses historical price for mine init
# _rerun_overhead_nohistinit is same as _rerun_overhead but doesn't use historical price for mine init
# _rerun_overhead_sxew is same as _rerun_overhead except with sxew fixes

# ------- Simple mining test ------- #
# import sys
# sys.path.append('generalization')
# from mining_class import miningModel
#
# m=miningModel(verbosity=10)
# for i in m.simulation_time:
#     m.i = i
#     m.run()


# ------- Testing mining ------- #
# import pickle
# import pandas as pd
# import numpy as np
# from modules.mining_class import miningModel
# from modules.integration import Integration
# from IPython.display import display
#
#
# input_file = pd.read_excel('input_files/user_defined/case study data.xlsx',index_col=0)
# commodity_inputs = input_file['Cu'].dropna()
# commodity_inputs.loc['incentive_require_tune_years'] = 10
# commodity_inputs.loc['presimulate_n_years'] = 10
# commodity_inputs.loc['end_calibrate_years'] = 10
# commodity_inputs.loc['start_calibrate_years'] = 5
# commodity_inputs.loc['close_price_method'] = 'max'
# integ = Integration(simulation_time=np.arange(2001,2020),byproduct=False,verbosity=0, commodity='Cu',
#                     scenario_name='input_files/user_defined/Scenario setup.xlsx++1')
# for i in commodity_inputs.index:
#     integ.hyperparam.loc[i,'Value'] = commodity_inputs[i]
# integ.run()
#
# file = open('pickleyshist','wb')
# pickle.dump(mining,file)
# file.close()

# ------- Running / testing sensitivity class functionality ------- #

# from integration_functions import Sensitivity
# import numpy as np
# material='Cu'
# element_commodity_map = {'Al':'Aluminum','Au':'Gold','Cu':'Copper','Steel':'Steel','Co':'Cobalt','REEs':'REEs','W':'Tungsten','Sn':'Tin','Ta':'Tantalum','Ni':'Nickel','Ag':'Silver','Zn':'Zinc','Pb':'Lead','Mo':'Molybdenum','Pt':'Platinum','Te':'Telllurium','Li':'Lithium'}
# filename='data/'+element_commodity_map[material].lower()+'_run_hist_DELETE.pkl'
# shist1 = Sensitivity(filename,changing_base_parameters_series=material,notes='Monte Carlo aluminum run',
#                 simulation_time=np.arange(2001,2020),OVERWRITE=True,use_alternative_gold_volumes=True,
#                     historical_price_rolling_window=5,verbosity=0)
# shist1.historical_sim_check_demand(50,demand_or_mining='mining')

# to_run = ['Steel','Al','Au','Sn','Cu','Ni','Ag','Zn','Pb']
to_run = ['Cu', 'Au', 'Al', 'Ag', 'Zn', 'Pb', 'Sn', 'Ni', 'Steel']
# to_run = ['Au','Al','Ag','Zn','Pb','Sn','Ni','Steel']
# to_run = ['Sn','Cu','Ni','Steel']
# to_run = ['Al','Au']
# run_future_scenarios(commodities=to_run,run_parallel=5,verbosity= -1,
#     scenario_name_base='_run_scenario_set',supply_or_demand='demand')
#
# run_future_scenarios(commodities=to_run,run_parallel=6,verbosity= -1,
#     scenario_name_base='_run_scenario_set_alt_hist_g',supply_or_demand='demand-alt',
#     simulation_time=np.arange(2001,2041), baseline_sampling='grouped')
#
# run_future_scenarios(commodities=to_run,run_parallel=6,verbosity= -1,
#     scenario_name_base='_run_scenario_set_hist_g',supply_or_demand='demand',
#     simulation_time=np.arange(2001,2041), baseline_sampling='grouped')
#
# run_future_scenarios(commodities=to_run,run_parallel=6,verbosity= -1,
#     scenario_name_base='_run_scenario_set_alt_hist',supply_or_demand='demand-alt',
#     simulation_time=np.arange(2019,2041), baseline_sampling='random')
#
# run_future_scenarios(commodities=to_run,run_parallel=6,verbosity= -1,
#     scenario_name_base='_run_scenario_set_hist',supply_or_demand='demand',
#     simulation_time=np.arange(2019,2041), baseline_sampling='random')

# run_future_scenarios(commodities=to_run,run_parallel=6,verbosity= -1,
#     scenario_name_base='_run_scenario_set_alt_hist_g_new',supply_or_demand='demand-alt',
#     simulation_time=np.arange(2019,2041), baseline_sampling='grouped')
#
# run_future_scenarios(commodities=to_run,run_parallel=6,verbosity= -1,
#     scenario_name_base='_run_scenario_set_hist_g_new',supply_or_demand='demand',
#     simulation_time=np.arange(2019,2041), baseline_sampling='grouped')

# These are the main four I've been running
# run_future_scenarios(commodities=to_run,run_parallel=6,verbosity=-1,
#     scenario_name_base='_run_scenario_set_alt_hist_act_10yr',supply_or_demand='demand-alt',
#     simulation_time=np.arange(2001,2041), baseline_sampling='actual',
#     years_of_increase=np.arange(10,11))
#
# run_future_scenarios(commodities=to_run,run_parallel=6,verbosity= -1,
#     scenario_name_base='_run_scenario_set_alt_act_10yr',supply_or_demand='demand-alt',
#     simulation_time=np.arange(2019,2041), baseline_sampling='actual',
#     years_of_increase=np.arange(10,11))
# #
# run_future_scenarios(commodities=to_run,run_parallel=-2,verbosity= -1,
#     scenario_name_base='_run_scenario_set_supply_hist_act',supply_or_demand='both-alt',
#     simulation_time=np.arange(2001,2041), baseline_sampling='clustered',
#     years_of_increase=np.arange(1,2),tuned_rmse_df_out_append='_mcpe0',
#     save_mining_info=False,
#     n_best_scenarios=10,n_per_baseline=5
#     )
# run_future_scenarios(commodities=to_run,run_parallel=-2,verbosity= -1,
#     scenario_name_base='_run_scenario_set_alt_hist_act',supply_or_demand='demand-alt',
#     simulation_time=np.arange(2001,2041), baseline_sampling='clustered',
#     years_of_increase=np.arange(1,2),tuned_rmse_df_out_append='_mcpe0',
#     save_mining_info=False,
#     n_best_scenarios=10,n_per_baseline=5
# )
#
# run_future_scenarios(commodities=to_run,run_parallel=6,verbosity= -1,
#     scenario_name_base='_run_scenario_set_alt_act',supply_or_demand='demand-alt',
#     simulation_time=np.arange(2019,2041), baseline_sampling='actual',
#     years_of_increase=np.arange(1,2))

# ------- Running a bunch of baselines ------- #
warnings.simplefilter('error')
run_future_scenarios(commodities=to_run, run_parallel=3, verbosity=-1,
                     scenario_name_base='_baselines', supply_or_demand=None,
                     simulation_time=np.arange(2001, 2041), baseline_sampling='clustered',
                     tuned_rmse_df_out_append='_split_grades',
                     save_mining_info=False, # 'cost_curve'
                     n_best_scenarios=2, n_per_baseline=2,
                     )
# to_run=['Cu']
# run_future_scenarios(commodities=to_run,run_parallel= 4,verbosity= -1,
#     scenario_name_base='_run_scenario_check_cumu_ot',supply_or_demand=None,
#     simulation_time=np.arange(2001,2041), baseline_sampling='clustered',
#     tuned_rmse_df_out_append='_rerun_overhead',
#     save_mining_info=True,
#     n_best_scenarios=1, n_per_baseline=1,
# )

# checking scenario file functionality
# to_run=['Cu','Ni']
# warnings.simplefilter('error')
# run_future_scenarios(commodities=to_run, run_parallel=3, verbosity=-1,
#                      scenario_sheet_file_path='input_files/user_defined/Scenario setup.xlsx',
#                      scenario_name_base='_TEST', supply_or_demand=None,
#                      simulation_time=np.arange(2001,2041), baseline_sampling='clustered',
#                      tuned_rmse_df_out_append='_split_grades',
#                      save_mining_info=False,
#                      n_best_scenarios=5, n_per_baseline=1)
