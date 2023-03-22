import warnings
from modules.integration_functions import Sensitivity
from modules.scenario_parser import get_scenario_dataframe
from modules.load_data import LoadFolderContents
import numpy as np
import pandas as pd
from modules.useful_functions import *
from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from joblib import Parallel, delayed
import os
from shutil import copyfile

import shap
from modules.Individual import *
from datetime import datetime
import warnings

# warnings.filterwarnings('error')
# np.seterr(all='raise')

# def setup_files(OVERWRITE=False):
#     if not os.path.exists('data'):
#         os.mkdir('output_files')
#         print('Directory \'output_files\' created')
#     if not os.path.exists('output_files/Historical tuning'):
#         os.mkdir('output_files/Historical tuning')
#         print('Directory \'output_files/Historical tuning\' created')
#     if not os.path.exists('output_files/Simulation'):
#         os.mkdir('output_files/Simulation')
#         print('Directory \'output_files/Simulation\' created')
#     tuning_files   = [i for i in os.listdir('generalization/output_files/Historical tuning') if 'run_hist_all_mcpe0' in i]
#     baseline_files = [i for i in os.listdir('generalization/output_files/Simulation') if '_run_scenario_baselines' in i]
#     tuning_files_copied, baseline_files_copied = [], []
#     if len(tuning_files)>0:
#         for i in tuning_files:
#             if not os.path.exists(f'output_files/Historical tuning/{i}') or OVERWRITE:
#                 copyfile(f'generalization/output_files/Historical tuning/{i}',f'output_files/Historical tuning/{i}')
#                 tuning_files_copied += [i]
#     if len(baseline_files)>0:
#         for i in baseline_files:
#             if not os.path.exists(f'output_files/Simulation/{i}') or OVERWRITE:
#                 copyfile(f'generalization/output_files/Simulation/{i}',f'output_files/Simulation/{i}')
#                 tuning_files_copied += [i]
#     print('The following files have been copied from the github directory (generalization/output_files/Historical tuning/ & generalization/output_files/Simulation/) to folder in the working directory (output_files/Historical tuning/ & output_files/Simulation):')
#     print(tuning_files_copied)
#     print(baseline_files_copied)
#     if (len(tuning_files_copied)==0 and len(tuning_files)!=0) or (len(baseline_files_copied)==0 and len(baseline_files)!=0):
#         print('Files already exist in target directory. If you would like to overwrite them, run the function:\n  setup_files(OVERWRITE=True)')
#
# setup_files(OVERWRITE=False)

class Many():
    '''
    Runs many commodities simultaneously, and contains methods for plotting
    the outcomes of those runs.

    Methods within Many include:
    - run_all_demand
    - run_mining
    - run_all_integration
    - get_variables (for loading all info for each commodity within a single
        object, e.g. mining or demand)
    - get_multiple (for loading all info for each commodity, creating separate
        Many instance for each object such that you have Many.demand,
        Many.mining, and/or Many.integ)
    - plot_all_demand
    - plot_all_mining
    - plot_all_integration

    The Many object can also be passed to many of the other functions included
    in this file, which include:

    For feature importance:
    - feature_importance
    - nice_feature_importance_plot
    - commodity_level_feature_importance
    - plot_all_feature_importance_plots
    - make_parameter_names_nice
    - prep_for_snsplots
    - plot_demand_parameter_correlation
    - plot_important_parameter_scatter
    - commodity_level_feature_importance_heatmap
    - nice_plot_pretuning

    For running future scenarios:
    - run_future_scenarios
        - op_run_future_scenarios
        - op_run_sensitivity_fn
        - op_run_future_scenarios_parallel
        - run_scenario_set
    - generate_clustered_hyperparam

    For comparing train and test sets:
    - get_pretuning_params
    - get_train_test_scores
    - get_commodity_scores
    - get_best_columns
    - plot_given_columns
    - plot_best_scenarios_train_test
    - plot_test_score_vs_train_score
        - plot_test_score_vs_train_score_commodity
    - plot_best_scores_history
    - plot_best_scores_hyperparam_distributions

    For comparing constrained and unconstrained tuning:
    - compare_constrained_unconstrained_tuning
        - get_rmse_df_results_hyperparam
        - get_expected_parameter_signs

    For SHAP and SRI analysis:
    - SHAP
        - __init__
        - initialize
        - run_tree_based_regression
        - initialize_shap
        - get_interactions
        - plot_shap
        - summary_plot
        - waterfall_plot
        - heatmap_plot
        - shap_vs_param_plot
        - gamma_facet
    - draw_facet_dendrogram
    - plot_sri_matrices

    For interpreting future or tuning scenarios:
    - plot_best_scenario_sd
    - stackplot_scrap_demand
    - make_parameter_mean_std_table
    - plot_violin_all

    Not all of the above take Many as an input; some are standalone or are
    called by other functions in this file.
    '''
    def __init__(self, user_data_folder=None, static_data_folder=None, output_data_folder=None):
        '''
        user_data_folder: str, the folder where historical data needing loading lives
        output_data_folder: str, folder where pkl files of results will be saved
        '''
        self.ready_commodities = ['Al','Au','Sn','Cu','Ni','Ag','Zn','Pb','Steel']
        self.element_commodity_map = {'Steel':'Steel','Al':'Aluminum','Au':'Gold','Cu':'Copper','Steel':'Steel','Co':'Cobalt','REEs':'REEs','W':'Tungstate','Sn':'Tin','Ta':'Tantalum','Ni':'Nickel','Ag':'Silver','Zn':'Zinc','Pb':'Lead','Mo':'Molybdenum','Pt':'Platinum','Te':'Telllurium','Li':'Lithium'}
        self.commodity_element_map = dict(zip(self.element_commodity_map.values(),self.element_commodity_map.keys()))
        self.user_data_folder = 'input_files/user_defined' if user_data_folder == None else user_data_folder
        self.static_data_folder = 'input_files/static' if static_data_folder == None else static_data_folder
        self.output_data_folder = 'output_files' if output_data_folder == None else output_data_folder
        self.objective_results_map = {'Total demand':'Total demand','Primary commodity price':'Refined price',
                                 'Primary demand':'Conc. demand','Primary supply':'Mine production',
                                'Conc. SD':'Conc. SD','Scrap SD':'Scrap SD','Ref. SD':'Ref. SD'}

    def run_all_demand(self, n_runs=50, commodities=None, save_mining_info=False, trim_result_df=True, constrain_tuning_to_sign=True, filename_base='_run_hist', filename_modifier='', n_parallel=1):
        """
        Runs pre-tuning of the demand-specific parameters, using historical
        price and having demand match to historical demand based on historical
        price. Tuning uses Bayesian optimization. Saves best results in
        updated_commodity_inputs.pkl file for reference in integration tuning.

        ------------
        n_runs: int, number of Bayesian optimization runs until stopping
        commodities: None or list, commodity names to run (formatted as in the
            case study data.xlsx file) if a list is given, if None will use
            self.ready_commodities
        save_mining_info: bool, whether or not to save mine-level information.
            Default is False, takes up a lot of memory if True.
        trim_result_df: bool, whether to save all years or just those simulated.
            Default is False, works to try and save some memory so everyting
            is not going back to 1912
        constrain_tuning_to_sign: bool, whether to require that tuning variables
            match signs with their signs in integration hyperparameters (setup
            in integration.py). If False, allows for +/- values, which planning
            to use for testing the statistical signficance of the results, and
            makes a updated_commodity_inputs.pkl file with the name
            updated_commodity_inputs_unconstrained.pkl that will be used for
            integration if this variable is False there.
        filename_base: str, default `_run_hist`. Base name of file given when
            running, is the part coming after the commodity name and before
            `_DEM`
        filename_modifier: str, filename modifier, comes after the `_DEM` but
            before `.pkl`
        n_parallel: int, number of cores to use for running commodities in
            parallel (number of commodities to run simultaneously)
        """
        t1 = datetime.now()
        self.time_str = str(datetime.now()).replace(':', '_').replace('.', '_')[:21]
        commodities = self.ready_commodities if commodities==None else commodities
        def run_individual_demand(material):
            print('-'*40)
            print(material)
            mat = self.element_commodity_map[material].lower()
            filename=f'{self.output_data_folder}/{mat}{filename_base}{filename_modifier}.pkl'
            self.shist1 = Sensitivity(pkl_filename=filename, user_data_folder=self.user_data_folder,
                                      static_data_folder=self.static_data_folder,
                                      changing_base_parameters_series=material, notes='Monte Carlo aluminum run',
                                      simulation_time=np.arange(2001,2020), OVERWRITE=True,
                                      use_alternative_gold_volumes=True,
                                      constrain_tuning_to_sign=constrain_tuning_to_sign,
                                      historical_price_rolling_window=5, verbosity=0, trim_result_df=trim_result_df)
            self.shist.time_str = self.time_str
            self.shist1.historical_sim_check_demand(n_runs,demand_or_mining='demand')
            print(f'time elapsed: {str(datetime.now()-t1)}')

        Parallel(n_jobs=n_parallel)(delayed(run_individual_demand)(material) for material in commodities)

    def run_all_mining(self, n_runs=50, commodities=None, save_mining_info=False, trim_result_df=True, constrain_tuning_to_sign=True, filename_base='_run_hist', filename_modifier='', n_parallel=1):
        """
        Runs pre-tuning of the mining-specific parameters, using historical
        price and having mining match to historical demand based on historical
        price. Tuning uses Bayesian optimization. Saves best results in
        updated_commodity_inputs.pkl file for reference in integration tuning.
        ------------
        n_runs: int, number of Bayesian optimization runs until stopping
        commodities: None or list, commodity names to run (formatted as in the
            case study data.xlsx file) if a list is given, if None will use
            self.ready_commodities
        save_mining_info: bool, whether or not to save mine-level information.
            Default is False, takes up a lot of memory if True. Can also take
            the str `cost_curve` to save a subset of parameters that enable cost
            curve construction
        trim_result_df: bool, whether to save all years or just those simulated.
            Default is False, works to try and save some memory so everyting
            is not going back to 1912
        constrain_tuning_to_sign: bool, whether to require that tuning variables
            match signs with their signs in integration hyperparameters (setup
            in integration.py). If False, allows for +/- values, which planning
            to use for testing the statistical signficance of the results, and
            makes a updated_commodity_inputs.pkl file with the name
            updated_commodity_inputs_unconstrained.pkl that will be used for
            integration if this variable is False there.
        filename_base: str, default `_run_hist`. Base name of file given when
            running, is the part coming after the commodity name and before
            `_mining`
        filename_modifier: str, filename modifier, comes after the `_mining` but
            before `.pkl`
        n_parallel: int, number of cores to use for running commodities in
            parallel (number of commodities to run simultaneously)
        """
        commodities = self.ready_commodities if commodities==None else commodities
        self.time_str = str(datetime.now()).replace(':', '_').replace('.', '_')[:21]
        # for material in commodities:
        def run_individual_mining(material):
            t1 = datetime.now()
            print('-'*40)
            print(material)
            mat = self.element_commodity_map[material].lower()
            filename=f'{self.output_data_folder}/{mat}{filename_base}{filename_modifier}.pkl'
            additional_base_parameters = pd.Series(1,['refinery_capacity_growth_lag'])
            if 'mcpe0' in filename_modifier:
                additional_base_parameters.loc['mine_cost_price_elas'] = 0
            self.shist = Sensitivity(pkl_filename=filename, user_data_folder=self.user_data_folder,
                                     static_data_folder=self.static_data_folder,
                                     changing_base_parameters_series=material, notes='Monte Carlo aluminum run',
                                     simulation_time=np.arange(2001,2020), OVERWRITE=True, use_alternative_gold_volumes=True,
                                     additional_base_parameters = additional_base_parameters,
                                     historical_price_rolling_window=5, verbosity=0,
                                     constrain_tuning_to_sign=constrain_tuning_to_sign,
                                     incentive_opening_probability_fraction_zero=0, save_mining_info=save_mining_info,
                                     trim_result_df=trim_result_df)
            self.shist.time_str = self.time_str
            self.shist.historical_sim_check_demand(n_runs,demand_or_mining='mining')
            print(f'time elapsed: {str(datetime.now()-t1)}')

        Parallel(n_jobs=n_parallel)(delayed(run_individual_mining)(material) for material in commodities)

    def run_all_integration(self, n_runs=200, n_params=3, n_jobs=3, tuned_rmse_df_out_append=None, commodities=None,
                            train_time=np.arange(2001,2020), simulation_time=np.arange(2001,2020),
                            normalize_objectives=False, force_integration_historical_price=False,
                            constrain_previously_tuned=False, verbosity=0, save_mining_info=False, trim_result_df=True,
                            constrain_tuning_to_sign=True, use_historical_price_for_mine_initialization=True,
                            filename_base='_run_hist', filename_modifier='', n_parallel=1):
        """
        Runs parameter tuning, trying to match to historical demand, price, and
        mine production. Tuning uses Bayesian optimization. Saves all results in
        `tuned_rmse_df_out`+tuned_rmse_df_out_append+`.pkl` for use in future
        runs (see run_future_scenarios function)
        ------------
        n_runs: int, number of Bayesian optimization runs until stopping
        n_params: int, number of columns from case study data.xlsx to use,
            starting from the leftmost and moving right. If set to 2, it will
            do the first three and drop any with `price` in the str (upper- or
            lowercase) - will still save the RMSEs of all three, but the score
            will be computed without it (see skopt_run_score function)
        n_jobs: int, number of parallel jobs run in the Bayesian optimization
        tuned_rmse_df_out_append: str, string appended to tuned_rmse_df_out
            filename so you can differentiate tuning results, if needed.
            Defaults to using the same value as filename_modifier input
        commodities: None or list, commodity names to run (formatted as in the
            case study data.xlsx file) if a list is given, if None will use
            self.ready_commodities
        train_time: np.ndarray, of years used for training/tuning the model.
            Can run for more years than this using simulation_time, to compare
            model outcome for non-training years with actual and test
            accuracy.
        simulation_time: np.ndarray of years to run the model in ascending
            order. Needs to contain all years in train_time.
        normalize_objectives: bool, whether or not to normalize the RMSE values
            in the tuning by dividing each computed RMSE by the simulation_time
            start value for the corresponding historical data. If False, can
            have issues due to the different orders of magnitude for demand vs
            price vs mine production, resulting in different relative importance
        force_integration_historical_price: bool, whether to force Integration
            models to use historical price rather than let it evolve
            independently.
        constrain_previously_tuned: bool, if True, requires any bayesian
            optimization tuning parameters that have previously been tuned
            (by historical_sim_check_demand/run_all_demand/run_all_mining,
            meaning they are in the index of self.updated_commodity_inputs(_sub))
            to be 0.001-2X their previously-tuned value, if the optimization
            is trying to tune them. If False, constraints are as they were
            previously and does not use updated_commodity_inputs.pkl. Does not
            apply to demand variables if dont_constrain_demand is True, which is
            the default for the Sensitivity class.
        save_mining_info: bool, whether or not to save mine-level information.
            Default is False, takes up a lot of memory if True. Can also take
            the str `cost_curve` to save a subset of parameters that enable cost
            curve construction.
        trim_result_df: bool, whether to save all years or just those simulated.
            Default is False, works to try and save some memory so everyting
            is not going back to 1912
        verbosity: float/int, the amount of outputs to print, typically -1 or 0,
            but can go above 3.
        constrain_tuning_to_sign: bool, whether to require that tuning variables
            match signs with their signs in integration hyperparameters (setup
            in integration.py). If False, allows for +/- values, which planning
            to use for testing the statistical signficance of the results, and
            uses the updated_commodity_inputs.pkl file with the name
            updated_commodity_inputs_unconstrained.pkl from running the demand
            and mining pre-tuning with this variable False
        filename_base: str, default `_run_hist`. Base name of file given when
            running, is the part coming after the commodity name and before
            `_mining`
        filename_modifier: str, filename modifier, comes after the `_mining` but
            before `.pkl`
        n_parallel: int, number of cores to use for running commodities in
            parallel (number of commodities to run simultaneously)
        """
        if tuned_rmse_df_out_append is None:
            tuned_rmse_df_out_append = filename_modifier

        commodities = self.ready_commodities if commodities==None else commodities
        if os.path.exists(f'output_files/tuned_rmse_df_out{tuned_rmse_df_out_append}.csv'):
            rmse_df_out = pd.read_csv(f'output_files/tuned_rmse_df_out{tuned_rmse_df_out_append}.csv', index_col=[0,1])
            if rmse_df_out.index.nlevels==1:
                rmse_df_out = pd.DataFrame()
        else:
            rmse_df_out = pd.DataFrame()

        for material in commodities:
            mat = self.element_commodity_map[material].lower()
            if rmse_df_out.index.nlevels>1:
                if mat in rmse_df_out.index.get_level_values(0):
                    rmse_df_out.drop(mat,level=0,inplace=True)

        t1 = datetime.now()
        self.time_str = str(datetime.now()).replace(':', '_').replace('.', '_')[:21]

        def run_individual_integration(self,material):
            print('-'*40)
            print(material)
            # timer=IterTimer()
            mat = self.element_commodity_map[material].lower()

            filename=f'{self.output_data_folder}/{mat}{filename_base}_all{filename_modifier}.pkl'
            print('--'*15+filename+'-'*15)
            additional_base_parameters = pd.Series(1,['refinery_capacity_growth_lag'])
            if 'mcpe0' in filename_modifier:
                additional_base_parameters.loc['mine_cost_price_elas'] = 0
            # comparison
            self.s = Sensitivity(pkl_filename=filename, user_data_folder=self.user_data_folder,
                                 static_data_folder=self.static_data_folder,
                                 changing_base_parameters_series=material,
                                 notes=f'{filename}: Monte Carlo {material} run',
                                 additional_base_parameters=additional_base_parameters,
                                 simulation_time=simulation_time, include_sd_objectives=False, train_time=train_time,
                                 OVERWRITE=True, verbosity=verbosity, historical_price_rolling_window=5,
                                 force_integration_historical_price=force_integration_historical_price,
                                 constrain_tuning_to_sign=constrain_tuning_to_sign,
                                 constrain_previously_tuned=constrain_previously_tuned, normalize_objectives=normalize_objectives,
                                 use_historical_price_for_mine_initialization=use_historical_price_for_mine_initialization,
                                 save_mining_info=save_mining_info, trim_result_df=trim_result_df)
            self.s.time_str = self.time_str
            sensitivity_parameters = [
                'pri CU price elas',
                'sec CU price elas',
                'pri CU TCRC elas',
                'sec CU TCRC elas',
                'sec ratio TCRC elas',
                'sec ratio scrap spread elas',
                'primary_commodity_price_elas_sd',
                'tcrc_elas_sd',
                'tcrc_elas_price',
                'scrap_spread_elas_sd',
                'scrap_spread_elas_primary_commodity_price',
                'direct_melt_elas_scrap_spread',
                'collection_elas_scrap_price',
                'refinery_capacity_fraction_increase_mining',
                'incentive_opening_probability',
                'mine_cu_margin_elas',
                'mine_cost_change_per_year',
                'mine_cost_og_elas',
                'primary_oge_scale',
                'initial_ore_grade_decline',
                'sector_specific_dematerialization_tech_growth',
                'intensity_response_to_gdp',
                'sector_specific_price_response',
                'primary_price_resources_contained_elas',
            ]
            if force_integration_historical_price:
                sensitivity_parameters = [
                    'pri CU price elas',
                    'sec CU price elas',
                    'pri CU TCRC elas',
                    'sec CU TCRC elas',
                    'sec ratio TCRC elas',
                    'sec ratio scrap spread elas',
                    'tcrc_elas_sd',
                    'tcrc_elas_price',
                    'scrap_spread_elas_sd',
                    'scrap_spread_elas_primary_commodity_price',
                    'direct_melt_elas_scrap_spread',
                    'collection_elas_scrap_price',
                    'refinery_capacity_fraction_increase_mining']
            if n_params==2:
                sensitivity_parameters = [i for i in sensitivity_parameters if i!='primary_commodity_price_elas_sd']
            self.s.run_historical_multiple_integration_models(n_scenarios=n_runs,
                                                              sensitivity_parameters=sensitivity_parameters,
                                                              bayesian_tune=True, n_params=n_params, n_jobs=n_jobs)
            rmse_df = self.s.rmse_df.copy()
            ind = [j for j in self.s.updated_commodity_inputs.index if j not in rmse_df.columns and j!='pareto_3p']
            for k in ind:
                rmse_df.loc[:,k] = self.s.updated_commodity_inputs[material][k]
            rmse_df = pd.concat([rmse_df],keys=[mat])
            return rmse_df

        if len(commodities) > 1:
            output = Parallel(n_jobs=n_parallel)(delayed(run_individual_integration)(self, material) for material in commodities)
            rmse_df_out = pd.concat([rmse_df_out, pd.concat(output)]).fillna(0)
            rmse_df_out.to_csv(f'{self.output_data_folder}/tuned_rmse_df_out{tuned_rmse_df_out_append}.csv')
        else:
            output = run_individual_integration(self, commodities[0])
            print(f'commodity info not updated in {self.output_data_folder}/tuned_rmse_df_out{tuned_rmse_df_out_append}.csv')

        print(f'time elapsed: {str(datetime.now()-t1)}')
        # add 'response','growth' to sensitivity_parameters input to allow demand parameters to change again

    def get_variables(self, demand_mining_all='demand',filename_base='_run_hist',filename_modifier='',tuned_rmse_df_out_append=None, commodities=None):
        '''
        loads the data from the output pkl files and concatenates the dataframes
        for each commodity. Main variable names are results, hyperparam, and
        rmse_df, where each of these can have _sorted appended to them to
        reorder their scenario numbering from min RMSE to maximum (when viewing
        the dataframe left to right, or smallest scenario number to highest)

        sorting takes place from min rmse to max
        ------
        demand_mining_all: str, can be demand, mining, or all
        filename_base: str, default `_run_hist`. Base name of file given when
            running, is the part coming after the commodity name and before
            `_mining`/`_DEM`/`_all`
        filename_modifier: str, filename modifier, comes after the `_mining`/
            `_DEM`/`_all` but before `.pkl`
        '''
        commodities = commodities if commodities is not None else self.ready_commodities
        self.filename_base = filename_base
        self.filename_modifier = filename_modifier
        self.tuned_rmse_df_out_append = tuned_rmse_df_out_append if tuned_rmse_df_out_append!=None else filename_modifier
        if demand_mining_all=='all':
            self.tuned_rmse_df_out = pd.read_csv(f'output_files/tuned_rmse_df_out{self.tuned_rmse_df_out_append}.csv', index_col=[0,1])

        for df_name in ['rmse_df','hyperparam','simulated_demand','results','historical_data','mine_supply']:
            df_outer = pd.DataFrame()
            df_outer_sorted = pd.DataFrame()

        output_data_folder = self.output_data_folder
        if output_data_folder=='output_files' or output_data_folder.split('/')[-1]=='output_files':
            output_data_folder = output_data_folder.replace('output_files','output_files/Historical tuning')

        for material in commodities:
            material = self.element_commodity_map[material].lower()
            if demand_mining_all=='demand':
                indiv = Individual(filename=f'{output_data_folder}/{material}{filename_base}{filename_modifier}_DEM.pkl',rmse_not_mae=False,dpi=50)
                rmse_or_score = 'RMSE'
            elif demand_mining_all=='mining':
                indiv = Individual(filename=f'{output_data_folder}/{material}{filename_base}{filename_modifier}_mining.pkl',rmse_not_mae=False,dpi=50)
                rmse_or_score = 'RMSE'
            elif demand_mining_all=='all':
                indiv = Individual(filename=f'{output_data_folder}/{material}{filename_base}_all{filename_modifier}.pkl',rmse_not_mae=False,dpi=50)
                rmse_or_score = 'score'
            else: raise ValueError('input for the demand_mining_all variable when calling the Many().get_variables() function must be a string of one of the following: demand, many, all')

            if 'limit' in self.filename_modifier:
                self.add_primary_commodity_price_elas_sd_commodity(indiv)
            setattr(self,'indiv_'+material,indiv)

            for df_name in ['rmse_df','hyperparam','simulated_demand','results','historical_data','mine_data']:
                df_name_sorted = f'{df_name}_sorted'
                if hasattr(indiv,df_name):
                    if not hasattr(self,df_name):
                        setattr(self,df_name,pd.DataFrame())
                        setattr(self,df_name_sorted,pd.DataFrame())
                    df_ph = pd.concat([getattr(indiv,df_name).dropna(how='all').dropna(axis=1,how='all')],keys=[material])
                    if df_ph.columns.nlevels>1 and 'Notes' in df_ph.columns.get_level_values(1): df_ph = df_ph.loc[:,idx[:,'Value']].droplevel(1,axis=1)
                    if df_name!='rmse_df' and df_name!='historical_data':
                        sorted_cols = self.rmse_df.loc[material,rmse_or_score].dropna().sort_values().index
                        if df_name=='results' and demand_mining_all=='all':
                            df_ph_sorted = df_ph.copy().loc[idx[:,sorted_cols,:],:].unstack()
                            df_ph_sorted.index = df_ph_sorted.index.set_names(['Commodity','scenario number old'])
                            prev_names = list(df_ph_sorted.index.names)
                            df_ph_sorted = df_ph_sorted.reset_index(drop=False)
                            df_ph_sorted.index.name = 'Scenario number'
                            df_ph_sorted = df_ph_sorted.reset_index(drop=False)
                            df_ph_sorted = df_ph_sorted.set_index(['Commodity','Scenario number']).sort_index().T.sort_index().T.drop(columns='scenario number old')
                            df_ph_sorted = df_ph_sorted.stack(1)
                        elif df_name=='mine_data' and demand_mining_all=='all':
                            df_ph_sorted = df_ph.copy().loc[idx[:,sorted_cols,:],:].unstack().unstack()
                            df_ph_sorted.index = df_ph_sorted.index.set_names(['Commodity','scenario number old'])
                            prev_names = list(df_ph_sorted.index.names)
                            df_ph_sorted = df_ph_sorted.reset_index(drop=False)
                            df_ph_sorted.index.name = 'Scenario number'
                            df_ph_sorted = df_ph_sorted.reset_index(drop=False)
                            df_ph_sorted = df_ph_sorted.set_index(['Commodity','Scenario number']).sort_index().T.sort_index().T.drop(columns='scenario number old')
                            df_ph_sorted = df_ph_sorted.stack().stack()
                        else:
                            df_ph_sorted = df_ph.copy().loc[:,sorted_cols]
                            df_ph_sorted = df_ph_sorted.T.reset_index(drop=True).T
                    elif df_name=='rmse_df':
                        df_ph_sorted = df_ph.copy().sort_values(by=(material,rmse_or_score),axis=1)
                        df_ph_sorted = df_ph_sorted.T.reset_index(drop=True).T
                    df_outer = pd.concat([getattr(self,df_name),df_ph])
                    if df_name!='historical_data':
                        df_outer_sorted = pd.concat([getattr(self,df_name_sorted),df_ph_sorted])
                    if df_outer.shape[0]>0 and type(df_outer.index[0])==tuple:
                        df_outer.index = pd.MultiIndex.from_tuples(df_outer.index)
                    if df_name!='historical_data':
                        if df_outer_sorted.shape[0]>0 and type(df_outer_sorted.index[0])==tuple:
                            df_outer_sorted.index = pd.MultiIndex.from_tuples(df_outer_sorted.index)
                    setattr(self,df_name,df_outer)
                    if df_name!='historical_data':
                        setattr(self,df_name_sorted,df_outer_sorted)

        if 'limit' in filename_modifier and demand_mining_all=='all':
            self.add_primary_commodity_price_elas_sd_tuned()

        types = pd.Series([type(i) for i in self.hyperparam.iloc[:,0]],self.hyperparam.index)
        self.types = types.copy()
        types = (types == float) | (types == int) | (types == np.float64)
        self.changing_hyperparam = self.hyperparam.loc[types].copy()
        self.changing_hyperparam = self.changing_hyperparam.loc[~(self.changing_hyperparam.apply(lambda x: x-x.mean(),axis=1)<1e-6).all(axis=1)]

    def get_multiple(self, demand=True, mining=True, integ=False, reinitialize=False, filename_base='_run_hist', filename_modifier='', filename_modify_non_integ=False, noninteg_modifier='', tuned_rmse_df_out_append=None, commodities=None):
        '''
        Runs the get_variables command on each type of model run, which are
        then accessible through self.mining, self.demand, and self.integ, each
        as an instance of the Many class.
        -----------------------
        demand: bool, whether to load the demand pre-tuning results
        mining: bool, whether to load the mining pre-tuning results
        integ: bool, whether to load the full tuning results
        reinitialize: bool, allows this to get called multiple times such that
          if you've already loaded e.g. demand and do not want to reload it
          when you load mining, you leave this value as False
        filename_base: str, default `_run_hist`. Base name of file given when
          running, is the part coming after the commodity name and before
          `_mining`/`_DEM`/`_all`
        filename_modifier: str, filename modifier, comes after the `_mining`/
          `_DEM`/`_all` but before `.pkl`
        '''
        self.filename_base = filename_base
        self.filename_modifier = filename_modifier
        self.noninteg_modifier = noninteg_modifier

        if demand and (not hasattr(self,'demand') or reinitialize):
            self.demand = Many()
            self.demand.get_variables('demand', filename_base=filename_base, filename_modifier=filename_modifier if filename_modify_non_integ else noninteg_modifier, commodities=commodities)
            feature_importance(self.demand,plot=False,objective='RMSE')

        if mining and (not hasattr(self,'mining') or reinitialize):
            self.mining = Many()
            self.mining.get_variables('mining', filename_base=filename_base, filename_modifier=filename_modifier if filename_modify_non_integ else noninteg_modifier, commodities=commodities)
            feature_importance(self.mining,plot=False,objective='score')

        if integ and (not hasattr(self,'integ') or reinitialize):
            self.integ = Many()
            self.integ.get_variables('all', filename_base=filename_base, filename_modifier=filename_modifier, tuned_rmse_df_out_append=tuned_rmse_df_out_append, commodities=commodities)
            feature_importance(self.integ,plot=False,objective='score')

    def add_primary_commodity_price_elas_sd_tuned(self):
        """

        """
        def get_regression_results(price, sd):
            price=price.copy()
            sd = sd.copy()
            sd.name = 'SD'
            m = sm.GLS(np.log(price), sm.add_constant(np.log(sd))).fit(cov_type='HC3')
            return pd.Series([m.params['SD'], round(m.pvalues['SD'],6), m.rsquared],['coefficient','pvalue','rsquared'])

        commodities = self.results.index.get_level_values(0).unique()
        regress_results = pd.DataFrame()
        for commodity in commodities:
            res = self.results.loc[commodity]
            sd = (res['Ref. supply']/res['Ref. demand']).unstack(0).shift(1).dropna()
            price = res['Refined price'].unstack(0).loc[sd.index]
            regr = sd.apply(lambda x: get_regression_results(price[0],x),axis=0)
            regr = pd.concat([regr],keys=[commodity])
            regress_results = pd.concat([regress_results,regr])

        rename_dict={'coefficient':'primary_commodity_price_elas_sd',
                     'pvalue':'primary_price_pval_score',
                     'rsquared':'primary_price_r2_score'}
        for k in rename_dict.values():
            if k in self.tuned_rmse_df_out.columns:
                self.tuned_rmse_df_out.drop(k,axis=1,inplace=True)
        self.tuned_rmse_df_out = pd.concat([
            self.tuned_rmse_df_out,
            regress_results.loc[idx[:,:],1:].rename(rename_dict,level=1).stack().unstack(1)],axis=1
            ).sort_index().T.sort_index().T

    def add_primary_commodity_price_elas_sd_commodity(self,indiv):
        def get_regression_results(price, sd):
            price=price.copy()
            sd = sd.copy()
            sd.name = 'SD'
            m = sm.GLS(np.log(price), sm.add_constant(np.log(sd))).fit(cov_type='HC3')
            return pd.Series([m.params['SD'], round(m.pvalues['SD'],6), m.rsquared],['coefficient','pvalue','rsquared'])


        regress_results = pd.DataFrame()
        res = indiv.results.copy()
        sd = (res['Ref. supply']/res['Ref. demand']).unstack(0).shift(1).dropna()
        price = res['Refined price'].unstack(0).loc[sd.index]
        regress_results = sd.apply(lambda x: get_regression_results(price[0],x),axis=0)
        if 0 in regress_results.columns:
            regress_results = regress_results.loc[:,1:]

        rename_dict={'coefficient':'primary_commodity_price_elas_sd',
                     'pvalue':'primary_price_pval_score',
                     'rsquared':'primary_price_R2_score'}
        for k in rename_dict.values():
            if k in indiv.rmse_df.index:
                indiv.rmse_df.drop(k,inplace=True)
        indiv.rmse_df = pd.concat([
            indiv.rmse_df,
            regress_results.rename(rename_dict)]).sort_index().T.sort_index().T

    def plot_all_demand(self, dpi=50, filename_base='_run_hist', filename_modifier=''):
        '''
        Loads each commodity in the Individual class and runs its
        plot_demand_results method.
        -----------
        dpi: int, dots per inch for controlling figure resolution
        filename_base: str, default `_run_hist`. Base name of file given when
            running, is the part coming after the commodity name and before
            `_mining`/`_DEM`/`_all`
        filename_modifier: str, filename modifier, comes after the `_mining`/
            `_DEM`/`_all` but before `.pkl`
        '''
        if hasattr(self,'filename_base') and filename_base=='_run_hist':
            filename_base = self.filename_base
        if hasattr(self,'filename_modifier') and filename_modifier=='':
            filename_modifier = self.filename_modifier

        for material in self.ready_commodities:
            material = self.element_commodity_map[material].lower()
            filename=f'{self.output_data_folder}/{material}{filename_base}{filename_modifier}_DEM.pkl'
            indiv = Individual(filename=filename,rmse_not_mae=False,dpi=dpi)
            indiv.plot_demand_results()
        plt.show()
        plt.close()

    def plot_all_mining(self,dpi=50, filename_base='_run_hist', filename_modifier=''):
        '''
        Loads each commodity in the Individual class and runs its
        plot_demand_results method.
        -----------
        dpi: int, dots per inch for controlling figure resolution
        filename_base: str, default `_run_hist`. Base name of file given when
            running, is the part coming after the commodity name and before
            `_mining`/`_DEM`/`_all`
        filename_modifier: str, filename modifier, comes after the `_mining`/
            `_DEM`/`_all` but before `.pkl`
        '''
        if hasattr(self,'filename_base') and filename_base=='_run_hist':
            filename_base = self.filename_base
        if hasattr(self,'filename_modifier') and filename_modifier=='':
            filename_modifier = self.filename_modifier

        for material in self.ready_commodities:
            material = self.element_commodity_map[material].lower()
            filename=f'{self.output_data_folder}/{material}{filename_base}{filename_modifier}_mining.pkl'
            indiv = Individual(filename=filename,rmse_not_mae=False,dpi=dpi)
            indiv.plot_demand_results()

    def plot_all_integration(self,dpi=50, plot_over_time=True, nth_best=1, weight_price=1, include_sd=False, plot_sd_over_time=False, plot_best_indiv_over_time=False, plot_hyperparam_heatmap=False, n_best=20, plot_hyperparam_distributions=False, n_per_plot=4, plot_hyperparam_vs_error=False, flip_yx=False, plot_best_params=False, plot_supply_demand_stack=False, filename_base='_run_hist', filename_modifier=''):
        '''
        Produces many different plots you can use to try and understand the
        model outputs. Loads each commodity in the Individual class, and runs
        its plot_results method on it.

        More info is given with each model input bool description below.

        ----------------------
        Inputs:
        plot_over_time: bool, True plots the best overall scenario over time
            (using NORM SUM or NORM SUM OBJ ONLY) for each objective to allow comparison
        n_best: int, the number of scenarios to include in plot_over_time or in
            plot_hyperparam_distributions
        include_sd: bool, True means we use the NORM SUM row to evaluate the
            best scenario, while False means we use NORM SUM OBJ ONLY
        plot_hyperparam_heatmap: bool, True plots a heatmap of the
            hyperparameter values for the best n scenarios
        plot_hyperparam_distributions: bool, True plots the hyperparameter
            distributions
        n_per_plot: int, for use with plot_hyperparam_distributions. Determines
            how many hyperparameter values are put in each plot, since it can be
            hard to tell what is going on when there are too many lines in a
            figure
        plot_hyperparam_vs_error: bool, plots the hyperparameter values vs the
            error value, separate plot for each hyperparameter. Use this to try
            and see if there are ranges for the best hyperparameter values.
        flip_yx: bool, False means plot hyperparam value vs error, while
            True means plot error vs hyperparam value
        filename_base: str, default `_run_hist`. Base name of file given when
            running, is the part coming after the commodity name and before
            `_mining`/`_DEM`/`_all`
        filename_modifier: str, filename modifier, comes after the `_mining`/
            `_DEM`/`_all` but before `.pkl`
        '''
        if hasattr(self,'filename_base') and filename_base=='_run_hist':
            filename_base = self.filename_base
        if hasattr(self,'filename_modifier') and filename_modifier=='':
            filename_modifier = self.filename_modifier

        for element in self.ready_commodities:
            material = self.element_commodity_map[element].lower()
            indiv = Individual(element, 3, filename=f'{self.output_data_folder}/{material}{filename_base}_all{filename_modifier}.pkl',
                               rmse_not_mae=True, weight_price=weight_price, dpi=dpi, price_rolling=5)
            # indiv.plot_best_all()
            # indiv.find_pareto(plot=True,log=True,plot_non_pareto=False)
            fig_list = indiv.plot_results(plot_over_time=plot_over_time, nth_best=nth_best,
                               include_sd=include_sd,
                               plot_sd_over_time=plot_sd_over_time,
                               plot_best_indiv_over_time=plot_best_indiv_over_time,
                               plot_hyperparam_heatmap=plot_hyperparam_heatmap,
                               n_best=n_best,
                               plot_hyperparam_distributions=plot_hyperparam_distributions,
                               n_per_plot=n_per_plot,
                               plot_hyperparam_vs_error=plot_hyperparam_vs_error,
                               flip_yx=flip_yx,
                               plot_best_params=plot_best_params,
                               plot_supply_demand_stack=plot_supply_demand_stack,
                               )
            fig_list[0].suptitle(material.capitalize(),weight='bold',y=1.02,x=0.515)
            plt.show()
            plt.close()

    def load_future_scenario_runs(self, output_data_folder=None, commodities=None, scenario_name_base='_run_scenario_set',verbosity=None):
        """
        loads values from the crazy number of scenarios generated by the
        run_future_scenarios function, stores them in:
        - self.multi_scenario_results
        - self.multi_scenario_hyperparam
        - self.multi_scenario_results_formatted
        - self.multi_scenario_hyperparam_formatted

        formatted versions have the columns renamed to be the scenario changes

        output_data_folder: str, folder where data is saved
        commodities: list or np.ndarray, when None uses all those from
          self.ready_commodities, but otherwise can use a list of commodities
          in the case study data.xlsx format
        scenario_name_base: str, base name of file given when running

        """
        if output_data_folder is None:
            output_data_folder = self.output_data_folder
        if commodities is None:
            commodities = self.ready_commodities
        if verbosity is None:
            verbosity=self.verbosity
        output_data_folder_ph = output_data_folder

        self.multi_scenario_results = pd.DataFrame()
        self.multi_scenario_hyperparam = pd.DataFrame()
        self.multi_scenario_mine_data = pd.DataFrame()
        self.historical_data = pd.DataFrame()
        loaded_commodities=[]
        for element in commodities:
            if verbosity>1: print(f'starting {element}')
            if output_data_folder_ph=='output_files' or output_data_folder_ph.split('/')[-1]=='output_files':
                output_data_folder = output_data_folder_ph.replace('output_files','output_files/Simulation')
            commodity = self.element_commodity_map[element].lower()

            dir_list = os.listdir(output_data_folder)
            name = f'{commodity}{scenario_name_base}'
            filename_list = [f'{output_data_folder}/{n}' for n in dir_list if name in n and len(n.split(name)[1].split('.')[0])<=2]
            if verbosity>0:
                print(filename_list)
            indiv_list = [Individual(element,3,name) if name.split('/')[-1] in dir_list else name for name in filename_list]
            print_list = [j for j in indiv_list if type(j)!=Individual]
            if len(print_list)>0:
                print(f'{element} is missing {len(print_list)} files, set verbosity>0 to view')
                if verbosity>0:
                    print('Missing files:')
                    missing_ints = [int(j.split(scenario_name_base)[1].split('.pkl')[0]) for j in print_list]
                    if len(missing_ints)==missing_ints[-1]-missing_ints[0]+1:
                        print(f'\t{print_list[0]} to {print_list[-1]}')
                    else:
                        for j in print_list:
                            print(f'\t{j}')
            indiv_list = [j for j in indiv_list if type(j)==Individual]
            # print(indiv_list)
            if len(indiv_list)!=0:
                multi_scenario_results_ph = pd.concat([indiv.results for indiv in indiv_list],keys=np.arange(0,len(indiv_list)))
                multi_scenario_hyperparam_ph = pd.concat([indiv.hyperparam for indiv in indiv_list],keys=np.arange(0,len(indiv_list)))
                multi_scenario_minedata_ph = pd.concat([indiv.mine_data for indiv in indiv_list],keys=np.arange(0,len(indiv_list)))

                multi_scenario_results_ph = pd.concat([multi_scenario_results_ph],keys=[commodity])
                multi_scenario_hyperparam_ph = pd.concat([multi_scenario_hyperparam_ph],keys=[commodity])
                multi_scenario_minedata_ph = pd.concat([multi_scenario_minedata_ph],keys=[commodity])
                self.multi_scenario_results = pd.concat([self.multi_scenario_results, multi_scenario_results_ph])
                self.multi_scenario_hyperparam = pd.concat([self.multi_scenario_hyperparam, multi_scenario_hyperparam_ph])
                self.multi_scenario_mine_data = pd.concat([self.multi_scenario_mine_data, multi_scenario_minedata_ph])
                hist_data = indiv_list[0].historical_data
                hist_data = pd.concat([hist_data],keys=[commodity])
                self.historical_data = pd.concat([self.historical_data, hist_data])
                if verbosity>0: print(f'{element} completed')
                loaded_commodities += [element]
            elif verbosity>0: print(f'{element} skipped due to missing scenarios')
        if verbosity>-1: print(f'loaded commodities: {loaded_commodities},\nskipped commodities: {[i for i in commodities if i not in loaded_commodities]}')

        if self.multi_scenario_results.shape[0]!=0:
            ph = self.multi_scenario_results.stack().unstack(2)
            ele = ph.index[0][0]
            ph = ph.rename(columns=dict(zip(ph.columns,[(self.multi_scenario_hyperparam.loc[idx[ele,0,'secondary_refined_duration'],i],round(self.multi_scenario_hyperparam.loc[idx[ele,0,['secondary_refined_pct_change_tot','direct_melt_pct_change_tot']],i].sum()-1,4)) for i in self.multi_scenario_hyperparam.columns])))
            ph.columns = pd.MultiIndex.from_tuples(ph.columns)
            ph = ph.loc[:,~ph.columns.duplicated()]
            ph = ph.stack(0).stack().unstack(3)
            self.multi_scenario_results_formatted = ph.copy()

            ph = self.multi_scenario_hyperparam.copy()
            ph = ph.rename(columns=dict(zip(ph.columns,[(self.multi_scenario_hyperparam.loc[idx[ele,0,'secondary_refined_duration'],i],round(self.multi_scenario_hyperparam.loc[idx[ele,0,['secondary_refined_pct_change_tot','direct_melt_pct_change_tot']],i].sum()-1,4)) for i in self.multi_scenario_hyperparam.columns])))
            ph.columns = pd.MultiIndex.from_tuples(ph.columns)
            ph = ph.loc[:,~ph.columns.duplicated()]
            self.multi_scenario_hyperparam_formatted = ph.copy()

            sec_ref_fraction_is_zero = (self.multi_scenario_hyperparam_formatted.loc[idx[:,:,'Secondary refinery fraction of recycled content, Global'],:]==0).any(axis=1)
            sec_ref_fraction_is_zero = sec_ref_fraction_is_zero[sec_ref_fraction_is_zero]
            glv = sec_ref_fraction_is_zero.index.get_level_values
            drop_ind = pd.MultiIndex.from_product([glv(0).unique(), glv(1).unique(), ['sec CU price elas','sec CU TCRC elas','sec ratio TCRC elas','sec ratio scrap spread elas']])
            if len(sec_ref_fraction_is_zero)>0:
                self.multi_scenario_hyperparam_formatted.drop(drop_ind, inplace=True)
        else:
            raise FileExistsError('No files matching the input found')

    def load_data(self, folder_path):
        """"
            Used to load all the csv files from within a given scenario folder.
            If you only want a single commodity, load the entire folder and down-select
            from there, as single commodity (or subsets) are not supported. Give the
            overall scenario folder, not the data_Element version. Folder names will
            correspond with the time the model was initialized, followed by a given
            scenario name.

            Will look for the folder in the current working directory, the path given,
            and in:
                generalization/output_files/Historical tuning
                generalization/output_files/Simulation
                output_files/Historical tuning
                output_files/Simulation

            ----------------------------------------
            OUTPUTS:
                No direct outputs, but saves the following variables as attributes of self
                - rmse_df
                - hyperparam
                - results
                - historical_data
                - rmse_df_sorted
                - hyperparam_sorted
                - results_sorted
        """
        lod = LoadFolderContents(folder_path)
        lod.load_scenario_data()
        lod.get_sorted_dataframes()
        dataframes_to_pull = ['results','rmse_df','hyperparam','historical_data','results_sorted','rmse_df_sorted','hyperparam_sorted']

        for i in dataframes_to_pull:
            dataframe = getattr(lod, i)
            setattr(self, i, dataframe.copy())

def get_X_df_y_df(self, commodity=None, objective=None, standard_scaler=True):
    """
    Sets up dataframes for processing in feature_importance function

    self: Many() instance with rmse_df object or
        multi_scenario_results_formatted object if objective!=None.
    commodity: None or str. If str, has to be lowercase commodity name, in which
        case this gets the right commodity from the corresponding dataframe. If
        None, uses the full dataframe, going across all commodities
    objective: None or str. If str, has to be one of the columns in
        multi_scenario_results_formatted, otherwise will be using `score` or
        `RMSE` from rmse_df dataframe depending on whether using the Integration
        or demandModel formulation. If str, will try to do the mean difference
        from baseline.
    standard_scaler: bool, whether to use standard scaler to rescale data so it
        is N(0,1), which we should probably always do.
    """
    using_rmse_y = objective is None or hasattr(self,'rmse_df')
    if objective is None and not hasattr(self,'rmse_df'):
        raise ValueError('Many object does not have rmse_df, so cannot support running with objective=None. Set objective to something you want to observe, e.g. `Conc. supply`')
    if using_rmse_y: using_rmse_y = using_rmse_y and np.any([x in self.rmse_df.index.get_level_values(1).unique() for x in [objective,'RMSE','score']])
    if hasattr(self,'rmse_df'): rmse_df = self.rmse_df.copy().astype(float)
    if not using_rmse_y:
        outer = self.multi_scenario_results_formatted.copy()
    if commodity!=None and hasattr(self,'rmse_df'): rmse_df = rmse_df.loc[idx[commodity,:],:]
    if not using_rmse_y and commodity!=None:
        outer = outer.loc[idx[commodity,:],:]
    if using_rmse_y:
        X_df = rmse_df.copy()
        r = [r for r in X_df.index.get_level_values(1) if np.any([j in r for j in ['R2','RMSE','region_specific_price_response','score','Region specific']])]
        if len(r)>0:
            X_df = X_df.drop(r,level=1)
        X_df = X_df.unstack().stack(level=0)
    else:
        if not hasattr(self,'processing_option'):
            X_df = self.multi_scenario_hyperparam_formatted.copy()
            if commodity!=None: X_df = X_df.loc[idx[commodity,:],:]
            X_df = X_df.loc[[type(i) not in [str,np.ndarray,bool] and not np.isnan(i) for i in X_df.iloc[:,0].values]]
            X_df = X_df.unstack(1)
            params_changing = X_df.loc[abs(X_df.subtract(X_df.loc[:,idx[0,1,0]],axis=0).sum(axis=1))>1e-12].index.get_level_values(1).unique()
            X_df = X_df.loc[idx[:,params_changing],:].unstack().stack(2).stack()
            X_df = X_df.stack(0).stack().unstack(2)
            X_df = X_df.loc[:,[i for i in X_df.columns if 'RMSE' not in i and 'R2' not in i and i!='score']]
            X_df = X_df.loc[idx[:,:,[j for j in X_df.index.get_level_values(2).unique() if j!=0],:],:]
        else:
            X_df = self.multi_scenario_hyperparam.copy()
            X_df = X_df.stack().droplevel(1).unstack()
            X_df = X_df.loc[[type(i) not in [str,np.ndarray,bool] and not np.isnan(i) for i in X_df.iloc[:,0].values]]
            params_changing = X_df.loc[abs(X_df.subtract(X_df.iloc[:,0],axis=0).sum(axis=1))>1e-12].index.get_level_values(1).unique()
            params_changing = [i for i in params_changing if np.all([j not in i for j in ['R2','RMSE','score']])]
            X_df = X_df.loc[idx[:,params_changing],:]
            X_df = X_df.stack().unstack(1)
            X_df = X_df.sort_index()
            if commodity!=None: X_df = X_df.loc[idx[commodity,:],:]

    if using_rmse_y and 'score' in rmse_df.index.get_level_values(1):
        y_df = rmse_df.loc[idx[:,'score'],:].unstack().stack(level=0)
    elif using_rmse_y:
        y_df = np.log(rmse_df.loc[idx[:,'RMSE'],:].unstack().stack(level=0))
    elif objective in self.multi_scenario_results_formatted.columns:
        if not hasattr(self, 'processing_option'):
            y_df = outer[objective]
            y_df = y_df.unstack(3).unstack().dropna(axis=1,how='all')
            y_df = y_df.subtract(y_df.loc[:,idx[0,1]],axis=0)
            y_df = y_df.groupby(level=[0,1]).mean()
            y_df = y_df.loc[:,(y_df!=0).any()]
            y_df = y_df.stack(0).stack()
            if type(y_df)==pd.core.series.Series:
                y_df = pd.DataFrame(y_df)
            y_df = y_df.rename(columns={y_df.columns[0]:objective})
        elif self.processing_option=='2040 ratio 2018 hist':
            y_df = self.multi_scenario_results[objective].droplevel(1)
            y_df = y_df.loc[idx[:,:,2040]]
            ph = self.objective_results_map
            results_objective_map = dict(zip(ph.values(),ph.keys()))
            y_df = y_df.unstack()
            y_df = y_df.divide(
                self.historical_data[results_objective_map['Mine production']].loc[idx[:,2018]],
                axis=0).stack()
            if type(y_df)==pd.core.series.Series:
                y_df = pd.DataFrame(y_df)
            y_df = y_df.rename(columns={y_df.columns[0]:'Mine production'})
            y_df = y_df.sort_index()
            if commodity!=None: y_df = y_df.loc[idx[commodity,:],:]
    else:
        print('Potential objective values:\n')
        print(rmse_df.index.get_level_values(1).unique(),'\n')
        print(self.multi_scenario_results_formatted.columns)
        raise ValueError('cannot load y_df objective')

    self.y_df_prescale = y_df.copy()
    self.X_df_prescale = X_df.copy()
    if standard_scaler:
        scaler = StandardScaler()
        x_std = scaler.fit_transform(X_df.values)
        X_df = pd.DataFrame(x_std,X_df.index,X_df.columns)
        y_std = scaler.fit_transform(y_df.values.reshape(-1,1))
        y_df = pd.DataFrame(y_std,y_df.index,y_df.columns)
    self.y_df = y_df.copy()
    self.X_df = X_df.copy()


def feature_importance(self, plot=None, recalculate=False, standard_scaler=True, plot_commodity_importances=False,
                       commodity=None, objective=None, processing_option=None, dpi=50):
    '''
    Calculates feature importances using three different tree-based machine
    learning algorithms: RandomForestRegressor, ExtraTreesRegressor, and
    GradientBoostingRegressor. Takes the mean of all three. Creates new
    variable in self called importances_df that includes the outcomes of this
    method.

    self: Many() instance with rmse_df object or
        multi_scenario_results_formatted object if objective!=None.
    plot: None, bool, or str. if True or None, plots using the
        plot_feature_importances method. If str `both`, plots
        using plot_train_test and plot_feature_importances variables True.
        Should have made these into separate methods but here we are.
        False means no plotting. (recalculate has to be True for
        plot_train_test to work)
        - plot_train_test plots the predicted vs actual scores for the test
          set, for each ML tree method, and shows the associated regression
          characteristics.
        - plot_feature_importances plots the bar plot of feature importances
          with bars for each regression method, with and without dummy variables
          included (dummy variables should only make a difference if we are
          doing all commodities simultaneously)
    recalculate: bool, whether to recalculate feature importance, only set
        False if this has already been run and you want to plot things.
    standard_scaler: bool, whether to use standard scaler to rescale data so it
        is N(0,1), which we should probably always do.
    plot_commodity_importances: bool, whether to include the importance of each
        commodity in the `with dummies` subplot for plot_feature_importances
    commodity: None or str. If str, has to be lowercase commodity name, in which
        case this gets the right commodity from the corresponding dataframe. If
        None, uses the full dataframe, going across all commodities
    objective: None or str. If str, has to be one of the columns in
        multi_scenario_results_formatted, otherwise will be using `score` or
        `RMSE` from rmse_df dataframe depending on whether using and Integration
        or demandModel formulation
    processing_option: None or str, used to determine which method we use to
        set up y_df in get_X_df_y_df function, particularly for criticality
        evaluation. Options are:
        - `2040 ratio 2018 hist`: y_df will contain the ratio between the
           simulated 2040 value and the historical 2018 value, for the given
           objective, provided the objective has a corresponding historical
           value
    dpi: float, dots per inch, controls figure resolution.
    '''

    if processing_option is not None:
        self.processing_option = processing_option

    if hasattr(self, 'objective'):
        objective = self.objective
    elif objective is None:
        objective = 'score'
    self.objective = objective

    split_frac = 0.5

    if plot == None or (type(plot) == bool and plot):
        plot_train_test = False
        plot_feature_importances = True
    elif type(plot) == bool and not plot:
        plot_train_test = False
        plot_feature_importances = False
    elif type(plot) == str and plot == 'both':
        plot_train_test = True
        plot_feature_importances = True

    get_X_df_y_df(self, commodity=commodity, objective=objective, standard_scaler=standard_scaler)

    if not hasattr(self, 'importances_df') or recalculate or plot_commodity_importances:
        self.importance_test = None
        importances_df = pd.DataFrame()
        importance_r2 = pd.DataFrame()
        for Regressor, name in zip([RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor],
                                   ['RandomForest', 'ExtraTrees', 'GradientBoosting']):
            if plot_train_test: fig2, ax2 = easy_subplots(2, dpi=dpi)
            for e, dummies in enumerate([True, False]):
                for quantile in [0.95, 1]:
                    X_df, y_df = self.X_df.copy().fillna(0), self.y_df.copy()

                    if dummies:
                        X_df.loc[:, 'commodity ='] = X_df.index.get_level_values(0)
                        X_df = pd.get_dummies(X_df, columns=['commodity ='])

                    y_df = y_df.loc[y_df['score'] < y_df['score'].quantile(quantile)]
                    X_df = X_df.loc[y_df.index]

                    X = X_df.reset_index(drop=True).values
                    y = y_df.values.flatten()

                    X_train, X_test, y_train, y_test, ind_train, ind_test = train_test_split(X,
                                                                                             y,
                                                                                             y_df.index,
                                                                                             test_size=0.333,
                                                                                             random_state=42)
                    X_df_test = X_df.loc[ind_test]
                    y_df_test = y_df.loc[ind_test]

                    regr = Regressor(random_state=0, n_estimators=200)
                    regr.fit(X_train, y_train)

                    test = pd.concat([X_df_test, y_df_test], axis=1)
                    if self.importance_test is None:
                        self.importance_test = test.copy()
                    test.loc[:, 'predicted ' + objective] = regr.predict(X_test)
                    dumm_name = 'w/ dummies' if dummies else 'no dummies'
                    self.importance_test.loc[:, 'predicted ' + objective + ' ' + name + ' ' + dumm_name] = test[
                        'predicted ' + objective]

                    if not plot_commodity_importances:
                        importances = pd.Series(regr.feature_importances_, X_df.columns).drop(
                            [i for i in X_df.columns if 'commodity =' in i]).sort_values(ascending=False)
                    else:
                        importances = pd.Series(regr.feature_importances_, X_df.columns).sort_values(ascending=False)
                    importances.name = name + (' w/ dummies' if dummies else ' no dummies') + f', {quantile} quantile'

                    r2_name = importances.name.replace(f', {quantile} quantile', '')
                    importance_r2.loc[r2_name, quantile] = r2_score(y_test, regr.predict(X_test))

                    if dummies:
                        importances /= importances.sum()
                        self.X_df_dummies = X_df.copy()
                        self.y_df_dummies = y_df.copy()
                    else:
                        self.X_df_no_dummies = X_df.copy()
                        self.y_df_no_dummies = y_df.copy()
                    importances_df = pd.concat([importances_df, importances], axis=1)

                    if plot_train_test:
                        if objective is None and 'RMSE' in test.columns:
                            target = 'RMSE'
                        elif objective is None:
                            target = 'score'
                        else:
                            target = objective
                        do_a_regress(test[target], test['predicted ' + objective], ax=ax2[e])
                        ax2[e].set(title=importances.name.replace(' w/', '\nw/').replace(' no', '\nno'))

        # adding columns to importances_df that don't have the quantile label, taking best of quantiles
        for dummies in [True, False]:
            for name in importance_r2.index:
                quantile = importance_r2.loc[name].idxmax()
                quantile = quantile if quantile != 1 else int(quantile)
                df_name = name + f', {quantile} quantile'
                importances_df.loc[:, name] = importances_df[df_name]

        dummy_cols = [i for i in importances_df.columns if 'w/ dummies' in i and 'quantile' not in i]
        no_dummy_cols = [i for i in importances_df.columns if 'no dummies' in i and 'quantile' not in i]
        importances_df.loc[:, 'Mean w/ dummies'] = importances_df[dummy_cols].mean(axis=1)
        importances_df.loc[:, 'Mean w/ dummies'] /= importances_df['Mean w/ dummies'].sum()
        importances_df.loc[:, 'Mean no dummies'] = importances_df[no_dummy_cols].mean(axis=1)
        importances_df.loc[:, 'Mean no dummies'] /= importances_df['Mean no dummies'].sum()
        dummy_cols += ['Mean w/ dummies']
        no_dummy_cols += ['Mean no dummies']

        importances_df.loc[:, 'Best test R2'] = importances_df[importance_r2.max(axis=1).idxmax()]
        self.importances_df = importances_df.copy()
        self.importance_r2 = importance_r2.copy()

    dummy_cols = [i for i in self.importances_df.columns if 'w/ dummies' in i]
    no_dummy_cols = [i for i in self.importances_df.columns if 'no dummies' in i]

    if plot_feature_importances:
        to_plot_du = self.importances_df.loc[:, dummy_cols].sort_values(by='Mean w/ dummies', ascending=False)
        to_plot_du.rename(columns=dict(zip(dummy_cols, [i.split(' w/ dummies')[0] for i in dummy_cols])), inplace=True)
        to_plot_no = self.importances_df.loc[:, no_dummy_cols].sort_values(by='Mean no dummies',
                                                                           ascending=False).dropna()
        to_plot_no.rename(columns=dict(zip(no_dummy_cols, [i.split(' no dummies')[0] for i in no_dummy_cols])),
                          inplace=True)
        to_plot_du.rename(make_parameter_names_nice(to_plot_du.index), inplace=True)
        to_plot_no.rename(make_parameter_names_nice(to_plot_no.index), inplace=True)
        height_scale = 2 if plot_commodity_importances and 'Mine cost reduction per year' in to_plot_no.index else 1.7 if plot_commodity_importances else 1.5
        fig1, ax1 = easy_subplots(2, height_scale=height_scale, width_scale=9 / len(self.importances_df.columns),
                                  dpi=dpi,
                                  width_ratios=(to_plot_du.shape[0], to_plot_no.shape[0]))
        to_plot_du.plot.bar(ax=ax1[0], ylabel='Feature importance').grid(axis='x')
        to_plot_no.plot.bar(ax=ax1[1], ylabel='Feature importance', legend=not plot_commodity_importances).grid(
            axis='x')
        ax1[0].set_title('With commodity dummies', weight='bold')
        ax1[1].set_title('No dummies', weight='bold')
        y1a, y1b = ax1[0].get_ylim()
        y2a, y2b = ax1[1].get_ylim()
        ya = min(y1a, y2a)
        yb = max(y1b, y2b)
        ax1[0].set(ylim=(ya, yb))
        ax1[1].set(ylim=(ya, yb))
        if plot_commodity_importances:
            ax1[1].set_yticklabels([])
            ax1[1].set_ylabel(None)
            ax1[1].yaxis.set_tick_params(width=0)
            fig1.tight_layout(pad=0.5)
        plt.show()
        plt.close()
        return fig1, ax1


def nice_feature_importance_plot(self, dpi=50):
    """
    For pre-tuning only, so have to have self.mining and self.demand loaded.
    Plots the demand parameter feature importance and mining parameter
    feature importance in separate subplots, for each regression method and the
    mean.
    """
    if not hasattr(self,'mining'):
        self.get_multiple()
    if not hasattr(self.mining,'importances_df') or np.any(['commodity' in i.lower() for i in self.mining.importances_df.index]):
        feature_importance(self.mining,plot=False,objective='score')
    if not hasattr(self.demand,'importances_df') or np.any(['commodity' in i.lower() for i in self.demand.importances_df.index]):
        feature_importance(self.demand,plot=False,objective='RMSE')

    no_dummy_cols = [i for i in self.mining.importances_df.columns if 'no dummies' in i]
    to_plot_demand = self.demand.importances_df.loc[:,no_dummy_cols].sort_values(by='Mean no dummies',ascending=False).dropna()
    to_plot_demand.rename(columns=dict(zip(no_dummy_cols,[i.split(' no dummies')[0] for i in no_dummy_cols])),inplace=True)
    to_plot_demand.rename(make_parameter_names_nice(to_plot_demand.index),inplace=True)

    to_plot_mining = self.mining.importances_df.loc[:,no_dummy_cols].sort_values(by='Mean no dummies',ascending=False).dropna()
    to_plot_mining.rename(columns=dict(zip(no_dummy_cols,[i.split(' no dummies')[0] for i in no_dummy_cols])),inplace=True)
    to_plot_mining.rename(make_parameter_names_nice(to_plot_mining.index),inplace=True)

    fig1,ax1 = easy_subplots(2,width_scale=1.4,height_scale=2,width_ratios=(to_plot_demand.shape[0],to_plot_mining.shape[0]),dpi=dpi)

    to_plot_demand.plot.bar(ax=ax1[0],ylabel='Feature importance',legend=False).grid(axis='x')
    ax1[0].set_title('Demand pre-tuning',weight='bold')

    to_plot_mining.plot.bar(ax=ax1[1]).grid(axis='x')
    ax1[1].set_title('Mining pre-tuning parameter importance',weight='bold')
    ax1[1].set_yticklabels([])
    ax1[1].yaxis.set_tick_params(width=0)

    y1a, y1b = ax1[0].get_ylim()
    y2a, y2b = ax1[1].get_ylim()
    ya = min(y1a,y2a)
    yb = max(y1b,y2b)
    ax1[0].set(ylim=(ya,yb))
    ax1[1].set(ylim=(ya,yb))

    xa = ax1[0].get_xlim()
    ax1[0].hlines(0.05, xa[0], xa[1],zorder=0,color='lightgray',linestyle='--',linewidth=3)
    xb = ax1[1].get_xlim()
    ax1[1].hlines(0.05, xb[0], xb[1],zorder=0,color='lightgray',linestyle='--',linewidth=3)

    fig1.tight_layout()
    plt.show()
    plt.close()
    return fig1, ax1

def commodity_level_feature_importance(self, dpi=50):
    """
    Plots the weird line plot showing feature importance for each commodity
    for each parameter. This one could use some updating, but the general format
    is what I imagined for the spider plot or something similar.
    """
    dpi=50
    df = pd.DataFrame()
    for i in self.rmse_df.index.get_level_values(0).unique():
        feature_importance(self,recalculate=True,commodity=i,plot=False)
        series = self.importances_df['Mean no dummies']
        series.name=i
        df = pd.concat([df,series],axis=1)

    fig,ax = easy_subplots(1,dpi=dpi)
    df.rename(make_parameter_names_nice(df.index)).T.plot(colormap='viridis',ax=ax[0]).grid(axis='x')
    ax[0].legend(loc=(1.1,0))
    ax[0].set(ylabel='Feature importance')
    self.commodity_importances_df = df.copy()

    plt.show()
    plt.close()
    return fig, ax

def plot_all_feature_importance_plots(self, dpi=50, plot_feature_importance_no_show_dummies=True, plot_feature_importance_show_dummies=True, plot_mining_and_demand_importance=True, plot_commodity_level_importances=True, commodity=None):
    """
    Does the plots for feature importance for all sub-instances of Many (mining,
    integ, demand).

    dpi: float, dots per inch, controls figure resolution.
    plot_feature_importance_no_show_dummies: bool, whether to do the bar plot of
        feature importances, not showing dummy variables
    plot_feature_importance_show_dummies: bool, whether to do the bar plot of
        feature importances, showing dummy variables
    plot_mining_and_demand_importance: bool, whether to run the
        nice_feature_importance_plot function for each commodity
    plot_commodity_level_importances: bool, whether to do the
        commodity_level_feature_importance funky lineplot
    commodity: None or str, if None does all commodities at the same time,
        otherwise need to give commodity in the lowercase form seen in rmse_df
    """
    self.feature_plot_figs = []
    self.feature_plot_axs = []
    ss = []
    for s in ['mining','demand','integ']:
        if hasattr(self,s): ss+=[getattr(self,s)]
    if plot_feature_importance_no_show_dummies:
        for s in ss:
            fig,ax = feature_importance(s,dpi=dpi,recalculate=True,commodity=commodity)
            self.feature_plot_figs += [fig]
            self.feature_plot_axs += [ax]

    if plot_mining_and_demand_importance:
        if not plot_feature_importance_no_show_dummies:
            for s in ss:
                feature_importance(s,dpi=dpi,recalculate=True, plot=False)
        fig,ax = nice_feature_importance_plot(self, dpi=dpi)
        self.feature_plot_figs += [fig]
        self.feature_plot_axs += [ax]

    if plot_feature_importance_show_dummies:
        for s in ss:
            fig,ax = feature_importance(s,dpi=dpi,plot_commodity_importances=True)
            self.feature_plot_figs += [fig]
            self.feature_plot_axs += [ax]
    if plot_commodity_level_importances:
        for s in ss:
            fig,ax = commodity_level_feature_importance(s,dpi=dpi)
            self.feature_plot_figs += [fig]
            self.feature_plot_axs += [ax]

def make_parameter_names_nice(ind):
    """
    Converts the ugly parameter names used in the actual simulation into nicer,
    plot-friendly names. Takes in an list of ugly parameter names and returns a
    dictionary mapping ugly names to nice names. Particularly useful when using
    the dataframe.rename() function on a pandas dataframe, otherwise you have to
    do:
    names_map = make_parameter_names_nice(ugly_names)
    nice_names = [names_map[i] for i in ugly_names]
    """
    updated = [i.replace('_',' ').replace('sector specific ','').replace('dematerialization tech growth','Intensity elasticity to time').replace(
        'price response','intensity response to price').capitalize().replace('gdp','GDP').replace(
        ' cu ',' CU ').replace(' og ',' OG ').replace('Primary price resources contained elas','Incentive tonnage response to price').replace(
        'OG elas','elasticity to ore grade').replace('Initial','Incentive').replace('Primary oge scale','Ore grade elasticity to COT distribution mean').replace(
        'Mine CU margin elas','Mine CU elasticity to TCM').replace('Mine cost tech improvements','Mine cost reduction per year').replace(
        'Incentive opening probability','Fraction of viable mines that open').replace('Mine cost price elas','Mine cost elasticity to commodity price').replace(
        'Close years back','Prior years used for price prediction').replace('Reserves ratio price lag','Price lag used for incentive pool tonnage').replace(
        'Incentive ore grade decline','Incentive ore grade elasticity to COT').replace('response','elasticity').replace('Tcrc','TCRC').replace('tcrc','TCRC').replace(
        'sd','SD').replace(' elas ',' elasticity to ').replace('Direct melt elas','Direct melt fraction elas').replace('CU price elas','CU elasticity to price').replace(
        'ratio TCRC elas','ratio elasticity to TCRC').replace('ratio scrap spread elas','ratio elasticity to scrap spread').replace(
        'Refinery capacity fraction increase mining','Ref. cap. growth frac. from mine prod. growth').replace('Pri ','Primary refinery ').replace(
        'Sec CU','Secondary refinery CU').replace('Sec ratio','Refinery SR').replace('primary commodity ','').replace('Primary commodity','Refined').replace(
        'tcm','TCM').replace(' sr',' SR').replace('CU TCRC elas','CU elasticity to TCRC').replace(' dist ',' fraction in ')
                                       for i in ind]
    return dict(zip(ind,updated))

def prep_for_snsplots(self,demand_mining_integ='demand',percentile=25,n_most_important=4):
    """
    Reformats data for use in the seaborn plotting, used in the functions
    plot_demand_parameter_correlation &  plot_important_parameter_scatter.

    For demand_mining_integ in ['mining','integ'], uses
    self.mining.importances_df or self.integ.importances_df to get feature
    importance scores, which allows us to down-select to the n_most_important
    features in the resulting dataframe. Uses the `Mean no dummies` column of
    importances_df

    For demand_mining_integ=='demand', does not do feature importance since
    there are only three pre-tuning parameters.

    self: Many() instance, with feature_importance(self) having already been run
        such that importances_df object exists.
    demand_mining_integ: str in the set [`demand`,`mining`,`integ`], determines
        the object of self used (assumes self was initialized with get_multiple)
    percentile: float, percentage of scenarios to use, ranked with lowest RMSE
        combined score.
    n_most_important: int, number of most important parameters to include
        in the resulting dataframe.

    Returns:
        - df: dataframe in seaborn plot format ready to go
        - demand_params: list of most important parameters
        - most_important: list of most important parameters as well?
          only returned for `integ` or `mining`, unsure if there is a difference
    """
    if demand_mining_integ=='mining' and hasattr(self,'mining'):
        df = self.mining.rmse_df_sorted.copy()
        most_important = self.mining.importances_df['Mean no dummies'].sort_values(ascending=False)
        if type(n_most_important)!=int:
            most_important = most_important.index[n_most_important[0]:n_most_important[1]]
        elif n_most_important>0:
            most_important = most_important.head(n_most_important).index
        else:
            most_important = most_important.tail(-n_most_important).index
        df = df.loc[idx[:,most_important],:].copy()
    elif demand_mining_integ=='integ' and hasattr(self,'integ'):
        df = self.integ.rmse_df_sorted.copy()
        most_important = self.integ.importances_df['Mean no dummies'].sort_values(ascending=False)
        if type(n_most_important)!=int:
            most_important = most_important.index[n_most_important[0]:n_most_important[1]]
        elif n_most_important>0:
            most_important = most_important.head(n_most_important).index
        else:
            most_important = most_important.tail(-n_most_important).index
        df = df.loc[idx[:,most_important],:].copy()
    elif demand_mining_integ=='demand' and hasattr(self,'demand'):
        df = self.demand.rmse_df_sorted.copy()
    else:
        df = self.rmse_df_sorted.copy()
        if demand_mining_integ=='integ':
            most_important = self.importances_df['Mean no dummies'].sort_values(ascending=False)
            if type(n_most_important)!=int:
                most_important = most_important.index[n_most_important[0]:n_most_important[1]]
            elif n_most_important>0:
                most_important = most_important.head(n_most_important).index
            else:
                most_important = most_important.tail(-n_most_important).index
            df = df.loc[idx[:,most_important],:].copy()
    percentile_converted = int(percentile/100*df.shape[1])
    df.rename(dict(zip(df.index.levels[0],[i.capitalize() for i in df.index.levels[0]])),inplace=True,level=0)
    for i in ['RMSE','R2']:
        if i in df.index.get_level_values(1).unique():
            df = df.drop(i,level=1)
    df = df.loc[:,:percentile_converted].stack().unstack(1).reset_index().rename(columns={'level_0':'Commodity','level_1':'Scenario number'})
    df.rename(columns=make_parameter_names_nice(df.columns),inplace=True)
    if 'Region specific intensity elasticity to price' in df.columns:
        df.drop(columns=['Region specific intensity elasticity to price'],inplace=True)
    demand_params = [i for i in df.columns if i not in ['Commodity','Scenario number','Region specific intensity response to price']]
    df.loc[:,demand_params] = df[demand_params].astype(float)
    if demand_mining_integ in ['mining','integ']:
        return df, demand_params, most_important
    else:
        return df, demand_params

def plot_demand_parameter_correlation(self,scatter=True, percentile=25, n=None, dpi=50, demand_or_integ='integ'):
    """
    Plots the three combinations of demand pre-tuning parameters against each other to show
    clustering for the best n scenarios.

    self: Many() instance, with feature_importance(self) having already been run
        which happens if you loaded using self.get_multiple()
    scatter: bool, whether to plot as a scatter plot, or if False, as a set of
        kernel density distributions
    percentile: float, percentage of total scenarios to plot, so if there are
        100 scenarios (as there typically are for demand pre-tuning), setting
        percentile=25 would plot the 25 best-fitting scenarios
    dpi: float, dots per inch, controls figure resolution
    """
    if n!=None: percentile=n/600*100
    if demand_or_integ=='demand':
        df,demand_params = prep_for_snsplots(self,demand_mining_integ='demand',percentile=percentile)
    else:
        df,_,_ = prep_for_snsplots(self,demand_mining_integ='integ',percentile=percentile,
                                   n_most_important=self.integ.importances_df.shape[0])
        demand_params = ['Intensity elasticity to GDP', 'Intensity elasticity to time', 'Intensity elasticity to price']
#         df = df.set_index(['Commodity','Scenario number','Parameter']).loc[idx[:,:,demand_params]]
#         df = df.unstack().reset_index(drop=False)
    self.df=df
    combos = list(combinations(demand_params,2))
    for pair in combos:
        if scatter:
            g = sns.jointplot(data=df,x=pair[0],y=pair[1],hue='Commodity',height=8)
            g.ax_joint.legend(loc=(1.17,0),title='Commodity')
        else:
            plt.figure(figsize=(8,8))
            g = sns.kdeplot(data=df,x=pair[0],y=pair[1],hue='Commodity',fill=False)
            sns.move_legend(g,loc=(1.1,0))
        g.figure.set_dpi(dpi)
    plt.show()
    plt.close()
    return g.fig, g.ax_joint

def plot_important_parameter_scatter(self, mining_or_integ='mining', percentile=25, n=None, n_most_important=4, scale_y_for_legend=1, plot_median=True, best_or_median='mean', legend=True, scale_fig_width=1, scale_fig_height=1, split_params=True, normalize=False, x_for_stat_sig=False, plot=True, dpi=50):

    """
    Plots the vertical seaborn stripplot of the most important parameters
    for either the mining pre-tuning or the full integration tuning.
    Automatically splits out any parameters that have values outside the [0,1]
    range and puts them on a separate subplot.

    self: Many() instance, with feature_importance(self) having already been run
        which happens if you loaded using self.get_multiple()
    percentile: float, selects the number of parameter sets to include, so
      percentile=25 would take the best-fitting 25% of parameter sets
    n: int, overrides percentile to allow you to select the number of best
      scenarios to plot directly.
    n_most_important: int, number of most important parameters to plot
    scale_y_for_legend: float, value to scale the upper y-limit to make room
      for the legend
    plot_median: bool, whether or not to plot the median or best parameter value
      as a semi-transparent gray square
    best_or_median: str, can be either `best` or `median`, such that if
      plot_median is True, either the best-fitting parameter set is plotted or
      just the median of the selected best-fitting parameter set.
    legend: bool, whether to plot the legend on the plot
    scale_fig_width: float, can be used to widen or narrow the plot, default 1
    split_params: bool, whether to split different parameters to plot on
        different subplot
    normalize: bool, whether to divide each parameter by the larger of its
        absolute value max and min.
    x_for_stat_sig: bool, whether to plot X`s at the y value 1 to show
        the parameter as statistically significantly different from zero
    dpi: int, dots per inch, figure resolution (higher = better)
    """
    if n!=None: percentile=n/600*100
    mining_or_integ_ph = mining_or_integ
    if mining_or_integ=='demand':
        n_most_important=self.integ.importances_df.shape[0]
        mining_or_integ_ph = 'integ'
    df2, demand_params, order = prep_for_snsplots(self,demand_mining_integ=mining_or_integ_ph,
                                                  percentile=percentile, n_most_important=n_most_important)
    if type(n_most_important)!=int:
        n_most_important = abs(n_most_important[1]-n_most_important[0])
    df2 = df2.set_index(['Commodity','Scenario number']).stack().reset_index(drop=False).rename(
        columns={'level_2':'Parameter',0:'Value'})
    if mining_or_integ=='demand':
        demand_params = ['Intensity elasticity to GDP', 'Intensity elasticity to time', 'Mine cost change per year']
        df2 = df2.loc[[i in demand_params for i in df2['Parameter']]]

    if normalize:
#         n = df2.groupby(['Commodity','Parameter']).apply(lambda x: max(abs(max(x['Value'])),abs(min(x['Value']))))
#         df2.loc[:,'Value'] = df2.apply(lambda x: x['Value']/n.loc[idx[x['Commodity'],x['Parameter']]],axis=1)
        df2.loc[(df2['Parameter']=='Incentive mine cost change per year')|(df2['Parameter']=='Mine cost change per year'),'Value']/=5
        df2.loc[(df2['Parameter']=='Intensity elasticity to time'),'Value']*=10
#         df2.loc[df2['Parameter']=='Incentive mine cost change per year','Parameter'] = r'$\frac{Incentive\:mine\:cost\:change\:per\:year}{5}$'
#         df2.loc[df2['Parameter']=='Mine cost change per year','Parameter'] = r'$\frac{Mine\:cost\:change\:per\:year}{5}$'
#         df2.loc[df2['Parameter']=='Intensity elasticity to time','Parameter'] = '10 x Intensity elasticity to time'

    else:
        df2.loc[df2['Parameter']=='Mine cost reduction per year','Value'] /= 10
         # df2a = df2.copy().loc[df2['Parameter']!='Mine cost reduction per year']
    order = df2['Parameter'].unique()
    if split_params:
        outer = df2.loc[((df2['Value']<0)|(df2['Value']>1))]['Parameter'].unique()
        if len(outer)==len(order):
            outer=[]
    else:
        outer = df2.loc[((df2['Value']<0)|(df2['Value']>1))&(df2['Value']==np.inf)]['Parameter'].unique()
    df2a = df2.copy().loc[[i not in outer for i in df2['Parameter']]]

    def replace_for_mining(string):
        return string.replace(' to T','\nto T').replace('y d','y\nd').replace('ing pr','ing\npr')\
            .replace('ge e','ge\ne')
#     if mining_or_integ=='mining':
#         for i in demand_params:
#             df2a.replace(i,replace_for_mining(i),inplace=True)
    df2b = df2.copy().loc[[i in outer for i in df2['Parameter']]]
    if best_or_median=='median':
        df2a_means = df2a.groupby(['Commodity','Parameter']).median().reset_index(drop=False)
        df2b_means = df2b.groupby(['Commodity','Parameter']).median().reset_index(drop=False)
    else:
        df2a_means = df2a.loc[df2a['Scenario number']==0].reset_index(drop=True)
        df2b_means = df2b.loc[df2b['Scenario number']==0].reset_index(drop=True)

    if x_for_stat_sig:
        df2a_means = df2a_means.set_index(['Commodity','Parameter'])
        stat_sig = df2.loc[[i not in outer for i in df2['Parameter']]].groupby(['Commodity','Parameter']).apply(lambda x: stats.ttest_1samp(x['Value'], popmean=0)[1]<0.05)
#         stat_sig[~stat_sig]=np.nan
        df2a_means.loc[:,'Statistically significant'] = stat_sig
        df2a_means = df2a_means.reset_index(drop=False)
        if len(outer)>0:
            df2b_means = df2b_means.set_index(['Commodity','Parameter'])
            df2b_means.loc[:,'Statistically significant'] = df2.loc[[i in outer for i in df2['Parameter']]].groupby(['Commodity','Parameter']).apply(lambda x: stats.ttest_1samp(x['Value'], popmean=0)[1]<0.05)
            df2b_means = df2b_means.reset_index(drop=False)

    self.df2 = df2
    self.df2a = df2a
    self.df2b = df2b
    self.df2a_means = df2a_means
    self.df2b_means = df2b_means
    if len(outer)>0:
        fig,ax=easy_subplots(2,width_scale=scale_fig_width*1.3+0.1*n_most_important/4,
                             width_ratios=[n_most_important-len(outer),len(outer)],
                             height_scale=scale_fig_height, dpi=dpi)
    else:
        fig,ax=easy_subplots(1,width_scale=scale_fig_width*1.5+0.1*n_most_important/4,
                             height_scale=scale_fig_height, dpi=dpi)
    if plot:
        a=ax[0]
        # sns.violinplot(data=df2a, x='Parameter', y='Value', hue='Commodity',ax=a, linewidth=2)
        order_rename = make_parameter_names_nice(order)
        order = [order_rename[i] for i in order]
        if mining_or_integ=='mining':
            order = [replace_for_mining(i) for i in order if i not in outer]
        linewidth = 0.5
        order_a = [i for i in order if i not in outer]
        order_b = [i for i in order if i in outer]

        orient='v'
        if orient=='v':
            kwd_args = {'x':'Parameter', 'y':'Value', 'hue':'Commodity', 'dodge':True,
                        'linewidth':linewidth, 'orient':orient}
        else:
            kwd_args = {'y':'Parameter', 'x':'Value', 'hue':'Commodity', 'dodge':True,
                        'linewidth':linewidth, 'orient':orient}
        sns.stripplot(data=df2a, ax=a, size=10, edgecolor='w',order=order_a, **kwd_args)
        if plot_median:
            marker='s'
            markersize=12
            alpha=0.3
            sns.stripplot(data=df2a_means,ax=a,
                          size=markersize, palette='dark:k', alpha=alpha, marker=marker,
                          edgecolors='k',order=order_a,**kwd_args)
        if x_for_stat_sig:
            alt_kwd_args = dict(pd.Series(kwd_args).drop(['y','linewidth']))
            sns.stripplot(data=df2a_means,ax=a,y='Statistically significant',linewidth=3,
                          size=markersize, palette='dark:k', alpha=alpha, marker='x',
                          edgecolors='k',order=order_a,**alt_kwd_args)
        h,l = a.get_legend_handles_labels()
        if plot_median:
            n_commodities = len(df2a.Commodity.unique())
            square_h, square_l = Line2D([0],[0],marker=marker,color='w',alpha=alpha+0.1,markersize=10,
                                        markerfacecolor='k', markeredgecolor='k'),best_or_median.capitalize()
            h = h[:n_commodities]
            l = l[:n_commodities]
        ncol=1
        h_update = list(np.concatenate([h[i::ncol] for i in np.arange(0,ncol)]))
        l_update = list(np.concatenate([l[i::ncol] for i in np.arange(0,ncol)]))
        if plot_median:
            h_update += [square_h]
            l_update += [square_l]
        if legend:
            a.legend(ncol=ncol,handles=h_update, labels=l_update, frameon=True, columnspacing=0.2,
                     handletextpad=0.1, borderpad=0.5, labelspacing=0.1)
        else:
            a.legend('')
        a.tick_params(axis='x',rotation=90)

        alim = a.get_ylim()
        a.set(xlabel=None, ylim=(alim[0],alim[1]*1.07*(scale_y_for_legend)))
        if mining_or_integ=='mining': title_string='Mine pre-tuning'
        elif mining_or_integ=='integ': title_string='Integration tuning'
        elif mining_or_integ=='demand': title_string='Demand parameter tuning'
        if len(outer)>0:
            b=ax[1]
            sns.stripplot(data=df2b,ax=b, size=10, edgecolor='w', order=order_b, **kwd_args)
            if plot_median:
                sns.stripplot(data=df2b_means, ax=b, size=markersize, palette='dark:k', alpha=alpha, marker=marker, order=order_b,
                              **kwd_args)
            if x_for_stat_sig:
                alt_kwd_args = dict(pd.Series(kwd_args).drop(['y','linewidth']))
                sns.stripplot(data=df2b_means,ax=b,y='Statistically significant',linewidth=3,
                              size=markersize, palette='dark:k', alpha=alpha, marker='x',
                              edgecolors='k',order=order_b,**alt_kwd_args)
            b.legend('')
            alim = a.get_ylim()
            scale = np.floor(df2b['Value'].min()) if df2b['Value'].min()<0 else np.ceil(df2b['Value'].max())
            if df2b['Value'].min()<0:
                b.set(ylim=[alim[0]+scale,alim[1]-1], ylabel=None, xlabel=None)
            else:
                b.set(ylim=[alim[0]*scale,alim[1]*scale], ylabel=None, xlabel=None)
            b.tick_params(axis='x',rotation=90)

        fig.suptitle(title_string+' parameter results',weight='bold')
        fig.tight_layout(pad=0.8)
        plt.show()
        plt.close()
    else:
        plt.close()

    return fig,ax,df2


def commodity_level_feature_importance_heatmap(self, dpi=50, recalculate=True, objective=None, only_plot_names=False,
                                               normalize=False, importances_df_column='Mean no dummies'):
    """
    Creates a plot showing commodity level feature importances in heatmap form.

    dpi: int, dots per inch, controls figure resolution.
    recalculate: bool, whether or not feature importance gets recalculated,
      should keep as True unless you have just run this function with
      recalculate=True and just want to update dpi or other plotting components.
    objective: None or str, string must correspond with a column in the results
      dataframe
    only_plot_names: bool, default False, restricts the variables shown to be
      those defined in the names variable within this function. Does not seem
      useful anymore
    normalize: bool, default False. If True, divides all feature importance
      values in each commodity by the commodity maximum; gives relative importance
      where most important parameter has the value 1, rather than all summing
      to 1
    """
    names = ['Intensity elasticity to GDP',
             'Intensity decline per year',
             'Intensity elasticity to price',
             'Mine CU elasticity to TCM',
             'Fraction of viable mines that open',
             'Ore grade elasticity distribution mean',
             'Mine cost reduction per year',
             'TCRC elasticity to SD',
             'Refinery SR elasticity to scrap spread',
             'Refinery SR elasticity to TCRC',
             'Secondary refinery CU elasticity to price',
             'Primary refinery CU elasticity to price',
             'Scrap spread elasticity to SD',
             'Ref. cap. growth frac. from mine prod. growth',
             'TCRC elasticity to price',
             'Collection elasticity to scrap price',
             'Direct melt fraction elasticity to scrap spread']
    if not hasattr(self, 'importances_df_reformed') or recalculate:
        importances_df = pd.DataFrame()
        importances_predict = pd.DataFrame()
        importance_r2 = pd.DataFrame()
        if hasattr(self, 'rmse_df'):
            outer = self.rmse_df.copy()
        else:
            outer = self.multi_scenario_results.copy()
        for comm in list(outer.index.get_level_values(0).unique()) + [None]:
            feature_importance(self, commodity=comm, recalculate=True, plot=False, objective=objective)
            ph = self.importances_df[importances_df_column]
            ph.name = comm if comm != None else 'None'
            importances_df = pd.concat([importances_df, ph], axis=1)
            ph = pd.concat([self.importance_test], keys=[comm])
            importances_predict = pd.concat([importances_predict, ph])
            ph = pd.concat([self.importance_r2], keys=[comm])
            importance_r2 = pd.concat([importance_r2, ph])

        self.importances_df_reformed = importances_df.rename(
            make_parameter_names_nice(importances_df.index)).rename(
            columns=dict(
                zip(importances_df.columns, [i.capitalize().replace('None', 'All') for i in importances_df.columns])))
        self.importances_predict = importances_predict.copy()
        self.importance_r2 = importance_r2.copy()
        if only_plot_names:
            self.importances_df_reformed = self.importances_df_reformed.loc[
                [i for i in names if i in self.importances_df_reformed.index]]

    fig, ax = easy_subplots(1, height_scale=1.1 * self.importances_df.shape[0] / len(names))
    a = ax[0]
    if normalize:
        self.importances_df_reformed = self.importances_df_reformed.div(self.importances_df_reformed.max())
        cbar_kws = {'label': 'Relative feature importance'}
    else:
        cbar_kws = {'label': 'Feature importance'}
    self.importances_df_reformed.rename(columns=self.commodity_element_map, inplace=True)
    self.importances_df_reformed = self.importances_df_reformed.sort_values(by='All', ascending=False)[[
        'Ag', 'Al', 'Au', 'Cu', 'Ni', 'Pb', 'Sn', 'Steel', 'Zn', 'All']]
    sns.heatmap(self.importances_df_reformed,
                xticklabels=True, yticklabels=True, ax=a, cbar_kws=cbar_kws, cmap='OrRd')
    fig.set_dpi(dpi)
    a.tick_params(axis='x', labelbottom=True, labeltop=True, labelrotation=90)
    a.set_title('Commodity level\nfeature importance', weight='bold')
    return fig, a

def nice_plot_pretuning(demand_or_mining='mining',dpi=50,output_data_folder='output_files',filename_base='_run_hist',filename_modifier=''):
    """
    Nicely plots just the best-fitting simulated and historical mine production
    or demand based on the demand_or_mining input, for each commodity.

    demand_or_mining: str, can be either `mining` or `demand`, determines
      whether the mining or demand pre-tuning result is plotted.
    dpi: int, dots per inch, controls figure resolution
    output_data_folder: str, folder where data is saved
    filename_base: str, base name of file given when running, is the part coming
      after the commodity name and before `_mining` or `_DEM`
    filename_modifier: str, filename modifier, comes after the `_mining` or
      `_DEM` but before `.pkl`
    """
    if demand_or_mining=='demand': demand_or_mining='DEM'
    ready_commodities = ['Steel','Al','Au','Sn','Cu','Ni','Ag','Zn','Pb']
    fig,ax = easy_subplots(ready_commodities,dpi=dpi)
    cmap = {'nickel':'Ni','gold':'Au','aluminum':'Al','tin':'Sn','zinc':'Zn','lead':'Pb','steel':'Steel','copper':'Cu','silver':'Ag'}
    cmap_r=dict(zip(cmap.values(),cmap.keys()))
    for c,a in zip(ready_commodities,ax):
        filename=f'{output_data_folder}/{cmap_r[c]}{filename_base}{filename_modifier}_{demand_or_mining}.pkl'
        big_df = pd.read_pickle(filename)
        rmse_df = big_df.loc['rmse_df'].iloc[-1][0]
        rmse_df.index = pd.MultiIndex.from_tuples(rmse_df.index)
        idx = rmse_df.unstack()['RMSE'].idxmin()
        rmse_df.unstack().loc[idx]
        ieieie = Individual(c,3,filename=filename, rmse_not_mae=True,weight_price=1,dpi=50,price_rolling=5)
        if 'DEM' in filename:
            hist = ieieie.historical_data['Total demand']
            sim = big_df.loc['results',idx]['Total demand']
            hist = pd.concat([sim.loc[:2000],hist.loc[2001:]])
            sim = sim.loc[2001:]
            diction = get_unit(sim,hist,'Total demand (kt)')
            hist, sim, unit = [diction[i] for i in ['historical','simulated','unit']]
            hist.plot(ax=a,label='Historical')
            sim.plot(ax=a,ylabel=f'Total demand ({unit})',label='Simulated').grid(axis='x')
        else:
            hist = ieieie.historical_data['Primary supply']
            sim = big_df.loc['results',idx]['Primary supply']
            hist = pd.concat([sim.loc[:2000],hist.loc[2001:]])
            sim = sim.loc[2001:]
            diction = get_unit(sim,hist,'Primary supply (kt)')
            hist, sim, unit = [diction[i] for i in ['historical','simulated','unit']]
            hist.plot(ax=a,label='Historical')
            sim.plot(ax=a,ylabel=f'Primary supply ({unit})',label='Simulated').grid(axis='x')
        a.set(title=cmap_r[c].capitalize(),xlabel='Year')
        a.legend(loc='upper left')
    fig.tight_layout()
    return fig,ax

def run_future_scenarios(output_data_folder='output_files', user_data_folder='input_files/user_defined',
                         static_data_folder='input_files/static', run_parallel=3, scenario_sheet_file_path=None,
                         supply_or_demand='demand', n_best_scenarios=25, n_per_baseline=25, price_response=True,
                         commodities=None, years_of_increase=np.arange(1,2),scenario_name_base='_run_scenario_set',
                         simulation_time=np.arange(2019,2041), baseline_sampling='clustered',
                         tuned_rmse_df_out_append='', save_mining_info=False, trim_result_df=True,
                         notes='Scenario run!', random_state=None, verbosity=2):
    """
    Runs scrap demand scenarios, for 0.01 to 20% of the market switching from
    refined consumption to scrap consumption, for years given (default is just
    one year).

    Runs n_baselines (default 100) different sets of hyperparameters. These are
    (for baseline_sampling==`grouped`) randomly sampled from a weighted
    sampling-with-replacement of the Bayesian optimization results
    for each commodity, with weighting going as score^(-10), which is
    automatically normalized within the pandas sampling function. This weighted
    sampling with replacement creates distributions for each parameter, and each
    parameter is then sampled independently to form the 100 different sets of
    hyperparameters

    -----------------------------
    Inputs:
    - output_data_folder: str, path to where the tuned_rmse_df_out.pkl file is stored
    - run_parallel: int, 0 to not use parallel function (op_run_future_scenarios)
      and any other number will be used as input for parallelization in
      op_run_future_scenarios_parallel
    - scenario_sheet_file_path: str or None, if str, should be absolute or
      relative file path for a properly formatted scenario sheet excel file.
      See documentation for more details on formatting.
    - supply_or_demand: str, can be `supply`, `demand`, None, for running a scrap
      supply-side or demand-side set of scenarios, or just baseline
    - n_best_scenarios: int, number of different baseline scenarios to run
    - n_per_baseline: int, number of scenarios generated using each baseline
      from n_best_scenarios as a basis.
    - price_response: bool, whether the scrap supply or demand scenario will
      include price response (typically True when scrap demand and False when
      doing scrap supply, but this is not enforced, must be set manually)
    - commodities: list, default is the list in the Many class called
      ready_commodities, which at least right now is ['Steel','Al','Au','Sn',
      'Cu','Ni','Ag','Zn','Pb']. If giving an input for this parameter, should
      be in list form and in the form given in case study data.xlsx (elemental
      except Steel)
    - years_of_increase: np.ndarray of years in which the scrap demand increase
      occurs, with default np.arange(1,2) so just one year.
    - simulation_time: np.ndarray of years in which the scenarios run.
    - scenario_name_base: str, where the resulting filename will be
      output_files/commodity+scenario_name_base+`.pkl`
    - baseline_sampling: str, can be `random`, `grouped`, `clustered`, `actual`
        -- `random`: randomly sampling from the param_samp variable, which
          resamples from the full hyperparameter set weighting by score^(-10),
          with samples independent of each other such that any dependencies
          between hyperparameters are ignored. Takes n_per_baseline samples
          (would run 100).
        -- `grouped`: similar to `clustered` below, but using the best paramters
          based on score alone, and more attempted adjustments such that the
          mean matches the original best scenario value.
        -- `clustered`: more complex, relies on having a train-test setup from
          tuning historical data, such that we select the 4*n_best_scenarios
          based on the train score, then select the n_best_scenarios from that
          set based on the test score. Each of these scenarios is modified to
          produce n_per_baseline scenarios by summing with a N(0,std(param))
          distribution, where std(param) is the standard deviation of the
          hyperparameter within the n_best_scenarios. Total number of scenarios
          is n_best_scenarios*n_per_baseline
        -- `actual`: takes the actual n_best_scenarios (used 10 previously)
          according to the score
    - tuned_rmse_df_out_append: str, default ``. Can add something to this if
      you have saved a differently named tuned_rmse_df_out with some additional
      str appended, and want to use that in baseline selection.
    - save_mining_info: bool, default False, can also take the str `cost_curve`
      to save a subset of parameters that enable cost curve construction
    - trim_result_df: bool, default True
    - notes: str, gets saved in hyperparam of every scenario
    """
    import numpy as np
    import os
    import pandas as pd
    idx = pd.IndexSlice
    import warnings
    from modules.Many import Many
    from modules.integration_functions import Sensitivity
    from datetime import datetime

    if verbosity>-2:
        print(f'using tuned_rmse_df_out_append={tuned_rmse_df_out_append}')
    if supply_or_demand is None and scenario_sheet_file_path is None:
        scenarios = ['']
    elif supply_or_demand is None:
        scenario_frame = get_scenario_dataframe(file_path_for_scenario_setup=scenario_sheet_file_path, default_year=2019)
        scenarios = ['++'.join([scenario_sheet_file_path,q])
                          for q in scenario_frame.index.get_level_values(0).unique()]+['']
        # TODO implementation to run a baseline somehow, whether to just always have baseline be the first, label baseline somehow.
    else:
        if supply_or_demand=='supply': s = 'ss'
        elif supply_or_demand=='demand': s = 'sd'
        elif supply_or_demand=='demand-alt': s = 'sd-alt'
        elif supply_or_demand=='both': s = 'bo'
        elif supply_or_demand=='both-alt': s = 'bo-alt'
        else: raise ValueError('supply_or_demand input to run_scrap_scenarios function must be str of either `supply` or `demand`')
        if price_response: p='pr'
        else: p='no'
        scenariosb = ['',f'{s}_{p}_1yr_0.01%tot_0%inc',f'{s}_{p}_1yr_0.1%tot_0%inc']
        scenarios2 = [f'{s}_{p}_'+str(yr)+'yr_'+str(round(pct,1))+'%tot_0%inc' for yr in years_of_increase
             for pct in np.arange(0.2,1.1,0.2)]
        scenarios3 = [f'{s}_{p}_'+str(yr)+'yr_'+str(round(pct,1))+'%tot_0%inc' for yr in years_of_increase
             for pct in np.arange(2,21,2)]
        scenarios4 = [f'{s}_{p}_'+str(yr)+'yr_'+str(round(pct,1))+'%tot_0%inc' for yr in years_of_increase
             for pct in np.arange(25,41,5)]
        scenarios = scenariosb+scenarios2+scenarios3+scenarios4
    if verbosity>0: print(scenarios)

    rmse_df = pd.read_csv(f'{output_data_folder}/tuned_rmse_df_out{tuned_rmse_df_out_append}.csv',index_col=[0,1]).stack().unstack(1)
    if commodities is None:
        commodities = ['Steel','Al','Au','Sn','Cu','Ni','Ag','Zn','Pb']

    exponent = 10
    element_commodity_map = {'Steel':'Steel','Al':'Aluminum','Au':'Gold','Cu':'Copper','Steel':'Steel','Co':'Cobalt','REEs':'REEs','W':'Tungstate','Sn':'Tin','Ta':'Tantalum','Ni':'Nickel','Ag':'Silver','Zn':'Zinc','Pb':'Lead','Mo':'Molybdenum','Pt':'Platinum','Te':'Telllurium','Li':'Lithium'}
    col_map = dict(zip(element_commodity_map.values(),element_commodity_map.keys()))

    time_str = str(datetime.now()).replace(':', '_').replace('.', '_')[:21]

    for commodity in commodities:
        # weighted sampling from Bayesian optimization
        commodity=element_commodity_map[commodity].lower()
        weights = np.exp(rmse_df.loc[idx[commodity,'score'],:])**-exponent
        params = rmse_df.loc[commodity]
        params = params.drop([i for i in params.index if 'RMSE' in i or 'R2' in i or i=='score'])
        params.loc['region_specific_price_response'] = params.loc['sector_specific_price_response']
        param_samp = params.T.sample(n=10000,replace=True,weights=weights,random_state=221017)

        # getting 100 hyperparameters for run from resulting distributions
        notes += f' baseline_sampling={baseline_sampling}'
        if baseline_sampling=='random':
            hyp_sample = pd.DataFrame(np.nan, np.arange(0,n_best_scenarios), param_samp.columns)
            rs = 1017
            for i in hyp_sample.index:
                for j in hyp_sample.columns:
                    hyp_sample.loc[i,j] = param_samp[j].sample(random_state=rs).values[0]
                    rs += 1
                hyp_sample.loc[:,'hyperparam_set_number'] = i
                hyp_sample.loc[:,'hyperparam_set_group'] = 0
            hyp_sample = hyp_sample.T

        elif baseline_sampling=='grouped':
            df = pd.DataFrame(np.nan, params.index, np.arange(0,n_per_baseline))
            stds = params[weights.sort_values(ascending=False).head(n_best_scenarios).index].std(axis=1)
            best10 = weights.sort_values(ascending=False).index
            rs = 0
            hyp_sample = pd.DataFrame()
            for n in np.arange(0,n_best_scenarios):
                for ix,i in enumerate(params.index):
                    sign = np.sign(params[best10[n]][i].mean())
                    mean = abs(params[best10[n]][i])
                    std = stds[i]
                    # df.loc[i] = stats.lognorm.rvs(loc=0,scale=mean,s=std,size=n_samp,random_state=rs)*sign
                    generated = stats.norm.rvs(loc=mean,scale=std,size=n_per_baseline*100,random_state=rs)
                    for _ in np.arange(0,3):
                        generated += mean-np.mean(generated)
                        generated = generated[generated>0]
#                         if i!= 'mine_cost_change_per_year':
                        if abs(params.loc[i]).max()<=1:
                            generated = generated[generated<1]
                    df.loc[i] = generated[:n_per_baseline]*sign
                    rs+=1
                df.loc['hyperparam_set_number'] = np.arange(0,n_per_baseline)
                df.loc['hyperparam_set_group'] = n
                hyp_sample = pd.concat([hyp_sample,df],axis=1)
            hyp_sample = hyp_sample.T.reset_index(drop=True).T
        elif baseline_sampling=='clustered':
            hyp_sample = generate_clustered_hyperparam(rmse_df=rmse_df, commodity=commodity, n_best_scenarios=n_best_scenarios,
                                                          n_per_baseline=n_per_baseline, plot=False);
        elif baseline_sampling=='actual':
            best_n = weights.sort_values(ascending=False).head(n_best_scenarios).index
            hyp_sample = params.copy()[best_n]
            hyp_sample.loc['hyperparam_set_number'] = np.arange(0,len(best_n))
            hyp_sample.loc['hyperparam_set_group'] = 0
            hyp_sample = hyp_sample.T.reset_index(drop=True).T

        # this is now redundant since we save updated_commodity_inputs in rmse_df
        # hyp_sample = get_pretuning_params(best_hyperparameters=hyp_sample, material=col_map[commodity.capitalize()], user_data_folder=user_data_folder, verbosity=verbosity)

        # running all
        if run_parallel==0:
            run_fn = op_run_future_scenarios
        elif run_parallel<0:
            run_fn = op_run_future_scenarios_parallel
        else:
            run_fn = op_run_sensitivity_fn
        run_fn(
            commodity=commodity,
            user_data_folder=user_data_folder,
            static_data_folder=static_data_folder,
            hyperparam_df=hyp_sample,
            scenario_list=scenarios,
            scenario_name_base=scenario_name_base,
            verbosity=verbosity,
            run_parallel=run_parallel,
            simulation_time=simulation_time,
            notes=notes,
            random_state=random_state,
            save_mining_info=save_mining_info,
            time_str=time_str
            )



def op_run_future_scenarios(commodity, hyperparam_df, scenario_list, user_data_folder='input_files/user_defined', static_data_folder='input_files/static', scenario_name_base='_run_scenario_set', verbosity=0, run_parallel=None, simulation_time=np.arange(2019,2041), notes='', save_mining_info=False, random_state=None, time_str=None):
    """
    Can be run by run_future_scenarios if run_parallel is set to zero; this is
    currently the most deprecated version of this process, see
    op_run_sensitivity_fn for the current version.
    """
    from integration_functions import Sensitivity
    from datetime import datetime
    if time_str is None:
        time_str = str(datetime.now()).replace(':', '_').replace('.', '_')[:21]

    if type(scenario_list[0])==str:
        scenario_list = [scenario_list]
    hyp_sample = hyperparam_df.copy()

    element_commodity_map = {'Steel':'Steel','Al':'Aluminum','Au':'Gold','Cu':'Copper','Steel':'Steel','Co':'Cobalt','REEs':'REEs','W':'Tungstate','Sn':'Tin','Ta':'Tantalum','Ni':'Nickel','Ag':'Silver','Zn':'Zinc','Pb':'Lead','Mo':'Molybdenum','Pt':'Platinum','Te':'Telllurium','Li':'Lithium'}
    col_map = dict(zip(element_commodity_map.values(),element_commodity_map.keys()))

    material=commodity
    t0 = datetime.now()
    t_per_batch = pd.Series(np.nan,np.arange(0,len(hyp_sample.columns)))
    filename_list = []
    n_scen = len(hyp_sample.columns)*len(scenario_list)
    for m,best_ind in enumerate(hyp_sample.columns):
        if m>1 and len(t_per_batch.dropna())<=3:
            time_remaining = (t_per_batch.dropna().sum()/len(t_per_batch.dropna()))*(n_scen-m*len(scenario_list))
        elif m>1:
            time_remaining = (t_per_batch.dropna().iloc[-3:].sum()/3)*(n_scen-m*len(scenario_list))
        else: time_remaining = (datetime(2022,10,17,23,21,50,0)-datetime(2022,10,17,23,13,20,0))*(n_scen-m*len(scenario_list))
        if verbosity>-1:
            print(f'Hyperparam set {m}/{len(hyp_sample.columns)},\nEst. finish time: {str(datetime.now()+time_remaining)}')
        best_params = hyp_sample[best_ind]
        if type(random_state)==int:
            random_state = [random_state]
        if random_state is None:
            random_states = np.arange(0,1)
        else:
            random_states = np.arange(0,len(random_state))
        for n,scenarios in enumerate(scenario_list):
            for rs in random_states:
                t1 = datetime.now()
                filename=f'output_files/Simulation/'+material+scenario_name_base+str(m)+'.pkl'
                if verbosity>-2: print('--'*15+filename+'-'*15)
                best_params.loc['refinery_capacity_growth_lag']=1
                if random_state is not None:
                    best_params.loc['random_state'] = random_state[rs]
                best_params.loc['reserves_ratio_price_lag'] = 5
                best_params.loc['close_years_back'] = 3

                s = Sensitivity(filename,changing_base_parameters_series=col_map[material.capitalize()],notes=notes,
                                additional_base_parameters=best_params, historical_price_rolling_window=5,
                                simulation_time=simulation_time,
                                scenarios=scenarios, save_mining_info=save_mining_info,
                                OVERWRITE=rs==0,verbosity=verbosity)
                s.time_str = time_str
                s.run_multiple_integration_models(n_scenarios=2, bayesian_tune=False, sensitivity_parameters=['Nothing, giving a string incompatible with any of the variable names'])
                if verbosity>-1: print(f'time for batch: {str(datetime.now()-t1)}')
                t_per_batch.loc[m*len(scenario_list)+n] = datetime.now()-t1
                filename_list += [filename]
    if verbosity>-1: print(f'total time elapsed: {str(datetime.now()-t0)}')

def op_run_sensitivity_fn(commodity, hyperparam_df, scenario_list, user_data_folder='input_files/user_defined', static_data_folder='input_files/static', scenario_name_base='_run_scenario_baseline', verbosity=0, run_parallel=None, simulation_time=np.arange(2019,2041), notes='', save_mining_info=False, trim_result_df=True, random_state=None, time_str=None):
    """
    Run by the run_future_scenarios function if run_parallel is greater than
    zero. Mainly used so that the large pickle files for all the data get
    segmented and do not cause memory usage errors while running.

    If you start running out of memory while running things, check out the
    integration_functions.py file, function complete_bayesian_trial, and change
    the 666 number to something smaller. This may make the results from
    run_all_integration faulty or incomplete when loaded by the
    Many().get_multiple or Many().get_variables functions. Could update the
    get_variables function in line with how Many().load_future_scenario_runs
    function loads its data to fix.
    """
    if time_str is None:
        time_str = str(datetime.now()).replace(':', '_').replace('.', '_')[:21]

    element_commodity_map = {'Steel':'Steel','Al':'Aluminum','Au':'Gold','Cu':'Copper','Steel':'Steel','Co':'Cobalt','REEs':'REEs','W':'Tungstate','Sn':'Tin','Ta':'Tantalum','Ni':'Nickel','Ag':'Silver','Zn':'Zinc','Pb':'Lead','Mo':'Molybdenum','Pt':'Platinum','Te':'Telllurium','Li':'Lithium'}
    col_map = dict(zip(element_commodity_map.values(),element_commodity_map.keys()))
    element = col_map[commodity.capitalize()]

    # comparison
    filename=f'output_files/Simulation/'+commodity+scenario_name_base+'.pkl'
    if verbosity>-2: print('--'*15+filename+'-'*15)
    best_params = pd.Series(dtype=float)
    best_params.loc['reserves_ratio_price_lag'] = 5
    best_params.loc['close_years_back'] = 3

    train_time = np.intersect1d(simulation_time, np.arange(2001,2020))
    s = Sensitivity(pkl_filename=filename, user_data_folder=user_data_folder,
                         static_data_folder=static_data_folder,
                         changing_base_parameters_series=element,
                         notes=notes,
                         scenarios=scenario_list,
                         additional_base_parameters=best_params,
                         simulation_time=simulation_time, include_sd_objectives=False, train_time=train_time,
                         OVERWRITE=True, verbosity=verbosity, historical_price_rolling_window=5,
                         force_integration_historical_price=False,
                         constrain_tuning_to_sign=True,
                         constrain_previously_tuned=False,
                         normalize_objectives=True,
                         use_historical_price_for_mine_initialization=True,
                         save_mining_info=save_mining_info, trim_result_df=trim_result_df)

    s.time_str = time_str
    s.run_multiple_integration_models(n_scenarios=2, bayesian_tune=False, sensitivity_parameters=hyperparam_df, n_jobs=abs(run_parallel))

def op_run_future_scenarios_parallel(commodity, hyperparam_df, scenario_list, user_data_folder='input_files/user_defined', static_data_folder='input_files/static', scenario_name_base='_run_scenario_set', verbosity=0, run_parallel=3, simulation_time=np.arange(2019,2041), notes='', save_mining_info=False, random_state=None, time_str=None):
    """
    Called by run_future_scenarios if its run_parallel input is below zero,
    since in my opinion this function is mostly deprecated
    """
    from integration_functions import Sensitivity
    from datetime import datetime
    from joblib import Parallel, delayed
    from IterTimer import IterTimer

    if time_str is None:
        time_str = str(datetime.now()).replace(':', '_').replace('.', '_')[:21]

    if type(scenario_list[0])==str:
        scenario_list = [scenario_list]
    hyp_sample = hyperparam_df.copy()

    timer = IterTimer(n_iters=len(scenario_list)*hyp_sample.shape[1], log_times=False)

    element_commodity_map = {'Steel':'Steel','Al':'Aluminum','Au':'Gold','Cu':'Copper','Steel':'Steel','Co':'Cobalt','REEs':'REEs','W':'Tungstate','Sn':'Tin','Ta':'Tantalum','Ni':'Nickel','Ag':'Silver','Zn':'Zinc','Pb':'Lead','Mo':'Molybdenum','Pt':'Platinum','Te':'Telllurium','Li':'Lithium'}
    col_map = dict(zip(element_commodity_map.values(),element_commodity_map.keys()))

    material=commodity
    t0 = datetime.now()
    t_per_batch = pd.Series(np.nan,np.arange(0,len(hyp_sample.columns)))
    filename_list = []
    n_scen = len(hyp_sample.columns)*len(scenario_list)
    hyp_sample.loc['reserves_ratio_price_lag'] = 5
    hyp_sample.loc['close_years_back'] = 3

    Parallel(n_jobs=abs(run_parallel))(delayed(run_scenario_set)(m, best_ind, hyp_sample, scenario_list, material, scenario_name_base, col_map, verbosity, simulation_time, save_mining_info, notes, timer, random_state, time_str) for m,best_ind in enumerate(hyp_sample.columns))

    if verbosity>-1: print(f'total time elapsed: {str(datetime.now()-t0)}')

def run_scenario_set(m,best_ind,hyp_sample,scenario_list,material,scenario_name_base,col_map,verbosity, simulation_time=np.arange(2019,2041), save_mining_info=False, notes='', timer=None, random_state=None, time_str=None):
    """
    Called by op_run_future_scenarios_parallel to run each set of scenarios
    """
    if time_str is None:
        time_str = str(datetime.now()).replace(':', '_').replace('.', '_')[:21]

    element_commodity_map = {'Steel':'Steel','Al':'Aluminum','Au':'Gold','Cu':'Copper','Steel':'Steel','Co':'Cobalt','REEs':'REEs','W':'Tungstate','Sn':'Tin','Ta':'Tantalum','Ni':'Nickel','Ag':'Silver','Zn':'Zinc','Pb':'Lead','Mo':'Molybdenum','Pt':'Platinum','Te':'Telllurium','Li':'Lithium'}
    col_map = dict(zip(element_commodity_map.values(),element_commodity_map.keys()))
    element = col_map[material.capitalize()]

    if verbosity>-1:
        print(f'Hyperparam set {m}/{len(hyp_sample.columns)}')
    best_params = hyp_sample[best_ind]
    if type(random_state)==int:
        random_state = [random_state]
    if random_state is None:
        random_states = np.arange(0,1)
    else:
        random_states = np.arange(0,len(random_state))
    for n,scenarios in enumerate(scenario_list):
        for rs_e,rs in enumerate(random_states):
            # timer.start_iter()
            t1 = datetime.now()
            if len(scenarios)<3:
                filename=f'output_files/Simulation/'+material+scenario_name_base+'.pkl'
            else:
                filename=f'output_files/Simulation/'+material+scenario_name_base+str(m)+'.pkl'
            if verbosity>-2: print('--'*15+filename+'-'*15)
            best_params.loc['refinery_capacity_growth_lag']=1
            OVERWRITE = m==0 if len(scenarios)<3 else rs_e==0
            if random_state is not None:
                best_params.loc['random_state'] = random_state[rs]
            best_params.loc['reserves_ratio_price_lag'] = 5
            best_params.loc['close_years_back'] = 3
            s = Sensitivity(filename,changing_base_parameters_series=col_map[material.capitalize()],notes=notes,
                            additional_base_parameters=best_params, historical_price_rolling_window=5,
                            simulation_time=simulation_time,
                            scenarios=scenarios, save_mining_info=save_mining_info,
                            OVERWRITE=OVERWRITE,verbosity=verbosity)
            s.time_str = time_str
            s.run_multiple_integration_models(n_scenarios=2, bayesian_tune=False, sensitivity_parameters=['Nothing, giving a string incompatible with any of the variable names'])
            if verbosity>-1: print(f'time for batch: {str(datetime.now()-t1)}')
        # timer.end_iter()

def get_pretuning_params(best_hyperparameters, material, output_data_folder='output_files', verbosity=0):
    """
    Used to load updated_commodity_inputs.pkl so it gets used for future scenario runs as well.
    """
    if os.path.exists(f'{output_data_folder}/updated_commodity_inputs.pkl'):
        updated_commodity_inputs = pd.read_pickle(f'{output_data_folder}/updated_commodity_inputs.pkl')
        if verbosity > -1: print(f'updated_commodity_inputs source: {output_data_folder}/updated_commodity_inputs.pkl')
    elif os.path.exists('output_files/updated_commodity_inputs.pkl'):
        updated_commodity_inputs = pd.read_pickle('output_files/updated_commodity_inputs.pkl')
        if verbosity > -1: print('updated_commodity_inputs source: output_files/updated_commodity_inputs.pkl')
    elif os.path.exists('updated_commodity_inputs.pkl'):
        updated_commodity_inputs = pd.read_pickle('updated_commodity_inputs.pkl')
        if verbosity>-1: print('updated_commodity_inputs source: updated_commodity_inputs.pkl')
    else:
        raise ValueError('updated_commodity_inputs.pkl does not exist in the expected locations (in this directory, in data relative path, in output_data_folder input given). Need to run the historical_sim_check_demand() function to create an initialization of updated_commodity_inputs.pkl')

    best_params = updated_commodity_inputs[material].copy().dropna()
    changing_base_parameters_series = best_params[[j for j in best_params.index if 'pareto' not in j]]

    if verbosity>1:
        print('without updates, hyperparam:\n',best_hyperparameters)

    for i in changing_base_parameters_series.index:
        if i not in best_hyperparameters.index:
            best_hyperparameters.loc[i,:] = changing_base_parameters_series[i]

    if verbosity>1:
        print('new hyperparam:\n',best_hyperparameters)
    return best_hyperparameters

def get_train_test_scores(rmse_df, weight_price=1):
    """
    Calculates train score, test score, total score,
    and sum score from the relevant RMSE values, since
    the score stored in rmse_df is actually only for
    train. This adds explicit labels.

    weight_price is exponent to which the price RMSE
    is raised to, to increase its contribution to the
    score and should weight it more strongly when
    summing over all RMSEs.
    """
    rmse_copy = rmse_df.copy()
    if rmse_copy.index.nlevels>1:
        for i in ['test','train']:
            string = i+' RMSE'
            test_rmses = rmse_copy.loc[idx[:,[i for i in rmse_copy.index.get_level_values(1).unique()
                                              if string in i]],:].sort_index()
            if test_rmses.shape[0]>0:
                test_rmses.loc[idx[:,[i for i in test_rmses.index.get_level_values(1).unique()
                                      if 'price' in i]],:] = \
                    test_rmses.loc[idx[:,[i for i in test_rmses.index.get_level_values(1).unique()
                                      if 'price' in i]],:]**weight_price
                if (test_rmses.groupby(level=0).sum()!=0).any().any():
                    test_score = np.log(test_rmses.groupby(level=0).sum().replace(0,1e6).div(3))
                else:
                    test_score = test_rmses.groupby(level=0).sum().div(3)
                test_score = pd.concat([test_score.unstack()],keys=[f'{i} score'],axis=1).stack().unstack(0)
                rmse_copy = pd.concat([rmse_copy,test_score])

        # now doing overall score (score saved from original rmse_df is for train set)
        test_rmses = rmse_copy.loc[idx[:,[i for i in rmse_copy.index.get_level_values(1).unique()
                                      if 'RMSE' in i and 'test' not in i and 'train' not in i]],:].sort_index()
        if test_rmses.shape[0]>0:
            if (test_rmses.groupby(level=0).sum()!=0).any().any():
                test_score = np.log(test_rmses.groupby(level=0).sum().div(3))
            else:
                test_score = test_rmses.groupby(level=0).sum().div(3)

            test_score = pd.concat([test_score.unstack()],keys=['total score'],axis=1).stack().unstack(0)
            rmse_copy = pd.concat([rmse_copy,test_score])

        # sum of scores to avoid weighting
        if 'test score' in rmse_copy.index.get_level_values(1).unique():
            test_score = rmse_copy.loc[idx[:,'train score'],:].droplevel(1)+\
                    rmse_copy.loc[idx[:,'test score'],:].droplevel(1)
        else:
            test_score = rmse_copy.loc[idx[:,'train score'],:].droplevel(1)
        test_score = pd.concat([test_score.unstack()],keys=['sum score'],axis=1).stack().unstack(0)
        rmse_copy = pd.concat([rmse_copy,test_score])
        rmse_copy = rmse_copy.sort_index()
    else:
        for i in ['test','train']:
            string = i+' RMSE'
            test_rmses = rmse_copy.loc[[i for i in rmse_copy.index.unique()
                                              if string in i],:].sort_index()
            test_rmses.loc[[i for i in test_rmses.index.unique()
                                  if 'price' in i],:] = \
                test_rmses.loc[[i for i in test_rmses.index.unique()
                                  if 'price' in i],:]**weight_price
            print(test_rmses)
            test_score = np.log(test_rmses.sum().div(3))
            test_score = pd.concat([test_score],keys=[f'{i} score'],axis=1).T
            rmse_copy = pd.concat([rmse_copy,test_score])

        # now doing overall score (score saved from original rmse_df is for train set)
        test_rmses = rmse_copy.loc[[i for i in rmse_copy.index.unique() if 'RMSE' in i and 'test'
                                    not in i and 'train' not in i],:].sort_index()
        test_rmses.loc[[i for i in test_rmses.index.unique()
                              if 'price' in i],:] = \
                                   test_rmses.loc[[i for i in test_rmses.index.unique()
                                   if 'price' in i],:]**weight_price
        test_score = np.log(test_rmses.sum().div(3))
        test_score = pd.concat([test_score],keys=['total score'],axis=1).T
        rmse_copy = pd.concat([rmse_copy,test_score])

        # sum of scores to avoid weighting
        test_score = rmse_copy.loc['train score',:]+\
                rmse_copy.loc['test score',:]
        test_score = pd.concat([test_score],keys=['sum score'],axis=1).T
        rmse_copy = pd.concat([rmse_copy,test_score])
        rmse_copy = rmse_copy.sort_index()

    return rmse_copy

def get_commodity_scores(rmse_train_test, commodity, n_best_scenarios=25):
    """
    Creates the `score` dataframe from the train-test rmse
    dataframe, which should come from the get_train_test_scores
    function so the train score, test score, total score, and
    sum score indeces are there.

    Dataframe contains these scores as columns, a column called
    `sum` that is just the direct sum of the test and train scores
    and boolean columns indicating whether the hyperparameter set
    is on the pareto front (col. `pareto`), or in selection1 or
    selection2.

    Column `selection1` relies on having a train-test setup from
    tuning historical data, and we select the 4*n_best_scenarios
    based on the train score, then select the n_best_scenarios from
    that set based on the test score. Shows True if in this set.

    Column `selection2` is True for the n_best_scenarios based on the
    `total score` column
    """
    if rmse_train_test.index.nlevels>1:
        scores = rmse_train_test.loc[idx[commodity,
                                          rmse_train_test.index.isin(['train score','test score','total score','sum score'],level=1)],:].droplevel(0).T
    else:
        scores = rmse_train_test.loc[rmse_train_test.index.isin(['train score','test score','total score','sum score'])].T
    scores['sum'] = scores.sum(axis=1)
    if 'test score' in scores.columns:
        scores['pareto'] = is_pareto_efficient_simple(scores[['train score','test score']].astype(float).values)
        pareto = scores.loc[scores['pareto']]
        selection = scores.loc[scores['train score'].sort_values().head(n_best_scenarios*4).index]
        selection = selection.loc[selection['test score'].sort_values().head(n_best_scenarios).index]
        selection2 = scores.loc[scores['total score'].sort_values().head(n_best_scenarios).index]
        scores['selection2'] = [i in selection2.index for i in scores.index]
    else:
        scores['pareto'] = False
        selection = scores.loc[scores['train score'].sort_values().head(n_best_scenarios).index]
        scores['selection2'] = [i in selection.index for i in scores.index]
    scores['selection1'] = [i in selection.index for i in scores.index]
    return scores

def get_best_columns(many, commodity, n_best_scenarios=25, weight_price=1):
    """
    I do not currently use this anywhere.

    returns a dictionary of the n_best_scenarios
    for each score (total score, train score, test
    score, sum score)
    """
    rmse_copy = get_train_test_scores(many.rmse_df, weight_price=weight_price)
    column_dict = {}
    for i in ['total score','train score','test score','sum score']:
        cols = rmse_copy.loc[idx[commodity,i],:].sort_index().sort_values().head(n_best_scenarios).index
        column_dict[i] = np.array(cols)
    return column_dict

def generate_clustered_hyperparam(rmse_df, commodity, n_best_scenarios=25, n_per_baseline=10, plot=False):
    """
    Generates a set of hyperparameters based on the best_n_scenarios,
    varied based on the deviation within each hyperparamter but maintaining
    the bulk of any hyperparameter correlations.

    Baseline selection is more complex, relies on having a train-test
    setup from tuning historical data. We select the 4*n_best_scenarios
    based on the train score, then select the n_best_scenarios from that
    set based on the test score. Each of these scenarios is modified to
    produce n_per_baseline scenarios by summing with a N(0,std(param))
    distribution, where std(param) is the standard deviation of the
    hyperparameter within the n_best_scenarios. Total number of scenarios
    is n_best_scenarios*n_per_baseline.

    Returns a hyperparameter dataframe with scenario numbers
    for columns, hyperparameters labels for index.
    """
    rmse_train_test = get_train_test_scores(rmse_df, weight_price=1)
    scores = get_commodity_scores(rmse_train_test,commodity,n_best_scenarios=10 if n_best_scenarios<10 else n_best_scenarios)

    if rmse_df.index.nlevels>1:
        rmse_df = rmse_df.copy().loc[commodity]
    rmse_df.loc['region_specific_price_response'] = rmse_df.loc['sector_specific_price_response']
    rmse_df = rmse_df.sort_values(by='score',axis=1)
    ind = rmse_df.columns
    droppers = [i for i in rmse_df.index.unique() if np.any([j in i for j in ['score','R2','RMSE']])]
    rmse_df.drop(droppers,inplace=True)

    rmse_ph = rmse_df[scores.loc[scores.selection1].index].T
    rmse_ph = rmse_ph.loc[[i for i in ind if i in rmse_ph.index]]

    hyperparam_ph = pd.DataFrame(np.nan,np.arange(0,n_best_scenarios*n_per_baseline),rmse_ph.columns)
    for ei,i in enumerate(rmse_ph.index[:n_best_scenarios]):
        ind = hyperparam_ph.index[ei*n_per_baseline:(ei+1)*n_per_baseline]
        for ef,f in enumerate(rmse_ph.columns):
            sign = np.sign(rmse_ph[f][i])
            if n_per_baseline>1:
                data = np.append(abs(rmse_ph[f][i]),
                                 abs(rmse_ph[f][i]+stats.norm.rvs(loc=0, scale=rmse_ph[f].std(),
                                                    size=n_per_baseline-1, random_state=ei+ef)))
            else:
                data = np.array([abs(rmse_ph[f][i])])
            if abs(rmse_ph[f]).max()<=1:
                data[data>1] = 2-data[data>1]
            data = sign*data
            hyperparam_ph.loc[ind,f]=data
        hyperparam_ph.loc[ind,'hyperparam_set_number'] = np.arange(0,n_per_baseline)
        hyperparam_ph.loc[ind,'hyperparam_set_group'] = ei

    if plot:
        fig,ax=easy_subplots(hyperparam_ph.columns)
        for i,a in zip(hyperparam_ph.columns,ax):
            hyperparam_ph[i].iloc[:n_per_baseline].plot.hist(ax=a,title=i)
        return hyperparam_ph.T, fig,ax
    return hyperparam_ph.T

def plot_given_columns(many, commodity, columns, column_name=None, ax=None, column_subset=None, start_year=None, end_year=2019, dpi=50):
    """
    Plots historical vs simulated demand, mining,
    and primary commodity price for a given commodity
    and whichever hyperparameter sets given as the
    `columns` variable. Can get the n_best_scenarios
    using the get_best_columns function, or more manually
    by running get_train_test_scores on an rmse_df (or
    commodity subset), then running its output through
    get_commodity_scores, e.g.

    rr = get_train_test_scores(many_test.integ.rmse_df.loc['silver'])
    s1 = get_commodity_scores(rr,None)
    then passing s1.loc[s1.selection2].index as the columns input.

    -------------
    many: Many() object, needs to have results object of its own,
        so if you ran get_multiple to load many, you should pass
        many.integ to this function
    commodity: str, commodity name in lowercase form
    columns: list, columns from rmse_df to plot. Will highlight
        the lowest-score one if no column_subset is passed
    column_name: str, gets included in the plot title if not None
    ax: matplotlib axes object, can leave out and this will
        create its own plot for you.
    column_subset: list, allows you to select a subset of the
        passed columns to highlight, or to just plot two groups
        of parameter sets simultaneously, since column_subset
        and columns do not have to intersect. Pass a list or
        array of numbers corresponding to rmse_df columns.
    dpi: dots per inch, controls resolution. Only functions if
        the ax input is None.
    """
    if ax is None:
        fig,ax=easy_subplots(3, dpi=dpi)
    else:
        fig = 0
    objective_results_map = {'Total demand':'Total demand','Primary commodity price':'Refined price',
                                 'Primary demand':'Conc. demand','Primary supply':'Mine production',
                                'Conc. SD':'Conc. SD','Scrap SD':'Scrap SD','Ref. SD':'Ref. SD'}
    for i,a in zip(['Total demand','Primary commodity price','Primary supply'], ax):
        results = many.results.copy()[objective_results_map[i]].sort_index()\
            .loc[idx[commodity,:,2001:end_year]].droplevel(0).unstack(0)
        if 'SD' not in i:
            historical_data = many.historical_data.copy()[i].loc[commodity].loc[:2019]
            if start_year is not None:
                historical_data = historical_data.loc[start_year:]
        else:
            historical_data = pd.Series(results.min(),[0])
        results_ph = results.copy()
        results = results[columns]

        diction = get_unit(results, historical_data, i)
        results, historical_data, unit = [diction[i] for i in ['simulated','historical','unit']]
        results_ph *= results[columns[0]].mean()/results_ph[columns[0]].mean()
        sim_line = a.plot(results,linewidth=1,color='gray',alpha=0.3,label=results.columns)
        if column_subset is None:
            best_line= a.plot(results[columns[0]],linewidth=6,label='Simulated',color='tab:blue')
        else:
            best_line= a.plot(results_ph[column_subset],linewidth=1,label='Simulated',color='tab:blue')
        mins = min(historical_data.min(),results[columns[0]].min())*0.95
        maxs = max(historical_data.max(),results[columns[0]].max())*1.1
        hist_line = a.plot(historical_data,label='Historical',color='k',linewidth=6)
        inter = np.intersect1d(results.index,historical_data.index)
        m = sm.GLS(historical_data.loc[inter], sm.add_constant(results[columns[0]].loc[inter])).fit(cov_type='HC3')
        mse = round(m.mse_resid**0.5,2)
        mse = round(m.rsquared,2)
        if column_name is not None:
            title=f'Best {i}, {column_name} {commodity},\n'+r'$R^2$'+f'={mse}, scenario {columns[0]}'
        else:
            title=f'Best {i}, {commodity}\nR2={mse}, {columns[0]}'
        a.set(title=title,
              ylabel=i+' ('+unit+')',xlabel='Year',ylim=(mins,maxs))
        if len(sim_line)<10:
            a.legend()
    return fig,ax

def plot_best_scenarios_train_test(many, weight_price=1, dpi=50):
    """
    plotting the 25 best scenarios for each of the score
    bases (train, test, total, sum) for each commodity.
    Makes a very large grid for each commodity.

    many: Many() object, needs to have results object of its own,
        so if you ran get_multiple to load many, you should pass
        many.integ to this function
    weight_price: exponent price gets raised to when summing the
        normed RMSEs - is an effort to try to see what the best
        scenario looks like with different weighting
    dpi: float, dots per inch, controls figure resolution
    """
    fig_list = []
    ax_list = []
    commodities = many.results.index.get_level_values(0).unique()
    for commodity in commodities:
        fig,axes = easy_subplots(12,4,dpi=dpi)
        n=-1
        column_dict = get_best_columns(many, commodity=commodity)
        for column_name in ['train','test','total','sum']:
            n+=1
            ax = axes[n::4]
            columns = column_dict[column_name+' score']
            plot_given_columns(many, commodity, columns, column_name, ax)
        fig.tight_layout()
        fig.suptitle(commodity.capitalize(),weight='bold',y=1.02,x=0.515)
        fig_list += [fig]
        ax_list += [ax]
        plt.show()
        plt.close()
    return fig_list

def plot_test_score_vs_train_score_commodity(rmse_df, commodity, ax, quantile=0.5, show_selection=None, n_best_scenarios=25, color=True, plot=True):
    """
    plots scatterplots of train vs test scores for a commodity,
    with the color option making it so they are colored according
    to the total score.

    show_selection: None or str, where str can be pareto, selection1,
    or selection2. Causes the corresponding scenarios to be highlighted
    in blue.
    """
    rmse_train_test = get_train_test_scores(rmse_df, weight_price=1)
    scores = get_commodity_scores(rmse_train_test,commodity,n_best_scenarios=n_best_scenarios)

    x = 'train score'
    y = 'test score'
    scores = scores.loc[scores['train score']<scores['train score'].quantile(quantile)]
    s=60
    if color and plot:
        ax.scatter(scores[x],scores[y],s=s, c=scores['total score'],cmap='gist_earth_r')
    elif plot:
        ax.scatter(scores[x],scores[y],s=s)

    if show_selection is not None:
        selection = scores.loc[scores[show_selection]]
        if plot:
            ax.scatter(selection[x],selection[y], s=s, c='blue')
    if plot:
        ax.set(xlabel=x.capitalize(), ylabel=y.capitalize())
    return scores

def plot_test_score_vs_train_score(many, quantile=0.5, n_best_scenarios=25, show_selection=None, color=True, plot=True):
    """
    plots scatterplots of train vs test scores for each commodity,
    with the color option making it so they are colored according
    to the total score.

    show_selection: None or str, where str can be pareto, selection1,
    or selection2. Causes the corresponding scenarios to be highlighted
    in blue.
    """
    commodities = [many.element_commodity_map[i].lower() for i in many.ready_commodities]
    if plot:
        fig,ax = easy_subplots(commodities)
    else:
        ax = commodities
    scores_dict = {}
    for commodity,a in zip(commodities,ax):
        scores_dict[commodity] = plot_test_score_vs_train_score_commodity(
            many.rmse_df, commodity, a, quantile=quantile, n_best_scenarios=25,
            show_selection=show_selection, color=color, plot=plot
        )
        if plot:
            a.set(title=commodity.capitalize())
    if plot:
        fig.tight_layout()
    else: fig,ax=None,None
    return fig, ax, scores_dict

def plot_best_scores_history(many, commodity, show_selection='selection1', n_best_train=0):
    """
    Just used to check the historical vs
    simulated for a single commodity, may
    be deprecated
    """
    fig,ax,scores_dict = plot_test_score_vs_train_score(many, show_selection=None,
                                                        quantile=0.5, color=True,plot=False)
    scores = scores_dict[commodity]
    fig,ax=easy_subplots(3)
    columns = [i for i in scores.loc[scores[show_selection]].index if i not in [0]]
    if n_best_train!=0:
        column_subset = columns
        columns = [i for i in scores['train score'].sort_values().head(n_best_train).index if i not in [0]]
    else: column_subset=None
    # columns = [i for i in scores.sort_values(by='sum').head(20).index if i not in [0]]
    plot_given_columns(many, commodity, columns, show_selection, ax, column_subset=column_subset)
    fig.tight_layout()

def plot_best_scores_hyperparam_distributions(many, commodity):
    """
    Trying to see how the hyperparameter
    distributions are affected by choosing
    selection1 vs selection2 for baseline
    """
    fig,ax,scores_dict = plot_test_score_vs_train_score(many, show_selection=None,
                                                        quantile=0.5, color=True,plot=False)
    scores = scores_dict[commodity]
    rmse_df = many.rmse_df.copy().loc[commodity]
    droppers = [i for i in rmse_df.index.unique() if np.any([j in i for j in ['score','R2','RMSE']])]
    rmse_df.drop(droppers,inplace=True)
    fig,ax = easy_subplots(rmse_df.index.unique())
    rmse_df_subset = pd.concat([
        rmse_df[scores.loc[scores.selection1].index].T,
        rmse_df[scores.loc[scores.selection2].index].T],
        keys=['Option 1','Option 2'])
    for i,a in zip(rmse_df_subset.columns,ax):
        mins = rmse_df_subset[i].min()
        maxs = rmse_df_subset[i].max()
        # rmse_df_subset[i].unstack(0).plot.hist(ax=a,alpha=0.5,bins=np.linspace(mins,maxs,50))
        rmse_df_subset[i].unstack(0).plot.kde(ax=a,bw_method=0.2)
        a.set(title=make_parameter_names_nice([i])[i])
    fig.tight_layout()

def get_rmse_df_results_hyperparam(commodity, filename_modifier, demand_or_mining='demand', filename_base='_run_hist',file_folder='output_files/Historical tuning'):
    """
    returns rmse_df, results, and hyperparam for an individual
    pkl file, taking in all the necessary inputs to describe the
    filename.
    """
    demand_or_mining = demand_or_mining.replace('demand','DEM')
    filename=f'{file_folder}/{commodity}{filename_base}{filename_modifier}_{demand_or_mining}.pkl'
    df = pd.read_pickle(filename)
    ph = df.loc['rmse_df'].iloc[-1][0]
    ph.index = pd.MultiIndex.from_tuples(ph.index)
    ph2 = pd.concat([df.loc['results'][i] for i in df.loc['results'].index],axis=0,keys=df.columns)
    ph3 = pd.concat([df.loc['hyperparam'][i] for i in df.loc['results'].index],axis=0,keys=df.columns)
    return ph,ph2,ph3

def get_expected_parameter_signs(demand_or_mining, rmse_df_multi):
    """
    Used in compare_constrained_unconstrained_tuning
    function. Inputs correspond with variables in that
    fuction.

    Returns a dataframe of `positive` or `negative` for
    each parameter based on its value in the `_constrain`
    run in the rmse_df_multi dataframe. If `_constrain`
    is not in the rmse_df_multi dataframe, puts `unknown`

    rmse_df_multi: pandas dataframe with three index levels:
        filename_modifier, scenario_number, parameter
    """
    if '_constrain' in rmse_df_multi.index.levels[0]:
        tuning_range = np.sign(rmse_df_multi.loc['_constrain'].loc[rmse_df_multi.index.get_level_values(1)[0]])
        tuning_range.loc[:] = ['positive' if i>0 else 'negative' for i in tuning_range.values]
    else:
        tuning_range = rmse_df_multi.loc[
            rmse_df_multi.index.get_level_values(0)[0]].loc[rmse_df_multi.index.get_level_values(1)[0]]
        tuning_range.loc[:] = ['unknown' for i in tuning_range.values]

    return tuning_range

def compare_constrained_unconstrained_tuning(plot_cummin_rmse=True,log_cummin_rmse=True,plot_best_distributions=True,n_best_distributions=25,demand_or_mining='demand', to_compare = ['_constrain','_unconstrain','_unconstrain1'],dpi=50):
    """
    plot_cummin_rmse: bool, whether to plot the cumulative minimum
        RMSE/score value vs the number of Bayesian optimization
        runs, for each commodity and tuning method
    log_cummin_rmse: bool, whether to take the log10 of RMSE so it is
        easier to visualize
    plot_best_distributions: bool, whether to plot the distributions
        for the parameters tuned for each commodity and tuning method
    n_best_distributions: int, number of best parameters to use in
        plot_best_distributions
    mining_or_demand: str, either `mining` or `demand`, selects which
        pre-tuning method parameters are shown
    to_compare: list, list containing the filename_modifier strings
        for each tuning method to compare
    dpi: float, dots per inch, figure resolution
    """
    ready_commodities = ['Al','Au','Sn','Cu','Ni','Ag','Zn','Pb','Steel']
    element_commodity_map = {'Al':'Aluminum','Au':'Gold','Cu':'Copper','Steel':'Steel','Co':'Cobalt','REEs':'REEs','W':'Tungstate','Sn':'Tin','Ta':'Tantalum','Ni':'Nickel','Ag':'Silver','Zn':'Zinc','Pb':'Lead','Mo':'Molybdenum','Pt':'Platinum','Te':'Telllurium','Li':'Lithium'}
    commodities = [element_commodity_map[i].lower() for i in ready_commodities]

    fig_dict = {}
    if plot_cummin_rmse:
        cummin_fig, cummin_ax = easy_subplots(commodities,dpi=dpi)

    if demand_or_mining in ['demand','mining']:
        rmse_or_score = 'RMSE'
    else:
        rmse_or_score = 'score'

    rmse_df_all = pd.DataFrame()
    for commodity,cummin_a in zip(commodities,cummin_ax):
        rmse_df_multi = pd.concat([get_rmse_df_results_hyperparam(commodity=commodity,
                                                                  filename_modifier=tc,
                                                                  demand_or_mining=demand_or_mining,
                                                                  filename_base='_run_hist',
                                                                  file_folder='output_files/Historical tuning')[0]
                                   for tc in to_compare],
                                  keys=to_compare)
        rmse_df_all = pd.concat([rmse_df_all,
                                 pd.concat([rmse_df_multi],keys=[commodity])])
        expected_parameter_signs = get_expected_parameter_signs(demand_or_mining,rmse_df_multi)

        if plot_cummin_rmse:
            for tc in to_compare:
                rmse_df_multi.loc[tc].loc[idx[:,rmse_or_score]].cummin().plot(
                    logy=log_cummin_rmse,
                    label=tc.replace('_','').capitalize(),
                    ax=cummin_a,
                    title=f'{commodity.capitalize()}',
                    xlabel='n iterations',
                    ylabel='RMSE'
                )
            cummin_a.legend()
            # ph.loc[idx[:,'RMSE']].plot()

        best_multi = {}
        for tc in to_compare:
            best = rmse_df_multi.loc[tc].loc[idx[:,rmse_or_score]].sort_values().head(n_best_distributions).index
            best_multi[tc] = best
        if plot_best_distributions:
            demand_params = [i for i in rmse_df_multi.index.get_level_values(2).unique() if not
                             np.any([j in i for j in ['RMSE','score','R2','region_specific_price_response']])]
            dist_fig,dist_ax = easy_subplots(demand_params)
            for b,a in zip(demand_params,dist_ax):
                for tc in to_compare:
                    c = expected_parameter_signs[b]
                    rmse_df_multi.loc[tc].loc[idx[best_multi[tc],b]].plot.hist(
                        ax=a,label=tc.replace('_','').capitalize(),alpha=0.5,
                        title=f'{b}\nexpect {c}'
                    )
                a.legend()
                # ph.loc[idx[]].sample(ph.loc[ph.loc[idx[:,'RMSE']]])
            dist_fig.tight_layout()
            dist_fig.suptitle(f'{commodity.capitalize()}',y=1.05,weight='bold')
            fig_dict['dist_fig_'+commodity] = dist_fig
    if plot_cummin_rmse:
        cummin_fig.tight_layout()
        fig_dict['cummin_fig'] = cummin_fig
    rmse_df_all.index = pd.MultiIndex.from_tuples(rmse_df_all.index)
    return rmse_df_all, fig_dict

def get_pca_tsne_results(all_hyperparam, n_best):
    """
    Fits the PCA and TSNE models to the data. Does
    the standard scaler transformation for both.
    returns 2 dataframes, pca_results and tsne_results
    in that order.

    all_hyperparam: pandas dataframe, from
        format_data_for_pca_tsne function
    n_best: int, number of best-fitting
        parameters to use in fitting the
        PCA and TSNE models
    """
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    n=n_best
    x = all_hyperparam.loc[:,:n].unstack().stack(0)
    tsne = TSNE(n_components=2, verbose=0, random_state=123)
    pca = PCA(n_components=2)
    scaler = StandardScaler()
    x_std = scaler.fit_transform(x.values)
    # x_std=x.values
    z_pca = pca.fit_transform(x_std)
    z_tsne = tsne.fit_transform(x_std)

    tsne_results = pd.DataFrame(z_tsne, index=x.index, columns=['T-SNE 1','T-SNE 2'])
    pca_results = pd.DataFrame(z_pca, index=x.index, columns=['PCA 1','PCA 2'])
    return pca_results,tsne_results

def format_data_for_pca_tsne(demand_or_mining='demand', filename_base='_run_hist', filename_modifier=''):
    """
    returns a dataframe including all the hyperparameters
    for each commodity for pre-tuning. For mining, currently
    picks just three parameters (which are typically important
    ones). Sorts each commodity sub-frame according to lowest RMSE.

    demand_or_mining: str, `demand` or `mining`, determines
        which set of pre-tuning hyperparameters are used.
    filename_base: str
    filename_modifier: str
    """
    mining_only = demand_or_mining=='mining'
    ready_commodities = ['Al','Au','Sn','Cu','Ni','Ag','Zn','Pb','Steel']
    element_commodity_map = {'Al':'Aluminum','Au':'Gold','Cu':'Copper','Steel':'Steel','Co':'Cobalt','REEs':'REEs','W':'Tungstate','Sn':'Tin','Ta':'Tantalum','Ni':'Nickel','Ag':'Silver','Zn':'Zinc','Pb':'Lead','Mo':'Molybdenum','Pt':'Platinum','Te':'Telllurium','Li':'Lithium'}
    all_rmse_df = pd.DataFrame()
    for material in ready_commodities:
        material = element_commodity_map[material].lower()
        indiv_demand = Individual(filename=f'output_files/Historical tuning/{material}{filename_base}{filename_modifier}_DEM.pkl',rmse_not_mae=False,dpi=50)
        indiv_mining = Individual(filename=f'output_files/Historical tuning/{material}{filename_base}{filename_modifier}_mining.pkl',rmse_not_mae=False,dpi=50)
        # both = pd.concat([indiv_mining.rmse_df.rename({'R2':'Mining R2','RMSE':'Mining RMSE'}),indiv_demand.rmse_df.rename({'R2':'Demand R2','RMSE':'Demand RMSE'})])
        if mining_only:
            df = indiv_mining.rmse_df.copy().sort_values(by='RMSE',axis=1).T.reset_index(drop=True).T
            df = df.rename({'R2':'Mining R2','RMSE':'Mining RMSE'})
            both = pd.concat([df])
        else:
            df = indiv_demand.rmse_df.copy().sort_values(by='RMSE',axis=1).T.reset_index(drop=True).T
            df = df.rename({'R2':'Demand R2','RMSE':'Demand RMSE'})
            both = pd.concat([df])
        both = pd.concat([both],keys=[material])
        all_rmse_df = pd.concat([all_rmse_df,both])
    # all_hyperparam = all_rmse_df.drop(['Mining R2','Mining RMSE','Demand R2','Demand RMSE'],level=1)
    if 'Mining R2' in all_rmse_df.index.get_level_values(1).unique():
        all_hyperparam = all_rmse_df.drop(['Mining R2','Mining RMSE'],level=1)
        all_hyperparam = all_hyperparam.loc[idx[:,['mine_cu_margin_elas','primary_oge_scale','mine_cost_change_per_year']],:]
    if 'Demand R2' in all_rmse_df.index.get_level_values(1).unique():
        all_hyperparam = all_rmse_df.drop(['Demand R2','Demand RMSE'],level=1)
    return all_hyperparam

def plot_pca_tsne(pca_results, tsne_results, demand_or_mining='demand', filename_modifier='', n_best=25, dpi=50):
    """
    plots 2-dimensional TSNE and PCA plots for the given
    results, for each commodity.

    Returns fig_dict, all_hyperparam, pca_results, tsne_results

    demand_or_mining: str, `demand` or `mining`, determines
        which set of pre-tuning hyperparameters are used.
    filename_base: str
    filename_modifier: str
    n_best: int, number of best-fitting
        parameters to use in fitting the
        PCA and TSNE models
    dpi: float, dots per inch, figure resolution
    """
    all_hyperparam = format_data_for_pca_tsne(demand_or_mining,filename_modifier=filename_modifier)
    pca_results,tsne_results = get_pca_tsne_results(all_hyperparam, n_best)

    fig_dict = {}
    fig,ax=easy_subplots(1,dpi=dpi)
    tsne_results_sns = tsne_results.reset_index().rename(columns={'level_0':'Commodity','index':'Commodity','level_1':'Scenario'})
    g = sns.scatterplot(data=tsne_results_sns, x='T-SNE 1', y='T-SNE 2', hue='Commodity',ax=ax[0])
    sns.move_legend(g,loc=(1,0))
    fig_dict['tsne'] = fig

    fig,ax=easy_subplots(1,dpi=dpi)
    pca_results_sns = pca_results.reset_index().rename(columns={'level_0':'Commodity','index':'Commodity','level_1':'Scenario'})
    g = sns.scatterplot(data=pca_results_sns, x='PCA 1', y='PCA 2', hue='Commodity', ax=ax[0])
    sns.move_legend(g,loc=(1,0))
    fig_dict['pca'] = fig
    return fig_dict, all_hyperparam, pca_results, tsne_results

class SHAP():
    '''
    A useful description of the way SHAP produces interaction effects:
      (see https://www.databricks.com/blog/2019/06/17/detecting-bias-with-shap.html)
      The total effect of identifying as female on the prediction can be
      broken down into the effect of identifying as female AND being an
      engineering manager, AND working with Windows, etc

    Another good link, with some resources listed at the bottom:
     (see https://blog.datascienceheroes.com/how-to-interpret-shap-values-in-r/#:~:text=How%20to%20interpret%20the%20shap%20summary%20plot%3F%201,point%20represents%20a%20row%20from%20the%20original%20dataset.)
     There is a vast literature around this technique, check the online book Interpretable Machine Learning by Christoph Molnar. It addresses in a nicely way Model-Agnostic Methods and one of its particular cases Shapley values. An outstanding work.
        https://christophm.github.io/interpretable-ml-book/shapley.html
      From classical variable, ranking approaches like weight and gain, to shap values: Interpretable Machine Learning with XGBoost by Scott Lundberg.
        https://towardsdatascience.com/interpretable-machine-learning-with-xgboost-9ec80d148d27
      A permutation perspective with examples: One Feature Attribution Method to (Supposedly) Rule Them All: Shapley Values.
        https://towardsdatascience.com/one-feature-attribution-method-to-supposedly-rule-them-all-shapley-values-f3e04534983d

    -----------------------------
    Initialization variables:
    many: instance of the Many class, with data loaded
    commodity: (None | str), commodity name, upper or lower case. None
        causes all commodities to be used; individual commodities can
        be used by giving their corresponding string
    standard_scaler: bool, whether to run
        sklearn.preprocessing.StandardScaler on each column of data such
        that it is normalized to have mean 0 and std 1
    dummies: bool, whether to include commodity-level dummy variables
        (e.g. a column of zeros with 1s for all Ag scenarios)
    split_frac: float (0,1), for train-test split in creating the ML
        model for the tree-based regression we get SHAP values from
    Regressor: tree-based regression class, can be RandomForestRegressor,
        ExtraTreesRegressor, GradientBoostingRegressor, likely others;
        regression to use for the tree-based regression that we get SHAP values from
    use_train_data: bool, whether to use the train data rather than test
        data for shap.TreeExplainer(regression_model).explainer(data)
    objective:  None or str. If str, has to be one of the columns in
        multi_scenario_results_formatted, otherwise will be using `score`
        or `RMSE` from rmse_df dataframe depending on whether using the
        Integration or demandModel formulation. If str, will try to do
        the mean difference from baseline.
    processing_option: None or str, used to determine which method we use to
        set up y_df in get_X_df_y_df function, particularly for criticality
        evaluation. Options are:
        - `2040 ratio 2018 hist`: y_df will contain the ratio between the
           simulated 2040 value and the historical 2018 value, for the given
           objective, provided the objective has a corresponding historical
           value
    -----------------------------
    Methods:
    initialize: runs all the main code (run_tree_based_regression, initialize_shap,
        and gamma_facet) to produce the data for variables rmse_df, shap_values_df,
        shap_interaction_df, synergy_matrix, independence_matrix, redundancy_matrix
    run_tree_based_regression: splits rmse_df and runs a tree-based regression model
        for RMSE from the parameter values, so we can then run SHAP on that
    initialize_shap: sets up the shap.TreeExplainer object from the regression model,
        runs its explainer method on the data, and sets up the arrays and dataframes
        for shap_values/shap_values_df and shap_interaction_values/shap_interaction_df
    get_interactions: precursor to gamma_facet, which uses the equations in Ittner et
        al 2021 paper to calculate the synergy, independence, and redundancy matrices.
        This has no correction for non-orthogonality but is much faster
    plot_shap: plots the various SHAP plots using the shap methods summary, waterfall,
        and heatmap, as well as plotting the shap value vs parameter value for each
        parameter
    summary_plot: Does the SHAP summary plot (the one with lots of
        little colored dots across lines for each parameter)
    waterfall_plot: does the SHAP waterfall plot for a single data point
    heatmap_plot: Does the SHAP heatmap plot, which is fairly similar to
        the summary plot in terms of information, as far as I
        can tell
    shap_vs_param_plot: plots the SHAP values as functions of the
        corresponding parameter value. Similar to an unstacked
        version of the summary plot.

    -------------
    Typical run looks like:
    sh = SHAP(many,commodity='aluminum',split_frac=0.6, use_train_data=False,
              standard_scaler=True, objective='Conc. supply')
    sh.initialize()
    sh.plot_shap(dpi=50,highlight_best=True)
    '''
    def __init__(self, many, commodity=None, standard_scaler=True, dummies=False, split_frac=0.5, Regressor=RandomForestRegressor, use_train_data=False, objective=None, processing_option=None):
        self.many = many
        if commodity!=None:
            commodity=commodity.lower()
        self.commodity = commodity
        self.standard_scaler = standard_scaler
        self.dummies = dummies
        self.split_frac = split_frac
        self.Regressor = Regressor
        self.use_train_data = use_train_data
        self.objective = objective
        if processing_option is not None:
            self.processing_option = processing_option

    def initialize(self):
        '''
        runs all the main code (run_tree_based_regression, initialize_shap,
        and gamma_facet) to produce the data for variables rmse_df,
        shap_values_df, shap_interaction_df, synergy_matrix,
        independence_matrix, redundancy_matrix
        '''
        self.run_tree_based_regression(self.many, commodity=self.commodity, standard_scaler=self.standard_scaler,
                                       dummies=self.dummies, split_frac=self.split_frac, Regressor=self.Regressor,)
        self.initialize_shap()
        self.gamma_facet() # was previously self.get_interactions(), but have shifted in favor of the package provided by Ittner (see function notes)

    def run_tree_based_regression(self, many, commodity=None, standard_scaler=True, dummies=False, split_frac=0.5, Regressor=RandomForestRegressor, objective=None):
#         self.rmse_df = many.rmse_df_sorted.copy().astype(float)
#         rmse_df = self.rmse_df.rename(make_parameter_names_nice([i for i in self.rmse_df.index.get_level_values(1).unique() if 'RMSE' not in i and 'R2' not in i and i!='score']),level=1)

#         if standard_scaler:
#             scaler = StandardScaler()
#             x_std = scaler.fit_transform(rmse_df.values)
#             rmse_df = pd.DataFrame(x_std,rmse_df.index,rmse_df.columns)

#         self.X_df = rmse_df.copy()
#         for r in ['R2','RMSE','score','region_specific_price_response','Region specific intensity elasticity to price']:
#             if self.X_df.index.nlevels==1 and r in self.X_df.index:
#                 self.X_df = self.X_df.drop(r)
#             elif self.X_df.index.nlevels>1 and r in self.X_df.index.get_level_values(1):
#                 self.X_df = self.X_df.drop(r,level=1)

#         self.y_df = rmse_df.loc[idx[:,'RMSE'],:]
#         self.y_df = np.log(self.y_df)
#         self.X_df = self.X_df.unstack().stack(level=0)
#         self.y_df = self.y_df.unstack().stack(level=0).iloc[:,0]
        # self.rmse_df = self.rmse_df if commodity==None else self.rmse_df.loc[commodity]
        # self.X_df = self.X_df if commodity==None else self.X_df.loc[commodity]
        # self.y_df = self.y_df if commodity==None else self.y_df.loc[commodity]

        get_X_df_y_df(self.many, commodity=commodity, standard_scaler=standard_scaler, objective=self.objective)
        self.X_df = self.many.X_df.copy()
        self.X_df_prescale = self.many.X_df_prescale.copy()
        self.X_df = self.X_df.rename(columns=make_parameter_names_nice(self.X_df.columns))
        self.X_df_prescale = self.X_df_prescale.rename(columns=make_parameter_names_nice(self.X_df_prescale.columns))
        self.y_df = self.many.y_df.copy()
        self.y_df_prescale = self.many.y_df_prescale.copy()

        if dummies:
            self.X_df.loc[:,'commodity =']=self.X_df.index.get_level_values(0)
            self.X_df = pd.get_dummies(self.X_df,columns=['commodity ='])
            self.X_df_prescale.loc[:,'commodity =']=self.X_df_prescale.index.get_level_values(0)
            self.X_df_prescale = pd.get_dummies(self.X_df_prescale,columns=['commodity ='])
        self.many.X_df = self.X_df.copy()

        if self.use_train_data:
            self.X_df_test = self.X_df.copy()
            self.X_test = self.X_df_test.values
            self.y_df_test = self.y_df.copy()
        else:
            self.X_df_test = self.X_df.sample(frac=split_frac,replace=False,random_state=0)
            self.X_test = self.X_df_test.reset_index(drop=True).values
            self.y_df_test = self.y_df.loc[self.X_df_test.index]
            self.X_df = self.X_df.loc[~self.X_df.index.isin(self.X_df_test.index)]
            self.y_df = self.y_df.loc[self.X_df.index]

        self.X_df_prescale_test = self.X_df_prescale.loc[self.X_df_test.index]
        self.y_df_prescale_test = self.y_df_prescale.loc[self.y_df_test.index]

        self.X = self.X_df.values
        self.y = self.y_df.values.flatten()

        self.regr = Regressor(random_state=0)
        if type(self.y_df)==pd.core.frame.DataFrame:
            self.y_df = self.y_df.iloc[:,0]
        self.regr.fit(self.X_df.fillna(0),self.y_df)

        if self.objective is None: self.objective='RMSE'
        self.test = pd.concat([self.X_df_test,self.y_df_test.rename(columns={self.y_df_test.columns[0]:self.objective})],axis=1)
        self.test_prescale = pd.concat([self.X_df_prescale_test,self.y_df_prescale_test.rename(columns={self.y_df_prescale_test.columns[0]:self.objective})],axis=1)
        self.test.loc[:,'predicted '+self.objective] = self.regr.predict(self.X_df_test.fillna(0))
        self.test_prescale.loc[:,'predicted '+self.objective] = self.test['predicted '+self.objective]

    def initialize_shap(self):
        from copy import deepcopy
        self.explainer = shap.TreeExplainer(self.regr)
        explainer_copy = deepcopy([self.explainer])[0]
        data = self.X_df.copy() if self.use_train_data else self.X_df_test.copy()
        data = data.rename(columns=make_parameter_names_nice(data.columns))
        self.data = data.copy()
        self.shap_values_object = self.explainer(data)
        self.shap_values = self.shap_values_object.values
        self.shap_values_df = pd.DataFrame(self.shap_values, data.index, data.columns)
        # alt_explainer = shap.TreeExplainer(self.regr)
        # self.shap_interaction_values = alt_explainer.shap_interaction_values(data)
        self.shap_interaction_values = explainer_copy.shap_interaction_values(data)
        x = self.shap_interaction_values
        data_ph = data.fillna(0).stack()
        if self.commodity!=None:
            self.shap_interaction_df = pd.DataFrame(x.reshape(x.shape[0]*x.shape[1],x.shape[2]), data_ph.index, data.columns)
        else:
            self.shap_interaction_df = pd.DataFrame(x.reshape(x.shape[0]*x.shape[1],x.shape[2]), data_ph.index, data.columns)

    def get_interactions(self):
        '''
        This code is now depracated in favor of gamma_facet,
        does the same general thing but without the
        orthogonality correction, so the results are a little
        different, but do tell the same story. The gamma_facet
        function has very specific package version requirements
        and it is very tempting to avoid it for that reason.

        Code pulled from Towards Data Science post by
        Tiago Toledo Jr.

        See blog post (1) and github repo (2):
          (1) https://towardsdatascience.com/identifying-global-feature-relationships-with-shap-values-f9e8b2b4121c#:~:text=The%20SHAP%20interaction%20vector%20between%20two%20features%20defines,it%20for%20all%20possibilities%20generates%20the%20interaction%20vector.
          (2) https://github.com/BCG-Gamma/facet

        Appears based on:
          Ittner et al., Feature Synergy, Redundancy,
          and Independence in Global Model Explanations
          using SHAP Vector Decomposition (2021),
          arXiv:2107.12436 [cs.LG]

        Feature synergy matrix: This yields an asymmetric matrix where each row and column represents one
        feature, and the values at the intersections are the pairwise feature synergies,
        ranging from `0.0` (no synergy - both features contribute to predictions fully
        autonomously of each other) to `1.0` (full synergy, both features rely on
        combining all of their information to achieve any contribution to predictions).
        Synergy with self is defined as `1.0`

        This yields an asymmetric matrix where each row and column represents one
        feature, and the values at the intersections are the pairwise feature
        redundancies, ranging from `0.0` (no redundancy - both features contribute to
        predictions fully independently of each other) to `1.0` (full redundancy, either
        feature can replace the other feature without loss of predictive power).
        Redundancy with self is defined as `1.0`
        '''
        shap_values = self.shap_values
        shap_interaction_values = self.shap_interaction_values
        # Define matrices to be filled
        s = np.zeros((shap_values.shape[1], shap_values.shape[1], shap_values.shape[0]))
        a = np.zeros((shap_values.shape[1], shap_values.shape[1], shap_values.shape[0]))
        r = np.zeros((shap_values.shape[1], shap_values.shape[1], shap_values.shape[0]))
        i_ = np.zeros((shap_values.shape[1], shap_values.shape[1], shap_values.shape[0]))
        S = np.zeros((shap_values.shape[1], shap_values.shape[1]))
        R = np.zeros((shap_values.shape[1], shap_values.shape[1]))
        I = np.zeros((shap_values.shape[1], shap_values.shape[1]))

        for i in np.arange(0,shap_values.shape[1]):
            for j in np.arange(0,shap_values.shape[1]):
                # Selects the p_i vector -> Shap Values vector for feature i
                pi = shap_values[:, i]
                # Selects pij -> SHAP interaction vector between features i and j
                pij = shap_interaction_values[:, i, j]

                # Other required vectors
                pji = shap_interaction_values[:, j, i]
                pj = shap_values[:, j]

                # Synergy vector
                s[i, j] = (np.inner(pi, pij) / np.linalg.norm(pij)**2) * pij
                s[j, i] = (np.inner(pj, pji) / np.linalg.norm(pji)**2) * pji
                # Autonomy vector
                a[i,j] = pi - s[i, j]
                a[j,i] = pj - s[j, i]
                # Redundancy vector
                r[i,j] = (np.inner(a[i, j], a[j, i]) / np.linalg.norm(a[j, i])**2) * a[j, i]
                r[j,i] = (np.inner(a[j, i], a[i, j]) / np.linalg.norm(a[i, j])**2) * a[i, j]
                # Independece vector
                i_[i, j] = a[i, j] - r[i, j]
                i_[j, i] = a[j, i] - r[j, i]

                # Synergy value
                S[i, j] = np.linalg.norm(s[i, j])**2 / np.linalg.norm(pi)**2
                # Redundancy value
                R[i, j] = np.linalg.norm(r[i, j])**2 / np.linalg.norm(pi)**2
                # Independence value
                I[i, j] = np.linalg.norm(i_[i, j])**2 / np.linalg.norm(pi)**2

        param_map = make_parameter_names_nice(self.shap_values_df.columns)
        param_names = [param_map[i] for i in self.shap_values_df.columns]
        self.synergy = pd.DataFrame(S, param_names, param_names)
        self.redundancy = pd.DataFrame(R, param_names, param_names)
        self.independence = pd.DataFrame(I, param_names, param_names)

    def plot_shap(self, which=['summary','waterfall','heatmap','shap_vs_param'], figsize=None, dpi=100, **kwargs):
        '''
        figsize can be given as an input, must be a list of
        tuples with length equal to the number of values passed
        in `which`


        --------------------------
        kwargs not shown in the main parameter list:
        scenario_number: int, used in waterfall_plot
          function, is the scenario being plotted
        highlight_best: bool, used in shap_vs_param_plot
          function, whether to identify the sample with
          lowest RMSE, from the predicted RMSEs and actual
        n_plots: int, used in shap_vs_param_plot
          function. If there are many parameters in a given
          model/tuning, can be used to limit the number of
          plots generated
        '''
        if type(which)==str: which = [which]

        self.fig_list = []
        for s, w in enumerate(which):
            fig = plt.figure()
            if w=='summary':
                self.summary_plot()
            elif w=='waterfall':
                n = kwargs['scenario_number'] if 'scenario_number' in kwargs.keys() else None
                self.waterfall_plot(scenario_number=n)
            elif w=='heatmap':
                self.heatmap_plot()
                fig.axes[1].remove()
            elif w=='shap_vs_param':
                highlight_best = kwargs['highlight_best'] if 'highlight_best' in kwargs.keys() else None
                n_plots = kwargs['n_plots'] if 'n_plots' in kwargs.keys() else None
                fig = self.shap_vs_param_plot(highlight_best=highlight_best, n_plots=n_plots)
            if figsize!=None:
                fig.set_size_inches(figsize[s][0],figsize[s][1])
            fig.set_dpi(dpi)
            self.fig_list += [fig]
            plt.show()
            plt.close()

    def summary_plot(self):
        """
        Does the SHAP summary plot (the one with lots of
        little colored dots across lines for each parameter)
        """
        shap.summary_plot(self.shap_values_object,self.data,show=False)

    def waterfall_plot(self, scenario_number=None):
        """
        Does the SHAP waterfall plot for a single data point
        given by scenario number (int)
        """
        scenario_number=0 if scenario_number==None else scenario_number
        shap.plots._waterfall.waterfall_legacy(self.explainer.expected_value[0], self.shap_values[scenario_number], self.X_df.iloc[scenario_number],show=False)

    def heatmap_plot(self):
        """
        Does the SHAP heatmap plot, which is fairly similar to
        the summary plot in terms of information, as far as I
        can tell
        """
        shap.plots.heatmap(self.shap_values_object,show=False)

    def shap_vs_param_plot(self, highlight_best=None, n_plots=None):
        """
        plots the SHAP values as functions of the corresponding parameter
        value. Similar to an unstacked version of the summary plot.

        highlight_best: bool or str in [True,False,'max'], where False
            means the best scenario will not be highlighted, True means
            the minimum objective value will be highlighted, and 'max'
            means the maximum objective value will be highlighted
        n_plots: number of plots to show, can be used to limit how many
            parameters you display but not typically useful.
        """
        highlight_best = False if highlight_best is None else highlight_best
        cols = self.shap_values_df.columns
        n_plots = len(cols) if n_plots==None else n_plots
        cols = cols[:n_plots]
        fig,ax = easy_subplots(n_plots)
        name_map = make_parameter_names_nice(cols)
        data = self.test_prescale.copy()

        for param,a in zip(cols,ax):
            a.scatter(data[param],self.shap_values_df[param],label='all')
            if highlight_best:
                min_rmse = data['predicted '+self.objective].idxmin()
                a.scatter(data[param][min_rmse],self.shap_values_df[param][min_rmse],marker='s',s=200,label='pred. min. RMSE')
                if highlight_best=='max':
                    min_rmse = data[self.objective].idxmin()
                else:
                    min_rmse = data[self.objective].idxmin()
                a.scatter(data[param][min_rmse],self.shap_values_df[param][min_rmse],marker='^',s=200, label='actual min. RMSE')

            line_series = pd.concat([self.shap_values_df[param],data[param]],axis=1,keys=['shap values','parameter values']).set_index('parameter values').sort_index()
            a.plot(line_series.rolling(5,min_periods=1,center=True).mean(),color='k',alpha=0.3,zorder=0)

            a.set(xlabel='Parameter value', ylabel='SHAP value')
            a.set_title(name_map[param],weight='bold')
            if highlight_best: a.legend()
        fig.tight_layout()
        return fig

    def gamma_facet(self):
        '''
        see the feature_interactions function
        docstring for full information.

        requires:
        scikit-learn==1.0.2
        shap==0.39.0
        '''
        # sklearn imports
        from sklearn.metrics import (
            classification_report,
            confusion_matrix,
            roc_curve,
            roc_auc_score,
            auc,
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            precision_recall_curve,
            ConfusionMatrixDisplay,
            PrecisionRecallDisplay,
        )
        from sklearn.model_selection import RepeatedKFold

        # some helpful imports from sklearndf
        from sklearndf.pipeline import RegressorPipelineDF
        from sklearndf.regression import RandomForestRegressorDF

        # relevant FACET imports
        from facet.data import Sample
        from facet.selection import LearnerRanker, LearnerGrid

        get_X_df_y_df(self.many, commodity=self.commodity, standard_scaler=self.standard_scaler, objective=self.objective)
        if self.objective is None:
            self.objective='score'
        self.many.y_df = self.many.y_df.rename(columns={self.many.y_df.columns[0]:self.objective})
        self.facet_data = pd.concat([self.many.X_df.copy(),self.many.y_df.copy()],axis=1)
        # initial data wrangling
        # self.facet_data = self.facet_data.T
        self.facet_data.columns = [i for i in self.facet_data.columns]
        self.facet_data.index = [i for i in self.facet_data.index]
        r = [i for i in self.facet_data.columns if np.any([j in i and j!=self.objective for j in ['R2','RMSE','score','region_specific_price_response','Region specific']])]
        self.facet_data = self.facet_data.drop(columns=r)

        # creating the sample object
        sample = Sample(observations=self.facet_data.fillna(0),
                        feature_names=self.facet_data.drop(columns=self.objective).columns,
                        target_name=self.objective,)

        # create a (trivial) pipeline for a random forest regressor
        rnd_forest_reg = RegressorPipelineDF(
            regressor=RandomForestRegressorDF(n_estimators=200, random_state=42)
        )

        # define grid of models which are "competing" against each other
        rnd_forest_grid = [
            LearnerGrid(
                pipeline=rnd_forest_reg,
                learner_parameters={
                    "min_samples_leaf": [8, 11, 15],
                    "max_depth": [4, 5, 6],
                }
            ),
        ]

        # create repeated k-fold CV iterator
        rkf_cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)

        # rank your candidate models by performance (default is mean CV score - 2*SD)
        self.ranker = LearnerRanker(
            grids=rnd_forest_grid, cv=rkf_cv, n_jobs=-3
        ).fit(sample=sample)

        # get summary report
        self.summary_report = self.ranker.summary_report()

        # fit the model inspector
        from facet.inspection import LearnerInspector
        self.inspector = LearnerInspector(n_jobs=-3)
        self.inspector.fit(crossfit=self.ranker.best_model_crossfit_)

        # get matrices
        self.synergy_matrix = self.inspector.feature_synergy_matrix()
        self.redundancy_matrix = self.inspector.feature_redundancy_matrix()

        map_names = make_parameter_names_nice(self.synergy_matrix.index)
        self.synergy_matrix.rename(index=map_names,columns=map_names,inplace=True)
        self.redundancy_matrix.rename(index=map_names,columns=map_names,inplace=True)

        # From the paper S_ij + R_ij + I_ij = 1, so
        self.independence_matrix = 1 - self.synergy_matrix - self.redundancy_matrix
        for i in self.independence_matrix.index:
            self.independence_matrix.loc[i,i] = 0

        self.names = self.synergy_matrix.index
        self.synergy_matrix = self.synergy_matrix.loc[self.names,self.names]
        self.redundancy_matrix = self.redundancy_matrix.loc[self.synergy_matrix.index, self.synergy_matrix.index]
        self.independence_matrix = self.independence_matrix.loc[self.synergy_matrix.index, self.synergy_matrix.index]
        setattr(self.many, str(self.commodity)+'_synergy_matrix', self.synergy_matrix)
        setattr(self.many, str(self.commodity)+'_redundancy_matrix', self.redundancy_matrix)
        setattr(self.many, str(self.commodity)+'_independence_matrix', self.independence_matrix)

def draw_facet_dendrogram(self):
    """
    visualise redundancy using a dendrogram,
    need to make sure SHAP.gamma_facet has
    already been run.
    """
    #
    from pytools.viz.dendrogram import DendrogramDrawer
    redundancy = self.inspector.feature_redundancy_linkage()
    DendrogramDrawer().draw(data=redundancy, title="Redundancy Dendrogram")

def plot_sri_matrices(self, n_param=3, width_scale=1, width_ratio_scale = 0.83, dpi = 100, cmap='viridis_r'):
    '''
    notes: the synergy matrix shows that from the perspective of nearly
    any parameter (rows), the information in "incentive pool opening
    probability" is required to predict RMSE well, e.g. for "price lag
    used for incentive pool tonnage," about 60% of its information is
    combined with opening probability, so it cannot predict RMSE well
    on its own. High values on the diagonal indicate the parameter is
    minimally informed by other parameters, since the rows sum to one. Each
    row is showing the total contribution to prediction, and the diagonal
    should align with feature importance.

    In the redundancy matrix, the row is still the "perspective from" feature.

    It also appears that the information contained in the three different
    plots is largely redundant in itself.

    ------------------------
    self: SHAP object (need SHAP.gamma_facet to have run before this function)
    n_params: int, can be 1-3, allows plotting of synergy, redundancy, and
        independence in that order (e.g. if =1, only plots synergy)
    width_ratio_scale: float, used to change the relative widths of the
        subplots since the colorbar makes the rightmost figure narrower
    dpi: int, dots per square inch / figure resolution
    ------------------------
    Returns figure

    '''
    dfs = [self.synergy_matrix,self.redundancy_matrix,self.independence_matrix][:n_param]
    names = ['Synergy','Redundancy','Independence'][:n_param]
    fig,ax = easy_subplots(dfs,use_subplots=True,
                            width_scale=width_scale,
                            width_ratios=[width_ratio_scale,width_ratio_scale,1][-n_param:],
                            sharey=True)
    for i,a,name in zip(dfs,ax,names):
        i = i.sort_index().T.sort_index().T
        b = sns.heatmap(i,ax=a,annot=True,fmt='.2f',annot_kws={'fontsize':16},cbar=(name==names[-1]),xticklabels=True,yticklabels=True,cmap=cmap)
        a.set_title(name,weight='bold')
        a.set(ylabel='Feature' if name==names[0] else '',xlabel='Feature')
    if len(self.synergy_matrix.index)>11:
        fig.set_size_inches((12*n_param+6)*width_scale,16)
    else:
        fig.set_size_inches((7*n_param+6)*width_scale,12)
    fig.set_dpi = dpi
    fig.tight_layout(pad=1)
    return fig

def plot_all_sri_matrices(many, commodities=None, standard_scaler=True, dummies=False, split_frac=0.6, Regressor=RandomForestRegressor, use_train_data=False, objective=None, n_param=3, width_scale=1, cmap='viridis_r', dpi=50):
    """
    Rather than needing to initialize the SHAP function every time,
    can just run this function to get all the SRI matrices. Takes
    essentially the same inputs as the SHAP initialization. Will run
    your computer hard.

    many: instance of the Many class, with data loaded
    commodity: (None | list), list of commodity names, lower case. None
        causes all commodities to be used
    standard_scaler: bool, whether to run
        sklearn.preprocessing.StandardScaler on each column of data such
        that it is normalized to have mean 0 and std 1
    dummies: bool, whether to include commodity-level dummy variables
        (e.g. a column of zeros with 1s for all Ag scenarios)
    split_frac: float (0,1), for train-test split in creating the ML
        model for the tree-based regression we get SHAP values from
    Regressor: tree-based regression class, can be RandomForestRegressor,
        ExtraTreesRegressor, GradientBoostingRegressor, likely others;
        regression to use for the tree-based regression that we get SHAP values from
    use_train_data: bool, whether to use the train data rather than test
        data for shap.TreeExplainer(regression_model).explainer(data).
        We are supposed to use the test data, so leave this one False.
    objective:  None or str. If str, has to be one of the columns in
        multi_scenario_results_formatted, otherwise will be using `score`
        or `RMSE` from rmse_df dataframe depending on whether using the
        Integration or demandModel formulation. If str, will try to do
        the mean difference from baseline.
    """
    if commodities is None:
        if hasattr(many,'rmse_df'):
            commodities = list(many.rmse_df.index.get_level_values(0).unique())+[None]
        elif hasattr(many,'multi_scenario_results'):
            commodities = list(many.multi_scenario_results.index.get_level_values(0).unique())+[None]
        else:
            raise ValueError('either the Many object fed into this function needs to have rmse_df or multi_scenario_results objects, or you need to provide a list of commodities')

    for comm in commodities:
        print(comm)
        sh1 = SHAP(many,commodity=comm, standard_scaler=standard_scaler, dummies=dummies, split_frac=split_frac, Regressor=Regressor, use_train_data=use_train_data, objective=objective)
        sh1.gamma_facet()
        plot_sri_matrices(sh1, n_param=n_param, width_scale=width_scale, width_ratio_scale=width_ratio_scale, cmap=cmap, dpi=dpi)
        plt.show()
        plt.close()

def plot_best_scenario_sd(self, commodity='aluminum', plot_supply_demand_stack=True, best=0, scrap_scenario=0, legend=True, end_year=2040):
    """
    Plots the many different variables and SD imbalances for
    a given scenario number (or several). Can be run on either
    future runs or historical tuning runs (if the Many object
    you pass has a multi_scenario_results object, it will do
    future runs). If doing historical tuning, can set
    best = -1 to have it pick the best-score scenario for you;
    scrap scenario will have no effect for historical tuning.

    e.g. looking at how well the historical part of the runs went:
    plot_best_scenario_sd(many_act_hist,commodity='nickel',
        best=np.arange(0,100),scrap_scenario=0,
        plot_supply_demand_stack=False,legend=False,end_year=2019);

    self: Many object that must have a multi_scenario_results
        variable or rmse_df and results variables
    commodity: str, lowercase full name of commodity as in
        multi_scenario_results index level 0
    plot_supply_demand_stack: bool, whether to plot the stackplot
        of refined production distribution and line for consumption
    best: int or list, corresponding to the scenario number/hyperparameter
        set in multi_scenario_results (level 1 of multi_scenario_results index)
    scrap_scenario: numerical representation of which scrap scenario to select
        (level 2 of multi_scenario_results index), doesn`t get used if doing
        the historical tuning version.
    end_year: int, typically either 2019 or 2040
    """
    if hasattr(self,'multi_scenario_results'):
        if type(best)!=int:
            best = self.multi_scenario_results.index.get_level_values(1).unique()[self.multi_scenario_results.index.get_level_values(1).unique().isin(best)]
        results = self.multi_scenario_results.loc[commodity].loc[idx[best,scrap_scenario,:],:].copy().dropna(how='all')
        while results.index.nlevels>1:
            results = results.droplevel(0)
    else:
        if type(best)!=int:
            best = self.rmse_df.columns[self.rmse_df.columns.isin(best)]
        elif type(best)==int and best==-1:
            best = int(self.rmse_df.loc[commodity].loc['score'].idxmin())
        results = self.results.loc[commodity].loc[best]
    if type(best)!=int:
        results = results.unstack(0).droplevel(0)
    variables = ['Total','Conc. supply','Conc. demand','Ref.','Scrap','Spread','TCRC','Refined','CU','SR','Direct','Mean total','mine grade','Conc. SD','Ref. SD','Scrap SD']
    fig,ax=easy_subplots(variables)
    for var,a in zip(variables,ax):
        if type(best)!=int:
            parameters = [i for i in results.columns.get_level_values(0).unique() if var in i and ('SD' not in i or 'SD' in var)]
        else:
            parameters = [i for i in results.columns if var in i and ('SD' not in i or 'SD' in var)]
        res = results[parameters].loc[2001:end_year]
        resdict = get_unit(res,res,parameters[0])
        res, unit = resdict['simulated'], resdict['unit']
        param_str = ', '.join(parameters) if len(', '.join(parameters))<30 else ',\n'.join(parameters)
        res.plot(ax=a,title=f'{param_str} ({unit})',legend=legend)
    fig.tight_layout()


    if plot_supply_demand_stack:
        fig1 = plt.figure()
        res = results[['Pri. ref. prod.','Sec. ref. prod.','Direct melt','Total demand']].loc[2001:]
        res = res.replace(0,np.nan)
        resdict = get_unit(res,res,parameters[0])
        res, unit = resdict['simulated'], resdict['unit']
        res = res.fillna(0)
        plt.plot(res['Total demand'],label='Total demand',color='k')
        plt.stackplot(res.index, res[['Pri. ref. prod.','Sec. ref. prod.','Direct melt']].T, labels=['Pri. ref.','Sec. ref.','Direct melt']);
        if legend:
            plt.legend(framealpha=0.5,frameon=True);
        plt.title(f'Total supply-demand imbalance ({unit})')
        return fig, fig1
    return fig

def stackplot_scrap_demand(many, commodity='aluminum', hyperparam_set=0, scrap_scenario=18):
    """
    Used to show how the additional scrap demand and rest of
    market scrap demand evolve over time, for a given scrap
    demand scenario (stackplot) and showing baseline as a line.
    Returns the figure, axes, and results dataframe in that order.

    many: Many object with multi_scenario_results variable
    commodity: str, lowercase full name as in multi_scenario_results
        index level 0
    hyperparam_set: level 1 of multi_scenario_results for plotting
    scrap_scenario: level 2 of multi_scenario results for plotting

    Can check what the above two mean using the
    many.multi_scenario_hyperparam variable.
    """
    results = many.multi_scenario_results.copy().loc[commodity].loc[idx[:,scrap_scenario],:].copy().dropna(how='all')
    res0 = many.multi_scenario_results.copy().loc[commodity].loc[idx[:,0],:].copy().dropna(how='all')
    dicty = get_unit(results['Scrap demand'], res0['Scrap demand'], 'scrap demand')
    resultsy,res0y,unit = [dicty[i] for i in ['simulated','historical','unit']]
    # results *= (resultsy/results['Scrap demand']).mean()
    # res0 *= (res0y/res0['Scrap demand']).mean()
    without = (results['Direct melt']-results['Additional direct melt']).loc[idx[:,:,2001:]]
    # results['Additional direct melt'].loc[idx[:,:,2001:]].unstack().T
    # without.unstack().T.plot()
    results['Rest of market'] = results['Direct melt']-results['Additional direct melt']
    ph = results[['Rest of market','Additional direct melt']].loc[hyperparam_set,scrap_scenario,2001:].droplevel([0,1])
    fig,ax = easy_subplots(3)
    ax[0].stackplot(ph.index, ph.T,labels=ph.columns)
    ax[0].plot(res0['Direct melt'].loc[hyperparam_set,0,2001:].droplevel([0,1]),color='k',label='Baseline demand')
    ax[0].legend(loc='upper left',fontsize=16)
    ax[0].set(title='Direct melt scrap demand',ylabel=f'Scrap demand ({unit})',xlabel='Year')

    without = (results['Sec. ref. cons.']-results['Additional secondary refined']).loc[idx[:,:,2001:]]
    # results['Additional direct melt'].loc[idx[:,:,2001:]].unstack().T
    # without.unstack().T.plot()
    results['Rest of market'] = results['Sec. ref. cons.']-results['Additional secondary refined']
    ph = results[['Rest of market','Additional secondary refined']].loc[hyperparam_set,scrap_scenario,2001:].droplevel([0,1])
    ax[1].stackplot(ph.index, ph.T,labels=ph.columns)
    ax[1].plot(res0['Sec. ref. cons.'].loc[hyperparam_set,0,2001:].droplevel([0,1]),color='k',label='Baseline demand')
    ax[1].legend(loc='upper left',fontsize=16)
    ax[1].set(title='Refined scrap demand',ylabel=f'Scrap demand ({unit})',xlabel='Year')

    without = (results['Scrap demand']-results[['Additional secondary refined','Additional direct melt']].sum(axis=1)).loc[idx[:,:,2001:]]
    # results['Additional direct melt'].loc[idx[:,:,2001:]].unstack().T
    # without.unstack().T.plot()
    results['Rest of market'] = results['Scrap demand']-results[['Additional secondary refined','Additional direct melt']].sum(axis=1)
    ph = results[['Rest of market','Additional secondary refined','Additional direct melt']].loc[hyperparam_set,scrap_scenario,2001:].droplevel([0,1])
    ax[2].stackplot(ph.index, ph.T,labels=ph.columns)
    ax[2].plot(res0['Scrap demand'].loc[hyperparam_set,0,2001:].droplevel([0,1]),color='k',label='Baseline demand')
    ax[2].legend(loc='upper left',fontsize=16)
    ax[2].set(title='Total scrap demand',ylabel=f'Scrap demand ({unit})',xlabel='Year')
    fig.tight_layout()
    return fig,ax,results

def plot_future_line_and_hist_one_commodity(many, commodity, parameter, restrict=0.95, use_sns=False, plot_historical=False, color='tab:blue', fig=None, ax=None, dpi=50):
    """
    Plots the combined set of transparent lines for historical-future time
    progression of the given parameter and commodity, with a histogram aligning
    with the y axis that shows the 2040 value distribution.

    --------
    many: Many() object, must have multi_scenario_results object
    commodity: str, lowercase commodity form
    parameter: str, any column in many.multi_scenario_results
    restrict: float or False, whether to restrict the y-axis limits to exclude
        outliers, with float values corresponding to the percentile being
        plotted (e.g. 0.95 causes the middle 95% of max/min values to be shown)
    use_sns: bool, no functionality yet but figured it would be to have the
        sns.lineplot shading functionality
    plot_historical: bool, whether to show the historical values alongside the
        simulated
    color: str, matplotlib color for the lines and histogram
    fig: figure on which to plot
    ax: axes on which to plot, must be a list of length 2, as the histogram
        plots on ax[1]
    dpi: int, dots per square inch, controls figure resolution
    ----------

    Returns: comm (the data being plotted), fig
    """
    if ax is None:
        fig,ax=easy_subplots(2,2,width_ratios=[4,1], sharey=False, width_scale=0.6)
    comm = many.multi_scenario_results.loc[commodity][parameter].droplevel(0)
    comm = comm.unstack().T.loc[2001:]
    hist_data_dict = dict(zip(many.objective_results_map.values(), many.objective_results_map.keys()))
    if parameter in hist_data_dict and plot_historical:
        hist = many.historical_data.loc[commodity][hist_data_dict[parameter]].loc[2001:2019]
        dicty = get_unit(comm, hist, parameter)
        comm = dicty['simulated']
        hist = dicty['historical']
        unit = dicty['unit']
    else:
        dicty = get_unit(comm, comm, parameter)
        comm = dicty['simulated']
        unit = dicty['unit']

    comm.plot(legend=False, alpha=0.05, linewidth=1, color=color,
              ylabel=f'{parameter} ({unit})',
              xlabel='Year',
              ax=ax[0],
             ).grid(axis='x')

    if parameter in hist_data_dict and plot_historical:
        hist.plot(
                color='k',
                alpha=0.5,
                ax=ax[0],
            ).grid(axis='x')
    if restrict:
        restrict = 1-(1-restrict)/2
        booly = (comm.min()>-comm.min().quantile(1-restrict)) &\
                (comm.max()<comm.max().quantile(restrict))
        restricted = booly[booly].index
    else: restricted = comm.columns
    comm.loc[2040][restricted].plot.hist(orientation='horizontal', ax=ax[1],
                             color=color
                            ).grid(axis='x')
    if restrict:
        ax[0].set(ylim=(-comm.min().quantile(1-restrict), comm.max().quantile(restrict)))
        ax[1].set(ylim=(-comm.min().quantile(1-restrict), comm.max().quantile(restrict)))

    ax[1].set(xlabel='2040 Freq.', yticklabels=[])

    fig.suptitle(f'{commodity.capitalize()} {parameter.lower()}',y=0.9, x=0.58, weight='bold')
    fig.tight_layout()
    fig.set_dpi(dpi)
    plt.show()
    plt.close()
    return comm, fig

def pval_to_star(pval, no_star_cut=0.1, period_cut=0.05, one_star_cut=0.01, two_star_cut=0.001):
    """
    Converts a value from its numerical value to a string where:
    *** < 0.001 < ** < 0.01 < * < 0.05 < . < 0.1
    """
    pval_str = '***' if pval < two_star_cut else '**' if pval < one_star_cut else '*' if \
        pval < no_star_cut else '(.)' if pval < period_cut else ''
    return pval_str

def make_parameter_mean_std_table(many, n_best, value_in_parentheses='standard error', stars='ttest'):
    """
    many: Many instance, from tuning, either the integ object or full thing is has integ
        object
    n_best: int, number of best scenarios to use in the calculation of mean and std error/
        std dev / variance
    value_in_parentheses: str, can be `standard error`, `standard deviation`, or `variance`
    """
    if hasattr(many, 'integ'):
        rmse_df = many.integ.rmse_df_sorted.copy()
    else:
        rmse_df = many.rmse_df_sorted.copy()
    r = [i for i in rmse_df.index.get_level_values(1).unique()
         if np.any([j in i for j in ['score', 'R2', 'RMSE', 'region_specific_price_response']])]
    rmse_df.drop(r, inplace=True, level=1)
    best_n = rmse_df.loc[:, :n_best]
    means = best_n.mean(axis=1).unstack(0).fillna('')
    if value_in_parentheses == 'standard error':
        stds = best_n.sem(axis=1).unstack(0).fillna('')
    elif value_in_parentheses == 'standard deviation':
        stds = best_n.std(axis=1).unstack(0).fillna('')
    elif value_in_parentheses == 'variance':
        stds = best_n.var(axis=1).unstack(0).fillna('')

    if stars == 'ttest':
        pvals = best_n.apply(lambda x: stats.ttest_1samp(x, popmean=0)[1], axis=1)
        display(pvals)
    elif stars == 'uniform':
        pvals = get_difference_from_uniform(rmse_df)
    else:
        raise ValueError('stars input must be either ttest or uniform')
    stars = pvals.apply(pval_to_star)

    demand_pretune = ['sector_specific_dematerialization_tech_growth', 'sector_specific_price_response',
                      'intensity_response_to_gdp']
    mining_pretune = ['primary_oge_scale', 'mine_cu_margin_elas', 'mine_cost_og_elas',
                      'mine_cost_tech_improvements', 'mine_cost_price_elas', 'initial_ore_grade_decline',
                      'primary_price_resources_contained_elas', 'incentive_opening_probability', 'close_years_back',
                      'reserves_ratio_price_lag', 'mine_cost_change_per_year', 'incentive_mine_cost_change_per_year']
    mean_std = pd.DataFrame(np.nan, means.index, means.columns)
    for i in mean_std.index:
        for c in mean_std.columns:
            if means[c][i] != '':
                mean_std.loc[i, c] = '{:.3f}{:s} ({:.3f})'.format(means[c][i], stars[c][i], stds[c][i])
            else:
                mean_std.loc[i, c] = ' '
    mean_std = mean_std.T
    params_nice = make_parameter_names_nice(mean_std.columns)

    def convert_param_names(v):
        if v in mining_pretune:
            return ('Mining parameters', params_nice[v])
        elif v in demand_pretune:
            return ('Demand parameters', params_nice[v])
        else:
            return ('Integration parameters', params_nice[v])

    mean_std = mean_std.rename(columns=dict(zip(mean_std.columns,
                                                [convert_param_names(i) for i in mean_std.columns])))
    mean_std = mean_std.rename(dict(zip(mean_std.index, [many.commodity_element_map[i.capitalize()]
                                                         for i in mean_std.index])))
    mean_std.columns = pd.MultiIndex.from_tuples(mean_std.columns)
    means = means.T.replace('', np.nan)
    means = means.rename(columns=dict(zip(means.columns,
                                          [convert_param_names(i) for i in means.columns])))
    means = means.rename(dict(zip(means.index, [many.commodity_element_map[i.capitalize()]
                                                for i in means.index])))
    means.columns = pd.MultiIndex.from_tuples(means.columns)
    return mean_std.T.sort_index().T, means.sort_index().T.sort_index().T

def get_difference_from_uniform(rmse_df):
    for_test = rmse_df.copy()

    uniform_rvs = pd.DataFrame()
    loc = 0
    scale = 1
    for rs in np.arange(0, 100):
        uniform_rvs[rs] = stats.uniform.rvs(loc=loc, scale=scale, size=25, random_state=rs)

    def ks100_uniform(array, size=1000):
        """
        Size is the size of the random variable generation from the uniform distribution
        """
        ks_pvals = []
        name = array.name[1]
        loc = 0
        scale = 1
        if name == 'incentive_opening_probability':
            scale = 0.5
        elif name in ['mine_cost_change_per_year', 'incentive_mine_cost_change_per_year']:
            loc = -5
            scale = 10
        elif name == 'sector_specific_dematerialization_tech_growth':
            loc = -0.1
            scale = 0.2
        elif name == 'intensity_response_to_gdp':
            loc = -0.5
            scale = 1.5
        elif name == 'sector_specific_price_response':
            loc = -0.6
            scale = 0.6
        elif np.max(array) < 0:
            loc = -1
        if size is not None:
            for rs in np.arange(0, 100):
                if True:
                    randoms = stats.uniform.rvs(loc=loc, scale=scale, size=size, random_state=rs)
                    val = stats.kstest(array, randoms)[1]
                else:
                    val = stats.kstest(array, uniform_rvs[rs])[1]
                ks_pvals += [val]
            return np.mean(ks_pvals)
        else:
            randoms = stats.uniform.rvs(loc=loc, scale=scale, size=1000, random_state=0)
            val = stats.kstest(array, randoms)[1]
            return val

    applied = for_test.apply(ks100_uniform, axis=1)
    return applied

def plot_violin_all(many, mining_or_integ='integ', percentile=25, n=25, n_most_important=100, legend=True,
                    normalize=False, dpi=50):
    """
    The huge violinplot.

    many: Many instance
    mining_or_integ:
    percentile: float, selects the number of parameter sets to include, so
      percentile=25 would take the best-fitting 25% of parameter sets
    n: int, overrides percentile to allow you to select the number of best
      scenarios to plot directly.
    n_most_important: int, number of most important parameters to plot
    legend: bool, whether to plot the legend on the plot
    scale_fig_width: float, can be used to widen or narrow the plot, default 1
    split_params: bool, whether to split different parameters to plot on
        different subplot
    dpi: int, dots per inch, figure resolution (higher = better)

    Returns fig, ax, df
    """
    plot_important_parameter_scatter(many, mining_or_integ=mining_or_integ, percentile=percentile, n=n,
                                     n_most_important=n_most_important, legend=legend, normalize=normalize, dpi=dpi,
                                     plot=False)

    df_use = many.df2.loc[
        (many.df2.Parameter != 'Price lag used for incentive pool tonnage')
        & (many.df2.Parameter != 'Prior years used for price prediction')]
    df_use.loc[(df_use['Parameter'] == 'Incentive mine cost change per year') | (
                df_use['Parameter'] == 'Mine cost change per year'), 'Value'] /= 5
    df_use.loc[(df_use['Parameter'] == 'Intensity elasticity to time'), 'Value'] *= 10
    df_use.loc[df_use[
                   'Parameter'] == 'Incentive mine cost change per year', 'Parameter'] = r'$\frac{Incentive\:mine\:cost\:change\:per\:year}{5}$'
    df_use.loc[
        df_use['Parameter'] == 'Mine cost change per year', 'Parameter'] = r'$\frac{Mine\:cost\:change\:per\:year}{5}$'
    df_use.loc[df_use['Parameter'] == 'Intensity elasticity to time', 'Parameter'] = '10 x Intensity elasticity to time'
    fig, a = plt.subplots(figsize=(10, 60))
    if hasattr(many, 'integ') and mining_or_integ == 'integ':
        order_many = many.integ
    else:
        order_many = many
    order = order_many.importances_df['Mean no dummies'].sort_values(ascending=False).index
    order = [make_parameter_names_nice(order)[i].replace(
        'Intensity elasticity to time', '10 x Intensity elasticity to time').replace(
        'Mine cost change per year', r'$\frac{Mine\:cost\:change\:per\:year}{5}$').replace(
        'Incentive mine cost change per year', r'$\frac{Incentive\:mine\:cost\:change\:per\:year}{5}$')
        for i in order]
    order = [i for i in order if i in df_use['Parameter'].unique()]

    sns.violinplot(data=df_use, y='Parameter', x='Value', hue='Commodity', ax=a, linewidth=2,
                   cut=0, orient='h', order=order)
    a.tick_params(axis='x', rotation=90)
    a.grid(axis='x')
    if not legend:
        a.get_legend().remove()
    return fig, a, df_use

def run_demand_pretuning():
    """
    Mostly for keeping track of things.

    demand pre-tuning scenario runs for showing statistical signficance?

    Order is somewhat important - the last run executed informs the values in
    updated_commodity_inputs (or updated_commodity_inputs_unconstrained) for
    future use, since we overwrite that file with every run.
    """
    mod = Many()
    mod.run_all_demand(100, constrain_tuning_to_sign=True, filename_modifier='_constrain_mcpe0',n_parallel=5)

    mod = Many()
    mod.run_all_demand(300, constrain_tuning_to_sign=True, filename_modifier='_constrain3_mcpe0',n_parallel=5)

    mod = Many()
    mod.run_all_demand(100, constrain_tuning_to_sign=False, filename_modifier='_unconstrain1_mcpe0',n_parallel=5)

    mod = Many()
    mod.run_all_demand(100, constrain_tuning_to_sign=False, filename_modifier='_unconstrain_mcpe0',n_parallel=5)

    mod = Many()
    mod.run_all_demand(300, constrain_tuning_to_sign=False, filename_modifier='_unconstrain3_mcpe0',n_parallel=5)

def run_mining_pretuning():
    """
    Mostly for keeping track of everything
    """
    to_run = ['Au','Al','Ag','Zn','Pb','Sn','Cu','Ni','Steel']
    mod = Many()
    mod.run_all_mining(200,commodities=None, constrain_tuning_to_sign=False, filename_modifier='_unconstrain1_mcpe0', n_parallel=2)

    mod = Many()
    mod.run_all_mining(200,commodities=None, constrain_tuning_to_sign=False, filename_modifier='_unconstrain_mcpe0', n_parallel=2)

    # removing mine_cost_price_elas variable from the system (setting 0), run 22-11-14
    mod = Many()
    mod.run_all_mining(200,commodities=to_run, constrain_tuning_to_sign=True, filename_modifier='_constrain_mcpe0', n_parallel=2)

def run_integration_tuning():
    """
    Mostly for keeping track of everything

    Order does not matter here since we differentiate between tuned_rmse_df_out
    with tuned_rmse_df_out_append.
    """
    to_run = ['Au','Al','Ag','Zn','Pb','Sn','Cu','Ni','Steel']
    # to_run = ['Ag','Zn','Pb','Al','Au']
    # to_run = ['Ni','Cu','Sn','Steel']

    mod = Many()
    mod.run_all_integration(200, tuned_rmse_df_out_append='_mcpe0_2016',
        train_time=np.arange(2001,2016), simulation_time=np.arange(2001,2020),
        normalize_objectives=True, constrain_previously_tuned=True,
        commodities=to_run, filename_modifier='_mcpe0_2016',
        n_parallel=2)

    mod = Many()
    mod.run_all_integration(200, tuned_rmse_df_out_append='_mcpe0_2015',
        train_time=np.arange(2001,2015), simulation_time=np.arange(2001,2020),
        normalize_objectives=True, constrain_previously_tuned=True,
        commodities=to_run, filename_modifier='_mcpe0_2015',
        n_parallel=2)

    mod = Many()
    mod.run_all_integration(200, tuned_rmse_df_out_append='_mcpe0_unconstrain',
        train_time=np.arange(2001,2020), simulation_time=np.arange(2001,2041),
        normalize_objectives=True, constrain_previously_tuned=True,
        constrain_tuning_to_sign=False,
        commodities=to_run, filename_modifier='_mcpe0_unconstrain',
        n_parallel=2)

    # removing mine_cost_price_elas variable from the system (setting 0), run 22-11-14
    mod = Many()
    mod.run_all_integration(200, tuned_rmse_df_out_append='_mcpe0',
        train_time=np.arange(2001,2020), simulation_time=np.arange(2001,2041),
        normalize_objectives=True, constrain_previously_tuned=True,
        commodities=to_run, filename_modifier='_mcpe0',
        n_parallel=2)

    # keeping mine cost price elas removed, and tuning only the remaining
    # variables that were not pre-tuned, with historical primary commodity price
    # rather than allowing price to be simulated
    mod = Many()
    mod.run_all_integration(n_runs=200, n_params=2, n_jobs=3,
                            tuned_rmse_df_out_append='_mcpe0_histprice',
                            train_time=np.arange(2001,2020),
                            normalize_objectives=True,
                            constrain_previously_tuned=True,
                            commodities=to_run,
                            force_integration_historical_price=True,
                            filename_modifier='_mcpe0_histprice',
                            n_parallel=2)

    # tuning demand and mining, ignoring price tuning to see what we get
    mod = Many()
    mod.run_all_integration(n_runs=200, n_params=2, n_jobs=3,
                            tuned_rmse_df_out_append='_mcpe0_noprice',
                            train_time=np.arange(2001,2020),
                            normalize_objectives=True,
                            constrain_previously_tuned=True,
                            commodities=to_run,
                            force_integration_historical_price=False,
                            filename_modifier='_mcpe0_noprice',
                            n_parallel=2)

def run_all_the_future_scenarios():
    """
    Mostly for keeping track of everything
    """
    to_run = ['Au','Al','Ag','Zn','Pb','Sn','Cu','Ni','Steel']
    # to_run = ['Ag','Zn','Pb','Al','Au']
    # to_run = ['Ni','Cu','Sn','Steel']

    # Running a bunch of baselines
    run_future_scenarios(commodities=to_run,run_parallel=6,verbosity= -1,
        scenario_name_base='_run_scenario_baselines_test',supply_or_demand=None,
        simulation_time=np.arange(2001,2041), baseline_sampling='clustered',
        n_best_scenarios=30, n_per_baseline=1,
    )

    # Fruity demand scenarios
    run_future_scenarios(commodities=to_run,run_parallel=6,verbosity= -1,
        scenario_name_base='_run_scenario_set_alt_hist_act',supply_or_demand='demand-alt',
        simulation_time=np.arange(2001,2041), baseline_sampling='actual',
        n_best_scenarios=10,
        years_of_increase=np.arange(1,2))

    run_future_scenarios(commodities=to_run,run_parallel=4,verbosity= -1,
        scenario_name_base='_run_scenario_set_alt_hist_act',supply_or_demand='demand-alt',
        simulation_time=np.arange(2001,2041), baseline_sampling='clustered',
        years_of_increase=np.arange(1,2),tuned_rmse_df_out_append='_mcpe0',
        n_best_scenarios=10,n_per_baseline=5
    )
    run_future_scenarios(commodities=to_run,run_parallel=4,verbosity= -1,
        scenario_name_base='_run_scenario_set_alt_hist_act',supply_or_demand='both-alt',
        simulation_time=np.arange(2001,2041), baseline_sampling='clustered',
        years_of_increase=np.arange(1,2),tuned_rmse_df_out_append='_mcpe0',
        n_best_scenarios=10,n_per_baseline=5
    )

    # Looking at what happens if we tune without price
    mod = Many()
    mod.run_all_integration(200, n_params=2, tuned_rmse_df_out_append='_norm_noprice',
        train_time=np.arange(2001,2020), simulation_time=np.arange(2001,2040),
        normalize_objectives=True, constrain_previously_tuned=True,
        commodities=to_run, filename_modifier='_norm_noprice')

    # Looking at what happens when we tune integration with forced historical price
    mod = Many()
    mod.run_all_integration(200, n_params=2, tuned_rmse_df_out_append='_norm_histprice',
        train_time=np.arange(2001,2020), simulation_time=np.arange(2001,2020),
        normalize_objectives=True, constrain_previously_tuned=True,
        force_integration_historical_price=True,
        commodities=to_run, filename_modifier='_norm_histprice')

def update_tuned_rmse_df_out_with_new_data(folder_name, element, tuned_rmse_df_out_append,
                                           generalization_file_path='generalization'):
    m = Many()
    m.load_data(folder_name)
    commodity = m.element_commodity_map[element].lower()
    rmse_ph = m.rmse_df_sorted.stack().unstack(1).rename({element:commodity})
    filename = f'{generalization_file_path}/output_files/tuned_rmse_df_out_{tuned_rmse_df_out_append}.csv'
    tuned = pd.read_csv(filename, index_col=[0,1])
    if commodity in tuned.index.get_level_values(0).unique():
        tuned.drop(commodity,inplace=True)
    rmse = pd.concat([tuned,rmse_ph]).fillna(0)
    rmse.to_csv(filename)