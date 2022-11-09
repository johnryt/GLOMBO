import warnings
from integration_functions import Sensitivity
import numpy as np
import pandas as pd
from useful_functions import *
from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler
import os

import shap
from Individual import *
from datetime import datetime

# warnings.filterwarnings('error')
# np.seterr(all='raise')

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
    - get_unit
    - run_future_scenarios
        - op_run_future_scenarios
        - op_run_sensitivity_fn
        - op_run_future_scenarios_parallel
        - run_scenario_set
    - get_pretuning_params
    - get_train_test_scores
    - get_commodity_scores
    - get_best_columns
    - generate_clustered_hyperparam
    Not all of the above take Many as an input; some are standalone or are
    called by other functions in this file.
    '''
    def __init__(self, data_folder=None, pkl_folder=None):
        '''
        data_folder: str, the folder where historical data needing loading lives
        pkl_folder: str, folder where pkl files of results will be saved
        '''
        self.ready_commodities = ['Al','Au','Sn','Cu','Ni','Ag','Zn','Pb','Steel']
        self.element_commodity_map = {'Steel':'Steel','Al':'Aluminum','Au':'Gold','Cu':'Copper','Steel':'Steel','Co':'Cobalt','REEs':'REEs','W':'Tungsten','Sn':'Tin','Ta':'Tantalum','Ni':'Nickel','Ag':'Silver','Zn':'Zinc','Pb':'Lead','Mo':'Molybdenum','Pt':'Platinum','Te':'Telllurium','Li':'Lithium'}
        self.commodity_element_map = dict(zip(self.element_commodity_map.values(),self.element_commodity_map.keys()))
        self.data_folder = 'generalization/data' if data_folder==None else data_folder
        self.pkl_folder = 'data' if pkl_folder==None else pkl_folder
        self.objective_results_map = {'Total demand':'Total demand','Primary commodity price':'Refined price',
                                 'Primary demand':'Conc. demand','Primary supply':'Mine production',
                                'Conc. SD':'Conc. SD','Scrap SD':'Scrap SD','Ref. SD':'Ref. SD'}

    def run_all_demand(self, n_runs=50, commodities=None, save_mining_info=False, trim_result_df=True, constrain_tuning_to_sign=True, filename_base='_run_hist', filename_modifier=''):
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
        """
        t1 = datetime.now()
        commodities = self.ready_commodities if commodities==None else commodities
        for material in commodities:
            print('-'*40)
            print(material)
            mat = self.element_commodity_map[material].lower()
            filename=f'{self.pkl_folder}/{mat}{filename_base}.pkl'
            self.shist1 = Sensitivity(pkl_filename=filename, data_folder=self.data_folder,changing_base_parameters_series=material,notes='Monte Carlo aluminum run',
                            simulation_time=np.arange(2001,2020),OVERWRITE=True,use_alternative_gold_volumes=True,
                            constrain_tuning_to_sign=constrain_tuning_to_sign,
                                historical_price_rolling_window=5,verbosity=0, trim_result_df=trim_result_df)
            self.shist1.historical_sim_check_demand(n_runs,demand_or_mining='demand')
            print(f'time elapsed: {str(datetime.now()-t1)}')

    def run_all_mining(self, n_runs=50, commodities=None, save_mining_info=False, trim_result_df=True, constrain_tuning_to_sign=True, filename_base='_run_hist', filename_modifier=''):
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
            `_mining`
        filename_modifier: str, filename modifier, comes after the `_mining` but
            before `.pkl`
        """
        commodities = self.ready_commodities if commodities==None else commodities
        for material in commodities:
            t1 = datetime.now()
            print('-'*40)
            print(material)
            mat = self.element_commodity_map[material].lower()
            filename=f'{self.pkl_folder}/{mat}{filename_base}{filename_modifier}.pkl'
            self.shist = Sensitivity(pkl_filename=filename, data_folder=self.data_folder,changing_base_parameters_series=material,notes='Monte Carlo aluminum run',
                            simulation_time=np.arange(2001,2020),OVERWRITE=True,use_alternative_gold_volumes=True,
                                historical_price_rolling_window=5,verbosity=0,
                                constrain_tuning_to_sign=constrain_tuning_to_sign,
                               incentive_opening_probability_fraction_zero=0, save_mining_info=save_mining_info,
                               trim_result_df=trim_result_df)
            self.shist.historical_sim_check_demand(n_runs,demand_or_mining='mining')
            print(f'time elapsed: {str(datetime.now()-t1)}')

    def run_all_integration(self, n_runs=200, tuned_rmse_df_out_append='', commodities=None, train_time=np.arange(2001,2020), simulation_time=np.arange(2001,2020), normalize_objectives=False,constrain_previously_tuned=True, verbosity=0, save_mining_info=False, trim_result_df=True, constrain_tuning_to_sign=True, filename_base='_run_hist', filename_modifier=''):
        """
        Runs parameter tuning, trying to match to historical demand, price, and
        mine production. Tuning uses Bayesian optimization. Saves all results in
        `tuned_rmse_df_out`+tuned_rmse_df_out_append+`.pkl` for use in future
        runs (see run_future_scenarios function)
        ------------
        n_runs: int, number of Bayesian optimization runs until stopping
        tuned_rmse_df_out_append: str, string appended to tuned_rmse_df_out
            filename so you can differentiate tuning results, if needed.
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
        constrain_previously_tuned: bool, if True, requires any bayesian
            optimization tuning parameters that have previously been tuned
            (by historical_sim_check_demand/run_all_demand/run_all_mining,
            meaning they are in the index of self.updated_commodity_inputs(_sub))
            to be 0.001-2X their previously-tuned value, if the optimization
            is trying to tune them. If False, constraints are as they were
            previously. Does not apply to demand variables if
            dont_constrain_demand is True, which is the default for the
            Sensitivity class.
        save_mining_info: bool, whether or not to save mine-level information.
            Default is False, takes up a lot of memory if True.
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
        """
        commodities = self.ready_commodities if commodities==None else commodities
        for material in commodities:
            t1 = datetime.now()
            print('-'*40)
            print(material)
            # timer=IterTimer()
            n=3
            mat = self.element_commodity_map[material].lower()
            filename=f'{self.pkl_folder}/{mat}{filename_base}_all{filename_modifier}.pkl'
            print('--'*15+filename+'-'*15)
            self.s = Sensitivity(pkl_filename=filename, data_folder=self.data_folder,changing_base_parameters_series=material,notes=f'Monte Carlo {material} run',
                            additional_base_parameters=pd.Series(1,['refinery_capacity_growth_lag']),
                            simulation_time=simulation_time, include_sd_objectives=False, train_time=train_time,
                            OVERWRITE=True,verbosity=verbosity,historical_price_rolling_window=5,
                            constrain_tuning_to_sign=constrain_tuning_to_sign,
                            constrain_previously_tuned=constrain_previously_tuned, normalize_objectives=normalize_objectives,
                            save_mining_info=save_mining_info, trim_result_df=trim_result_df)
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
                'mine_cost_tech_improvements',
                'primary_oge_scale',
                'sector_specific_dematerialization_tech_growth',
                'intensity_response_to_gdp',
                'sector_specific_price_response',
            ]
            self.s.run_historical_monte_carlo(n_scenarios=n_runs,bayesian_tune=True,n_params=n,
                sensitivity_parameters=sensitivity_parameters)
            self.s.rmse_df.to_pickle(f'data/tuned_rmse_df_out{tuned_rmse_df_out_append}.pkl')
            print(f'time elapsed: {str(datetime.now()-t1)}')
            # add 'response','growth' to sensitivity_parameters input to allow demand parameters to change again

    def get_variables(self, demand_mining_all='demand',filename_base='_run_hist',filename_modifier=''):
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
        self.filename_base = filename_base
        self.filename_modifier = filename_modifier

        for df_name in ['rmse_df','hyperparam','simulated_demand','results','historical_data','mine_supply']:
            df_outer = pd.DataFrame()
            df_outer_sorted = pd.DataFrame()

        pkl_folder = self.pkl_folder
        if pkl_folder=='data' or pkl_folder.split('/')[-1]=='data':
            pkl_folder = pkl_folder.replace('data','data/Historical tuning')

        for material in self.ready_commodities:
            material = self.element_commodity_map[material].lower()
            if demand_mining_all=='demand':
                indiv = Individual(filename=f'{pkl_folder}/{material}{filename_base}{filename_modifier}_DEM.pkl',rmse_not_mae=False,dpi=50)
                rmse_or_score = 'RMSE'
            elif demand_mining_all=='mining':
                indiv = Individual(filename=f'{pkl_folder}/{material}{filename_base}{filename_modifier}_mining.pkl',rmse_not_mae=False,dpi=50)
                rmse_or_score = 'RMSE'
            elif demand_mining_all=='all':
                indiv = Individual(filename=f'{pkl_folder}/{material}{filename_base}_all{filename_modifier}.pkl',rmse_not_mae=False,dpi=50)
                rmse_or_score = 'score'
            else: raise ValueError('input for the demand_mining_all variable when calling the Many().get_variables() function must be a string of one of the following: demand, many, all')

            setattr(self,'indiv_'+material,indiv)

            for df_name in ['rmse_df','hyperparam','simulated_demand','results','historical_data','mine_supply']:
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

        types = pd.Series([type(i) for i in self.hyperparam.iloc[:,0]],self.hyperparam.index)
        self.types = types.copy()
        types = (types == float) | (types == int) | (types == np.float64)
        self.changing_hyperparam = self.hyperparam.loc[types].copy()
        self.changing_hyperparam = self.changing_hyperparam.loc[~(self.changing_hyperparam.apply(lambda x: x-x.mean(),axis=1)<1e-6).all(axis=1)]

    def get_multiple(self, demand=True, mining=True, integ=False, reinitialize=False, filename_base='_run_hist', filename_modifier='', filename_modify_non_integ=False):
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

        if demand and (not hasattr(self,'demand') or reinitialize):
            self.demand = Many()
            self.demand.get_variables('demand', filename_base=filename_base, filename_modifier=filename_modifier if filename_modify_non_integ else '')
            feature_importance(self.demand,plot=False,objective='RMSE')

        if mining and (not hasattr(self,'mining') or reinitialize):
            self.mining = Many()
            self.mining.get_variables('mining', filename_base=filename_base, filename_modifier=filename_modifier if filename_modify_non_integ else '')
            feature_importance(self.mining,plot=False,objective='score')

        if integ and (not hasattr(self,'integ') or reinitialize):
            self.integ = Many()
            self.integ.get_variables('all', filename_base=filename_base, filename_modifier=filename_modifier)
            feature_importance(self.integ,plot=False,objective='score')

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
            filename=f'{self.pkl_folder}/{material}{filename_base}{filename_modifier}_DEM.pkl'
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
            filename=f'{self.pkl_folder}/{material}{filename_base}{filename_modifier}_mining.pkl'
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
            indiv = Individual(element,3,filename=f'{self.pkl_folder}/{material}{filename_base}_all{filename_modifier}.pkl',
                   rmse_not_mae=True,weight_price=weight_price,dpi=dpi,price_rolling=5)
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

    def load_future_scenario_runs(self, pkl_folder=None, commodities=None, scenario_name_base='_run_scenario_set',verbosity=None):
        """
        loads values from the crazy number of scenarios generated by the
        run_future_scenarios function, stores them in:
        - self.multi_scenario_results
        - self.multi_scenario_hyperparam
        - self.multi_scenario_results_formatted
        - self.multi_scenario_hyperparam_formatted

        formatted versions have the columns renamed to be the scenario changes

        pkl_folder: str, folder where data is saved
        commodities: list or np.ndarray, when None uses all those from
          self.ready_commodities, but otherwise can use a list of commodities
          in the case study data.xlsx format
        scenario_name_base: str, base name of file given when running

        """
        if pkl_folder is None:
            pkl_folder = self.pkl_folder
        if commodities is None:
            commodities = self.ready_commodities
        if verbosity is None:
            verbosity=self.verbosity
        pkl_folder_ph = pkl_folder

        self.multi_scenario_results = pd.DataFrame()
        self.multi_scenario_hyperparam = pd.DataFrame()
        loaded_commodities=[]
        for element in commodities:
            if verbosity>1: print(f'starting {element}')
            if pkl_folder_ph=='data' or pkl_folder_ph.split('/')[-1]=='data':
                pkl_folder = pkl_folder_ph.replace('data','data/Simulation')
            commodity = self.element_commodity_map[element].lower()

            dir_list = os.listdir(pkl_folder)
            name = f'{commodity}{scenario_name_base}'
            filename_list = [f'{pkl_folder}/{n}' for n in dir_list if name in n and len(n.split(name)[1].split('.')[0])<=2]
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

                multi_scenario_results_ph = pd.concat([multi_scenario_results_ph],keys=[commodity])
                multi_scenario_hyperparam_ph = pd.concat([multi_scenario_hyperparam_ph],keys=[commodity])
                self.multi_scenario_results = pd.concat([self.multi_scenario_results, multi_scenario_results_ph])
                self.multi_scenario_hyperparam = pd.concat([self.multi_scenario_hyperparam, multi_scenario_hyperparam_ph])
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
        `RMSE` from rmse_df dataframe depending on whether using and Integration
        or demandModel formulation
    standard_scaler: bool, whether to use standard scaler to rescale data so it
        is N(0,1), which we should probably always do.
    """
    using_rmse_y = objective is None or hasattr(self,'rmse_df')
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
        X_df = self.multi_scenario_hyperparam_formatted.copy()
        if commodity!=None: X_df = X_df.loc[idx[commodity,:],:]
        X_df = X_df.loc[[type(i) not in [str,np.ndarray,bool] and not np.isnan(i) for i in X_df.iloc[:,0].values]]
        X_df = X_df.unstack(1)
        params_changing = X_df.loc[abs(X_df.subtract(X_df.loc[:,idx[0,1,0]],axis=0).sum(axis=1))>1e-12].index.get_level_values(1).unique()
        X_df = X_df.loc[idx[:,params_changing],:].unstack().stack(2).stack()
        X_df = X_df.stack(0).stack().unstack(2)
        X_df = X_df.loc[:,[i for i in X_df.columns if 'RMSE' not in i and 'R2' not in i and i!='score']]
        X_df = X_df.loc[idx[:,:,[j for j in X_df.index.get_level_values(2).unique() if j!=0],:],:]

    if using_rmse_y and 'score' in rmse_df.index.get_level_values(1):
        y_df = rmse_df.loc[idx[:,'score'],:].unstack().stack(level=0)
    elif using_rmse_y:
        y_df = np.log(rmse_df.loc[idx[:,'RMSE'],:].unstack().stack(level=0))
    elif objective in self.multi_scenario_results_formatted.columns:
        y_df = outer[objective]
        y_df = y_df.unstack(3).unstack().dropna(axis=1,how='all')
        y_df = y_df.subtract(y_df.loc[:,idx[0,1]],axis=0)
        y_df = y_df.groupby(level=[0,1]).mean()
        y_df = y_df.loc[:,(y_df!=0).any()]
        y_df = y_df.stack(0).stack()
        if type(y_df)==pd.core.series.Series:
            y_df = pd.DataFrame(y_df)
        y_df = y_df.rename(columns={y_df.columns[0]:objective})
    else:
        print(rmse_df.index.get_level_values(1).unique())
        print(self.multi_scenario_results_formatted.columns)
        raise ValueError('cannot load y_df objective')

    if standard_scaler:
        scaler = StandardScaler()
        x_std = scaler.fit_transform(X_df.values)
        X_df = pd.DataFrame(x_std,X_df.index,X_df.columns)
        y_std = scaler.fit_transform(y_df.values.reshape(-1,1))
        y_df = pd.DataFrame(y_std,y_df.index,y_df.columns)
    self.y_df = y_df.copy()
    self.X_df = X_df.copy()

def feature_importance(self,plot=None,recalculate=False, standard_scaler=True, plot_commodity_importances=False, commodity=None, objective=None, dpi=50):
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
    dpi: float, dots per inch, controls figure resolution.
    '''
    if hasattr(self,'objective'):
        objective=self.objective
    elif objective is None:
        objective='score'
    self.objective = objective

    split_frac = 0.5

    if plot==None or (type(plot)==bool and plot):
        plot_train_test=False
        plot_feature_importances=True
    elif type(plot)==bool and not plot:
        plot_train_test=False
        plot_feature_importances=False
    elif type(plot)==str and plot=='both':
        plot_train_test=True
        plot_feature_importances=True

    get_X_df_y_df(self, commodity=commodity, objective=objective, standard_scaler=standard_scaler)

    if not hasattr(self,'importances_df') or recalculate or plot_commodity_importances:
        importances_df = pd.DataFrame()
        for Regressor, name in zip([RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor],['RandomForest','ExtraTrees','GradientBoosting']):
            if plot_train_test: fig2,ax2 = easy_subplots(2,dpi=dpi)
            for e,dummies in enumerate([True,False]):
                X_df, y_df = self.X_df.copy().fillna(0), self.y_df.copy()

                if dummies:
                    X_df.loc[:,'commodity =']=X_df.index.get_level_values(0)
                    X_df = pd.get_dummies(X_df,columns=['commodity ='])
                X_df_test = X_df.sample(frac=split_frac,replace=False,random_state=0)
                X_test = X_df_test.reset_index(drop=True).values
                y_df_test = y_df.loc[X_df_test.index]
                X_df = X_df.loc[~X_df.index.isin(X_df_test.index)]
                y_df = y_df.loc[X_df.index]

                X = X_df.reset_index(drop=True).values
                y = y_df.values.flatten()

                regr = Regressor(random_state=0)
                regr.fit(X,y)

                test = pd.concat([X_df_test,y_df_test],axis=1)
                test.loc[:,'predicted '+objective] = regr.predict(X_test)

                if not plot_commodity_importances:
                    importances = pd.Series(regr.feature_importances_, X_df.columns).drop([i for i in X_df.columns if 'commodity' in i]).sort_values(ascending=False)
                else:
                    importances = pd.Series(regr.feature_importances_, X_df.columns).sort_values(ascending=False)
                importances.name =  name + (' w/ dummies' if dummies else ' no dummies')
                if dummies:
                    importances /= importances.sum()
                    self.X_df_dummies = X_df.copy()
                    self.y_df_dummies = y_df.copy()
                else:
                    self.X_df_no_dummies = X_df.copy()
                    self.y_df_no_dummies = y_df.copy()
                importances_df = pd.concat([importances_df, importances],axis=1)

                if plot_train_test:
                    if objective is None and 'RMSE' in test.columns:
                        target='RMSE'
                    elif objective is None:
                        target = 'score'
                    else: target = objective
                    do_a_regress(test[target], test['predicted '+objective],ax=ax2[e])
                    ax2[e].set(title=importances.name.replace(' w/','\nw/').replace(' no','\nno'))

        dummy_cols = [i for i in importances_df.columns if 'w/ dummies' in i]
        no_dummy_cols=[i for i in importances_df.columns if 'no dummies' in i]
        importances_df.loc[:,'Mean w/ dummies'] = importances_df[dummy_cols].mean(axis=1)
        importances_df.loc[:,'Mean w/ dummies'] /= importances_df['Mean w/ dummies'].sum()
        importances_df.loc[:,'Mean no dummies'] = importances_df[no_dummy_cols].mean(axis=1)
        importances_df.loc[:,'Mean no dummies'] /= importances_df['Mean no dummies'].sum()
        dummy_cols += ['Mean w/ dummies']
        no_dummy_cols += ['Mean no dummies']
        self.importances_df = importances_df.copy()

    dummy_cols = [i for i in self.importances_df.columns if 'w/ dummies' in i]
    no_dummy_cols=[i for i in self.importances_df.columns if 'no dummies' in i]

    if plot_feature_importances:
        to_plot_du = self.importances_df.loc[:,dummy_cols].sort_values(by='Mean w/ dummies',ascending=False)
        to_plot_du.rename(columns=dict(zip(dummy_cols,[i.split(' w/ dummies')[0] for i in dummy_cols])),inplace=True)
        to_plot_no = self.importances_df.loc[:,no_dummy_cols].sort_values(by='Mean no dummies',ascending=False).dropna()
        to_plot_no.rename(columns=dict(zip(no_dummy_cols,[i.split(' no dummies')[0] for i in no_dummy_cols])),inplace=True)
        to_plot_du.rename(make_parameter_names_nice(to_plot_du.index),inplace=True)
        to_plot_no.rename(make_parameter_names_nice(to_plot_no.index),inplace=True)
        height_scale = 2 if plot_commodity_importances and 'Mine cost reduction per year' in to_plot_no.index else 1.7 if plot_commodity_importances else 1.5
        fig1,ax1 = easy_subplots(2, height_scale=height_scale, width_scale=9/len(self.importances_df.columns), dpi=dpi,
                            width_ratios=(to_plot_du.shape[0],to_plot_no.shape[0]))
        to_plot_du.plot.bar(ax=ax1[0],ylabel='Feature importance').grid(axis='x')
        to_plot_no.plot.bar(ax=ax1[1],ylabel='Feature importance',legend=not plot_commodity_importances).grid(axis='x')
        ax1[0].set_title('With commodity dummies',weight='bold')
        ax1[1].set_title('No dummies',weight='bold')
        y1a, y1b = ax1[0].get_ylim()
        y2a, y2b = ax1[1].get_ylim()
        ya = min(y1a,y2a)
        yb = max(y1b,y2b)
        ax1[0].set(ylim=(ya,yb))
        ax1[1].set(ylim=(ya,yb))
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
    updated = [i.replace('_',' ').replace('sector specific ','').replace('dematerialization tech growth','Intensity decline per year').replace(
        'price response','intensity response to price').capitalize().replace('gdp','GDP').replace(
        ' cu ',' CU ').replace(' og ',' OG ').replace('Primary price resources contained elas','Incentive tonnage response to price').replace(
        'OG elas','elasticity to ore grade decline').replace('Initial','Incentive').replace('Primary oge scale','Ore grade elasticity distribution mean').replace(
        'Mine CU margin elas','Mine CU elasticity to TCM').replace('Mine cost tech improvements','Mine cost reduction per year').replace(
        'Incentive opening','Incentive pool opening').replace('Mine cost price elas','Mine cost elasticity to commodity price').replace(
        'Close years back','Prior years used for price prediction').replace('Reserves ratio price lag','Price lag used for incentive pool tonnage').replace(
        'Incentive pool ore grade decline','Incentive pool ore grade decline per year').replace('response','elasticity').replace('Tcrc','TCRC').replace('tcrc','TCRC').replace(
        'sd','SD').replace(' elas ',' elasticity to ').replace('Direct melt elas','Direct melt fraction elas').replace('CU price elas','CU elasticity to price').replace(
        'ratio TCRC elas','ratio elasticity to TCRC').replace('ratio scrap spread elas','ratio elasticity to scrap spread').replace(
        'Refinery capacity fraction increase mining','Ref. cap. growth frac. from mine prod. growth').replace('Pri ','Primary refinery ').replace(
        'Sec CU','Secondary refinery CU').replace('Sec ratio','Refinery SR').replace('primary commodity ','').replace('Primary commodity','Refined').replace('tcm','TCM')
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
    if demand_mining_integ=='mining':
        df = self.mining.rmse_df_sorted.copy()
        most_important = self.mining.importances_df['Mean no dummies'].sort_values(ascending=False).head(n_most_important).index
        df = df.loc[idx[:,most_important],:].copy()
    elif demand_mining_integ=='integ':
        df = self.integ.rmse_df_sorted.copy()
        most_important = self.integ.importances_df['Mean no dummies'].sort_values(ascending=False).head(n_most_important).index
        df = df.loc[idx[:,most_important],:].copy()
    else:
        df = self.demand.rmse_df_sorted.copy()
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

def plot_demand_parameter_correlation(self,scatter=True, percentile=25, dpi=50):
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
    df, demand_params = prep_for_snsplots(self,demand_mining_integ='demand',percentile=percentile)
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

def plot_important_parameter_scatter(self, mining_or_integ='mining', percentile=25, n=None, n_most_important=4, scale_y_for_legend=1, plot_median=True, best_or_median='mean', legend=True, scale_fig_width=1, dpi=50):
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
    dpi: int, dots per inch, figure resolution (higher = better)
    """
    if n!=None: percentile=n/600*100
    df2, demand_params, order = prep_for_snsplots(self,demand_mining_integ=mining_or_integ, percentile=percentile, n_most_important=n_most_important)
    df2 = df2.set_index(['Commodity','Scenario number']).stack().reset_index(drop=False).rename(columns={'level_2':'Parameter',0:'Value'})
    # df2a = df2.copy().loc[df2['Parameter']!='Mine cost reduction per year']
    outer = df2.loc[(df2['Value']<0)|(df2['Value']>1)]['Parameter'].unique()
    df2a = df2.copy().loc[[i not in outer for i in df2['Parameter']]]

    def replace_for_mining(string):
        return string.replace(' to T','\nto T').replace('y d','y\nd').replace('ing pr','ing\npr').replace('ge e','ge\ne')
    for i in demand_params:
        df2a.replace(i,replace_for_mining(i),inplace=True)
    df2b = df2.copy().loc[[i in outer for i in df2['Parameter']]]
    if best_or_median=='median':
        df2a_means = df2a.groupby(['Commodity','Parameter']).median().reset_index(drop=False)
        df2b_means = df2b.groupby(['Commodity','Parameter']).median().reset_index(drop=False)
    else:
        df2a_means = df2a.loc[df2a['Scenario number']==0]
        df2b_means = df2b.loc[df2b['Scenario number']==0]
    self.df2 = df2
    self.df2a = df2a
    self.df2b = df2b
    if len(outer)>0:
        fig,ax=easy_subplots(2,width_scale=scale_fig_width*1.3+0.1*n_most_important/4,width_ratios=[n_most_important-len(outer),len(outer)],dpi=dpi)
    else:
        fig,ax=easy_subplots(1,width_scale=scale_fig_width*1.5+0.1*n_most_important/4,dpi=dpi)
    a=ax[0]
    # sns.violinplot(data=df2a, x='Parameter', y='Value', hue='Commodity',ax=a, linewidth=2)
    order_rename = make_parameter_names_nice(order)
    order = [order_rename[i] for i in order]
    order = [replace_for_mining(i) for i in order if i not in outer]
    linewidth = 0.5

    sns.stripplot(data=df2a, x='Parameter', y='Value', hue='Commodity',ax=a, dodge=True, size=10,
                 order=order, edgecolor='w', linewidth=linewidth)
    if plot_median:
        marker='s'
        markersize=12
        alpha=0.3
        sns.stripplot(data=df2a_means, x='Parameter', y='Value', hue='Commodity',ax=a, dodge=True, size=markersize, palette='dark:k', alpha=alpha, marker=marker,
                 order=order, edgecolors='k')
    h,l = a.get_legend_handles_labels()
    if plot_median:
        n_commodities = len(df2a.Commodity.unique())
        square_h, square_l = Line2D([0],[0],marker=marker,color='w',alpha=alpha+0.1,markersize=10,markerfacecolor='k', markeredgecolor='k'),best_or_median.capitalize()
        h = h[:n_commodities]
        l = l[:n_commodities]
    ncol=1
    h_update = list(np.concatenate([h[i::ncol] for i in np.arange(0,ncol)]))
    l_update = list(np.concatenate([l[i::ncol] for i in np.arange(0,ncol)]))
    if plot_median:
        h_update += [square_h]
        l_update += [square_l]
    if legend:
        a.legend(ncol=ncol,handles=h_update, labels=l_update, frameon=True, columnspacing=0.2, handletextpad=0.1, borderpad=0.5, labelspacing=0.1)
    else:
        a.legend('')


    alim = a.get_ylim()
    a.set(xlabel=None, ylim=(alim[0],alim[1]*1.07*(scale_y_for_legend)))
    if mining_or_integ=='mining': title_string='Mine pre-tuning'
    elif mining_or_integ=='integ': title_string='Integration tunining'
    if len(outer)>0:
        a.set_title('                                      '+title_string+' parameter results\n',weight='bold')
    else:
        a.set_title(title_string+' parameter results',weight='bold')
    if len(outer)>0:
        b=ax[1]
        sns.stripplot(data=df2b, x='Parameter', y='Value', hue='Commodity',ax=b, dodge=True, size=10, linewidth=linewidth, edgecolor='w')
        if plot_median:
            sns.stripplot(data=df2b_means, x='Parameter', y='Value', hue='Commodity',ax=b, dodge=True, size=markersize, palette='dark:k', alpha=alpha, marker=marker)
        b.legend('')
        alim = a.get_ylim()
        scale = np.floor(df2b['Value'].min()) if df2b['Value'].min()<0 else np.ceil(df2b['Value'].max())
        if df2b['Value'].min()<0:
            b.set(ylim=[alim[0]+scale,alim[1]-1], ylabel=None, xlabel=None)
        else:
            b.set(ylim=[alim[0]*scale,alim[1]*scale], ylabel=None, xlabel=None)
    fig.tight_layout(pad=0.8)
    plt.show()
    plt.close()

    return fig,ax,df2

def commodity_level_feature_importance_heatmap(self,dpi=50,recalculate=True):
    """
    Creates a plot showing commodity level feature importances in heatmap form.

    dpi: int, dots per inch, controls figure resolution.
    recalculate: bool, whether or not feature importance gets recalculated,
      should keep as True unless you have just run this function with
      recalculate=True and just want to update dpi.
    """
    names = ['Intensity elasticity to GDP',
                 'Intensity decline per year',
                 'Intensity elasticity to price',
                 'Mine CU elasticity to TCM',
                 'Incentive pool opening probability',
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
    if not hasattr(self,'importances_df_reformed') or recalculate:
        importances_df = pd.DataFrame()
        for comm in list(self.rmse_df.index.get_level_values(0).unique())+[None]:
            feature_importance(self,commodity=comm,recalculate=True,plot=False)
            ph = self.importances_df['Mean no dummies']
            ph.name=comm if comm!=None else 'None'
            importances_df = pd.concat([importances_df,ph],axis=1)
        self.importances_df_reformed = importances_df.rename(
            make_parameter_names_nice(importances_df.index)).rename(
            columns=dict(zip(importances_df.columns,[i.capitalize().replace('None','All') for i in importances_df.columns])))
        self.importances_df_reformed = self.importances_df_reformed.loc[[i for i in names if i in self.importances_df_reformed.index]]

    fig,ax = easy_subplots(1,height_scale=1.1)
    a = ax[0]

    sns.heatmap(self.importances_df_reformed,
                xticklabels=True,yticklabels=True,ax=a,cbar_kws={'label':'Feature importance'},cmap='OrRd')
    fig.set_dpi(dpi)
    a.set_title('Integrated model\nfeature importance',weight='bold')
    return fig,a

def nice_plot_pretuning(demand_or_mining='mining',dpi=50,pkl_folder='data',filename_base='_run_hist',filename_modifier=''):
    """
    Nicely plots just the best-fitting simulated and historical mine production
    or demand based on the demand_or_mining input, for each commodity.

    demand_or_mining: str, can be either `mining` or `demand`, determines
      whether the mining or demand pre-tuning result is plotted.
    dpi: int, dots per inch, controls figure resolution
    pkl_folder: str, folder where data is saved
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
        filename=f'{pkl_folder}/{cmap_r[c]}{filename_base}{filename_modifier}_{demand_or_mining}.pkl'
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

def run_future_scenarios(data_folder='data', run_parallel=3, supply_or_demand='demand', n_best_scenarios=25, n_per_baseline=25, price_response=True, commodities=None, years_of_increase=np.arange(1,2),scenario_name_base='_run_scenario_set', simulation_time=np.arange(2019,2041), baseline_sampling='grouped', tuned_rmse_df_out_append='', notes='Scenario run!', random_state=None, verbosity=2):
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
    - data_folder: str, path to where the tuned_rmse_df_out.pkl file is stored
    - run_parallel: int, 0 to not use parallel function (op_run_future_scenarios)
      and any other number will be used as input for parallelization in
      op_run_future_scenarios_parallel
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
      data/commodity+scenario_name_base+`.pkl`
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
    - notes: str, gets saved in hyperparam of every scenario
    """
    import numpy as np
    import os
    import pandas as pd
    idx = pd.IndexSlice
    import warnings
    from Many import Many
    from integration_functions import Sensitivity
    from datetime import datetime

    if supply_or_demand is None:
        scenarios = ['']
    else:
        if supply_or_demand=='supply': s = 'ss'
        elif supply_or_demand=='demand': s = 'sd'
        elif supply_or_demand=='demand-alt': s = 'sd-alt'
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

    rmse_df = pd.read_pickle(f'{data_folder}/tuned_rmse_df_out{tuned_rmse_df_out_append}.pkl')
    if commodities is None:
        commodities = ['Steel','Al','Au','Sn','Cu','Ni','Ag','Zn','Pb']

    exponent = 10
    element_commodity_map = {'Steel':'Steel','Al':'Aluminum','Au':'Gold','Cu':'Copper','Steel':'Steel','Co':'Cobalt','REEs':'REEs','W':'Tungsten','Sn':'Tin','Ta':'Tantalum','Ni':'Nickel','Ag':'Silver','Zn':'Zinc','Pb':'Lead','Mo':'Molybdenum','Pt':'Platinum','Te':'Telllurium','Li':'Lithium'}
    col_map = dict(zip(element_commodity_map.values(),element_commodity_map.keys()))
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
            hyp_sample = pd.DataFrame(np.nan, np.arange(0,n_baselines), param_samp.columns)
            rs = 1017
            for i in hyp_sample.index:
                for j in hyp_sample.columns:
                    hyp_sample.loc[i,j] = param_samp[j].sample(random_state=rs).values[0]
                    rs += 1
            hyp_sample = hyp_sample.T
        elif baseline_sampling=='grouped':
            df = pd.DataFrame(np.nan, params.index, np.arange(0,n_per_baseline))
            stds = params[weights.sort_values(ascending=False).head(n_best_scenarios).index].std(axis=1)
            best10 = weights.sort_values(ascending=False).index
            rs = 0
            hyp_sample = pd.DataFrame()
            for n in np.arange(0,n_best_scenarios):
                for i in params.index:
                    sign = np.sign(params[best10[n]][i].mean())
                    mean = abs(params[best10[n]][i])
                    std = stds[i]
                    # df.loc[i] = stats.lognorm.rvs(loc=0,scale=mean,s=std,size=n_samp,random_state=rs)*sign
                    generated = stats.norm.rvs(loc=mean,scale=std,size=n_per_baseline*100,random_state=rs)
                    for _ in np.arange(0,3):
                        generated += mean-np.mean(generated)
                        generated = generated[generated>0]
                        if i!= 'mine_cost_tech_improvements':
                            generated = generated[generated<1]
                    df.loc[i] = generated[:n_per_baseline]*sign
                    rs+=1
                hyp_sample = pd.concat([hyp_sample,df],axis=1)
            hyp_sample = hyp_sample.T.reset_index(drop=True).T
        elif baseline_sampling=='clustered':
            hyp_sample = generate_clustered_hyperparam(rmse_df=rmse_df, commodity=commodity, n_best_scenarios=n_best_scenarios,
                                                          n_per_baseline=n_per_baseline, plot=False);
        elif baseline_sampling=='actual':
            best_n = weights.sort_values(ascending=False).head(n_best_scenarios).index
            hyp_sample = params[best_n]
            hyp_sample = hyp_sample.T.reset_index(drop=True).T

        hyp_sample = get_pretuning_params(best_hyperparameters=hyp_sample, material=col_map[commodity.capitalize()], data_folder=data_folder, verbosity=verbosity)

        # running all
        if run_parallel==0:
            run_fn = op_run_future_scenarios
        elif run_parallel<0:
            run_fn = op_run_future_scenarios_parallel
        else:
            run_fn = op_run_sensitivity_fn
        run_fn(
            commodity=commodity,
            hyperparam_df=hyp_sample,
            scenario_list=scenarios,
            scenario_name_base=scenario_name_base,
            verbosity=verbosity,
            run_parallel=run_parallel,
            simulation_time=simulation_time,
            notes=notes,
            random_state=random_state,
            )

def op_run_future_scenarios(commodity, hyperparam_df, scenario_list, scenario_name_base='_run_scenario_set', verbosity=0, run_parallel=None, simulation_time=np.arange(2019,2041), notes='', random_state=None):
    """
    Can be run by run_future_scenarios if run_parallel is set to zero; this is
    currently the most deprecated version of this process, see
    op_run_sensitivity_fn for the current version.
    """
    from integration_functions import Sensitivity
    from datetime import datetime

    if type(scenario_list[0])==str:
        scenario_list = [scenario_list]
    hyp_sample = hyperparam_df.copy()

    element_commodity_map = {'Steel':'Steel','Al':'Aluminum','Au':'Gold','Cu':'Copper','Steel':'Steel','Co':'Cobalt','REEs':'REEs','W':'Tungsten','Sn':'Tin','Ta':'Tantalum','Ni':'Nickel','Ag':'Silver','Zn':'Zinc','Pb':'Lead','Mo':'Molybdenum','Pt':'Platinum','Te':'Telllurium','Li':'Lithium'}
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
                filename=f'data/Simulation/'+material+scenario_name_base+str(m)+'.pkl'
                if verbosity>-2: print('--'*15+filename+'-'*15)
                best_params.loc['refinery_capacity_growth_lag']=1
                if random_state is not None:
                    best_params.loc['random_state'] = random_state[rs]
                s = Sensitivity(filename,changing_base_parameters_series=col_map[material.capitalize()],notes=notes,
                                additional_base_parameters=best_params, historical_price_rolling_window=5,
                                simulation_time=simulation_time,
                                scenarios=scenarios,
                                OVERWRITE=rs==0,verbosity=verbosity)
                s.run_monte_carlo(n_scenarios=2,bayesian_tune=False, sensitivity_parameters=['Nothing, giving a string incompatible with any of the variable names'])
                if verbosity>-1: print(f'time for batch: {str(datetime.now()-t1)}')
                t_per_batch.loc[m*len(scenario_list)+n] = datetime.now()-t1
                filename_list += [filename]
    if verbosity>-1: print(f'total time elapsed: {str(datetime.now()-t0)}')

def op_run_sensitivity_fn(commodity, hyperparam_df, scenario_list, scenario_name_base='_run_scenario_baseline', verbosity=0, run_parallel=None, simulation_time=np.arange(2019,2041), notes='', random_state=None):
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
    element_commodity_map = {'Steel':'Steel','Al':'Aluminum','Au':'Gold','Cu':'Copper','Steel':'Steel','Co':'Cobalt','REEs':'REEs','W':'Tungsten','Sn':'Tin','Ta':'Tantalum','Ni':'Nickel','Ag':'Silver','Zn':'Zinc','Pb':'Lead','Mo':'Molybdenum','Pt':'Platinum','Te':'Telllurium','Li':'Lithium'}
    col_map = dict(zip(element_commodity_map.values(),element_commodity_map.keys()))
    element = col_map[commodity.capitalize()]

    filename=f'data/Simulation/'+commodity+scenario_name_base+'.pkl'
    if verbosity>-2: print('--'*15+filename+'-'*15)
    s = Sensitivity(filename,changing_base_parameters_series=col_map[commodity.capitalize()],notes=notes,
                    additional_base_parameters=0, historical_price_rolling_window=5,
                    simulation_time=simulation_time,
                    scenarios=scenario_list,
                    OVERWRITE=True,verbosity=verbosity)
    s.run_monte_carlo(n_scenarios=2,bayesian_tune=False, sensitivity_parameters=hyperparam_df,n_jobs=abs(run_parallel))

def op_run_future_scenarios_parallel(commodity, hyperparam_df, scenario_list, scenario_name_base='_run_scenario_set', verbosity=0, run_parallel=3, simulation_time=np.arange(2019,2041), notes='', random_state=None):
    """
    Called by run_future_scenarios if its run_parallel input is below zero,
    since in my opinion this function is mostly deprecated
    """
    from integration_functions import Sensitivity
    from datetime import datetime
    from joblib import Parallel, delayed
    from IterTimer import IterTimer

    if type(scenario_list[0])==str:
        scenario_list = [scenario_list]
    hyp_sample = hyperparam_df.copy()

    timer = IterTimer(n_iters=len(scenario_list)*hyp_sample.shape[1], log_times=False)

    element_commodity_map = {'Steel':'Steel','Al':'Aluminum','Au':'Gold','Cu':'Copper','Steel':'Steel','Co':'Cobalt','REEs':'REEs','W':'Tungsten','Sn':'Tin','Ta':'Tantalum','Ni':'Nickel','Ag':'Silver','Zn':'Zinc','Pb':'Lead','Mo':'Molybdenum','Pt':'Platinum','Te':'Telllurium','Li':'Lithium'}
    col_map = dict(zip(element_commodity_map.values(),element_commodity_map.keys()))

    material=commodity
    t0 = datetime.now()
    t_per_batch = pd.Series(np.nan,np.arange(0,len(hyp_sample.columns)))
    filename_list = []
    n_scen = len(hyp_sample.columns)*len(scenario_list)


    Parallel(n_jobs=abs(run_parallel))(delayed(run_scenario_set)(m, best_ind, hyp_sample, scenario_list, material, scenario_name_base, col_map, verbosity, simulation_time, notes, timer, random_state) for m,best_ind in enumerate(hyp_sample.columns))

    if verbosity>-1: print(f'total time elapsed: {str(datetime.now()-t0)}')

def run_scenario_set(m,best_ind,hyp_sample,scenario_list,material,scenario_name_base,col_map,verbosity, simulation_time=np.arange(2019,2041), notes='', timer=None, random_state=None):
    """
    Called by op_run_future_scenarios_parallel to run each set of scenarios
    """
    element_commodity_map = {'Steel':'Steel','Al':'Aluminum','Au':'Gold','Cu':'Copper','Steel':'Steel','Co':'Cobalt','REEs':'REEs','W':'Tungsten','Sn':'Tin','Ta':'Tantalum','Ni':'Nickel','Ag':'Silver','Zn':'Zinc','Pb':'Lead','Mo':'Molybdenum','Pt':'Platinum','Te':'Telllurium','Li':'Lithium'}
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
                filename=f'data/Simulation/'+material+scenario_name_base+'.pkl'
            else:
                filename=f'data/Simulation/'+material+scenario_name_base+str(m)+'.pkl'
            if verbosity>-2: print('--'*15+filename+'-'*15)
            best_params.loc['refinery_capacity_growth_lag']=1
            OVERWRITE = m==0 if len(scenarios)<3 else rs_e==0
            if random_state is not None:
                best_params.loc['random_state'] = random_state[rs]
            s = Sensitivity(filename,changing_base_parameters_series=col_map[material.capitalize()],notes=notes,
                            additional_base_parameters=best_params, historical_price_rolling_window=5,
                            simulation_time=simulation_time,
                            scenarios=scenarios,
                            OVERWRITE=OVERWRITE,verbosity=verbosity)
            s.run_monte_carlo(n_scenarios=2,bayesian_tune=False, sensitivity_parameters=['Nothing, giving a string incompatible with any of the variable names'])
            if verbosity>-1: print(f'time for batch: {str(datetime.now()-t1)}')
        # timer.end_iter()

def get_pretuning_params(best_hyperparameters, material, data_folder='data', verbosity=0):
    """
    Used to lead updated_commodity_inputs.pkl so it gets used for future scenario runs as well.
    """
    if os.path.exists('data/updated_commodity_inputs.pkl'):
        updated_commodity_inputs = pd.read_pickle('data/updated_commodity_inputs.pkl')
        if verbosity>-1: print('updated_commodity_inputs source: data/updated_commodity_inputs.pkl')
    elif os.path.exists('updated_commodity_inputs.pkl'):
        updated_commodity_inputs = pd.read_pickle('updated_commodity_inputs.pkl')
        if verbosity>-1: print('updated_commodity_inputs source: updated_commodity_inputs.pkl')
    elif os.path.exists(f'{data_folder}/updated_commodity_inputs.pkl'):
        updated_commodity_inputs = pd.read_pickle(f'{data_folder}/updated_commodity_inputs.pkl')
        if verbosity>-1: print(f'updated_commodity_inputs source: {data_folder}/updated_commodity_inputs.pkl')
    else:
        raise ValueError('updated_commodity_inputs.pkl does not exist in the expected locations (in this directory, in data relative path, in data_folder input given). Need to run the historical_sim_check_demand() function to create an initialization of updated_commodity_inputs.pkl')

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
            test_rmses.loc[idx[:,[i for i in test_rmses.index.get_level_values(1).unique()
                                  if 'price' in i]],:] = \
                test_rmses.loc[idx[:,[i for i in test_rmses.index.get_level_values(1).unique()
                                  if 'price' in i]],:]**weight_price
            test_score = np.log(test_rmses.groupby(level=0).sum().div(3))
            test_score = pd.concat([test_score.unstack()],keys=[f'{i} score'],axis=1).stack().unstack(0)
            rmse_copy = pd.concat([rmse_copy,test_score])

        # now doing overall score (score saved from original rmse_df is for train set)
        test_rmses = rmse_copy.loc[idx[:,[i for i in rmse_copy.index.get_level_values(1).unique()
                                      if 'RMSE' in i and 'test' not in i and 'train' not in i]],:].sort_index()
        test_score = np.log(test_rmses.groupby(level=0).sum().div(3))

        test_score = pd.concat([test_score.unstack()],keys=['total score'],axis=1).stack().unstack(0)
        rmse_copy = pd.concat([rmse_copy,test_score])

        # sum of scores to avoid weighting
        test_score = rmse_copy.loc[idx[:,'train score'],:].droplevel(1)+\
                rmse_copy.loc[idx[:,'test score'],:].droplevel(1)
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
                                          ['train score','test score','total score','sum score']],:].droplevel(0).T
    else:
        scores = rmse_train_test.loc[['train score','test score','total score','sum score']].T
    scores['sum'] = scores.sum(axis=1)
    scores['pareto'] = is_pareto_efficient_simple(scores[['train score','test score']].astype(float).values)
    pareto = scores.loc[scores['pareto']]
    selection = scores.loc[scores['train score'].sort_values().head(n_best_scenarios*4).index]
    selection = selection.loc[selection['test score'].sort_values().head(n_best_scenarios).index]
    scores['selection1'] = [i in selection.index for i in scores.index]
    selection2 = scores.loc[scores['total score'].sort_values().head(n_best_scenarios).index]
    scores['selection2'] = [i in selection2.index for i in scores.index]
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
    scores = get_commodity_scores(rmse_train_test,commodity,n_best_scenarios=n_best_scenarios)

    if rmse_df.index.nlevels>1:
        rmse_df = rmse_df.copy().loc[commodity]
    rmse_df.loc['region_specific_price_response'] = rmse_df.loc['sector_specific_price_response']
    droppers = [i for i in rmse_df.index.unique() if np.any([j in i for j in ['score','R2','RMSE']])]
    rmse_df.drop(droppers,inplace=True)

    rmse_ph = rmse_df[scores.loc[scores.selection1].index].T
    hyperparam_ph = pd.DataFrame(np.nan,np.arange(0,rmse_ph.shape[0]*n_per_baseline),rmse_ph.columns)
    for ei,i in enumerate(rmse_ph.index):
        for ef,f in enumerate(hyperparam_ph.columns):
            sign = np.sign(rmse_ph[f][i])
            if n_per_baseline>1:
                data = abs(rmse_ph[f][i]+stats.norm.rvs(loc=0, scale=rmse_ph[f].std(),
                                                    size=n_per_baseline, random_state=ei+ef))
            else:
                data = np.array([abs(rmse_ph[f][i])])
            if f!='mine_cost_tech_improvements':
                data[data>1] = 2-data[data>1]
            data = sign*data
            hyperparam_ph.iloc[ei*n_per_baseline:(ei+1)*n_per_baseline,ef]=data

    if plot:
        fig,ax=easy_subplots(hyperparam_ph.columns)
        for i,a in zip(hyperparam_ph.columns,ax):
            hyperparam_ph[i].iloc[:n_per_baseline].plot.hist(ax=a,title=i)
        return hyperparam_ph.T, fig,ax
    return hyperparam_ph.T
