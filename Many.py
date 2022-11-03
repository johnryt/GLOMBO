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
    Not all of the above take Many as an input; some are standalone or are
    called by other functions in this file.
    '''
    def __init__(self, data_folder=None, pkl_folder=None):
        '''
        data_folder: str, the folder where historical data needing loading lives
        pkl_folder: str, folder where pkl files of results will be saved
        '''
        self.ready_commodities = ['Steel','Al','Au','Sn','Cu','Ni','Ag','Zn','Pb']
        self.element_commodity_map = {'Steel':'Steel','Al':'Aluminum','Au':'Gold','Cu':'Copper','Steel':'Steel','Co':'Cobalt','REEs':'REEs','W':'Tungsten','Sn':'Tin','Ta':'Tantalum','Ni':'Nickel','Ag':'Silver','Zn':'Zinc','Pb':'Lead','Mo':'Molybdenum','Pt':'Platinum','Te':'Telllurium','Li':'Lithium'}
        self.commodity_element_map = dict(zip(self.element_commodity_map.values(),self.element_commodity_map.keys()))
        self.data_folder = 'generalization/data' if data_folder==None else data_folder
        self.pkl_folder = 'data' if pkl_folder==None else pkl_folder

    def run_all_demand(self, n_runs=50, commodities=None, save_mining_info=False, trim_result_df=True, filename_base='_run_hist', filename_modifier=''):
        t1 = datetime.now()
        commodities = self.ready_commodities if commodities==None else commodities
        for material in commodities:
            print('-'*40)
            print(material)
            mat = self.element_commodity_map[material].lower()
            filename=f'{self.pkl_folder}/{mat}{filename_base}.pkl'
            self.shist1 = Sensitivity(pkl_filename=filename, data_folder=self.data_folder,changing_base_parameters_series=material,notes='Monte Carlo aluminum run',
                            simulation_time=np.arange(2001,2020),OVERWRITE=True,use_alternative_gold_volumes=True,
                                historical_price_rolling_window=5,verbosity=0, trim_result_df=trim_result_df)
            self.shist1.historical_sim_check_demand(n_runs,demand_or_mining='demand')
            print(f'time elapsed: {str(datetime.now()-t1)}')

    def run_all_mining(self, n_runs=50, commodities=None, save_mining_info=False, trim_result_df=True, filename_base='_run_hist', filename_modifier=''):
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
                               incentive_opening_probability_fraction_zero=0, save_mining_info=save_mining_info,
                               trim_result_df=trim_result_df)
            self.shist.historical_sim_check_demand(n_runs,demand_or_mining='mining')
            print(f'time elapsed: {str(datetime.now()-t1)}')

    def run_all_integration(self, n_runs=200, commodities=None, train_time=np.arange(2001,2020), normalize_objectives=False,constrain_previously_tuned=True, verbosity=0, save_mining_info=False, trim_result_df=True, filename_base='_run_hist', filename_modifier=''):
        '''

        '''
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
                            simulation_time=np.arange(2001,2020), include_sd_objectives=False, train_time=train_time,
                            OVERWRITE=True,verbosity=verbosity,historical_price_rolling_window=5,
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
        '''
        for df_name in ['rmse_df','hyperparam','simulated_demand','results','historical_data','mine_supply']:
            df_outer = pd.DataFrame()
            df_outer_sorted = pd.DataFrame()


        for material in self.ready_commodities:
            material = self.element_commodity_map[material].lower()
            if demand_mining_all=='demand':
                indiv = Individual(filename=f'{self.pkl_folder}/{material}{filename_base}{filename_modifier}_DEM.pkl',rmse_not_mae=False,dpi=50)
                rmse_or_score = 'RMSE'
            elif demand_mining_all=='mining':
                indiv = Individual(filename=f'{self.pkl_folder}/{material}{filename_base}{filename_modifier}_mining.pkl',rmse_not_mae=False,dpi=50)
                rmse_or_score = 'RMSE'
            elif demand_mining_all=='all':
                indiv = Individual(filename=f'{self.pkl_folder}/{material}{filename_base}_all{filename_modifier}.pkl',rmse_not_mae=False,dpi=50)
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
        '''
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
        '''
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
        '''
        for material in self.ready_commodities:
            material = self.element_commodity_map[material].lower()
            filename=f'{self.pkl_folder}/{material}{filename_base}{filename_modifier}_mining.pkl'
            indiv = Individual(filename=filename,rmse_not_mae=False,dpi=dpi)
            indiv.plot_demand_results()

    def plot_all_integration(self,dpi=50,
                    plot_over_time=True,
                    nth_best=1,
                    weight_price=1,
                    include_sd=False,
                    plot_sd_over_time=False,
                    plot_best_indiv_over_time=False,
                    plot_hyperparam_heatmap=False,
                    n_best=20,
                    plot_hyperparam_distributions=False,
                    n_per_plot=4,
                    plot_hyperparam_vs_error=False,
                    flip_yx=False,
                    plot_best_params=False,
                    plot_supply_demand_stack=False,
                    filename_base='_run_hist',
                    filename_modifier=''):
        '''
        Produces many different plots you can use to try and understand the
        model outputs. Loads each commodity in the Individual class, and runs
        its plot_results method on it.

        More info is given with each model input bool description below.

        ----------------------
        Inputs:
        plot_over_time: bool, True plots the best overall scenario over time (using NORM SUM or NORM SUM OBJ ONLY) for each objective to allow comparison
        n_best: int, the number of scenarios to include in plot_over_time or in plot_hyperparam_distributions
        include_sd: bool, True means we use the NORM SUM row to evaluate the best scenario, while False means we use NORM SUM OBJ ONLY
        plot_hyperparam_heatmap: bool, True plots a heatmap of the hyperparameter values for the best n scenarios
        plot_hyperparam_distributions: bool, True plots the hyperparameter distributions
        n_per_plot: int, for use with plot_hyperparam_distributions. Determines how many hyperparameter values are put in each plot, since it can be hard to tell what is going on when there are too many lines in a figure
        plot_hyperparam_vs_error: bool, plots the hyperparameter values vs the error value, separate plot for each hyperparameter. Use this to try and see if there are ranges for the best hyperparameter values.
        flip_yx: bool, False means plot hyperparam value vs error, while True means plot error vs hyperparam value
        '''

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

    def load_future_scenario_runs(self, pkl_folder=None, commodities=None, scenario_name_base='_run_scenario_set',n_scenarios=100,verbosity=0):
        """
        loads values from the crazy number of scenarios generated by the
        run_future_scenarios function, stores them in:
        - self.multi_scenario_results
        - self.multi_scenario_hyperparam
        - self.multi_scenario_results_formatted
        - self.multi_scenario_hyperparam_formatted

        formatted versions have the columns renamed to be the scenario changes
        """
        if pkl_folder is None:
            pkl_folder = self.pkl_folder

        if commodities is None:
            commodities = self.ready_commodities
        self.multi_scenario_results = pd.DataFrame()
        self.multi_scenario_hyperparam = pd.DataFrame()
        loaded_commodities=[]
        for element in commodities:
            if verbosity>1: print(f'starting {element}')
            # filename_list = [f'data/aluminum_run_scenario_set0{chr(i)}.pkl' for i in np.arange(97,97+10)]
            commodity = self.element_commodity_map[element].lower()
            filename_list = [f'{pkl_folder}/{commodity}{scenario_name_base}{i}.pkl' for i in np.arange(0,n_scenarios)]

            dir_list = os.listdir(pkl_folder)
            indiv_list = [Individual(element,3,name) if name.split('/')[-1] in dir_list else name for name in filename_list]
            print_list = [j for j in indiv_list if type(j)!=Individual]
            if len(print_list)>0:
                print(f'{element} is missing {len(print_list)}/{n_scenarios} files, set verbosity>0 to view')
                if verbosity>0:
                    print('Missing files:')
                    missing_ints = [int(j.split(scenario_name_base)[1].split('.pkl')[0]) for j in print_list]
                    if len(missing_ints)==missing_ints[-1]-missing_ints[0]+1:
                        print(f'\t{print_list[0]} to {print_list[-1]}')
                    else:
                        for j in print_list:
                            print(f'\t{j}')
            indiv_list = [j for j in indiv_list if type(j)==Individual]

            if len(indiv_list)==n_scenarios:
                multi_scenario_results_ph = pd.concat([indiv.results for indiv in indiv_list],keys=np.arange(0,n_scenarios))
                multi_scenario_hyperparam_ph = pd.concat([indiv.hyperparam for indiv in indiv_list],keys=np.arange(0,n_scenarios))

                multi_scenario_results_ph = pd.concat([multi_scenario_results_ph],keys=[commodity])
                multi_scenario_hyperparam_ph = pd.concat([multi_scenario_hyperparam_ph],keys=[commodity])
                self.multi_scenario_results = pd.concat([self.multi_scenario_results, multi_scenario_results_ph])
                self.multi_scenario_hyperparam = pd.concat([self.multi_scenario_hyperparam, multi_scenario_hyperparam_ph])
                if verbosity>0: print(f'{element} completed')
                loaded_commodities += [element]
            elif verbosity>0: print(f'{element} skipped due to missing scenarios')
        if verbosity>-1: print(f'loaded commodities: {loaded_commodities},\nskipped commodities: {[i for i in commodities if i not in loaded_commodities]}')

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

def get_X_df_y_df(self, commodity=None, objective=None, standard_scaler=True):
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

def feature_importance(self,plot=None, dpi=50,recalculate=False, standard_scaler=True, plot_commodity_importances=False, commodity=None, objective=None):
    '''
    objective: str, which variable to try and calculate feature importance for.
      If left as None, will go ahead with either `score` or `RMSE` depending on
      whether using an Integration or demandModel formulation.
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
                    do_a_regress(test.RMSE, test['predicted '+objective],ax=ax2[e])
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
    plot_feature_importance_no_show_dummies: bool,
    plot_feature_importance_show_dummies: bool,
    plot_mining_and_demand_importance: bool,
    plot_commodity_level_importances: bool,
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

def plot_important_parameter_scatter(self, percentile=25, n=None, dpi=50, n_most_important=4, scale_y_for_legend=1, plot_median=True, legend=True, best_or_median='mean', mining_or_integ='mining'):
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
        fig,ax=easy_subplots(2,width_scale=1.3+0.1*n_most_important/4,width_ratios=[n_most_important-len(outer),len(outer)],dpi=dpi)
    else:
        fig,ax=easy_subplots(1,width_scale=1.3+0.1*n_most_important/4,dpi=dpi)
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
    a.set_title('                                      Mine pre-tuning parameter results\n',weight='bold')

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

def nice_plot_pretuning(demand_or_mining='mining',dpi=50,filename_base='_run_hist',filename_modifier=''):
    if demand_or_mining=='demand': demand_or_mining='DEM'
    ready_commodities = ['Steel','Al','Au','Sn','Cu','Ni','Ag','Zn','Pb']
    fig,ax = easy_subplots(ready_commodities,dpi=dpi)
    cmap = {'nickel':'Ni','gold':'Au','aluminum':'Al','tin':'Sn','zinc':'Zn','lead':'Pb','steel':'Steel','copper':'Cu','silver':'Ag'}
    cmap_r=dict(zip(cmap.values(),cmap.keys()))
    for c,a in zip(ready_commodities,ax):
        filename=f'{self.pkl_folder}/{cmap_r[c]}{filename_base}{filename_modifier}_{demand_or_mining}.pkl'
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

def run_future_scenarios(data_folder='data', run_parallel=3, supply_or_demand='demand', n_baselines=100, price_response=True, commodities=None, years_of_increase=np.arange(1,2),scenario_name_base='_run_scenario_set', simulation_time=np.arange(2019,2041), baseline_sampling='grouped', notes='Scenario run!', random_state=None, verbosity=2):
    """
    Runs scrap demand scenarios, for 0.01 to 20% of the market switching from
    refined consumption to scrap consumption, for years given (default is just
    one year).

    Runs 100 different sets of hyperparameters. These are randomly sampled from
    a weighted sampling-with-replacement of the Bayesian optimization results
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
    - n_baselines: int, number of different baseline scenarios to run
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
    - baseline_sampling: str, can be either `grouped` or `random`
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

    rmse_df = pd.read_pickle(f'{data_folder}/tuned_rmse_df_out.pkl')
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
            n_samp = 10
            df = pd.DataFrame(np.nan, params.index, np.arange(0,n_samp))
            stds = params[weights.sort_values(ascending=False).head(25).index].std(axis=1)
            best10 = weights.sort_values(ascending=False).index
            rs = 0
            n_best=int(n_baselines/10)
            hyp_sample = pd.DataFrame()
            for n in np.arange(0,n_best):
                for i in params.index:
                    sign = np.sign(params[best10[n]][i].mean())
                    mean = abs(params[best10[n]][i])
                    std = stds[i]
                    # df.loc[i] = stats.lognorm.rvs(loc=0,scale=mean,s=std,size=n_samp,random_state=rs)*sign
                    generated = stats.norm.rvs(loc=mean,scale=std,size=n_samp*100,random_state=rs)
                    for _ in np.arange(0,3):
                        generated += mean-np.mean(generated)
                        generated = generated[generated>0]
                        if i!= 'mine_cost_tech_improvements':
                            generated = generated[generated<1]
                    df.loc[i] = generated[:n_samp]*sign
                    rs+=1
                hyp_sample = pd.concat([hyp_sample,df],axis=1)
            hyp_sample = hyp_sample.T.reset_index(drop=True).T
        elif baseline_sampling=='actual':
            best_n = weights.sort_values(ascending=False).head(10).index
            hyp_sample = params[best_n]
            hyp_sample = hyp_sample.T.reset_index(drop=True).T

        hyp_sample = get_pretuning_params(best_hyperparameters=hyp_sample, material=col_map[commodity.capitalize()], data_folder=data_folder, verbosity=verbosity)

        # running all
        if run_parallel==0:
            run_fn = op_run_future_scenarios
        else:
            run_fn = op_run_future_scenarios_parallel
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
                filename='data/'+material+scenario_name_base+str(m)+'.pkl'
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

def op_run_future_scenarios_parallel(commodity, hyperparam_df, scenario_list, scenario_name_base='_run_scenario_set', verbosity=0, run_parallel=3, simulation_time=np.arange(2019,2041), notes='', random_state=None):
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


    Parallel(n_jobs=run_parallel)(delayed(run_scenario_set)(m, best_ind, hyp_sample, scenario_list, material, scenario_name_base, col_map, verbosity, simulation_time, notes, timer, random_state) for m,best_ind in enumerate(hyp_sample.columns))

    if verbosity>-1: print(f'total time elapsed: {str(datetime.now()-t0)}')

def run_scenario_set(m,best_ind,hyp_sample,scenario_list,material,scenario_name_base,col_map,verbosity, simulation_time=np.arange(2019,2041), notes='', timer=None, random_state=None):
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
        for rs in random_states:
            # timer.start_iter()
            t1 = datetime.now()
            filename='data/'+material+scenario_name_base+str(m)+'.pkl'
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
        # timer.end_iter()

def get_pretuning_params(best_hyperparameters, material, data_folder='data', verbosity=0):
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
