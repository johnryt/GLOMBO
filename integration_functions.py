import numpy as np
import pandas as pd
idx = pd.IndexSlice
from matplotlib import pyplot as plt
from scipy import stats
from integration import Integration
from random import seed, sample, shuffle
from demand_class import demandModel
import os
from useful_functions import easy_subplots
import statsmodels.api as sm

from copy import deepcopy

from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties

from ax.metrics.noisy_function import NoisyFunctionMetric
from ax import RangeParameter, ParameterType, SearchSpace, MultiObjective, Objective, ObjectiveThreshold, MultiObjectiveOptimizationConfig, Models, Experiment, Data
from ax.runners.synthetic import SyntheticRunner
from ax.modelbridge.factory import get_MOO_EHVI
from ax.modelbridge.modelbridge_utils import observed_hypervolume
from ax.service.utils.report_utils import exp_to_df

import torch
from botorch.test_functions.multi_objective import BraninCurrin
branin_currin = BraninCurrin(negate=True).to(
    dtype=torch.double,
    device= torch.device("cuda" if torch.cuda.is_available() else "cpu"),
)

def create_result_df(integ):
    '''
    takes Integration object, returns regional results. Used within the senstivity
    function to convert the individual model run to results we can interpret later.
    '''
    reg = 'Global'
    reg_results = pd.Series(np.nan,['Global','China','RoW'],dtype=object)
    
    new = integ.mining.ml.loc[integ.mining.ml['Opening']>integ.simulation_time[0]]
    old = integ.mining.ml.loc[integ.mining.ml['Opening']<=integ.simulation_time[0]]
    old_new_mines = pd.concat([
        old.loc[:,'Production (kt)'].groupby(level=0).sum(),
        new.loc[:,'Production (kt)'].groupby(level=0).sum(),
        old.loc[:,['Production (kt)','Head grade (%)']].product(axis=1).groupby(level=0).sum()/old.loc[:,'Production (kt)'].groupby(level=0).sum(),
        new.loc[:,['Production (kt)','Head grade (%)']].product(axis=1).groupby(level=0).sum()/new.loc[:,'Production (kt)'].groupby(level=0).sum(),
        old.loc[:,['Production (kt)','Minesite cost (USD/t)']].product(axis=1).groupby(level=0).sum()/old.loc[:,'Production (kt)'].groupby(level=0).sum(),
        new.loc[:,['Production (kt)','Minesite cost (USD/t)']].product(axis=1).groupby(level=0).sum()/new.loc[:,'Production (kt)'].groupby(level=0).sum(),
        old.loc[:,['Production (kt)','Total cash margin (USD/t)']].product(axis=1).groupby(level=0).sum()/old.loc[:,'Production (kt)'].groupby(level=0).sum(),
        new.loc[:,['Production (kt)','Total cash margin (USD/t)']].product(axis=1).groupby(level=0).sum()/new.loc[:,'Production (kt)'].groupby(level=0).sum(),
        integ.mining.resources_contained_series, integ.mining.reserves_ratio_with_demand_series
        ],
        keys=['Old mine prod.','New mine prod.',
              'Old mine grade','New mine grade',
              'Old mine cost','New mine cost',
              'Old mine margin','New mine margin',
              'Reserves','Reserves ratio with production'],axis=1).fillna(0)

    addl_scrap = integ.additional_scrap.sum(axis=1).unstack()
    addl_scrap.loc[:,'Global'] = addl_scrap.sum(axis=1)
    for reg in reg_results.index:
        results = pd.concat([integ.total_demand.loc[:,reg],integ.scrap_demand.loc[:,reg],integ.scrap_supply[reg],
               integ.concentrate_demand[reg],integ.concentrate_supply,
               integ.mining.ml.loc[:,['Production (kt)','Head grade (%)']].product(axis=1).groupby(level=0).sum()/integ.mining.ml.loc[:,'Production (kt)'].groupby(level=0).sum(),
               integ.mining.ml.loc[:,['Production (kt)','Minesite cost (USD/t)']].product(axis=1).groupby(level=0).sum()/integ.mining.ml.loc[:,'Production (kt)'].groupby(level=0).sum(),
               integ.mining.ml.loc[:,['Production (kt)','Total cash margin (USD/t)']].product(axis=1).groupby(level=0).sum()/integ.mining.ml.loc[:,'Production (kt)'].groupby(level=0).sum(),
               old_new_mines['Old mine prod.'],old_new_mines['New mine prod.'],
               old_new_mines['Old mine grade'],old_new_mines['New mine grade'],
               old_new_mines['Old mine cost'],old_new_mines['New mine cost'],
               old_new_mines['Old mine margin'],old_new_mines['New mine margin'],
               integ.refined_demand.loc[:,reg],integ.refined_supply[reg],
               integ.secondary_refined_demand.loc[:,reg],integ.direct_melt_demand.loc[:,reg],
               integ.scrap_spread[reg],integ.tcrc,integ.primary_commodity_price,
               integ.refine.ref_stats[reg]['Primary CU'], integ.refine.ref_stats[reg]['Secondary CU'],
               integ.refine.ref_stats[reg]['Secondary ratio'],
               integ.refine.ref_stats[reg]['Primary capacity'], integ.refine.ref_stats[reg]['Secondary capacity'],
               integ.refine.ref_stats[reg]['Primary production'], integ.refine.ref_stats[reg]['Secondary production'],
               integ.additional_direct_melt[reg],integ.additional_secondary_refined[reg],addl_scrap[reg],
               integ.sxew_supply,integ.primary_supply[reg],integ.primary_demand[reg]
              ],axis=1,
              keys=['Total demand','Scrap demand','Scrap supply',
                    'Conc. demand','Conc. supply',
                    'Mean mine grade','Mean total minesite cost','Mean total cash margin',
                    'Old mine prod.','New mine prod.',
                    'Old mine grade','New mine grade',
                    'Old mine cost','New mine cost',
                    'Old mine margin','New mine margin',
                    'Ref. demand','Ref. supply',
                    'Sec. ref. cons.','Direct melt',
                    'Spread','TCRC','Refined price',
                    'Refinery pri. CU','Refinery sec. CU','Refinery SR',
                    'Pri. ref. capacity','Sec. ref. capacity',
                    'Pri. ref. prod.','Sec. ref. prod.',
                    'Additional direct melt','Additional secondary refined','Additional scrap',
                    'SX-EW supply','Primary supply','Primary demand'])
        if reg=='Global':
            collection = integ.demand.old_scrap_collected.groupby(level=0).sum()/integ.demand.eol.groupby(level=0).sum()
            old_scrap = integ.demand.old_scrap_collected.groupby(level=0).sum().sum(axis=1)
            new_scrap = integ.demand.new_scrap_collected.groupby(level=0).sum().sum(axis=1)
        else:
            collection = integ.collection_rate.loc[idx[:,reg],:].droplevel(1)
            old_scrap = integ.demand.old_scrap_collected.loc[idx[:,reg],:].droplevel(1).sum(axis=1)
            new_scrap = integ.demand.new_scrap_collected.loc[idx[:,reg],:].droplevel(1).sum(axis=1)
        collection = collection.rename(columns=dict(zip(collection.columns,['Collection rate '+j.lower() for j in collection.columns])))
        scraps = pd.concat([old_scrap,new_scrap],axis=1,keys=['Old scrap collection','New scrap collection'])
        results = pd.concat([results,collection,scraps],axis=1)
        reg_results.loc[reg] = [results]
    return reg_results

def check_equivalence(big_df, potential_append):
    '''
    Returns True if equivalent, False if not equivalent and we should add to our df
    True means equivalent; Used to check whether each model run in the sensitivity
    has already been run, so we can skip those that have already been run
    '''
    v = big_df.copy()
    bools = []
    for i in big_df.index:
        c = potential_append.loc[i].iloc[0]
        if i in ['hyperparam']:
            col = 'Global' if i=='refine.hyperparam' else 'Value'
            j = v.loc[i].iloc[0]
            c_ind = [i for i in c.index if i!='simulation_time']
            if not np.any([len(c.dropna(how='all').index)==len(j.dropna(how='all').index) and
             len(np.intersect1d(c.dropna(how='all').index,j.dropna(how='all').index))==len(c.dropna(how='all').index) and
             len(np.intersect1d(c.dropna(how='all').index,j.dropna(how='all').index))==len(j.dropna(how='all').index) and
             np.all([q in c.dropna(how='all').index for q in j.dropna(how='all').index]) and
             np.all([q in j.dropna(how='all').index for q in c.dropna(how='all').index]) and
             (c[col][c_ind].dropna(how='all')==j[col][c_ind].dropna(how='all')).all().all() for j in v.loc[i]]):
#                 print(i,'not already a thing, adding to our big dataframe')
                bools += [False]
            else:
#                 print(i,'already a thing')
                bools += [True]
        elif i in ['version']:
            if c in v.loc[i]:
#                 print(i,'not already a thing, adding to our big dataframe')
                bools += [False]
            else:
#                 print(i,'already a thing')
                bools += [True]
    return not np.all(bools),bools

class Sensitivity():
    '''
    Look at the initialization (__init__) docstring for input
    information, below is more general information like the
    available most-used methods/functions and what they do.
    
    changing_base_parameters_series: can be a string referring to a
      commodity in case_study_data, or a series with hyperparameter
      inputs in the same format

    Methods/functions (those not listed here are utility functions):
    - run: runs a sensitivity changing only one variable at a time,
        using the variables in the params_to_change variable from
        Sensitivity initialization. Also from initialization: n_per_param
        gives the number of scenarios to run per parameter, and param_scale
        (from 1-param_scale to 1+param_scale) gives the multiplier for
        the parameters included
    - run_monte_carlo: takes in a list of sensitivity_parameters, where
        if any of the strings in that list are in any of the hyperparameters,
        that hyperparameter will be subject to random generation, with values
        between 0 and 1. The demand parameters sector_specific_dematerialization_tech_growth,
        sector_specific_price_response, region_specific_price_response, and
        incentive_opening_probability are multiplied by smaller factors to
        be between 0 and 0.15.
    - historical_sim_check_demand: uses a Bayesian optimization algorithm
        to find the best values for the demand parameters
        sector_specific_dematerialization_tech_growth,
        sector_specific_price_response, region_specific_price_response, and
        intensity_response_to_gdp, simulating only demand using the demand
        module. **Requires** that case_study_data.xlsx have a sheet named after
        the commodity being input via changing_base_parameters_series (should
        use changing_base_parameters_series=commodity string here too). This
        sheet must also have columns [Total demand] and [Primary commodity price].
    - check_hist_demand_convergence: plots the total demand results from
        historical_sim_check_demand to ensure our algorithm converged correctly,
        plotting the simulated demands, the best simulated demand, and historical
        demand. Also plots the RMSE vs the demand parameters.
    - run_historical_monte_carlo: similar to the run_monte_carlo function, but also
        requires the presence of a 

    The update_changing_base_parameters_series function could be a source of error,
        as it updates the model start values with historical values within
        the Sensitivity class

    If you add more than the columns Total demand, Primary production, Primary
        commodity price, Primary demand, and Scrap demand for the material-
        specific sheets in case study data.xlsx, you will need to:

        1. update the param_variable_map in the complete_bayesian_trial function
        within the Sensitivity class. You put the columns name, then the
        string of the variable name you want to tune it to. All the variable
        names you could want would be in the Integration class, and you can
        see them in the functions __init__, initialize_integration, run_mining,
        or update_integration_variables. Looking in __init__ is the closest
        to a direct list, but it's more clear their meaning looking in the
        other functions.
        2. update the update_changing_base_parameters_series function within
        the Sensitivity class with any necessary additions such that there
        is agreement across variables and time.

    In the case of MemoryError, try running the following code (can go up
    module levels):
        var = vars(s.mod.mining.inc) # or s.mod, s.mod.mining, etc.
        for v in var.keys():
            print(v, getsizeof(var[v]))
     Finding the largest variables can help identify the location of the problem
    '''
    def __init__(self,
                 pkl_filename='integration_big_df.pkl',
                 case_study_data_file_path='generalization/data/case study data.xlsx',
                 changing_base_parameters_series=0,
                 additional_base_parameters=0,
                 params_to_change=0,
                 n_per_param=5,
                 notes='Initial run',
                 simulation_time = np.arange(2019,2041),
                 byproduct=False,
                 verbosity=0,
                 param_scale=0.5,
                 scenarios=[''],
                 OVERWRITE=False,
                 random_state=220620,
                 incentive_opening_probability_fraction_zero=0.1,
                 include_sd_objectives=False,
                 use_rmse_not_r2=True,
                 dpi=50,
                 using_thresholds=True,
                 N_INIT=3,
                 normalize_objectives=False,
                 use_alternative_gold_volumes=True):
        '''
        Initializing Sensitivity class.

        ---------------
        pkl_filename: str, path/filename ending in pkl for where you want to save the results of the
            sensitivity you are about to run
        case_study_data_file_path: str, path/filename to the \'case study data.xlsx\' file being used
            for the input hyperparameters
        changing_base_parameters_series: pd.series.Series | str. If string, must correspond with a column and
            sheet name in the case study data excel file, which then causes the base parameters series
            to be loaded using parameters for that commodity. Alternatively, a pandas series can be input that
            contains the hyperparameter names you wish to change
        additional_base_parameters: pd.series.Series. pandas series of model hyperparameters that is merged onto
            the eventual changing_base_parameters_series dataframe, such that even when changing_base_parameters_series
            is a string, we can add additional model hyperparameters to change its functionality.
        params_to_change: list of hyperparameters to be changed when running the most
            simple sensitivity analysis, where we only change one parameter at a time using
            the run() function.
        n_per_param: int, number of trials to run per parameter when using the run() function to run a
            a very simple one-parameter sensitivity. Used in conjunction with param_scale.
        notes: str, anything you want to be included in the resulting pickle file under the notes index
        simulation_time: np.ndarray, list of years for running the simulation. Typically either 2001 through
            2019 or 2019 through 2040.
        byproduct: bool, whether we are simulating a mono-product mined commodity or a multi-product mined
            commodity. Have not yet had the chance to test out the True setting
        verbosity: int, tells how much we print as the model runs. Can be anything including -1 which suppresses
            nearly all output, and I think the highest value (above which nothing changes) is 4.
        param_scale: float, used in conjunction with params_to_change and n_per_param in the run() simple
            sensitivity function, where whichever parameter is selected will be scaled by +/- param_scale
        scenarios: list or array, including only strings. The contents of each string determine the scrap
            supply or demand changes that will be implemented. See the decode_scrap_scenario_name() function
            in integration.py (Integration class) for more detail, but some are given here:
                takes the form 00_11_22_33_44
                where:
                00: ss, sd, bo (scrap supply, scrap demand, both)
                11: pr or no (price response included or no)
                22: Xyr, where X is any integer and represents
                 the number of years the increase occurs
                33: X%tot, where X is any float/int and is the
                 increase/decrease in scrap supply or demand
                 relative to the initial year total demand
                44: X%inc, where X is any float/int and is the
                 increase/decrease in the %tot value per year

                for 22-44, an additional X should be placed at
                 the end when 00==both, describing the sd values
        OVERWRITE: bool, whether or not the pkl_filename given should overwrite the existing file or not.
            If doing a Bayesian tuning sensitivity, needs to be set True
        random_state: int, can be anything, but have been using the default for all scenarios for
            reproducibility
        incentive_opening_probability_fraction_zero: used in the run_monte_carlo() method to determine the
            fraction of incentive_opening_probability values generated that are then set to zero, since setting
            to zero allows the model to determine the incentive_opening_probability value endogenously, picking
            the mean value from the most recent n simulations where incentive tuning used incentive_opening_probability
            to set the number of mines opening such that concentrate supply=demand
        dpi: int, stands for dots per inch, and is used to change the resolution of the plots created using
            the check_hist_demand_convergence() method on a historical_sim_check_demand() method run
        using_thresholds: bool, determines whether to use the old AxClient approach instead of 
            updated Bayesian optimization approach which uses the Service API, which allows us to set thresholds
            for our objectives. Hoping that the new one works better than the old one due to presence of thresholds.
            Both are visible at: https://ax.dev/tutorials/multiobjective_optimization, where thresholds are set
            using the algorithms farther down the page, under the heading Set Objective Thresholds. I cannot tell
            whether the AxClient approach uses the same algorithm for multi-objective optimization. In the 
            Service API approach, we specify we are using the qNEHVI option, Noisy Expected Hypervolume Improvement.
            It is the one they recommend so that is my assumption/hope that the AxClient approach uses the same
            algorithm. From the two Pareto Frontiers they show, it appears this is the case.
        N_INIT: int, number of randomly generated (SOBOL) initialization runs for the updated Bayesian optimization
            approach before it switches to the qNEHVI algorithm.
        normalize_objectives: bool, determines whether the objectives should be normalized by their first-year value
        use_alternative_gold_volumes: bool, if True uses the alternative gold volume drivers since industrial
            represents bar and coin, and transport represents jewelry. False uses the default, which fails to
            capture the plateau seen in gold demand 2010-2019.
        '''
        self.simulation_time = simulation_time
        self.changing_base_parameters_series = changing_base_parameters_series
        self.additional_base_parameters = additional_base_parameters
        self.case_study_data_file_path = case_study_data_file_path
        self.use_alternative_gold_volumes = use_alternative_gold_volumes
        self.update_changing_base_parameters_series()

        self.byproduct = byproduct
        self.verbosity = verbosity
        self.param_scale = param_scale
        self.pkl_filename = pkl_filename
        self.params_to_change = params_to_change
        self.n_per_param = n_per_param
        self.notes = notes
        self.scenarios = scenarios
        self.overwrite = OVERWRITE
        self.random_state = random_state
        self.include_sd_objectives = include_sd_objectives
        self.incentive_opening_probability_fraction_zero = incentive_opening_probability_fraction_zero
        self.use_rmse_not_r2 = use_rmse_not_r2
        self.dpi = dpi
        
        self.using_thresholds = using_thresholds
        self.N_INIT = N_INIT
        self.normalize_objectives = normalize_objectives

        if self.overwrite: print('WARNING, YOU ARE OVERWRITING AN EXISTING FILE')

    def initialize_big_df(self):
        '''
        Initializes the big dataframe used to save all the results
        '''
        if os.path.exists(self.pkl_filename) and not self.overwrite:
            big_df = pd.read_pickle(self.pkl_filename)
        else:
            self.mod = Integration(simulation_time=self.simulation_time,verbosity=self.verbosity,byproduct=self.byproduct,scenario_name='')
            for base in self.changing_base_parameters_series.index:
                self.mod.hyperparam.loc[base,'Value'] = self.changing_base_parameters_series[base]
            self.mod.run()
            big_df = pd.DataFrame(np.nan,index=[
                'version','notes','hyperparam','mining.hyperparam','refine.hyperparam','demand.hyperparam','results','mine_data'
            ],columns=[])
            reg_results = create_result_df(self.mod)
            big_df.loc[:,0] = np.array([self.mod.version, self.notes, self.mod.hyperparam, self.mod.mining.hyperparam,
                                        self.mod.refine.hyperparam, self.mod.demand.hyperparam, reg_results, self.mod.mining.ml],dtype=object)
            big_df.to_pickle(self.pkl_filename)
        self.big_df = big_df.copy()

    def update_changing_base_parameters_series(self):
        '''
        This function is called within the __init__() method and updates the hyperparameters
        using default changes (setting incentive_require_tune_years, presimulate_n_years,
        and end_calibrate_years hyperparameter values to 10, and start_calibrate_years to 5.
        This means that for all scenarios we run, we have it run on an additional 10 years of
        data before the simulation_time variable start, where it is required to tune the
        incentive pool such that concentrate supply=demand in each of those years. The
        start and end calibrate years give the time after the additional early simulation
        start that we take the mean opening probability and use it for the case where
        incentive_opening_probability==0.

        Otherwise, this function uses the value of changing_base_parameters_series from
        initialization to update the self.changing_base_parameters_series variable. The input
        to initialization can be a string or pandas series of hyperparameters and their values.
        If it is a string, the function gets the values from the case study data excel file that
        match the string.
        '''
        changing_base_parameters_series = self.changing_base_parameters_series
        simulation_time = self.simulation_time
        if type(changing_base_parameters_series)==str:
            self.material = changing_base_parameters_series
            input_file = pd.read_excel(self.case_study_data_file_path,index_col=0)
            commodity_inputs = input_file[changing_base_parameters_series].dropna()
            commodity_inputs.loc['incentive_require_tune_years'] = 10
            commodity_inputs.loc['presimulate_n_years'] = 10
            commodity_inputs.loc['end_calibrate_years'] = 10
            commodity_inputs.loc['start_calibrate_years'] = 5
            commodity_inputs.loc['close_price_method'] = 'probabilistic'
            commodity_inputs = commodity_inputs.dropna()

            history_file = pd.read_excel(self.case_study_data_file_path,index_col=0,sheet_name=changing_base_parameters_series)
            historical_data = history_file.loc[simulation_time[0]:].dropna(axis=1)

            original_demand = commodity_inputs['initial_demand']
            original_primary_production = commodity_inputs['primary_production']
            if 'Total demand' in historical_data.columns:
                commodity_inputs.loc['initial_demand'] = historical_data['Total demand'][simulation_time[0]]
            elif 'Primary production' in historical_data.columns:
                commodity_inputs.loc['initial_demand'] = historical_data['Primary production'][simulation_time[0]]*original_demand/original_primary_production
                historical_data.loc[:,'Total demand'] = historical_data['Primary production']*original_demand/historical_data['Primary production'][simulation_time[-1]]
            elif 'Primary supply' in historical_data.columns:
                commodity_inputs.loc['initial_demand'] = historical_data['Primary supply'][simulation_time[0]]*original_demand/original_primary_production
                historical_data.loc[:,'Total demand'] = historical_data['Primary supply']*original_demand/historical_data['Primary supply'][simulation_time[-1]]
            else:
                raise ValueError('Need either [Total demand] or [Primary production] in historical data columns (ignore the brackets, but case sensitive)')

            if 'Scrap demand' in historical_data.columns:
                commodity_inputs.loc['Recycling input rate, Global'] = historical_data['Scrap demand'][simulation_time[0]]/historical_data['Total demand'][simulation_time[0]]
                commodity_inputs.loc['Recycling input rate, China'] = historical_data['Scrap demand'][simulation_time[0]]/historical_data['Total demand'][simulation_time[0]]

            if 'Primary production' in historical_data.columns:
                commodity_inputs.loc['primary_production'] = historical_data['Primary production'][simulation_time[0]]
            elif 'Primary supply' in historical_data.columns:
                commodity_inputs.loc['primary_production'] = historical_data['Primary supply'][simulation_time[0]]
            else:
                commodity_inputs.loc['primary_production'] *= commodity_inputs['initial_demand']/original_demand

            if 'Primary commodity price' in historical_data.columns:
                commodity_inputs.loc['primary_commodity_price'] = historical_data['Primary commodity price'][simulation_time[0]]
            if 'Scrap demand' in historical_data.columns and 'Primary production' in historical_data.columns:
                commodity_inputs.loc['Total production, Global'] = historical_data['Primary production'][simulation_time[0]]+historical_data['Scrap demand'][simulation_time[0]]
            elif 'Primary production' in historical_data.columns:
                commodity_inputs.loc['Total production, Global'] = historical_data['Primary production'][simulation_time[0]]*original_demand/original_primary_production
            elif 'Primary supply' in historical_data.columns:
                commodity_inputs.loc['Total production, Global'] = historical_data['Primary supply'][simulation_time[0]]*original_demand/original_primary_production
            else:
                commodity_inputs.loc['Total production, Global'] = commodity_inputs['initial_demand']*original_demand/commodity_inputs['Total production, Global']
            self.historical_data = historical_data.copy()
            self.changing_base_parameters_series = commodity_inputs.copy()
        elif not hasattr(self,'material'):
            self.material = ''
        cbps = self.changing_base_parameters_series
        if type(cbps)==pd.core.frame.DataFrame:
            cbps = self.changing_base_parameters_series.copy()
            cbps = cbps.iloc[:,0]
            self.changing_base_parameters_series = cbps.copy()
        elif type(cbps)==pd.core.series.Series:
            pass
        elif type(cbps)==int:
            self.changing_base_parameters_series = pd.Series(dtype=object)
        else:
            raise ValueError('changing_base_parameters_series input is incorrectly formatted (should be dataframe with parameter values in first column, or a series)')

        if type(self.additional_base_parameters)==pd.core.series.Series:
            for q in self.additional_base_parameters.index:
                self.changing_base_parameters_series.loc[q] = self.additional_base_parameters[q]
        elif type(self.additional_base_parameters)==pd.core.frame.DataFrame:
            for q in self.additional_base_parameters.index:
                self.changing_base_parameters_series.loc[q] = self.additional_base_parameters.iloc[:,0][q]
        if 'commodity' not in self.changing_base_parameters_series.index and self.use_alternative_gold_volumes:
            self.changing_base_parameters_series.loc['commodity'] = self.material
            
    def get_params_to_change(self):
        '''
        gets a list of parameters to change that are already within the
        Integration class hyperparam dataframe
        '''
        if type(self.params_to_change)==int:
            self.mod = Integration(simulation_time=self.simulation_time,verbosity=self.verbosity,byproduct=self.byproduct)
            for base in self.changing_base_parameters_series.index:
                self.mod.hyperparam.loc[base,'Value'] = self.changing_base_parameters_series[base]
            self.params_to_change = pd.concat([
                self.mod.hyperparam.loc['price elasticities':'determining model structure'].dropna(),
                self.mod.hyperparam.loc['mining only':].dropna(how='all')])

    def run(self):
        '''
        Runs a simplistic sensitivity, where only one parameter is
        changed each time. Currently uses get_params_to_change() method
        to select a preset bunch of hyperparameters to go through. Have
        stopped using this one as much, but can still be useful.
        '''
        self.update_changing_base_parameters_series()
        self.initialize_big_df()
        self.get_params_to_change()
        changing_base_parameters_series = self.changing_base_parameters_series.copy()
        params_to_change = self.params_to_change.copy()
        params_to_change_ind = [i for i in params_to_change.index if type(params_to_change['Value'][i]) not in [bool,str,np.ndarray] and 'years' not in i]
        big_df = self.big_df.copy()

        n = self.n_per_param + 1
        n_sig_dig = 3

        total_num_scenarios = len(params_to_change_ind)*(n-1)
        count = 0

        for i in params_to_change_ind:
            init_val = params_to_change['Value'][i]
            vals = np.append(np.linspace(init_val*(1-self.param_scale),init_val,int(n/2)-1,endpoint=False), np.linspace(init_val,init_val*(1+self.param_scale),int(n/2)))
            for val in vals:
                if self.verbosity>0:
                    print(i,val)
                if val!=0:
                    val = round(val, n_sig_dig-1 - int(np.floor(np.log10(abs(val)))))
                self.val = val
                self.mod = Integration(simulation_time=self.simulation_time,verbosity=self.verbosity,byproduct=self.byproduct)
                self.hyperparam_copy = self.mod.hyperparam.copy()

                ###### CHANGING BASE PARAMETERS ######
                for base in changing_base_parameters_series.index:
                    self.mod.hyperparam.loc[base,'Value'] = changing_base_parameters_series[base]
                    if count==0:
                        print(base,changing_base_parameters_series[base])

                ###### UPDATING FROM params_to_change_ind ######
                self.mod.hyperparam.loc[i,'Value'] = val
                print(f'Scenario {count}/{total_num_scenarios}: {i} = {val}')
                count += 1

                self.check_run_append()

    def run_monte_carlo(self, n_scenarios, random_state=220530,
                        sensitivity_parameters=['elas','response','growth','improvements','refinery_capacity_fraction_increase_mining','incentive_opening_probability'],
                        bayesian_tune=False,n_params=1):
        '''
        Runs a Monte Carlo based approach to the sensitivity, where all the sensitivity_parameters
        given are then given values randomly selected from between zero and one.

        Always runs an initial scenario with default parameters, so remember to skip that one
        when looking at the resulting pickle file
        ----------------------
        n_scenarios: int, number of scenarios to run in total
        random_state: int, hold constant to keep same set of generated values
          each time this function is run
        senstivity_parameters: list, includes ['elas','response','growth','improvements']
          by default to capture the different elasticity values, mine improvements, and
          demand responses to price. The parameters you are targeting must already be in
          the main mod.hyperparam dataframe, which must be updated within that function;
          cannot be accessed here. collection_rate_price_response, direct_melt_price_response,
          and secondary_refined_price_response are excluded from the parameters to change
          regardless of the inputs here, since they are True/False values and are determined
          by the scenario input given when the Senstivity class is initialized.
        bayesian_tune: bool, whether to use the Bayesian optimization over historical data
          to try to select the next sentivitity parameter values
        n_params: int, number of columns from self.historical_data to use in the objective
          function for tuning. >1 means multi-objective optimization, and depends on the
          order of the columns, since it grabs the leftmost n_params columns.
        '''
        self.random_state = random_state
        self.update_changing_base_parameters_series()
        self.initialize_big_df()
        self.mod = Integration(simulation_time=self.simulation_time,verbosity=self.verbosity,byproduct=self.byproduct)
        scenario_params_dont_change = ['collection_rate_price_response','direct_melt_price_response','secondary_refined_price_response','refinery_capacity_growth_lag']
#         params_to_change = [i for i in self.mod.hyperparam.dropna(how='all').index if ('elas' in i or 'response' in i or 'growth' in i or 'improvements' in i) and i not in scenario_params_dont_change]
        params_to_change = [i for i in self.mod.hyperparam.dropna(how='all').index if np.any([j in i for j in sensitivity_parameters]) and i not in scenario_params_dont_change]
        self.sensitivity_param = params_to_change
        if self.verbosity>-1:
            print(params_to_change)
        # do something with incentive_opening_probability?

        if bayesian_tune:
            self.setup_bayesian_tune(n_params=n_params)

        for n in np.arange(1,n_scenarios):
            self.scenario_number = n-1
            if self.verbosity>-1:
                print(f'Scenario {n+1}/{n_scenarios}')
            for enum,scenario_name in enumerate(self.scenarios):
                if self.verbosity>-1:
                    print(f'\tSub-scenario {enum+1}/{len(self.scenarios)}: {scenario_name} checking if exists...')
                self.mod = Integration(simulation_time=self.simulation_time,verbosity=self.verbosity,byproduct=self.byproduct,scenario_name=scenario_name)
                self.notes = scenario_name

                ###### CHANGING BASE PARAMETERS ######
                changing_base_parameters_series = self.changing_base_parameters_series.copy()
                if self.verbosity>0: print('parameters getting updated from outside using changing_base_parameters_series input:\n',changing_base_parameters_series.index)
                for base in changing_base_parameters_series.index:
                    self.mod.hyperparam.loc[base,'Value'] = changing_base_parameters_series[base]
                    if n==0 and self.verbosity>0:
                        print(base,changing_base_parameters_series[base])
                self.hyperparam_copy = self.mod.hyperparam.copy()

                ###### UPDATING MONTE CARLO PARAMETERS ######
                if len(params_to_change)>0:
                    rs = random_state+n
                    values = stats.uniform.rvs(loc=0,scale=1,size=len(params_to_change),random_state=rs)
                    new_param_series = pd.Series(values, params_to_change)
                    if bayesian_tune:
                        new_param_series = self.get_bayesian_trial()
                    if self.verbosity>0:
                        print('Parameters getting changed via Monte Carlo random selection:\n',params_to_change)
                    if 'sector_specific_dematerialization_tech_growth' in params_to_change:
                        new_param_series.loc['sector_specific_dematerialization_tech_growth'] *= 0.15
                    if 'sector_specific_price_response' in params_to_change:
                        new_param_series.loc['sector_specific_price_response'] *= 0.15
                    if 'region_specific_price_response' in params_to_change:
                        new_param_series.loc['region_specific_price_response'] *= 0.15
                    if 'incentive_opening_probability' in params_to_change:
                        new_param_series.loc['incentive_opening_probability']*=0.1/(1-self.incentive_opening_probability_fraction_zero)
                        if new_param_series['incentive_opening_probability']>0.1:
                            new_param_series.loc['incentive_opening_probability'] = 0
                    # ^ these values should be small, since small changes make big changes
                    all_three_here = np.all([q in params_to_change for q in ['close_probability_split_max','close_probability_split_mean','close_probability_split_min']])
                    if 'close_probability_split_max' in params_to_change and not all_three_here:
                        new_param_series.loc['close_probability_split_max'] *= 0.8
                    if 'close_probability_split_mean' in params_to_change and not all_three_here:
                        new_param_series.loc['close_probability_split_mean'] *= 0.8
                    if 'close_probability_split_max' in params_to_change and 'close_probability_split_mean' in params_to_change and not all_three_here:
                        sum_mean_max = new_param_series.loc[['close_probability_split_max','close_probability_split_mean']].sum()
                        if sum_mean_max>0.95:
                            new_param_series.loc[['close_probability_split_max','close_probability_split_mean']] *= 0.95/sum_mean_max
                        new_param_series.loc['close_probability_split_mean'] = 1-new_param_series.loc[['close_probability_split_max','close_probability_split_mean']].sum()
                    if 'close_years_back' in params_to_change:
                        new_param_series.loc['close_years_back'] = int(10*new_param_series.loc['close_years_back']+3)
                    if all_three_here:
                        new_param_series.loc[['close_probability_split_mean','close_probability_split_min','close_probability_split_max']] /=  new_param_series.loc[['close_probability_split_mean','close_probability_split_min','close_probability_split_max']].sum()

                    for param in params_to_change:
                        if type(self.mod.hyperparam['Value'][param])!=bool:
                            self.mod.hyperparam.loc[param,'Value'] = new_param_series[param]*np.sign(self.mod.hyperparam.loc[param,'Value'])
                        else:
                            self.mod.hyperparam.loc[param,'Value'] = new_param_series[param]
                self.check_run_append()
                if bayesian_tune:
                    self.complete_bayesian_trial(n_params=n_params,
                                                new_param_series=new_param_series, scenario_number=n+enum)

        if bayesian_tune:
            self.save_bayesian_results(n_params=n_params)

    def get_model_results(self,x1):
        '''
        Runs only in the updated Bayesian optimization setup, and is used to save
        and then read model run results. 
        '''
        x = pd.Series(x1, self.sensitivity_param)
        check = np.any([(x==self.model_data.loc[x.index,m]).all() for m in self.model_data.columns])
        if check:
            check_idx = self.model_data.apply(lambda b: (b.loc[x.index]==x).all()).idxmax()
            result = self.model_data.loc[self.objective_parameters,check_idx].values
            if self.verbosity>5:
                print('598  read result',check_idx,result)
        else:
            rmse_list, r2_list = self.get_rmse_r2()
            rmse_list = rmse_list if self.use_rmse_not_r2 else r2_list
            result = [q[0] for q in rmse_list]
            if self.verbosity>5:
                print('603 write result',self.model_data.shape[1],'\n\t',x.values,result)
            for q,r in zip(self.objective_parameters,result):
                x.loc[q] = r
            self.model_data.loc[:,self.model_data.shape[1]] = x
            self.model_data = self.model_data.copy()
        self.x = x.copy()

        return result
    
    def setup_bayesian_tune(self, n_params=1):
        '''
        Initializes the things needed to run the Bayesian optimization
        '''
        if n_params==1:
            self.ax_client = AxClient(random_seed=self.random_state,verbose_logging=self.verbosity>-1)
            experiment_param=[{'name':i,'type':'range','bounds':[0.001,1],'value_type':'float'} for i in self.sensitivity_param]
            self.ax_client.create_experiment(
                name="minimize_RMSE_with_real",
                parameters=experiment_param,
                objective_name="RMSE",
                minimize=self.use_rmse_not_r2, # default False, true if using RMSE, false if using R2
            )
        elif n_params>1:
            n_params = min([self.historical_data.shape[1],n_params])
            objective_parameters = self.historical_data.columns[:n_params]
            hist_ph = self.historical_data.copy()
            h = self.mod.hyperparam['Value']
            if self.include_sd_objectives:
                objective_parameters = np.append(objective_parameters,['Scrap SD','Conc. SD','Ref. SD'])
                hist_ph.loc[:,'Scrap SD'] = h['initial_demand']*h['Recycling input rate, Global']*0.1
                hist_ph.loc[:,'Conc. SD'] = h['initial_demand']*(1-h['Recycling input rate, Global'])*0.1
                hist_ph.loc[:,'Ref. SD'] = h['initial_demand']*(1-h['Recycling input rate, Global'])*0.1
            self.objective_parameters = objective_parameters
            self.model_data = pd.DataFrame(np.nan, np.append(self.sensitivity_param,self.objective_parameters), [])
            
            if not self.using_thresholds:
                self.ax_client = AxClient(random_seed=self.random_state,verbose_logging=self.verbosity>-1)
                experiment_param=[{'name':i,'type':'range','bounds':[0.001,1],'value_type':'float'} for i in self.sensitivity_param]
#                 objective_param = dict([[c, ObjectiveProperties(minimize=True, threshold=branin_currin.ref_point[i])] for i,c in enumerate(objective_parameters)])
                objective_param = dict([[c, ObjectiveProperties(minimize=self.use_rmse_not_r2)] for i,c in enumerate(objective_parameters)])
                all_three_here = np.all([q in self.sensitivity_param for q in ['close_probability_split_max','close_probability_split_mean','close_probability_split_min']])
                if not all_three_here:
                    self.ax_client.create_experiment(
                        name="minimize_RMSE_with_real",
                        parameters=experiment_param,
                        objectives=objective_param
                    )
                else:
                    self.ax_client.create_experiment(
                        name="minimize_RMSE_with_real",
                        parameters=experiment_param,
                        objectives=objective_param,
                        parameter_constraints=['close_probability_split_max + close_probability_split_mean + close_probability_split_min <= 1']
                    )
            else:
                parameters = [RangeParameter(name=x, lower=0, upper=1, parameter_type=ParameterType.FLOAT) for x in self.sensitivity_param]
                self.search_space = SearchSpace(
                    parameters=parameters,
                )
                
                class MetricA(NoisyFunctionMetric):
                    def f(self2, x:np.ndarray) -> float:
                        return float(self.get_model_results(x)[0])
                class MetricB(NoisyFunctionMetric):
                    def f(self2, x:np.ndarray) -> float:
                        return float(self.get_model_results(x)[1])
                class MetricC(NoisyFunctionMetric):
                    def f(self2, x:np.ndarray) -> float:
                        return float(self.get_model_results(x)[2])
                
                metric_a = MetricA(objective_parameters[0], self.sensitivity_param, noise_sd=0.0, lower_is_better=self.use_rmse_not_r2)
                metric_b = MetricB(objective_parameters[1], self.sensitivity_param, noise_sd=0.0, lower_is_better=self.use_rmse_not_r2)
                if n_params==2:
                    self.mo = MultiObjective(objectives=[Objective(metric=metric_a),Objective(metric=metric_b)])
                elif n_params==3:
                    metric_c = MetricC(objective_parameters[2], self.sensitivity_param, noise_sd=0.0, lower_is_better=self.use_rmse_not_r2)
                    self.mo = MultiObjective(objectives=[Objective(metric=metric_a),Objective(metric=metric_b),Objective(metric=metric_c)])
                else:
                    raise ValueError('number of parameters is incorrect for current Bayesian optimization setup')
                if self.normalize_objectives:
                    thresholds = [0.1 for i in n_params]
                else:
                    thresholds = [hist_ph[i].iloc[0] for i in objective_parameters[:n_params]]
                
                self.objective_thresholds = [ObjectiveThreshold(metric=metric, bound=val, relative=False) for metric,val in zip(self.mo.metrics,thresholds)]
                
                self.optimization_config = MultiObjectiveOptimizationConfig(objective=self.mo, objective_thresholds=self.objective_thresholds)
        
        self.rmse_df = pd.DataFrame()
        
    def build_experiment(self):
        experiment = Experiment(
            name="pareto_experiment",
            search_space=self.search_space,
            optimization_config=self.optimization_config,
            runner=SyntheticRunner(),
        )
        return experiment

    def get_bayesian_trial(self):
        if not self.using_thresholds:
            parameters, self.trial_index = self.ax_client.get_next_trial()
#                         trial_index = n+enum
            new_param_series = pd.Series(parameters,self.sensitivity_param)
        else:
            if self.scenario_number==0:
                self.experiment = self.build_experiment()
                self.sobol = Models.SOBOL(search_space=self.experiment.search_space, seed=self.random_state)
                self.ehvi_hv_list = []
            if self.scenario_number<self.N_INIT:
                self.trial = self.experiment.new_trial(self.sobol.gen(1))
            else:
                self.ehvi_data = self.experiment.fetch_data()
                self.ehvi_model = get_MOO_EHVI(experiment=self.experiment, data=self.ehvi_data)
                self.trial = self.experiment.new_trial(generator_run=self.ehvi_model.gen(1))
            new_param_series = self.experiment.arms_by_name[str(self.scenario_number)+'_0'].parameters
            new_param_series = pd.Series(new_param_series)
        return new_param_series
        
    def complete_bayesian_trial(self, n_params=1, new_param_series=pd.Series(dtype=float), scenario_number=0):
        '''
        calculates root mean squared errors (RMSE) from current variables and historical
        values to give the error the Bayesian optimization is trying to minimize.
        '''
        if not self.using_thresholds: trial_index = self.trial_index
        if n_params==1:
            rmse = ((self.mod.total_demand['Global']-self.historical_data['Total demand'])**2).loc[self.simulation_time].astype(float).sum()**0.5
            new_param_series.loc['RMSE'] = rmse
            rmse_dict=(rmse,0)
        elif n_params>1:
            rmse_list, r2_list = self.get_rmse_r2()
            for param, rmse, r2 in zip(self.objective_parameters, rmse_list, r2_list):
                new_param_series.loc[param+' RMSE'] = rmse[0]
                new_param_series.loc[param+' R2'] = r2[0]
            rmse_list = rmse_list if self.use_rmse_not_r2 else r2_list
            rmse_dict = dict(zip(self.objective_parameters,rmse_list))
            
        new_param_series = pd.concat([new_param_series],keys=[scenario_number])
        self.rmse_df = pd.concat([self.rmse_df,new_param_series])
        
        if not self.using_thresholds:
            self.ax_client.complete_trial(trial_index=trial_index, raw_data=rmse_dict)
        else:
            self.rmse_list = [q[0] for q in rmse_list]
            if self.verbosity>5:
                display('764 this is the one',self.rmse_list)
            self.trial.run()
            self.trial_data = self.trial.fetch_data()
            # self.ehvi_data = self.experiment.fetch_data()
            if self.scenario_number>self.N_INIT:
                self.ehvi_data = Data.from_multiple_data([self.ehvi_data, self.trial.fetch_data()])
                self.exp_df = exp_to_df(self.experiment)
                self.outcomes = np.array(self.exp_df[self.objective_parameters], dtype=np.double)
                try:
                    hv = observed_hypervolume(modelbridge=self.ehvi_model)
                except:
                    hv = 0
                    print("\tFailed to compute hv")
                self.ehvi_hv_list.append(hv)
                if self.verbosity>0:
                    print(f"\tIteration: {self.scenario_number}, HV: {hv}")
            

    def calculate_rmse_r2(self, sim, hist, use_rmse):
        n = len(self.simulation_time)
        x, y = sim.loc[self.simulation_time].astype(float), hist.loc[self.simulation_time].astype(float)
        if hasattr(x,'columns') and 'Global' in x.columns: x=x['Global']
        if hasattr(y,'columns') and 'Global' in y.columns: y=y['Global']
        m = sm.GLS(x,sm.add_constant(y)).fit(cov_type='HC3')
        if use_rmse:
            result = m.mse_resid**0.5
        else:
            result = m.rsquared
        return result
    
    def save_bayesian_results(self,n_params=1):
        '''
        saves the results of the Bayesian optimization in the
        updated_commodity_inputs.pkl file
        '''
        rmse_df = self.rmse_df.copy()
        rmse_df.index = pd.MultiIndex.from_tuples(rmse_df.index)
        rmse_df = rmse_df[0]
        rmse_df = rmse_df.unstack()
        self.rmse_df = rmse_df.copy()

        if n_params==1:
            best_params = pd.DataFrame(rmse_df.loc[rmse_df['RMSE'].idxmin()].drop('RMSE'))
            best_params = best_params.rename(columns={best_params.columns[0]:self.material})
#             best_parameters, values = self.ax_client.get_best_parameters()
        elif n_params>1:
            best_params = self.ax_client.get_pareto_optimal_parameters()
            best_params = pd.DataFrame(best_params)

        path=''
        if os.path.exists('data/updated_commodity_inputs.pkl'):
            path = 'data/updated_commodity_inputs.pkl'
        elif os.path.exists('updated_commodity_inputs.pkl'):
            path = 'updated_commodity_inputs.pkl'

        if path!='':
            self.updated_commodity_inputs = pd.read_pickle(path)
            if n_params==1:
                for i in best_params.index:
                    self.updated_commodity_inputs.loc[i,self.material] = best_params[self.material][i]
            elif n_params>1:
                self.updated_commodity_inputs.loc['pareto_'+str(n_params)+'p',self.material] = [best_params]
            self.updated_commodity_inputs.to_pickle(path)

        else:
            self.updated_commodity_inputs = best_params.copy()
            if os.path.exists('data'):
                self.updated_commodity_inputs.to_pickle('data/updated_commodity_inputs_all.pkl')
            else:
                self.updated_commodity_inputs.to_pickle('updated_commodity_inputs_all.pkl')

    def get_rmse_r2(self):
        param_variable_map = {'Total demand':'total_demand','Primary demand':'primary_demand',
            'Primary commodity price':'primary_commodity_price','Primary supply':'primary_supply',
            'Scrap demand':'scrap_demand'}
        rmse_list = []
        r2_list = []
        for param in self.objective_parameters:
            if 'SD' not in param:
                historical = self.historical_data[param]
                simulated = getattr(self.mod,param_variable_map[param])
                if hasattr(simulated,'columns') and 'Global' in simulated.columns:
                    simulated = simulated['Global']
                if self.normalize_objectives:
                    historical /= historical.iloc[0]
                    simulated /= simulated.iloc[0]
                rmse = self.calculate_rmse_r2(simulated,historical,True)
                r2 = self.calculate_rmse_r2(simulated,historical,False)
            else:
                if 'Conc' in param:
                    rmse = self.calculate_rmse_r2(self.mod.primary_supply,self.mod.primary_demand,True)
                    if self.normalize_objectives:
                        rmse /= self.mod.primary_demand.iloc[0] if not hasattr(self.mod.primary_demand,'columns') else self.mod.primary_demand['Global'].iloc[0]
                    r2 = self.calculate_rmse_r2(self.mod.primary_supply,self.mod.primary_demand,False)
                elif 'Scrap' in param:
                    rmse = self.calculate_rmse_r2(self.mod.scrap_supply,self.mod.scrap_demand,True)
                    if self.normalize_objectives:
                        rmse /= self.mod.scrap_demand.iloc[0] if not hasattr(self.mod.scrap_demand,'columns') else self.mod.scrap_demand['Global'].iloc[0]
                    r2 = self.calculate_rmse_r2(self.mod.scrap_supply,self.mod.scrap_demand,False)
                elif 'Ref' in param:
                    rmse = self.calculate_rmse_r2(self.mod.refined_supply,self.mod.refined_demand,True)
                    if self.normalize_objectives:
                        rmse /= self.mod.refined_demand.iloc[0] if not hasattr(self.mod.refined_demand,'columns') else self.mod.refined_demand['Global'].iloc[0]
                    r2= self.calculate_rmse_r2(self.mod.refined_supply,self.mod.refined_demand,False)
            
            rmse_list += [(rmse,0)]
            r2_list+= [(r2,0)]
        return rmse_list, r2_list
            
    def historical_sim_check_demand(self, n_scenarios):
        '''
        Varies the parameters for demand (sector_specific_dematerialization_tech_growth,
        sector_specific_price_response, region_specific_price_response, and
        intensity_response_to_gdp) to minimize the RMSE between simulated
        demand and historical.

        Uses Ax Adaptive Experimentation Platform Bayesian Optimization to find these
        parameters: https://ax.dev/tutorials/gpei_hartmann_service.html,
        install: https://ax.dev/, pip3 install ax-platform

        Adds/updates a dataframe self.updated_commodity_inputs that contains all demand
        updates for all commodities, which is saved in updated_commodity_inputs.pkl
        '''
        self.pkl_filename = self.pkl_filename.split('.pkl')[0]+'_DEM.pkl'
        self.notes += ' check demand'
        if os.path.exists(self.pkl_filename) and not self.overwrite:
            big_df = pd.read_pickle(self.pkl_filename)
        else:
            big_df = pd.DataFrame([],['version','notes','hyperparam','results'],[])
            big_df.to_pickle(self.pkl_filename)
        if 'Primary commodity price' not in self.historical_data.columns:
            raise ValueError('require a price input in primary commodity price for historical_sim_check_demand to work properly')
        self.update_changing_base_parameters_series()

        self.mod = demandModel(verbosity=0,simulation_time=self.simulation_time)
        params_to_change = ['sector_specific_dematerialization_tech_growth','sector_specific_price_response','region_specific_price_response','intensity_response_to_gdp']
        print(params_to_change)

        ax_client = AxClient(random_seed=self.random_state)
        ax_client.create_experiment(
            name="minimize_RMSE_with_real",
            parameters=[
                {
                    "name": "sector_specific_dematerialization_tech_growth",
                    "type": "range",
                    "bounds": [0.001, 0.08],
                    "value_type": "float",  # Optional, defaults to inference from type of "bounds".
                    "log_scale": False,  # Optional, defaults to False.
                },
                {
                    "name": "sector_specific_price_response",
                    "type": "range",
                    "value_type": "float",
                    "bounds": [0.001, 0.3],
                },
                {
                    "name": "region_specific_price_response",
                    "type": "range",
                    "value_type": "float",
                    "bounds": [0.001, 0.3],
                },
                {
                    "name": "intensity_response_to_gdp",
                    "type": "range",
                    "value_type": "float",
                    "bounds": [0.001, 1.5],
                },
            ],
            objective_name="RMSE",
            minimize=True, # default False
        )
        self.rmse_df = pd.DataFrame()
        for n in np.arange(0,n_scenarios):
            if self.verbosity>-1:
                print(f'Scenario {n+1}/{n_scenarios}')
            self.mod = demandModel(verbosity=0,simulation_time=self.simulation_time)
            self.mod.commodity_price_series = self.historical_data['Primary commodity price']
            self.mod.commodity_price_series = pd.concat([pd.Series(self.mod.commodity_price_series.iloc[0],np.arange(1900,self.simulation_time[0])),
                                                    self.mod.commodity_price_series])
            self.mod.version = '220620'

            ###### CHANGING BASE PARAMETERS ######
            changing_base_parameters_series = self.changing_base_parameters_series.copy()
            for base in np.intersect1d(self.mod.hyperparam.index, changing_base_parameters_series.index):
                self.mod.hyperparam.loc[base,'Value'] = changing_base_parameters_series[base]
                if n==0 and self.verbosity>-1:
                    print(base,changing_base_parameters_series[base])
            self.hyperparam_copy = self.mod.hyperparam.copy()

            ###### UPDATING PARAMETERS ######
            parameters, trial_index = ax_client.get_next_trial()
            new_param_series = pd.Series(parameters,params_to_change)
            self.mod.hyperparam.loc[params_to_change,'Value'] = new_param_series*np.sign(self.mod.hyperparam.loc[params_to_change,'Value'])
            self.check_run_append(self.mod)
            rmse = ((self.mod.demand.sum(axis=1)-self.historical_data['Total demand'])**2).loc[self.simulation_time].astype(float).sum()**0.5
            new_param_series.loc['RMSE'] = rmse
            new_param_series = pd.concat([new_param_series],keys=[n])
            self.rmse_df = pd.concat([self.rmse_df,new_param_series])

            ax_client.complete_trial(trial_index=trial_index, raw_data=rmse)

        rmse_df = self.rmse_df.copy()
        rmse_df.index = pd.MultiIndex.from_tuples(rmse_df.index)
        rmse_df = rmse_df[0]
        rmse_df = rmse_df.unstack()
        self.rmse_df = rmse_df.copy()
        best_params = pd.DataFrame(rmse_df.loc[rmse_df.where(rmse_df!=0).dropna()['RMSE'].idxmin()].drop('RMSE'))
        best_params = best_params.rename(columns={best_params.columns[0]:self.material})
        if os.path.exists('data/updated_commodity_inputs.pkl'):
            self.updated_commodity_inputs = pd.read_pickle('data/updated_commodity_inputs.pkl')
            self.updated_commodity_inputs.loc[:,self.material] = best_params[self.material]
            self.updated_commodity_inputs.to_pickle('data/updated_commodity_inputs.pkl')
        elif os.path.exists('updated_commodity_inputs.pkl'):
            self.updated_commodity_inputs = pd.read_pickle('updated_commodity_inputs.pkl')
            self.updated_commodity_inputs.loc[:,self.material] = best_params[self.material]
            self.updated_commodity_inputs.to_pickle('updated_commodity_inputs.pkl')
        else:
            self.updated_commodity_inputs = best_params.copy()
            if os.path.exists('data'):
                self.updated_commodity_inputs.to_pickle('data/updated_commodity_inputs.pkl')
            else:
                self.updated_commodity_inputs.to_pickle('updated_commodity_inputs.pkl')
        for i in best_params.index:
            self.changing_base_parameters_series.loc[i] = best_params[self.material][i]
        self.notes = self.notes.split(' check demand')[0]
        self.pkl_filename = self.pkl_filename.split('_DEM')[0]+'.pkl'

    def check_hist_demand_convergence(self):
        '''
        Run after historical_sim_check_demand() method to plot
        the results.
        '''
        historical_data = self.historical_data.copy()
        big_df = pd.read_pickle(self.pkl_filename.split('.pkl')[0]+'_DEM.pkl')
        ind = big_df.loc['results'].dropna().index
        res = pd.concat([big_df.loc['results',i] for i in ind],keys=ind)
        tot_demand = res['Total demand'].unstack(0).loc[self.simulation_time[0]:self.simulation_time[-1]]
        # below: calculates RMSE (root mean squared error) for each column and idxmin gets the index corresponding to the minimum RMSE
        best_demand = tot_demand[(tot_demand.astype(float).apply(lambda x: x-historical_data['Total demand'])**2).sum().astype(float).idxmin()]
        fig,ax = easy_subplots(3,dpi=self.dpi)
        for i,a in enumerate(ax[:2]):
            if i==0:
                tot_demand.plot(linewidth=1,alpha=0.3,legend=False,ax=a)
            historical_data['Total demand'].plot(ax=a,label='Historical',color='k',linewidth=4)
            best_demand.plot(ax=a,label='Best simulated',color='blue',linewidth=4)
            if i==1:
                a.legend()
            a.set(title='Total demand over time',xlabel='Year',ylabel='Total demand (kt)')
        do_a_regress(best_demand.astype(float),historical_data['Total demand'].astype(float),ax=ax[2],xlabel='Simulated',ylabel='Historical')
        ax[-1].set(title='Historical regressed on simulated')
        plt.suptitle('Total demand, varying demand parameters (sensitivity historical_sim_check_demand result)',fontweight='bold')
        fig.tight_layout()

        hyps=['sector_specific_dematerialization_tech_growth','sector_specific_price_response','region_specific_price_response','intensity_response_to_gdp']
        hyper = pd.concat([big_df.loc['hyperparam'].dropna().loc[i].loc[hyps,'Value'] for i in ind],keys=ind,axis=1)
        hyper.loc['RMSE'] = (tot_demand.astype(float).apply(lambda x: x-historical_data['Total demand'])**2).sum().astype(float)**0.5
        fig,ax=easy_subplots(4,dpi=self.dpi)
        for i,a in zip([i for i in hyper.index if i!='RMSE'],ax):
            a.scatter(hyper.loc[i],hyper.loc['RMSE'])
            a.set(title=i,ylabel='RMSE (kt)',xlabel='Elasticity value')
        plt.suptitle('Checking correlation betwen RMSE and parameter value (sensitivity historical_sim_check_demand_result)',fontweight='bold')

    def run_historical_monte_carlo(self, n_scenarios, random_state=220621,
                                   sensitivity_parameters=['elas','incentive_opening_probability','improvements','refinery_capacity_fraction_increase_mining'],
                                   bayesian_tune=False,n_params=2):
        '''
        Wrapper to run the run_monte_carlo() method on historical data

        Always runs an initial scenario with default parameters, so remember to skip that one
        when looking at the resulting pickle file
        --------------
        n_scenarios: int, typically use 200, but unsure yet whether we could improve
            performance by using more.
        bayesian_tune: bool, True allows tuning to the historical values given.
        n_params: int, the number of columns used in tuning for Bayesian optimization,
            counting from the left in the sheet with the matching commodity name in
            case study data.xlsx. We have been able to find 3 for everything so far:
            Total demand, Primary commodity price, and Primary production
        '''
        self.random_state = random_state
        if os.path.exists('data/updated_commodity_inputs.pkl'):
            self.updated_commodity_inputs = pd.read_pickle('data/updated_commodity_inputs.pkl')
            if self.verbosity>-1: print('updated_commodity_inputs source: data/updated_commodity_inputs.pkl')
        elif os.path.exists('updated_commodity_inputs.pkl'):
            self.updated_commodity_inputs = pd.read_pickle('updated_commodity_inputs.pkl')
            if self.verbosity>-1: print('updated_commodity_inputs source: updated_commodity_inputs.pkl')
        elif hasattr(self,'updated_commodity_inputs'):
            pass
        else:
            raise ValueError('updated_commodity_inputs.pkl does not exist in the expected locations (in this directory, in data folder, as attribute of Sensitivity)')

        if hasattr(self,'material') and self.material!='':
            best_params = self.updated_commodity_inputs[self.material].copy()
        else:
            raise ValueError('need to use a string input to changing_base_parameters_series in Sensitivity initialization to run this method')

        demand_params = ['sector_specific_dematerialization_tech_growth','sector_specific_price_response','region_specific_price_response','intensity_response_to_gdp']
        for i in demand_params:
            self.changing_base_parameters_series.loc[i] = best_params[i]

        self.run_monte_carlo(n_scenarios=n_scenarios,
                             random_state=random_state,
                             sensitivity_parameters=sensitivity_parameters,
                             bayesian_tune=bayesian_tune,
                             n_params=n_params)

    def create_potential_append(self,big_df,notes,reg_results,initialize=False):
        '''
        Sets up a pandas series that could be appended to our big dataframe
        that is used for saving, such that we can check whether this
        combination of parameters already exists in the big dataframe or not
        '''
        new_col_name=0 if len(big_df.columns)==0 else max(big_df.columns)+1
        if type(self.mod)==Integration:
            if initialize:
                mining = pd.DataFrame([],[],['hyperparam','ml'])
                refine = pd.DataFrame([],[],['hyperparam'])
                demand = pd.DataFrame([],[],['hyperparam'])
            else:
                mining = deepcopy([self.mod.mining])[0]
                refine = deepcopy([self.mod.refine])[0]
                demand = deepcopy([self.mod.demand])[0]
            potential_append = pd.DataFrame(np.array([self.mod.version, notes, self.mod.hyperparam, mining.hyperparam,
                                refine.hyperparam, demand.hyperparam, reg_results, mining.ml],dtype=object)
                                             ,index=[
                                    'version','notes','hyperparam','mining.hyperparam','refine.hyperparam','demand.hyperparam','results','mine_data'
                                ],columns=[new_col_name])
        elif type(self.mod)==demandModel:
            potential_append = pd.DataFrame(np.array([self.mod.version, notes, self.mod.hyperparam, reg_results],dtype=object)
                                         ,index=[
                                'version','notes','hyperparam','results'
                            ],columns=[new_col_name])
        return potential_append

    def check_run_append(self):
        '''
        Checks whether the proposed set of hyperparameters has already been run and saved
        in the current big result dataframe. If it has, it skips. Otherwise, it runs the
        scenario and appends it to the big dataframe, resaving it.
        '''
        big_df = pd.read_pickle(self.pkl_filename)
        potential_append = self.create_potential_append(big_df=big_df,notes=self.notes,reg_results=[],initialize=True)
        if type(self.mod)==demandModel or self.overwrite or check_equivalence(big_df, potential_append)[0]:
            if self.verbosity>-1:
                print('\tScenario does not already exist, running...')
            if type(self.mod)==Integration:
                try:
                    self.mod.run()
                except MemoryError:
                    if self.verbosity>-1:
                        print('************************MemoryError, no clue what to do about this************************')
                    param_variable_map = {'Total demand':'total_demand','Primary demand':'primary_demand',
                        'Primary commodity price':'primary_commodity_price','Primary supply':'primary_supply',
                        'Scrap demand':'scrap_demand'}
                    for param in self.objective_parameters:
                        historical = self.historical_data[param]
                        simulated = getattr(self.mod,param_variable_map[param])
                        if hasattr(simulated,'columns') and 'Global' in simulated.columns:
                            getattr(self.mod,param_variable_map[param]).loc[:,'Global'] = historical*5
                        else:
                            getattr(self.mod,param_variable_map[param]).loc[:] = historical*5
                    raise MemoryError
            elif type(self.mod)==demandModel:
                for i in self.mod.simulation_time:
                    self.mod.i = i
                    self.mod.run()

            if hasattr(self,'val'):
                notes = self.notes+ f', {i}={self.val}'
            else:
                notes = self.notes+''
            ind = [j for j in self.hyperparam_copy.index if type(self.hyperparam_copy['Value'][j]) not in [np.ndarray,list]]
            z = self.hyperparam_copy['Value'][ind].dropna()!=self.mod.hyperparam['Value'][ind].dropna()
            z = [j for j in z[z].index]
            if len(z)>0:
                for zz in z:
                    notes += ', {}={}'.format(zz,self.mod.hyperparam['Value'][zz])

            if type(self.mod)==Integration: reg_results = create_result_df(self.mod)
            elif type(self.mod) == demandModel: reg_results = pd.concat([self.mod.demand.sum(axis=1),self.mod.commodity_price_series],axis=1,keys=['Total demand','Primary commodity price'])
            potential_append = self.create_potential_append(big_df=big_df,notes=notes,reg_results=reg_results,initialize=False)

            big_df = pd.concat([big_df,potential_append],axis=1)
            # self.big_df = pd.concat([self.big_df,potential_append],axis=1)
            big_df.to_pickle(self.pkl_filename)
            if self.verbosity>-1:
                print('\tScenario successfully saved\n')
        else:
            if self.verbosity>-1:
                print('\tScenario already exists\n')

#         direct_melt_elas_scrap_spread
#         collection_elas_scrap_price

def grade_predict(ci):
    '''
    see slide 116 in the file:
    C:/Users/ryter/Dropbox (MIT)/
    Group Research Folder_Olivetti/
    Displacement/04 Presentations/
    John/Weekly Updates/20210825
    Generalization.pptx
    a + b*log(price),
    a = 8.1748
    b = -0.9477
    R^2=0.944 for commodities in SNL
    '''
    price = ci['primary_commodity_price']
    grade = 8.1748 -0.9477*np.log(price)
    grade = np.exp(grade)
    ci.loc['primary_ore_grade_mean'] = grade
    return ci

def generate_commodity_inputs(commodity_inputs, random_state):
    '''
    Used for generating random \"materials\" that can then be run
    through a Monte Carlo using the run_monte_carlo() method.
    '''
    ci = commodity_inputs.copy()
    if 'Byproduct status' in ci.index:
        ci = ci.drop('Byproduct status')
    if 'mine_cu_margin_elas' in ci.index:
        ci = ci.drop('mine_cu_margin_elas')
        # this one already gets covered in the Sensitivity class
    dists = pd.DataFrame(np.nan,ci.index,['dist','p1','p2'])
    ci.loc[:] = np.nan
    rs = random_state
    if 'values in case_study_data.xlsx':
        ci.loc['initial_demand'] = 10000
        ci.loc['Regional production fraction of total production, Global'] = 1
        dists.loc['historical_growth_rate'] = 'uniform',0.01,0.6
        dists.loc['china_fraction_demand'] = 'uniform',0.05,0.7
        sector_dists = [i for i in dists.index if 'sector_dist' in i]
        dists.loc[sector_dists] = 'uniform',0,1
        mine_types_main = [i for i in dists.index if i in ['minetype_prod_frac_underground','minetype_prod_frac_openpit']]
        mine_types_alt  = [i for i in dists.index if 'minetype' in i and i not in mine_types_main]
        mine_types  = [i for i in dists.index if 'minetype' in i]
        dists.loc[mine_types_main] = 'uniform',0,1
        dists.loc[mine_types_alt] = 'uniform',0,0.1
        dists.loc['Recycling input rate, Global'] = 'uniform',0,0.5
        dists.loc['Regional production fraction of total production, China'] = 'uniform',0.05,0.7
        dists.loc['Secondary refinery fraction of recycled content, Global'] = 'uniform',0,1
        dists.loc['SX-EW fraction of production, Global'] = 'uniform',0,0.5
        dists.loc['primary_production_mean'] = 'uniform',1e-4,1e-2
        dists.loc['primary_production_var'] = 'uniform',0.5,2
        dists.loc['primary_ore_grade_mean'] = 'norm',1,0.2
        dists.loc['primary_ore_grade_var'] = 'uniform',0.01,1.5
        dists.loc['primary_commodity_price'] = 'uniform',100,1e8
        dists.loc['initial_scrap_spread'] = 'uniform',0,0.4
        dists.loc['lifetime_mean_construction'] = 'uniform',10,50
        dists.loc['lifetime_mean_electrical'] = 'uniform',10,30
        dists.loc['lifetime_mean_industrial'] = 'uniform',10,30
        dists.loc['lifetime_mean_other'] = 'uniform',1,15
        dists.loc['lifetime_mean_transport'] = 'uniform',5,20

    if 'in integration hyperparam':
        dists.loc['incentive_opening_probability'] = 'uniform',0.001,0.1
        dists.loc['refinery_follows_concentrate_supply'] = 'bool',0,1
        dists.loc['random_state'] = 'discrete',1,220609

    if 'in mining':
        dists.loc['ramp_up_years'] = 'discrete',2,5
        dists.loc['ramp_up_cu'] = 'uniform',0.4,0.8
        dists.loc['primary_oge_scale'] = 'uniform',0.1,0.5
        dists.loc['discount_rate'] = 'uniform',0.05,0.15
        dists.loc['ramp_down_cu'] = 'uniform',0.2,0.6
        dists.loc['close_years_back'] = 'discrete',2,10
        dists.loc['years_for_roi'] = 'discrete',10,20
        dists.loc['close_price_method'] = 'close_price_method'
        dists.loc['close_probability_split_max'] = 'uniform',0,1
        dists.loc['close_probability_split_mean'] = 'uniform',0,1
        dists.loc['close_probability_split_min'] = 'uniform',0,1
        close_prob = [i for i in dists.index if 'close_probability_split' in i]
        dists.loc['reserves_ratio_price_lag'] = 'discrete',1,10

    unis = dists['dist']=='uniform'
    dists.loc[unis,'p2'] = dists.loc[unis,'p2']-dists.loc[unis,'p1']
    for i in dists.dropna().index:
        if dists['dist'][i] in ['uniform']:
            val = getattr(stats,dists['dist'][i]).rvs(dists['p1'][i],dists['p2'][i],random_state=rs)
        elif dists['dist'][i] in ['discrete']:
            seed(rs)
            val = int(sample(list(np.arange(dists['p1'][i],dists['p2'][i]+1)),1)[0])
        elif dists['dist'][i] in ['bool']:
            seed(rs)
            val = sample([False,True],1)[0]
        elif i=='close_price_method':
            val = 'probabilistic'
    #     print(i, dists['p1'][i], dists['p2'][i], val)


        if i in ['Secondary fraction of recycled content, Global']:
            seed(rs)
            ci.loc[i] = sample([0,val,1],1)[0]
        elif i in ['SX-EW fraction of production, Global']:
            # 10% chance of having SX-EW production
            seed(rs)
            ci.loc[i] = sample(list(np.repeat([0],9))+[val],1)[0]
#         elif i in ['ramp_up_cu']:
#             seed(rs)
#             ci.loc[i] = sample([0,val],1)[0]
    #     elif i in ['incentive_opening_probability']:
              # currently commented out because I want to keep these scenario sets separate
    #         # 10% chance of having the incentive_opening_probability==0
    #         seed(rs)
    #         ci.loc[i] = sample(list(np.repeat(val,9))+[0],1)[0]
        else:
            ci.loc[i] = val
        rs += 1

    if 'values in case_study_data.xlsx':
        ci.loc[sector_dists] /= ci.loc[sector_dists].sum()
        ci.loc[mine_types] /= ci.loc[mine_types].sum()
        fab_and_life = [i for i in ci.index if 'fabrication_efficiency' in i]
        ci.loc[fab_and_life] = commodity_inputs.loc[fab_and_life]
        ci.loc['Recycling input rate, China'] = ci.loc['Recycling input rate, Global']
        ci.loc['Secondary refinery fraction of recycled content, China'] = ci.loc['Secondary refinery fraction of recycled content, Global']
        ci.loc['SX-EW fraction of production, China'] = ci.loc['SX-EW fraction of production, Global']
        ci.loc['Total production, Global'] = ci.loc['initial_demand'] * stats.uniform.rvs(0.95,0.1, random_state=rs)
        ci.loc['primary_production'] = ci['Total production, Global']*(1-ci['Recycling input rate, Global'])
        ci.loc['primary_production_mean'] *= ci['primary_production']
        grade_multiplier = abs(ci['primary_ore_grade_mean'])
        ci = grade_predict(ci)
        ci.loc['primary_ore_grade_mean'] *= grade_multiplier

    if 'values in integration hyperparam':
        ci.loc['presimulate_n_years'] = 10 if ci['ramp_up_cu']!=0 else 5*ci['ramp_up_years']
        if ci['incentive_opening_probability']==0:
            ci.loc['presimulate_n_years'] = 10*ci['ramp_up_years']
            ci.loc['start_calibrate_years'] = 5
            ci.loc['end_calibrate_years'] = 10 if 10<ci['presimulate_n_years'] else ci['presimulate_n_years']
        else:
            ci.loc['presimulate_n_years'] = 5*ci['ramp_up_years']
            ci.loc['incentive_require_tune_years'] = ci['presimulate_n_years']

    if 'values in mining param':
        ci.loc[close_prob] /= ci[close_prob].sum()

    if 'values in demand param':
        n = 0
        for i in ['construction','electrical','industrial','other','transport']:
            ci.loc['lifetime_sigma_'+i] = stats.uniform.rvs(0.1,0.4,random_state=n+rs)*ci['lifetime_mean_'+i]
            n += 1

    ci.loc['incentive_require_tune_years'] = 10
    ci.loc['presimulate_n_years'] = 10
    ci.loc['end_calibrate_years'] = 10
    ci.loc['start_calibrate_years'] = 5
    ci.loc['refinery_follows_concentrate_supply'] = False
    ci.loc['incentive_opening_probability'] = 0
    return ci
