import numpy as np
import pandas as pd
idx = pd.IndexSlice
from matplotlib import pyplot as plt
from scipy import stats
from modules.integration import Integration
from random import seed, sample, shuffle
from modules.demand_class import demandModel
from modules.mining_class import miningModel
from modules.scenario_parser import *
from modules.useful_functions import easy_subplots, do_a_regress
import os

from copy import deepcopy

from skopt import Optimizer
from joblib import Parallel, delayed
from sklearn.metrics import mean_squared_error, r2_score

import warnings
# warnings.filterwarnings('error')
# np.seterr(all='raise')

def create_result_df(self,integ):
    '''
    takes Integration object, returns regional results. Used within the sensitivity
    function to convert the individual model run to results we can interpret later.
    '''
    reg_results = pd.Series(np.nan,['Global','China','RoW'],dtype=object)

    if type(integ.mining.ml)!=pd.core.frame.DataFrame:
        integ.mining.ml = integ.mining.ml.generate_df(redo_strings=True)
    new = integ.mining.ml.loc[integ.mining.ml['Opening']>integ.simulation_time[0]]
    old = integ.mining.ml.loc[integ.mining.ml['Opening']<=integ.simulation_time[0]]
    old_new_mines = pd.concat([
        old.loc[:,'Production (kt)'].groupby(level=0).sum(),
        new.loc[:,'Production (kt)'].groupby(level=0).sum(),
        (old['Production (kt)']*old['Head grade (%)']).groupby(level=0).sum()/old.loc[:,'Production (kt)'].groupby(level=0).sum().replace(0,np.nan),
        (new['Production (kt)']*new['Head grade (%)']).groupby(level=0).sum()/new.loc[:,'Production (kt)'].groupby(level=0).sum().replace(0,np.nan),
        (old['Production (kt)']*old['Minesite cost (USD/t)']).groupby(level=0).sum()/old.loc[:,'Production (kt)'].groupby(level=0).sum().replace(0,np.nan),
        (new['Production (kt)']*new['Minesite cost (USD/t)']).groupby(level=0).sum()/new.loc[:,'Production (kt)'].groupby(level=0).sum().replace(0,np.nan),
        (old['Production (kt)']*old['Total cash margin (USD/t)']).groupby(level=0).sum()/old.loc[:,'Production (kt)'].groupby(level=0).sum().replace(0,np.nan),
        (new['Production (kt)']*new['Total cash margin (USD/t)']).groupby(level=0).sum()/new.loc[:,'Production (kt)'].groupby(level=0).sum().replace(0,np.nan),
        integ.mining.resources_contained_series, integ.mining.reserves_ratio_with_demand_series
        ],
        keys=['Old mine prod.','New mine prod.',
              'Old mine grade','New mine grade',
              'Old mine cost','New mine cost',
              'Old mine margin','New mine margin',
              'Reserves','Reserves ratio with production'],axis=1).fillna(0)

    # print(integ.additional_scrap)
    addl_scrap = integ.additional_scrap.sum(axis=1).unstack()
    addl_scrap.loc[:,'Global'] = addl_scrap.sum(axis=1)
    for reg in reg_results.index:
        results = pd.concat([integ.total_demand.loc[:,reg],integ.scrap_demand.loc[:,reg],integ.scrap_supply[reg],
               integ.concentrate_demand[reg],integ.concentrate_supply,
               (integ.mining.ml['Production (kt)']*integ.mining.ml['Head grade (%)']).groupby(level=0).sum()/integ.mining.ml.loc[:,'Production (kt)'].groupby(level=0).sum().replace(0,np.nan),
               (integ.mining.ml['Production (kt)']*integ.mining.ml['Minesite cost (USD/t)']).groupby(level=0).sum()/integ.mining.ml.loc[:,'Production (kt)'].groupby(level=0).sum().replace(0,np.nan),
               (integ.mining.ml['Production (kt)']*integ.mining.ml['Total cash margin (USD/t)']).groupby(level=0).sum()/integ.mining.ml.loc[:,'Production (kt)'].groupby(level=0).sum().replace(0,np.nan),
               old_new_mines['Old mine prod.'],old_new_mines['New mine prod.'],
               old_new_mines['Old mine grade'],old_new_mines['New mine grade'],
               old_new_mines['Old mine cost'],old_new_mines['New mine cost'],
               old_new_mines['Old mine margin'],old_new_mines['New mine margin'],
               (integ.mining.ml['Production (kt)']>0).groupby(level=0).sum(),
               integ.mining.n_mines_opening, integ.mining.n_mines_closing,
               integ.refined_demand.loc[:,reg],integ.refined_supply[reg],
               integ.secondary_refined_demand.loc[:,reg],integ.direct_melt_demand.loc[:,reg],
               integ.scrap_spread[reg],integ.tcrc,integ.primary_commodity_price,
               integ.refine.ref_stats[reg]['Primary CU'], integ.refine.ref_stats[reg]['Secondary CU'],
               integ.refine.ref_stats[reg]['Secondary ratio'],
               integ.refine.ref_stats[reg]['Primary capacity'], integ.refine.ref_stats[reg]['Secondary capacity'],
               integ.refine.ref_stats[reg]['Primary production'], integ.refine.ref_stats[reg]['Secondary production'],
               integ.additional_direct_melt[reg],integ.additional_secondary_refined[reg], addl_scrap[reg],
               integ.sxew_supply,integ.primary_supply[reg],integ.primary_demand[reg],integ.mine_production
              ],axis=1,
              keys=['Total demand','Scrap demand','Scrap supply',
                    'Conc. demand','Conc. supply',
                    'Mean mine grade','Mean total minesite cost','Mean total cash margin',
                    'Old mine prod.','New mine prod.',
                    'Old mine grade','New mine grade',
                    'Old mine cost','New mine cost',
                    'Old mine margin','New mine margin',
                    'Number of operating mines',
                    'Number of mines opening','Number of mines closing',
                    'Ref. demand','Ref. supply',
                    'Sec. ref. cons.','Direct melt',
                    'Spread','TCRC','Refined price',
                    'Refinery pri. CU','Refinery sec. CU','Refinery SR',
                    'Pri. ref. capacity','Sec. ref. capacity',
                    'Pri. ref. prod.','Sec. ref. prod.',
                    'Additional direct melt','Additional secondary refined','Additional scrap',
                    'SX-EW supply','Primary supply','Primary demand','Mine production'])
        if reg=='Global':
            scrap_collected = integ.demand.old_scrap_collected.groupby(level=0).sum()
            collection = integ.demand.old_scrap_collected.groupby(level=0).sum()/integ.demand.eol.groupby(level=0).sum().replace(0,np.nan)
            old_scrap = integ.demand.old_scrap_collected.groupby(level=0).sum().sum(axis=1)
            new_scrap = integ.demand.new_scrap_collected.groupby(level=0).sum().sum(axis=1)
        else:
            scrap_collected = integ.demand.old_scrap_collected.loc[idx[:,reg],:].droplevel(1)
            collection = integ.collection_rate.loc[idx[:,reg],:].droplevel(1).fillna(0)
            old_scrap = integ.demand.old_scrap_collected.loc[idx[:,reg],:].droplevel(1).sum(axis=1)
            new_scrap = integ.demand.new_scrap_collected.loc[idx[:,reg],:].droplevel(1).sum(axis=1)
        scrap_collected = scrap_collected.rename(columns=dict(zip(scrap_collected.columns,['Old scrap '+j.lower() for j in scrap_collected.columns])))
        collection = collection.rename(columns=dict(zip(collection.columns,['Collection rate '+j.lower() for j in collection.columns])))
        scraps = pd.concat([old_scrap,new_scrap],axis=1,keys=['Old scrap collection','New scrap collection'])
        results = pd.concat([results,collection,scrap_collected,scraps],axis=1)
        time_index = np.arange(self.simulation_time[0]-self.changing_base_parameters_series['presimulate_n_years'],self.simulation_time[-1]+1)
        time_index = [i for i in results.index if i in time_index]
        if self.trim_result_df: results = results.loc[time_index]
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

def is_pareto_efficient_simple(costs):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient

    from Peter at https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
    """
    is_efficient = np.ones(costs.shape[0], dtype = bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient]<c, axis=1)  # Keep any point with a lower cost
            is_efficient[i] = True  # And keep self
    return is_efficient

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
                 user_data_folder=None,
                 static_data_folder=None,
                 output_data_folder=None,
                 changing_base_parameters_series=0,
                 additional_base_parameters=0,
                 params_to_change=0,
                 n_per_param=5,
                 notes='Initial run',
                 simulation_time = np.arange(2019,2041),
                 train_time = np.arange(2001,2020),
                 byproduct=False,
                 verbosity=0,
                 param_scale=0.5,
                 scenarios=[''],
                 OVERWRITE=False,
                 random_state=220620,
                 incentive_opening_probability_fraction_zero=0,
                 include_sd_objectives=False,
                 use_rmse_not_r2=True,
                 dpi=50,
                 normalize_objectives=False,
                 use_alternative_gold_volumes=True,
                 historical_price_rolling_window=5,
                 force_integration_historical_price=False,
                 constrain_tuning_to_sign=True,
                 constrain_previously_tuned=False,
                 dont_constrain_demand=True,
                 price_to_use='log',
                 use_historical_price_for_mine_initialization=True,
                 timer=None,
                 save_mining_info=False,
                 trim_result_df=True,
                 save_regional_data=False,
                 ):
        '''
        Initializing Sensitivity class.

        ---------------
        pkl_filename: str, path/filename ending in pkl for where you want to save the results of the
            sensitivity you are about to run
        user_data_folder: str, path to where case study data.xlsx and price adjustment results.xlsx are saved
        static_data_folder: str, path to where additional do-not-edit input files are saved
        output_data_folder: str, path to where model outputs will be saved
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
        simulation_time: np.ndarray, array of years for running the simulation. Typically either 2001 through
            2019, 2019 through 2040, or 2001 through 2040.
        train_time: np.ndarray, array of years to use for training the Bayesian optimization model. Test years
            would be any in simulation_time not in train_time. Currently trying to use 2001 through 2014 for
            train_time (approx. 75% of historical period we have), but default is set to 2001-2019 so that
            demand and mining pre-tuning and integration pre-tuning are not disturbed unless intentionally
            (train-test split is not the default)
        byproduct: bool, whether we are simulating a mono-product mined commodity or a multi-product mined
            commodity. Have not yet had the chance to test out the True setting
        verbosity: int, tells how much we print as the model runs. Can be anything including -1 which suppresses
            nearly all output, and I think the highest value (above which nothing changes) is 4.
        param_scale: float, used in conjunction with params_to_change and n_per_param in the run() simple
            sensitivity function, where whichever parameter is selected will be scaled by +/- param_scale
        scenarios: list or array, including only strings. The contents of each string determine the scrap
            supply or demand changes that will be implemented. See the decode_scenario_name() function
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
        incentive_opening_probability_fraction_zero: float, used in the run_monte_carlo() method to determine the
            fraction of incentive_opening_probability values generated that are then set to zero, since setting
            to zero allows the model to determine the incentive_opening_probability value endogenously, picking
            the mean value from the most recent n simulations where incentive tuning used incentive_opening_probability
            to set the number of mines opening such that concentrate supply=demand
        include_sd_objectives: bool, whether to make the optimization try to also minimize the RMSE between
            supply and demand, default (and pretty much always should be) is False
        use_rmse_not_r2: bool, default True, whether to use RMSE instead of R2 as the variable we try to
            minimize in the optimization
        dpi: int, stands for dots per inch, and is used to change the resolution of the plots created using
            the check_hist_demand_convergence() method on a historical_sim_check_demand() method run
        normalize_objectives: bool, default False, determines whether the objectives should be normalized by
            their first-year value. Having it True appears to cause weird convergence behavior and I think this
            option should be avoided
        use_alternative_gold_volumes: bool, if True uses the alternative gold volume drivers since industrial
            represents bar and coin, and transport represents jewelry. False uses the default, which fails to
            capture the plateau seen in gold demand 2010-2019.
        historical_price_rolling_window: int, window size for rolling mean of historical price, if 1, no rolling mean
            is done.
        force_integration_historical_price: bool, whether to force Integration models to use historical price rather
            than let it evolve independently.
        constrain_tuning_to_sign: bool, whether to constrain parameter tuning to be within the (0,1) range (pre-scaling). If
            False, constrains tuning to be in (-1,1) range (pre-scaling). Default is True, False is used for statistical
            significance validation.
        constrain_previously_tuned: bool, if True, requires any bayesian optimization tuning parameters that have
            previously been tuned (by historical_sim_check_demand, meaning they are in the index of
            self.updated_commodity_inputs(_sub)) to be 0.001-2X their previously-tuned value, if the optimization
            is trying to tune them. If False, constraints are as they were previously.
        dont_constrain_demand: bool, if True, makes it so that the constrain_previously_tuned above does not apply
            to the parameters associated with demand
        price_to_use: str within the set [log,original,diff,case study data]. The first three refer to
            the respective columns in the data/price adjustment results.xlsx excel file, for the selected commodity.
            Using case study data causes the system to use the values from the case study data.xlsx excel sheet
            corresponding with the selected commodity. Using the other three values overwrites this input value,
            with the purpose being to have more historical price data prior to the historical simulation time
            such that the mining price expectation calculation has more data to draw from and we are not just
            giving constant values for data before simulation start time.
        timer: callback, if provided, use this function to measure time and print mean iteration time as well as ETA,
            if interested, ask Luca Montanelli for his function.
        save_mining_info: bool, default False. If True, saves mine-level data in the result dataframe, which takes
            up a lot of memory. Setting to False means it just saves a zero, and if you need to then access the
            mine-level data you have to re-run the simulation with this value as True
        trim_result_df: bool, default True. If True, the result dataframe will be trimmed to only include years of
            data for which the simulation was run, rather than going all the way back to 1912, to save memory
        save_regional_data: bool, default False. Whether to save the separate data for each region for refining and
            demand
        '''
        self.save_mining_info = save_mining_info
        self.trim_result_df = trim_result_df
        self.simulation_time = simulation_time
        self.use_historical_price_for_mine_initialization = use_historical_price_for_mine_initialization
        self.train_time = train_time
        self.changing_base_parameters_series = changing_base_parameters_series
        self.additional_base_parameters = additional_base_parameters
        self.user_data_folder = 'input_files/user_defined' if user_data_folder is None else user_data_folder
        self.static_data_folder = 'input_files/static' if static_data_folder is None else static_data_folder
        self.output_data_folder = 'output_files' if output_data_folder is None else output_data_folder
        self.case_study_data_file_path = f'{self.user_data_folder}/case study data.xlsx'
        self.price_adjustment_results_file_path = f'{self.user_data_folder}/price adjustment results.xlsx'
        self.price_to_use = price_to_use
        self.force_integration_historical_price = force_integration_historical_price
        self.use_alternative_gold_volumes = use_alternative_gold_volumes
        self.historical_price_rolling_window = historical_price_rolling_window
        self.constrain_tuning_to_sign = constrain_tuning_to_sign
        self.constrain_previously_tuned = constrain_previously_tuned
        self.dont_constrain_demand = dont_constrain_demand
        self.element_commodity_map = {'Al':'Aluminum','Au':'Gold','Cu':'Copper','Steel':'Steel','Co':'Cobalt','REEs':'REEs','W':'Tungsten','Sn':'Tin','Ta':'Tantalum','Ni':'Nickel','Ag':'Silver','Zn':'Zinc','Pb':'Lead','Mo':'Molybdenum','Pt':'Platinum','Te':'Telllurium','Li':'Lithium'}
        self.update_changing_base_parameters_series()

        self.byproduct = byproduct
        self.verbosity = verbosity
        self.param_scale = param_scale
        self.pkl_filename = pkl_filename

        self.params_to_change = params_to_change
        self.n_per_param = n_per_param
        self.notes = f'{notes}, price version: {price_to_use}, price rolling: {historical_price_rolling_window}, train_time: {train_time[0]}-{train_time[-1]}'
        self.scenarios = scenarios
        self.overwrite = OVERWRITE
        self.save_regional_data = save_regional_data
        self.random_state = random_state
        self.include_sd_objectives = include_sd_objectives
        self.incentive_opening_probability_fraction_zero = incentive_opening_probability_fraction_zero
        self.use_rmse_not_r2 = use_rmse_not_r2
        self.dpi = dpi

        self.normalize_objectives = normalize_objectives

        self.bayesian_tune = False # added here, gets overwritten in any of the bayesian tuning runs; just a flag so we know when regular scenarios are running so they get saved correctly

        self.timer = timer

        if self.overwrite and self.verbosity>0: print('WARNING, YOU ARE OVERWRITING AN EXISTING FILE')

        self.demand_params = ['sector_specific_dematerialization_tech_growth','sector_specific_price_response','intensity_response_to_gdp']
        self.historical_data_column_list = ['Total demand','Primary commodity price','Primary supply','Primary production','Scrap demand','Total production','Primary demand']
        self.historical_data_column_list = [j for j in self.historical_data_column_list if j in self.historical_data.columns]
        self.demand_or_mining = None

        self.n_files = 0

    def update_pkl_filename_path(self):
        if len(self.pkl_filename.split('output_data/'))>2:
            error_print_value = len(self.pkl_filename.split('output_data/'))-1
            raise ValueError('pkl_filename has {error_print_value} occurrences of `data/` which is not compatible with running methods, see update_pkl_filename_path function in integration_functions.py')
        if not os.path.exists('output_files'):
            os.mkdir('output_files')
        if not os.path.exists('output_files/Historical tuning'):
            os.mkdir('output_files/Historical tuning')
        if not os.path.exists('output_files/Simulation'):
            os.mkdir('output_files/Simulation')
        if self.material!='':
            if self.bayesian_tune and 'output_files/Historical tuning' not in self.pkl_filename:
                self.pkl_filename = self.pkl_filename.replace('output_files/','output_files/Historical tuning/')
            if not self.bayesian_tune and 'output_files/Simulation' not in self.pkl_filename:
                self.pkl_filename = self.pkl_filename.replace('output_files/','output_files/Simulation/')
        else:
            if not os.path.exists('output_files/Other'):
                os.mkdir('output_files/Other')
            self.pkl_filename = self.pkl_filename.replace('output_files/','output_files/Other/')

    def initialize_big_df(self):
        '''
        Initializes the big dataframe used to save all the results
        '''
        if os.path.exists(self.pkl_filename) and not self.overwrite:
            big_df = pd.read_pickle(self.pkl_filename)
        else:
            self.mod = Integration(static_data_folder=self.static_data_folder, user_data_folder=self.user_data_folder,
                                   simulation_time=self.simulation_time,
                                   verbosity=self.verbosity,byproduct=self.byproduct,scenario_name='',
                                   commodity=self.material, price_to_use=self.price_to_use,
                                   historical_price_rolling_window=self.historical_price_rolling_window,
                                   force_integration_historical_price=self.force_integration_historical_price,
                                   use_historical_price_for_mine_initialization=self.use_historical_price_for_mine_initialization)
            for base in self.changing_base_parameters_series.index:
                if base in self.demand_params:
                    self.mod.hyperparam.loc[base,'Value'] = abs(self.changing_base_parameters_series[base])*np.sign(self.mod.hyperparam.loc[base,'Value'])
                else:
                    self.mod.hyperparam.loc[base,'Value'] = self.changing_base_parameters_series[base]

            if self.bayesian_tune or (
                    hasattr(self, 'historical_data') and 'Primary commodity price' in self.historical_data.columns):
                self.mod.primary_commodity_price = self.historical_data['Primary commodity price'].dropna()
                self.mod.primary_commodity_price = pd.concat([pd.Series(self.mod.primary_commodity_price.iloc[0],
                                                                        np.arange(1900,
                                                                                  self.mod.primary_commodity_price.dropna().index[
                                                                                      0])),
                                                              self.mod.primary_commodity_price]).sort_index()
            if hasattr(self,'historical_data'):
                self.mod.historical_data = self.historical_data.copy()


            self.mod.run()
            big_df = pd.DataFrame(np.nan,index=[
                'version','notes','hyperparam','mining.hyperparam','refine.hyperparam','demand.hyperparam','results','mine_data'
            ],columns=[])
            reg_results = create_result_df(self, self.mod)
            if not self.save_mining_info: ml = [0]
            elif self.save_mining_info=='cost_curve':
                ml = self.mod.mining.ml.copy()[['Commodity price (USD/t)','Minesite cost (USD/t)','Total cash margin (USD/t)','TCRC (USD/t)','Head grade (%)','Recovery rate (%)','Payable percent (%)','Production (kt)','Opening','Simulated closure']]
            else: ml = self.mod.mining.ml.copy()
            big_df.loc[:,0] = np.array([self.mod.version, self.notes, self.mod.hyperparam, self.mod.mining.hyperparam,
                                        self.mod.refine.hyperparam, self.mod.demand.hyperparam, reg_results, ml],dtype=object)
            big_df.to_pickle(self.pkl_filename)
        self.big_df = big_df.copy()

    def update_changing_base_parameters_series(self):
        """
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
        """
        changing_base_parameters_series = self.changing_base_parameters_series
        simulation_time = self.simulation_time
        if type(changing_base_parameters_series)==str:
            self.material = changing_base_parameters_series
            input_file = pd.read_excel(self.case_study_data_file_path,index_col=0)
            commodity_inputs = input_file[changing_base_parameters_series]
            n_years_tune = 20
            commodity_inputs.loc['incentive_require_tune_years'] = n_years_tune
            commodity_inputs.loc['presimulate_n_years'] = n_years_tune
            commodity_inputs.loc['end_calibrate_years'] = n_years_tune
            commodity_inputs.loc['start_calibrate_years'] = 5
            commodity_inputs.loc['close_price_method'] = 'max'
            commodity_inputs = commodity_inputs.dropna()

            history_file = pd.read_excel(self.case_study_data_file_path,index_col=0,sheet_name=changing_base_parameters_series)
            historical_data = history_file.loc[[i for i in history_file.index if i!='Source(s)']].dropna(axis=1,how='all').astype(float)
            historical_data.index = historical_data.index.astype(int)
            if simulation_time[0] in historical_data.index and simulation_time[0]!=2019:
                historical_data = history_file.loc[[i for i in simulation_time if i in history_file.index]]
                if self.price_to_use!='case study data':
                    price_update_file = pd.read_excel(self.price_adjustment_results_file_path,index_col=0)
                    cap_mat = self.element_commodity_map[self.material]
                    price_map = {'log':'log('+cap_mat+')',  'diff':'∆'+cap_mat,  'original':cap_mat+' original'}
                    historical_price = price_update_file[price_map[self.price_to_use]].astype(float)
                    if not self.use_historical_price_for_mine_initialization:
                        historical_price = historical_price.loc[
                            [i for i in historical_price.index if i in self.simulation_time]]
                    historical_price.name = 'Primary commodity price'
                    historical_price.index = historical_price.index.astype(int)
                    if 'Primary commodity price' in historical_data.columns:
                        historical_data = pd.concat([historical_data.drop('Primary commodity price',axis=1),historical_price],axis=1)
                        historical_data.index = historical_data.index.astype(int)
                        historical_data = historical_data.sort_index().dropna(how='all')
                    else:
                        historical_data = pd.concat([historical_data,historical_price],axis=1).sort_index().dropna(how='all')
                if 'Primary commodity price' in historical_data.columns:
                    historical_data.loc[historical_data.index,'Primary commodity price'] = historical_data['Primary commodity price'].rolling(self.historical_price_rolling_window,min_periods=1,center=True).mean()
                original_demand = commodity_inputs['initial_demand']
                original_production = commodity_inputs['Total production, Global']
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

                if 'Total production' in historical_data.columns:
                    commodity_inputs.loc['Total production, Global'] = historical_data['Total production'][simulation_time[0]]
                elif 'Scrap demand' in historical_data.columns and 'Primary production' in historical_data.columns:
                    commodity_inputs.loc['Total production, Global'] = historical_data['Primary production'][simulation_time[0]]+historical_data['Scrap demand'][simulation_time[0]]
                elif 'Scrap demand' in historical_data.columns and 'Primary supply' in historical_data.columns:
                    commodity_inputs.loc['Total production, Global'] = historical_data['Primary supply'][simulation_time[0]]+historical_data['Scrap demand'][simulation_time[0]]
                elif 'Primary production' in historical_data.columns:
                    commodity_inputs.loc['Total production, Global'] = historical_data['Primary production'][simulation_time[0]]*original_production/original_primary_production
                elif 'Primary supply' in historical_data.columns:
                    commodity_inputs.loc['Total production, Global'] = historical_data['Primary supply'][simulation_time[0]]*original_production/original_primary_production
                else:
                    commodity_inputs.loc['Total production, Global'] = commodity_inputs['initial_demand']*original_production/commodity_inputs['Total production, Global']
                # if self.material=='Al': commodity_inputs.loc['Total production, Global'] = 36000
            self.historical_data = historical_data.copy()
            self.changing_base_parameters_series = commodity_inputs.copy()
        elif not hasattr(self,'material'):
            self.material = ''
            self.historical_data = pd.DataFrame()
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
            self.mod = Integration(static_data_folder=self.static_data_folder, user_data_folder=self.user_data_folder,
                                    simulation_time=self.simulation_time,
                                   verbosity=self.verbosity,byproduct=self.byproduct, commodity=self.material,
                                   price_to_use=self.price_to_use,
                                   historical_price_rolling_window=self.historical_price_rolling_window,
                                   force_integration_historical_price=self.force_integration_historical_price,
                                   use_historical_price_for_mine_initialization=self.use_historical_price_for_mine_initialization)
            if self.bayesian_tune or (
                    hasattr(self, 'historical_data') and 'Primary commodity price' in self.historical_data.columns):
                self.mod.primary_commodity_price = self.historical_data['Primary commodity price'].dropna()
                self.mod.primary_commodity_price = pd.concat([pd.Series(self.mod.primary_commodity_price.iloc[0],
                                                                        np.arange(1900,
                                                                                  self.mod.primary_commodity_price.dropna().index[
                                                                                      0])),
                                                              self.mod.primary_commodity_price]).sort_index()
            if hasattr(self, 'historical_data'):
                self.mod.historical_data = self.historical_data.copy()


            for base in self.changing_base_parameters_series.index:
                if base in self.demand_params:
                    self.mod.hyperparam.loc[base,'Value'] = abs(self.changing_base_parameters_series[base])*np.sign(self.mod.hyperparam.loc[base,'Value'])
                else:
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
                self.mod = Integration(static_data_folder=self.static_data_folder, user_data_folder=self.user_data_folder,
                                       simulation_time=self.simulation_time,
                                       verbosity=self.verbosity,byproduct=self.byproduct,commodity=self.material,
                                       price_to_use=self.price_to_use,
                                       historical_price_rolling_window=self.historical_price_rolling_window,
                                       force_integration_historical_price=self.force_integration_historical_price,
                                       use_historical_price_for_mine_initialization=self.use_historical_price_for_mine_initialization)
                if self.bayesian_tune or (
                        hasattr(self, 'historical_data') and 'Primary commodity price' in self.historical_data.columns):
                    self.mod.primary_commodity_price = self.historical_data['Primary commodity price'].dropna()
                    self.mod.primary_commodity_price = pd.concat([pd.Series(self.mod.primary_commodity_price.iloc[0],
                                                                            np.arange(1900,
                                                                                      self.mod.primary_commodity_price.dropna().index[
                                                                                          0])),
                                                                  self.mod.primary_commodity_price]).sort_index()
                if hasattr(self, 'historical_data'):
                    self.mod.historical_data = self.historical_data.copy()

                self.hyperparam_copy = self.mod.hyperparam.copy()

                ###### CHANGING BASE PARAMETERS ######
                for base in changing_base_parameters_series.index:
                    self.mod.hyperparam.loc[base,'Value'] = changing_base_parameters_series[base]
                    if count==0 and self.verbosity>0:
                        print(base,changing_base_parameters_series[base])

                ###### UPDATING FROM params_to_change_ind ######
                self.mod.hyperparam.loc[i,'Value'] = val
                print(f'Scenario {count}/{total_num_scenarios}: {i} = {val}')
                count += 1

                self.check_run_append()

    def run_monte_carlo(self, n_scenarios, random_state=220530,sensitivity_parameters=['elas','response','growth','improvements','refinery_capacity_fraction_increase_mining','incentive_opening_probability'],bayesian_tune=False,n_params=1, n_jobs=3, surrogate_model='GBRT'):

        '''
        Runs a Monte Carlo based approach to the sensitivity, where all the sensitivity_parameters
        given are then given values randomly selected from between zero and one.

        Always runs an initial scenario with default parameters, so remember to skip that one
        when looking at the resulting pickle file
        ----------------------
        n_scenarios: int, number of scenarios to run in total
        random_state: int, hold constant to keep same set of generated values
          each time this function is run
        sensitivity_parameters: list, includes ['elas','response','growth','improvements']
          by default to capture the different elasticity values, mine improvements, and
          demand responses to price. The parameters you are targeting must already be in
          the main mod.hyperparam dataframe, which must be updated within that function;
          cannot be accessed here. collection_rate_price_response, direct_melt_price_response,
          and secondary_refined_price_response are excluded from the parameters to change
          regardless of the inputs here, since they are True/False values and are determined
          by the scenario input given when the Sensitivity class is initialized.
        bayesian_tune: bool, whether to use the Bayesian optimization over historical data
          to try to select the next sentivitity parameter values
        n_params: int, number of columns from self.historical_data to use in the objective
          function for tuning. >1 means multi-objective optimization, and depends on the
          order of the columns, since it grabs the leftmost n_params columns.
        n_jobs: int, the number of points to sample at each Bayesian iteration, also
            the number of cores to use to parallelise Integration() calculation.
        surrogate_model: str, which type of surrogate model to use in BO, can be ET, GBRT,
            GP, RF, or DUMMY.
        '''
        self.random_state = random_state
        self.bayesian_tune = bayesian_tune
        self.n_jobs = n_jobs
        given_hyperparam_df = type(sensitivity_parameters)==pd.core.frame.DataFrame
        self.given_hyperparam_df = given_hyperparam_df
        self.update_changing_base_parameters_series()
        self.update_pkl_filename_path()
        if not given_hyperparam_df:
            self.initialize_big_df()
        else:
            self.big_df=pd.DataFrame()
            # self.big_df.to_pickle(self.pkl_filename)
        if np.all(['++' not in q for q in self.scenarios]):
            self.mod = Integration(static_data_folder=self.static_data_folder, user_data_folder=self.user_data_folder,
                                    simulation_time=self.simulation_time,
                                   verbosity=self.verbosity,byproduct=self.byproduct,commodity=self.material,
                                   price_to_use=self.price_to_use,
                                   historical_price_rolling_window=self.historical_price_rolling_window,
                                   force_integration_historical_price=self.force_integration_historical_price,
                                   use_historical_price_for_mine_initialization=self.use_historical_price_for_mine_initialization)
            if self.bayesian_tune or (
                    hasattr(self, 'historical_data') and 'Primary commodity price' in self.historical_data.columns):
                self.mod.primary_commodity_price = self.historical_data['Primary commodity price'].dropna()
                self.mod.primary_commodity_price = pd.concat([pd.Series(self.mod.primary_commodity_price.iloc[0],
                                                                        np.arange(1900,
                                                                                  self.mod.primary_commodity_price.dropna().index[
                                                                                      0])),
                                                              self.mod.primary_commodity_price]).sort_index()
            if hasattr(self, 'historical_data'):
                self.mod.historical_data = self.historical_data.copy()

        if not given_hyperparam_df:
            scenario_params_dont_change = ['collection_rate_price_response','direct_melt_price_response','secondary_refined_price_response','refinery_capacity_growth_lag','region_specific_price_response']
            if self.changing_base_parameters_series.loc['Secondary refinery fraction of recycled content, Global']==0:
                scenario_params_dont_change += ['sec CU price elas','sec CU TCRC elas','sec ratio TCRC elas','sec ratio scrap spread elas']
            if self.changing_base_parameters_series.loc['Secondary refinery fraction of recycled content, Global']==1:
                scenario_params_dont_change += ['direct_melt_elas_scrap_spread']
    #         params_to_change = [i for i in self.mod.hyperparam.dropna(how='all').index if ('elas' in i or 'response' in i or 'growth' in i or 'improvements' in i) and i not in scenario_params_dont_change]
            params_to_change = [i for i in self.mod.hyperparam.dropna(how='all').index if np.any([j == i for j in sensitivity_parameters]) and i not in scenario_params_dont_change]
            self.sensitivity_param = params_to_change
            if self.verbosity>-1:
                print(params_to_change)
        else:
            n_scenarios = sensitivity_parameters.shape[1]+1
            # ^ when we're running scenarios rather than tuning, we give a dataframe of hyperparameters to use
            # this dataframe is saved in the sensitivity_parameters variable

        if bayesian_tune:
            self.setup_bayesian_tune(n_params=n_params, surrogate_model=surrogate_model)
        if given_hyperparam_df:
            self.rmse_df = pd.DataFrame()

        notes_ph = self.notes
        for n in np.arange(1,n_scenarios):
            if self.timer is not None: self.timer.start_iter()

            self.scenario_number = n-1
            if self.verbosity>-1:
                print(f'Scenario {n+1}/{n_scenarios}')

            if bayesian_tune:
                next_parameters = self.opt.ask(n_points=self.n_jobs)
                n_jobs = self.n_jobs
            if bayesian_tune or (given_hyperparam_df and n == 1):
                mods = []
                new_param_series_all = []
                if given_hyperparam_df:
                    next_parameters=0
                    n_jobs = 1
            elif not bayesian_tune and not given_hyperparam_df:
                self.n_jobs = 1

            if len(self.scenarios)>1:
                if '++' in self.scenarios[1]:
                    self.scenario_frame = get_scenario_dataframe(
                        file_path_for_scenario_setup=self.scenarios[1].split('++')[0],
                        default_year=2019)
            for i in range(n_jobs):
                for enum,scenario_name in enumerate(self.scenarios):
                    if len(self.scenarios)>1:
                        if '++' in scenario_name:
                            original_scenario_name = scenario_name
                            self.scenario_frame = get_scenario_dataframe(
                                file_path_for_scenario_setup=scenario_name.split('++')[0], default_year=2019)
                            pass_scenario_name = self.scenario_frame.loc[scenario_name.split('++')[1]]
                        else:
                            pass_scenario_name = scenario_name
                    else:
                        pass_scenario_name = scenario_name

                    self.last_scenario_flag = (n==n_scenarios-1) and (i==self.n_jobs-1) and (scenario_name==self.scenarios[-1])
                    if self.verbosity>-1:
                        print(f'\tSub-scenario {enum+1}/{len(self.scenarios)}: {scenario_name} checking if exists...')
                    self.mod = Integration(static_data_folder=self.static_data_folder, user_data_folder=self.user_data_folder,
                                           simulation_time=self.simulation_time,
                                           verbosity=self.verbosity,byproduct=self.byproduct,
                                           scenario_name=pass_scenario_name,commodity=self.material,
                                           price_to_use=self.price_to_use,
                                           historical_price_rolling_window=self.historical_price_rolling_window,
                                           force_integration_historical_price=self.force_integration_historical_price,
                                           use_historical_price_for_mine_initialization=self.use_historical_price_for_mine_initialization)
                    self.mod.hyperparam.loc['scenario_name'] = scenario_name

                    if self.bayesian_tune or (
                            hasattr(self,
                                    'historical_data') and 'Primary commodity price' in self.historical_data.columns):
                        self.mod.primary_commodity_price = self.historical_data['Primary commodity price'].dropna()
                        self.mod.primary_commodity_price = pd.concat(
                            [pd.Series(self.mod.primary_commodity_price.iloc[0],
                                       np.arange(1900,
                                                 self.mod.primary_commodity_price.dropna().index[
                                                     0])),
                             self.mod.primary_commodity_price]).sort_index()
                    if hasattr(self, 'historical_data'):
                        self.mod.historical_data = self.historical_data.copy()

                    self.notes = notes_ph+' '+scenario_name

                    ###### CHANGING BASE PARAMETERS ######
                    changing_base_parameters_series = self.changing_base_parameters_series.copy()
                    if self.verbosity>0: print('parameters getting updated from outside using changing_base_parameters_series input:\n',changing_base_parameters_series.index)
                    for base in changing_base_parameters_series.index:
                        if base in self.demand_params:
                            self.mod.hyperparam.loc[base,'Value'] = abs(self.changing_base_parameters_series[base])*np.sign(self.mod.hyperparam.loc[base,'Value'])
                        else:
                            self.mod.hyperparam.loc[base,'Value'] = self.changing_base_parameters_series[base]
                        if n==0 and self.verbosity>0:
                            print(base,changing_base_parameters_series[base])
                    self.hyperparam_copy = self.mod.hyperparam.copy()

                    ###### UPDATING MONTE CARLO PARAMETERS ######
                    if not given_hyperparam_df:
                        if len(params_to_change)>1:
                            if bayesian_tune:
                                new_param_series = pd.Series(next_parameters[i], params_to_change)
                            else:
                                rs = random_state+n
                                values = stats.uniform.rvs(loc=0,scale=1,size=len(params_to_change),random_state=rs)
                                new_param_series = pd.Series(values, params_to_change)

                            always_unconstrain = ['mine_cost_change_per_year','incentive_mine_cost_change_per_year',
                                                  'sector_specific_dematerialization_tech_growth',
                                                  'intensity_response_to_gdp']

                            if 'sector_specific_price_response' in params_to_change:
                                new_param_series.loc['region_specific_price_response'] = new_param_series['sector_specific_price_response']
                            elif 'region_specific_price_response' in params_to_change:
                                new_param_series.loc['sector_specific_price_response'] = new_param_series['region_specific_price_response']
                            if self.verbosity>0:
                                print('Parameters getting changed via Monte Carlo random selection:\n',params_to_change)
                            if 'incentive_opening_probability' in params_to_change and self.check_for_previously_tuned('incentive_opening_probability'):
                                new_param_series.loc['incentive_opening_probability']*=0.5/(1-self.incentive_opening_probability_fraction_zero)
                                if new_param_series['incentive_opening_probability']>0.5 and self.incentive_opening_probability_fraction_zero!=0:
                                    new_param_series.loc['incentive_opening_probability'] = 0
                            # ^ these values should be small, since small changes make big changes
                            all_three_here = np.all([q in params_to_change for q in ['close_probability_split_max','close_probability_split_mean','close_probability_split_min']])
                            if 'close_probability_split_max' in params_to_change and not all_three_here and self.check_for_previously_tuned('close_probability_split_max'):
                                new_param_series.loc['close_probability_split_max'] *= 0.8
                            if 'close_probability_split_mean' in params_to_change and not all_three_here and self.check_for_previously_tuned('close_probability_split_mean'):
                                new_param_series.loc['close_probability_split_mean'] *= 0.8
                            if 'close_probability_split_max' in params_to_change and 'close_probability_split_mean' in params_to_change and not all_three_here and self.check_for_previously_tuned('close_probability_split_max') and self.check_for_previously_tuned('close_probability_split_mean'):
                                sum_mean_max = new_param_series.loc[['close_probability_split_max','close_probability_split_mean']].sum()
                                if sum_mean_max>0.95:
                                    new_param_series.loc[['close_probability_split_max','close_probability_split_mean']] *= 0.95/sum_mean_max
                                new_param_series.loc['close_probability_split_mean'] = 1-new_param_series.loc[['close_probability_split_max','close_probability_split_mean']].sum()
                            if 'close_years_back' in params_to_change and self.check_for_previously_tuned('close_years_back'):
                                new_param_series.loc['close_years_back'] = int(7*new_param_series.loc['close_years_back']+3)
                            if 'reserves_ratio_price_lag' in params_to_change and self.check_for_previously_tuned('reserves_ratio_price_lag'):
                                new_param_series.loc['reserves_ratio_price_lag'] = int(7*new_param_series['reserves_ratio_price_lag']+3)
                            if all_three_here and np.all([self.check_for_previously_tuned(j) for j in ['close_probability_split_mean','close_probability_split_min','close_probability_split_max']]):
                                new_param_series.loc[['close_probability_split_mean','close_probability_split_min','close_probability_split_max']] /=  new_param_series.loc[['close_probability_split_mean','close_probability_split_min','close_probability_split_max']].sum()
                            if 'mine_cost_change_per_year' in params_to_change and self.check_for_previously_tuned('mine_cost_change_per_year'):
                                new_param_series.loc['mine_cost_change_per_year'] = new_param_series['mine_cost_change_per_year']*10-5
                                if 'incentive_mine_cost_change_per_year' not in params_to_change:
                                    new_param_series.loc['incentive_mine_cost_change_per_year'] = new_param_series['mine_cost_change_per_year']
                            if 'primary_overhead_const' in params_to_change and self.check_for_previously_tuned('primary_overhead_const'):
                                new_param_series.loc['primary_overhead_const'] = (new_param_series['primary_overhead_const']-0.5)*1
                            max_dict = dict(zip(['sector_specific_dematerialization_tech_growth','sector_specific_price_response','region_specific_price_response','intensity_response_to_gdp'],[0.2,0.6,0.6,1.5]))
                            if 'sector_specific_price_response' in new_param_series.index and self.check_for_previously_tuned('sector_specific_price_response'):
                                new_param_series.loc['sector_specific_price_response'] *= max_dict['sector_specific_price_response']
                            if 'sector_specific_dematerialization_tech_growth' in params_to_change and self.check_for_previously_tuned('sector_specific_dematerialization_tech_growth'):
                                new_param_series.loc['sector_specific_dematerialization_tech_growth'] *= max_dict['sector_specific_dematerialization_tech_growth']
                                new_param_series.loc['sector_specific_dematerialization_tech_growth'] -= max_dict['sector_specific_dematerialization_tech_growth']/2
                            if 'region_specific_price_response' in new_param_series.index and self.check_for_previously_tuned('region_specific_price_response'):
                                new_param_series.loc['region_specific_price_response'] *= max_dict['region_specific_price_response']
                            if 'intensity_response_to_gdp' in params_to_change and self.check_for_previously_tuned('intensity_response_to_gdp'):
                                new_param_series.loc['intensity_response_to_gdp'] *= max_dict['intensity_response_to_gdp']
                                new_param_series.loc['intensity_response_to_gdp'] -= max_dict['intensity_response_to_gdp']/3
                            if 'primary_oge_scale' in params_to_change and 'initial_ore_grade_decline' not in params_to_change:
                                new_param_series.loc['initial_ore_grade_decline'] = new_param_series['primary_oge_scale']
                            self.sensitivity_param = params_to_change
                            int_params = ['reserves_ratio_price_lag','close_years_back']
                            cannot_unconstrain = ['primary_oge_scale', 'incentive_opening_probability',
                                                  'close_years_back', 'reserves_ratio_price_lag',
                                                  'refinery_capacity_fraction_increase_mining']

                            for j in int_params:
                                if j in new_param_series.index: new_param_series.loc[j] = int(new_param_series[j])
                            for param in new_param_series.index:
                                if type(self.mod.hyperparam['Value'][param])!=bool and self.mod.hyperparam['Value'][param]!=np.nan and (self.constrain_tuning_to_sign or param in cannot_unconstrain) and param not in always_unconstrain:
                                    new_param_series.loc[param] = abs(new_param_series[param])*np.sign(self.mod.hyperparam.loc[param,'Value'])
                                    self.mod.hyperparam.loc[param,'Value'] = abs(new_param_series[param])*np.sign(self.mod.hyperparam.loc[param,'Value'])
                                else:
                                    self.mod.hyperparam.loc[param,'Value'] = new_param_series[param]


                    if bayesian_tune:
                        mods.append(self.mod)
                        new_param_series_all.append(new_param_series)
                    elif given_hyperparam_df:
                        ind = (n-1)
                        if ind in sensitivity_parameters.columns:
                            for base in sensitivity_parameters.index:
                                self.mod.hyperparam.loc[base,'Value'] = sensitivity_parameters[ind][base]
                            self.hyperparam_copy = self.mod.hyperparam.copy()
                            new_param_series = self.mod.hyperparam.copy()
                            mods.append(self.mod)
                            new_param_series_all.append(new_param_series)
                    else:
                        self.check_run_append()

            # If tuning, want to run complete_bayesian_trial now so that we can update our optimization model
            # with the runs from that scenario round.
            if bayesian_tune:
                self.complete_bayesian_trial(mods=mods,
                                             new_param_series_all=new_param_series_all,
                                             scenario_numbers=range(n_jobs*(n-1)+1, n_jobs*(n)+1),
                                             next_parameters=next_parameters,
                                             bayesian_tune=bayesian_tune,
                                             n_params=n_params)
                if self.timer is not None: self.timer.end_iter()

        # If we're not tuning, we want to run the complete_bayesian_trial one level out, so that all the
        # models have been properly initialized prior to running the parallel part.
        if given_hyperparam_df:
            n_iterations = int(np.ceil(len(mods) / self.n_jobs))+1
            for k in np.arange(0,n_iterations):
                from_here = k*self.n_jobs
                to_here = (k+1)*self.n_jobs
                if to_here > len(mods):
                    to_here = len(mods)
                if from_here <= len(mods):
                    if type(next_parameters) == int:
                        next_params = next_parameters
                    else:
                        next_params = next_parameters[from_here:to_here]
                    self.complete_bayesian_trial(mods=mods[from_here:to_here],
                                                 new_param_series_all=new_param_series_all[from_here:to_here],
                                                 scenario_numbers=np.arange(from_here,to_here),
                                                 next_parameters=next_params,
                                                 bayesian_tune=bayesian_tune,
                                                 n_params=n_params)
            if self.timer is not None: self.timer.end_iter()

        if bayesian_tune:
            self.save_bayesian_results(n_params=n_params)

    def check_for_previously_tuned(self,param):
        """
        returns True if we are not constraining to previously tuned, or if the
        parameter is not in updated_commodity_inputs.pkl. Meaning that if we are
        constraining previously tuned and the value is in the pickle file, we
        will not do the scaling used for parameters generated from the Bayesian
        optimization, and will use the value from the pickle instead.
        """
        bool1 = not self.constrain_previously_tuned
        bool2 = param not in self.updated_commodity_inputs_sub.dropna().index
        return bool1 or bool2

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

    def setup_bayesian_tune(self, n_params=1, surrogate_model='GBRT'):
        '''
        Initializes the things needed to run the Bayesian optimization
        '''

        if n_params!=2:
            self.objective_parameters = self.historical_data_column_list[:n_params]
        else:
            self.objective_parameters = self.historical_data_column_list[:3]
        variance_from_previous = 1
        lowerbound = 1-variance_from_previous
        lowerbound = 0.001 if lowerbound<=0 else lowerbound
        cannot_unconstrain = ['primary_oge_scale','incentive_opening_probability',
                      'close_years_back','reserves_ratio_price_lag','refinery_capacity_fraction_increase_mining']
        default_bounds = dict([(_,(0.001,1)) if self.constrain_tuning_to_sign
                                or _ in cannot_unconstrain else (_,(0.001,2))
                                for _ in self.sensitivity_param])
        self.opt = Optimizer(
            dimensions=[default_bounds[_]
                        if (not self.constrain_previously_tuned or _ not in self.updated_commodity_inputs_sub.dropna().index)
                        else (abs(self.updated_commodity_inputs_sub[_])*(lowerbound),
                              abs(self.updated_commodity_inputs_sub[_])*(1+variance_from_previous))
                        for _ in self.sensitivity_param],
            base_estimator=surrogate_model,
            n_initial_points=20,
            initial_point_generator='random' if surrogate_model=='dummy' else 'lhs',
            n_jobs=self.n_jobs,
            acq_func='gp_hedge',
            acq_optimizer='auto',
            acq_func_kwargs={"kappa": 1.96},
            random_state=self.random_state
        )

        self.rmse_df = pd.DataFrame()

    def complete_bayesian_trial(self, mods, new_param_series_all, next_parameters, scenario_numbers, bayesian_tune=True, n_params=3):
        '''
        calculates root mean squared errors (RMSE) from current variables and historical
        values to give the error the Bayesian optimization is trying to minimize.
        '''
        #output is of the form [(score_0, new_params_0, potential_append_0), (score_1, new_params_1, potential_append_0), ...]
        output = Parallel(n_jobs=self.n_jobs)(delayed(self.skopt_run_score)(
            mod, param_series, s_n, bayesian_tune, n_params
        ) for mod, param_series, s_n in zip(mods, new_param_series_all, scenario_numbers))

        #give scores to skopt
        if bayesian_tune:
            self.opt.tell(next_parameters, [out[0] for out in output])

        #save new_param_series in rmse_df
        for new_param_series in [out[1] for out in output]:
            self.rmse_df = pd.concat([self.rmse_df, new_param_series])

        #save scenario results in pickle file
        # big_df = pd.read_pickle(self.pkl_filename)
        if self.big_df.shape[1]>666 and not self.bayesian_tune:
            big_df = pd.DataFrame()
            self.n_files += 1
        else:
            big_df = self.big_df.copy()
        for potential_append in [out[2] for out in output]:
            if potential_append is None:
                raise Exception("Scenario has already been run, this case has not been implemented yet. Ask Luca Montanelli why.")
            potential_append.loc['rmse_df'] = [self.rmse_df]
            big_df = pd.concat([big_df,potential_append],axis=1)
        if not self.bayesian_tune:
            temp_filename = self.pkl_filename.split('.')
            temp_filename = ''.join([temp_filename[0],str(self.n_files)+'.',temp_filename[1]])
        else:
            temp_filename = self.pkl_filename
        big_df.to_pickle(temp_filename)
        self.big_df = big_df.copy()
        if self.verbosity>-1:
            print('\tScenario successfully saved\n')

    def skopt_run_score(self, mod, new_param_series, s_n, bayesian_tune=True, n_params=3):

        #run model
        potential_append = self.check_run_append(mod, s_n)
        self.potential_append = potential_append.copy()
        if bayesian_tune:
            #get scores
            if type(mod)==Integration and self.demand_or_mining!='mining':
                test_time = [i for i in self.simulation_time if i not in self.train_time]
                # loop over three RMSE calculations using different evaluation time periods (full simulation time, test set, train set), with train set last so it is used in the score calculation/tuning
                for time,time_name in zip([self.simulation_time, test_time, self.train_time],['','test','train']):
                    if len(time)>1:
                        rmse_list, r2_list = self.get_rmse_r2(mod, time=time)

                        #add rmse and r2 score in new_param_series to then be added into rmse_df
                        for param, rmse, r2 in zip(self.objective_parameters, rmse_list, r2_list):
                            new_param_series.loc[f'{param} {time_name} RMSE'] = rmse[0]
                            new_param_series.loc[f'{param} {time_name} R2'] = r2[0]
                # removing price if n_params==2
                rmse_list = [j for z,j in zip(self.objective_parameters, rmse_list) if (n_params!=2 or 'price' not in z.lower())]
                r2_list   = [j for z,j in zip(self.objective_parameters, r2_list)   if (n_params!=2 or 'price' not in z.lower())]
                if self.verbosity>3:
                    print(969,'rmse_list:',rmse_list)

            else:
                rmse_list, r2_list = self.get_rmse_r2_mining(mod)

                #add rmse and r2 score in new_param_series to then be added into rmse_df
                for rmse, r2 in zip(rmse_list, r2_list):
                    new_param_series.loc['RMSE'] = rmse[0]
                    new_param_series.loc['R2'] = r2[0]

            #if we optimise on rmse, return that, else return r2
            if self.use_rmse_not_r2:
                if self.log:
                    score = np.log(sum(r[0]/n_params for r in rmse_list))
                else:
                    score = sum(r[0]/n_params for r in rmse_list)
            else: #normalize_objectives
                if self.log:
                    #r2 is [-inf, 1] so we change it to [1, inf] to use in log, it then becomes [0, inf]
                    #+1 intead of +2 would have made it into [-inf, inf]; not good
                    score = np.log(sum(-r[0]/n_params+2 for r in r2_list))
                else:
                    #we flip the sign because skopt only minimises
                    score = sum(-r[0]/n_params for r in r2_list)

            new_param_series.loc['score'] = score
        else:
            score=0
        new_param_series = pd.concat([new_param_series],keys=[s_n])
        return score, new_param_series, potential_append

    def calculate_rmse_r2(self, sim, hist, use_rmse):
        n = len(self.simulation_time)
        try:
            x, y = sim.astype(float), hist.astype(float)
        except KeyError as e:
            print('train_time input variable must fit entirely within simulation_time variable')
            raise e
        if hasattr(x,'columns') and 'Global' in x.columns: x=x['Global']
        if hasattr(y,'columns') and 'Global' in y.columns: y=y['Global']
        # m = sm.GLS(x,sm.add_constant(y)).fit(cov_type='HC3')
        if use_rmse:
            # result = m.mse_resid**0.5
            result = mean_squared_error(y, x)**0.5
        else:
            result = r2_score(y, x)
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

    def get_rmse_r2(self, mod=None, time=np.arange(2001,2020)):
        #if no mod is provided, use self.mod, otherwise use mod
        #this is because default model behaviour was to use self.mod but that doesn't work with
        #having multiple samples per BO iteration
        if mod is None:
            mod = self.mod

        param_variable_map = {'Total demand':'total_demand','Primary demand':'primary_demand',
            'Primary commodity price':'primary_commodity_price','Primary supply':'mine_production',
            'Primary production':'mine_production','Scrap demand':'scrap_demand'}
        rmse_list = []
        r2_list = []
        for param in self.objective_parameters:
            if 'SD' not in param:
                try:
                    historical = self.historical_data[param].loc[time]
                    simulated = getattr(mod,param_variable_map[param]).loc[time]
                    if hasattr(simulated,'columns') and 'Global' in simulated.columns:
                        simulated = simulated['Global']
                    rmse = self.calculate_rmse_r2(simulated,historical,True)
                    if self.normalize_objectives:
                        rmse/=self.historical_data[param].loc[self.simulation_time[0]]
                    r2 = self.calculate_rmse_r2(simulated,historical,False)
                except KeyError as e:
                    if self.verbosity>0:
                        print('train_time input variable must fit entirely within simulation_time variable for R2 and RMSE to be correctly calculated. Not an issue if you do not care about those.')
                    elif self.verbosity>1:
                        print('Causes exception:',e)
                    rmse = np.nan
                    r2 = np.nan
            else:
                if 'Conc' in param:
                    rmse = self.calculate_rmse_r2(mod.concentrate_supply,mod.concentrate_demand,True)
                    if self.normalize_objectives:
                        rmse /= mod.concentrate_demand.loc[self.simulation_time[0]] if not hasattr(mod.concentrate_demand,'columns') else mod.concentrate_demand['Global'].iloc[0]
                    r2 = self.calculate_rmse_r2(mod.primary_supply,mod.primary_demand,False)
                elif 'Scrap' in param:
                    rmse = self.calculate_rmse_r2(mod.scrap_supply,mod.scrap_demand,True)
                    if self.normalize_objectives:
                        rmse /= mod.scrap_demand.loc[self.simulation_time[0]] if not hasattr(mod.scrap_demand,'columns') else mod.scrap_demand['Global'].iloc[0]
                    r2 = self.calculate_rmse_r2(mod.scrap_supply,mod.scrap_demand,False)
                elif 'Ref' in param:
                    rmse = self.calculate_rmse_r2(mod.refined_supply,mod.refined_demand,True)
                    if self.normalize_objectives:
                        rmse /= mod.refined_demand.loc[self.simulation_time[0]] if not hasattr(mod.refined_demand,'columns') else mod.refined_demand['Global'].iloc[0]
                    r2= self.calculate_rmse_r2(mod.refined_supply,mod.refined_demand,False)

            rmse_list += [(rmse,0)]
            r2_list+= [(r2,0)]

        return rmse_list, r2_list

    def get_rmse_r2_mining(self, mod=None):
        #if no mod is provided, use self.mod, otherwise use mod
        #this is because default model behaviour was to use self.mod but that doesn't work with
        #having multiple samples per BO iteration
        if mod is None:
            mod = self.mod

        rmse_list = []
        r2_list = []

        if type(mod)==Integration:
            sim = mod.mine_production.loc[self.simulation_time]
        else:
            sim = mod.supply_series.loc[self.simulation_time]
        hist = self.historical_data['Primary supply' if 'Primary supply' in self.historical_data.columns else 'Primary production'].loc[self.simulation_time]
        rmse = mean_squared_error(hist, sim)**0.5
        r2 = r2_score(hist, sim)

        rmse_list += [(rmse,0)]
        r2_list+= [(r2,0)]

        return rmse_list, r2_list

    def historical_sim_check_demand(self, n_scenarios, surrogate_model='ET', log=True, demand_or_mining='demand'):
        '''
        Varies the parameters for demand (sector_specific_dematerialization_tech_growth,
        sector_specific_price_response, region_specific_price_response, and
        intensity_response_to_gdp) to minimize the RMSE between simulated
        demand and historical.

        Uses skopt Optimize to optimize the parameters

        Adds/updates a dataframe self.updated_commodity_inputs that contains all demand
        updates for all commodities, which is saved in updated_commodity_inputs.pkl
        '''
        self.bayesian_tune = True
        self.demand_or_mining = demand_or_mining
        self.log = log
        self.update_pkl_filename_path()

        if demand_or_mining=='demand':
            self.pkl_filename = self.pkl_filename.split('.pkl')[0]+'_DEM.pkl'
            self.notes += ' check demand'
        elif demand_or_mining=='mining':
            self.pkl_filename = self.pkl_filename.split('.pkl')[0]+'_mining.pkl'
            self.notes += ' check mining'
        else:
            raise ValueError('the demand_or_mining input must be either demand or mining, no other inputs are supported')
        if os.path.exists(self.pkl_filename) and not self.overwrite:
            big_df = pd.read_pickle(self.pkl_filename)
        else:
            big_df = pd.DataFrame([],['version','notes','hyperparam','results'],[])
            big_df.to_pickle(self.pkl_filename)
        if demand_or_mining=='demand' and 'Primary commodity price' not in self.historical_data.columns:
            raise ValueError('require a price input in primary commodity price for historical_sim_check_demand to work properly')
        elif demand_or_mining=='mining' and 'Primary commodity price' not in self.historical_data.columns and 'Primary production' not in self.historial_data.columns and 'Primary supply' not in self.historial_data.columns:
            raise ValueError('require a price input in primary commodity price for historical_sim_check_demand to work properly, also require primary supply')
        self.update_changing_base_parameters_series()

        cannot_unconstrain = ['primary_oge_scale','incentive_opening_probability',
                              'close_years_back','reserves_ratio_price_lag']
        if 'constrain' in self.pkl_filename:
            if self.pkl_filename.split('constrain')[1][0]=='1':
                cannot_unconstrain += ['intensity_response_to_gdp','mine_cost_change_per_year']
                print('1126, added intensity_response_to_gdp and mine_cost_change_per_year to cannot_unconstrain')

        if demand_or_mining=='demand':
            self.mod = demandModel(verbosity=self.verbosity, simulation_time=self.simulation_time,
                                   static_data_folder=self.static_data_folder,
                                   user_data_folder=self.user_data_folder)
            params_to_change = self.demand_params
            default_bounds = (0.001,1) if self.constrain_tuning_to_sign else (0.001,2)
            dimensions = [default_bounds if i not in cannot_unconstrain else (0.001,1) for i in params_to_change]
        else:
            # self.mod = miningModel(verbosity=self.verbosity, simulation_time=self.simulation_time,byproduct=self.byproduct)
            self.mod = Integration(static_data_folder=self.static_data_folder, user_data_folder=self.user_data_folder,
                                    simulation_time=self.simulation_time,
                                   verbosity=self.verbosity,byproduct=self.byproduct,commodity=self.material,
                                   price_to_use=self.price_to_use,
                                   historical_price_rolling_window=self.historical_price_rolling_window,
                                   force_integration_historical_price=self.force_integration_historical_price,
                                   use_historical_price_for_mine_initialization=self.use_historical_price_for_mine_initialization)
            if self.bayesian_tune or (
                    hasattr(self, 'historical_data') and 'Primary commodity price' in self.historical_data.columns):
                self.mod.primary_commodity_price = self.historical_data['Primary commodity price'].dropna()
                self.mod.primary_commodity_price = pd.concat([pd.Series(self.mod.primary_commodity_price.iloc[0],
                                                                        np.arange(1900,
                                                                                  self.mod.primary_commodity_price.dropna().index[
                                                                                      0])),
                                                              self.mod.primary_commodity_price]).sort_index()
            if hasattr(self, 'historical_data'):
                self.mod.historical_data = self.historical_data.copy()

            params_to_change = ['primary_oge_scale','mine_cu_margin_elas','mine_cost_og_elas','mine_cost_change_per_year','mine_cost_price_elas','initial_ore_grade_decline','primary_price_resources_contained_elas','incentive_opening_probability','close_years_back','reserves_ratio_price_lag']
            checker = np.intersect1d(params_to_change,self.additional_base_parameters.index)
            if len(checker)>0 and self.verbosity>-1:
                print(f'1152, checking and removing: {checker}')
            params_to_change = [k for k in params_to_change if k not in self.additional_base_parameters.index]
            default_bounds = (0.001,1) if self.constrain_tuning_to_sign else (0.001,2)
            dimensions = [default_bounds if i not in cannot_unconstrain else (0.001,1) for i in params_to_change]
        mod = self.mod
        self.hyperparam_changing = params_to_change
        if self.verbosity>0: print(params_to_change)

        n_jobs = 1 if demand_or_mining=='demand' else 3

        opt = Optimizer(
            dimensions=dimensions,
            base_estimator=surrogate_model,
            n_initial_points=15,
            initial_point_generator='lhs',
            n_jobs=n_jobs,
            acq_func='gp_hedge',
            acq_optimizer='auto',
            acq_func_kwargs={"kappa": 1.96},
            random_state=self.random_state
        )

        self.rmse_df = pd.DataFrame()

        for n in np.arange(0,n_scenarios):
            new_parameters = opt.ask(n_points=n_jobs)
            mods = []
            new_param_series_all = []

            for j in range(n_jobs):
                if self.timer is not None: self.timer.start_iter()

                if self.verbosity>-1:
                    print(f'Scenario {n+1}/{n_scenarios}')
                if demand_or_mining=='demand':
                    mod = demandModel(verbosity=self.verbosity, simulation_time=self.simulation_time,
                                      static_data_folder=self.static_data_folder,
                                      user_data_folder=self.user_data_folder)
                    mod.commodity_price_series = self.historical_data['Primary commodity price']
                    mod.commodity_price_series = pd.concat([pd.Series(mod.commodity_price_series.iloc[0],np.arange(1900,self.historical_data['Primary commodity price'].dropna().index[0])),
                                                            mod.commodity_price_series]).sort_index()
                else:
                    # mod = miningModel(verbosity=self.verbosity, simulation_time=self.simulation_time,byproduct=self.byproduct)
                    mod = Integration(static_data_folder=self.static_data_folder, user_data_folder=self.user_data_folder,
                                      simulation_time=self.simulation_time,
                                      verbosity=self.verbosity,byproduct=self.byproduct,commodity=self.material,
                                      price_to_use=self.price_to_use,
                                      historical_price_rolling_window=self.historical_price_rolling_window,
                                      force_integration_historical_price=self.force_integration_historical_price,
                                      use_historical_price_for_mine_initialization=self.use_historical_price_for_mine_initialization)
                    if self.bayesian_tune or (
                            hasattr(self,
                                    'historical_data') and 'Primary commodity price' in self.historical_data.columns):
                        self.mod.primary_commodity_price = self.historical_data['Primary commodity price'].dropna()
                        self.mod.primary_commodity_price = pd.concat(
                            [pd.Series(self.mod.primary_commodity_price.iloc[0],
                                       np.arange(1900,
                                                 self.mod.primary_commodity_price.dropna().index[
                                                     0])),
                             self.mod.primary_commodity_price]).sort_index()
                    if hasattr(self, 'historical_data'):
                        self.mod.historical_data = self.historical_data.copy()

                mod.version = '1.0'

                ###### CHANGING BASE PARAMETERS ######
                changing_base_parameters_series = self.changing_base_parameters_series.copy()
                for base in np.intersect1d(mod.hyperparam.index, changing_base_parameters_series.index):
                    if type(base) not in [str]:
                        mod.hyperparam.loc[base,'Value'] = abs(self.changing_base_parameters_series[base])*np.sign(mod.hyperparam.loc[base,'Value'])
                    else:
                        mod.hyperparam.loc[base,'Value'] = self.changing_base_parameters_series[base]

                    if n==0 and self.verbosity>-1:
                        print(base,changing_base_parameters_series[base])
                self.hyperparam_copy = mod.hyperparam.copy()

                ###### UPDATING PARAMETERS ######
                if n_jobs==1:
                    new_param_series = pd.Series(new_parameters[0],params_to_change)
                else:
                    new_param_series = pd.Series(new_parameters[j],params_to_change)
                if not self.constrain_tuning_to_sign:
                    for x in new_param_series.index:
                        if x not in cannot_unconstrain:
                            new_param_series.loc[x] -= 1
                max_dict = dict(zip(['sector_specific_dematerialization_tech_growth','sector_specific_price_response','region_specific_price_response','intensity_response_to_gdp'],[0.08,0.6,0.6,1.5]))
                if 'sector_specific_price_response' in params_to_change:
                    new_param_series.loc['region_specific_price_response'] = new_param_series['sector_specific_price_response']
                elif 'region_specific_price_response' in params_to_change:
                    new_param_series.loc['sector_specific_price_response'] = new_param_series['region_specific_price_response']
                if 'incentive_opening_probability' in params_to_change:
                    new_param_series.loc['incentive_opening_probability']*=0.5/(1-self.incentive_opening_probability_fraction_zero)
                    if new_param_series['incentive_opening_probability']>0.5 and self.incentive_opening_probability_fraction_zero!=0:
                        new_param_series.loc['incentive_opening_probability'] = 0
                if 'primary_oge_scale' in params_to_change:
                    new_param_series.loc['primary_oge_scale']*=0.5
                if 'initial_ore_grade_decline' in params_to_change:
                    new_param_series.loc['initial_ore_grade_decline']*=0.5
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
                    new_param_series.loc['close_years_back'] = int(7*new_param_series.loc['close_years_back']+3)
                if 'reserves_ratio_price_lag' in params_to_change:
                    new_param_series.loc['reserves_ratio_price_lag'] = int(7*new_param_series['reserves_ratio_price_lag']+3)
                if 'mine_cost_change_per_year' in params_to_change:
                    new_param_series.loc['mine_cost_change_per_year'] *= 5
                if 'primary_overhead_const' in params_to_change:
                    new_param_series.loc['primary_overhead_const'] = (new_param_series['primary_overhead_const']-0.5)*10
                if 'sector_specific_price_response' in new_param_series.index:
                    new_param_series.loc['sector_specific_price_response'] *= max_dict['sector_specific_price_response']
                if 'sector_specific_dematerialization_tech_growth' in params_to_change:
                    new_param_series.loc['sector_specific_dematerialization_tech_growth'] *= max_dict['sector_specific_dematerialization_tech_growth']
                if 'region_specific_price_response' in new_param_series.index:
                    new_param_series.loc['region_specific_price_response'] *= max_dict['region_specific_price_response']
                if 'intensity_response_to_gdp' in params_to_change:
                    new_param_series.loc['intensity_response_to_gdp'] *= max_dict['intensity_response_to_gdp']
                if all_three_here:
                    new_param_series.loc[['close_probability_split_mean','close_probability_split_min','close_probability_split_max']] /=  new_param_series.loc[['close_probability_split_mean','close_probability_split_min','close_probability_split_max']].sum()

                for param in new_param_series.index:
                    if mod.hyperparam.loc[param,'Value']!=0 and param!='primary_overhead_const' and (self.constrain_tuning_to_sign or param in cannot_unconstrain):
                        new_param_series.loc[param] = abs(new_param_series[param])*np.sign(self.mod.hyperparam.loc[param,'Value'])
                        mod.hyperparam.loc[param,'Value'] = abs(new_param_series.loc[param])*np.sign(mod.hyperparam.loc[param,'Value'])
                    else:
                        mod.hyperparam.loc[param,'Value'] = new_param_series[param]
                if n_jobs>1:
                    self.mod = mod
                    mods.append(mod)
                    new_param_series_all.append(new_param_series)
                    #output is of the form [(score_0, new_params_0, potential_append_0), (score_1, new_params_1, potential_append_0),
                else:
                    self.mod = mod
                    potential_append = self.check_run_append(self.mod,n+j)
                    mod = self.mod

                    if demand_or_mining=='demand':
                        sim = mod.demand.sum(axis=1).loc[self.simulation_time]
                        hist = self.historical_data['Total demand'].loc[self.simulation_time]
                    else:
                        # sim = mod.supply_series.loc[self.simulation_time]
                        sim = mod.mine_production.loc[self.simulation_time]
                        hist = self.historical_data['Primary supply' if 'Primary supply' in self.historical_data.columns else 'Primary production'].loc[self.simulation_time]
                    rmse = mean_squared_error(hist, sim)**0.5
                    r2 = r2_score(hist, sim)
                    # ((self.mod.demand.sum(axis=1)-self.historical_data['Total demand'])**2).loc[self.simulation_time].astype(float).sum()**0.5
                    new_param_series.loc['RMSE'] = rmse
                    new_param_series.loc['R2'] = r2
                    new_param_series = pd.concat([new_param_series],keys=[n])
                    self.rmse_df = pd.concat([self.rmse_df,new_param_series]).astype(float)

                    if self.use_rmse_not_r2:
                        if log: opt.tell(new_parameters[0], np.log(rmse))
                        else: opt.tell(new_parameters[0], rmse)
                    else:
                        #r2 is [-inf, 4] so we change it to [1, inf] to use in log to that it becomes [0, inf]
                        #+4 intead of +5 would have made it into [-inf, inf]; not good
                        if log: opt.tell(new_parameters[0], np.log(-r2+5))
                        #we flip the sign because skopt only minimises
                        else: opt.tell(new_parameters[0], -r2)

                    big_df = pd.read_pickle(self.pkl_filename)
                    if potential_append is None:
                        raise Exception("Scenario has already been run, this case has not been implemented yet. Ask Luca Montanelli why.")
                    big_df = pd.concat([big_df,potential_append],axis=1)

                    big_df.to_pickle(self.pkl_filename)

                    if self.timer is not None: self.timer.end_iter()

            if n_jobs>1:
                scenario_numbers=range(n_jobs*(n-1)+1, n_jobs*n+1)
                output = Parallel(n_jobs=n_jobs)(delayed(self.skopt_run_score)(mod, param_series, s_n) for mod, param_series, s_n in zip(mods, new_param_series_all, scenario_numbers))

                #give scores to skopt
                opt.tell(new_parameters, [out[0] for out in output])

                #save new_param_series in rmse_df
                for new_param_series in [out[1] for out in output]:
                    self.rmse_df = pd.concat([self.rmse_df, new_param_series])

                #save scenario results in pickle file
                big_df = pd.read_pickle(self.pkl_filename)
                for potential_append in [out[2] for out in output]:
                    if potential_append is None:
                        raise Exception("Scenario has already been run, this case has not been implemented yet. Ask Luca Montanelli why.")
                    big_df = pd.concat([big_df,potential_append],axis=1)

                big_df.to_pickle(self.pkl_filename)
                if self.verbosity>-1:
                    print('\tScenario successfully saved\n')
        rmse_df = self.rmse_df.copy()
        rmse_df.index = pd.MultiIndex.from_tuples(rmse_df.index)
        rmse_df = rmse_df[0]
        rmse_df = rmse_df.unstack()
        self.rmse_df = rmse_df.copy().astype(float)
        best_params = pd.DataFrame(rmse_df.loc[rmse_df.where(rmse_df!=0).dropna()['RMSE'].astype(float).idxmin()])
        if 'RMSE' in best_params.index:
            best_params.drop('RMSE',inplace=True)
        if 'R2' in best_params.index:
            best_params.drop('R2',inplace=True)
        best_params = best_params.rename(columns={best_params.columns[0]:self.material})
        updated_commodity_inputs = 'updated_commodity_inputs'
        if not self.constrain_tuning_to_sign:
            updated_commodity_inputs += '_unconstrained'
        if 'mcpe0' in self.pkl_filename or (hasattr(self.additional_base_parameters,'index') and 'mine_cost_price_elas' in self.additional_base_parameters.index):
            print(1351,updated_commodity_inputs)
            updated_commodity_inputs += '_mcpe0'
        if os.path.exists(f'data/{updated_commodity_inputs}.pkl'):
            self.updated_commodity_inputs = pd.read_pickle(f'data/{updated_commodity_inputs}.pkl')
            for param in best_params.index:
                self.updated_commodity_inputs.loc[param,self.material] = best_params[self.material][param]
            self.updated_commodity_inputs.to_pickle(f'data/{updated_commodity_inputs}.pkl')
        elif os.path.exists(f'{updated_commodity_inputs}.pkl'):
            self.updated_commodity_inputs = pd.read_pickle(f'{updated_commodity_inputs}.pkl')
            for param in best_params.index:
                self.updated_commodity_inputs.loc[param,self.material] = best_params[self.material][param]
            self.updated_commodity_inputs.to_pickle(f'{updated_commodity_inputs}.pkl')
        else:
            self.updated_commodity_inputs = best_params.copy()
            if os.path.exists('data'):
                self.updated_commodity_inputs.to_pickle(f'data/{updated_commodity_inputs}.pkl')
            else:
                self.updated_commodity_inputs.to_pickle(f'{updated_commodity_inputs}.pkl')
        for i in best_params.index:
            self.changing_base_parameters_series.loc[i] = best_params[self.material][i]
        self.notes = self.notes.split(' check demand')[0] if len(self.notes.split(' check demand'))==1 else ' '.join(self.notes.split(' check demand'))
        self.notes = self.notes.split(' check mining')[0] if len(self.notes.split(' check mining'))==1 else ' '.join(self.notes.split(' check mining'))
        if demand_or_mining=='demand':
            self.pkl_filename = self.pkl_filename.split('_DEM')[0]+'.pkl'
        else:
            self.pkl_filename = self.pkl_filename.split('_mining')[0]+'.pkl'

    def check_hist_demand_convergence(self):
        '''
        Run after historical_sim_check_demand() method to plot
        the results.
        '''
        historical_data = self.historical_data.copy()
        if self.demand_or_mining=='demand':
            if '_DEM' not in self.pkl_filename:
                big_df = pd.read_pickle(self.pkl_filename.split('.pkl')[0]+'_DEM.pkl')
            else:
                big_df = pd.read_pickle(self.pkl_filename)
        else:
            if '_mining' not in self.pkl_filename:
                big_df = pd.read_pickle(self.pkl_filename.split('.pkl')[0]+'_mining.pkl')
            else:
                big_df = pd.read_pickle(self.pkl_filename)
        ind = big_df.loc['results'].dropna().index
        res = pd.concat([big_df.loc['results',i] for i in ind],keys=ind)
        self.results = res.copy()
        if self.demand_or_mining=='demand':
            tot_demand = res['Total demand'].unstack(0).loc[self.simulation_time[0]:self.simulation_time[-1]]
            best_demand = tot_demand[(tot_demand.astype(float).apply(lambda x: x-historical_data['Total demand'])**2).sum().astype(float).idxmin()]
        else:
            tot_demand = res['Primary supply'].unstack(0).loc[self.simulation_time[0]:self.simulation_time[-1]]
            best_demand = tot_demand[(tot_demand.astype(float).apply(lambda x: x-historical_data['Primary production' if 'Primary production' in historical_data.columns else 'Primary supply'])**2).sum().astype(float).idxmin()]

        # below: calculates RMSE (root mean squared error) for each column and idxmin gets the index corresponding to the minimum RMSE
        fig,ax = easy_subplots(3,dpi=self.dpi)
        for i,a in enumerate(ax[:2]):
            if i==0:
                tot_demand.plot(linewidth=1,alpha=0.3,legend=False,ax=a)
            if self.demand_or_mining=='demand':
                historical_data['Total demand'].plot(ax=a,label='Historical',color='k',linewidth=4)
                a.set(title='Total demand over time',xlabel='Year',ylabel='Total demand (kt)')
            else:
                historical_data['Primary supply' if 'Primary supply' in historical_data.columns else 'Primary production'].plot(ax=a,label='Historical',color='k',linewidth=4)
                a.set(title='Mine production over time',xlabel='Year',ylabel='Mine production (kt)')
            best_demand.plot(ax=a,label='Best simulated',color='blue',linewidth=4)
            if i==1:
                a.legend()

        if self.demand_or_mining=='demand':
            do_a_regress(best_demand.astype(float),historical_data['Total demand'].astype(float).loc[self.simulation_time],ax=ax[2],xlabel='Simulated',ylabel='Historical')
        else:
            do_a_regress(best_demand.astype(float),historical_data['Primary production' if 'Primary production' in historical_data.columns else 'Primary supply'].astype(float).loc[self.simulation_time],ax=ax[2],xlabel='Simulated',ylabel='Historical')

        ax[-1].set(title='Historical regressed on simulated')
        if self.demand_or_mining=='demand':
            plt.suptitle('Total demand, varying demand parameters (sensitivity historical_sim_check_demand result)',fontweight='bold')
        else:
            plt.suptitle('Mine production, varying demand parameters (sensitivity historical_sim_check_demand result)',fontweight='bold')
        fig.tight_layout()

        hyps=self.hyperparam_changing
        if 'region_specific_price_response' not in hyps and self.demand_or_mining=='demand': hyps = list(hyps)+['region_specific_price_response']
        hyper = pd.concat([big_df.loc['hyperparam'].dropna().loc[i].loc[hyps,'Value'] for i in ind],keys=ind,axis=1)
        if self.demand_or_mining=='demand':
            fig,ax=easy_subplots(4,dpi=self.dpi)
            hyper.loc['RMSE'] = (tot_demand.astype(float).apply(lambda x: x-historical_data['Total demand'])**2).sum().astype(float)**0.5
        else:
            fig,ax=easy_subplots(self.hyperparam_changing,dpi=self.dpi)
            hyper.loc['RMSE'] = (tot_demand.astype(float).apply(lambda x: x-historical_data['Primary production' if 'Primary production' in historical_data.columns else 'Primary supply'])**2).sum().astype(float)**0.5

        for i,a in zip([i for i in hyper.index if i!='RMSE'],ax):
            a.scatter(hyper.loc[i],hyper.loc['RMSE'])
            a.set(title=i,ylabel='RMSE (kt)',xlabel='Elasticity value')
        plt.suptitle('Checking correlation betwen RMSE and parameter value (sensitivity historical_sim_check_demand_result)',fontweight='bold')
        fig.tight_layout()
        plt.show()

    def run_historical_monte_carlo(self, n_scenarios, random_state=220621,sensitivity_parameters=['elas','incentive_opening_probability','improvements','refinery_capacity_fraction_increase_mining'],bayesian_tune=False, n_params=2, n_jobs=3, surrogate_model='ET', log=True):

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
        n_jobs: int, the number of points to sample at each Bayesian iteration, also
            the number of cores to use to parallelise Integration() calculation
        surrogate_model: str, which type of surrogate model to use in BO, can be ET, GBRT,
            GP, RF, or DUMMY
        log: bool, whether to log the rmse (or r2) score prior to giving it to BO, setting it
            to True leads to better performance
        '''
        n_scenarios += 1
        self.random_state = random_state
        self.bayesian_tune = bayesian_tune
        updated_commodity_inputs = 'updated_commodity_inputs'
        if self.constrain_previously_tuned:
            if not self.constrain_tuning_to_sign:
                updated_commodity_inputs += '_unconstrained'
            if 'mine_cost_price_elas' in self.additional_base_parameters.index:
                updated_commodity_inputs += '_mcpe0'
            if os.path.exists(f'data/{updated_commodity_inputs}.pkl'):
                self.updated_commodity_inputs = pd.read_pickle(f'data/{updated_commodity_inputs}.pkl')
                if self.verbosity>-1: print(f'{updated_commodity_inputs} source: data/{updated_commodity_inputs}.pkl')
            elif os.path.exists(f'{updated_commodity_inputs}.pkl'):
                self.updated_commodity_inputs = pd.read_pickle(f'{updated_commodity_inputs}.pkl')
                if self.verbosity>-1: print(f'{updated_commodity_inputs} source: {updated_commodity_inputs}.pkl')
            elif os.path.exists(f'{self.output_data_folder}/{updated_commodity_inputs}.pkl'):
                self.updated_commodity_inputs = pd.read_pickle(f'{self.output_data_folder}/{updated_commodity_inputs}.pkl')
                if self.verbosity>-1: print(f'{updated_commodity_inputs} source: {self.output_data_folder}/{updated_commodity_inputs}.pkl')
            elif hasattr(self,'updated_commodity_inputs'):
                pass
            else:
                raise ValueError(f'{updated_commodity_inputs}.pkl does not exist in the expected locations (in this directory, in data folder, as attribute of Sensitivity). Need to run the historical_sim_check_demand() function to create an initialization of updated_commodity_inputs.pkl')
        else:
            self.updated_commodity_inputs = pd.DataFrame()
        if hasattr(self,'material') and self.material!='':
            if self.constrain_previously_tuned:
                best_params = self.updated_commodity_inputs[self.material].copy().dropna()
                self.updated_commodity_inputs_sub = best_params.copy()
                for q in self.demand_params:
                    if q in self.updated_commodity_inputs_sub.index and self.dont_constrain_demand: self.updated_commodity_inputs_sub.drop(q,inplace=True)
            else:
                best_params = pd.Series(dtype=float)
                best_params.loc['reserves_ratio_price_lag'] = 5
                best_params.loc['close_years_back'] = 3
                self.updated_commodity_inputs_sub = pd.Series(dtype=float)
        else:
            raise ValueError('need to use a string input to changing_base_parameters_series in Sensitivity initialization to run this method')

        # demand_params = ['sector_specific_dematerialization_tech_growth','sector_specific_price_response','region_specific_price_response','intensity_response_to_gdp']
        # for i in demand_params:
        #     self.changing_base_parameters_series.loc[i] = best_params[i]

        for i in [j for j in best_params.index if 'pareto' not in j]:
            self.changing_base_parameters_series.loc[i] = best_params[i]

        if not bayesian_tune:
            n_jobs = 1

        self.n_jobs = n_jobs
        self.log = log

        self.run_monte_carlo(n_scenarios=n_scenarios,
                             random_state=random_state,
                             sensitivity_parameters=sensitivity_parameters,
                             bayesian_tune=bayesian_tune,
                             n_params=n_params,
                             surrogate_model=surrogate_model,
                             n_jobs=n_jobs)

    def create_potential_append(self,big_df,notes,reg_results,initialize=False, mod=None):
        '''
        Sets up a pandas series that could be appended to our big dataframe
        that is used for saving, such that we can check whether this
        combination of parameters already exists in the big dataframe or not

        mod should be None except as part of a BO loop
        '''

        #if no mod is provided, use self.mod, otherwise use mod
        #this is because default model behaviour was to use self.mod but that doesn't work with
        #having multiple samples per BO iteration
        if mod is None:
            mod = self.mod

        new_col_name=0 if len(big_df.columns)==0 else max(big_df.columns)+1
        if type(mod)==Integration:
            if initialize:
                mining = pd.DataFrame([],[],['hyperparam','ml'])
                refine = pd.DataFrame([],[],['hyperparam'])
                demand = pd.DataFrame([],[],['hyperparam'])
                if not hasattr(self,'rmse_df'):
                    self.rmse_df=0
            else:
                mining = deepcopy([mod.mining])[0]
                refine = deepcopy([mod.refine])[0]
                demand = deepcopy([mod.demand])[0]
            if not self.save_mining_info: ml = [0]
            elif self.save_mining_info=='cost_curve':
                if mining.ml.shape[0]>0:
                    ml = mining.ml.copy()[['Commodity price (USD/t)','Minesite cost (USD/t)','Total cash margin (USD/t)','TCRC (USD/t)','Head grade (%)','Recovery rate (%)','Payable percent (%)','Production (kt)','Opening','Simulated closure']]
                else:
                    ml = [0]
            else: ml = mining.ml.copy()
            potential_append = pd.DataFrame(np.array([mod.version, notes, mod.hyperparam, mining.hyperparam,
                                refine.hyperparam, demand.hyperparam, reg_results, ml, self.rmse_df],dtype=object)
                                             ,index=[
                                    'version','notes','hyperparam','mining.hyperparam','refine.hyperparam','demand.hyperparam','results','mine_data','rmse_df'
                                ],columns=[new_col_name])
        elif type(mod)==demandModel or type(mod)==miningModel:
            if not hasattr(self,'rmse_df'):
                self.rmse_df = 0
            potential_append = pd.DataFrame(np.array([mod.version, notes, mod.hyperparam, reg_results, self.rmse_df],dtype=object)
                                         ,index=[
                                'version','notes','hyperparam','results','rmse_df'
                            ],columns=[new_col_name])
        return potential_append

    def check_run_append(self, mod=None, s_n=None):
        '''
        Checks whether the proposed set of hyperparameters has already been run and saved
        in the current big result dataframe. If it has, it skips. Otherwise, it runs the
        scenario and appends it to the big dataframe, resaving it.

        mod and s_n should be None except as part of a BO loop
        '''
        #if no mod is provided, use self.mod, otherwise use mod
        #this is because default model behaviour was to use self.mod but that doesn't work with
        #having multiple samples per BO iteration
        if mod is None:
            mod = self.mod

        if mod is None or (self.bayesian_tune==False and self.given_hyperparam_df==False):
            big_df = pd.read_pickle(self.pkl_filename)
        else: big_df=pd.DataFrame()
        potential_append = self.create_potential_append(big_df=big_df,notes=self.notes,reg_results=[],initialize=True)
        check = self.bayesian_tune or self.given_hyperparam_df or type(mod)==demandModel or type(mod)==miningModel or self.overwrite
        if not check: check = check or check_equivalence(big_df, potential_append)[0]
        if check:
            if self.verbosity>-1:
                print('\tScenario does not already exist, running...')
            if type(mod)==Integration:
                try:
                    if hasattr(self,'demand_or_mining') and self.demand_or_mining=='mining':
                        mod.run_mining_only()
                    else:
                        mod.run()
                except MemoryError:
                    if self.verbosity>-1:
                        print('************************MemoryError, no clue what to do about this************************')
                    param_variable_map = {'Total demand':'total_demand','Primary demand':'primary_demand',
                        'Primary commodity price':'primary_commodity_price','Primary supply':'primary_supply',
                        'Scrap demand':'scrap_demand'}
                    for param in self.objective_parameters:
                        historical = self.historical_data[param]
                        simulated = getattr(mod,param_variable_map[param])
                        if hasattr(simulated,'columns') and 'Global' in simulated.columns:
                            getattr(mod,param_variable_map[param]).loc[:,'Global'] = historical*5
                        else:
                            getattr(mod,param_variable_map[param]).loc[:] = historical*5
                    raise MemoryError
            elif type(mod)==demandModel:
                for i in mod.simulation_time:
                    mod.i = i
                    mod.commodity_price_series.loc[i] = self.historical_data['Primary commodity price'][i]
                    mod.run()
            elif type(mod)==miningModel:
                for i in mod.simulation_time:
                    mod.i = i
                    mod.primary_price_series.loc[i] = self.historical_data['Primary commodity price'][i]
                    if i>mod.simulation_time[0]:
                        mod.primary_tcrc_series.loc[i] = mod.primary_tcrc_series[i-1]
                    if i-1 not in self.historical_data.index:
                        mod.demand_series.loc[i-1] = self.historical_data['Total demand'][i]*self.historical_data['Total demand'][i]/self.historical_data['Total demand'][i+1]
                    else:
                        mod.demand_series.loc[i-1] = self.historical_data['Total demand'][i-1]
                    mod.run()

            if hasattr(self,'val'):
                notes = self.notes+ f', {i}={self.val}'
            else:
                notes = self.notes+''
            ind = [j for j in self.hyperparam_copy.index if type(self.hyperparam_copy['Value'][j]) not in [np.ndarray,list,pd.core.series.Series]]
            z = self.hyperparam_copy['Value'][ind].dropna()!=mod.hyperparam['Value'][ind].dropna()
            z = [j for j in z[z].index]
            if len(z)>0:
                for zz in z:
                    notes += ', {}={}'.format(zz,mod.hyperparam['Value'][zz])

            if type(mod)==Integration:
                if hasattr(self,'demand_or_mining') and self.demand_or_mining=='mining':
                    reg_results = pd.concat([mod.mine_production,mod.mining.primary_price_series,mod.mining.demand_series],axis=1,keys=['Primary supply','Primary commodity price','Total demand'])
                else:
                    reg_results = create_result_df(self,mod)
            elif type(mod) == demandModel:
                reg_results = pd.concat([mod.demand.sum(axis=1),mod.commodity_price_series],axis=1,keys=['Total demand','Primary commodity price'])
            elif type(mod) == miningModel: reg_results = pd.concat([mod.supply_series,mod.primary_price_series,mod.demand_series],axis=1,keys=['Primary supply','Primary commodity price','Total demand'])

            if type(mod)!=Integration or self.demand_or_mining=='mining':
                time_index = np.arange(self.simulation_time[0]-self.changing_base_parameters_series['presimulate_n_years'],self.simulation_time[-1]+1)
                time_index = [i for i in reg_results.index if i in time_index]
                if self.trim_result_df: reg_results = reg_results.loc[time_index]
            potential_append = self.create_potential_append(big_df=big_df,notes=notes,reg_results=reg_results,initialize=False, mod=mod)
            if mod is None or (self.bayesian_tune==False and self.given_hyperparam_df==False):
                big_df = pd.concat([big_df,potential_append],axis=1)
                # self.big_df = pd.concat([self.big_df,potential_append],axis=1)
                big_df.to_pickle(self.pkl_filename)
                if self.verbosity>-1:
                    print('\tScenario successfully saved\n')
            else:
                #if there is a scenario number, use that intead of the default index from create_potential_append()
                #prevents the overwriting of scenarios when having multiple samples at each iteration
                if s_n is not None:
                    potential_append.columns = [s_n]

                return potential_append

        else:
            if mod is None:
                if self.verbosity>-1:
                    print('\tScenario already exists\n')
            else:
                return None

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
