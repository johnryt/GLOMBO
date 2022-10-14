import numpy as np
import pandas as pd
idx = pd.IndexSlice
from matplotlib import pyplot as plt
from scipy import stats
from integration import Integration
from random import seed, sample, shuffle
from demand_class import demandModel
from mining_class import miningModel
import os

from useful_functions import easy_subplots, do_a_regress
import seaborn as sns
import statsmodels.api as sm

from itertools import combinations
from matplotlib.lines import Line2D

def get_unit(simulated, historical, param):
    """
    returns dictionary of simulated, historical, and unit.
    Unit is simply the unit, supply your own parentheses, etc.
    - e.g. USD/t, fraction, Mt, t, kt
    """
    simulated, historical = simulated.copy(), historical.copy()
    if np.any([i in param.lower() for i in ['price','tcrc','spread']]):
        unit = 'USD/t'
    elif 'CU' in param or 'SR' in param:
        unit = 'fraction'
    else:
        min_simulated = abs(simulated).min() if len(simulated.shape)<=1 else abs(simulated).min().min()
        max_simulated = abs(simulated).max() if len(simulated.shape)<=1 else abs(simulated).max().max()
        mean_simulated = abs(simulated).mean() if len(simulated.shape)<=1 else abs(simulated).mean().mean()
        min_historical = abs(historical).min() if len(historical.shape)<=1 else abs(historical).min().min()
        max_historical = abs(historical).max() if len(historical.shape)<=1 else abs(historical).max().max()
        mean_historical = abs(historical).mean() if len(historical.shape)<=1 else abs(historical).mean().mean()
        if np.mean([mean_historical,mean_simulated])>1000:
            historical /= 1000
            simulated /= 1000
            unit = 'Mt'
        elif np.mean([mean_historical,mean_simulated])<1:
            historical *= 1000
            simulated *= 1000
            unit = 't'
        else:
            unit = 'kt'
    return {'simulated':simulated, 'historical':historical, 'unit':unit}

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

class Individual():
    '''
    Use this class to interpret Sensitivity runs. Input variables to the
    class initialization are listed here, methods and resulting variables
    for interpretation are listed farther down:
    - material: str, Al, Steel, etc., in the format of the case study data.xlsx file column/sheet names. Not necessary if giving the filename input. The combination of material and n_params can be used to get specific files, but they have to match the aluminum_run_hist_3p.pkl format and be in the same folder as this file. Easier to just give the filename.
    - n_params: int, can be used with the material input to get the file. If filename is given, can be used to only plot a smaller number of columns (where n_params is any integer <= the number of parameters used in run_historical_monte_carlo()). This can be particularly useful when calling the find_pareto function, so you can see the tradeoff between just the two parameters.
    - filename: str, the pickle path/filename produced by your Sensitivity() run.
    - historical_data_file_path: str, the path/filename to access the case study data excel file
    - rmse_not_mae: bool, True means root mean squared error is used to calculate the error and select the best-performing scenarios, while False means the mean absolute error is used.
    - drop_no_price_elasticitiy: bool, True means that the price elasticity to SD must be nonzero and all scenarios where price elasticity to SD (primary_commodity_price_elas_sd in hyperparam) are dropped. False means we leave them in.
    - weight_price: float, allows scaling of the normalized Primary commodity price RMSE/MAE such that NORM SUM and NORM SUM OBJ ONLY rows will consider price error more or less heavily. Is in exponential form.
    - dpi: int, dots per inch controls the resolution of the figures displayed.

    -------------------------------------------

    After initialization, if the instance of Individual() is called indiv,
    use the following variable names and methods/functions:
    - if using a file after running Sensitivity().historical_sim_check_demand():
        - var indiv.historical_data: dataframe with the historical data loaded from case study data excel sheet
        - var indiv.simulated_demand: dataframe with simulated demand for each scenario run
        - var indiv.price: dataframe with prices for each run (should all be identical here)
        - func indiv.plot_demand_results(): plots the demand results, with historical and best-fit scenarios using thicker lines and all others semi-transparent thin lines
    - if using a file after running Sensitivity().run_historical_monte_carlo()
      or any other where the model used is the Integration() model rather than
      demandModel():
        - var indiv.results: multi-indexed dataframe with all scenarios and their outputs over time
        - var indiv.hyperparam: dataframe with all scenarios and their hyperparameters, as well as their RMSE values for each objective (still called RMSE even when MAE is used). Gets updated when the normalize_rmses() function is called to include NORM SUM and NORM SUM OBJ ONLY rows. The NORM SUM row uses the objectives from the Bayesian optimization as well as the scrap, refined, and concentrate supply-demand imbalances. The NORM SUM OBJ ONLY row does not include the SD imblanaces, just the objectives.
        - var indiv.historical_data: dataframe with historical data loaded from case study data excel sheet
        - func indiv.plot_best_all(): plots the scenario results for each objective given, with historical and the best-case scenario given thicker outlines. Also does a regression of the best and historical to give statistical measure of fit
        - func indiv.find_pareto(): adds the is_pareto row to the indiv.hyperparam dataframe, which gives True for all scenarios on the pareto front. Can plot the pareto front for each objective
        - func indiv.normalize_rmses(): adds the NORM SUM and NORM SUM OBJ ONLY rows to the indiv.hyperparam dataframe
        - func indiv.plot_results(): produces many different plots you can use to try and understand the model outputs. Plots the best overall scenario over time (using NORM SUM or NORM SUM OBJ ONLY) for each objective to allow comparison, plots a heatmap of the hyperparameter values for the best n scenarios, plots the hyperparameter distributions, plots the hyperparameter values vs the error value
    '''

    def __init__(self,material='Al',n_params=3,filename='',historical_data_path=None,rmse_not_mae=True, drop_no_price_elasticity=True, weight_price=1, dpi=50, price_rolling=1):
        self.material = material
        self.price_rolling = price_rolling
        self.n_params = n_params
        self.filename = filename
        element_commodity_map = {'Al':'Aluminum','Au':'Gold','Cu':'Copper','Steel':'Steel','Co':'Cobalt','REEs':'REEs','W':'Tungsten','Sn':'Tin','Ta':'Tantalum','Ni':'Nickel','Ag':'Silver','Zn':'Zinc','Pb':'Lead','Mo':'Molybdenum','Pt':'Platinum','Te':'Telllurium','Li':'Lithium'}
        commodity_element_map = dict(zip(element_commodity_map.values(),element_commodity_map.keys()))
        if filename!='':
            if '/' in filename: filename=filename.split('/')[1]
            self.material = commodity_element_map[filename.split('_')[0].capitalize()]
        self.rmse_not_mae = rmse_not_mae
        self.drop_no_price_elasticity = drop_no_price_elasticity
        self.weight_price = weight_price
        self.dpi = dpi
        self.historical_data_path = 'generalization/data/' if historical_data_path==None else historical_data_path
        self.historical_data_file_path = self.historical_data_path+'case study data.xlsx'
        self.price_adjustment_results_file_path = self.historical_data_path+'price adjustment results.xlsx'

        self.get_results()

    def get_results(self):
        '''
        Handles the FileNotFound error for pickle files so the possible files available get printed out, while calling get_results_hyperparam_history for Integration() based models and getting the right variables for demandModel() based models.
        '''
        try:
            if '_DEM' in self.filename or '_mining' in self.filename:
                self.demand_flag = '_DEM' in self.filename
                self.big_df = pd.read_pickle(self.filename)
                big_df = self.big_df.copy()
                self.simulated_demand = pd.concat([big_df.loc['results'][i]['Total demand'] for i in big_df.columns],keys=big_df.columns,axis=1).loc[2001:2019]
                self.historical_data = pd.read_excel(self.historical_data_file_path,sheet_name=self.material,index_col=0).loc[self.simulated_demand.index].astype(float)
                self.historical_data.index = self.historical_data.index.astype(int)
                self.historical_data_column_list = ['Total demand','Primary commodity price','Primary supply','Scrap demand','Total production','Primary demand']
                self.historical_data_column_list = [j for j in self.historical_data_column_list if j in self.historical_data.columns]

                notes = self.big_df[0]['notes']
                if 'price rolling: ' in notes:
                    self.price_rolling = notes.split('price rolling: ')[1].split(',')[0]
                    try: self.price_rolling = int(self.price_rolling)
                    except: self.price_rolling = int(self.price_rolling.split()[0])
                if 'price version: ' in notes:
                    price_to_use = notes.split('price version: ')[1].split(',')[0]
                    commodity = self.filename if '/' not in self.filename else self.filename.split('/')[1]
                    commodity = commodity.split('_')[0].capitalize()
                    if price_to_use!='case study data':
                        price_update_file = pd.read_excel(self.price_adjustment_results_file_path,index_col=0)
                        price_map = {'log':'log('+commodity+')',  'diff':'∆'+commodity,  'original':commodity+' original'}
                        historical_price = price_update_file[price_map[price_to_use]].astype(float)
                        historical_price.name = 'Primary commodity price'
                        if 'Primary commodity price' in self.historical_data.columns:
                            self.historical_data = pd.concat([self.historical_data.drop('Primary commodity price',axis=1),historical_price],axis=1).sort_index().dropna(how='all')
                        else:
                            self.historical_data = pd.concat([self.historical_data,historical_price],axis=1).sort_index().dropna(how='all')
                if 'Primary commodity price' in self.historical_data.columns:
                    self.historical_data.loc[:,'Primary commodity price'] = self.historical_data.loc[:,'Primary commodity price'].rolling(self.price_rolling,min_periods=1,center=True).mean()
                self.price =  pd.concat([big_df.loc['results'][i]['Primary commodity price'] for i in big_df.columns],keys=big_df.columns,axis=1).loc[2001:2019]
                self.hyperparam =  pd.concat([big_df.loc['hyperparam'][i] for i in big_df.columns],keys=big_df.columns,axis=1)
                if 'Primary supply' in big_df.loc['results'][big_df.columns[0]].columns:
                    self.mine_supply = pd.concat([big_df.loc['results'][i]['Primary supply'] for i in big_df.columns],keys=big_df.columns,axis=1).loc[2001:2019]
                if 'rmse_df' in big_df.index:
                    self.rmse_df = big_df.loc['rmse_df'].iloc[-1][0]
                    self.rmse_df.index = pd.MultiIndex.from_tuples(self.rmse_df.index)
                    self.rmse_df = self.rmse_df.unstack(0)
            else:
                self.get_results_hyperparam_history()
        except Exception as e:
            print('Files within current path:')
            self.check_available_files()
            for i in self.filename.split('/')[:-1]:
                print('Files within folder ['+i+'/] specified:')
                self.check_available_files(i)
            raise e

    def check_available_files(self,path=None):
        '''
        displays the available files for the path given, or for the current directory if no path given.
        '''
        display(os.listdir(path))

    def get_results_hyperparam_history(self):
        '''
        gets all the relevant parameters for Integration() based models, and calculates the RMSE/MAE between scenario results and historical values, in both cases saving them as RMSE+objective name in the self.hyperparam dataframe
        '''

        material = self.material
        n_params = self.n_params
        filename = self.filename
        historical_data_file_path = self.historical_data_file_path
        material_map={'Al':'aluminum','Steel':'steel'}
        if n_params==1 and filename=='':
            filename = material_map[material]+'_run_hist.pkl'
            if not os.path.exists(filename):
                filename = material_map[material]+'_run_hist_1p.pkl'
        elif n_params>1 and filename=='':
            filename = material_map[material]+'_run_hist_'+str(n_params)+'p.pkl'
        if type(filename)==str:
            big_df = pd.read_pickle(filename)
        else:
            big_df = pd.concat([pd.read_pickle(i) for i in filename],axis=1).T.reset_index().T
        ind = big_df.loc['results'].dropna().index
        results = pd.concat([big_df.loc['results',i]['Global'][0] for i in ind],keys=ind)
        hyperparameters = pd.concat([big_df.loc['hyperparam'].dropna().loc[i].loc[:,'Value'] for i in ind],keys=ind,axis=1)
        if self.drop_no_price_elasticity:
            cols = (hyperparameters.loc['primary_commodity_price_elas_sd']!=0)
            cols = cols[cols].index
            results = results.loc[idx[cols,:],:].copy()
            hyperparameters = hyperparameters[cols].copy()

        historical_data = pd.read_excel(historical_data_file_path,sheet_name=material,index_col=0).loc[2001:].astype(float)
        historical_data.index = historical_data.index.astype(int)
        self.historical_data_column_list = ['Total demand','Primary commodity price','Primary supply','Scrap demand','Total production','Primary demand']
        self.historical_data_column_list = [j for j in self.historical_data_column_list if j in historical_data.columns]

        notes = big_df[0]['notes']
        if 'price rolling: ' in notes:
            self.price_rolling = notes.split('price rolling: ')[1].split(',')[0]
            try: self.price_rolling = int(self.price_rolling)
            except: self.price_rolling = int(self.price_rolling.split()[0])
        if 'price version: ' in notes:
            price_to_use = notes.split('price version: ')[1].split(',')[0]
            commodity = self.filename if '/' not in self.filename else self.filename.split('/')[1]
            commodity = commodity.split('_')[0].capitalize()
            if price_to_use!='case study data':
                price_update_file = pd.read_excel(self.price_adjustment_results_file_path,index_col=0)
                price_map = {'log':'log('+commodity+')',  'diff':'∆'+commodity,  'original':commodity+' original'}
                historical_price = price_update_file[price_map[price_to_use]].astype(float)
                historical_price.name = 'Primary commodity price'
                if 'Primary commodity price' in historical_data.columns:
                    historical_data = pd.concat([historical_data.drop('Primary commodity price',axis=1),historical_price],axis=1).sort_index().dropna(how='all')
                else:
                    historical_data = pd.concat([historical_data,historical_price],axis=1).sort_index().dropna(how='all')

        if 'Primary commodity price' in historical_data.columns:
            historical_data.loc[:,'Primary commodity price'] = historical_data.loc[:,'Primary commodity price'].rolling(self.price_rolling,min_periods=1,center=True).mean()

        self.objective_params = self.historical_data_column_list[:n_params]
        self.objective_results_map = {'Total demand':'Total demand','Primary commodity price':'Refined price',
                                 'Primary demand':'Conc. demand','Primary supply':'Mine production',
                                'Conc. SD':'Conc. SD','Scrap SD':'Scrap SD','Ref. SD':'Ref. SD'}
        for i in ['Conc.','Ref.','Scrap']:
            results.loc[:,i+' SD'] = results[i+' supply']-results[i+' demand']
        self.sd_ind = [i for i in results.columns if 'SD' in i]
        if 'Conc. SD' not in self.objective_params:
            self.all_params = np.append(self.objective_params,self.sd_ind)
        for obj in self.all_params:
            simulated = results[self.objective_results_map[obj]].unstack(0).loc[2001:2019].astype(float)
            if self.rmse_not_mae:
                if 'SD' in obj:
                    rmses = (simulated**2).sum().div(simulated.shape[0])**0.5
                else:
                    rmses = (simulated.apply(lambda x: x-historical_data[obj])**2).sum().div(simulated.shape[0])**0.5
            else:
                if 'SD' in obj:
                    rmses = abs(simulated).sum()
                else:
                    rmses = abs(simulated.apply(lambda x: x-historical_data[obj])).sum()
            hyperparameters.loc['RMSE '+obj] = rmses

        self.big_df = big_df.copy()

        if 'rmse_df' in big_df.index and type(big_df.loc['rmse_df'].iloc[-1])!=int:
            self.rmse_df = big_df.loc['rmse_df'].iloc[-1][0]
            self.rmse_df.index = pd.MultiIndex.from_tuples(self.rmse_df.index)
            self.rmse_df = self.rmse_df.unstack(0)

        if 'mine_data' in big_df.index and type(big_df.loc['mine_data'].iloc[1])==pd.core.frame.DataFrame:
            self.mine_data = pd.concat([big_df[i]['mine_data'] for i in big_df.columns],keys=big_df.columns)
        else:
            self.mine_data = pd.DataFrame(0,index=results.index,columns=pd.MultiIndex.from_product([['Mine data'],np.arange(0,10)])).stack()
        self.mine_data.index.set_names(['scenario','year','mine_id'],inplace=True)

        self.results, self.hyperparam, self.historical_data = results.astype(float), hyperparameters, historical_data.astype(float)

    def plot_best_all(self):
        '''
        plots the scenario results for each objective given, with historical and the best-case scenario given thicker outlines. Also does a regression of the best and historical to give statistical measure of fit
        '''
        if not hasattr(self,'results'):
            self.get_results()

        hyperparam = self.hyperparam.copy()
        results = self.results.copy()
        historical_data = self.historical_data.copy().loc[2001:2019]

        for obj in self.objective_params:
            simulated = results[self.objective_results_map[obj]].unstack(0).loc[2001:2019]
            best = simulated[(simulated.astype(float).apply(lambda x: x-historical_data[obj])**2).sum().astype(float).idxmin()]

            fig,ax = easy_subplots(3,dpi=self.dpi)
            for i,a in enumerate(ax[:2]):
                if i==0:
                    simulated.plot(linewidth=1,alpha=0.5,legend=False,ax=a)
                historical_data[obj].plot(ax=a,label='Historical')
                best.plot(ax=a,label='Best simulated')
                if i==1:
                    a.legend()
                a.set(title=obj+' over time',xlabel='Year',ylabel=obj)

            do_a_regress(best.astype(float),historical_data[obj].astype(float),ax=ax[2],xlabel='Simulated',ylabel='Historical')
            ax[-1].set(title='Historical regressed on simulated')
            plt.suptitle(obj+', varying demand parameters (historical sensitivity result)',fontweight='bold')
            fig.tight_layout()

    def find_pareto(self, plot=False, log=True, plot_non_pareto=True):
        '''
        adds the is_pareto row to the indiv.hyperparam dataframe, which gives True for all scenarios on the pareto front. Can plot the pareto front for each objective

        Inputs:
        - plot: bool, whether to plot the pareto result
        - log: bool, whether to use log scaling while plotting the pareto result
        - plot_non_pareto: bool, whether to include scenarios not on the pareto front when plotting
        '''
        if not hasattr(self,'results'):
            self.get_results()
        combos = list(combinations(self.objective_params,2))
        cost_df = self.rmse_df.loc[[i for i in self.rmse_df.index if 'RMSE' in i]].T
        cost_array = cost_df.values
        cost_df.loc[:,'is_pareto'] = is_pareto_efficient_simple(cost_array)
        yes = cost_df.loc[cost_df.is_pareto].astype(float)
        no = cost_df.loc[cost_df.is_pareto==False].astype(float)

        if plot:
            fig,ax = easy_subplots(combos, dpi=self.dpi)
            for c,a in zip(combos,ax):
                if plot_non_pareto:
                    no.plot.scatter(x='RMSE '+c[0],y='RMSE '+c[1],loglog=log,ax=a,color='tab:blue')
                yes.plot.scatter(x='RMSE '+c[0],y='RMSE '+c[1],loglog=log,ax=a,color='tab:orange')
            fig.tight_layout()
        self.pareto_ind = yes

    def normalize_rmses(self):
        '''
        adds the NORM SUM and NORM SUM OBJ ONLY rows to the indiv.hyperparam dataframe
        '''
        for i in [i for i in self.hyperparam.index if 'RMSE' in i]:
            self.hyperparam.loc['NORM '+i] = self.hyperparam.loc[i]/self.hyperparam.loc[i].replace(0,1e-6).min()
        if 'NORM RMSE Primary commodity price' in self.hyperparam.index:
            self.hyperparam.loc['NORM RMSE Primary commodity price'] = self.hyperparam.loc['NORM RMSE Primary commodity price']**self.weight_price
        normed = [i for i in self.hyperparam.index if 'NORM' in i]
        normed_obj = [i for i in self.hyperparam.index if 'NORM' in i and np.any([j in i for j in self.objective_params])]
        self.hyperparam.loc['NORM SUM'] = self.hyperparam.loc[normed].sum()
        self.hyperparam.loc['NORM SUM OBJ ONLY'] = self.hyperparam.loc[normed_obj].sum()

        # repeating above for rmse_df
        for i in [i for i in self.rmse_df.index if 'RMSE' in i]:
            self.rmse_df.loc['NORM '+i] = np.log(self.rmse_df.loc[i].astype(float))/np.log(self.rmse_df.loc[i].astype(float)).replace(0,1e-6).min()
            self.rmse_df.loc['LOG '+i] = np.log(self.rmse_df.loc[i].astype(float))
        if 'NORM Primary commodity price RMSE' in self.rmse_df.index:
            self.rmse_df.loc['NORM Primary commodity price RMSE'] = self.rmse_df.loc['NORM Primary commodity price RMSE']**self.weight_price
            self.rmse_df.loc['LOG Primary commodity price RMSE'] = self.rmse_df.loc['LOG Primary commodity price RMSE']**self.weight_price
        normed = [i for i in self.rmse_df.index if 'NORM' in i]
        logged = [i for i in self.rmse_df.index if 'LOG' in i]
        normed_obj = [i for i in self.rmse_df.index if 'NORM' in i and np.any([j in i for j in self.objective_params])]
        self.rmse_df.loc['NORM SUM'] = self.rmse_df.loc[normed].sum()
        self.rmse_df.loc['LOG SUM'] = np.log(np.exp(self.rmse_df.loc[logged]).sum())
        self.rmse_df.loc['NORM SUM OBJ ONLY'] = self.rmse_df.loc[normed_obj].sum()

    def plot_results(self, plot_over_time=True, n_best=0, include_sd=False, nth_best=1,
                     plot_sd_over_time=True,
                     plot_best_indiv_over_time=True,
                    plot_hyperparam_heatmap=True,
                    plot_hyperparam_distributions=True, n_per_plot=3,
                    plot_hyperparam_vs_error=True, flip_yx=False,
                    plot_best_params=True, plot_supply_demand_stack=True, best_ind=-1):
        '''
        produces many different plots you can use to try and understand the model outputs. More info is given with each model input bool description below.

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
        if not hasattr(self,'rmse_df'):
            plot_best_indiv_over_time=False
            plot_hyperparam_heatmap=False
            plot_hyperparam_distributions=False
            plot_hyperparam_vs_error=False
            plot_best_params=False
            plot_supply_demand_stack=False

        fig_list = []
        if hasattr(self,'rmse_df'):
            self.normalize_rmses()
        norm_sum = 'NORM SUM' if include_sd else 'NORM SUM OBJ ONLY'
        if plot_over_time:
            fig,ax = easy_subplots(self.all_params[:3],dpi=self.dpi)
            for i,a in zip(self.all_params[:3], ax):
                results = self.results.copy()[self.objective_results_map[i]].loc[idx[:,2001:]]
                if 'SD' not in i:
                    historical_data = self.historical_data.copy()[i]
                else:
                    historical_data = pd.Series(results.min(),[0])

                n_best = n_best if n_best>1 else 2
                # simulated = self.hyperparam.loc[norm_sum].astype(float).sort_values().head(n_best).index
                if hasattr(self,'rmse_df'):
                    if nth_best<0:
                        simulated = self.rmse_df.loc['score'].astype(float).sort_values().head(n_best).index
                    else:
                        simulated = self.rmse_df.loc['LOG SUM'].astype(float).sort_values().head(n_best).index
                    results = results.loc[idx[simulated,:]]

                diction = get_unit(results, historical_data, i)
                results, historical_data, unit = [diction[i] for i in ['simulated','historical','unit']]
                # best = self.hyperparam.loc[norm_sum].astype(float).sort_values().index[nth_best-1]
                if hasattr(self,'rmse_df'):
                    if nth_best<0:
                        best_ind = self.rmse_df.loc['score'].astype(float).sort_values().index[-nth_best-1]
                    else:
                        best_ind = self.rmse_df.loc['LOG SUM'].astype(float).sort_values().index[nth_best-1]

                if hasattr(self,'rmse_df'):
                    best = results.loc[best_ind]
                if 'SD' not in i:
                    hist_line = a.plot(historical_data,label='Historical',color='k',linewidth=6)
                    if hasattr(self,'rmse_df'):
                        regr = sm.GLS(historical_data.loc[2001:2019],sm.add_constant(best.loc[2001:2019])).fit(cov_type='HC3')
                        reg_res = r'$R^2$'+': {:.3f}'.format(regr.rsquared)
                else: reg_res=''
                if hasattr(self,'rmse_df'):
                    sim_line = a.plot(best,label='Simulated',color='tab:blue',linewidth=6)
                    reg_point = plt.plot(best,color='w',zorder=0,alpha=0)
                if hasattr(self,'rmse_df'):
                    if n_best>1:
                        a.plot(results.unstack(0),linewidth=1,alpha=0.5,zorder=0)
                else:
                    a.plot(results.unstack(0),linewidth=4,alpha=0.8)

                if hasattr(self,'rmse_df'):
                    mins = min(historical_data.min(),best.min())*0.95
                    maxs = max(historical_data.max(),best.max())*1.1
                # else:
                #     mins = min(historical_data.min(),results.min().min())*0.95
                #     maxs = max(historical_data.max(),results.max().max())*1.1

                if hasattr(self,'rmse_df'):
                    a.set(title='Best '+i,ylabel=i+' ('+unit+')',xlabel='Year',ylim=(mins,maxs))
                    a.legend([hist_line[0],sim_line[0],reg_point[0]],['Historical','Simulated',reg_res])
                else:
                    a.set(title=i,ylabel=f'{i} ({unit})',xlabel='Year')
                self.sim = results
                self.hist = historical_data
            fig.tight_layout()
            fig_list += [fig]
            if hasattr(self,'rmse_df'):
                print(f'best scenario number is: {best_ind}')

        if plot_sd_over_time:
            fig,ax = easy_subplots(self.all_params[3:],dpi=self.dpi)
            for i,a in zip(self.all_params[3:], ax):
                results = self.results.copy()[self.objective_results_map[i]].loc[idx[:,2001:]]
                if 'SD' not in i:
                    historical_data = self.historical_data.copy()[i]
                else:
                    historical_data = pd.Series(results.min(),[0])

                n_best = n_best if n_best>1 else 2
                # simulated = self.hyperparam.loc[norm_sum].astype(float).sort_values().head(n_best).index
                if hasattr(self,'rmse_df'):
                    if nth_best<0:
                        simulated = self.rmse_df.loc['score'].astype(float).sort_values().head(n_best).index
                    else:
                        simulated = self.rmse_df.loc['LOG SUM'].astype(float).sort_values().head(n_best).index
                    results = results.loc[idx[simulated,:]]

                results, historical_data, unit = self.get_unit(results, historical_data, i)
                # best = self.hyperparam.loc[norm_sum].astype(float).sort_values().index[nth_best-1]
                if hasattr(self,'rmse_df'):
                    if nth_best<0:
                        best_ind = self.rmse_df.loc['score'].astype(float).sort_values().index[-nth_best-1]
                    else:
                        best_ind = self.rmse_df.loc['LOG SUM'].astype(float).sort_values().index[nth_best-1]

                    best = results.loc[best_ind]
                    sim_line = a.plot(best,label='Simulated',color='tab:blue')
                if 'SD' not in i:
                    hist_line = a.plot(historical_data,label='Historical',color='k')
                    if hasattr(self,'rmse_df'):
                        regr = sm.GLS(historical_data.loc[2001:2019],best.loc[2001:2019]).fit(cov_type='HC3')
                        reg_res = r'$R^2$'+': {:.3f}'.format(regr.rsquared)
                else: reg_res=''
                if hasattr(self,'rmse_df'):
                    reg_point = plt.scatter([historical_data.index[0]],[historical_data.iloc[0]],color='w',zorder=0,alpha=0)
                if hasattr(self,'rmse_df'):
                    if n_best>1:
                        a.plot(results.unstack(0),linewidth=1,alpha=0.5,zorder=0)
                else:
                    a.plot(results.unstack(0),linewidth=4,alpha=0.8)

                if hasattr(self,'rmse_df'):
                    a.set(title='Best '+i,ylabel=i+' ('+unit+')',xlabel='Year')
                else:
                    a.set(title=i,ylabel=i+' ('+unit+')',xlabel='Year')

                if hasattr(self,'rmse_df'):
                    a.legend([hist_line[0],sim_line[0],reg_point],['Historical','Simulated',reg_res])

            fig.tight_layout()
            fig_list += [fig]

        if hasattr(self,'rmse_df'):
            cols = self.hyperparam.loc[norm_sum].sort_values().head(n_best).index
            ind = [i for i in self.hyperparam.index if type(self.hyperparam.loc[i].iloc[0]) in [float,int]]
            ind = [i for i in ind if 'RMSE' not in i and 'NORM' not in i]
            ind = self.hyperparam.loc[ind].loc[(self.hyperparam.loc[ind].std(axis=1)>1e-3)].index
            self.hyperparams_changing = ind
            best_hyperparam = self.hyperparam.copy().loc[ind,cols].astype(float)
            if 'close_years_back' in best_hyperparam.index:
                best_hyperparam.loc['close_years_back'] /= 10

        if plot_best_indiv_over_time:
            fig,ax = easy_subplots(self.objective_params,dpi=self.dpi)
            for i,a in zip(self.objective_params, ax):
                results = self.results.copy()[self.objective_results_map[i]].loc[idx[:,2001:]]
                if 'SD' not in i:
                    historical_data = self.historical_data.copy()[i]
                else:
                    historical_data = pd.Series(results.min(),[0])

                n_best = n_best if n_best>1 else 2
                simulated = self.hyperparam.loc['RMSE '+i].astype(float).sort_values().head(n_best).index
                results = results.loc[idx[simulated,:]]

                results, historical_data, unit = self.get_unit(results, historical_data, i)
                best_ind = self.hyperparam.loc['RMSE '+i].astype(float).idxmin()
                best = results.loc[best_ind]
                best.plot(ax=a,label='Simulated',color='blue')
                if 'SD' not in i:
                    historical_data.plot(ax=a,label='Historical',color='k')
                if n_best>1:
                    results.unstack(0).plot(ax=a,linewidth=1,alpha=0.5,zorder=0,legend=False)
                a.set(title=i+f' ({best_ind})',ylabel=i+' ('+unit+')',xlabel='Year')
                a.legend(['Simulated','Historical'])
            fig_list += [fig]

        if plot_hyperparam_heatmap:
            fig = plt.figure(figsize=(1.2*n_best,10),dpi=self.dpi)
            heat = best_hyperparam.copy()
            while (heat.max(axis=1)>1).any():
                lrg=heat.max(axis=1)>1
                lrg=lrg[lrg].index
                heat.loc[lrg,:] /= 10
                heat.rename(dict(zip(lrg,[i+'*' for i in lrg])),inplace=True)
            sns.heatmap(heat,yticklabels=True,annot=True)
            fig_list += [fig]

        if plot_hyperparam_distributions:
            breaks = np.arange(0,int(np.ceil(len(ind)/n_per_plot)))
            fig,ax = easy_subplots(breaks,dpi=self.dpi)
            for a,b in zip(ax,breaks):
                best_hyperparam.iloc[b*n_per_plot:(b+1)*n_per_plot].T.plot(kind='kde',bw_method=0.1,ax=a)
                a.legend(fontsize=16)
            fig_list += [fig]

        if plot_hyperparam_vs_error:
            fig,ax=easy_subplots(self.hyperparams_changing,dpi=self.dpi)
            for a,i in zip(ax,self.hyperparams_changing):
                if flip_yx:
                    v = self.hyperparam.sort_values(by=i,axis=1).T.reset_index(drop=True).T
                else:
                    v = self.hyperparam.sort_values(by=norm_sum,axis=1).T.reset_index(drop=True).T
                x = v.copy().loc[i]
                y = v.copy().loc[norm_sum]
                y = y.where(y<y.quantile(0.7)).dropna()

                if flip_yx:
                    a.plot(x[y.index],y)
                    a.set(title=i,xlabel='Hyperparameter value',ylabel='Normalized error')
                else:
                    a.plot(y,x[y.index])
                    a.set(title=i,ylabel='Hyperparameter value',xlabel='Normalized error')

            fig.tight_layout()
            fig_list += [fig]

        if plot_best_params:
            fig = self.plot_best_scenario_sd(include_sd=include_sd, plot_supply_demand_stack=plot_supply_demand_stack, best=best_ind)
            fig_list += [fig]
        return fig_list

    def plot_demand_results(self, mining=False, n_best=25):
        '''
        plots the demand results, with historical and best-fit scenarios using thicker lines and all others semi-transparent thin lines
        '''
        if self.demand_flag:
            historical_demand = self.historical_data['Total demand']
            simulated_demand = self.simulated_demand.copy()
        else:
            historical_demand = self.historical_data['Primary supply']
            simulated_demand = self.mine_supply.copy()
        simulated_demand, historical_demand, unit = self.get_unit(simulated_demand, historical_demand, 'Demand')
        rmse = (simulated_demand.subtract(historical_demand,axis=0)**2).sum()**0.5
        best = simulated_demand[rmse.idxmin()]
        best_n = simulated_demand.loc[:,rmse.sort_values().tail(n_best).index]
        fig,ax = easy_subplots(3,dpi=self.dpi)
        best_n.plot(ax=ax[0],linewidth=1,alpha=0.5,legend=False)
        best.plot(ax=ax[0],linewidth=4,color='blue')
        historical_demand.plot(ax=ax[0],linewidth=4,color='k').grid(axis='x')
        material = self.filename.split('_')[0]
        material = material if '/' not in material else material.split('/')[1]

        if self.demand_flag:
            ax[0].set(title='Historical '+material+' demand',xlabel='Years',ylabel=f'{material} demand ({unit})'.capitalize())
        else:
            ax[0].set(title='Historical '+material+' mine production',xlabel='Years',ylabel=f'{material} production ({unit})'.capitalize())
        custom_lines = [Line2D([0], [0], color='k', lw=4),
                        Line2D([0], [0], color='blue', lw=4)]
        ax[0].legend(custom_lines,['Historical','Simulated'],loc='upper left')
        best.name = 'Simulated'
        best.plot(ax=ax[1],linewidth=4,color='blue',label='Simulated')
        historical_demand.loc[best.index].plot(ax=ax[1],linewidth=4,color='k',label='Historical').grid(axis='x')
        ax[1].legend()
        if self.demand_flag:
            ax[1].set(title=f'Historical {material} demand',xlabel='Years',ylabel=f'{material} demand ({unit})'.capitalize())
        else:
            ax[1].set(title=f'Historical {material} mine production',xlabel='Years',ylabel=f'{material} production ({unit})'.capitalize())

        do_a_regress(best.astype(float),historical_demand.loc[best.index].astype(float),ax=ax[2])

    def get_unit(self, simulated_demand, historical_demand, param):
        if 'price' in param.lower():
            unit = 'USD/t'
        else:
            min_simulated = abs(simulated_demand).min() if len(simulated_demand.shape)<=1 else abs(simulated_demand).min().min()
            max_simulated = abs(simulated_demand).max() if len(simulated_demand.shape)<=1 else abs(simulated_demand).max().max()
            mean_simulated = abs(simulated_demand).mean() if len(simulated_demand.shape)<=1 else abs(simulated_demand).mean().mean()
            if np.mean([historical_demand.mean(),mean_simulated])>1000:
                historical_demand /= 1000
                simulated_demand /= 1000
                unit = 'Mt'
            elif np.mean([historical_demand.mean(),mean_simulated])<1:
                historical_demand *= 1000
                simulated_demand *= 1000
                unit = 't'
            else:
                unit = 'kt'
        return simulated_demand, historical_demand, unit

    def get_unit_df(self, res):
        maxx, minn, mean = abs(res).max().max(), abs(res).min().min(), abs(res).mean().mean()
        if np.any([i in res.columns[0].lower() for i in ['price','cost','tcrc','spread']]):
            unit = ' (USD/t)'
        elif 'CU' in res.columns[0] or 'SR' in res.columns[0]:
            unit = ''
        elif 'grade' in res.columns[0]:
            unit = ' (%)'
        elif mean>1000:
            res/=1000
            unit=' (Mt)'
        elif mean<1:
            res*=1000
            unit=' (t)'
        else:
            unit=' (kt)'
        return res, unit

    def plot_best_scenario_sd(self, include_sd=False, plot_supply_demand_stack=True, best=-1):
        self.normalize_rmses()
        norm_sum = 'NORM SUM' if include_sd else 'NORM SUM OBJ ONLY'
        results = self.results.loc[best].copy().dropna(how='all')
        variables = ['Total','Conc','Ref.','Scrap','Spread','TCRC','Refined','CU','SR','Direct','Mean total','mine grade','Conc. SD','Ref. SD','Scrap SD']
        fig,ax=easy_subplots(variables)
        for var,a in zip(variables,ax):
            parameters = [i for i in results.columns if var in i and ('SD' not in i or 'SD' in var)]
            res = results[parameters].loc[2001:]
            res, unit = self.get_unit_df(res)
            param_str = ', '.join(parameters) if len(', '.join(parameters))<30 else ',\n'.join(parameters)
            res.plot(ax=a,title=param_str+unit)
        fig.tight_layout()


        if plot_supply_demand_stack:
            fig1 = plt.figure()
            res = results[['Pri. ref. prod.','Sec. ref. prod.','Direct melt','Total demand']].loc[2001:]
            res = res.replace(0,np.nan)
            res, unit = self.get_unit_df(res)
            res = res.fillna(0)
            plt.plot(res['Total demand'],label='Total demand',color='k')
            plt.stackplot(res.index, res[['Pri. ref. prod.','Sec. ref. prod.','Direct melt']].T, labels=['Pri. ref.','Sec. ref.','Direct melt']);
            plt.legend(framealpha=0.5,frameon=True);
            plt.title('Total supply-demand imbalance'+unit)
        return fig, fig1
