import numpy as np
import pandas as pd
idx = pd.IndexSlice
from matplotlib import pyplot as plt
from scipy import stats
from integration import Integration
from random import seed, sample, shuffle
from demand_class import demandModel
import os
from itertools import combinations
from matplotlib.lines import Line2D

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
    - historical_data_filename: str, the path/filename to access the case study data excel file
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
    
    def __init__(self,material='Al',n_params=3,filename='',historical_data_filename='',rmse_not_mae=True, drop_no_price_elasticity=True, weight_price=1, dpi=50):
        self.material = material
        self.n_params = n_params
        self.filename = filename
        self.historical_data_filename = historical_data_filename
        self.rmse_not_mae = rmse_not_mae
        self.drop_no_price_elasticity = drop_no_price_elasticity
        self.weight_price = weight_price
        self.get_results()            
        self.dpi = dpi
        
    def get_results(self):
        '''
        Handles the FileNotFound error for pickle files so the possible files available get printed out, while calling get_results_hyperparam_history for Integration() based models and getting the right variables for demandModel() based models.
        '''
        try:
            if '_DEM' in self.filename:
                self.big_df = pd.read_pickle(self.filename)
                big_df = self.big_df.copy()
                if self.historical_data_filename=='':
                    self.historical_data_filename='generalization/data/case study data.xlsx'
                self.historical_data = pd.read_excel(self.historical_data_filename,sheet_name=self.material,index_col=0).loc[2001:].astype(float)
                self.simulated_demand = pd.concat([big_df.loc['results'][i]['Total demand'] for i in big_df.columns],keys=big_df.columns,axis=1).loc[2001:2019]
                self.price =  pd.concat([big_df.loc['results'][i]['Primary commodity price'] for i in big_df.columns],keys=big_df.columns,axis=1).loc[2001:2019]
            else:
                self.results, self.hyperparam, self.historical_data = self.get_results_hyperparam_history()
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
        historical_data_filename = self.historical_data_filename
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
        
        if historical_data_filename=='':
            historical_data_filename='generalization/data/case study data.xlsx'
        historical_data = pd.read_excel(historical_data_filename,sheet_name=material,index_col=0).loc[2001:].astype(float)

        self.objective_params = historical_data.columns[:n_params]
        self.objective_results_map = {'Total demand':'Total demand','Primary commodity price':'Refined price',
                                 'Primary demand':'Conc. demand','Primary supply':'Conc. supply',
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
        
        if 'mine_data' in big_df.index: 
            self.mine_data = pd.concat([big_df[i]['mine_data'] for i in big_df.columns],keys=big_df.columns)
            self.mine_data.index.set_names(['scenario','year','mine_id'],inplace=True)
        return results.astype(float), hyperparameters, historical_data.astype(float)
    
    def plot_best_all(self):
        '''
        plots the scenario results for each objective given, with historical and the best-case scenario given thicker outlines. Also does a regression of the best and historical to give statistical measure of fit
        '''
        if not hasattr(self,'results'):
            self.get_results()
            
        hyperparam = self.hyperparam.copy()
        results = self.results.copy()
        historical_data = self.historical_data.copy()

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
        cost_df = self.hyperparam.loc[[i for i in self.hyperparam.index if 'RMSE' in i]].T
        cost_array = cost_df.values
        cost_df.loc[:,'is_pareto'] = is_pareto_efficient_simple(cost_array)
        self.hyperparam.loc['is_pareto',:] = cost_df.is_pareto
        yes = cost_df.loc[cost_df.is_pareto].astype(float)
        no = cost_df.loc[cost_df.is_pareto==False].astype(float)

        if plot:
            fig,ax = easy_subplots(combos, dpi=self.dpi)
            for c,a in zip(combos,ax):
                if plot_non_pareto:
                    no.plot.scatter(x='RMSE '+c[0],y='RMSE '+c[1],loglog=log,ax=a,color='tab:blue')
                yes.plot.scatter(x='RMSE '+c[0],y='RMSE '+c[1],loglog=log,ax=a,color='tab:orange')
            fig.tight_layout()
    
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
        
    def plot_results(self, plot_over_time=True, n_best=0, include_sd=False,
                    plot_hyperparam_heatmap=True,
                    plot_hyperparam_distributions=True, n_per_plot=3,
                    plot_hyperparam_vs_error=True, flip_yx=False):
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
        self.normalize_rmses()
        norm_sum = 'NORM SUM' if include_sd else 'NORM SUM OBJ ONLY'
        if plot_over_time:
            fig,ax = easy_subplots(self.all_params,dpi=self.dpi)
            for i,a in zip(self.all_params, ax):
                results = self.results.copy()[self.objective_results_map[i]].loc[idx[:,2001:]]
                if 'SD' not in i:
                    historical_data = self.historical_data.copy()[i]
                else:
                    historical_data = pd.Series(results.min(),[0])
                
                n_best = n_best if n_best>1 else 2
                simulated = self.hyperparam.loc[norm_sum].astype(float).sort_values().head(n_best).index
                results = results.loc[idx[simulated,:]]

                results, historical_data, unit = self.get_unit(results, historical_data, i)
                best = self.hyperparam.loc[norm_sum].astype(float).idxmin()
                best = results.loc[best]
                best.plot(ax=a,label='Simulated',color='blue')
                if 'SD' not in i:
                    historical_data.plot(ax=a,label='Historical',color='k')
                if n_best>1:
                    results.unstack(0).plot(ax=a,linewidth=1,alpha=0.5,zorder=0,legend=False)
                a.set(title=i,ylabel=i+' ('+unit+')',xlabel='Year')
                a.legend(['Simulated','Historical'])
                
            fig.tight_layout()
        cols = self.hyperparam.loc[norm_sum].sort_values().head(n_best).index
        ind = [i for i in self.hyperparam.index if type(self.hyperparam.loc[i].iloc[0]) in [float,int]]
        ind = [i for i in ind if 'RMSE' not in i and 'NORM' not in i]
        ind = self.hyperparam.loc[ind].loc[(self.hyperparam.loc[ind].std(axis=1)>1e-3)].index
        self.hyperparams_changing = ind
        best_hyperparam = self.hyperparam.copy().loc[ind,cols].astype(float)
        if 'close_years_back' in best_hyperparam.index:
            best_hyperparam.loc['close_years_back'] /= 10
        
        if plot_hyperparam_heatmap:
            plt.figure(figsize=(1.2*n_best,10),dpi=self.dpi)
            sns.heatmap(best_hyperparam,yticklabels=True,annot=True)
        
        if plot_hyperparam_distributions:
            breaks = np.arange(0,int(np.ceil(len(ind)/n_per_plot)))
            fig,ax = easy_subplots(breaks,dpi=self.dpi)
            for a,b in zip(ax,breaks):
                best_hyperparam.iloc[b*n_per_plot:(b+1)*n_per_plot].T.plot(kind='kde',bw_method=0.1,ax=a)
                a.legend(fontsize=16)
        
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
            
    def plot_demand_results(self):
        '''
        plots the demand results, with historical and best-fit scenarios using thicker lines and all others semi-transparent thin lines
        '''
        historical_demand = self.historical_data['Total demand']
        simulated_demand = self.simulated_demand.copy()
        simulated_demand, historical_demand, unit = self.get_unit(simulated_demand, historical_demand, 'Demand')
        rmse = (simulated_demand.subtract(historical_demand,axis=0)**2).sum()**0.5
        best = simulated_demand[rmse.idxmin()]
        fig,ax = easy_subplots(1,dpi=self.dpi)
        simulated_demand.plot(ax=ax[0],linewidth=1,alpha=0.5,legend=False)
        best.plot(ax=ax[0],linewidth=4,color='blue')
        historical_demand.plot(ax=ax[0],linewidth=4,color='k').grid(axis='x')
        material = self.filename.split('_')[0]
        material = material if '/' not in material else material.split('/')[1]
        ax[0].set(title='Historical '+material+' demand',xlabel='Years',ylabel=material.title()+' demand ('+unit+')')
        custom_lines = [Line2D([0], [0], color='k', lw=4),
                        Line2D([0], [0], color='blue', lw=4)]
        ax[0].legend(custom_lines,['Historical','Simulated'],loc='upper left')
        
    def get_unit(self, simulated_demand, historical_demand, param):
        if 'price' in param.lower():
            unit = 'USD/t'
        else:
            min_simulated = abs(simulated_demand).min() if len(simulated_demand.shape)<=1 else abs(simulated_demand).min().min()
            max_simulated = abs(simulated_demand).max() if len(simulated_demand.shape)<=1 else abs(simulated_demand).max().max()
            if min(historical_demand.min(),min_simulated)>1000:
                historical_demand /= 1000
                simulated_demand /= 1000
                unit = 'Mt'
            elif max(historical_demand.max(),max_simulated)<1:
                historical_demand *= 1000
                simulated_demand *= 1000
                unit = 't'
            else:
                unit = 'kt'
        return simulated_demand, historical_demand, unit

    def get_unit_df(self, res):
        maxx, minn = abs(res).max().max(), abs(res).min().min()
        if np.any([i in res.columns[0].lower() for i in ['price','cost','tcrc','spread']]):
            unit = ' (USD/t)'
        elif 'CU' in res.columns[0]:
            unit = ''
        elif 'grade' in res.columns[0]:
            unit = ' (%)'
        elif minn>1000:
            res/=1000
            unit=' (Mt)'
        elif maxx<1:
            res*=1000
            unit=' (t)'
        else:
            unit=' (kt)'
        return res, unit
        
    def plot_best_scenario_sd(self, include_sd=False, plot_supply_demand_stack=True):
        self.normalize_rmses()
        norm_sum = 'NORM SUM' if include_sd else 'NORM SUM OBJ ONLY'
        best = self.hyperparam.loc[norm_sum].astype(float).idxmin()
        print(f'Best scenario is scenario number: {best}')
        results = self.results.loc[best].copy().dropna(how='all')
        variables = ['Total','Conc','Ref.','Scrap','Spread','TCRC','Refined','CU','SR','Direct','Mean total','mine grade','Conc. SD','Ref. SD','Scrap SD']
        fig,ax=easy_subplots(variables)
        for var,a in zip(variables,ax):
            parameters = [i for i in results.columns if var in i and ('SD' not in i or 'SD' in var)]
            res = results[parameters]
            res, unit = self.get_unit_df(res)
            param_str = ', '.join(parameters) if len(', '.join(parameters))<30 else ',\n'.join(parameters)
            res.plot(ax=a,title=param_str+unit)
        fig.tight_layout()
        
        if plot_supply_demand_stack:
            plt.figure()
            res = results[['Pri. ref. prod.','Sec. ref. prod.','Direct melt','Total demand']]
            res = res.replace(0,np.nan)
            res, unit = self.get_unit_df(res)
            res = res.fillna(0)
            plt.plot(res['Total demand'],label='Total demand',color='k')
            plt.stackplot(res.index, res[['Pri. ref. prod.','Sec. ref. prod.','Direct melt']].T, labels=['Pri. ref.','Sec. ref.','Direct melt']);
            plt.legend(framealpha=0.5,frameon=True);
            plt.title('Total supply-demand imbalance'+unit)