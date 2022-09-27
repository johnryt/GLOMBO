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
import shap
from Individual import Individual

class Many():
    def __init__(self, data_folder=None, pkl_folder=None):
        self.ready_commodities = ['Steel','Al','Au','Sn','Cu','Ni','Ag','Zn','Pb']
        self.element_commodity_map = {'Steel':'Steel','Al':'Aluminum','Au':'Gold','Cu':'Copper','Steel':'Steel','Co':'Cobalt','REEs':'REEs','W':'Tungsten','Sn':'Tin','Ta':'Tantalum','Ni':'Nickel','Ag':'Silver','Zn':'Zinc','Pb':'Lead','Mo':'Molybdenum','Pt':'Platinum','Te':'Telllurium','Li':'Lithium'}
        self.commodity_element_map = dict(zip(self.element_commodity_map.values(),self.element_commodity_map.keys()))
        self.data_folder = 'generalization/data' if data_folder==None else data_folder
        self.pkl_folder = 'data' if pkl_folder==None else pkl_folder

    def run_all_demand(self, n_runs=50, commodities=None):
        commodities = self.ready_commodities if commodities==None else commodities
        for material in commodities:
            print('-'*40)
            print(material)
            mat = self.element_commodity_map[material].lower()
            filename=f'{self.pkl_folder}/{mat}_run_hist.pkl'
            self.shist1 = Sensitivity(pkl_filename=filename, data_folder=self.data_folder,changing_base_parameters_series=material,notes='Monte Carlo aluminum run',
                            simulation_time=np.arange(2001,2020),OVERWRITE=True,use_alternative_gold_volumes=True,
                                historical_price_rolling_window=5,verbosity=0)
            self.shist1.historical_sim_check_demand(n_runs,demand_or_mining='demand')

    def run_all_mining(self, n_runs=50, commodities=None):
        commodities = self.ready_commodities if commodities==None else commodities
        for material in commodities:
            print('-'*40)
            print(material)
            mat = self.element_commodity_map[material].lower()
            filename=f'{self.pkl_folder}/{mat}_run_hist.pkl'
            self.shist = Sensitivity(pkl_filename=filename, data_folder=self.data_folder,changing_base_parameters_series=material,notes='Monte Carlo aluminum run',
                            simulation_time=np.arange(2001,2020),OVERWRITE=True,use_alternative_gold_volumes=True,
                                historical_price_rolling_window=5,verbosity=0,
                               incentive_opening_probability_fraction_zero=0)
            self.shist.historical_sim_check_demand(n_runs,demand_or_mining='mining')

    def run_all_integration(self, n_runs=200, commodities=None, normalize_objectives=False, constrain_previously_tuned=True):
        commodities = self.ready_commodities if commodities==None else commodities
        for material in commodities:
            print('-'*40)
            print(material)
            # timer=IterTimer()
            n=3
            thing = '_'+str(n)+'p' if n>1 else ''
            mat = self.element_commodity_map[material].lower()
            filename=f'{self.pkl_folder}/{mat}_run_hist_all{thing}.pkl'
            print('--'*15+filename+'-'*15)
            s = Sensitivity(pkl_filename=filename, data_folder=self.data_folder,changing_base_parameters_series=material,notes=f'Monte Carlo {material} run',
                            additional_base_parameters=pd.Series(1,['refinery_capacity_growth_lag']),
                            simulation_time=np.arange(2001,2020), include_sd_objectives=False,
                            OVERWRITE=True,verbosity=0,historical_price_rolling_window=5,
                            constrain_previously_tuned=constrain_previously_tuned, normalize_objectives=normalize_objectives)
            sensitivity_parameters = [
                'pri CU price elas',
                'sec CU price elas',
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
            s.run_historical_monte_carlo(n_scenarios=n_runs,bayesian_tune=True,n_params=n,
                sensitivity_parameters=sensitivity_parameters)
            # add 'response','growth' to sensitivity_parameters input to allow demand parameters to change again

    def get_variables(self, demand_mining_all='demand'):
        '''
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
                indiv = Individual(filename=f'data/{material}_run_hist_DEM.pkl',rmse_not_mae=False,dpi=50)
                rmse_or_score = 'RMSE'
            elif demand_mining_all=='mining':
                indiv = Individual(filename=f'data/{material}_run_hist_mining.pkl',rmse_not_mae=False,dpi=50)
                rmse_or_score = 'RMSE'
            elif demand_mining_all=='all':
                indiv = Individual(filename=f'data/{material}_run_hist_all_3p.pkl',rmse_not_mae=False,dpi=50)
                rmse_or_score = 'score'
            else: raise ValueError('input for the demand_mining_all variable when calling the Many().get_variables() function must be a string of one of the following: demand, many, all')


            for df_name in ['rmse_df','hyperparam','simulated_demand','results','historical_data','mine_supply']:
                df_name_sorted = f'{df_name}_sorted'
                if hasattr(indiv,df_name):
                    if not hasattr(self,df_name):
                        setattr(self,df_name,pd.DataFrame())
                        setattr(self,df_name_sorted,pd.DataFrame())
                    df_ph = pd.concat([getattr(indiv,df_name).dropna(how='all').dropna(axis=1,how='all')],keys=[material])
                    if df_ph.columns.nlevels>1 and 'Notes' in df_ph.columns.get_level_values(1): df_ph = df_ph.loc[:,idx[:,'Value']].droplevel(1,axis=1)
                    if df_name!='rmse_df' and df_name!='historical_data':
                        sorted_cols = self.rmse_df.loc[material,rmse_or_score].sort_values().index
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

    def get_multiple(self, demand=True, mining=True, integration=False, reinitialize=False):
        if demand and (not hasattr(self,'demand') or reinitialize):
            self.demand = Many()
            self.demand.get_variables('demand')
            feature_importance(many.demand,plot=False)

        if mining and (not hasattr(self,'mining') or reinitialize):
            self.mining = Many()
            self.mining.get_variables('mining')
            feature_importance(many.mining,plot=False)

        if integration and (not hasattr(self,'integ') or reinitialize):
            self.integ = Many()
            self.integ.get_variables('all')
            feature_importance(many.all,plot=False)

    def plot_all_demand(self, dpi=50):
        for material in self.ready_commodities:
            material = self.element_commodity_map[material].lower()
            filename=f'{self.pkl_folder}/{material}_run_hist_DEM.pkl'
            indiv = Individual(filename=filename,rmse_not_mae=False,dpi=dpi)
            indiv.plot_demand_results()
        plt.show()
        plt.close()

    def plot_all_mining(self,dpi=50):
        for material in self.ready_commodities:
            material = self.element_commodity_map[material].lower()
            filename=f'{self.pkl_folder}/{material}_run_hist_mining.pkl'
            indiv = Individual(filename=filename,rmse_not_mae=False,dpi=dpi)
            indiv.plot_demand_results()

    def plot_all_integration(self,dpi=50,
                    plot_over_time=True,
                    include_sd=False,
                    plot_best_indiv_over_time=False,
                    plot_hyperparam_heatmap=False,
                    n_best=20,
                    plot_hyperparam_distributions=False,
                    n_per_plot=4,
                    plot_hyperparam_vs_error=False,
                    flip_yx=False,
                    plot_best_params=False,
                    plot_supply_demand_stack=False):
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

        for element in self.ready_commodities:
            material = self.element_commodity_map[element].lower()
            indiv = Individual(element,3,filename=f'data/{material}_run_hist_all_3p.pkl',
                   rmse_not_mae=True,weight_price=1,dpi=50,price_rolling=5)
            # indiv.plot_best_all()
            # indiv.find_pareto(plot=True,log=True,plot_non_pareto=False)
            indiv.plot_results(plot_over_time=plot_over_time,
                               include_sd=include_sd,
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

def feature_importance(self,plot=None, dpi=50,recalculate=False, standard_scaler=True, plot_commodity_importances=False, commodity=None):
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

    rmse_df = self.rmse_df.copy().astype(float)
    if commodity!=None: rmse_df = rmse_df.loc[idx[commodity,:],:]
    if not hasattr(self,'importances_df') or recalculate or plot_commodity_importances:
        importances_df = pd.DataFrame()
        for Regressor, name in zip([RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor],['RandomForest','ExtraTrees','GradientBoosting']):
            if plot_train_test: fig2,ax2 = easy_subplots(2,dpi=dpi)
            for e,dummies in enumerate([True,False]):
                X_df = rmse_df.copy()
                for r in ['R2','RMSE','region_specific_price_response','score']:
                    if r in X_df.index.get_level_values(1):
                        X_df = X_df.drop(r,level=1)
                X_df = X_df.unstack().stack(level=0)
                y_df = np.log(rmse_df.loc[idx[:,'RMSE'],:].unstack().stack(level=0))

                if standard_scaler:
                    scaler = StandardScaler()
                    x_std = scaler.fit_transform(X_df.values)
                    X_df = pd.DataFrame(x_std,X_df.index,X_df.columns)

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
                test.loc[:,'predicted RMSE'] = regr.predict(X_test)

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
                    do_a_regress(test.RMSE, test['predicted RMSE'],ax=ax2[e])
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
        feature_importance(self.mining,plot=False)
    if not hasattr(self.demand,'importances_df') or np.any(['commodity' in i.lower() for i in self.demand.importances_df.index]):
        feature_importance(self.demand,plot=False)

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

def plot_all_feature_importance_plots(self, dpi=50,
                                      plot_feature_importance_no_show_dummies=True,
                                      plot_feature_importance_show_dummies=True,
                                      plot_mining_and_demand_importance=True,
                                      plot_commodity_level_importances=True,
                                      commodity=None,
                                     ):
    self.feature_plot_figs = []
    self.feature_plot_axs = []
    if plot_feature_importance_no_show_dummies:
        for s in [self.mining,self.demand]:
            fig,ax = feature_importance(s,dpi=dpi,recalculate=True,commodity=commodity)
            self.feature_plot_figs += [fig]
            self.feature_plot_axs += [ax]

    if plot_mining_and_demand_importance:
        if not plot_feature_importance_no_show_dummies:
            for s in [self.mining,self.demand]:
                feature_importance(s,dpi=dpi,recalculate=True, plot=False)
        fig,ax = nice_feature_importance_plot(self, dpi=dpi)
        self.feature_plot_figs += [fig]
        self.feature_plot_axs += [ax]

    if plot_feature_importance_show_dummies:
        for s in [self.mining,self.demand]:
            fig,ax = feature_importance(s,dpi=dpi,plot_commodity_importances=True)
            self.feature_plot_figs += [fig]
            self.feature_plot_axs += [ax]
    if plot_commodity_level_importances:
        for s in [self.mining,self.demand]:
            fig,ax = commodity_level_feature_importance(s,dpi=dpi)
            self.feature_plot_figs += [fig]
            self.feature_plot_axs += [ax]

def make_parameter_names_nice(ind):
    updated = [i.replace('_',' ').replace('sector specific ','').replace('dematerialization tech growth','Intensity decline per year').replace(
        'price response','intensity response to price').capitalize().replace('gdp','GDP').replace(
        ' cu ',' CU ').replace(' og ',' OG ').replace('Primary price resources contained elas','Incentive tonnage response to price').replace(
        'OG elas','elasticity to ore grade decline').replace('Initial','Incentive').replace('Primary oge scale','Ore grade elasticity distribution mean').replace(
        'Mine CU margin elas','Mine CU elasticity to TCM').replace('Mine cost tech improvements','Mine cost reduction per year').replace(
        'Incentive','Incentive pool').replace('Mine cost price elas','Mine cost elasticity to commodity price').replace(
        'Close years back','Prior years used for price prediction').replace('Reserves ratio price lag','Price lag used for incentive pool tonnage').replace(
        'Incentive pool ore grade decline','Incentive pool ore grade decline per year').replace('response','elasticity').replace('Tcrc','TCRC').replace('tcrc','TCRC').replace(
        'sd','SD').replace(' elas ',' elasticity to ').replace('Direct melt','Direct melt fraction').replace('CU price elas','CU elasticity to price').replace(
        'ratio TCRC elas','ratio elasticity to TCRC').replace('ratio scrap spread elas','ratio elasticity to scrap spread').replace(
        'Refinery capacity fraction increase mining','Ref. cap. growth frac. from mine prod. growth').replace('Pri ','Primary refinery ').replace(
        'Sec CU','Secondary refinery CU').replace('Sec ratio','Refinery SR').replace('primary commodity ','').replace('Primary commodity','Refined')
                                       for i in ind]
    return dict(zip(ind,updated))

def prep_for_snsplots(self,mining=False,percentile=25,n_most_important=4):
    if mining:
        df = self.mining.rmse_df_sorted.copy()
        most_important = self.mining.importances_df['Mean no dummies'].sort_values(ascending=False).head(n_most_important).index
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
    if mining:
        return df, demand_params, most_important
    else:
        return df, demand_params

def plot_demand_parameter_correlation(self,scatter=True, percentile=25, dpi=50):
    df, demand_params = prep_for_snsplots(self,mining=False,percentile=percentile)
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

def plot_mining_parameter_scatter(self, percentile=25, n=None, dpi=50, n_most_important=4, scale_y_for_legend=1,
                                  plot_median=True, legend=True, best_or_median='mean'):
    if n!=None: percentile=n/600*100
    df2, demand_params, order = prep_for_snsplots(self,mining=True, percentile=percentile, n_most_important=n_most_important)
    df2 = df2.set_index(['Commodity','Scenario number']).stack().reset_index(drop=False).rename(columns={'level_2':'Parameter',0:'Value'})
    df2a = df2.copy().loc[df2['Parameter']!='Mine cost reduction per year']

    def replace_for_mining(string):
        return string.replace(' to T','\nto T').replace('y d','y\nd').replace('ing pr','ing\npr').replace('ge e','ge\ne')
    for i in demand_params:
        df2a.replace(i,replace_for_mining(i),inplace=True)
    df2b = df2.copy().loc[df2['Parameter']=='Mine cost reduction per year']
    if best_or_median=='median':
        df2a_means = df2a.groupby(['Commodity','Parameter']).median().reset_index(drop=False)
        df2b_means = df2b.groupby(['Commodity','Parameter']).median().reset_index(drop=False)
    else:
        df2a_means = df2a.loc[df2a['Scenario number']==0]
        df2b_means = df2b.loc[df2b['Scenario number']==0]

    fig,ax=easy_subplots(2,width_scale=1.3+0.1*n_most_important/4,width_ratios=[n_most_important-1,1],dpi=dpi)
    a=ax[0]
    # sns.violinplot(data=df2a, x='Parameter', y='Value', hue='Commodity',ax=a, linewidth=2)
    order_rename = make_parameter_names_nice(order)
    order = [order_rename[i] for i in order]
    order = [replace_for_mining(i) for i in order if 'Mine cost reduction per year'!=i]
    linewidth = 0.5

    sns.stripplot(data=df2a, x='Parameter', y='Value', hue='Commodity',ax=a, dodge=True, size=10,
                 order=order, edgecolor='w', linewidth=linewidth)
    if plot_median:
        marker='s'
        markersize=12
        alpha=0.3
        sns.stripplot(data=df2a_means, x='Parameter', y='Value', hue='Commodity',ax=a, dodge=True, size=markersize, color='k', alpha=alpha, marker=marker,
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

    b=ax[1]
    sns.stripplot(data=df2b, x='Parameter', y='Value', hue='Commodity',ax=b, dodge=True, size=10, linewidth=linewidth, edgecolor='w')
    if plot_median:
        sns.stripplot(data=df2b_means, x='Parameter', y='Value', hue='Commodity',ax=b, dodge=True, size=markersize, color='k', alpha=alpha, marker=marker)
    b.legend('')
    alim = a.get_ylim()
    b.set(ylim=[alim[0]*5,alim[1]*5], ylabel=None, xlabel=None)
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

def get_unit(simulated, historical, param):
    if 'price' in param.lower():
        unit = 'USD/t'
    else:
        min_simulated = abs(simulated).min() if len(simulated.shape)<=1 else abs(simulated).min().min()
        max_simulated = abs(simulated).max() if len(simulated.shape)<=1 else abs(simulated).max().max()
        mean_simulated = abs(simulated).mean() if len(simulated.shape)<=1 else abs(simulated).mean().mean()
        if np.mean([historical.mean(),mean_simulated])>1000:
            historical /= 1000
            simulated /= 1000
            unit = 'Mt'
        elif np.mean([historical.mean(),mean_simulated])<1:
            historical *= 1000
            simulated *= 1000
            unit = 't'
        else:
            unit = 'kt'
    return {'simulated':simulated, 'historical':historical, 'unit':unit}
