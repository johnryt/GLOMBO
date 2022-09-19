import warnings
from integration_functions import Sensitivity
import numpy as np
import pandas as pd
from useful_functions import *
from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor

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

    def run_all_integration(self, n_runs=200, commodities=None):
        commodities = self.ready_commodities if commodities==None else commodities
        for material in commodities:
            print('-'*40)
            print(material)
            with warnings.catch_warnings():
                # timer=IterTimer()
                warnings.simplefilter('error')
                n=3
                thing = '_'+str(n)+'p' if n>1 else ''
                mat = self.element_commodity_map[material].lower()
                filename=f'{self.pkl_folder}/{mat}_run_hist_all{thing}.pkl'
                print('--'*15+filename+'-'*15)
                s = Sensitivity(pkl_filename=filename, data_folder=self.data_folder,changing_base_parameters_series=material,notes=f'Monte Carlo {material} run',
                                additional_base_parameters=pd.Series(1,['refinery_capacity_growth_lag']),
                                simulation_time=np.arange(2001,2020), include_sd_objectives=False,
                                OVERWRITE=True,verbosity=0,historical_price_rolling_window=5,
                                constrain_previously_tuned=True)
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
                    'mine_cu_margin_elas',
                    'mine_cost_tech_improvements',
                    'mine_cost_price_elas',
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
        '''
        for df_name in ['rmse_df','hyperparam','simulated_demand','results','historical_data','mine_supply']:
            df_outer = pd.DataFrame()
            df_outer_sorted = pd.DataFrame()


        for material in self.ready_commodities:
            material = self.element_commodity_map[material].lower()
            if demand_mining_all=='demand':
                indiv = Individual(filename=f'data/{material}_run_hist_DEM.pkl',rmse_not_mae=False,dpi=50)
            elif demand_mining_all=='mining':
                indiv = Individual(filename=f'data/{material}_run_hist_mining.pkl',rmse_not_mae=False,dpi=50)
            elif demand_mining_all=='all':
                indiv = Individual(filename=f'data/{material}_run_hist.pkl',rmse_not_mae=False,dpi=50)
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
                        sorted_cols = self.rmse_df.loc[material,'RMSE'].sort_values().index
                        df_ph_sorted = df_ph.copy().loc[:,sorted_cols]
                        df_ph_sorted = df_ph_sorted.T.reset_index(drop=True).T
                    elif df_name=='rmse_df':
                        df_ph_sorted = df_ph.copy().sort_values(by=(material,'RMSE'),axis=1)
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


    def plot_all_demand(self, dpi=50):
        for material in self.ready_commodities:
            material = self.element_commodity_map[material].lower()
            filename=f'{self.pkl_folder}/{material}_run_hist_DEM.pkl'
            indiv = Individual(filename=filename,rmse_not_mae=False,dpi=dpi)
            indiv.plot_demand_results()

    def plot_all_mining(self,dpi=50):
        for material in self.ready_commodities:
            material = self.element_commodity_map[material].lower()
            filename=f'{self.pkl_folder}/{material}_run_hist_mining.pkl'
            indiv = Individual(filename=filename,rmse_not_mae=False,dpi=dpi)
            indiv.plot_demand_results()

    def feature_importance(self,dpi=50):

        split_frac = 0.5

        plot_train_test=False
        plot_feature_importances=True

        rmse_df = self.rmse_df.copy().astype(float)
        if not hasattr(self,'importances_df'):
            importances_df = pd.DataFrame()
            for Regressor, name in zip([RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor],['RandomForest','ExtrTrees','GradientBoosting']):
                if plot_train_test: fig2,ax2 = easy_subplots(2,dpi=dpi)
                for e,dummies in enumerate([True,False]):
                    X_df = rmse_df.copy()
                    for r in ['R2','RMSE','region_specific_price_response','score']:
                        if r in X_df.index.get_level_values(1):
                            X_df = X_df.drop(r,level=1)
                    X_df = X_df.unstack().stack(level=0)
                    y_df = np.log(rmse_df.loc[idx[:,'RMSE'],:].unstack().stack(level=0))

                    if dummies:
                        X_df.loc[:,'commodity']=X_df.index.get_level_values(0)
                        X_df = pd.get_dummies(X_df,columns=['commodity'])
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

                    importances = pd.Series(regr.feature_importances_, X_df.columns).drop([i for i in X_df.columns if 'commodity' in i]).sort_values(ascending=False)
                    importances.name =  name + (' w/ dummies' if dummies else ' no dummies')
                    if dummies: importances /= importances.sum()
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

        dummy_cols = [i for i in importances_df.columns if 'w/ dummies' in i]
        no_dummy_cols=[i for i in importances_df.columns if 'no dummies' in i]

        if plot_feature_importances:
            fig1,ax1 = easy_subplots(2, height_scale=1.5, width_scale=9/len(self.importances_df.columns), dpi=dpi)
            to_plot_du = self.importances_df.loc[:,dummy_cols].sort_values(by='Mean w/ dummies',ascending=False)
            to_plot_du.rename(columns=dict(zip(dummy_cols,[i.split(' w/ dummies')[0] for i in dummy_cols])),inplace=True)
            to_plot_du.plot.bar(ax=ax1[0],ylabel='Feature importance').grid(axis='x')
            to_plot_no = self.importances_df.loc[:,no_dummy_cols].sort_values(by='Mean no dummies',ascending=False)
            to_plot_no.rename(columns=dict(zip(no_dummy_cols,[i.split(' no dummies')[0] for i in no_dummy_cols])),inplace=True)
            to_plot_no.plot.bar(ax=ax1[1],ylabel='Feature importance').grid(axis='x')
            ax1[0].set_title('With dummies',weight='bold')
            ax1[1].set_title('No dummies',weight='bold')
