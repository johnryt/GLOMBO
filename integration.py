from mining_class import *
from demand_class import *
from refining_class import *
import numpy as np

class Integration():
    '''
    
    '''
    def __init__(self,simulation_time=np.arange(2019,2041),verbosity=0,byproduct=False,input_hyperparam=0):
        self.version = '2022-05-30 09:12:42' # str(datetime.now())[:19]
        self.i = simulation_time[0]
        self.simulation_time = simulation_time
        self.verbosity = verbosity
        self.byproduct = byproduct
        self.input_hyperparam = input_hyperparam
        
        self.demand = demandModel(simulation_time=simulation_time, verbosity=verbosity)
        self.refine = refiningModel(simulation_time=simulation_time, verbosity=verbosity)
        self.initialize_hyperparam()
        self.hyperparam.loc['simulation_time',:] = np.array([simulation_time,'simulation time from model initialization'],dtype=object)
        self.update_hyperparam()
        
        self.concentrate_supply = pd.Series(np.nan,simulation_time)
        self.concentrate_demand = pd.DataFrame(np.nan,simulation_time,['Global','China','RoW'])
        self.refined_supply = self.concentrate_demand.copy()
        self.refined_demand = self.concentrate_demand.copy()
        self.scrap_supply = self.concentrate_demand.copy()
        self.scrap_demand = self.concentrate_demand.copy()
        self.direct_melt_demand = self.concentrate_demand.copy()
        self.direct_melt_fraction = self.concentrate_demand.copy()
        self.total_demand = self.concentrate_demand.copy()
        
        self.scrap_spread = self.concentrate_demand.copy()
        self.primary_commodity_price = self.concentrate_supply.copy()
        self.tcrc = self.concentrate_supply.copy()
    
    def initialize_hyperparam(self):
        hyperparameters = pd.DataFrame(np.nan,index=[],columns=['Value','Notes'])
        
        # refining and demand
        hyperparameters.loc['refining and demand',:] = np.nan
        hyperparameters.loc['initial_demand',:] = 1,'initial overall demand'
        hyperparameters.loc['Recycling input rate, Global',['Value','Notes']] = 0.4, 'float, fraction of demand in the initial year satisfied by recycled inputs; includes refined and direct melt'
        hyperparameters.loc['Recycling input rate, China',['Value','Notes']] = 0.4, 'float, fraction of demand in the initial year satisfied by recycled inputs; includes refined and direct melt'
        hyperparameters.loc['china_fraction_demand',['Value','Notes']] = 0.7, 'China fraction of demand, was 0.52645 for copper in 2019'
        hyperparameters.loc['scrap_to_cathode_eff',:] = 0.99,'Efficiency of remelting and refining scrap'

        # refining only
        hyperparameters.loc['refining only',:] = np.nan
        hyperparameters.loc['pri CU TCRC elas',:] = 0.1, 'Capacity utilization at primary-only refineries'
        hyperparameters.loc['sec CU TCRC elas',:] = 0.1, 'Capacity utilization at secondary-consuming refineries'
        hyperparameters.loc['sec ratio TCRC elas',:] = -0.4, 'Secondary ratio elasticity to TCRC'
        hyperparameters.loc['sec ratio scrap spread elas',:] = 0.8, 'Secondary ratio elasticity to scrap spread'
        hyperparameters.loc['Use regions'] = False, 'True makes it so global refining is determined from combo of China and RoW; False means we ignore the region level'
        
        # price
        hyperparameters.loc['price',:] = np.nan
        hyperparameters.loc['primary_commodity_price',:] = 6000,'primary commodity price'
        hyperparameters.loc['initial_scrap_spread',:] = 500,'scrap spread at simulation start'
        
        # price elasticities
        hyperparameters.loc['price elasticities',:] = np.nan
        hyperparameters.loc['primary_commodity_price_elas_sd',:] = -0.4, 'primary commodity price elasticity to supply-demand imbalance (here using S/D ratio)'
        hyperparameters.loc['tcrc_elas_sd',:] = 0.7, 'TCRC elasticity to supply-demand imbalance (here using S/D ratio)'
        hyperparameters.loc['tcrc_elas_price',:] = 0.7, 'TCRC elasticity to primary commodity price change'
        hyperparameters.loc['scrap_spread_elas_sd',:] = 0.4, 'scrap spread elasticity to supply-demand imbalance (here using S/D ratio)'
        hyperparameters.loc['scrap_spread_elas_primary_commodity_price',:] = 0.4, 'scrap spread elasticity to primary commodity price, using ratio year-over-year'
        hyperparameters.loc['direct_melt_elas_scrap_spread',:] = 0.4, 'direct melt demand elasticity to scrap spread'
        hyperparameters.loc['collection_elas_scrap_price',:] = 0.6,'% increase in collection corresponding with a 1% increase in scrap price'

        # determining model structure
        hyperparameters.loc['determining model structure',:] = np.nan
        hyperparameters.loc['scrap_trade_simplistic',:] = True,'if True, sets scrap supply in each region equal to the demand scaled by the global supply/demand ratio. If False, we should set up a price-informed trade algorithm, but have not done that yet so this should stay True for now'
        hyperparameters.loc['refinery_follows_concentrate_supply',:] = False, 'True causes refinery capacity evolution to follow concentrate supply; False causes refinery capacity evolution to follow refined demand'
        hyperparameters.loc['presimulate_mining',:] = True, 'True means we simulate mining production for presimulate_n_years before the simulation starts, theoretically to force alignment with simulation start time mine production'
        hyperparameters.loc['presimulate_n_years',:] = 10, 'int, number of years in the past to simulate mining to establish baseline'
        
        # mining only
        hyperparameters.loc['mining only',:] = np.nan
        hyperparameters.loc['primary_ore_grade_mean',:] = 0.1,'Ore grade mean for lognormal distribution'
        hyperparameters.loc['use_ml_to_accelerate','Value'] = True
        hyperparameters.loc['ml_accelerate_initialize_years','Value'] = 20
        hyperparameters.loc['ml_accelerate_every_n_years','Value'] = 5
        hyperparameters.loc['incentive_tuning_option','Value'] = 'pid-2019'
        hyperparameters.loc['internal_price_formation','Value'] = False
        hyperparameters.loc['primary_production_mean','Value'] = 0.001
        hyperparameters.loc['primary_production_var','Value'] = 0.5
        hyperparameters.loc['primary_ore_grade_var','Value'] = 1
        hyperparameters.loc['incentive_opening_probability','Value'] = 0.05
        hyperparameters.loc['incentive_require_tune_years',:] = 5,'requires incentive tuning for however many years such that supply=demand, with no requirements on incentive_opening_probability and allowing the given incentive_opening_probability to be used'
        hyperparameters.loc['demand_series_method','Value'] = 'none'
        hyperparameters.loc['end_calibrate_years','Value'] = 20
        hyperparameters.loc['start_calibrate_years','Value'] = 10
        hyperparameters.loc['ramp_up_cu','Value'] = 0.7
        hyperparameters.loc['ml_accelerate_initialize_years','Value'] = max(hyperparameters['Value'][['ml_accelerate_initialize_years','end_calibrate_years']])
        hyperparameters.loc['mine_cu_margin_elas','Value'] = 0.8
        
        self.hyperparam = hyperparameters.copy()
        self.h = hyperparameters.copy()['Value']
        
    def update_hyperparam(self):
        update = 0
        if type(self.input_hyperparam) == pd.core.frame.DataFrame:
            if 'Value' in self.input_hyperparam.columns:
                update = self.input_hyperparam.copy()['Value']
            elif self.input_hyperparam.shape[1]==1:
                update = self.input_hyperparam.copy().iloc[:,0]
        elif type(self.input_hyperparam) == pd.core.series.Series:
            update = self.input_hyperparam.copy()
        if type(update)!=int:
            for ind in update.index:
                self.hyperparam.loc[ind,'Value'] = update[ind]
        
    def hyperparam_agreement(self):
        dem = self.demand
        ref = self.refine
        h = self.h
        
        dem.hyperparam.loc['initial_demand','Value'] = h['initial_demand']
        ref.hyperparam.loc['Total production','Global'] = h['initial_demand'] # will need updated later

        dem.hyperparam.loc['china_fraction_demand','Value'] = h['china_fraction_demand']
        ref.hyperparam.loc['Regional production fraction of total production','China'] = h['china_fraction_demand']
        
        ref.hyperparam.loc['Recycling input rate','Global'] = h['Recycling input rate, Global']
        ref.hyperparam.loc['Recycling input rate','China'] = h['Recycling input rate, China']
        dem.hyperparam.loc['recycling_input_rate_china','Value'] = h['Recycling input rate, China']
        dem.hyperparam.loc['recycling_input_rate_row','Value'] = (h['Recycling input rate, Global']-h['Recycling input rate, China']*h['china_fraction_demand'])/(1-h['china_fraction_demand'])
        
        dem.hyperparam.loc['scrap_to_cathode_eff','Value'] = h['scrap_to_cathode_eff']
        ref.hyperparam.loc['scrap_to_cathode_eff',:] = h['scrap_to_cathode_eff']
        
        if self.verbosity>1: print('Refinery parameters updated:')
        for param in np.intersect1d(ref.hyperparam.index, h.index):
            ref.hyperparam.loc[param,:] = h[param]
            if self.verbosity>1:
                print('   ref',param,'now',h[param])
        
        h_index_split = [j for j in h.index if ', ' in j]
        h_index_ind = np.unique([j.split(', ')[0] for j in h_index_split])
        for param in np.intersect1d(ref.hyperparam.index,h_index_ind):
            regions = np.unique([j.split(', ')[1] for j in h_index_split if param in j])
            for reg in regions:
                ref.hyperparam.loc[param,reg] = h[', '.join([param,reg])]
                if self.verbosity>1:
                    print('   ref',param,reg,'now',h[', '.join([param,reg])])
        
        for param in np.intersect1d(ref.ref_hyper_param.index, h.index):
            ref.ref_hyper_param.loc[param,:] = h[param]
            if self.verbosity>1:
                print('   rhp',param)
            
        self.demand = dem
        self.refine = ref
    
    def initialize_integration(self):
        i = self.i
        self.conc_to_cathode_eff = self.refine.ref_param['Global']['conc to cathode eff']
        self.scrap_to_cathode_eff = self.refine.ref_param['Global']['scrap to cathode eff']
        
        # initializing demand
        self.hyperparam_agreement()
        self.demand.run()
        self.collection_rate = self.demand.collection_rate.copy()
        
        # initializing refining
        self.refine.run()
        
        # total_demand
        self.total_demand = pd.concat([self.demand.demand.sum(axis=1),
                                       self.demand.demand['China'].sum(axis=1),
                                       self.demand.demand['RoW'].sum(axis=1)],
                                      axis=1,keys=['Global','China','RoW'])
        self.total_demand.loc[self.i+1:] = np.nan

        # refined_demand, direct_melt_demand
        self.direct_melt_demand = self.total_demand.apply(lambda x: x*self.refine.hyperparam.loc['Direct melt fraction of production'],axis=1)
        self.direct_melt_demand.loc[i+1:] = np.nan
        self.refined_demand = self.total_demand - self.direct_melt_demand
        self.direct_melt_demand = self.direct_melt_demand.apply(lambda x: x/self.scrap_to_cathode_eff,axis=1)
        
        # concentrate_demand, secondary_refined_demand, refined_supply
        self.concentrate_demand = self.refine.ref_stats.loc[:,idx[:,'Primary production']].droplevel(1,axis=1)
        self.concentrate_demand.loc[i+1:] = np.nan
        self.secondary_refined_demand = self.refine.ref_stats.loc[:,idx[:,'Secondary production']].droplevel(1,axis=1)
        self.secondary_refined_demand.loc[i+1:] = np.nan
        self.refined_supply = self.concentrate_demand + self.secondary_refined_demand
        self.concentrate_demand = self.concentrate_demand.apply(lambda x: x/self.conc_to_cathode_eff,axis=1)
        self.secondary_refined_demand = self.secondary_refined_demand.apply(lambda x: x/self.scrap_to_cathode_eff,axis=1)

        # concentrate_supply,    initializing mining
        if self.h['presimulate_mining']:
            self.presimulate_mining()
        else:
            self.mining = miningModel(simulation_time=self.simulation_time, verbosity=self.verbosity, byproduct=self.byproduct)
            self.mining.hyperparam.loc['primary_production','Value'] = self.concentrate_demand['Global'][i]
            if self.verbosity>1: print('Mining parameters updated:')
            for param in np.intersect1d(self.mining.hyperparam.index, self.h.index):
                self.mining.hyperparam.loc[param,'Value'] = self.h[param]
                if self.verbosity>1:
                    print('  ',param)
            self.mining.run()
        self.concentrate_supply = self.mining.supply_series.copy()
        
        # scrap_supply, scrap_demand
        self.scrap_supply = self.demand.scrap_supply.copy()
        self.scrap_demand = self.secondary_refined_demand+self.direct_melt_demand
        
    def initialize_price(self):
        i = self.i
        h = self.h
        self.scrap_spread.loc[i] = h['initial_scrap_spread']
        self.primary_commodity_price.loc[i] = h['primary_commodity_price']
        self.tcrc.loc[i] = self.mining.primary_tcrc_series[i]
        
    def price_evolution(self):
        i = self.i
        h = self.h
        # refined
        self.primary_commodity_price.loc[i] = self.primary_commodity_price.loc[i-1]*(self.refined_supply.replace(0,1e-6)['Global'][i-1]/self.refined_demand.replace(0,1e-6)['Global'][i-1])**h['primary_commodity_price_elas_sd']
        
        # tcrc
        self.tcrc.loc[i] = self.tcrc.loc[i-1]*(self.concentrate_supply.replace(0,1e-6)[i-1]/self.concentrate_demand.replace(0,1e-6)['Global'][i-1])**h['tcrc_elas_sd']\
                                *(self.primary_commodity_price.loc[i]/self.primary_commodity_price.loc[i-1])**h['tcrc_elas_price']
        
        # scrap trade
        if h['scrap_trade_simplistic'] and i>self.simulation_time[2]:
            self.scrap_supply.loc[i-1,'China'] = self.scrap_supply['China'][i-2]*self.scrap_supply['Global'][i-1]/self.scrap_supply['Global'][i-2]
            self.scrap_supply.loc[i-1,'RoW'] = self.scrap_supply['RoW'][i-2]*self.scrap_supply['Global'][i-1]/self.scrap_supply['Global'][i-2]                
        elif h['scrap_trade_simplistic']:
            self.scrap_supply.loc[i-1,'China'] = self.scrap_supply['Global'][i-1]*self.scrap_demand['China'][i-1]/self.scrap_demand['Global'][i-1]
            self.scrap_supply.loc[i-1,'RoW'] = self.scrap_supply['Global'][i-1]*self.scrap_demand['RoW'][i-1]/self.scrap_demand['Global'][i-1]
            
        # scrap spread
        self.scrap_spread.loc[i,'China'] = self.scrap_spread['China'][i-1]*(self.scrap_supply['China'][i-1]/self.scrap_demand['China'][i-1])**h['scrap_spread_elas_sd']\
            * (self.primary_commodity_price[i]/self.primary_commodity_price[i-1])**h['scrap_spread_elas_primary_commodity_price']
        self.scrap_spread.loc[i,'RoW'] = self.scrap_spread['RoW'][i-1]*(self.scrap_supply['RoW'][i-1]/self.scrap_demand['RoW'][i-1])**h['scrap_spread_elas_sd']\
            * (self.primary_commodity_price[i]/self.primary_commodity_price[i-1])**h['scrap_spread_elas_primary_commodity_price']
        self.scrap_spread.loc[i,'Global'] = self.scrap_spread['Global'][i-1]*(self.scrap_supply['Global'][i-1]/self.scrap_demand['Global'][i-1])**h['scrap_spread_elas_sd']\
            * (self.primary_commodity_price[i]/self.primary_commodity_price[i-1])**h['scrap_spread_elas_primary_commodity_price']
        
    def run_demand(self):
        self.demand.i = self.i
        i = self.i
        h = self.h.copy()
        if self.i-2 not in self.primary_commodity_price.index:
            self.primary_commodity_price.loc[self.i-2] = self.primary_commodity_price[self.i-1]
        if self.i-2 not in self.scrap_spread.index:
            self.scrap_spread.loc[i-2] = self.scrap_spread.loc[i-1]
        self.demand.commodity_price_series = self.primary_commodity_price.copy()
        
#                 hyperparameters.loc['collection_elas_scrap_price',:] = 0.6,'% increase in collection corresponding with a 1% increase in scrap price'
        scrap_price_l1 = self.primary_commodity_price[i-1]-self.scrap_spread.loc[i-1]
        scrap_price_l2 = self.primary_commodity_price[i-2]-self.scrap_spread.loc[i-2]
        scrap_price_l1[scrap_price_l1<0] = 1e-6
        scrap_price_l2[scrap_price_l2<0] = 1e-6
        if h['Use regions']:
            self.collection_rate.loc[idx[i,'China'],:] = self.collection_rate.loc[idx[i-1,'China'],:].rename({i-1:i})*(scrap_price_l1['China']/scrap_price_l2['China'])**h['collection_elas_scrap_price']
            self.collection_rate.loc[idx[i,'RoW'],:]   = self.collection_rate.loc[idx[i-1,'RoW'],:].rename({i-1:i})*(scrap_price_l1['RoW']/scrap_price_l2['RoW'])**h['collection_elas_scrap_price']
        else:
            self.collection_rate.loc[idx[i,:],:] = self.collection_rate.loc[idx[i-1,:],:].rename({i-1:i})*(scrap_price_l1['Global']/scrap_price_l2['Global'])**h['collection_elas_scrap_price']
        max_cr = self.demand.hyperparam['Value']['maximum_collection_rate']
        self.collection_rate[self.collection_rate>max_cr] = max_cr
        self.demand.collection_rate = self.collection_rate.copy()
        self.demand.run()
        
    def run_refine(self):
        self.refine.i = self.i
        self.refine.tcrc_series = self.tcrc.copy()
        self.refine.scrap_spread_series = self.scrap_spread.copy()
        if self.i-2 not in self.concentrate_supply.index:
            self.concentrate_supply.loc[self.i-2] = self.concentrate_supply[self.i-1]*self.refined_demand['Global'][self.i-2]/self.refined_demand['Global'][self.i-1]
        if self.i-2 not in self.scrap_supply.index:
            self.scrap_supply.loc[self.i-2] = self.scrap_supply.loc[self.i-1]*self.refined_demand.loc[self.i-2]/self.refined_demand.loc[self.i-1]
        if self.h['refinery_follows_concentrate_supply']:
            self.refine.pri_cap_growth_series = self.concentrate_supply.copy()
        else:
            self.refine.pri_cap_growth_series = self.refined_demand['Global'].copy()
        self.refine.sec_cap_growth_series = self.scrap_supply.copy()
        self.refine.run()
        
    def run_mining(self):
        self.mining.i = self.i
        self.mining.primary_price_series.loc[self.i] = self.primary_commodity_price[self.i]
        self.mining.primary_tcrc_series.loc[self.i] = self.tcrc[self.i]
        self.mining.demand_series.loc[self.i] = self.concentrate_demand['Global'][self.i]
        self.mining.run()
        # concentrate_supply
        self.concentrate_supply.loc[self.i] = self.mining.supply_series[self.i]
    
    def presimulate_mining(self):
        h = self.h.copy()
        end_yr = self.simulation_time[0]
        hist_simulation_time = np.arange(end_yr-h['presimulate_n_years'],end_yr+1)
        mine_simulation_time = np.arange(end_yr-h['presimulate_n_years'],self.simulation_time[-1]+1)
        self.mining = miningModel(simulation_time=mine_simulation_time, byproduct=self.byproduct,verbosity=self.verbosity)
        m = self.mining
        if self.verbosity>1: print('Mining parameters updated:')
        for param in np.intersect1d(m.hyperparam.index, h.index):
            m.hyperparam.loc[param,'Value'] = h[param]
            if self.verbosity>1:
                print('  ',param,'now',h[param])
        
        initial_ore_grade_decline = m.hyperparam['Value']['initial_ore_grade_decline']
        incentive_mine_cost_improvement = m.hyperparam['Value']['incentive_mine_cost_improvement']
        annual_reserves_ratio_with_initial_production_slope = m.hyperparam['Value']['annual_reserves_ratio_with_initial_production_slope']
        m.hyperparam.loc['internal_price_formation','Value'] = True
        m.hyperparam.loc['initial_ore_grade_decline','Value'] = 0
        m.hyperparam.loc['incentive_mine_cost_improvement','Value'] = 0
        m.hyperparam.loc['annual_reserves_ratio_with_initial_production_slope','Value'] = 0
        
#         primary_production
        m.demand_series = self.demand.alt_demand[[j for j in self.demand.alt_demand.columns if j!='China Fraction']].sum(axis=1).rolling(5).mean()
        m.demand_series *= self.concentrate_demand['Global'][end_yr]/m.demand_series[end_yr]
        m.hyperparam.loc['primary_production','Value'] = m.demand_series[mine_simulation_time[0]]
        primary_production_mean_series = m.demand_series*h['primary_production_mean']/m.demand_series[end_yr]
        self.primary_production_mean_series = primary_production_mean_series.copy()
#         display(m.demand_series)
        for year in hist_simulation_time:
            m.i = year
            if self.verbosity>1:
                print('sim mine history year:',year)
            m.hyperparam.loc['primary_production_mean','Value'] = primary_production_mean_series[year]
            m.run()
#             display(m.demand_series)
#             raise ValueError('no')

            if self.verbosity>4:
                fig,ax=easy_subplots(2,2)
                ax[0].plot(m.demand_series,label='Demand',marker='o')
                ax[0].plot(m.supply_series,label='Supply',marker='o')
                ax[0].legend()
                ax[1].plot(m.primary_tcrc_series,marker='o')
                plt.show()
        
        m.hyperparam.loc['initial_ore_grade_decline','Value'] = initial_ore_grade_decline
        m.hyperparam.loc['incentive_mine_cost_improvement','Value'] = incentive_mine_cost_improvement
        m.hyperparam.loc['annual_reserves_ratio_with_initial_production_slope','Value'] = annual_reserves_ratio_with_initial_production_slope
        m.hyperparam.loc['internal_price_formation','Value'] = False
        self.concentrate_supply = m.supply_series.copy()
        self.tcrc = m.primary_tcrc_series.copy()
        self.mining = m
                
    def calculate_direct_melt_demand(self):
        i = self.i
        self.direct_melt_fraction.loc[i-1] = self.direct_melt_demand.loc[i-1]/self.total_demand.loc[i-1]
        if False:
            self.direct_melt_demand.loc[i] = self.direct_melt_demand.loc[i-1]*self.scrap_supply.loc[i]/self.scrap_supply.loc[i-1]\
                * (self.scrap_spread.loc[i]/self.scrap_spread.loc[i-1])**self.h['direct_melt_elas_scrap_spread']
        elif False:
            direct_melt_fraction = self.refine.hyperparam.loc['Direct melt fraction of production']\
                * (self.scrap_spread.loc[i]/self.scrap_spread.loc[i-1])**self.h['direct_melt_elas_scrap_spread']
        else:
            direct_melt_fraction = self.direct_melt_fraction.loc[i-1]\
                * (self.scrap_spread.loc[i]/self.scrap_spread.loc[i-1])**self.h['direct_melt_elas_scrap_spread']
            if (direct_melt_fraction > 1).any(): direct_melt_fraction[direct_melt_fraction>1] = 1
            if (direct_melt_fraction < 0).any(): direct_melt_fraction[direct_melt_fraction<0] = 0
            self.direct_melt_demand.loc[i] = self.total_demand.loc[i]*direct_melt_fraction
    
    def update_integration_variables(self):
        i = self.i
        
        # scrap supply
        self.scrap_supply = self.demand.scrap_supply.copy()
        
        # total_demand
        self.total_demand.loc[i,'Global'] = self.demand.demand.sum(axis=1)[i]
        self.total_demand.loc[i,'China'] = self.demand.demand['China'].sum(axis=1)[i]
        self.total_demand.loc[i,'RoW'] = self.demand.demand['RoW'].sum(axis=1)[i]
        self.total_demand[self.total_demand<0] = 0
        
        # refined_demand, direct_melt_demand
        self.calculate_direct_melt_demand()
        self.refined_demand.loc[i] = self.total_demand.loc[i] - self.direct_melt_demand.loc[i]
        self.refined_demand[self.refined_demand<0] = 0
        self.direct_melt_demand.loc[i] /= self.scrap_to_cathode_eff
        
        # concentrate_demand, secondary_refined_demand, refined_supply
        self.concentrate_demand.loc[i] = self.refine.ref_stats.loc[i,idx[:,'Primary production']].droplevel(1)
        self.concentrate_demand[self.concentrate_demand<0] = 0
        self.secondary_refined_demand.loc[i] = self.refine.ref_stats.loc[i,idx[:,'Secondary production']].droplevel(1)
        self.secondary_refined_demand[self.secondary_refined_demand<0] = 0
        self.refined_supply.loc[i] = self.concentrate_demand.loc[i] + self.secondary_refined_demand.loc[i]
        self.concentrate_demand.loc[i] /= self.conc_to_cathode_eff
        self.secondary_refined_demand.loc[i] /= self.scrap_to_cathode_eff

        # scrap_demand
        self.scrap_demand.loc[i] = self.secondary_refined_demand.loc[i]+self.direct_melt_demand.loc[i]
        
    def run(self):
        self.h = self.hyperparam['Value'].copy()
        for year in self.simulation_time:
            if self.verbosity>0: print(year)
            self.i = year
            if self.i==self.simulation_time[0]:
                self.initialize_integration()
                self.initialize_price()
            else:
                self.price_evolution()
                self.run_demand()
                self.run_refine()
                self.update_integration_variables()
                self.run_mining()
        self.concentrate_demand = pd.concat([self.mining.demand_series,self.concentrate_demand['China'],
                                            self.concentrate_demand['RoW']],axis=1,keys=['Global','China','RoW'])
