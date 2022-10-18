from mining_class import *
from demand_class import *
from refining_class import *
import numpy as np
from warnings import warn

class Integration():
    '''
    scenario_name takes the form 00_11_22_33_44
        where:
        00: ss, sd, bo (scrap supply, scrap demand, both).
         Can also be sd-alt, which uses an alternative
         implementation of the scrap demand increase (see below)
        11: pr or no (price response included or no)
        22: Xyr, where X is any integer and represents
         the number of years the increase occurs
        33: X%tot, where X is any float/int and is the
         increase/decrease in scrap supply or demand
         relative to the initial year total demand
        44: X%inc, where X is any float/int and is the
         increase/decrease in the %tot value per year

        e.g. ss_pr_1yr_1%tot_0%inc

        for 22-44, an additional X should be placed at
         the end when 00==both, describing the sd values
         e.g. ss_pr_1yr1_1%tot1_0%inc0

        Can also have 11 as nono, prno, nopr, or prpr to
         control the ss and sd price response individually
         (in that order)

        for sd-alt, the default (no alt) formulation is that an
        increase in scrap demand to 5% of demand over 2
        years would be
        [0, 2.5%, 2.5%, 0%, 0%, ..., 0%]
        while for alt it would be
        [0, 2.5%, 5%,   5%, 5%, ..., 5%]
        Can ctrl+F `direct_melt_duration` or `secondary_refined_duration`
        to see the actual methods
    '''
    def __init__(self, data_folder=None, simulation_time=np.arange(2019,2041), verbosity=0, byproduct=False, input_hyperparam=0, scenario_name='', commodity=None, price_to_use=None):
        self.version = '2022-09-19 18:03:44' # str(datetime.now())[:19]

        self.price_to_use = 'log' if price_to_use==None else price_to_use
        self.element_commodity_map = {'Al':'Aluminum','Au':'Gold','Cu':'Copper','Steel':'Steel','Co':'Cobalt','REEs':'REEs','W':'Tungsten','Sn':'Tin','Ta':'Tantalum','Ni':'Nickel','Ag':'Silver','Zn':'Zinc','Pb':'Lead','Mo':'Molybdenum','Pt':'Platinum','Te':'Telllurium','Li':'Lithium'}

        self.i = simulation_time[0]
        self.commodity = commodity
        self.data_folder = data_folder
        self.simulation_time = simulation_time
        self.verbosity = verbosity
        self.byproduct = byproduct
        self.input_hyperparam = input_hyperparam
        self.scenario_name = scenario_name
        self.demand = demandModel(data_folder=data_folder, simulation_time=simulation_time, verbosity=verbosity)
        self.refine = refiningModel(simulation_time=simulation_time, verbosity=verbosity)
        self.initialize_hyperparam()
        self.hyperparam.loc['simulation_time',:] = np.array([simulation_time,'simulation time from model initialization'],dtype=object)
        self.update_hyperparam()

        price_simulation_time = np.arange(simulation_time[0]-15,simulation_time[-1]+1)
        self.concentrate_supply = pd.Series(np.nan,price_simulation_time)
        self.sxew_supply = pd.Series(np.nan,price_simulation_time)
        self.concentrate_demand = pd.DataFrame(np.nan,price_simulation_time,['Global','China','RoW'])
        self.refined_supply = self.concentrate_demand.copy()
        self.refined_demand = self.concentrate_demand.copy()
        self.scrap_supply = self.concentrate_demand.copy()
        self.scrap_demand = self.concentrate_demand.copy()
        self.direct_melt_demand = self.concentrate_demand.copy()
        self.direct_melt_fraction = self.concentrate_demand.copy()
        self.total_demand = self.concentrate_demand.copy()
        self.primary_supply = self.concentrate_demand.copy()
        self.primary_demand = self.concentrate_demand.copy()
        self.secondary_supply = self.concentrate_demand.copy()
        self.secondary_demand = self.concentrate_demand.copy()

        self.scrap_spread = self.concentrate_demand.copy()
        self.primary_commodity_price = self.concentrate_supply.copy()
        self.tcrc = self.concentrate_supply.copy()

        self.decode_scrap_scenario_name()

    def load_historical_data(self):
        if self.commodity!=None and type(self.commodity)==str and not hasattr(self,'historical_data'):
            self.case_study_data_file_path = f'{self.data_folder}/case study data.xlsx'
            history_file = pd.read_excel(self.case_study_data_file_path,index_col=0,sheet_name=self.commodity)
            self.historical_data = history_file.iloc[1:]['Primary supply' if 'Primary supply' in history_file.columns else 'Primary production'].astype(float)
            self.historical_data = self.historical_data.dropna()
            self.historical_data.index = self.historical_data.index.astype(int)
        self.price_adjustment_results_file_path = f'{self.data_folder}/price adjustment results.xlsx'
        if self.price_to_use!='case study data' or 'Primary commodity price' not in self.historical_data.dropna().columns:
            self.historical_price_data = pd.read_excel(self.price_adjustment_results_file_path,index_col=0)
            cap_mat = self.element_commodity_map[self.commodity]
            price_map = {'log':'log('+cap_mat+')',  'diff':'∆'+cap_mat,  'original':cap_mat+' original'}
            self.historical_price_data = self.historical_price_data[price_map[self.price_to_use]].astype(float).dropna().sort_index()
            self.historical_price_data.name='Primary commodity price'
            if 'Primary commodity price' in self.historical_data.columns:
                self.historical_data.drop('Primary commodity price',axis=1,inplace=True)
            self.historical_data = pd.concat([self.historical_data,self.historical_price_data],axis=1)

    def initialize_hyperparam(self):
        hyperparameters = pd.DataFrame(np.nan,index=[],columns=['Value','Notes'])

        # refining and demand
        hyperparameters.loc['refining and demand',:] = np.nan
        hyperparameters.loc['initial_demand',:] = 1,'initial overall demand'
        hyperparameters.loc['Recycling input rate, Global',['Value','Notes']] = 0.4, 'float, fraction of demand in the initial year satisfied by recycled inputs; includes refined and direct melt'
        hyperparameters.loc['Recycling input rate, China',['Value','Notes']] = 0.4, 'float, fraction of demand in the initial year satisfied by recycled inputs; includes refined and direct melt'
        hyperparameters.loc['Secondary refinery fraction of recycled content, Global',:] = 0.6,'float, fraction of recycled content demanded by refineries, remainder is direct melt'
        hyperparameters.loc['china_fraction_demand',['Value','Notes']] = 0.7, 'China fraction of demand, was 0.52645 for copper in 2019'
        hyperparameters.loc['scrap_to_cathode_eff',:] = 0.99,'Efficiency of remelting and refining scrap'

        # refining only
        hyperparameters.loc['refining only',:] = np.nan
        hyperparameters.loc['pri CU TCRC elas',:] = 0.1, 'Capacity utilization at primary-only refineries'
        hyperparameters.loc['sec CU TCRC elas',:] = 0.1, 'Capacity utilization at secondary-consuming refineries'
        hyperparameters.loc['pri CU price elas',:] = 0.01, 'primary refinery capacity uitlization elasticity to primary commodity price'
        hyperparameters.loc['sec CU price elas',:] = 0.01, 'secondary refinery capacity uitlization elasticity to primary commodity price'
        hyperparameters.loc['sec ratio TCRC elas',:] = -0.4, 'Secondary ratio elasticity to TCRC'
        hyperparameters.loc['sec ratio scrap spread elas',:] = 0.8, 'Secondary ratio elasticity to scrap spread'
        hyperparameters.loc['Use regions'] = False, 'True makes it so global refining is determined from combo of China and RoW; False means we ignore the region level'
        hyperparameters.loc['refinery_capacity_growth_lag'] = 1,'capacity growth lag, can be 1 or 0. 1 means that we use the year_i-1 and year_i-2 years to calculated the growth associated with mining or demand for calculating capacity growth, while 0 means we use year_i and year_i-1.'

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
        hyperparameters.loc['presimulate_mining',:] = True, 'True means we simulate mining production for presimulate_n_years before the simulation starts, theoretically to force alignment with simulation start time mine production'
        hyperparameters.loc['presimulate_n_years',:] = 10, 'int, number of years in the past to simulate mining to establish baseline'
        hyperparameters.loc['collection_rate_price_response',:] = True,'bool, whether or not this parameter responds to price'
        hyperparameters.loc['direct_melt_price_response',:] = True,'bool, whether or not this parameter responds to price'
        hyperparameters.loc['refinery_capacity_fraction_increase_mining'] = 0.001,'0 means refinery capacity growth is determined by demand growth alone; 1 means refinery capacity growth is determined by mining production growth alone. Values in between allow for a weighted mixture.'

        # mining only
        hyperparameters.loc['mining only',:] = np.nan
        hyperparameters.loc['primary_ore_grade_mean',:] = 0.1,'Ore grade mean for lognormal distribution'
        hyperparameters.loc['use_ml_to_accelerate','Value'] = False
        hyperparameters.loc['ml_accelerate_initialize_years','Value'] = 20
        hyperparameters.loc['ml_accelerate_every_n_years','Value'] = 5
        hyperparameters.loc['incentive_tuning_option','Value'] = 'pid-2019'
        hyperparameters.loc['internal_price_formation','Value'] = False
        hyperparameters.loc['primary_production_mean','Value'] = 0.001
        hyperparameters.loc['primary_production_var','Value'] = 0.5
        hyperparameters.loc['primary_ore_grade_var','Value'] = 1
        hyperparameters.loc['incentive_opening_probability','Value'] = 0.05
        hyperparameters.loc['incentive_require_tune_years',:] = 10,'requires incentive tuning for however many years such that supply=demand, with no requirements on incentive_opening_probability and allowing the given incentive_opening_probability to be used'
        hyperparameters.loc['demand_series_method','Value'] = 'none'
        hyperparameters.loc['end_calibrate_years','Value'] = 10
        hyperparameters.loc['start_calibrate_years','Value'] = 5
        hyperparameters.loc['ramp_up_cu','Value'] = 0.4
        hyperparameters.loc['ml_accelerate_initialize_years','Value'] = max(hyperparameters['Value'][['ml_accelerate_initialize_years','end_calibrate_years']])
        hyperparameters.loc['mine_cu_margin_elas','Value'] = 0.8
        hyperparameters.loc['mine_cost_price_elas','Value'] = 0.125
        hyperparameters.loc['mine_cost_og_elas','Value'] = -0.113
        hyperparameters.loc['mine_cost_tech_improvements','Value'] = 0.05
        hyperparameters.loc['primary_price_resources_contained_elas','Value'] = 0.5
        hyperparameters.loc['close_price_method','Value']='max'
        hyperparameters.loc['close_years_back','Value']=3
        hyperparameters.loc['close_probability_split_max','Value']=0.3
        hyperparameters.loc['close_probability_split_mean','Value']=0.5
        hyperparameters.loc['close_probability_split_min','Value']=0.2
        hyperparameters.loc['primary_oge_scale','Value'] = 0.399365
        hyperparameters.loc['initial_ore_grade_decline','Value'] = -0.05# 'Initial ore grade for new mines, elasticity to cumulative ore treated'
        hyperparameters.loc['annual_reserves_ratio_with_initial_production_const','Value'] = 1.1
        hyperparameters.loc['primary_overhead_const','Value'] = 0
        hyperparameters.loc['ramp_up_fraction',['Value','Notes']] = np.array([0.02,'fraction of mines in the initial mine generation step that are in any of the ramp up stages (e.g. if ramp_up_year is 3 and ramp_up_fraction is 0.1, then 10% of the mines will have ramp up flag=1, 10% ramp up flag=2, etc.). Value is currently 0.02 based on an initial guess.'],dtype='object')
        hyperparameters.loc['demand_series_method',['Value','Notes']] = np.array(['','This is for setting up the so-called demand series, which is what the mining module tries to tune mine production to match when historical presimulation is done. This value is normally either yoy or target, and setting it to anything else allows us to ensure the demand_series used for mine tuning is the one from our historical data file.'],dtype='object')
        hyperparameters.loc['reserves_ratio_price_lag',:] = 7, 'lag on price change price(t-lag)/price(t-lag-1) used for informing incentive pool size change, paired with resources_contained_elas_primary_price (and byproduct if byproduct==True)'

        # demand
        hyperparameters.loc['demand only',:] = np.nan
        hyperparameters.loc['commodity'] = 'notAu'
        hyperparameters.loc['sector_specific_dematerialization_tech_growth','Value'] = -0.03
        hyperparameters.loc['sector_specific_price_response','Value'] = -0.06
        hyperparameters.loc['region_specific_price_response','Value'] = -0.1
        hyperparameters.loc['intensity_response_to_gdp','Value'] = 0.69

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
        dem.collection_rate_price_response = self.collection_rate_price_response
        dem.scenario_type = self.scenario_type
        dem.collection_rate_duration = self.collection_rate_duration
        dem.collection_rate_pct_change_tot = self.collection_rate_pct_change_tot
        dem.collection_rate_pct_change_inc = self.collection_rate_pct_change_inc

        ref = self.refine
        ref.scenario_type = self.scenario_type
        ref.secondary_refined_price_response = self.secondary_refined_price_response
        ref.secondary_refined_duration = self.secondary_refined_duration
        ref.secondary_refined_pct_change_tot = self.secondary_refined_pct_change_tot
        ref.secondary_refined_pct_change_inc = self.secondary_refined_pct_change_inc
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
                print('   rhp',param, h[param])

        for param in np.intersect1d(dem.hyperparam.index, h.index):
            dem.hyperparam.loc[param,'Value'] = h[param]
            if self.verbosity>1:
                print('   demand',param, h[param])

        self.demand = dem
        self.refine = ref

    def initialize_integration(self):
        '''
        Sets up the integration, also sets up the scenario corresponding with scenario_name
        '''
        self.load_historical_data()

        i = self.i
        self.conc_to_cathode_eff = self.refine.ref_param['Global']['conc to cathode eff']
        self.scrap_to_cathode_eff = self.refine.ref_param['Global']['scrap to cathode eff']

        # initializing demand
        self.hyperparam_agreement()
        self.demand.run()

        # initializing collection_rate (scenario modification is done in the demand class)
        self.collection_rate = self.demand.collection_rate.copy()
        self.additional_scrap = self.demand.additional_scrap.copy()

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
        self.direct_melt_fraction = self.direct_melt_demand/self.total_demand
        self.direct_melt_fraction = self.direct_melt_fraction.apply(lambda x: self.direct_melt_fraction.loc[i],axis=1)
#         self.direct_melt_demand.loc[i+1:] = np.nan
        self.refined_demand = self.total_demand - self.direct_melt_demand
        self.direct_melt_demand = self.direct_melt_demand.apply(lambda x: x/self.scrap_to_cathode_eff,axis=1)
        self.additional_direct_melt = self.direct_melt_demand.copy()
        self.additional_direct_melt.loc[:] = 0
        if self.scenario_type in ['scrap demand','both']:
            if not self.direct_melt_alt:# trying an alternative method, seems like adding more each year is not quite in line with how the market would work. Instead, it should be that once someone increases demand, their new demand is implicit within the rest of the market so we do not need to keep adding it each year
                multiplier_array = np.append([1],np.append(
                    np.repeat(1+(self.direct_melt_pct_change_tot-1)/self.direct_melt_duration,self.direct_melt_duration),
                    [self.direct_melt_pct_change_inc for j in np.arange(1,len(self.simulation_time)-self.direct_melt_duration)]))
            else:
                multiplier_array = np.append(
                    np.linspace(1,self.direct_melt_pct_change_tot,self.direct_melt_duration+1),
                    [self.direct_melt_pct_change_tot*self.direct_melt_pct_change_inc**j for j in np.arange(1,len(self.simulation_time)-self.direct_melt_duration)])
            if self.direct_melt_price_response==False:
                for yr,mul in zip(self.simulation_time,multiplier_array):
                    self.direct_melt_fraction.loc[yr,:]*=mul
            else:
                multiplier_array -= 1
#                 ph2 = self.demand.scrap_collected.stack().unstack(1).unstack()
                self.additional_direct_melt = self.demand.demand.loc[self.simulation_time[0]:].apply(lambda x: self.demand.demand.loc[self.simulation_time[0]]*multiplier_array[x.name-self.simulation_time[0]],axis=1)
                self.additional_direct_melt = self.additional_direct_melt.groupby(level=0,axis=1).sum()
                self.additional_direct_melt.loc[:,'Global'] = self.additional_direct_melt.sum(axis=1)
                self.additional_direct_melt.loc[self.simulation_time[0]-1,:] = 0
                self.additional_direct_melt = self.additional_direct_melt.sort_index()

        # concentrate_demand, secondary_refined_demand, refined_supply
        self.concentrate_demand = self.refine.ref_stats.loc[:,idx[:,'Primary production']].droplevel(1,axis=1)
        self.concentrate_demand.loc[i+1:] = np.nan
        self.secondary_refined_demand = self.refine.ref_stats.loc[:,idx[:,'Secondary production']].droplevel(1,axis=1)
        self.secondary_refined_demand.loc[i+1:] = np.nan
        self.secondary_ratio = self.refine.ref_stats.loc[:,idx[:,'Secondary ratio']].droplevel(1,axis=1)
        self.refined_supply = self.concentrate_demand + self.secondary_refined_demand
        self.primary_supply = self.concentrate_demand/self.refined_supply*self.refined_demand
        self.primary_demand = self.refined_demand * self.secondary_refined_demand / self.refined_supply

        self.concentrate_demand = self.concentrate_demand.apply(lambda x: x/self.conc_to_cathode_eff,axis=1)
        self.secondary_refined_demand = self.secondary_refined_demand.apply(lambda x: x/self.scrap_to_cathode_eff,axis=1)
        self.additional_secondary_refined = self.direct_melt_demand.copy()
        self.additional_secondary_refined.loc[:] = 0
        if self.scenario_type in ['scrap demand','both']:
            if not self.secondary_refined_alt:
                multiplier_array = np.append([1],np.append(
                    np.repeat(1+(self.secondary_refined_pct_change_tot-1)/self.secondary_refined_duration,self.secondary_refined_duration),
                    [self.secondary_refined_pct_change_inc for j in np.arange(1,len(self.simulation_time)-self.secondary_refined_duration)]))
            else:
                multiplier_array = np.append(
                    np.linspace(1,self.secondary_refined_pct_change_tot,self.secondary_refined_duration+1),
                    [self.secondary_refined_pct_change_tot*self.secondary_refined_pct_change_inc**j for j in np.arange(1,len(self.simulation_time)-self.secondary_refined_duration)])
            if self.secondary_refined_price_response==False:
                for yr,mul in zip(self.simulation_time,multiplier_array):
                    self.secondary_ratio.loc[yr,:]*=mul
                self.refine.secondary_ratio = self.secondary_ratio.copy()
            else:
                multiplier_array -= 1
#                 ph2 = self.demand.scrap_collected.stack().unstack(1).unstack()
#                 self.additional_secondary_refined = self.demand.demand.loc[self.simulation_time[0]:].apply(lambda x: ph2.loc[self.simulation_time[0]]*multiplier_array[x.name-self.simulation_time[0]],axis=1)
                self.additional_secondary_refined = self.demand.demand.loc[self.simulation_time[0]:].apply(lambda x: self.demand.demand.loc[self.simulation_time[0]]*multiplier_array[x.name-self.simulation_time[0]],axis=1)
                self.additional_secondary_refined = self.additional_secondary_refined.groupby(level=0,axis=1).sum()
                self.additional_secondary_refined.loc[:,'Global'] = self.additional_secondary_refined.sum(axis=1)
                self.additional_secondary_refined.loc[self.simulation_time[0]-1,:] = 0
                self.additional_secondary_refined = self.additional_secondary_refined.sort_index()
                self.refine.additional_secondary_refined = self.additional_secondary_refined.copy()

        # concentrate_supply,    initializing mining
        if self.h['presimulate_mining']:
            self.presimulate_mining()
        else:
            self.mining = miningModel(simulation_time=self.simulation_time, verbosity=self.verbosity, byproduct=self.byproduct)
            self.mining.hyperparam.loc['primary_production','Value'] = self.concentrate_demand['Global'][i]
            if hasattr(self,'historical_data'):
                self.mining.demand_series = self.historical_data['Primary supply' if 'Primary supply' in self.historical_data.columns else 'Primary production']
            else: raise ValueError('if simulating a real commodity, the Integration class initialization should take a str input for its commodity variable, which should correspond with a sheet name in case study data.xlsx. See the presimulate_mining function for how to deal with creating an alternate demand, if that is desired')
            if self.verbosity>1: print('Mining parameters updated:')
            for param in np.intersect1d(self.mining.hyperparam.index, self.h.index):
                self.mining.hyperparam.loc[param,'Value'] = self.h[param]
                if self.verbosity>1:
                    print('  ',param)
            self.mining.run()
        self.concentrate_supply = self.mining.concentrate_supply_series.copy()
        self.sxew_supply = self.mining.sxew_supply_series.copy()
        self.mine_production = self.concentrate_supply+self.sxew_supply

        # scrap_supply, scrap_demand
        self.scrap_supply = self.demand.scrap_supply.copy()
        self.scrap_demand = self.secondary_refined_demand+self.direct_melt_demand

    def initialize_price(self):
        i = self.i
        h = self.h
        self.scrap_spread.loc[:i] = h['initial_scrap_spread']
        self.primary_commodity_price.loc[:i] = h['primary_commodity_price']
        self.tcrc.loc[:i] = self.mining.primary_tcrc_series[i]

    def price_evolution(self):
        i = self.i
        h = self.h
        # refined
        self.refined_supply = self.refined_supply.where(self.refined_supply>1e-9).fillna(1e-9)
        self.refined_demand = self.refined_demand.where(self.refined_demand>1e-9).fillna(1e-9)
        self.primary_commodity_price.loc[i] = self.primary_commodity_price.loc[i-1]*(self.refined_supply['Global'][i-1]/self.refined_demand['Global'][i-1])**h['primary_commodity_price_elas_sd']
        self.primary_commodity_price = self.primary_commodity_price.where(self.primary_commodity_price>1e-9).fillna(1e-9)

        # if hasattr(self,'historical_data'):
        #     self.primary_commodity_price.loc[i] = self.historical_data['Primary commodity price'][i]

        # tcrc
        self.concentrate_supply = self.concentrate_supply.where(self.concentrate_supply>1e-9).fillna(1e-9)
        self.concentrate_demand = self.concentrate_demand.where(self.concentrate_demand>1e-9).fillna(1e-9)
        self.tcrc.loc[i] = self.tcrc.loc[i-1]*(self.concentrate_supply[i-1]/self.concentrate_demand['Global'][i-1])**h['tcrc_elas_sd']\
                                *(self.primary_commodity_price.loc[i]/self.primary_commodity_price.loc[i-1])**h['tcrc_elas_price']
        self.tcrc = self.tcrc.where(self.tcrc>1e-9).fillna(1e-9)

        # scrap trade
        self.scrap_supply = self.scrap_supply.where(self.scrap_supply>1e-9).fillna(1e-9)
        self.scrap_demand = self.scrap_demand.where(self.scrap_demand>1e-9).fillna(1e-9)
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
        self.scrap_spread = self.scrap_spread.where(self.scrap_spread>1e-9).fillna(1e-9)

    def run_demand(self):
        self.demand.i = self.i
        i = self.i
        h = self.h.copy()
        if self.i-2 not in self.primary_commodity_price.index:
            self.primary_commodity_price.loc[self.i-2] = self.primary_commodity_price[self.i-1]
        if self.i-2 not in self.scrap_spread.index:
            self.scrap_spread.loc[i-2] = self.scrap_spread.loc[i-1]
        self.demand.commodity_price_series = self.primary_commodity_price.copy()
        self.demand.collection_rate = self.collection_rate.copy()
        self.demand.run()
        self.additional_scrap = self.demand.additional_scrap.copy()

    def run_refine(self):
        self.refine.i = self.i
        self.refine.tcrc_series = self.tcrc.copy()
        self.refine.scrap_spread_series = self.scrap_spread.copy()
        self.refine.refined_price_series = self.primary_commodity_price.copy()
        if self.i-2 not in self.concentrate_supply.index:
            self.concentrate_supply.loc[self.i-2] = self.concentrate_supply[self.i-1]*self.refined_demand['Global'][self.i-2]/self.refined_demand['Global'][self.i-1]
        if self.i-2 not in self.scrap_supply.index:
            self.scrap_supply.loc[self.i-2] = self.scrap_supply.loc[self.i-1]*self.refined_demand.loc[self.i-2]/self.refined_demand.loc[self.i-1]

        conc_supply = self.concentrate_supply.copy()/self.concentrate_supply[self.simulation_time[0]]
        ref_demand = self.refined_demand['Global'].copy()/self.refined_demand['Global'][self.simulation_time[0]]
        pri_growth_series = conc_supply*self.h['refinery_capacity_fraction_increase_mining'] + ref_demand*(1-self.h['refinery_capacity_fraction_increase_mining'])
        self.refine.pri_cap_growth_series = pri_growth_series

        self.refine.sec_cap_growth_series = self.scrap_supply.copy()
        self.refine.run()

    def run_mining(self):
        self.mining.i = self.i
        self.mining.primary_price_series.loc[self.i] = self.primary_commodity_price[self.i]
        self.mining.primary_tcrc_series.loc[self.i] = self.tcrc[self.i]
        if self.concentrate_demand['Global'][self.i-1]==np.nan and self.mining.demand_series[self.i-1]!=np.nan:
            self.concentrate_demand.loc[self.i-1,'Global']=self.mining.demand_series.loc[self.i-1]
            print(433, 'did this one')
        self.mining.demand_series.loc[self.i-1] = self.concentrate_demand['Global'][self.i-1]
        self.mining.run()
        # concentrate_supply
        self.concentrate_supply.loc[self.i] = self.mining.concentrate_supply_series[self.i]
        self.sxew_supply.loc[self.i] = self.mining.sxew_supply_series[self.i]
        self.mine_production.loc[self.i] = self.concentrate_supply.loc[self.i]+self.sxew_supply.loc[self.i]

        self.primary_supply.loc[self.i,'Global'] += self.sxew_supply.loc[self.i]
        self.primary_supply.loc[self.i,'RoW'] += self.sxew_supply.loc[self.i]

    def presimulate_mining(self):
        h = self.h.copy()
        end_yr = self.simulation_time[0]
        hist_simulation_time = np.arange(end_yr-h['presimulate_n_years'],end_yr+1)
        mine_simulation_time = np.arange(end_yr-h['presimulate_n_years'],self.simulation_time[-1]+1)
        self.mining = miningModel(simulation_time=mine_simulation_time, byproduct=self.byproduct,verbosity=self.verbosity)
        m = self.mining
        if self.verbosity>1: print('Mining parameters updated:')
        for param in [j for j in h.index if j in m.hyperparam]:
            m.hyperparam[param] = h[param]
            if self.verbosity>1:
                print('  ',param,'now',h[param])

        initial_ore_grade_decline = m.hyperparam['initial_ore_grade_decline']
        incentive_mine_cost_improvement = m.hyperparam['incentive_mine_cost_improvement']
        annual_reserves_ratio_with_initial_production_slope = m.hyperparam['annual_reserves_ratio_with_initial_production_slope']
        m.hyperparam['internal_price_formation'] = False
        m.hyperparam['initial_ore_grade_decline'] = 0
        m.hyperparam['incentive_mine_cost_improvement'] = 0
        m.hyperparam['annual_reserves_ratio_with_initial_production_slope'] = 0

#         primary_production
        if hasattr(self,'historical_data'):
            self.historical_mining = self.historical_data['Primary supply' if 'Primary supply' in self.historical_data.columns else 'Primary production'].astype(float)
            self.historical_mining = self.historical_mining.dropna()
            self.historical_mining.index = self.historical_mining.index.astype(int)
            start_hist_year = self.historical_mining.sort_index().index[0]
            m.demand_series = self.demand.alt_demand[[j for j in self.demand.alt_demand.columns if j!='China Fraction']].sum(axis=1).rolling(5).mean()
            m.demand_series *= self.historical_mining[start_hist_year]/m.demand_series[start_hist_year]
            m.demand_series = pd.concat([m.demand_series.loc[~m.demand_series.index.isin(self.historical_mining.index)],self.historical_mining]).sort_index()

            m.primary_price_series = self.historical_data['Primary commodity price'].copy().dropna()
            m.primary_price_series.name = 'Primary commodity price'
            m.primary_price_series = pd.concat([pd.Series(m.primary_price_series.iloc[0],[i for i in mine_simulation_time if i not in m.primary_price_series.index]),
                m.primary_price_series]).sort_index()
            m.hyperparam['primary_commodity_price'] = m.primary_price_series.iloc[0]

        else:
            m.demand_series = self.demand.alt_demand[[j for j in self.demand.alt_demand.columns if j!='China Fraction']].sum(axis=1).rolling(5).mean()
            m.demand_series *= self.concentrate_demand['Global'][end_yr]/m.demand_series[end_yr]
            warn('if simulating a real commodity, the Integration class initialization should take a str input for its commodity variable, which should correspond with a sheet name in case study data.xlsx')

        m.hyperparam['primary_production'] = m.demand_series[mine_simulation_time[0]]
        primary_production_mean_series = m.demand_series*h['primary_production_mean']/m.demand_series[end_yr]
        self.primary_production_mean_series = primary_production_mean_series.copy()
#         display(m.demand_series)
        for year in hist_simulation_time:
            m.i = year
            if self.verbosity>1:
                print('sim mine history year:',year)
            m.hyperparam['primary_production_mean'] = primary_production_mean_series[year]
            m.run()
#             display(m.demand_series)
#             raise ValueError('no')

            if self.verbosity>4:
                fig,ax=easy_subplots(3)
                ax[0].plot(m.supply_series,label='Supply',marker='o')
                ax[0].plot(m.demand_series,label='Demand',marker='v',zorder=0)
                ax[0].legend()

                ax[1].plot(m.supply_series/m.demand_series,marker='o')

                ax[2].plot(m.primary_tcrc_series,marker='o')

                plt.show()
                plt.close()

        m.hyperparam['initial_ore_grade_decline'] = initial_ore_grade_decline
        m.hyperparam['incentive_mine_cost_improvement'] = incentive_mine_cost_improvement
        m.hyperparam['annual_reserves_ratio_with_initial_production_slope'] = annual_reserves_ratio_with_initial_production_slope
        m.hyperparam['internal_price_formation'] = False
        self.concentrate_supply = m.concentrate_supply_series.copy()
        self.sxew_supply = m.sxew_supply_series.copy()
        self.mine_production = self.concentrate_supply+self.sxew_supply
        self.tcrc = m.primary_tcrc_series.copy()
        self.tcrc = pd.concat([self.tcrc.dropna(), pd.Series(self.tcrc.dropna().sort_index().iloc[-1],[i for i in self.simulation_time if i not in self.tcrc.dropna().index])]).sort_index()
        self.primary_commodity_price = m.primary_price_series.copy()
        self.concentrate_demand.loc[:,'Global'] = m.demand_series.copy()

        self.mining = m

    def calculate_direct_melt_demand(self):
        i = self.i
#         self.direct_melt_fraction.loc[i-1] = self.direct_melt_demand.loc[i-1]/self.total_demand.loc[i-1]
        if False and self.direct_melt_price_response:
            self.direct_melt_demand.loc[i] = self.direct_melt_demand.loc[i-1]*self.scrap_supply.loc[i]/self.scrap_supply.loc[i-1]\
                * (self.scrap_spread.loc[i]/self.scrap_spread.loc[i-1])**self.h['direct_melt_elas_scrap_spread']
        elif False and self.direct_melt_price_response:
            direct_melt_fraction = self.refine.hyperparam.loc['Direct melt fraction of production']\
                * (self.scrap_spread.loc[i]/self.scrap_spread.loc[i-1])**self.h['direct_melt_elas_scrap_spread']
        elif self.direct_melt_price_response:
            temp_direct_melt_demand = self.direct_melt_demand.copy()
            temp_direct_melt_demand.loc[:i,:] -= self.additional_direct_melt.loc[:i,:]
            temp_direct_melt_fraction = temp_direct_melt_demand/self.total_demand
            self.direct_melt_demand.loc[i-1,:] -= self.additional_direct_melt.loc[i-1,:]
            direct_melt_fraction = temp_direct_melt_fraction.loc[i-1]\
                * (self.scrap_spread.loc[i]/self.scrap_spread.loc[i-1])**self.h['direct_melt_elas_scrap_spread']
            if (direct_melt_fraction > 1).any(): direct_melt_fraction[direct_melt_fraction>1] = 1
            if (direct_melt_fraction < 0).any(): direct_melt_fraction[direct_melt_fraction<0] = 0
            self.direct_melt_demand.loc[i] = self.total_demand.loc[i]*direct_melt_fraction
            self.direct_melt_demand.loc[i-1:i,:] += self.additional_direct_melt.loc[i-1:i,:]
            self.direct_melt_fraction.loc[i] = self.direct_melt_demand.loc[i]/self.total_demand.loc[i]
            self.temp_direct_melt_fraction = temp_direct_melt_fraction.copy()
        else:
            self.direct_melt_demand.loc[i] = self.total_demand.loc[i]*self.direct_melt_fraction.loc[i]

    def update_integration_variables_post_demand(self):
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

    def update_integration_variables_post_refine(self):
        i = self.i

        # concentrate_demand, secondary_refined_demand, refined_supply
        self.concentrate_demand.loc[i] = self.refine.ref_stats.loc[i,idx[:,'Primary production']].droplevel(1)
        self.concentrate_demand[self.concentrate_demand<0] = 0
        self.secondary_refined_demand.loc[i] = self.refine.ref_stats.loc[i,idx[:,'Secondary production']].droplevel(1)
        self.secondary_refined_demand[self.secondary_refined_demand<0] = 0
        self.refined_supply.loc[i] = self.concentrate_demand.loc[i] + self.secondary_refined_demand.loc[i]

        # primary supply and demand, assuming all sxew is done in RoW (this update is done in the mining module, and here we do primary supply as primary refined supply)
        self.primary_supply.loc[i] = self.concentrate_demand.copy().loc[i]/self.refined_supply.loc[i]*self.refined_demand.loc[i]
        self.primary_demand.loc[i] = self.refined_demand.loc[i]*self.secondary_refined_demand.loc[i]/self.refined_supply.loc[i]

        # correcting concentrate demand and secondary refined demand for efficiency loss
        self.concentrate_demand.loc[i] /= self.conc_to_cathode_eff
        self.secondary_refined_demand.loc[i] /= self.scrap_to_cathode_eff

        # scrap_demand
        self.scrap_demand.loc[i] = self.secondary_refined_demand.loc[i]+self.direct_melt_demand.loc[i]

        self.mining.demand_series.loc[self.i] = self.concentrate_demand['Global'][self.i]

    def run(self):
        self.h = self.hyperparam['Value'].copy()
        for year in self.simulation_time:
            if self.verbosity>0: print(year)
            self.i = year
            if self.i==self.simulation_time[0]:
                self.decode_scrap_scenario_name()
                self.h = self.hyperparam['Value'].copy()
                self.initialize_integration()
                self.initialize_price()
            else:
                self.price_evolution()
                self.run_demand()
                self.update_integration_variables_post_demand()
                self.run_mining()

                self.update_integration_variables_post_demand()
                self.run_refine()
                self.update_integration_variables_post_refine()
        self.concentrate_demand = pd.concat([self.mining.demand_series,self.concentrate_demand['China'],
                                            self.concentrate_demand['RoW']],axis=1,keys=['Global','China','RoW'])

    def run_mining_only(self):
        self.h = self.hyperparam['Value'].copy()
        for year in self.simulation_time:
            if self.verbosity>0: print(year)
            self.i = year
            if self.i==self.simulation_time[0]:
                self.h = self.hyperparam['Value'].copy()
                self.initialize_integration()
            else:
                self.primary_commodity_price.loc[year] = self.historical_data['Primary commodity price'].copy().dropna().loc[year]
                self.run_mining()

    def decode_scrap_scenario_name(self):
        '''
        scenario_name takes the form 00_11_22_33_44
        where:
        00: ss, sd, bo (scrap supply, scrap demand, both).
         Can also be sd-alt, which uses an alternative
         implementation of the scrap demand increase (see below)
        11: pr or no (price response included or no)
        22: Xyr, where X is any integer and represents
         the number of years the increase occurs
        33: X%tot, where X is any float/int and is the
         increase/decrease in scrap supply or demand
         relative to the initial year total demand
        44: X%inc, where X is any float/int and is the
         increase/decrease in the %tot value per year

        e.g. ss_pr_1yr_1%tot_0%inc

        for 22-44, an additional X should be placed at
         the end when 00==both, describing the sd values
         e.g. ss_pr_1yr1_1%tot1_0%inc0

        Can also have 11 as nono, prno, nopr, or prpr to
         control the ss and sd price response individually
         (in that order)

        for sd-alt, the default (no alt) formulation is that an
        increase in scrap demand to 5% of demand over 2
        years would be
        [0, 2.5%, 2.5%, 0%, 0%, ..., 0%]
        while for alt it would be
        [0, 2.5%, 5%,   5%, 5%, ..., 5%]
        Can ctrl+F `direct_melt_duration` or `secondary_refined_duration`
        to see the actual methods
        '''
        scenario_name = self.scenario_name
        error_string = 'improper format for scenario_name. Takes the form of 00_11_22_33_44 where:'+\
                                 '\n\t00: ss, sd, bo (scrap supply, scrap demand, both)'+\
                                 '\n\t11: pr or no (price response included or no)'+\
                                 '\n\t22: Xyr, where X is any integer and represents the number of years the increase occurs'+\
                                 '\n\t33: X%tot, where X is any float/int and is the increase/decrease in scrap supply or demand relative\n\t\tto the initial year total demand'+\
                                 '\n\t44: X%inc, where X is any float/int and is the increase/decrease in the %tot value per year'+\
                                 '\n\n\tfor 22-44, an additional X should be placed at the end when 00==bo, describing the sd values'+\
                                 '\n\n\tscenario_name: '+scenario_name+\
                                 '\n\tproper format: 00_11_Xyr_X%tot_X%inc or 00_11_XyrX_X%totX_X%incX if 00=bo'
        collection_rate_price_response=self.hyperparam['Value']['collection_rate_price_response']
        direct_melt_price_response=self.hyperparam['Value']['direct_melt_price_response']
        collection_rate_duration = 0
        collection_rate_pct_change_tot = 0
        collection_rate_pct_change_inc = 0
        direct_melt_duration = 0
        direct_melt_pct_change_tot = 0
        direct_melt_pct_change_inc = 0
        secondary_refined_duration = 0
        secondary_refined_pct_change_tot = 0
        secondary_refined_pct_change_inc = 0
        self.secondary_refined_alt = False
        self.direct_melt_alt = False

        if scenario_name=='':
            scenario_type = scenario_name
        else:
            if '_' in scenario_name:
                name = scenario_name.split('_')
            else:
                raise ValueError(error_string)
            if np.any([j not in scenario_name for j in ['yr','%tot','%inc']])\
               or len(name)!=5\
               or not np.any([j in name[0] for j in ['ss','sd','bo']])\
               or not np.any([j in name[1] for j in ['pr','no']])\
               or 'yr' not in name[2] or '%tot' not in name[3] or '%inc' not in name[4]:
                raise ValueError(error_string)
            scenario_type = name[0].replace('ss','scrap supply').replace('sd','scrap demand').replace('bo','both')

            if scenario_type in ['scrap supply','both']:
                collection_rate_duration= int(name[2].split('yr')[0])
                collection_rate_pct_change_tot = float(name[3].split('%tot')[0])
                collection_rate_pct_change_inc = float(name[4].split('%inc')[0])
                if name[1]=='pr':
                    collection_rate_price_response=True
                elif name[1]=='no':
                    collection_rate_price_response=False

            if scenario_type in ['scrap demand','both','scrap demand-alt']:
                if scenario_type=='both':
                    integ = 1
                    if name[2].split('yr')[1]=='' or name[3].split('%tot')[1]=='' or name[4].split('%inc')[1]=='':
                        print('WARNING: scenario name does not fit BOTH format, using SCRAP SUPPLY value for SCRAP DEMAND')
                        integ=0
                else:
                    integ = 0
                direct_melt_duration = int(name[2].split('yr')[integ])
                direct_melt_pct_change_tot = float(name[3].split('%tot')[integ])
                direct_melt_pct_change_inc = float(name[4].split('%inc')[integ])
                if name[1]=='pr':
                    direct_melt_price_response=True
                elif name[1]=='no':
                    direct_melt_price_response=False
                self.direct_melt_alt = '-alt' in scenario_type
                self.secondary_refined_alt = self.direct_melt_alt

            if len(name[1])>2:
                if name[1]=='nono':
                    collection_rate_price_response=False
                    direct_melt_price_response=False
                elif name[1]=='prno':
                    collection_rate_price_response=True
                    direct_melt_price_response=False
                elif name[1]=='nopr':
                    collection_rate_price_response=False
                    direct_melt_price_response=True
                elif name[1]=='prpr':
                    collection_rate_price_response=True
                    direct_melt_price_response=True
                else:
                    print(f'WARNING, scenario name does not fit price response format, using default value from hyperparam input which is:\n\tcollection_rate_price_response={collection_rate_price_response}\n\tdirect_melt_price_response={direct_melt_price_response}')

        self.scenario_type = scenario_type
        self.collection_rate_price_response = collection_rate_price_response
        self.direct_melt_price_response = direct_melt_price_response
        self.secondary_refined_price_response = direct_melt_price_response
        self.collection_rate_duration = collection_rate_duration
        self.collection_rate_pct_change_tot = 1+collection_rate_pct_change_tot/100
        self.collection_rate_pct_change_inc = 1+collection_rate_pct_change_inc/100
        self.direct_melt_duration = direct_melt_duration
        self.direct_melt_pct_change_tot = 1+direct_melt_pct_change_tot*(1-self.hyperparam['Value']['Secondary refinery fraction of recycled content, Global'])/100
        self.direct_melt_pct_change_inc = 1+direct_melt_pct_change_inc/100
        self.secondary_refined_duration = direct_melt_duration
        self.secondary_refined_pct_change_tot = 1+direct_melt_pct_change_tot*self.hyperparam['Value']['Secondary refinery fraction of recycled content, Global']/100
        self.secondary_refined_pct_change_inc = 1+direct_melt_pct_change_inc/100

        self.hyperparam.loc['scenario_type','Value'] = self.scenario_type
        self.hyperparam.loc['scenario_type','Notes'] = 'empty string, scrap supply, scrap demand, or both'
        self.hyperparam.loc['collection_rate_price_response','Value'] = self.collection_rate_price_response
        self.hyperparam.loc['collection_rate_price_response','Notes'] = 'whether or not there should be a price response for collection rate'
        self.hyperparam.loc['direct_melt_price_response','Value'] = self.direct_melt_price_response
        self.hyperparam.loc['direct_melt_price_response','Notes'] = 'whether there should be a price response for direct melt fraction'
        self.hyperparam.loc['secondary_refined_price_response','Value'] = self.direct_melt_price_response
        self.hyperparam.loc['secondary_refined_price_response','Notes'] = 'whether there should be a price response for direct melt fraction'
        self.hyperparam.loc['collection_rate_duration','Value'] = self.collection_rate_duration
        self.hyperparam.loc['collection_rate_duration','Notes'] = 'length of the increase in collection rate described by collection_rate_pct_change_tot'
        self.hyperparam.loc['collection_rate_pct_change_tot','Value'] = self.collection_rate_pct_change_tot
        self.hyperparam.loc['collection_rate_pct_change_tot','Notes'] = 'without price response, describes the percent increase in collection rate attained at the end of the linear ramp with duration collection_rate_duration. Given as 1+%change/100'
        self.hyperparam.loc['collection_rate_pct_change_inc','Value'] = self.collection_rate_pct_change_inc
        self.hyperparam.loc['collection_rate_pct_change_inc','Notes'] = 'once the collection_rate_pct_change_tot is reached, the collection rate will then increase by this value per year. Given as 1+%change/100'
        self.hyperparam.loc['direct_melt_duration','Value'] = self.direct_melt_duration
        self.hyperparam.loc['direct_melt_duration','Notes'] = 'length of the increase in direct melt fraction described by direct_melt_pct_change_tot'
        self.hyperparam.loc['direct_melt_alt','Value'] = self.direct_melt_alt
        self.hyperparam.loc['direct_melt_pct_change_tot','Value'] = self.direct_melt_pct_change_tot
        self.hyperparam.loc['direct_melt_pct_change_tot','Notes'] = 'without price response, describes the percent increase in collection rate attained at the end of the linear ramp with duration direct_melt_duration. Given as 1+%change/100'
        self.hyperparam.loc['direct_melt_pct_change_inc','Value'] = self.direct_melt_pct_change_inc
        self.hyperparam.loc['direct_melt_pct_change_inc','Notes'] = 'once the direct_melt_pct_change_tot is reached, the direct melt fraction will then increase by this value per year. Given as 1+%change/100'
        self.hyperparam.loc['secondary_refined_duration','Value'] = self.secondary_refined_duration
        self.hyperparam.loc['secondary_refined_duration','Notes'] = 'length of the increase in direct melt fraction described by direct_melt_pct_change_tot'
        self.hyperparam.loc['secondary_refined_alt','Value'] = self.secondary_refined_alt
        self.hyperparam.loc['secondary_refined_pct_change_tot','Value'] = self.secondary_refined_pct_change_tot
        self.hyperparam.loc['secondary_refined_pct_change_tot','Notes'] = 'without price response, describes the percent increase in collection rate attained at the end of the linear ramp with duration direct_melt_duration. Given as 1+%change/100'
        self.hyperparam.loc['secondary_refined_pct_change_inc','Value'] = self.secondary_refined_pct_change_inc
        self.hyperparam.loc['secondary_refined_pct_change_inc','Notes'] = 'once the direct_melt_pct_change_tot is reached, the direct melt fraction will then increase by this value per year. Given as 1+%change/100'
