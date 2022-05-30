from refining_functions import *
import numpy as np
import pandas as pd
idx = pd.IndexSlice

class refiningModel():
    
    def __init__(self,simulation_time=np.arange(2019,2041),verbosity=0):
        self.simulation_time = simulation_time
        self.i = simulation_time[0]
        self.verbosity = verbosity
        self.init_input_parameters_ref_hyperparam()
        self.update_hyperparam()
        self.update_ref_param()
        
        sim_time2 = np.arange(simulation_time[0]-1,1+simulation_time[-1])
        rate = 0.01
        self.tcrc_series = pd.Series([1*(1.0+rate)**(y-simulation_time[0]) for y in sim_time2], sim_time2)
        rate = 0
        self.scrap_spread_series = pd.Series([1*(1.0+rate)**(y-simulation_time[0]) for y in sim_time2], sim_time2)
        self.scrap_spread_series = pd.concat([self.scrap_spread_series for i in [0,0,0]],axis=1,keys=['Global','China','RoW'])
        rate = 0.02
        self.pri_cap_growth_series = pd.Series([1*(1.0+rate)**(y-simulation_time[0]) for y in sim_time2], sim_time2)
        rate = 0.01
        self.sec_cap_growth_series = pd.Series([1*(1.0+rate)**(y-simulation_time[0]) for y in sim_time2], sim_time2)
        self.sec_cap_growth_series = pd.concat([self.sec_cap_growth_series for i in [0,0,0]],axis=1,keys=['Global','China','RoW'])
        
    def init_input_parameters_ref_hyperparam(self):
        hyperparam = pd.DataFrame(0,['Recycling input rate','Secondary refinery fraction of recycled content','Regional production fraction of total production'],['Global','China','RoW'])
        hyperparam.loc['Recycling input rate','Global'] = 0.2
        hyperparam.loc['Recycling input rate','China'] = 0.2
        hyperparam.loc['Secondary refinery fraction of recycled content','Global'] = 0.5
        hyperparam.loc['Secondary refinery fraction of recycled content','China'] = 0.5
        hyperparam.loc['Regional production fraction of total production','Global'] = 1
        hyperparam.loc['Regional production fraction of total production','China'] = 0.5
        hyperparam.loc['SX-EW fraction of production','Global'] = 0
        hyperparam.loc['SX-EW fraction of production','China'] = 0
        hyperparam.loc['Total production','Global'] = 4 # kt
        hyperparam.loc['Use regions'] = True
        
        ref_hyper_param = pd.DataFrame(np.nan,['pri cap','pri CU','pri CU TCRC elas','sec cap','sec CU','sec CU TCRC elas','sec ratio','sec ratio TCRC elas','sec ratio scrap spread elas','conc to cathode eff','scrap to cathode eff'],['Value','Notes'])
        ref_hyper_param.loc['pri CU',:] = 0.85, 'Capacity at primary-only refineries'
        ref_hyper_param.loc['sec CU',:] = 0.85, 'Capacity at refineries that process secondary material (and also primary if the secondary ratio is less than 1)'
        ref_hyper_param.loc['pri CU TCRC elas',:] = 0.057, 'Capacity at primary-only refineries'
        ref_hyper_param.loc['sec CU TCRC elas',:] = 0.153, 'Capacity at secondary-consuming refineries'
        ref_hyper_param.loc['sec ratio',:] = 0.3, 'Secondary ratio, or the fraction of secondary-consuming refinery material consumption coming from scrap'
        ref_hyper_param.loc['sec ratio TCRC elas',:] = -0.197, 'Secondary ratio elasticity to TCRC'
        ref_hyper_param.loc['sec ratio scrap spread elas',:] = 0.316, 'Secondary ratio elasticity to scrap spread'
        ref_hyper_param.loc['conc to cathode eff',:] = 0.99, 'Efficiency of the concentrate to refined metal conversion (fraction)'
        ref_hyper_param.loc['scrap to cathode eff',:] = 0.99, 'Efficiency of the scrap to refined metal conversion (fraction)'
        ref_hyper_param.loc['pri cap','Notes'] = 'Primary-only refinery capacity (kt)'
        ref_hyper_param.loc['sec cap','Notes'] = 'Secondary-consuming refinery capacity (kt)'
        
        self.hyperparam = hyperparam.copy()
        ref_hyper_param_cn = ref_hyper_param.copy()
        ref_hyper_param_rw = ref_hyper_param.copy()
        self.ref_hyper_param = ref_hyper_param.copy()
        self.ref_hyper_param_cn = ref_hyper_param_cn.copy()
        self.ref_hyper_param_rw = ref_hyper_param_rw.copy()

    def update_hyperparam(self):
        input_param = self.hyperparam.copy()
        global_rir,china_rir=input_param.loc['Recycling input rate',['Global','China']] 
        global_sec_ref_fraction_of_recycled,china_sec_ref_fraction_of_recycled=input_param.loc['Secondary refinery fraction of recycled content',['Global','China']] 
        china_fraction_of_total_production = input_param.loc['Regional production fraction of total production','China']
        sxew_fraction, sxew_fraction_cn = input_param.loc['SX-EW fraction of production',['Global','China']]

        row_rir = (global_rir - china_rir*china_fraction_of_total_production) / (1-china_fraction_of_total_production)
        sxew_fraction_rw = (sxew_fraction - sxew_fraction_cn*china_fraction_of_total_production) / (1-china_fraction_of_total_production)

        china_fraction_recycled = china_rir*china_fraction_of_total_production/global_rir

        row_sec_ref_fraction_of_recycled = \
         (global_sec_ref_fraction_of_recycled - china_sec_ref_fraction_of_recycled*china_fraction_recycled) / \
         (1-china_fraction_recycled)

        sec_frac_of_refined = 1/ (1+(1-global_rir-sxew_fraction)/global_sec_ref_fraction_of_recycled/global_rir)
        sec_frac_of_refined_cn = 1/ (1+(1-china_rir-sxew_fraction_cn)/china_sec_ref_fraction_of_recycled/china_rir)
        sec_frac_of_refined_rw = 1/ (1+(1-row_rir-sxew_fraction_rw)/row_sec_ref_fraction_of_recycled/row_rir)

        global_ref_frac_production = global_sec_ref_fraction_of_recycled*global_rir + (1-global_rir) - sxew_fraction
        china_ref_frac_production = china_sec_ref_fraction_of_recycled*china_rir + (1-china_rir) - sxew_fraction_cn
        row_ref_frac_production = row_sec_ref_fraction_of_recycled*row_rir + (1-row_rir) - sxew_fraction_rw

        global_direct_melt_frac_production = -global_sec_ref_fraction_of_recycled*global_rir + global_rir
        china_direct_melt_frac_production = -china_sec_ref_fraction_of_recycled*china_rir + china_rir
        row_direct_melt_frac_production = -row_sec_ref_fraction_of_recycled*row_rir + row_rir

        input_param.loc['Total production','China'] = china_fraction_of_total_production*input_param.loc['Total production','Global']
        input_param.loc['Recycling input rate','RoW'] = row_rir
        input_param.loc['Secondary refinery fraction of recycled content','RoW'] = row_sec_ref_fraction_of_recycled
        input_param.loc['Regional production fraction of total production', 'RoW'] = 1-input_param.loc['Regional production fraction of total production','China']
        input_param.loc['SX-EW fraction of production','RoW'] = sxew_fraction_rw
        input_param.loc['Total production','RoW'] = (1-china_fraction_of_total_production)*input_param.loc['Total production','Global']
        
        input_param.loc['Secondary fraction of refinery production'] = sec_frac_of_refined,sec_frac_of_refined_cn,sec_frac_of_refined_rw
        input_param.loc['Refining fraction of production'] = global_ref_frac_production, china_ref_frac_production, row_ref_frac_production
        input_param.loc['Direct melt fraction of production'] = global_direct_melt_frac_production, china_direct_melt_frac_production, row_direct_melt_frac_production
        input_param.loc['Secondary refining fraction of production'] = global_sec_ref_fraction_of_recycled*global_rir, china_sec_ref_fraction_of_recycled*china_rir, row_sec_ref_fraction_of_recycled*row_rir
        self.hyperparam = input_param.copy()
        
    def update_ref_param(self):
        h = self.hyperparam.copy()
        ref_param = pd.DataFrame()
        for region, rhp in zip(
          ['Global','China','RoW'],
          [self.ref_hyper_param.copy(), self.ref_hyper_param.copy(), self.ref_hyper_param.copy()]):
            total_ref_production = h[region]['Total production']*h[region]['Refining fraction of production']
            secondary_fraction_of_refined_production = h[region]['Secondary fraction of refinery production']
            sr = rhp['Value']['sec ratio']
            scu = rhp['Value']['sec CU']
            pcu = rhp['Value']['pri CU']
            new_sec_prod = total_ref_production * secondary_fraction_of_refined_production
            new_pri_prod = total_ref_production - new_sec_prod
            new_sec_capacity = new_sec_prod/sr/scu
            new_pri_capacity = (new_pri_prod - \
             new_sec_capacity*scu*(1-sr))/pcu
            pri_frac_at_pri_refineries = new_pri_capacity*pcu/(new_pri_capacity*pcu+new_sec_capacity*scu*(1-sr))

            new_ref_hyper_param = rhp.copy()
            new_ref_hyper_param.loc['sec ratio','Value'] = sr
            new_ref_hyper_param.loc['pri cap','Value'] = new_pri_capacity
            new_ref_hyper_param.loc['sec cap','Value'] = new_sec_capacity

            new_ref_hyper_param.loc['pri production',['Value','Notes']] = new_pri_prod,'Production from primary sources, calculated from primary capacity, CU, secondary capacity, CU, (1-SR)'
            new_ref_hyper_param.loc['sec production',['Value','Notes']] = new_sec_prod,'Production from secondary sources, calculated from secondary capacity, CU, SR'

            ref_param = pd.concat([ref_param, new_ref_hyper_param.rename(columns={'Value':region})[region]],axis=1)
        ref_param.loc[:,'Notes'] = new_ref_hyper_param['Notes']
        self.ref_param = ref_param.copy()

    def initialize_ref_stats(self):
        self.regions = [i for i in self.ref_param.columns if i!='Notes']
        if self.hyperparam['Global']['Use regions']: self.regions = [i for i in self.regions if i!='Global']
        rs = []
        for region in self.regions:
            ref_hp = self.ref_param[region]
            ref_stats = ref_stats_init(self.simulation_time,ref_hp)
            rs += [ref_stats]
        self.ref_stats = pd.concat(rs,keys=self.regions,axis=1) 
        if self.hyperparam['Global']['Use regions']:
            self.need_correction = ['Primary capacity','Secondary capacity','Primary production','Secondary production',
                                    'Primary CU','Secondary CU','Secondary ratio']
#             self.ref_stats.drop('Global',axis=1,level=0,inplace=True)
            self.ref_stats = pd.concat([self.ref_stats,pd.concat([self.ref_stats['China'][self.need_correction]+self.ref_stats['RoW'][self.need_correction]],keys=['Global'],axis=1)],axis=1)
            self.ref_stats.loc[:,idx['Global','Primary CU']] = (self.ref_stats['China']['Primary CU']*self.ref_stats['China']['Primary capacity']+self.ref_stats['RoW']['Primary CU']*self.ref_stats['RoW']['Primary capacity'])/(self.ref_stats['China']['Primary capacity']+self.ref_stats['RoW']['Primary capacity'])
            self.ref_stats.loc[:,idx['Global','Secondary CU']] = (self.ref_stats['China']['Secondary CU']*self.ref_stats['China']['Secondary capacity']+self.ref_stats['RoW']['Secondary CU']*self.ref_stats['RoW']['Secondary capacity'])/(self.ref_stats['China']['Secondary capacity']+self.ref_stats['RoW']['Secondary capacity'])
            self.ref_stats.loc[:,idx['Global','Secondary ratio']] = self.ref_stats['Global']['Secondary production']/(self.ref_stats['Global']['Secondary capacity']*self.ref_stats['RoW']['Secondary CU'])
    
    def simulate_refinery_one_year(self):
        rs = []
        for region in self.regions:
            ref_stats = self.ref_stats.copy()[region]
            ref_hp = self.ref_param[region]
            ref_stats = simulate_refinery_production_oneyear(self.i, self.tcrc_series, self.scrap_spread_series[region], 
                                         self.pri_cap_growth_series, self.sec_cap_growth_series[region],
                                         ref_stats, ref_hp, 
                                         sec_coef=0, growth_lag=1, ref_bal = 0, 
                                         pri_CU_ref_bal_elas = 0, sec_CU_ref_bal_elas = 0,
                                         ref_cu_pct_change = 0, ref_sr_pct_change = 0)
            rs += [ref_stats]
        ref_stats = pd.concat(rs, keys=self.regions)
        if self.hyperparam['Global']['Use regions']:
            if 'Global' in ref_stats.index.get_level_values(0).unique():
                ref_stats.drop('Global',axis=1,level=0,inplace=True)
            ref_stats = pd.concat([ref_stats,pd.concat([ref_stats['China'][self.need_correction]+ref_stats['RoW'][self.need_correction]],keys=['Global'])])
            ref_stats.loc[:,idx['Global','Primary CU']] = (ref_stats['China']['Primary CU']*ref_stats['China']['Primary capacity']+ref_stats['RoW']['Primary CU']*ref_stats['RoW']['Primary capacity'])/(ref_stats['China']['Primary capacity']+ref_stats['RoW']['Primary capacity'])
            ref_stats.loc[:,idx['Global','Secondary CU']] = (ref_stats['China']['Secondary CU']*ref_stats['China']['Secondary capacity']+ref_stats['RoW']['Secondary CU']*ref_stats['RoW']['Secondary capacity'])/(ref_stats['China']['Secondary capacity']+ref_stats['RoW']['Secondary capacity'])
            ref_stats.loc[:,idx['Global','Secondary ratio']] = ref_stats['Global']['Secondary production']/(ref_stats['Global']['Secondary capacity']*ref_stats['RoW']['Secondary CU'])
#             (ref_stats['China']['Secondary CU']*ref_stats['China']['Secondary capacity']+ref_stats['RoW']['Secondary CU']*ref_stats['RoW']['Secondary capacity'])/(ref_stats['China']['Secondary capacity']+ref_stats['RoW']['Secondary capacity'])
        self.ref_stats.loc[self.i] = ref_stats
    
    def run(self):
        i = self.i
        if i==self.simulation_time[0]:
            self.update_hyperparam()
            self.update_ref_param()
            self.initialize_ref_stats()
        else:
            self.simulate_refinery_one_year()
    