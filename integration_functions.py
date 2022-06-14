import numpy as np
import pandas as pd
idx = pd.IndexSlice
from matplotlib import pyplot as plt
import os

def create_result_df(integ):
    '''
    takes Integration object, returns regional results.
    '''
    reg = 'Global'
    reg_results = pd.Series(np.nan,['Global','China','RoW'],dtype=object)
    old_new_mines = pd.concat([
        integ.mining.ml.loc[idx[:,:20190000],'Production (kt)'].groupby(level=0).sum(),
        integ.mining.ml.loc[idx[:,20200000:],'Production (kt)'].groupby(level=0).sum(),
        integ.mining.ml.loc[idx[:,:20190000],['Production (kt)','Head grade (%)']].product(axis=1).groupby(level=0).sum()/integ.mining.ml.loc[idx[:,:20190000],'Production (kt)'].groupby(level=0).sum(),
        integ.mining.ml.loc[idx[:,20200000:],['Production (kt)','Head grade (%)']].product(axis=1).groupby(level=0).sum()/integ.mining.ml.loc[idx[:,20200000:],'Production (kt)'].groupby(level=0).sum(),
        integ.mining.ml.loc[idx[:,:20190000],['Production (kt)','Minesite cost (USD/t)']].product(axis=1).groupby(level=0).sum()/integ.mining.ml.loc[idx[:,:20190000],'Production (kt)'].groupby(level=0).sum(),
        integ.mining.ml.loc[idx[:,20200000:],['Production (kt)','Minesite cost (USD/t)']].product(axis=1).groupby(level=0).sum()/integ.mining.ml.loc[idx[:,20200000:],'Production (kt)'].groupby(level=0).sum(),
        integ.mining.ml.loc[idx[:,:20190000],['Production (kt)','Total cash margin (USD/t)']].product(axis=1).groupby(level=0).sum()/integ.mining.ml.loc[idx[:,:20190000],'Production (kt)'].groupby(level=0).sum(),
        integ.mining.ml.loc[idx[:,20200000:],['Production (kt)','Total cash margin (USD/t)']].product(axis=1).groupby(level=0).sum()/integ.mining.ml.loc[idx[:,20200000:],'Production (kt)'].groupby(level=0).sum(),
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
        results = pd.concat([integ.total_demand.loc[2018:,reg],integ.scrap_demand.loc[2018:,reg],integ.scrap_supply[reg],
               integ.concentrate_demand[reg],integ.concentrate_supply,
               integ.mining.ml.loc[:,['Production (kt)','Head grade (%)']].product(axis=1).groupby(level=0).sum()/integ.mining.ml.loc[:,'Production (kt)'].groupby(level=0).sum(),
               integ.mining.ml.loc[:,['Production (kt)','Minesite cost (USD/t)']].product(axis=1).groupby(level=0).sum()/integ.mining.ml.loc[:,'Production (kt)'].groupby(level=0).sum(),
               integ.mining.ml.loc[:,['Production (kt)','Total cash margin (USD/t)']].product(axis=1).groupby(level=0).sum()/integ.mining.ml.loc[:,'Production (kt)'].groupby(level=0).sum(),
               old_new_mines['Old mine prod.'],old_new_mines['New mine prod.'],
               old_new_mines['Old mine grade'],old_new_mines['New mine grade'],
               old_new_mines['Old mine cost'],old_new_mines['New mine cost'],
               old_new_mines['Old mine margin'],old_new_mines['New mine margin'],
               integ.refined_demand.loc[2018:,reg],integ.refined_supply[reg],
               integ.secondary_refined_demand.loc[:,reg],integ.direct_melt_demand.loc[2018:,reg],
               integ.scrap_spread[reg],integ.tcrc,integ.primary_commodity_price,
               integ.refine.ref_stats[reg]['Primary CU'], integ.refine.ref_stats[reg]['Secondary CU'],
               integ.refine.ref_stats[reg]['Secondary ratio'],
               integ.refine.ref_stats[reg]['Primary capacity'], integ.refine.ref_stats[reg]['Secondary capacity'],
               integ.refine.ref_stats[reg]['Primary production'], integ.refine.ref_stats[reg]['Secondary production'],
               integ.additional_direct_melt[reg],addl_scrap[reg]
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
                    'Additional direct melt','Additional scrap'])
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
    True means equivalent; 
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
    
    '''
    def __init__(self,
                 pkl_filename='integration_big_df.pkl', 
                 changing_base_parameters_series=0, 
                 params_to_change=0, 
                 n_per_param=5, 
                 notes='Initial run',
                 simulation_time = np.arange(2019,2041),
                 byproduct=False,
                 verbosity=0,
                 param_scale=0.5,
                 scenarios=[''],
                 OVERWRITE=False):
        '''
        scenario_name takes the form 00_11_22_33_44
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
        
        e.g. ss_pr_1yr_1%tot_0%inc
        
        for 22-44, an additional X should be placed at
         the end when 00==both, describing the sd values
         e.g. ss_pr_1yr1_1%tot1_0%inc0
         
        Can also have 11 as nono, prno, nopr, or prpr to
         control the ss and sd price response individually
         (in that order)
        '''
        self.simulation_time = simulation_time
        self.byproduct = byproduct
        self.verbosity = verbosity
        self.param_scale = param_scale
        self.pkl_filename = pkl_filename
        self.changing_base_parameters_series = changing_base_parameters_series
        self.params_to_change = params_to_change
        self.n_per_param = n_per_param
        self.notes = notes
        self.scenarios = scenarios
        self.overwrite = OVERWRITE
        if overwrite: print('WARNING, YOU ARE OVERWRITING AN EXISTING FILE')
        
    def initialize_big_df(self):
        if os.path.exists(self.pkl_filename) and not self.overwrite:
            big_df = pd.read_pickle(self.pkl_filename)
        else:
            mod = Integration(simulation_time=self.simulation_time,verbosity=self.verbosity,byproduct=self.byproduct,scenario_name='')
            for base in self.changing_base_parameters_series.index:
                mod.hyperparam.loc[base,'Value'] = self.changing_base_parameters_series[base]
            self.mod = mod
            mod.run()
            big_df = pd.DataFrame(np.nan,index=[
                'version','notes','hyperparam','mining.hyperparam','refine.hyperparam','demand.hyperparam','results'
            ],columns=[])
            reg_results = create_result_df(mod)
            big_df.loc[:,0] = np.array([mod.version, self.notes, mod.hyperparam, mod.mining.hyperparam, 
                                        mod.refine.hyperparam, mod.demand.hyperparam, reg_results],dtype=object)
            big_df.to_pickle(self.pkl_filename)
            self.mod = mod
        self.big_df = big_df.copy()

    def update_changing_base_parameters_series(self):
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

    def get_params_to_change(self):
        if type(self.params_to_change)==int:
            mod = Integration(simulation_time=self.simulation_time,verbosity=self.verbosity,byproduct=self.byproduct)
            for base in self.changing_base_parameters_series.index:
                mod.hyperparam.loc[base,'Value'] = self.changing_base_parameters_series[base]
            self.params_to_change = pd.concat([
                mod.hyperparam.loc['price elasticities':'determining model structure'].dropna(),
                mod.hyperparam.loc['mining only':].dropna(how='all')])
        
    def run(self):
        '''
        pkl_filename: .pkl file path. If the file does not exist,
           a new one will be created. If the file already exists,
           scenarios will be appended to the existing dataframe.
        changing_base_parameters_series: 
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
                mod = Integration(simulation_time=self.simulation_time,verbosity=self.verbosity,byproduct=self.byproduct)
                self.hyperparam_copy = mod.hyperparam.copy()

                ###### CHANGING BASE PARAMETERS ######
                for base in changing_base_parameters_series.index:
                    mod.hyperparam.loc[base,'Value'] = changing_base_parameters_series[base]
                    if count==0:
                        print(base,changing_base_parameters_series[base])

                ###### UPDATING FROM params_to_change_ind ######
                mod.hyperparam.loc[i,'Value'] = val
                print(f'Scenario {count}/{total_num_scenarios}: {i} = {val}')
                count += 1
                
                self.check_run_append(mod)
                self.mod = mod
    
    def run_monte_carlo(self, n_scenarios):
        self.update_changing_base_parameters_series()
        self.initialize_big_df()
        mod = Integration(simulation_time=self.simulation_time,verbosity=self.verbosity,byproduct=self.byproduct)
        scenario_params_dont_change = ['collection_rate_price_response','direct_melt_price_response']
        params_to_change = [i for i in mod.hyperparam.dropna(how='all').index if ('elas' in i or 'response' in i or 'growth' in i or 'improvements' in i) and i not in scenario_params_dont_change]
        
        for n in np.arange(0,n_scenarios):
            if self.verbosity>-1: 
                print(f'Scenario {n+1}/{n_scenarios}')
            for enum,scenario_name in enumerate(self.scenarios):
                if self.verbosity>-1:
                    print(f'\tSub-scenario {enum+1}/{len(self.scenarios)}: {scenario_name} checking if exists...')
                mod = Integration(simulation_time=self.simulation_time,verbosity=self.verbosity,byproduct=self.byproduct,scenario_name=scenario_name)
                self.mod = mod
                self.notes = scenario_name
                
                ###### CHANGING BASE PARAMETERS ######
                changing_base_parameters_series = self.changing_base_parameters_series.copy()
                for base in changing_base_parameters_series.index:
                    mod.hyperparam.loc[base,'Value'] = changing_base_parameters_series[base]
                    if n==0 and self.verbosity>0:
                        print(base,changing_base_parameters_series[base])
                self.hyperparam_copy = mod.hyperparam.copy()

                ###### UPDATING MONTE CARLO PARAMETERS ######
                rs = 220530+n
                values = stats.uniform.rvs(loc=0,scale=1,size=len(params_to_change),random_state=rs)
                new_param_series = pd.Series(values, params_to_change)
                if self.verbosity>0:
                    print(params_to_change)
                new_param_series.loc['sector_specific_dematerialization_tech_growth'] *= 0.15
                new_param_series.loc['sector_specific_price_response'] *= 0.15
                new_param_series.loc['region_specific_price_response'] *= 0.15
                # ^ these values should be small, since small changes make big changes
                for param in params_to_change:
                    if type(mod.hyperparam['Value'][param])!=bool:
                        mod.hyperparam.loc[param,'Value'] = new_param_series[param]*np.sign(mod.hyperparam.loc[param,'Value'])
                    else:
                        mod.hyperparam.loc[param,'Value'] = new_param_series[param]
                self.check_run_append(mod)
    
    def run_monte_carlo_across_base(self, n_scenarios):
        self.update_changing_base_parameters_series()
        self.initialize_big_df()
        mod = Integration(simulation_time=self.simulation_time,verbosity=self.verbosity,byproduct=self.byproduct)
        params_to_change = [i for i in mod.hyperparam.dropna().index if 'elas' in i]
        for n in np.arange(0,n_scenarios):
            if self.verbosity>-1: 
                print(f'Scenario {n+1}/{n_scenarios}')
            mod = Integration(simulation_time=self.simulation_time,verbosity=self.verbosity,byproduct=self.byproduct)
            
            ###### CHANGING BASE PARAMETERS ######
            changing_base_parameters_series = self.changing_base_parameters_series.copy()
            for base in np.intersect1d(mod.hyperparam.index, changing_base_parameters_series.index):
                mod.hyperparam.loc[base,'Value'] = changing_base_parameters_series[base]
                if n==0 and self.verbosity>0:
                    print(base,changing_base_parameters_series[base])
            self.hyperparam_copy = mod.hyperparam.copy()
            
            ###### UPDATING MONTE CARLO PARAMETERS ######
            rs = 220530+n
            values = stats.uniform.rvs(loc=0,scale=1,size=len(params_to_change),random_state=rs)
            new_param_series = pd.Series(values, params_to_change)
            new_param_series
            mod.hyperparam.loc[params_to_change,'Value'] = new_param_series*np.sign(mod.hyperparam.loc[params_to_change,'Value'])
            self.check_run_append(mod)
    
    def check_run_append(self, mod):
        big_df = pd.read_pickle(self.pkl_filename)
        potential_append = pd.DataFrame(np.array([mod.version, self.notes, mod.hyperparam, [], 
                            [], [], []],dtype=object)
                                         ,index=[
                                'version','notes','hyperparam','mining.hyperparam','refine.hyperparam','demand.hyperparam','results'
                            ],columns=[max(big_df.columns)+1])
        if check_equivalence(big_df, potential_append)[0]:
            if self.verbosity>-1:
                print('\tScenario does not already exist, running...')
            self.mod = mod
            mod.run()

            if hasattr(self,'val'):
                notes = self.notes+ f', {i}={self.val}'
            else:
                notes = self.notes+''
            ind = [j for j in self.hyperparam_copy.index if type(self.hyperparam_copy['Value'][j]) not in [np.ndarray,list]]
            z = self.hyperparam_copy['Value'][ind].dropna()!=mod.hyperparam['Value'][ind].dropna()
            z = [j for j in z[z].index if j!=i]
            if len(z)>0:
                for zz in z:
                    notes += ', {}={}'.format(zz,mod.hyperparam['Value'][zz])

            reg_results = create_result_df(mod)
            potential_append = pd.DataFrame(np.array([mod.version, notes, mod.hyperparam, mod.mining.hyperparam, 
                                mod.refine.hyperparam, mod.demand.hyperparam, reg_results],dtype=object)
                                             ,index=[
                                    'version','notes','hyperparam','mining.hyperparam','refine.hyperparam','demand.hyperparam','results'
                                ],columns=[max(big_df.columns)+1])
            big_df = pd.concat([big_df,potential_append],axis=1)
            big_df.to_pickle(self.pkl_filename)
            if self.verbosity>-1:
                print('\tScenario successfully saved\n')
            self.mod = mod
        else:
            if self.verbosity>-1:
                print('\tScenario already exists\n')
            
#         direct_melt_elas_scrap_spread
#         collection_elas_scrap_price
        