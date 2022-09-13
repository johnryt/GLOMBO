from demand_functions import *
import numpy as np
import pandas as pd
idx = pd.IndexSlice
from matplotlib import pyplot as plt

class demandModel():
    '''
    collection_rate refers to both collection rate and sorting efficiency for simplicity.

    See demandModel.hyperparam for list of inputs. Run a single year using run_demand()

    Values that can/should be updated between years:
        - commodity_price_series
        - collection_rate
        - fabrication_efficiency
        - hyperparam.loc['new_scrap_fraction']

    Outputs of note:
        - demand
        - semis_demand
        - eol
        - old_scrap_collected

    '''
    def __init__(self, data_folder=None, simulation_time=np.arange(2019,2041), verbosity=0):
        self.simulation_time = simulation_time
        self.verbosity = verbosity
        self.data_folder = "generalization/data" if data_folder is None else data_folder
        self.i = self.simulation_time[0]
        self.init_hyperparams()

        self.collection_rate_price_response = True
        self.scrap_spread = pd.DataFrame(0.1,np.arange(simulation_time[0]-1,simulation_time[-1]+1),['Global','China','RoW'])

        file = f'{self.data_folder}/Demand prediction data-copper.xlsx'
        with pd.ExcelFile(file) as xl:
            self.volumes = pd.read_excel(xl, sheet_name='All sectors', header=[0,1], index_col=0).sort_index().sort_index(axis=1).stack(0).unstack()
            self.all_time = np.sort(np.union1d(self.volumes.index,self.simulation_time))
        self.commodity_price_series = pd.Series(self.hyperparam['Value']['primary_commodity_price'],self.all_time)
        self.scenario_type = ''
        self.collection_rate_duration = 0
        self.collection_rate_pct_change_tot = 1
        self.collection_rate_pct_change_inc = 1

    def init_hyperparams(self):
        hyperparameters = pd.DataFrame()
        hyperparameters.loc['sector_specific_dematerialization_tech_growth','Value'] = -0.03
        hyperparameters.loc['sector_specific_dematerialization_tech_growth','Notes'] = 'should be negative, sets intercept mean parameter for sectors in intensity_param/intensity_parameters, can be float or series with indices from self.sectors'
        hyperparameters.loc['sector_specific_price_response',['Value','Notes']] = -0.06,'should be negative, sets elasticity mean parameter for sectors in intensity_param/intensity_parameters, can be float or series with indices from self.sectors'
        hyperparameters.loc['region_specific_price_response',['Value','Notes']] = -0.1,'should be negative, sets elasticity mean parameter for regions in intensity_param/intensity_parameters, can be float or series with indices from self.regions'
        hyperparameters.loc['intensity_response_to_gdp',['Value','Notes']] = 0.69,'float, should be positive, sets elasticity mean parameter for gdp in intensity_param/intensity_parameters'
        hyperparameters.loc['initial_demand',['Value','Notes']] = 1000,'float/int, demand in first year in kt'
        hyperparameters.loc['volume_growth_rate',['Value','Notes']] = 0.02546855,'float, default 0.02546855, annual growth rate'
        hyperparameters.loc['recycling_input_rate_china',['Value','Notes']] = 0.3, 'float, fraction of demand in the initial year satisfied by recycled inputs; includes refined and direct melt'
        hyperparameters.loc['recycling_input_rate_row',['Value','Notes']] = 0.3, 'float, fraction of demand in the initial year satisfied by recycled inputs; includes refined and direct melt'
        hyperparameters.loc['maximum_collection_rate',['Value','Notes']] = 0.95, 'float, maximum allowable collection rate for any one sector/region. For simplicity, collection rate refers to both collection rate and sorting efficiency here for simplicity'
        hyperparameters.loc['historical_growth_rate',['Value','Notes']] = 0.03, 'growth rate used for demand pre-simulation start date, set to -1 to use a scaled version of copper growth, which was ~3%'
        hyperparameters.loc['china_fraction_demand',['Value','Notes']] = 0.7, 'China fraction of demand, was 0.52645 for copper in 2019'
        hyperparameters.loc['new_scrap_fraction',['Value','Notes']] = 0.5, 'fraction of manufacuring scrap generated that is sold on the market as new scrap. The class variable new_scrap fraction is the fraction of demand that is new scrap, having already accounted for manufacturing efficiency'
        hyperparameters.loc['scrap_to_cathode_eff',:] = 0.99,'efficiency of remelting and/or refining scrap'
        hyperparameters.loc['primary_commodity_price',:] = 5000

        hyperparameters.loc['sector_dist_construction',['Value','Notes']] = 0.2, 'fraction of sector demand'
        hyperparameters.loc['sector_dist_electrical',['Value','Notes']] = 0.2, 'fraction of sector demand'
        hyperparameters.loc['sector_dist_industrial',['Value','Notes']] = 0.2, 'fraction of sector demand'
        hyperparameters.loc['sector_dist_other',['Value','Notes']] = 0.2, 'fraction of sector demand'
        hyperparameters.loc['sector_dist_transport',['Value','Notes']] = 0.2, 'fraction of sector demand'

        hyperparameters.loc['fabrication_efficiency_construction',['Value','Notes']] = 0.80, 'fraction of material demanded by semis that ends up in final product; one minus this value gives the post-industrial scrap generation, of which some fraction is home and some is new'
        hyperparameters.loc['fabrication_efficiency_electrical',['Value','Notes']] = 0.90, 'fraction of material demanded by semis that ends up in final product; one minus this value gives the post-industrial scrap generation, of which some fraction is home and some is new'
        hyperparameters.loc['fabrication_efficiency_industrial',['Value','Notes']] = 0.80, 'fraction of material demanded by semis that ends up in final product; one minus this value gives the post-industrial scrap generation, of which some fraction is home and some is new'
        hyperparameters.loc['fabrication_efficiency_other',['Value','Notes']] = 0.80, 'fraction of material demanded by semis that ends up in final product; one minus this value gives the post-industrial scrap generation, of which some fraction is home and some is new'
        hyperparameters.loc['fabrication_efficiency_transport',['Value','Notes']] = 0.80, 'fraction of material demanded by semis that ends up in final product; one minus this value gives the post-industrial scrap generation, of which some fraction is home and some is new'
        hyperparameters.loc['fabrication_efficiency_improvement_slope',['Value','Notes']] = 0.001, 'fraction fabrication efficiency improvement in each year'

        hyperparameters.loc['lifetime_mean_construction',['Value','Notes']] = 50, 'mean lifetime for sector (normal distribution of lifetime)'
        hyperparameters.loc['lifetime_mean_electrical',['Value','Notes']] = 25, 'mean lifetime for sector (normal distribution of lifetime)'
        hyperparameters.loc['lifetime_mean_industrial',['Value','Notes']] = 20, 'mean lifetime for sector (normal distribution of lifetime)'
        hyperparameters.loc['lifetime_mean_other',['Value','Notes']] = 10, 'mean lifetime for sector (normal distribution of lifetime)'
        hyperparameters.loc['lifetime_mean_transport',['Value','Notes']] = 15, 'mean lifetime for sector (normal distribution of lifetime)'

        hyperparameters.loc['lifetime_sigma_construction',['Value','Notes']] = 15, 'lifetime variance for sector (normal distribution)'
        hyperparameters.loc['lifetime_sigma_electrical',['Value','Notes']] = 7.5, 'lifetime variance for sector (normal distribution)'
        hyperparameters.loc['lifetime_sigma_industrial',['Value','Notes']] = 6, 'lifetime variance for sector (normal distribution)'
        hyperparameters.loc['lifetime_sigma_other',['Value','Notes']] = 3, 'lifetime variance for sector (normal distribution)'
        hyperparameters.loc['lifetime_sigma_transport',['Value','Notes']] = 4.5, 'lifetime variance for sector (normal distribution)'

        hyperparameters.loc['collection_rate_target_construction',['Value','Notes']] = 0.6, 'anticipated collection rate, will be modified so numbers actually align with recycling input rates given'
        hyperparameters.loc['collection_rate_target_electrical',['Value','Notes']] = 0.6, 'anticipated collection rate, will be modified so numbers actually align with recycling input rates given'
        hyperparameters.loc['collection_rate_target_industrial',['Value','Notes']] = 0.6, 'anticipated collection rate, will be modified so numbers actually align with recycling input rates given'
        hyperparameters.loc['collection_rate_target_other',['Value','Notes']] = 0.2, 'anticipated collection rate, will be modified so numbers actually align with recycling input rates given'
        hyperparameters.loc['collection_rate_target_transport',['Value','Notes']] = 0.6, 'anticipated collection rate, will be modified so numbers actually align with recycling input rates given'

        hyperparameters.loc['Use regions',:] = False,'whether to use regions in collection rate response to scrap price or to use the global value'
        hyperparameters.loc['collection_elas_scrap_price',:] = 0.1,'Collection rate elasticity to scrap price'
        hyperparameters.loc['commodity',:] = 'notAu','used to allow for other volumes to be substituted in the case of materials that need it, e.g. Au (set to Au to allow Au volume substitution)'
        hyperparameters.loc['gold_rolling_window',:] = 5, 'number of years to use in the rolling mean for gold volume drivers'
        self.hyperparam = hyperparameters.copy()

    def load_demand_data(self):
        file = f'{self.data_folder}/Demand prediction data-copper.xlsx'
        with pd.ExcelFile(file) as xl:
            self.volumes = pd.read_excel(xl, sheet_name='All sectors', header=[0,1], index_col=0).sort_index().sort_index(axis=1).stack(0).unstack()
            self.gdp_growth = pd.read_excel(xl, sheet_name='GDP growth', header=[0], index_col=0, usecols='A:F').sort_index().sort_index(axis=1).dropna()
            self.intensities = pd.read_excel(xl, sheet_name='Intensity', header=[0,1], index_col=0).sort_index().sort_index(axis=1).stack(0).unstack()
        self.alt_demand = pd.read_excel(f'{self.data_folder}/End use combined data-copper.xlsx',sheet_name='Combined',index_col=0)
        intensity_parameters_cu = pd.read_excel(f'{self.data_folder}/elasticity estimates-copper.xlsx', sheet_name='S+R S intercept only', header=[0], index_col=0).sort_index(axis=1)
        self.intensity_parameters_al = pd.read_excel(f'{self.data_folder}/baseline_scenario_aluminum.xlsx', sheet_name='intensity_parameters', header=[0,1], index_col=0).sort_index().sort_index(axis=1)
        self.original_growth_rate = 1.02546855
        intensity_parameters_cu_original = intensity_parameters_cu.copy()
        self.intensity_parameters = intensity_parameters_cu.copy()

        if self.hyperparam['Value']['commodity'] in ['Au','Ag']:
            gold_vols = pd.read_excel(f'{self.data_folder}/Gold demand volume indicators.xlsx',sheet_name='Volume drivers',index_col=0).loc[2001:]
            gold_vols1 = gold_vols.loc[2001:2019].rolling(self.hyperparam['Value']['gold_rolling_window'],min_periods=1,center=True).mean()
            gold_vols1 = pd.concat([gold_vols1,gold_vols.rolling(5,min_periods=1,center=True).mean().loc[2020:]])
            gold_vols2 = gold_vols.rolling(self.hyperparam['Value']['gold_rolling_window']+2,min_periods=1,center=True).mean()
            self.volumes.loc[:,idx[:,'Industrial']] = self.volumes.loc[:,idx[:,'Industrial']].apply(lambda x: x/x.sum(), axis=1).apply(lambda x: x*gold_vols2['Global cash reserves (USD$2021)'])#['US circulating coin production (million coins)'])
            self.volumes.loc[:,idx[:,'Transport']] = self.volumes.loc[:,idx[:,'Transport']].apply(lambda x: x/x.sum(), axis=1).apply(lambda x: x*gold_vols1['Diamond demand ($B)'])

            gold_dem = pd.read_excel(f'{self.data_folder}/case study data.xlsx',sheet_name='Au',index_col=0).loc[2001:,'Total demand'].astype(float)
            ad = self.alt_demand.copy()
            sectors = ['Construction','Electrical','Industrial','Transport','Other']
            ad.loc[2001:,sectors] = ad.loc[2001:,sectors].apply(lambda x: x/x.sum(),axis=1).apply(lambda x: x*gold_dem.loc[:2019])
            ad.loc[:2000,sectors] *= ad.loc[2001,sectors].sum()/ad.loc[2000,sectors].sum()**2*self.alt_demand.loc[2001,sectors].sum()
            self.alt_demand = ad.copy()

    def update_volumes(self):
        imported_volumes = self.volumes.copy()
        new_volumes = imported_volumes.copy()
        growth_rate = 1+self.hyperparam['Value']['volume_growth_rate']
        imported_growth_rates = imported_volumes/imported_volumes.shift(1)
        new_growth_rates = imported_growth_rates.apply(lambda x: (x)*growth_rate/self.original_growth_rate)

        for year_i in self.simulation_time[1:]:
            new_volumes.loc[year_i] = new_volumes.loc[year_i-1]*new_growth_rates.loc[year_i]
        self.volumes = new_volumes.copy()

    def setup_intensity_param(self):
        intensity_parameters = self.intensity_parameters.copy()
        sectors, regions = self.sectors, self.regions

        if type(self.hyperparam['Value']['sector_specific_dematerialization_tech_growth']) != pd.core.series.Series:
            self.sector_specific_dematerialization_tech_growth = pd.Series(self.hyperparam['Value']['sector_specific_dematerialization_tech_growth'],sectors)
        if type(self.hyperparam['Value']['sector_specific_price_response']) != pd.core.series.Series:
            self.sector_specific_price_response = pd.Series(self.hyperparam['Value']['sector_specific_price_response'],sectors)
        if type(self.hyperparam['Value']['region_specific_price_response']) != pd.core.series.Series:
            self.region_specific_price_response = pd.Series(self.hyperparam['Value']['region_specific_price_response'],regions)

        intensity_parameters.loc[sectors,'Intercept mean'] = self.hyperparam['Value']['sector_specific_dematerialization_tech_growth']
        intensity_parameters.loc[sectors,'Elasticity mean'] = self.hyperparam['Value']['sector_specific_price_response']
        intensity_parameters.loc[regions,'Elasticity mean'] = self.hyperparam['Value']['region_specific_price_response']
        intensity_parameters.loc['GDP','Elasticity mean'] = self.hyperparam['Value']['intensity_response_to_gdp']
        intensity_param = pd.DataFrame(np.nan,self.intensity_parameters_al.index,self.volumes.columns)
        for r in regions:
            for s in sectors:
                intensity_param.loc['Intercept mean',idx[r,s]] = intensity_parameters.loc[[r,s],'Intercept mean'].sum()
                intensity_param.loc['Intercept SD',idx[r,s]] = intensity_parameters.loc[[r,s],'Intercept SD'].mean()
                intensity_param.loc['Elasticity mean',idx[r,s]] = intensity_parameters.loc[[r,s],'Elasticity mean'].sum()
                intensity_param.loc['Elasticity SD',idx[r,s]] = intensity_parameters.loc[[r,s],'Elasticity SD'].mean()
                intensity_param.loc['GDPPC_Elasticity mean',idx[r,s]] = intensity_parameters.loc['GDP','Elasticity mean']+intensity_parameters.loc[r,'Elasticity mean']
                intensity_param.loc['GDPPC_Elasticity SD',idx[r,s]] = (intensity_parameters.loc['GDP','Elasticity SD']+intensity_parameters.loc[r,'Elasticity SD'])/2
        self.intensity_param = intensity_param.copy()

    def setup_region_sector_fractions(self):
        sectors, regions = self.sectors, self.regions
        all_time = np.sort(np.union1d(self.volumes.index,self.simulation_time))
        all_time = np.sort(np.union1d(self.alt_demand.index,all_time))
        cn_frac = self.hyperparam['Value']['china_fraction_demand']
        initial_region_dist = pd.Series((1-cn_frac)/(len(regions)-1),regions)
        initial_region_dist.loc['China'] = cn_frac
        initial_region_dist /= initial_region_dist.sum()

        self.region_dist = pd.DataFrame(np.tile(initial_region_dist.values,len(all_time)).reshape(len(all_time),len(regions)),
                                  all_time, regions)
        cn_frac = self.alt_demand.copy()['China Fraction']
        cn_frac *= self.hyperparam['Value']['china_fraction_demand']/cn_frac.iloc[-1]
        for r in [i for i in regions if i!= 'China']:
            self.region_dist.loc[cn_frac.index,r] = (1-cn_frac)/(len(regions)-1)
        self.region_dist.loc[cn_frac.index,'China'] = cn_frac

        initial_sector_dist = pd.Series(1/len(sectors),sectors)
        initial_sector_dist.loc['Construction'] = self.hyperparam['Value']['sector_dist_construction']
        initial_sector_dist.loc['Electrical']   = self.hyperparam['Value']['sector_dist_electrical']
        initial_sector_dist.loc['Industrial']   = self.hyperparam['Value']['sector_dist_industrial']
        initial_sector_dist.loc['Other']        = self.hyperparam['Value']['sector_dist_other']
        initial_sector_dist.loc['Transport']    = self.hyperparam['Value']['sector_dist_transport']
        self.sector_dist = pd.DataFrame(np.tile(initial_sector_dist.values,len(all_time)).reshape(len(all_time),len(sectors)),
                                  all_time, sectors)

    def setup_intensity_baseline(self):
        intensities = self.intensities.copy()
        for year_i in [i for i in self.volumes.index if i not in intensities.dropna().index and i>min(intensities.index) and i<=self.simulation_time[-1]]:
            #predict the aluminium intensity for each sector based on commodity price and gdp growth
            intensities.loc[year_i] = intensity_prediction(year_i, self.commodity_price_series, self.gdp_growth,
                                                                intensities.loc[year_i-1], self.volumes,
                                                                self.intensity_param).loc[year_i]
        self.intensities = intensities.copy()
        self.demand_glo = intensities*self.volumes

    def convert_intensities_to_unit(self):
        intensities, volumes, region_dist, sector_dist = self.intensities.copy(), self.volumes.copy(), self.region_dist.copy(), self.sector_dist.copy()
        initial_year = self.simulation_time[0]

        cn_frac = self.alt_demand.copy()['China Fraction']
        cn_frac *= self.hyperparam['Value']['china_fraction_demand']/cn_frac.iloc[-1]

        demand_yr = volumes.index[0]
        total_demand = intensities*volumes
        alt_demand = pd.concat([self.alt_demand[total_demand.columns.get_level_values(1).unique()]
                                for i in np.arange(0,len(total_demand.columns.get_level_values(0).unique()))],
                               keys=total_demand.columns.get_level_values(0).unique(),axis=1)
        alt_demand = alt_demand.loc[:demand_yr-1].copy()*total_demand.sum(axis=1)[demand_yr]/alt_demand.sum(axis=1)[demand_yr]
        alt_volumes = self.alt_demand.sum(axis=1)*volumes.sum(axis=1)[demand_yr]/self.alt_demand.sum(axis=1)[demand_yr]
        vol_dist = volumes.loc[demand_yr]/volumes.loc[demand_yr].sum()
        alt_volumes = alt_volumes.apply(lambda x: x*vol_dist).loc[:demand_yr-1]

        total_demand = pd.concat([alt_demand,total_demand.loc[demand_yr:]])
        volumes = pd.concat([alt_volumes,volumes.loc[demand_yr:]])
        self.volumes = volumes.copy()

        total_demand_series = total_demand.sum(axis=1)
        if self.hyperparam['Value']['historical_growth_rate']!=-1:
            total_demand_series.loc[:initial_year-1] = [total_demand_series.loc[initial_year]*(1+self.hyperparam['Value']['historical_growth_rate'])**(t-initial_year) for t in np.arange(total_demand_series.index[0],initial_year)]
        new_demand = total_demand.copy()
        new_demand.loc[:] = 1
        for r in region_dist.columns:
            new_demand.loc[:,idx[r,:]] = new_demand.loc[:,idx[r,:]].apply(lambda x: x*region_dist[r]/new_demand[r].sum(axis=1))
            for s in sector_dist.columns:
                new_demand.loc[:,idx[r,s]] = new_demand.loc[:,idx[r,:]].sum(axis=1)*sector_dist[s]
        new_demand = new_demand.apply(lambda x: x*total_demand_series)
        new_demand *= self.hyperparam['Value']['initial_demand']/new_demand.loc[initial_year].sum()
        self.new_demand = new_demand.copy()
        new_intensities = new_demand/volumes

        for year_i in self.simulation_time[1:]:
            new_intensities.loc[year_i] = intensity_prediction(year_i, self.commodity_price_series, self.gdp_growth,
                                                            new_intensities.loc[year_i-1], volumes,
                                                            self.intensity_param).loc[year_i]
            new_demand.loc[year_i] = new_intensities.loc[year_i]*volumes.loc[year_i]
        self.demand_glo, self.intensities = new_demand.copy(), new_intensities.copy()

        self.demand = pd.DataFrame(np.nan,self.demand_glo.index,pd.MultiIndex.from_product([['China','RoW'],self.demand_glo.columns.get_level_values(1).unique()]))
        self.demand.loc[:,idx['China',:]] = self.demand_glo.copy().loc[:,idx['China',:]]
        row = self.demand_glo[[i for i in self.demand_glo.columns if 'China' not in i]].groupby(level=1,axis=1).sum()
        row = pd.concat([row],keys=['RoW'],axis=1)
        self.demand.loc[:,idx['RoW',:]] = row

    def run_intensity_prediction(self):
        intensity = intensity_prediction(self.i,
                                         self.commodity_price_series, self.gdp_growth,
                                         self.intensities.loc[self.i-1], self.volumes,
                                         self.intensity_param).loc[self.i]
        self.intensities.loc[self.i] = intensity.copy()
        self.demand_glo.loc[self.i] = intensity*self.volumes.loc[self.i]
        cn_row_demand = pd.concat([self.demand_glo['China'].loc[self.i],
                                   self.demand_glo[self.row].groupby(level=1,axis=1).sum().loc[self.i]],keys=['China','RoW'])
        self.demand.loc[self.i] = cn_row_demand

    def initialize_lifetimes(self):
        h = self.hyperparam['Value'].copy()
        lifetimes = pd.DataFrame(np.nan, ['lifetime','sigma','distribution'],self.demand.columns)
        lifetimes.loc['lifetime',idx[:,'Construction']] = h['lifetime_mean_construction']
        lifetimes.loc['lifetime',idx[:,'Electrical']] = h['lifetime_mean_electrical']
        lifetimes.loc['lifetime',idx[:,'Industrial']] = h['lifetime_mean_industrial']
        lifetimes.loc['lifetime',idx[:,'Other']] = h['lifetime_mean_other']
        lifetimes.loc['lifetime',idx[:,'Transport']] = h['lifetime_mean_transport']

        lifetimes.loc['sigma',idx[:,'Construction']] = h['lifetime_sigma_construction']
        lifetimes.loc['sigma',idx[:,'Electrical']] = h['lifetime_sigma_electrical']
        lifetimes.loc['sigma',idx[:,'Industrial']] = h['lifetime_sigma_industrial']
        lifetimes.loc['sigma',idx[:,'Other']] = h['lifetime_sigma_other']
        lifetimes.loc['sigma',idx[:,'Transport']] = h['lifetime_sigma_transport']

        lifetimes.loc['distribution',idx[:,'Construction']] = 'normal'
        lifetimes.loc['distribution',idx[:,'Electrical']] = 'normal'
        lifetimes.loc['distribution',idx[:,'Industrial']] = 'normal'
        lifetimes.loc['distribution',idx[:,'Other']] = 'normal'
        lifetimes.loc['distribution',idx[:,'Transport']] = 'normal'
        self.lifetimes = lifetimes.copy()

    def initialize_collection(self):
        '''
        Start with ratios, then move to actual rates based on recycling input rate.
        Assuming that demand is the total demand associated with the end products, since intensity is the
        metal required to make the products. This means that the demand input for old scrap generation
        has to be corrected for new scrap that is sold and does not go into their products
        '''
        simulation_time = self.simulation_time
        h = self.hyperparam['Value'].copy()

        self.demand_no_new = self.demand*(1-self.new_scrap_fraction)
        initial_eol = reaching_end_of_life(simulation_time[0],self.demand_no_new,self.lifetimes).sum().unstack()
        self.initial_eol = initial_eol.copy()
        self.eol = pd.concat([initial_eol],keys=[simulation_time[0]])

        init_demand = self.demand.loc[simulation_time[0]].unstack()
        self.init_demand = init_demand.copy()
        init_new_scrap = init_demand*self.new_scrap_fraction.loc[simulation_time[0]].unstack()
        self.init_new_scrap = init_new_scrap.copy()

        collection_rate = pd.Series(np.nan,self.demand.columns.levels[1])

        cr = []
        regions = self.demand.columns.get_level_values(0).unique()
        for region in regions:
            recycling_input_rate = pd.DataFrame(np.nan,regions,self.init_demand.columns)
            for reg in regions:
                recycling_input_rate.loc[reg] = h['recycling_input_rate_'+reg.lower()] / self.hyperparam['Value']['scrap_to_cathode_eff']
            max_recycling_input_rate = (initial_eol.sum().sum()+init_new_scrap.sum().sum())/init_demand.sum().sum()*h['maximum_collection_rate']
            if recycling_input_rate.max().max() > max_recycling_input_rate:
                if region==regions[0]:
                    for reg in recycling_input_rate.idxmax().unique():
                        print(reg,'recycling input rate exceeds maximum given the existing sector lifetimes.\nRecycling input rate set to maximum ({:.1f}% collection) value: {:0.3f}'.format(100*h['maximum_collection_rate'],max_recycling_input_rate.max().max()))
                recycling_input_rate[recycling_input_rate>max_recycling_input_rate] = max_recycling_input_rate
                self.hyperparam.loc['recycling_input_rate_'+region.lower(),'Value']=max_recycling_input_rate * self.hyperparam['Value']['scrap_to_cathode_eff']
                collection_rate.loc[:] = h['maximum_collection_rate']
            else:
                collection_rate.loc['Construction'] = h['collection_rate_target_construction']
                collection_rate.loc['Electrical']   = h['collection_rate_target_electrical']
                collection_rate.loc['Industrial']   = h['collection_rate_target_industrial']
                collection_rate.loc['Other']        = h['collection_rate_target_other']
                collection_rate.loc['Transport']    = h['collection_rate_target_transport']

            target_collected = recycling_input_rate*init_demand - init_new_scrap
            flag = 0
            if (target_collected[target_collected!=0]<1e-10).any().any():
                flag = 1
                prev_new_scrap_fraction = self.hyperparam['Value'].copy()['new_scrap_fraction']
                ('new scrap rate is too high or recycling input rate is too low, getting a negative number for old scrap collection target.')
            n=0
            while (target_collected[target_collected!=0]<1e-10).any().any():
                self.hyperparam.loc['new_scrap_fraction','Value']*=0.99
                self.initialize_new_scrap_fraction()
                init_new_scrap = init_demand*self.new_scrap_fraction.loc[simulation_time[0]].unstack()
                self.init_new_scrap = init_new_scrap.copy()
                target_collected = recycling_input_rate*init_demand - init_new_scrap
                n+=1
                if n>1e4:
                    print('new scrap rate issue, could be a problem if target_collected<0',target_collected)
                    break
            if flag!=0: print('New scrap rate reset to from {:.4f} to {:.4f}'.format(prev_new_scrap_fraction,self.hyperparam['Value']['new_scrap_fraction']))

            self.target_collected = target_collected

            self.cr = collection_rate.copy()
            self.cr = self.update_collection_rate(region)
            n=0
            while (self.cr>h['maximum_collection_rate']+1e-10).any().any():
                self.cr[self.cr>h['maximum_collection_rate']+1e-10] *= 0.99
                self.cr = self.update_collection_rate(region)
                n+=1
                if n>1e4:
                    print('collection rates exceeding maximum_collection_rate:',self.cr)
                    break
            cr += [self.cr]
        self.collection_rate = pd.concat(cr,keys=regions,axis=1).T

        self.old_scrap_collected = pd.concat([self.collection_rate * initial_eol],keys=[simulation_time[0]])
        self.new_scrap_collected = self.demand*self.new_scrap_fraction
        self.scrap_collected = self.new_scrap_collected.stack(0).loc[idx[self.simulation_time[0],:],:]+self.old_scrap_collected
        self.collection_rate = pd.concat([self.collection_rate for i in simulation_time],keys=simulation_time)

    def initialize_collection_scenario(self):
        self.additional_scrap = self.demand.copy().stack(0)
        self.additional_scrap.loc[:] = 0
        multiplier_array = np.append(
            np.linspace(1,self.collection_rate_pct_change_tot,self.collection_rate_duration+1),
            [self.collection_rate_pct_change_tot*self.collection_rate_pct_change_inc**j for j in np.arange(0,len(self.simulation_time)-self.collection_rate_duration-1)])
        if self.scenario_type in ['scrap supply','both'] and self.collection_rate_duration>0:
            if self.collection_rate_price_response==False:
                for yr,mul in zip(self.simulation_time,multiplier_array):
                    self.collection_rate.loc[idx[yr,:],:]*=mul
            else:
                multiplier_array-=1
                start = self.simulation_time[0]
                ph = self.eol.stack().unstack(1).unstack()
                ph2 = self.scrap_collected.stack().unstack(1).unstack()
                demand_adj = self.demand.apply(lambda x: x.sum()*ph.loc[start]/ph.loc[start].sum(),axis=1)
        #         demand_adj = self.eol.stack().unstack(1).unstack() * self.demand.loc[start,:].sum().sum() / self.eol.loc[start].sum().sum()
                self.additional_scrap = demand_adj.loc[self.simulation_time[0]:].apply(lambda x: ph2.loc[self.simulation_time[0]]*multiplier_array[x.name-self.simulation_time[0]],axis=1)
                self.additional_scrap.loc[self.simulation_time[0]-1,:] = 0
                self.additional_scrap = self.additional_scrap.sort_index()
                self.additional_scrap = self.additional_scrap.stack(0)

    def initialize_fabrication_efficiency(self):
        h = self.hyperparam['Value'].copy()
        simulation_time = self.simulation_time
        self.fabrication_efficiency = self.demand.copy()
        for sector in self.fabrication_efficiency.columns.get_level_values(1).unique():
            self.fabrication_efficiency.loc[:,idx[:,sector]] = h['_'.join(['fabrication_efficiency',sector.lower()])]
        change = pd.Series(1,self.fabrication_efficiency.index)
        change.loc[simulation_time] = [1+x*h['fabrication_efficiency_improvement_slope'] for x in np.arange(0,len(simulation_time))]
        self.fabrication_efficiency = self.fabrication_efficiency.apply(lambda x: x*change)
        self.fabrication_efficiency[self.fabrication_efficiency>0.95] = 0.95

    def initialize_new_scrap_fraction(self):
        self.new_scrap_fraction = (1-self.fabrication_efficiency)*self.hyperparam['Value']['new_scrap_fraction']

    def update_collection_rate(self,region):
        w1 = self.target_collected.sum().sum()/(self.cr/self.cr['Construction']*self.initial_eol.sum()).sum()
        new_collect_rate = w1*self.cr/self.cr['Construction']
        return new_collect_rate

    def scrap_generation_collection(self):
        i = self.i
        h = self.hyperparam['Value']
        self.demand_no_new = self.demand*(1-self.new_scrap_fraction)
        eol = reaching_end_of_life(self.i,self.demand_no_new,self.lifetimes).sum().unstack()

        old_scrap_collected = self.collection_rate.loc[i] * eol
        old_scrap_collected = pd.concat([old_scrap_collected],keys=[i])

        eol = pd.concat([eol],keys=[i])
        self.eol = pd.concat([self.eol,eol])
        self.old_scrap_collected = pd.concat([self.old_scrap_collected,old_scrap_collected])

        if self.collection_rate_price_response and i>self.simulation_time[0]:
            if not hasattr(self,'additional_scrap'):
                print('WARNING: additional_scrap variable did not exist and is now being initialized and set to zero')
                self.additional_scrap = pd.DataFrame(2,
                                                    pd.MultiIndex.from_product([self.simulation_time,['China','RoW']]),
                                                    self.eol.columns)
            temp_old_scrap = self.old_scrap_collected.copy()
            temp_old_scrap.loc[idx[:i-1,:],:] -= self.additional_scrap.loc[idx[:i-1,:],:]
            temp_collection_rate = temp_old_scrap/self.eol
            self.old_scrap_collected.loc[idx[i-1,:],:] -= self.additional_scrap.loc[idx[i-1,:],:]

            scrap_price_l1 = self.commodity_price_series[i-1]-self.scrap_spread.loc[i-1]
            scrap_price_l2 = self.commodity_price_series[i-2]-self.scrap_spread.loc[i-2]
            scrap_price_l1[scrap_price_l1<0] = 1e-6
            scrap_price_l2[scrap_price_l2<0] = 1e-6
            if not hasattr(self,'scrap_price_l1'):
                self.scrap_price_l1 = pd.Series(np.nan,self.simulation_time)
                self.scrap_price_l2 = pd.Series(np.nan,self.simulation_time)
            self.scrap_price_l1.loc[i] = scrap_price_l1['Global']
            self.scrap_price_l2.loc[i] = scrap_price_l2['Global']
            if h['Use regions']:
                temp_collection_rate.loc[idx[i,'China'],:] = temp_collection_rate.loc[idx[i-1,'China'],:].rename({i-1:i})*(scrap_price_l1['China']/scrap_price_l2['China'])**h['collection_elas_scrap_price']
                temp_collection_rate.loc[idx[i,'RoW'],:]   = temp_collection_rate.loc[idx[i-1,'RoW'],:].rename({i-1:i})*(scrap_price_l1['RoW']/scrap_price_l2['RoW'])**h['collection_elas_scrap_price']
            else:
                multiple = (scrap_price_l1['Global']/scrap_price_l2['Global'])**h['collection_elas_scrap_price']
                temp_collection_rate.loc[idx[i,:],:] = temp_collection_rate.loc[idx[i-1,:],:].rename({i-1:i})*multiple
            max_cr = h['maximum_collection_rate']
            temp_collection_rate[temp_collection_rate>max_cr] = max_cr
            self.temp_collection_rate = temp_collection_rate.copy()
            self.old_scrap_collected.loc[idx[i,:],:] = self.eol.loc[idx[i,:],:]*temp_collection_rate.loc[idx[i,:],:].rename({i-1:i})
            self.old_scrap_collected.loc[idx[i-1:i,:],:] += self.additional_scrap.loc[idx[i-1:i,:],:]
            self.collection_rate.loc[idx[:i,:],:] = self.old_scrap_collected.loc[idx[:i,:],:]/self.eol.loc[idx[:i,:],:]
            if (self.collection_rate.loc[i]>max_cr).any().any() and self.verbosity>0:
                print('WARNING {}: {}/10 sector-region collection rates (max. {:.3f}) exceed maximum allowable ({:.3f})'.format(i,(self.collection_rate.loc[i]>max_cr).sum().sum(),self.collection_rate.loc[i].max().max(), max_cr))

        self.new_scrap_collected = self.demand*self.new_scrap_fraction
        self.new_scrap_collected = self.new_scrap_collected.loc[self.old_scrap_collected.index.get_level_values(0).unique()].stack(0)
        self.scrap_collected = self.new_scrap_collected+self.old_scrap_collected

    def initialize_demand(self):
        self.update_volumes()
        self.setup_intensity_param()
        self.setup_region_sector_fractions()
        self.setup_intensity_baseline()
        self.convert_intensities_to_unit()

    def run(self):
        if self.i==self.simulation_time[0]:
            simulation_time = self.simulation_time
            self.load_demand_data()
            self.all_time = np.sort(np.union1d(self.volumes.index,self.simulation_time))
            if simulation_time[0]-1 not in self.commodity_price_series.index:
                self.commodity_price_series.loc[simulation_time[0]-1] = self.commodity_price_series.loc[simulation_time[0]]
                self.commodity_price_series = self.commodity_price_series.sort_index()
            self.regions = list(self.intensities.columns.get_level_values(0).unique())
            self.sectors = list(self.intensities.columns.get_level_values(1).unique())

            self.row = [i for i in self.volumes.columns.levels[0] if i!='China']

            self.initialize_demand()
            self.initialize_lifetimes()
            self.initialize_fabrication_efficiency()
            self.initialize_new_scrap_fraction()
            self.initialize_collection()
            self.initialize_collection_scenario()
        else:
            self.run_intensity_prediction()
            self.scrap_generation_collection()
        self.scrap_supply = self.scrap_collected.unstack().groupby(level=1,axis=1).sum()
        self.scrap_supply.loc[:,'Global'] = self.scrap_supply['China']+self.scrap_supply['RoW']
