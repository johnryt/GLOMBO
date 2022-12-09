import numpy as np
import pandas as pd
idx = pd.IndexSlice
from scipy import stats
from math import erf
from IPython.display import display
from datetime import datetime
from joblib import Parallel, delayed

def sum_distribution(year_i, y, mu, sigma, distr):
    '''
    Calculates for a given year the value of a distribution at year_i.

    Args:
        year_i: (int) Year for which to calculate end of life material.
        year: (int) Year at which the material was demanded.
        mu: (float) Mean of distribution.
        sigma: (float) Standard deviation of distribution.
        distr: (str) Type of distribution.

    Returns:
        Amount that year y contributed to material reaching end of life at year_i based on lifetime assumptions.
    '''
    if distr == 'uniform':
        if y == year_i-1: return 1
        else: return 0
    if distr == 'normal':
        return 0.5*(erf((year_i+1-mu-y)/(np.sqrt(2)*sigma)) - erf((year_i-mu-y)/(np.sqrt(2)*sigma)))
        # return stats.norm.cdf(year_i+1, loc=y+mu, scale=sigma) - stats.norm.cdf(year_i, loc=y+mu, scale=sigma)

def reaching_end_of_life(year_i, demand_sectors, lifetime_parameters):
    '''
    Calculates material reaching end of life given previous years' demand and lifetime assumptions.

    Args:
        year_i: (int) Year for which to calculate end of life material.
        demand_sectors: (pd.DataFrame) Demand for every sector and region as far back as possible.
        lifetime_parameters: (pd.DataFrame) Lifetime assumptions with mean, std and type.

    Returns:
        Amount of material reaching end of life at year_i for every region, sector and divided by year.
    '''

    #included for readability
    mus = lifetime_parameters.loc['lifetime'].values #means of lifetimes
    sigmas = lifetime_parameters.loc['sigma'].values #standard deviations of lifetimes
    distrs = lifetime_parameters.loc['distribution'].values #types of distribution
    all_years = demand_sectors.index[demand_sectors.index<year_i] #all years available before year_i

    def fn(regsec,y,year_i,mu,sigma,distr):
        return demand_sectors[regsec][y]*sum_distribution(year_i, y, mu=mu, sigma=sigma, distr=distr)
    def fn2(regsec, i, y):
        return demand_sectors[regsec][y]*sum_distribution(year_i, y, mu=mus[i], sigma=sigmas[i], distr=distrs[i])

    dsc = list(enumerate(demand_sectors.columns))
    big_year_i = [year_i    for y in all_years for i,a in dsc]
    big_y      = [y         for y in all_years for i,a in dsc]
    big_mu     = [mus[i]    for y in all_years for i,a in dsc]
    big_sigma  = [sigmas[i] for y in all_years for i,a in dsc]
    big_distr  = [distrs[i] for y in all_years for i,a in dsc]
    big_regsec = [a         for y in all_years for i,a in dsc]
    outie = np.array(list(map(fn, big_regsec, big_y, big_year_i, big_mu, big_sigma, big_distr)))
    outie = outie.reshape(len(all_years),len(dsc))
    end_of_life = pd.DataFrame(outie, index=all_years, columns=demand_sectors.columns)
    # end_of_life = pd.DataFrame([], index=all_years, columns=demand_sectors.columns)
    # for i, regsec in enumerate(demand_sectors.columns):
    #     end_of_life[regsec] = [demand_sectors[regsec][y]*sum_distribution(year_i, y, mu=mus[i], sigma=sigmas[i], distr=distrs[i]) for y in all_years]

    # outie2 = np.array(Parallel(n_jobs=2)(delayed(fn2)(regsec,i,y) for y in all_years for i,regsec in dsc))
    # outie2 = outie2.reshape(len(all_years),len(dsc))
    # eol2 = pd.DataFrame(outie, index=all_years, columns=demand_sectors.columns)
    # now4 = datetime.now()

    return end_of_life

def intensity_integral(mu, sigma, base=np.exp(1)):
    '''Formula to calculate components of intensity calculation.'''
    return base**mu*np.exp(0.5*sigma**2*np.log(base)**2)

# ∆log(Intensity) = β0SR × 1SR + βSR(∆log(Price)) × 1SR + βGDP(∆log(GDPSR)) × 1SR
def intensity_growth_sec_reg_pooled(mu_int_sec_reg, sigma_int_sec_reg, mu_p_sec_reg, sigma_p_sec_reg, p_growth, mu_gdp_sec_reg, sigma_gdp_sec_reg, gdp_growth):
    # Intercept β0S and β0R
    intercept_integral_sec_reg = intensity_integral(mu_int_sec_reg, sigma_int_sec_reg)
    # Price βS and βR
    price_integral_sec_reg = intensity_integral(mu_p_sec_reg, sigma_p_sec_reg, p_growth)
    # GDP βGDP
    gdp_integral = intensity_integral(mu_gdp_sec_reg, sigma_gdp_sec_reg, gdp_growth)

    return intercept_integral_sec_reg * price_integral_sec_reg * gdp_integral

def intensity_growth_sec_reg_specified_pooled(elas_sec_reg, reg, sec, gdp_growth, p_growth=1):
    mu_int_sec_reg = elas_sec_reg[reg, sec]['Intercept mean']
    sigma_int_sec_reg = elas_sec_reg[reg, sec]['Intercept SD']
    mu_p_sec_reg = elas_sec_reg[reg, sec]['Elasticity mean']
    sigma_p_sec_reg = elas_sec_reg[reg, sec]['Elasticity SD']
    mu_gdp_sec_reg = elas_sec_reg[reg, sec]['GDPPC_Elasticity mean']
    sigma_gdp_sec_reg = elas_sec_reg[reg, sec]['GDPPC_Elasticity SD']

    int_growth = intensity_growth_sec_reg_pooled(mu_int_sec_reg, sigma_int_sec_reg, mu_p_sec_reg, sigma_p_sec_reg, p_growth, mu_gdp_sec_reg, sigma_gdp_sec_reg, gdp_growth)

    return int_growth

def intensity_prediction(year_i, price_series, gdp_growth_prediction, intensity_last, volume_prediction, elas_mat, price_lag=0):
    '''
    Calculates new aluminium intensities.

    Args:
        year_i: (int) Year for which to calculate new intensities.
        price_series: (pd.DataFrame) Prices for aluminium throughout life.
        gdp_growth_prediction: (pd.DataFrame) Prediction, per region for every year, of gdp growth.
        intensity_last: (pd.Series) Previous intensities for every region and sector.
        volume_prediction: (pd.DataFrame) Prediction, per region and sector for every year, of volumes.
        elas_mat: (pd.DataFrame) Fitted parameters for calculating the intensity.

    Returns:
        Aluminium intensities at year_i for every region and sector.
    '''
    regions = volume_prediction.columns.get_level_values(0).unique()
    sectors = volume_prediction.columns.get_level_values(1).unique()
    pooled_intensity_growth_prediction = pd.DataFrame(0, index=[year_i], columns=pd.MultiIndex.from_product([regions, sectors]))

    p_growth = (price_series.loc[year_i-price_lag]+price_series.loc[year_i-price_lag-1])/(price_series.loc[year_i-price_lag-1]+price_series.loc[year_i-price_lag-2])
    if p_growth > 1e12:
        p_growth = 1e12

    for reg in regions:
        gdp_growth = gdp_growth_prediction.loc[year_i, reg]+1

        for sec in sectors:
            int_growth = intensity_growth_sec_reg_specified_pooled(elas_mat, reg, sec, gdp_growth, p_growth)
            pooled_intensity_growth_prediction.loc[year_i, idx[reg, sec]] = int_growth

    intensity_prediction = pooled_intensity_growth_prediction.mul(intensity_last)

    return intensity_prediction
