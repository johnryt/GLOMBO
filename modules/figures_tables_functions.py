try:
    from modules.Many import *
    from modules.useful_functions import *
    from modules.demand_class import demandModel
except:
    from Many import *
    from useful_functions import *
    from demand_class import demandModel
init_plot2()


from statsmodels.stats.diagnostic import lilliefors
from scipy.special import ndtri

import pandas as pd
import numpy as np
import os
import zipfile
idx = pd.IndexSlice
import matplotlib.pyplot as plt
import statsmodels.api as sm
import matplotlib as mpl
from linearmodels import PanelOLS,RandomEffects
from linearmodels.panel import compare
from itertools import combinations
from scipy import stats
import seaborn as sns
import warnings
from statsmodels.stats.outliers_influence import variance_inflation_factor
from IPython.display import display
from string import printable as character_list
import statsmodels.tsa.stattools as ts

def cointegration_test(y, x):
    # Step 1: regress one variable on the other
    ols_result = sm.OLS(y, x).fit()
    # Step 2: obtain the residual
    residual = ols_result.resid
    # Step 3: apply Augmented Dickey-Fuller test to see whether# the residual is unit root
    return ts.adfuller(residual)

def pval_to_star(pval, no_star_cut=0.1, period_cut=0.05, one_star_cut=0.01, two_star_cut=0.001):
    """
    Converts a value from its numerical value to a string where:
    *** <= 0.001 < ** <= 0.01 < * <= 0.05 < (.) <= 0.1
    """
    pval_str = '***' if pval <= two_star_cut else '**' if pval <= one_star_cut else '*' if \
        pval <= period_cut else '(.)' if pval <= no_star_cut else ''
    return pval_str

def make_parameter_mean_std_table(many, n_best, value_in_parentheses='standard error', stars='ttest',
                                rand_size=1000, how_many_tests=100):
    """
    many: Many instance, from tuning, either the integ object or full thing is has integ
        object
    n_best: int, number of best scenarios to use in the calculation of mean and std error/
        std dev / variance
    value_in_parentheses: str, can be `standard error`, `standard deviation`, or `variance`
    stars: str, can be `ttest`, `uniform-ks`, `both`, `uniform-sw`, `uniform-lilliefors`. 
        uniform-sw (Shapiro-Wilk) and uniform-lilliefors use conversion from uniform to
        normal and test for normality, avoiding the q-value issues.
    """
    if hasattr(many, 'integ'):
        rmse_df = many.integ.rmse_df_sorted.copy()
    else:
        rmse_df = many.rmse_df_sorted.copy()
    r = [i for i in rmse_df.index.get_level_values(1).unique()
        if np.any([j in i for j in ['score', 'R2', 'RMSE', 'region_specific_price_response']])]
    rmse_df.drop(r, inplace=True, level=1)
    best_n = rmse_df.loc[:, :n_best]
    best_n.loc[idx[:,'primary_oge_scale'],:] *= -1
    means = best_n.mean(axis=1).unstack(0).fillna('')
    if value_in_parentheses == 'standard error':
        stds = best_n.sem(axis=1).unstack(0).fillna('')
    elif value_in_parentheses == 'standard deviation':
        stds = best_n.std(axis=1).unstack(0).fillna('')
    elif value_in_parentheses == 'variance':
        stds = best_n.var(axis=1).unstack(0).fillna('')
    elif value_in_parentheses in ['None',None]:
        stds = best_n.var(axis=1).unstack(0)
        stds.loc[:] = ''
    
    if stars=='ttest':
        pvals = best_n.apply(lambda x: stats.ttest_1samp(x,popmean=0)[1], axis=1)
        display(pvals)
    elif stars=='uniform-ks':
        pvals = get_difference_from_uniform(best_n, rand_size=rand_size, how_many_tests=how_many_tests)
    elif stars=='both':
        pvals = best_n.apply(lambda x: stats.ttest_1samp(x,popmean=0)[1], axis=1)
        pvals_u = get_difference_from_uniform(best_n, rand_size=rand_size, how_many_tests=how_many_tests)
        stars_u = pvals_u.apply(pval_to_star)
    elif stars=='uniform-min':
        pvals = best_n.apply(lambda x: test_normality_from_uniform_min(x, False), axis=1)
    elif stars=='uniform-min-normality':
        pvals = best_n.apply(lambda x: test_normality_from_uniform_min(x, True), axis=1)
    elif 'uniform' in stars:
        which = '-'.join(stars.split('-')[1:])
        pvals = best_n.apply(lambda x: test_normality_from_uniform_any(x, which), axis=1)
    else:
        raise ValueError('stars input must be one of `ttest`, `uniform-ks`, `both`,  `uniform-sw`, `uniform-lilliefors`')
    stars_df = pvals.apply(pval_to_star)

    demand = ['sector_specific_dematerialization_tech_growth', 'sector_specific_price_response',
            'intensity_response_to_gdp', 'direct_melt_elas_scrap_spread']
    mine_production = ['primary_oge_scale', 'mine_cu_margin_elas', 'mine_cost_og_elas',
                    'mine_cost_change_per_year', 'mine_cost_price_elas']
    incentive = ['initial_ore_grade_decline',
                'incentive_opening_probability', 'close_years_back', 'primary_price_resources_contained_elas',
                'reserves_ratio_price_lag', 'mine_cost_tech_improvements', 'incentive_mine_cost_change_per_year']
    price = ['primary_commodity_price_elas_sd', 'scrap_spread_elas_primary_commodity_price', 
            'scrap_spread_elas_sd', 'tcrc_elas_price','tcrc_elas_sd']
    refinery = ['pri CU TCRC elas', 'pri CU price elas', 'refinery_capacity_fraction_increase_mining', 
                'sec CU TCRC elas', 'sec CU price elas', 'sec ratio TCRC elas', 'sec ratio scrap spread elas',]
    scrap_avail = ['collection_elas_scrap_price']
    
    mean_std = pd.DataFrame(np.nan, means.index, means.columns)
    for i in mean_std.index:
        for c in mean_std.columns:
            if means[c][i] != '':
                if stars != 'both' and value_in_parentheses in ['None',None]:
                    mean_std.loc[i, c] = '{:.3f}{:s}'.format(means[c][i], stars_df[c][i])
                elif stars != 'both':
                    mean_std.loc[i, c] = '{:.3f}{:s} ({:.3f})'.format(means[c][i], stars_df[c][i], stds[c][i])
                elif value_in_parentheses in ['None',None]:
                    mean_std.loc[i, c] = r'{:.3f}{:s}/{:s}'.format(means[c][i], stars_df[c][i], stars_u[c][i])
                else:
                    mean_std.loc[i, c] = r'{:.3f}{:s}/{:s} ({:.3f})'.format(means[c][i], stars_df[c][i], stars_u[c][i], stds[c][i])
            else:
                mean_std.loc[i, c] = ' '
    mean_std = mean_std.T
    params_nice = make_parameter_names_nice(mean_std.columns)

    def convert_param_names(v):
        if v in mine_production:
            return ('Mining production', params_nice[v])
        elif v in demand:
            return ('Demand response', params_nice[v])
        elif v in incentive:
            return ('Reserve development', params_nice[v])
        elif v in price:
            return ('Price formation', params_nice[v])
        elif v in refinery:
            return ('Refinery operation', params_nice[v])
        elif v in scrap_avail:
            return ('Secondary supply', params_nice[v])
        else:
            print(v)
            return ('Integration parameters', params_nice[v])

    mean_std = mean_std.rename(columns=dict(zip(mean_std.columns,
                                                [convert_param_names(i) for i in mean_std.columns])))
    pvals = pvals.unstack()
    pvals = pvals.rename(columns=dict(zip(pvals.columns,
                                                [convert_param_names(i) for i in pvals.columns])))
    pvals.columns = pd.MultiIndex.from_tuples(pvals.columns)
#     mean_std = mean_std.rename(dict(zip(mean_std.index, [i
#                                                          for i in mean_std.index])))
    mean_std.columns = pd.MultiIndex.from_tuples(mean_std.columns)
    means = means.T.replace('',np.nan)
    means = means.rename(columns=dict(zip(means.columns,
                                                [convert_param_names(i) for i in means.columns])))
#     means = means.rename(dict(zip(means.index, [many.commodity_element_map[i.capitalize()]
#                                                          for i in means.index])))
    means.columns = pd.MultiIndex.from_tuples(means.columns)
    return mean_std.T.sort_index().T, means.sort_index().T.sort_index().T, pvals.sort_index().T.sort_index().T

def get_correct_loc_scale(parameter):
    loc = 0
    scale = 1
    if parameter=='incentive_opening_probability':
        scale = 0.5
    elif parameter in ['mine_cost_change_per_year','incentive_mine_cost_change_per_year']:
        loc = -5
        scale = 10
    elif parameter=='sector_specific_dematerialization_tech_growth':
        loc = -0.1
        scale = 0.2
    elif parameter=='intensity_response_to_gdp':
        loc = -0.5
        scale = 1.5
    elif parameter in ['sector_specific_price_response','region_specific_price_response']:
        loc = -0.6
        scale = 0.6
    elif parameter in ['sec ratio TCRC elas','mine_cost_og_elas',
                    'primary_commodity_price_elas_sd','initial_ore_grade_decline']:
        loc = -1
    return loc, scale

def test_normality_from_uniform_any(X, which):
    """
    Converts from uniform distribution (X) to normal (Y) and exponential (Z) based on probability integral transform.
    Conversion from uniform to normal: https://stackoverflow.com/questions/20626994/how-to-calculate-the-inverse-of-the-normal-cumulative-distribution-function-in-p
    Conversion from uniform to exponential: https://en.wikipedia.org/wiki/Probability_integral_transform 
    """
    loc, scale = get_correct_loc_scale(X.name[1])
    X = (X.values-loc)/scale
    Y = ndtri(X)
    X[X==0]=1e-4
    X[X==1]=1-1e-4
    Z = -np.log(1-X)
    if which=='shapiro':
        s = stats.shapiro(Y)[1]
    elif which=='lilliefors-norm':
        s = lilliefors(Y)[1]
    elif which=='lilliefors-exp':
        s = lilliefors(Z, dist='exp')[1]
    elif which=='lilliefors-both':
        s = np.min([lilliefors(Z, dist='exp')[1], lilliefors(Y, dist='norm')[1]])
    elif which=='dagostino':
        s = stats.normaltest(Y)[1]
    elif which=='anderson-norm':
        s = anderson(Y, dist='norm')
    elif which=='anderson-exp':
        s = anderson(Z, dist='exp')
    elif which=='anderson-both':
        s = np.min([anderson(Z, dist='exp'), anderson(Y, dist='norm')])
    elif which=='ttest':
        s = stats.ttest_1samp(Y, 0)[1]
    else:
        raise ValueError('incorrect normality test given: '+which+'''. 
            Possible options: shapiro, lilliefors-norm, lilliefors-exp, lilliefors-both, dagostino, 
            anderson-norm, anderson-exp, anderson-both''')
    return s

def test_normality_from_uniform_min(X, normality_only=False):
    loc, scale = get_correct_loc_scale(X.name[1])
    X = (X.values-loc)/scale
    Y = ndtri(X)
    X[X==0]=1e-4
    X[X==1]=1-1e-4
    Z = -np.log(1-X)
    l1 = lilliefors(Y, dist='norm')[1]
    l2 = lilliefors(Z, dist='exp')[1]
    a1 = anderson(Y, dist='norm')
    a2 = anderson(Z, dist='exp')
    s = stats.shapiro(Y)[1]
    d = stats.normaltest(Y)[1]
    t = stats.ttest_1samp(Y, 0)[1]
    if normality_only:
        return min([s,l1,l2,a1,a2,d])
    return min([s,l1,l2,a1,a2,d,t])

def anderson(Y, dist='norm'):
    """
    Returns the lowest p-value achieved within the scipy.stats.anderson function call
    """
    dist = 'expon' if dist=='exp' else dist
    s = stats.anderson(Y, dist=dist)
    crit = s.critical_values
    sig = s.significance_level
    stat = s.statistic
    check = stat>crit
    if np.any(check):
        loc = np.argmin(stat>crit)-1
        a = sig[loc]/100
    else:
        a = 1
    return a

def get_difference_from_uniform(rmse_df, rand_size=25, how_many_tests=100):
    for_test = rmse_df.copy()
    
    uniform_rvs = pd.DataFrame()
    loc = 0
    scale = 1
    for rs in np.arange(0,how_many_tests):
        uniform_rvs[rs] = stats.uniform.rvs(loc=loc,scale=scale,size=25, random_state=rs)

    def ks100_uniform(array, size=25, how_many_tests=100):
        """
        Size is the size of the random variable generation from the uniform distribution
        """
        ks_pvals = []
        name = array.name[1]
        loc, scale = get_correct_loc_scale(name)
        if size is not None:
            for rs in np.arange(0,how_many_tests):
                if True:
                    randoms = stats.uniform.rvs(loc=loc, scale=scale, size=size, random_state=rs)
                    val = stats.kstest(array, randoms)[1]
                else:
                    val = stats.kstest(array, uniform_rvs[rs])[1]
                ks_pvals += [val]
#             return np.median(ks_pvals)
            return pi0est(np.array(ks_pvals))['pi0']
#             if array.name[0]=='Au':
#                 if array.name[1]=='sector_specific_price_response':
#                     print(np.median(ks_pvals), np.max(pval_to_qval(ks_pvals)))
#             return np.median(pval_to_qval(ks_pvals))
        else: 
            randoms = stats.uniform.rvs(loc=loc, scale=scale, size=10000, random_state=0)
            val = stats.kstest(array, randoms)[1]
#             val = stats.kstest(array, 'uniform')[1]
            return val
    
    applied = for_test.apply(lambda x: ks100_uniform(x, size=rand_size, how_many_tests=how_many_tests), axis=1)
    return applied

def plot_colorful_table(many, stars='ttest', value_in_parentheses='standard error', 
                        no_color_for_insignificant=True,
                        rand_size=1000, dpi=50, show=False):
    """
    Creates the table showing parameter values, stars for statistical significance, and
    standard error/other metric in parentheses. 
    
    many: Many object
    stars: str, how to calculate pvalues for the stars. Can be `ttest`, `uniform`, or `both`. 
        The ttest version uses a simple one-sample t-test for difference from zero. The 
        uniform version checks for whether the distribution is different from a uniform
        distribution with the corresponding bounds, performing 100 Kolmogorov-Smirnov tests
        and taking the mean. Uniform takes forever.
    value_in_parentheses: str, can be `standard error`, `standard deviation`, or `variance`
    dpi: int, controls resolution
    """
    table,means,pvals = make_parameter_mean_std_table(many,25, stars=stars, 
                                                value_in_parentheses=value_in_parentheses, rand_size=rand_size)
    alt_table = table.replace(dict(zip(table.values.flatten(),
                                    [i.replace(' (','\n(') for i in table.values.flatten()])))\
        .sort_index().T.sort_index().droplevel(0)
    alt_means = means.div(abs(means).max()).T.droplevel(0)
    alt_pvals = pvals.T.droplevel(0)
    if no_color_for_insignificant:
        alt_means[alt_pvals>0.1] = 0
    fig,ax = plt.subplots()
    sns.heatmap(alt_means,
                ax=ax,
                annot=alt_table,
                fmt='s',
                annot_kws={'fontsize':16},
                xticklabels=True,
                yticklabels=True,
                cmap='vlag',
                cbar=False,
            )
    # ax.set_yticks(np.arange(ax.get_yticks()[0],ax.get_yticks()[-1]+1,1))
    ax.tick_params(labelbottom=True, labeltop=True, labelsize=24)
    if stars != 'both':
        fig.set_size_inches(12,16)
    else:
        fig.set_size_inches(16,16)
    fig.set_dpi(dpi)
    if show:
        plt.show()
    plt.close()
    return table, means, pvals, alt_means, fig

def plot_colorful_table2(many, stars='ttest', value_in_parentheses='standard error', 
                            no_color_for_insignificant=True, how_many_tests=100,
                            rand_size=1000, dpi=50, show=False):
        """
        Creates the table showing parameter values, stars for statistical significance, and
        standard error/other metric in parentheses. 
        
        many: Many object
        stars: str, how to calculate pvalues for the stars. Can be `ttest`, `uniform`, or `both`. 
            The ttest version uses a simple one-sample t-test for difference from zero. The 
            uniform version checks for whether the distribution is different from a uniform
            distribution with the corresponding bounds, performing 100 Kolmogorov-Smirnov tests
            and taking the mean. Uniform takes forever.
        value_in_parentheses: str, can be `standard error`, `standard deviation`, or `variance`
        dpi: int, controls resolution
        """
        table,means,pvals = make_parameter_mean_std_table(many,25, stars=stars, 
                                                    value_in_parentheses=value_in_parentheses, rand_size=rand_size,
                                                    how_many_tests=how_many_tests)
        alt_table = table.replace(dict(zip(table.values.flatten(),
                                        [i.replace(' (','\n(') for i in table.values.flatten()])))\
            .sort_index().T.sort_index()
        alt_means = means.div(abs(means).max()).T
        alt_pvals = pvals.T
        if no_color_for_insignificant:
            alt_means[alt_pvals>0.1] = 0
        
        plots = alt_means.index.get_level_values(0).unique()
        height_ratios = [alt_means.loc[i].shape[0] for i in plots]
        fig,axes = easy_subplots(plots,1, height_ratios=height_ratios, use_subplots=True, sharex=True)
        for i,ax in zip(plots, axes):
            sub_means = alt_means.loc[i].rename(columns={'Steel':'Fe'})
            sub_table = alt_table.loc[i].rename(columns={'Steel':'Fe'})
            sns.heatmap(sub_means,
                        ax=ax,
                        annot=sub_table,
                        fmt='s',
                        annot_kws={'fontsize':18},
                        xticklabels=True,
                        yticklabels=True,
                        cmap='vlag',
                        cbar=False,
                        vmin=-1,
                        vmax=1,
                    )
            # ax.set_yticks(np.arange(ax.get_yticks()[0],ax.get_yticks()[-1]+1,1))
            ax.tick_params(labelbottom= i==plots[-1], labeltop= i==plots[0], labelsize=20, color='white')
            ax.tick_params(axis='y',rotation=0)
            ax.set_ylabel(i,rotation='horizontal',horizontalalignment='left', labelpad=170, fontsize=22)
        if stars != 'both':
            fig.set_size_inches(19,14)
        else:
            fig.set_size_inches(19,16)
        fig.set_dpi(dpi)
        fig.align_ylabels(axes)
        fig.tight_layout()
        if show:
            plt.show()
        plt.close()
        return table, means, pvals, alt_means, fig

def plot_given_columns_for_paper(many, commodity, columns, column_name=None, 
                                ax=None, column_subset=None, start_year=None, end_year=2019, 
                                show_all_lines=False, r2_on_own_line=True, plot_actual_price=False, 
                                apply_difference_for_regression=False, use_r2_instead_of_mape=False,
                                dpi=50):
    """
    Plots historical vs simulated demand, mining,
    and primary commodity price for a given commodity
    and whichever hyperparameter sets given as the
    `columns` variable. Can get the n_best_scenarios
    using the get_best_columns function, or more manually
    by running get_train_test_scores on an rmse_df (or
    commodity subset), then running its output through
    get_commodity_scores, e.g.

    rr = get_train_test_scores(many_test.integ.rmse_df.loc['silver'])
    s1 = get_commodity_scores(rr,None)
    then passing s1.loc[s1.selection2].index as the columns input.

    -------------
    many: Many() object, needs to have results object of its own,
        so if you ran get_multiple to load many, you should pass
        many.integ to this function
    commodity: str, commodity name in lowercase form
    columns: list, int, or None. If list, columns from rmse_df to 
        plot. If int, will selected the int lowest-score (RMSE)
        columns. If None, will select the 25 lowest-score columns.
        Will highlight the lowest-score one if no column_subset is 
        passed.
    column_name: str, gets included in the plot title if not None
    ax: matplotlib axes object, can leave out and this will
        create its own plot for you.
    column_subset: list, allows you to select a subset of the
        passed columns to highlight, or to just plot two groups
        of parameter sets simultaneously, since column_subset
        and columns do not have to intersect. Pass a list or
        array of numbers corresponding to rmse_df columns.
    plot_actual_price: bool, whether to include the unadjusted 
        historical price in the plot
    apply_difference_for_regression: bool, whether to apply
        difference for regressions where we calculate R2 to
        deal with non-stationarity problem pointed out by
        reviewer 1
    dpi: dots per inch, controls resolution. Only functions if
        the ax input is None.
    """
    if ax is None:
        fig,ax=easy_subplots(3, dpi=dpi)
    else:
        fig = 0
    if columns is None:
        columns = many.rmse_df.loc[commodity].sort_values(by='score',axis=1).iloc[:,:25].columns
    elif type(columns)==int:
        columns = many.rmse_df.loc[commodity].sort_values(by='score',axis=1).iloc[:,:columns].columns
    objective_results_map = {'Total demand':'Total demand','Primary commodity price':'Refined price',
                                'Primary demand':'Conc. demand','Primary supply':'Mine production',
                                'Conc. SD':'Conc. SD','Scrap SD':'Scrap SD','Ref. SD':'Ref. SD'}
    hold_results = pd.DataFrame()
    comm_not_element = many.element_commodity_map[commodity]
    for i,a in zip(['Total demand','Primary commodity price','Primary supply'], ax):
        results = many.results.copy()[objective_results_map[i]].sort_index()\
            .loc[idx[commodity,:,2001:end_year]].droplevel(0).unstack(0)
        if 'SD' not in i:
            historical_data = many.historical_data.copy()[i].loc[commodity].loc[:2019]
            if i=='Primary commodity price':
                original_price = many.historical_data.copy()[
                    'Original primary commodity price'].loc[commodity].loc[:2019]
            if start_year is not None:
                historical_data = historical_data.loc[start_year:]
                if i=='Primary commodity price':
                    original_price = original_price.loc[start_year:]
        else:
            historical_data = pd.Series(results.min(),[0])
        results_ph = results.copy()
        results = results[columns]

        diction = get_unit(results, historical_data, i)
        results, historical_data, unit = [diction[i] for i in ['simulated','historical','unit']]
        results_ph *= results[columns[0]].mean()/results_ph[columns[0]].mean()
        if show_all_lines:
            sim_line = a.plot(results,linewidth=1,color='gray',alpha=0.3,label=results.columns)
            if column_subset is None:
                best_line= a.plot(results[columns[0]],linewidth=6,label='Simulated',color='tab:blue')
            else:
                best_line= a.plot(results_ph[column_subset],linewidth=1,label='Simulated',color='tab:blue')
        else:
            sns_results = results.stack().reset_index().rename(columns={
                'level_0':'Year','level_1':'Scenario',0:'Value'
            })
            sim_line = sns.lineplot(data=sns_results, x='Year', y='Value', ax=a)
            sim_line = sim_line.get_lines()
        mins = min(historical_data.min(),results[columns[0]].min())*0.95
        maxs = max(historical_data.max(),results[columns[0]].max())*1.1
        extra_label=''
        if i=='Primary commodity price' and plot_actual_price:
            if original_price.isna().all():
                comm_not_element = many.element_commodity_map[commodity]
                original_price = pd.read_csv(
                    'input_files/user_defined/price adjustment results.csv',
                    index_col=0)[f'log({comm_not_element})'].sort_index().loc[original_price.index]
            orig_hist_line = a.plot(original_price, label='Historical', color='lightgray', linewidth=6,
                                    linestyle=':')
            extra_label=', rolling mean'
        hist_line = a.plot(historical_data,label='Historical'+extra_label,color='k',linewidth=6)
        inter = np.intersect1d(results.index,historical_data.index)
        endog = historical_data.loc[inter]
        if show_all_lines:
            exog = results[columns[0]].loc[inter]
        else:
            exog = sim_line[0].get_data()[1]
        m = sm.GLS(endog,sm.add_constant(exog)).fit(cov_type='HC3')
        mse = round(m.mse_resid**0.5,2)
        mse = round(m.rsquared,2)
        cointegration_test_pval = round(cointegration_test(exog, endog)[1],3)
        mape = round(100*np.mean(abs((exog-endog.values)/endog.values)),1)
        r2_or_mape_str = r'$R^2$' if use_r2_instead_of_mape else 'MAPE'
        r2_or_mape = mse if use_r2_instead_of_mape else str(mape)+'%'
        if apply_difference_for_regression:
            endog = np.diff(endog, axis=0)
            exog = np.diff(exog, axis=0)
            m = sm.GLS(endog,sm.add_constant(exog)).fit(cov_type='HC3')
            diff_mse = round(m.mse_resid**0.5,2)
            diff_mse = round(m.rsquared,2)
        alt_comm = commodity.replace('Steel','Fe')
        if column_name is not None:
            title=f'{i}, {column_name} {alt_comm},\n'+f'{r2_or_mape_str}={r2_or_mape}, scenario {columns[0]}'
        else:
            title=f'{i}, {alt_comm}'
        if i=='Primary commodity price':
            maxs *= 1.1
            if show_all_lines:
                maxs *= 1.1
            if plot_actual_price:
                maxs *= 1.15
                if commodity=='Sn':
                    maxs *= 1
                    mins *= 0.85
                elif commodity=='Cu':
                    mins *= 0.9
                    maxs*=1.15
#                 if commodity=='Ag':
#                     maxs *= 1.1
            title = title.replace('Primary commodity','Refined metal')
        elif i=='Primary supply' and show_all_lines:
            maxs *= 1.1
#         elif i=='Total demand' and commodity=='Ag':
#             maxs *= 0.6
        elif i=='Primary supply' and commodity=='Sn':
            mins *= 0.9
            maxs *= 1.2
        elif i=='Primary supply' and commodity=='Cu':
            mins *=0.9
        title = title.replace('Primary supply','Mine production')
        title = title.replace('Steel','Fe')
        a.set(title=title,
            ylabel=i+' ('+unit+')',xlabel='Year',ylim=(mins,maxs))
#         a.text(0.05,0.9, r'$R^2$'+f'={mse}', transform=a.transAxes)
        
        if len(sim_line)<10 and show_all_lines:
            a.legend()
        else:
            handles = []
            labels = []
            simulated_string = 'Simulated, best'
            r2_string = f'{r2_or_mape_str}={r2_or_mape}'
            if apply_difference_for_regression:
                diff_r2_string = r'Diff. $R^2$'+f'={diff_mse}, avg%err: {mape}'
            r2_handle = Line2D([0],[0], color='w', linewidth=0)
            if plot_actual_price and i=='Primary commodity price':
                handles += [orig_hist_line[0]]
                labels += ['Historical']
            handles += [hist_line[0]]
            labels += ['Historical'+extra_label]
            if len(sim_line)!=1:
                handles += [best_line[0], sim_line[0]]
                if r2_on_own_line:
                    labels += [simulated_string, 'Simulated, other', r2_string]
                    handles += [r2_handle]
                    if apply_difference_for_regression:
                        labels += [diff_r2_string]
                        handles += [r2_handle]
                else:
                    labels += [' '.join([simulated_string, r2_string]), 'Simulated, other']
                    a.set(title=title+', '+r2_string)
            else:
                handles += [sim_line[0]]
                if r2_on_own_line:
                    labels += [simulated_string, r2_string]
                    handles += [r2_handle]
                    if apply_difference_for_regression:
                        labels += [diff_r2_string]
                        handles += [r2_handle]
                else:
                    labels += [' '.join([simulated_string, r2_string])]
                    a.set(title=title+', '+r2_string)
            a.legend(handles, labels, loc='upper left')
        res = results.copy()
        res = pd.concat([res],keys=[commodity])
        res = pd.concat([res],keys=[i])
        hist = pd.DataFrame(historical_data.copy()).rename(columns={i:'Historical'})
        if i=='Primary commodity price' and plot_actual_price:
            op = pd.DataFrame(original_price.copy()).rename(columns={f'log({comm_not_element})':'Historical, actual'}) # actual original, hist is then the rolling mean
            hist = hist.rename(columns={'Historical':'Historical, rolling mean'})
            hist = pd.concat([op,hist],axis=1)
        hist = pd.concat([hist],keys=[commodity])
        hist = pd.concat([hist],keys=[i])
        
        both = pd.concat([hist,res],axis=1)
        hold_results = pd.concat([hold_results, both])
    hold_results = hold_results.stack()
    return fig,ax,hold_results

def plot_given_columns_for_paper_many(many, commodity, columns, column_name=None, 
                                 ax=None, column_subset=None, start_year=None, end_year=2019, 
                                 show_all_lines=False, r2_on_own_line=True, use_r2_instead_of_mape=False, 
                                 dpi=50):
    """
    Plots historical vs simulated demand, mining,
    and primary commodity price for a given commodity
    and whichever hyperparameter sets given as the
    `columns` variable. Can get the n_best_scenarios
    using the get_best_columns function, or more manually
    by running get_train_test_scores on an rmse_df (or
    commodity subset), then running its output through
    get_commodity_scores, e.g.

    rr = get_train_test_scores(many_test.integ.rmse_df.loc['silver'])
    s1 = get_commodity_scores(rr,None)
    then passing s1.loc[s1.selection2].index as the columns input.

    -------------
    many: Many() object, needs to have results object of its own,
        so if you ran get_multiple to load many, you should pass
        many.integ to this function
    commodity: str, commodity name in lowercase form
    columns: list, int, or None. If list, columns from rmse_df to 
        plot. If int, will selected the int lowest-score (RMSE)
        columns. If None, will select the 25 lowest-score columns.
        Will highlight the lowest-score one if no column_subset is 
        passed.
    column_name: str, gets included in the plot title if not None
    ax: matplotlib axes object, can leave out and this will
        create its own plot for you.
    column_subset: list, allows you to select a subset of the
        passed columns to highlight, or to just plot two groups
        of parameter sets simultaneously, since column_subset
        and columns do not have to intersect. Pass a list or
        array of numbers corresponding to rmse_df columns.
    dpi: dots per inch, controls resolution. Only functions if
        the ax input is None.
    """
    if ax is None:
        fig,ax=easy_subplots(3, dpi=dpi)
    else:
        fig = 0
    if columns is None:
        columns = many.rmse_df.loc[commodity].sort_values(by='score',axis=1).iloc[:,:25].columns
    elif type(columns)==int:
        columns = many.rmse_df.loc[commodity].sort_values(by='score',axis=1).iloc[:,:columns].columns
    objective_results_map = {'Total demand':'Total demand','Primary commodity price':'Refined price',
                                 'Primary demand':'Conc. demand','Primary supply':'Mine production',
                                'Conc. SD':'Conc. SD','Scrap SD':'Scrap SD','Ref. SD':'Ref. SD'}
    data = pd.DataFrame()
    for i,a in zip(['Total demand','Primary commodity price','Primary supply'], ax):
        results = many.results_sorted.copy()[objective_results_map[i]].sort_index()\
            .loc[idx[commodity,:,2001:end_year]].droplevel(0).unstack(0)
        if 'SD' not in i:
            historical_data = many.historical_data.copy()[i].loc[commodity].loc[:2019]
            if start_year is not None:
                historical_data = historical_data.loc[start_year:]
        else:
            historical_data = pd.Series(results.min(),[0])
        results_ph = results.copy()
#         results = results[columns]
        results = results.loc[:,:24]

        # print(commodity, i, results.stack().median(), results.stack().mean())
        # if 10*results.stack().median() < results.stack().mean():
        #     print(commodity, i)
        results[results>50*results.stack().median()] = np.nan
        diction = get_unit(results, historical_data, i)
        results, historical_data, unit = [diction[i] for i in ['simulated','historical','unit']]
#         results_ph *= results[columns[0]].mean()/results_ph[columns[0]].mean()
        if show_all_lines:
            sim_line = a.plot(results,linewidth=1,color='gray',alpha=0.3,label=results.columns)
            if column_subset is None:
                best_line= a.plot(results[columns[0]],linewidth=6,label='Simulated',color='tab:blue')
            else:
                best_line= a.plot(results_ph[column_subset],linewidth=1,label='Simulated',color='tab:blue')
        else:
            sns_results = results.stack().reset_index().rename(columns={
                'level_0':'Year','level_1':'Scenario',0:'Value'
            })
            sim_line = sns.lineplot(data=sns_results, x='Year', y='Value', ax=a)
            sim_line = sim_line.get_lines()
            if len(sim_line)>1:
                sim_line[-1].set_linestyle(['-','--',':','-.'][int((len(sim_line)-1)/2)])
        mins = min(historical_data.min(),results[0].min())*0.95
        maxs = max(historical_data.max(),results[0].max())*1.1
        hist = historical_data.copy()
        hist = pd.DataFrame(hist).reset_index().rename(columns={'index':'Year',i:'Value'})
        hist['Scenario'] = 'Historical'
        out = pd.concat([sns_results, hist])
        out['Historical or simulated'] = 'Simulated'
        out.loc[out['Scenario']=='Historical','Historical or simulated'] = 'Historical'
        out['Variable'] = i
        data = pd.concat([data, out])
        hist_line = a.plot(historical_data,label='Historical',color='k',linewidth=6)
        inter = np.intersect1d(results.index,historical_data.index)
        if show_all_lines:
            m = sm.GLS(historical_data.loc[inter], 
                       sm.add_constant(results[columns[0]].loc[inter])
                      ).fit(cov_type='HC3')
        else:
            m = sm.GLS(historical_data.loc[inter], 
                       sm.add_constant(sim_line[-1].get_data()[1])
                      ).fit(cov_type='HC3')
        mse = round(m.mse_resid**0.5,2)
        mse = round(m.rsquared,2)
        mape = 100*np.mean(abs((historical_data.loc[inter]-sim_line[-1].get_data()[1])/historical_data.loc[inter]))
        mape = round(mape,1)
        r2_or_mape_str = r'$R^2$' if use_r2_instead_of_mape else 'MAPE'
        r2_or_mape = mse if use_r2_instead_of_mape else str(mape)+'%'
        if column_name is not None:
            title=f'{i}, {column_name} {commodity},\n'+r2_or_mape_str+f'={r2_or_mape}, scenario {columns[0]}'
        else:
            title=f'{i}, {commodity}'
        if i=='Primary commodity price':
            maxs *= 1.1
            title = title.replace('Primary commodity','Refined metal')
        elif i=='Primary supply' and show_all_lines:
            maxs *= 1.1
        title = title.replace('Primary supply','Mine production')
        a.set(title=title.replace('Steel','Fe'),
              ylabel=i+' ('+unit+')',xlabel='Year',ylim=(mins,maxs))
#         a.text(0.05,0.9, r'$R^2$'+f'={mse}', transform=a.transAxes)

        if len(sim_line)<10 and show_all_lines:
            a.legend()
        elif len(sim_line)==1:
            r2_handle = Line2D([0],[0], color='w', linewidth=0)
            if r2_on_own_line:
                a.legend([hist_line[0], sim_line[0], r2_handle],
                         ['Historical','Simulated',f'{r2_or_mape_str}={r2_or_mape}'])
            else:
                a.legend([hist_line[0], sim_line[0]],
                         ['Historical','Simulated'+f' {r2_or_mape_str}={r2_or_mape}'], loc='upper left')
                sim_line[-1].set_label('Simulated:'+f' {r2_or_mape_str}={r2_or_mape}')

        elif 'best_line' not in locals():
            lines = [i for i in sim_line if i.get_color()!='k']
            labels = [i.get_label().replace('Simulated',j) for i,j in zip(lines,many.many_keys)][:-1]
            a.legend([hist_line[0]]+lines,
                     ['Historical']+labels+[many.plot_label+f': {r2_or_mape_str}={r2_or_mape}'], loc='upper left')
            sim_line[-1].set_label(many.plot_label+f': {r2_or_mape_str}={r2_or_mape}')
            a.set_ylim(a.get_ylim()[0], a.get_ylim()[1]*(1+len(sim_line)/4*0.2))
        else:
            r2_handle = Line2D([0],[0], color='w', linewidth=0)
            if r2_on_own_line:
                a.legend([hist_line[0], best_line[0], sim_line[0], r2_handle],
                         ['Historical','Simulated, best','Simulated, other',f'{r2_or_mape_str}={r2_or_mape}'])
            else:
                a.legend([hist_line[0], best_line[0], sim_line[0]],
                         ['Historical','Simulated, best:'+f' {r2_or_mape_str}={r2_or_mape}','Simulated, other'])
    return fig,ax,data

def plot_best_fits_many(many, show_all_lines=False, commodities='all', dpi=50, show=False):
    """
    many: Many object or dict full of labeled Many objects
    """
    if show_all_lines and type(many)==dict:
        show_all_lines=False
        warnings.warn('dict type input for the many variable is not accepted with show_all_lines=True; show_all_lines has been set to False')
    if type(many)==Many:
        many_dict = {'':many}
    else:
        many_dict = many
        many = many[list(many.keys())[0]]
    
    if type(commodities)==str:
        if commodities=='subset2':
            commodities=['Ag','Sn','Al','Steel']
        elif commodities=='subset1':
            commodities=['Cu','Ni','Pb','Zn','Au']
        elif commodities=='all':
            commodities=['Cu','Ni','Pb','Zn','Au','Ag','Sn','Al','Steel']
        else:
            raise ValueError('invalid commodity string given, should be either list/array of commodity (element) names, or among the strings: all, subset1, subset2')
    fig, ax = easy_subplots(len(commodities)*3)
    data = pd.DataFrame()
    for e,comm in enumerate(commodities):
        outer = pd.DataFrame()
        for many in many_dict:
            a = ax[3*e:3*(e+1)]
            many_dict[many].plot_label = many
            many_dict[many].many_keys = list(many_dict.keys())
            fig1,a,out=plot_given_columns_for_paper_many(many_dict[many],
                                                comm,
                                                columns=None, 
                                                show_all_lines=show_all_lines, 
                                                r2_on_own_line=False,
                                                ax=a
                                               )
            out['Tuning years'] = many
            outer = pd.concat([outer, out])
        outer['Commodity'] = comm
        data = pd.concat([data, outer])
    fig.tight_layout()
    add_axis_labels(fig, 'price_middle_column')
    fig.set_dpi(dpi)
    if show:
        plt.show()
    plt.close()
    return fig, data

def plot_best_fits(many, show_all_lines=False, plot_actual_price=False, commodities=None, dpi=300, 
                start_year=None, end_year=2019, show=False):
    if type(commodities)==str:
        if commodities=='subset2':
            commodities=['Ag','Sn','Al','Steel']
        elif commodities=='subset1':
            commodities=['Cu','Ni','Pb','Zn','Au']
        elif commodities=='all':
            commodities=['Cu','Ni','Pb','Zn','Au','Ag','Sn','Al','Steel']
        else:
            raise ValueError('invalid commodity string given, should be either list/array of commodity (element) names, or among the strings: all, subset1, subset2')
    
    fig, ax = easy_subplots(len(commodities)*3,height_scale=1.1)
    chars = character_list[10:36]
    all_data = pd.DataFrame()
    for e,comm in enumerate(commodities):
        a = ax[3*e:3*(e+1)]
        best = many.rmse_df.loc[comm].sort_values(by='score',axis=1).iloc[:,:25].columns
        fig1,a,data=plot_given_columns_for_paper(many,comm,columns=best, show_all_lines=show_all_lines, 
                                            plot_actual_price=plot_actual_price, 
                                            start_year=start_year, end_year=end_year,
                                            r2_on_own_line=True, ax=a)
        all_data = pd.concat([all_data, data])
#         a[1].set_ylim(10500,29000)
    
    add_axis_labels(fig, 'price_middle_column')
    fig.set_dpi(dpi)
    all_data.index = pd.MultiIndex.from_tuples(all_data.index)
    all_data = all_data.reset_index()
    all_data = all_data.rename(columns={
        0:'Value', 
        'level_0':'Variable',
        'level_1':'Commodity',
        'level_2':'Year',
        'level_3':'Scenario'
    })
    all_data['Historical or simulated'] = 'Simulated'
    all_data.loc[all_data['Scenario']=='Historical', 'Historical or simulated'] = 'Historical'
    all_data.loc[all_data['Scenario']=='Historical, actual', 'Historical or simulated'] = 'Historical, actual'
    all_data.loc[all_data['Scenario']=='Historical, rolling mean', 'Historical or simulated'] = 'Historical, rolling mean'
    all_data = all_data[['Variable','Commodity','Year','Historical or simulated','Scenario','Value']]
    all_data = all_data.reset_index(drop=True)
    fig.tight_layout()
    if show:
        plt.show()
    plt.close()
    return fig, all_data

def get_r2(many_sg, many_15, many_16, many_17, use_r2_instead_of_mape=False):
    years = np.arange(2001,2020)
    r2_values = pd.DataFrame()
    commodities = many_sg.rmse_df.index.get_level_values(0).unique()
    historical_names = {'Total demand':'Total demand', 'Mine production':'Primary supply', 
                        'Refined price':'Primary commodity price'}
    rsquared_df = pd.DataFrame()
    for param in ['Total demand', 'Mine production', 'Refined price']:
        rsquared_df_ph = pd.DataFrame()
        for comm in commodities:
            regr_y = many_sg.historical_data[historical_names[param]].loc[comm].loc[years]
            for many,label in zip([many_sg, many_15, many_16, many_17],['Full','To 2014','To 2015','To 2016']):
                regr_x = many.results_sorted[param].loc[comm].loc[idx[:24,years]].groupby(level=1).mean()
                if use_r2_instead_of_mape:
                    m = do_a_regress(regr_x, regr_y,plot=False)[1]
                    rsquared_df_ph.loc[comm, label] = m.rsquared
                else:
                    rsquared_df_ph.loc[comm, label] = np.mean(abs((regr_x-regr_y)/regr_y))*100
        rsquared_df_ph = pd.concat([rsquared_df_ph],keys=[param])
        rsquared_df = pd.concat([rsquared_df, rsquared_df_ph])
    return rsquared_df

def get_r2_from_plot(many_sg, many_15, many_16, many_17, use_r2_instead_of_mape=False):
    """
    Getting R2 differences from plot above
    """
    if not hasattr(many_sg, 'fig_fits'):
        figures_s28_and_s29(many_sg, many_17, many_16, many_15, show=False, all_only=True)
    fig_fits = many_sg.fig_fits
    r2_df = pd.DataFrame()
    for a in fig_fits.get_axes():
        title = a.get_title()
        lines = a.get_lines()
        labels = [i.get_label() for i in lines]
        if use_r2_instead_of_mape:
            labels = [i.replace('Simulated','Full') for i in labels if i!='Historical']
            keys = [i.split(':')[0] for i in labels]
            r2 = [float(i.split('=')[1]) for i in labels]
            r2_series = pd.Series(r2,keys)
            r2_series.name = title.split(', ')[1]
            r2_series = pd.concat([r2_series], keys=[title.split(',')[0]])
        else:
            labels = [i.replace('Simulated','Full') for i in labels]
            line_data = {l.split(':')[0]:pd.Series(d.get_xydata()[:,1], d.get_xydata()[:,0]) for l,d in zip(labels, lines)}
            line_data = {l: line_data[l][line_data[l]>=0] for l in line_data}
            r2_dict = {i: np.mean(abs((line_data[i]-line_data['Historical'])/line_data['Historical'])) for i in line_data if i!='Historical'}
            r2_series = pd.concat([pd.Series(r2_dict)], keys=[title.split(',')[0]])
            r2_series.name = title.split(', ')[1]
        r2_df = pd.concat([r2_df, r2_series],axis=1)
    r2 = r2_df.stack().unstack(level=1)
    return r2

def run_r2_parameter_change_regressions(many_sg, many_17, many_16, many_15, use_r2_instead_of_mape=False):
    # cutoff selection:
    cutoff_dict = {}
    if not hasattr(many_sg, 'means_coef_of_var'):
        figure_s30(many_sg, many_17, many_16, many_15, show=False)
    means_coef_of_var = many_sg.means_coef_of_var.copy()
    means_all = many_sg.means_all.copy()
    if not hasattr(many_sg, 'largest_diff'):
        figure_s27(many_sg, many_15, many_16, many_17, show=False)
    largest_diff = many_sg.largest_diff.copy()
    for cutoff in [0.03, 0.05, 0.08, 0.1, 0.12, 0.13, 0.14, 0.16, 0.18, 0.2, 0.3, 0.4, 0.5]:
        include_vals = (means_coef_of_var>cutoff).any(axis=1)
        include_vals = include_vals[include_vals].index
        include_vals = include_vals[~include_vals.isin(['Incentive mine cost change per year'])]
        cutoff_dict[cutoff] = len(include_vals)
    print('number of parameters included for different cutoff values:', cutoff_dict)

    # Using cutoff 0.12:
    cutoff = 0.12
    tune_to_list =['Mine production','Refined price','Total demand']
    results_all = pd.DataFrame()
    for tune_to in tune_to_list:
        means_diff = means_all.stack().unstack(1)
        means_diff = means_diff.subtract(means_diff['Full'],axis=0).drop(columns='Full')
        means_diff = means_diff.rename(columns={'Split, 2014':'To 2014', 'Split, 2016':'To 2016',
                                            'Split, 2015':'To 2015'})
        means_diff = abs(means_diff.stack().unstack(0).fillna(0))
        r2_d = largest_diff.copy().loc[tune_to]
    #     r2_d = r2_d.rename(dict([(i,many_sg.commodity_element_map[i.capitalize()]) for i in r2_d.index.get_level_values(0).unique()]))
        r2_d = abs(r2_d.sort_index())
        means_diff = means_diff.sort_index()
        if False:
            include_vals = (means_diff.groupby(level=0).std()>cutoff).any()
            include_vals = include_vals[include_vals].index
            include_vals = include_vals[~include_vals.isin(['Incentive mine cost change per year'])]
        elif True:
            include_vals = (means_coef_of_var>cutoff).any(axis=1)
            include_vals = include_vals[include_vals].index
            include_vals = include_vals[~include_vals.isin(['Incentive mine cost change per year'])]
        else:
            include_vals = ['Intensity elasticity to time','Intensity elasticity to GDP','Mine cost change per year',
                            'Ore grade elasticity to COT distribution mean','Mine CU elasticity to TCM',
                            'Secondary refinery CU elasticity to price','Fraction of viable mines that open']
        combos = []
        for i in np.arange(1,len(include_vals)):
            combos += list(combinations(include_vals,i))

        results = pd.DataFrame()
        for e,combo in enumerate(combos):
            x = means_diff.loc[:,combo]
            if use_r2_instead_of_mape:
                r2_d.name = tune_to+' R2 difference from full tuning'
            else:
                r2_d.name = tune_to+' MAPE difference from full tuning'
            m = sm.GLS(r2_d, sm.add_constant(x)).fit(cov_type='HC3')
            results.loc[e,'AIC'] = m.aic
            results.loc[e,'BIC'] = m.bic
            results.loc[e,'f_pvalue'] = m.f_pvalue
            results.loc[e,'rsquared'] = m.rsquared
            results.loc[e,'n_sig'] = (m.pvalues<0.1).sum()
            results.loc[e,'frac_sig'] = results['n_sig'][e]/len(m.pvalues)
            results.loc[e,'n_positive'] = ((m.params.drop('const')>0)&(m.pvalues.drop('const')<0.1)).sum()
            results.loc[e,'frac_positive'] = results['n_positive'][e]/(m.pvalues.drop('const')<0.1).sum() if (m.pvalues.drop('const')<0.1).sum()!=0 else 0
            results.loc[e,'m'] = m
    #     display(results.loc[results.f_pvalue<0.1].sort_values(by='AIC').m.iloc[0].summary())
        results = pd.concat([results],keys=[tune_to])
        results_all = pd.concat([results_all, results])
        many_sg.results_all = results_all.copy()
    return results_all

def quick_hist(d_f,log=True,norm=False,normer=0,plot=True, show=False, print_progress=True, 
               bins=40, show_sim=True, dist='norm', xlabel='',
               height_scale=1,width_scale=1,rounded=False):
    '''takes dataframe with single columns, plots histograms for each commodity where 
    the dataframe index has two levels: Property ID and Primary Commodity
    log: takes log10 of data before plotting or giving statistics
    norm: divides d_f by normer
    normer: dataframe of same size/shape as d_f
    plot: bool, whether to plot the histograms
    bins: int, number of bins in histogram
    show_sim: bool, whether to show the simulated hist'''
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        warnings.simplefilter(action='ignore', category=UserWarning)
        warnings.simplefilter(action='ignore', category=RuntimeWarning)
        
        if type(d_f) == type(pd.DataFrame()):
            df = d_f.copy().loc[:,d_f.columns[0]]
        else:
            df = d_f.copy()

        commodities = [i for i in np.unique(df.index.get_level_values(1)) if df.loc[idx[:,i]].notna().sum()>2
                      and i not in ['Heavy Mineral Sands','Chromite','Diamonds','Graphite','Phosphate','U3O8',
                                    'Coal','Potash']]
        iterate_over = commodities + ['All']
        if plot:
            fig, ax = easy_subplots(len(iterate_over),height_scale=height_scale, width_scale=width_scale, ncol=3)
        result_df = pd.DataFrame(np.nan,[i.split('|')[0] for i in commodities],['n','Mean','Log mean','std','Log std','KS diff from norm','Shapiro diff from norm','KS diff from all','KW diff from all'])

        if type(d_f)==type(pd.DataFrame()):
            rec_all = d_f.loc[idx[:,commodities],d_f.columns[0]]
        else:
            rec_all = d_f.loc[idx[:,commodities]]
        if norm and type(normer)!=int:
            ph_all = rec_all/normer
        elif norm:
            raise TypeError('normer variable for normalizing the given d_f input is required')
        else:
            ph_all = rec_all.copy()
        ph_all = ph_all.loc[(ph_all>0)&(ph_all!=np.inf)]
        if log:
            ph_all = np.log10(ph_all)
        elif norm:
            ph_all *= 1e6

        if not plot:
            ax = np.array(iterate_over)
            
        hist_data = pd.DataFrame()
        for i,a in zip(iterate_over,ax.flatten()):
            if print_progress:
                print(i,)
            if i=='All':
                i = commodities
                j = result_df.index
            else:
                j = i.split('|')[0]

            rec = df.copy().loc[idx[:,i]]

            if norm:
                ph = rec/normer
            else:
                ph = rec.copy()

            ph = ph.loc[(ph>0)&(ph!=np.inf)]

            if log:
                ph_mod = np.log10(ph)
            else:
                ph_mod = ph.copy()

            # Given we're generating random distributions or sampling randomly, get mean p-values
            # across 100 iterations for Kolmogoriv-Smirnov test of whether the variable is drawn from
            # a normal distribution (generate a normal distribution)
            
            stats_dist = getattr(stats,dist)
            dist_params = stats_dist.fit(ph_mod)
    
            p_ph = stats.kstest(ph_mod,dist,args=dist_params,N=len(ph_mod))[1] # figured out how to do this finally
            p_ks = []
            p_kw = []
            
            for n in np.arange(0,100):
#                 ph_sim = stats.norm.rvs(loc=ph_mod.mean(),scale=ph_mod.std(),size=len(ph_mod),random_state=n)
                
                if len(dist_params)==2:
                    ph_sim = stats_dist.rvs(dist_params[0],dist_params[1],size=len(ph_mod),random_state=0)
                elif len(dist_params)==3:
                    ph_sim = stats_dist.rvs(dist_params[0],dist_params[1],dist_params[2],size=len(ph_mod),random_state=0)
                elif len(dist_params)==4:
                    ph_sim = stats_dist.rvs(dist_params[0],dist_params[1],dist_params[2],dist_params[3],size=len(ph_mod),random_state=0)
                if rounded:
                    ph_sim=[round(i,0) for i in ph_sim]
                
                p_ks += [stats.kstest(ph_mod,ph_all.sample(len(ph_mod),random_state=n))[1]]
                p_kw += [stats.kruskal(ph_mod,ph_all.sample(len(ph_mod),random_state=n))[1]]
            if i==commodities:
                i,j='All','All'
            result_df.loc[j,:] = ph_mod.shape[0],ph.mean(),ph_mod.mean(),ph.std(),ph_mod.std(),p_ph, stats.shapiro(ph_mod)[1], \
                np.mean(p_ks),np.mean(p_kw)
            if plot:
                if show_sim:
                    alpha = 0.5
                else:
                    alpha = 1
                if np.min(np.min(ph_mod))==np.max(np.max(ph_mod)):
                    bins_sp = np.linspace(np.min(np.min(ph_mod))-10,np.max(np.max(ph_mod))+10,bins)
                else:
                    bins_sp = np.linspace(np.min(np.min(ph_mod)),np.max(np.max(ph_mod)),bins)

                (n_hist, bins_hist, patches_hist) = a.hist(ph_mod,bins=bins_sp,alpha=alpha, color='tab:blue')
                if show_sim:
                    (n_sim, bins_sim, patches_sim) = a.hist(ph_sim,bins=bins_sp,color='tab:orange',alpha=alpha)
                a.legend(['Real','Sim.'])
                a.set(title=i.split('|')[0].replace('Steel','Fe')+f', n={len(ph_mod)}')#+'\n|'+', '.join(['{:.3f}'.format(v) if abs(v)>1e-3 and abs(v)<1e3 else '{:.3e}'.format(v) for v in dist_params])+'|')
                try:
                    if log: a.set(xlabel=r'$log_{10}$('+df.name.split(' (')[0]+')')
                    else: a.set(xlabel=df.name.split('(')[0])
                except:
                    if log:
                        a.set(xlabel=r'$log_{10}$('+xlabel+')')
                    else:
                        a.set(xlabel=xlabel)
                hist_data_i = pd.concat([
                    pd.concat([pd.Series(n_hist),pd.Series(n_sim)],axis=1,keys=['Historical','Simulated']),
                    pd.concat([pd.Series(bins_hist),pd.Series(bins_sim)],axis=1,keys=['Historical','Simulated']),
                    ],keys=['Values','Edges'])
                hist_data_i = pd.concat([hist_data_i],keys=[i.split('|')[0]])
                hist_data = pd.concat([hist_data,hist_data_i])

        if plot:
            fig.tight_layout()
            for j in range(1,len(ax.flatten())-len(iterate_over)+1):
                ax[-j].axis('off')
            if show:
                plt.show()
            plt.close()
        return result_df, fig, hist_data

def rename_reshuffle_data(data):
    """
    takes in series in same format as needed for quick_hist()
    """
    data = data.copy()
    level_0 = np.array(data.index.get_level_values(0).unique())
    np.random.shuffle(level_0)
    level_2 = np.array(data.index.get_level_values(2).unique())
    np.random.shuffle(level_2)
    rename_0 = dict(zip(level_0, range(0,len(level_0))))
    rename_2 = dict(zip(level_2, range(0,len(level_2))))
    data = data.rename(rename_0, level=0).rename(rename_2, level=2)
    data.index = data.index.set_names(['','',''])
    return data

def save_quick_hist(many_sg, fig, data, statistics, figure_description, show=False, alt_statistics_name=None):
    # data = rename_reshuffle_data(data)
    fig.savefig(f'{many_sg.folder_path}/figures/figure_{figure_description}.pdf')
    fig.savefig(f'{many_sg.folder_path}/figures/figure_{figure_description}.png')
    if show:
        plt.show()
    plt.close()
    if len(data.shape)>1 and data.shape[1]>data.shape[0]:
        data = data.T
    data = data.rename(columns={'Steel':'Fe'}).rename({'Steel':'Fe'}).replace({'Steel','Fe'})
    data.to_csv(f'{many_sg.folder_path}/figures/figure_{figure_description}_plot.csv')
    if statistics is not None:
        if alt_statistics_name is None:
            alt_statistics_name = 'stats'
        statistics.T.to_csv(f'{many_sg.folder_path}/figures/figure_{figure_description}_{alt_statistics_name}.csv')

def get_regress(primary_only, pri_and_co):
    tmc = primary_only.loc[:,idx['Total Minesite Cost (USD/t)',:]].groupby(level=[0,1]).sum().stack().replace({0:np.nan})
    x_vals = ['Mine Type 1','Numerical Risk','Country Operational Risk','Country Political Risk','Country Terrorism Risk','Country Security Risk','Head Grade (%)','Commodity Price (USD/t)','Metal Payable Percent (%)']
    other = pri_and_co.loc[:,x_vals].droplevel(-1).stack()
    other.loc[:,'Numerical Risk'] = other.replace({'Insignificant':1,'Low':2,'Medium':3,'High':4,'Extreme':5}).loc[:,[i for i in other.columns if 'Risk' in i and 'Country' in i]].sum(axis=1)
    regress = pd.concat([tmc,other],axis=1).replace({0:np.nan})
    regress = regress.astype({'Mine Type 1':'category','Numerical Risk':'float',
                            'Country Operational Risk':'category','Country Political Risk':'category',
                            'Country Terrorism Risk':'category','Country Security Risk':'category',
                            'Total Minesite Cost (USD/t)':'float','Head Grade (%)':'float'})
    regress.loc[:,'SXEW'] = 0
    regress.loc[regress['Metal Payable Percent (%)']==100,'SXEW'] = 1
    regress.loc[:,'Commodity'] = regress.index.get_level_values(1)
    regress = pd.get_dummies(regress, columns=['Mine Type 1'],drop_first=True)
    log = ['Head Grade (%)','Total Minesite Cost (USD/t)','Commodity Price (USD/t)']
    regress.loc[:,log] = np.log(regress.loc[:,log])
    regress = regress.rename(columns=dict(zip(log,['log_'+i for i in log])))
    return regress

def get_regress_yearso(primary_only, pri_and_co, tcrc_primary):
    do_log=False
    individual = True
    metal = 'Gold'
    yearso = (primary_only.loc[:,idx['Sustaining CAPEX ($M)',:]].astype(float).groupby(level=[0,1]).sum().stack().replace({0:np.nan}))
    x_vals = ['Mine Type 1','Numerical Risk','Country Operational Risk','Country Political Risk',
            'Country Terrorism Risk','Country Security Risk','Head Grade (%)',
            'Commodity Price (USD/t)','Mill Capacity - tonnes/year','Metal Payable Percent (%)']
    other = pri_and_co.loc[:,x_vals].droplevel(-1).stack()
    other.loc[:,'Numerical Risk'] = other.replace({'Insignificant':1,'Low':2,'Medium':3,'High':4,'Extreme':5}).loc[:,[i for i in other.columns if 'Risk' in i and 'Country' in i]].sum(axis=1)
    other = other.rename(columns={'Mill Capacity - tonnes/year':'Capacity (kt)'})
    other.loc[:,'Capacity (kt)'] = other.loc[:,'Capacity (kt)'].astype(float)/1e3
    regress_yearso = pd.concat([yearso,other,
                            tcrc_primary['Refining charge type'].droplevel(2).drop('26490a',level=0)
                        ],axis=1).replace({0:np.nan})
    regress_yearso = regress_yearso.astype({'Mine Type 1':'category','Numerical Risk':'float',
                            'Country Operational Risk':'category','Country Political Risk':'category',
                            'Country Terrorism Risk':'category','Country Security Risk':'category',
                            'Sustaining CAPEX ($M)':'float','Head Grade (%)':'float',
                            'Refining charge type':'category','Capacity (kt)':'float'})
    regress_yearso.loc[:,'SXEW'] = 0
    regress_yearso.loc[regress_yearso['Metal Payable Percent (%)']==100,'SXEW'] = 1
    regress_yearso.loc[:,'Commodity'] = regress_yearso.index.get_level_values(1)
    regress_yearso = pd.get_dummies(regress_yearso, columns=['Mine Type 1'],drop_first=True)
    regress_yearso = pd.get_dummies(regress_yearso, columns=['Refining charge type'],drop_first=True)
    # below sets to use means
    regress_yearso = pd.concat([regress_yearso.loc[:,regress_yearso.dtypes==float].groupby(level=[0,1]).mean(),regress_yearso.loc[idx[:,:,2018],regress_yearso.dtypes!=float].droplevel(-1)],axis=1)
    regress_yearso = pd.concat([regress_yearso,
                                primary_only['Ore Treated (kt)'].cumsum(axis=1).rename(columns={2019:'Cumulative Ore Treated (kt)'})['Cumulative Ore Treated (kt)'].droplevel(2).astype(float).replace({0:np.nan}),
                                primary_only['Ore Treated (kt)'].rename(columns={2019:'Ore Treated (kt)'})['Ore Treated (kt)'].droplevel(2).astype(float).replace({0:np.nan}),
                                primary_only['Reserves: Ore Tonnage (tonnes)'].astype(float).mul(1e-3).rename(columns={2018:'Reserves (kt)'})['Reserves (kt)'].droplevel(2).replace({0:np.nan})],axis=1)
    # dividing by ore treated for normalization
    regress_yearso.loc[:,'Reserves (kt)'] /= regress_yearso['Ore Treated (kt)']
    regress_yearso.loc[:,'Cumulative Ore Treated (kt)'] /= regress_yearso['Ore Treated (kt)']
    if do_log:
        log = ['Head Grade (%)','Commodity Price (USD/t)','Sustaining CAPEX ($M)','Capacity (kt)',
        'Cumulative Ore Treated (kt)','Reserves (kt)']
        regress_yearso.loc[:,log] = np.log(regress_yearso.loc[:,log])
        regress_yearso = regress_yearso.rename(columns=dict(zip(log,['log_'+i for i in log]))).dropna()
    else:
        log = []
        regress_yearso = regress_yearso.dropna()
    return regress_yearso

def get_regress_oge(primary_only, pri_and_co, tcrc_primary, oge_results):
    do_log=True
    individual = True
    metal = 'Gold'
    oge = (primary_only.loc[:,idx['Sustaining CAPEX ($M)',:]].astype(float).groupby(level=[0,1]).sum().stack().replace({0:np.nan}))
    x_vals = ['Mine Type 1','Numerical Risk','Country Operational Risk','Country Political Risk',
            'Country Terrorism Risk','Country Security Risk','Head Grade (%)',
            'Commodity Price (USD/t)','Mill Capacity - tonnes/year','Metal Payable Percent (%)']
    other = pri_and_co.loc[:,x_vals].droplevel(-1).stack()
    other.loc[:,'Numerical Risk'] = other.replace({'Insignificant':1,'Low':2,'Medium':3,'High':4,'Extreme':5}).loc[:,[i for i in other.columns if 'Risk' in i and 'Country' in i]].sum(axis=1)
    other = other.rename(columns={'Mill Capacity - tonnes/year':'Capacity (kt)'})
    other.loc[:,'Capacity (kt)'] = other.loc[:,'Capacity (kt)'].astype(float)/1e3
    regress_oge = pd.concat([oge,other,
                            tcrc_primary['Refining charge type'].droplevel(2).drop('26490a',level=0)
                        ],axis=1).replace({0:np.nan})
    regress_oge = regress_oge.astype({'Mine Type 1':'category','Numerical Risk':'float',
                            'Country Operational Risk':'category','Country Political Risk':'category',
                            'Country Terrorism Risk':'category','Country Security Risk':'category',
                            'Sustaining CAPEX ($M)':'float','Head Grade (%)':'float',
                            'Refining charge type':'category','Capacity (kt)':'float'})
    regress_oge.loc[:,'SXEW'] = 0
    regress_oge.loc[regress_oge['Metal Payable Percent (%)']==100,'SXEW'] = 1
    regress_oge.loc[:,'Commodity'] = regress_oge.index.get_level_values(1)
    regress_oge = pd.get_dummies(regress_oge, columns=['Mine Type 1'],drop_first=True)
    regress_oge = pd.get_dummies(regress_oge, columns=['Refining charge type'],drop_first=True)
    # below sets to use means
    regress_oge = pd.concat([regress_oge.loc[:,regress_oge.dtypes==float].groupby(level=[0,1]).mean(),regress_oge.loc[idx[:,:,2018],regress_oge.dtypes!=float].droplevel(-1)],axis=1)
    regress_oge = pd.concat([regress_oge,
                                primary_only['Ore Treated (kt)'].cumsum(axis=1).rename(columns={2019:'Cumulative Ore Treated (kt)'})['Cumulative Ore Treated (kt)'].droplevel(2).astype(float).replace({0:np.nan}),
                                primary_only['Ore Treated (kt)'].rename(columns={2019:'Ore Treated (kt)'})['Ore Treated (kt)'].droplevel(2).astype(float).replace({0:np.nan}),
                                primary_only['Reserves: Ore Tonnage (tonnes)'].astype(float).mul(1e-3).rename(columns={2018:'Reserves (kt)'})['Reserves (kt)'].droplevel(2).replace({0:np.nan}),
                                oge_results.rename(columns={'slope':'OGE'}).OGE],axis=1)
    # dividing by ore treated for normalization
    # regress_oge.loc[:,'Reserves norm (kt)'] = regress_oge['Reserves (kt)'] / regress_oge['Ore Treated (kt)']
    # regress_oge.loc[:,'Cumulative Ore Treated norm (kt)'] = regress_oge['Cumulative Ore Treated (kt)'] / regress_oge['Ore Treated (kt)']
    if do_log:
        log = ['Head Grade (%)','Commodity Price (USD/t)','Sustaining CAPEX ($M)','Capacity (kt)',
        'Cumulative Ore Treated (kt)','Reserves (kt)']
        regress_oge.loc[:,log] = np.log(regress_oge.loc[:,log])
        regress_oge = regress_oge.rename(columns=dict(zip(log,['log_'+i for i in log]))).dropna()
    else:
        log = []
        regress_oge = regress_oge.dropna()
    regress_oge = regress_oge.loc[regress_oge['OGE']>-0.8,:]
    regress_oge = regress_oge.loc[regress_oge['OGE']<0,:]
    return regress_oge

def get_regress_rr(primary_only, pri_and_co, tcrc_primary):
    rr = (primary_only.loc[:,idx['Recovery Rate (%)',:]].astype(float).groupby(level=[0,1]).sum().stack().replace({0:np.nan}))
    x_vals = ['Mine Type 1','Numerical Risk','Country Operational Risk','Country Political Risk',
            'Country Terrorism Risk','Country Security Risk','Head Grade (%)',
            'Commodity Price (USD/t)','Metal Payable Percent (%)']
    other = pri_and_co.loc[:,x_vals].droplevel(-1).stack()
    other.loc[:,'Numerical Risk'] = other.replace({'Insignificant':1,'Low':2,'Medium':3,'High':4,'Extreme':5}).loc[:,[i for i in other.columns if 'Risk' in i and 'Country' in i]].sum(axis=1)
    regress_rr = pd.concat([rr,other,
                            tcrc_primary['Refining charge type'].droplevel(2).drop('26490a',level=0)
                        ],axis=1).replace({0:np.nan})
    regress_rr = regress_rr.astype({'Mine Type 1':'category','Numerical Risk':'float',
                            'Country Operational Risk':'category','Country Political Risk':'category',
                            'Country Terrorism Risk':'category','Country Security Risk':'category',
                            'Recovery Rate (%)':'float','Head Grade (%)':'float','Refining charge type':'category'})
    ph = (other['Metal Payable Percent (%)']==100).astype(int)
    ph.name='SX-EW'
    regress_rr = pd.concat([regress_rr,ph],axis=1)
    regress_rr.loc[:,'Commodity'] = regress_rr.index.get_level_values(1)
    regress_rr = pd.get_dummies(regress_rr, columns=['Mine Type 1'],drop_first=True)
    regress_rr = pd.get_dummies(regress_rr, columns=['Refining charge type'],drop_first=True)
    log = ['Head Grade (%)','Commodity Price (USD/t)']
    regress_rr.loc[:,log] = np.log(regress_rr.loc[:,log])
    regress_rr = regress_rr.rename(columns=dict(zip(log,['log_'+i for i in log])))
    return regress_rr

def get_regress_tcrc_conc(tcrc_pri, pri_and_co):
    tcrc_conc = tcrc_pri.loc[:,['TCRC (USD/t ore)','Refining charge type']].replace({0:np.nan})
    x_vals = ['Mine Type 1','Numerical Risk','Country Operational Risk','Country Political Risk',
            'Country Terrorism Risk','Country Security Risk','Head Grade (%)',
            'Commodity Price (USD/t)','Recovery Rate (%)','Metal Payable Percent (%)']
    other = pri_and_co.loc[:,x_vals].droplevel(-1).stack()
    other.loc[:,'Numerical Risk'] = other.replace({'Insignificant':1,'Low':2,'Medium':3,'High':4,'Extreme':5}).loc[:,[i for i in other.columns if 'Risk' in i and 'Country' in i]].sum(axis=1)
    regress_tcrc_conc = pd.concat([tcrc_conc,other],axis=1).replace({0:np.nan})
    if '26490a' in regress_tcrc_conc.index.get_level_values(0):
        regress_tcrc_conc.drop('26490a',inplace=True,level=0)
    regress_tcrc_conc.loc[:,'SXEW'] = 0
    regress_tcrc_conc.loc[regress_tcrc_conc['Metal Payable Percent (%)']==100,'SXEW'] = 1
    regress_tcrc_conc = regress_tcrc_conc.astype({'Mine Type 1':'category','Numerical Risk':'float',
                            'Country Operational Risk':'category','Country Political Risk':'category',
                            'Country Terrorism Risk':'category','Country Security Risk':'category',
                            'TCRC (USD/t ore)':'float','Head Grade (%)':'float','Refining charge type':'category'})
    regress_tcrc_conc.loc[:,'Commodity'] = regress_tcrc_conc.index.get_level_values(1)
    regress_tcrc_conc = pd.get_dummies(regress_tcrc_conc, columns=['Mine Type 1'],drop_first=True)
    regress_tcrc_conc = regress_tcrc_conc.loc[regress_tcrc_conc['Refining charge type'].notna()]
    regress_tcrc_conc = pd.get_dummies(regress_tcrc_conc, columns=['Refining charge type'],drop_first=True)
    log = ['Head Grade (%)','TCRC (USD/t ore)','Commodity Price (USD/t)']
    regress_tcrc_conc.loc[:,log] = np.log(regress_tcrc_conc.loc[:,log])
    regress_tcrc_conc = regress_tcrc_conc.rename(columns=dict(zip(log,['log_'+i for i in log])))
    return regress_tcrc_conc

def get_regress_scapex(primary_only, pri_and_co, tcrc_primary):
    scapex = (primary_only.loc[:,idx['Sustaining CAPEX ($M)',:]].astype(float).groupby(level=[0,1]).sum().stack().replace({0:np.nan}))
    x_vals = ['Mine Type 1','Numerical Risk','Country Operational Risk','Country Political Risk',
            'Country Terrorism Risk','Country Security Risk','Head Grade (%)',
            'Commodity Price (USD/t)','Mill Capacity - tonnes/year','Metal Payable Percent (%)']
    other = pri_and_co.loc[:,x_vals].droplevel(-1).stack()
    other.loc[:,'Numerical Risk'] = other.replace({'Insignificant':1,'Low':2,'Medium':3,'High':4,'Extreme':5}).loc[:,[i for i in other.columns if 'Risk' in i and 'Country' in i]].sum(axis=1)
    other = other.rename(columns={'Mill Capacity - tonnes/year':'Capacity (kt)'})
    other.loc[:,'Capacity (kt)'] = other.loc[:,'Capacity (kt)'].astype(float)/1e3
    regress_scapex = pd.concat([scapex,other,
                            tcrc_primary['Refining charge type'].droplevel(2).drop('26490a',level=0)
                    ],axis=1).replace({0:np.nan})
    regress_scapex = regress_scapex.astype({'Mine Type 1':'category','Numerical Risk':'float',
                            'Country Operational Risk':'category','Country Political Risk':'category',
                            'Country Terrorism Risk':'category','Country Security Risk':'category',
                            'Sustaining CAPEX ($M)':'float','Head Grade (%)':'float',
                            'Refining charge type':'category','Capacity (kt)':'float'})
    regress_scapex.loc[:,'SXEW'] = 0
    regress_scapex.loc[regress_scapex['Metal Payable Percent (%)']==100,'SXEW'] = 1
    regress_scapex.loc[:,'Commodity'] = regress_scapex.index.get_level_values(1)
    regress_scapex.loc[:,'sCAPEX norm ($M/kt)'] = regress_scapex['Sustaining CAPEX ($M)']/regress_scapex['Capacity (kt)']
    regress_scapex = pd.get_dummies(regress_scapex, columns=['Mine Type 1'],drop_first=True)
    regress_scapex = pd.get_dummies(regress_scapex, columns=['Refining charge type'],drop_first=True)
    # below sets to use means
    # regress_scapex = pd.concat([regress_scapex.loc[:,regress_scapex.dtypes==float].groupby(level=[0,1]).mean(),regress_scapex.loc[idx[:,:,2018],regress_scapex.dtypes!=float].droplevel(-1)],axis=1)
    log = ['Head Grade (%)','Commodity Price (USD/t)','Sustaining CAPEX ($M)','Capacity (kt)','sCAPEX norm ($M/kt)']
    regress_scapex = regress_scapex.drop(columns=['Country Operational Risk', 'Country Political Risk',
    'Country Security Risk', 'Country Terrorism Risk','Commodity']).groupby(level=[0,1]).mean()
    regress_scapex.loc[:,log] = np.log(regress_scapex.loc[:,log])
    regress_scapex = regress_scapex.rename(columns=dict(zip(log,['log_'+i for i in log]))).dropna()
    return regress_scapex

def get_tcrc_primary(concentrates, opening):
    # Checking that none overlap (initially used to make corrections below):
    # ind_check = refining_charges.notna().sum(axis=1)>1
    # ind_check = ind_check[ind_check].index
    # refining_charges.loc[ind_check]

    treatment_charges = concentrates.loc[:,[i for i in concentrates.columns.levels[0] if 'Treatment' in i]].stack().stack().sum(axis=1).replace(0,np.nan)
    treatment_charge_type =  concentrates.replace(0,np.nan).loc[:,[i for i in concentrates.columns.levels[0] if 'Treatment' in i]].stack().stack().dropna(how='all').notna().idxmax(axis=1)
    treatment_charges = pd.concat([treatment_charges, treatment_charge_type],axis=1,keys=['Treatment charges (USD/t)','Treatment charge type'])
    treatment_charges = pd.concat([treatment_charges, pd.concat([treatment_charges.loc['26490']],keys=['26490a'])])

    refining_charges = concentrates.replace(0,np.nan).loc[:,[i for i in concentrates.columns.levels[0] if 'Refining' in i]].stack().stack().sort_index()
    refining_charges = pd.concat([refining_charges,pd.concat([refining_charges.loc['26490']],keys=['26490a'])]).sort_index()
    for i in np.arange(1994,1999):
        refining_charges.loc[idx['26490','Copper','Copper',i],'SxCu Refining Charge (cents/lb) (¢/lb)'] = np.nan
        refining_charges.loc[idx['26490a','Copper','Copper',i],'Conc Refining Charge (cents/lb) (¢/lb)'] = np.nan
    ind_check = refining_charges.notna().sum(axis=1)>1
    ind_check = ind_check[ind_check].index

    refining_charge_type = refining_charges.dropna(how='all').notna().idxmax(axis=1)
    refining_charges.loc[:,[i for i in refining_charges.columns if 'cents/lb' in str(i)]] *= 22.0462
    refining_charges.loc[:,[i for i in refining_charges.columns if '$/oz' in str(i)]] *= 16*2204.62
    refining_charges = refining_charges.sum(axis=1)
    refining_charges = pd.concat([refining_charges, refining_charge_type],axis=1,keys=['Refining charges (USD/t)','Refining charge type'])

    rc_pri = refining_charges.loc[[i for i in refining_charges.index if i[1]==i[2]]].droplevel(2)
    rc_by = refining_charges.loc[[i for i in refining_charges.index if i[1]!=i[2]]]
    tc_pri = treatment_charges.loc[[i for i in treatment_charges.index if i[1]==i[2]]].droplevel(2)
    tc_by = treatment_charges.loc[[i for i in treatment_charges.index if i[1]!=i[2]]]

    conc_prod = concentrates.loc[:,[i for i in concentrates.columns.levels[0] if 'Production' in i]].stack().stack().sum(axis=1).replace(0,np.nan)
    conc_prod_type = concentrates.replace(0,np.nan).loc[:,[i for i in concentrates.columns.levels[0] if 'Production' in i]].stack().stack().dropna(how='all').notna().idxmax(axis=1)
    conc_prod = pd.concat([conc_prod, conc_prod_type],axis=1,keys=['Concentrate production (kt)','Treatment charge type'])
    conc_prod = pd.concat([conc_prod, pd.concat([conc_prod.loc['26490']],keys=['26490a'])])

    # Getting an alternate paid metal production: 
    prod = concentrates.astype(float).loc[:,[i for i in concentrates.columns if 'Production' in i[0]]].stack().stack().sum(axis=1).replace(0,np.nan)
    grad = concentrates.astype(float).loc[:,[i for i in concentrates.columns if 'Grade' in i[0] and not ('%' in i[0] and i[2]=='Palladium')]].stack()
    grad = grad.loc[[i for i in grad.index if i[1]==i[2]]].droplevel(2)
    grad1 = grad.loc[:,[i for i in grad.columns if 'g/tonne' in i[0]]]*1e-4
    grad2 = grad.loc[:,[i for i in grad.columns if '%' in i[0]]]
    grad = pd.concat([grad1,grad2],axis=1).stack().sum(axis=1)
    payp = concentrates.loc[:,[i for i in concentrates.columns if 'Metal Payable' in i[0]]].stack().stack().sum(axis=1)
    payp = payp.loc[[i for i in payp.index if i[1]==i[2]]].droplevel(2)
    prod = prod.loc[[i for i in prod.index if i[1]==i[2]]].droplevel(2)
    cols = np.intersect1d(prod.index,grad.index)
    cols = np.intersect1d(cols,payp.index)
    prod = prod.loc[cols]
    grad = grad.loc[cols]
    payp = payp.loc[cols]
    pmp_conc = prod*grad*payp/1e4
    pmp_conc.index = pmp_conc.index.set_names(['Property ID','Primary Commodity','Year'])

    # Converting RC from paid metal basis to total
    pmp = opening.loc[:,idx['Paid Metal Produced (kilotonnes)',:]].droplevel(0,axis=1).stack()#.droplevel(2)
    pmp_pri = pmp.loc[[i for i in pmp.index if i[1]==i[2]]].droplevel(2).stack()
    pmp_by = pmp.loc[[i for i in pmp.index if i[1]==i[2]]].droplevel(2).stack()
    rc_pri.index = rc_pri.index.set_names(['Property ID','Primary Commodity','Year'])
    rc_pri_abs = (pmp_conc* rc_pri['Refining charges (USD/t)'])

    # Converting TC from DMT concentrate basis to ore treated basis
    tc_pri.index = tc_pri.index.set_names(['Property ID','Primary Commodity','Year'])
    cp_pri = conc_prod.loc[[i for i in conc_prod.index if i[1]==i[2]],'Concentrate production (kt)'].droplevel(2)
    cp_pri.index = cp_pri.index.set_names(['Property ID','Primary Commodity','Year'])
    tc_pri_abs = (cp_pri*tc_pri['Treatment charges (USD/t)'])

    tcrc_pri_abs = (tc_pri_abs.unstack().replace(np.nan,0)+rc_pri_abs.unstack().replace(np.nan,0)).replace(0,np.nan).stack().dropna()
    ot = opening.loc[:,idx['Ore Treated (kilotonnes)',:]].droplevel([0,2],axis=1).stack()
    tcrc_pri_o = (tcrc_pri_abs.unstack()/ot.unstack()).stack()
    ref_ph = refining_charges.loc[[i for i in refining_charges.index if i[1]==i[2]]].droplevel(2)['Refining charge type']
    tre_ph = treatment_charges.loc[[i for i in treatment_charges.index if i[1]==i[2]]].droplevel(2)['Treatment charge type']
    ref_ph.index = ref_ph.index.set_names(['Property ID','Primary Commodity','Year'])
    tre_ph.index = tre_ph.index.set_names(['Property ID','Primary Commodity','Year'])
    tcrc_pri = pd.concat([tcrc_pri_o,
                        ref_ph, tre_ph
                        ],axis=1)
    tcrc_pri.index = tcrc_pri.index.set_names([0,1,2])
    tcrc_pri = tcrc_pri.rename(columns={0:'TCRC (USD/t ore)'})
    # tcrc_pri.replace(dict(zip(tcrc_pri['Refining charge type'].unique(), [str(i).split('Refining')[0] for i in tcrc_pri['Refining charge type'].unique()])),inplace=True)
    # tcrc_pri.replace(dict(zip(tcrc_pri['Treatment charge type'].unique(), [str(i).split('Treatment')[0] for i in tcrc_pri['Treatment charge type'].unique()])),inplace=True)
    # tcrc_primary = tcrc_pri.copy()

    # # Below is the previous version that didn't correct for the different basis
    tcrc = pd.concat([treatment_charges,refining_charges],axis=1)

    tcrc.loc[:,'TCRC (USD/t)'] = tcrc['Treatment charges (USD/t)'].fillna(0) + tcrc['Refining charges (USD/t)'].fillna(0)
    tcrc = tcrc.replace(0,np.nan).dropna(how='all')
    tcrc.index = tcrc.index.set_names([0,1,2,3])
    tcrc.replace(dict(zip(tcrc['Refining charge type'].unique(), [str(i).split('Refining')[0] for i in tcrc['Refining charge type'].unique()])),inplace=True)
    tcrc.replace(dict(zip(tcrc['Treatment charge type'].unique(), [str(i).split('Treatment')[0] for i in tcrc['Treatment charge type'].unique()])),inplace=True)
    tcrc.replace('nan',np.nan,inplace=True)

    tcrc_byprod = tcrc.loc[[i for i in tcrc.index if i[1]!=i[2]]]
    tcrc_primary = tcrc.loc[[i for i in tcrc.index if i[1]==i[2]]]
    return tcrc_primary, tcrc_pri

def get_regress_tcm(primary_only, pri_and_co):
    tcm = pd.concat([
        (primary_only.loc[:,'Commodity Price (USD/t)']-
        primary_only.loc[:,'Total Cash Cost (USD/t)']+
        primary_only['Smelting & Refining Cost (USD/t)']).astype(float)],keys=['Total Cash Margin (USD/t)'],axis=1).groupby(level=[0,1]).sum().stack().replace({0:np.nan})
    x_vals = ['Mine Type 1','Numerical Risk','Country Operational Risk','Country Political Risk','Country Terrorism Risk','Country Security Risk','Head Grade (%)','Commodity Price (USD/t)','Metal Payable Percent (%)','Global Region']
    other = pri_and_co.loc[:,x_vals].droplevel(-1).stack()
    other.loc[:,'Numerical Risk'] = other.replace({'Insignificant':1,'Low':2,'Medium':3,'High':4,'Extreme':5}).loc[:,[i for i in other.columns if 'Risk' in i and 'Country' in i]].sum(axis=1)
    regress_tcm = pd.concat([tcm,other],axis=1).replace({0:np.nan})
    regress_tcm = regress_tcm.astype({'Mine Type 1':'category','Numerical Risk':'float',
                            'Country Operational Risk':'category','Country Political Risk':'category',
                            'Country Terrorism Risk':'category','Country Security Risk':'category',
                            'Total Cash Margin (USD/t)':'float','Head Grade (%)':'float'})
    regress_tcm.loc[:,'SXEW'] = 0
    regress_tcm.loc[regress_tcm['Metal Payable Percent (%)']==100,'SXEW'] = 1
    regress_tcm.loc[:,'Commodity'] = regress_tcm.index.get_level_values(1)
    regress_tcm.loc[:,'Capacity (kt)'] = primary_only['Mill Capacity - tonnes/year'].stack().droplevel(2).astype(float)/1e3
    regress_tcm.loc[:,'TCM norm (USD/t/kt)'] = regress_tcm['Total Cash Margin (USD/t)']/regress_tcm['Capacity (kt)']
    regress_tcm = pd.get_dummies(regress_tcm, columns=['Mine Type 1'],drop_first=True)
    regress_tcm = pd.get_dummies(regress_tcm, columns=['Global Region'],drop_first=True)
    log = ['Head Grade (%)','Total Cash Margin (USD/t)','Commodity Price (USD/t)','TCM norm (USD/t/kt)','Capacity (kt)']
    regress_tcm.loc[:,log] = np.log(regress_tcm.loc[:,log])
    regress_tcm = regress_tcm.rename(columns=dict(zip(log,['log_'+i for i in log])))
    return regress_tcm

def get_regress_grade(primary_only, pri_and_co, tcrc_primary):
    grade = (primary_only.loc[:,idx['Recovery Rate (%)',:]].astype(float).groupby(level=[0,1]).sum().stack().replace({0:np.nan}))
    x_vals = ['Mine Type 1','Numerical Risk','Country Operational Risk','Country Political Risk',
            'Country Terrorism Risk','Country Security Risk','Head Grade (%)',
            'Commodity Price (USD/t)']
    other = pri_and_co.loc[:,x_vals].droplevel(-1).stack()
    other.loc[:,'Numerical Risk'] = other.replace({'Insignificant':1,'Low':2,'Medium':3,'High':4,'Extreme':5}).loc[:,[i for i in other.columns if 'Risk' in i and 'Country' in i]].sum(axis=1)
    regress_grade = pd.concat([grade,other,
                            tcrc_primary['Refining charge type'].droplevel(2).drop('26490a',level=0)
                        ],axis=1).replace({0:np.nan})
    regress_grade = regress_grade.astype({'Mine Type 1':'category','Numerical Risk':'float',
                            'Country Operational Risk':'category','Country Political Risk':'category',
                            'Country Terrorism Risk':'category','Country Security Risk':'category',
                            'Recovery Rate (%)':'float','Head Grade (%)':'float','Refining charge type':'category'})
    regress_grade.loc[:,'Commodity'] = regress_grade.index.get_level_values(1)
    regress_grade = pd.get_dummies(regress_grade, columns=['Mine Type 1'],drop_first=True)
    regress_grade = pd.get_dummies(regress_grade, columns=['Refining charge type'],drop_first=True)
    log = ['Head Grade (%)','Commodity Price (USD/t)']
    regress_grade.loc[:,log] = np.log(regress_grade.loc[:,log])
    regress_grade = regress_grade.rename(columns=dict(zip(log,['log_'+i for i in log])))
    return regress_grade

def discont_to_cont(df_twolevelindex_yearcolumns):
    ph = df_twolevelindex_yearcolumns.copy()
    df = pd.DataFrame()
    for ind in ph.index:
        ph6 = ph.loc[ind]
        g = ph6.isna().cumsum()
        if ph6.notna().sum()>0:
            ph7 = pd.concat([ph6[g==i] for i in np.arange(0,g.max()+1) if ph6[g==i].notna().sum()>0],axis=1).T
            ph7.index = pd.MultiIndex.from_tuples([(ind[0],ind[1],i) for i in np.arange(0,ph7.shape[0])])
        else:
            ph7 = pd.DataFrame()
        df = pd.concat([df,ph7])
    return df

def get_oge(primary_only, pri_and_co, tcrc_primary):
    ore_treated_d = discont_to_cont(primary_only['Ore Treated (kt)'].droplevel(2))
    head_grade_d = discont_to_cont(primary_only['Head Grade (%)'].droplevel(2))
    head_grade_d = head_grade_d.stack()
    ore_treated_d = ore_treated_d.stack()
    ind = np.intersect1d(ore_treated_d.index, head_grade_d.index)
    head_grade_d, ore_treated_d = head_grade_d.loc[ind], ore_treated_d.loc[ind]
    hgd = head_grade_d.unstack()
    otd = ore_treated_d.unstack()
    otd = otd.astype(float)
    hgd = hgd.astype(float)

    oge_results = pd.DataFrame(np.nan,otd.index,['const','slope','rsq'])
    for i in otd.index:
        y1 = hgd.loc[i].dropna()
        x1 = otd.loc[i].cumsum().dropna()
        x1 = np.log(x1.replace(0,np.nan))
        y1 = np.log(y1.replace(0,np.nan))
        if x1.notna().sum()>2 and y1.notna().sum()>2 and y1.iloc[0]>y1.iloc[-1]:
            r,m = do_a_regress(x1,y1,plot=False)
            oge_results.loc[i] = r['const'], r['slope'], m.rsquared
    # oge_results = oge_results.loc[oge_results['slope']<0]
    oge_results = oge_results.loc[idx[:,:,0]]
    regress_oge = get_regress_oge(primary_only, pri_and_co, tcrc_primary, oge_results)
    return oge_results, regress_oge

def find_best_regress_plot(regress_yearso_, individual=True,metal='Gold', exp=False,
                          verbose = True, ax = 0): 
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        regress_yearso = regress_yearso_.copy()
        if individual:
            regress_yearso = regress_yearso.loc[idx[:,metal],:]
        summ = pd.DataFrame()
        for j in ['none','none']:
            for k in ['Grade','none']:
                for n in ['Reserves','none']:
                    for o in ['Price','none']:
                        for p in ['Numerical','none']:
                            columnar = [i for i in regress_yearso.columns if (j in i or k in i or n in i or o in i or p in i
                                                                                      or 'Mine Type' in i)]
                            x = sm.add_constant(regress_yearso[columnar])
                            y = regress_yearso[[i for i in regress_yearso.columns if 'Cumulative Ore Treated (kt)' in i]]
                            m = sm.GLS(y.astype(float),x.astype(float),missing='drop').fit(cov_type='HC3')
                            summ.loc['-'.join([j,k,n,o,p]),'AIC'] = m.aic
                            summ.loc['-'.join([j,k,n,o,p]),'BIC'] = m.bic
                            summ.loc['-'.join([j,k,n,o,p]),'rsq'] = m.rsquared
                            summ.loc['-'.join([j,k,n,o,p]),'model'] = m
        if verbose:
            display(summ.loc[summ.AIC.idxmin(),'model'].summary())  
        # warning is due to small value of Placer placeholder, can exclude it and warning goes away and no values change

        if type(ax)==int:
            fig,ax = plt.subplots(1,1,figsize=(10,10))
        j,k,n,o,p = summ.AIC.idxmin().split('-')
        columnar = [i for i in regress_yearso.columns if (j in i or k in i or n in i or o in i or p in i
                                                                  or 'Mine Type' in i)]
        x = sm.add_constant(regress_yearso[columnar]).astype(float)
        y = regress_yearso[[i for i in regress_yearso.columns if 'Cumulative Ore Treated (kt)' in i]].astype(float)
        m = sm.GLS(y.astype(float),x.astype(float),missing='drop').fit(cov_type='HC3')
        if exp:
            actual = np.exp(regress_yearso[[i for i in regress_yearso.columns if 'Cumulative Ore Treated' in i]])
            predict = np.exp(m.predict(x))
        else:
            actual = (regress_yearso[[i for i in regress_yearso.columns if 'Cumulative Ore Treated' in i]])
            predict = (m.predict(x))

        lin_predict = pd.concat([
            actual,predict,
            regress_yearso.Commodity],
            axis=1,keys=['actual','predicted','Commodity']).droplevel(1,axis=1)
        do_a_regress(lin_predict.actual,lin_predict.predicted,ax=ax,loc='lower right')
        sns.scatterplot(data=lin_predict, x='actual',y='predicted',hue='Commodity',ax=ax)
        ax.legend(title=None, labelspacing=0.2,markerscale=2,fontsize=22,loc='upper left')
        ax.set(ylim=ax.get_xlim())
        return summ

def save_regression_table(many_sg, m, table_description):
    out = m.summary().tables
    out = pd.DataFrame(out[-2])
    out.loc[0,0] = 'R2={:.3f}, n={:.0f}'.format(m.rsquared, m.nobs)
    out.to_csv(f'{many_sg.folder_path}/tables/table_{table_description}.csv')

def plot_lin_predict(lin_predict_, ax, xlabel):
    lin_predict = lin_predict_.copy()
    if 'Commodity' in lin_predict.columns:
        lin_predict = lin_predict.drop('Commodity', axis=1)
    bins = np.linspace(lin_predict.min().min(), lin_predict.max().max(), 50)
    (n_hist, bins_hist, patches_hist) = ax.hist((lin_predict.Actual.dropna()), bins=bins, label='Actual')
    (n_sim, bins_sim, patches_sim) = ax.hist((lin_predict.Predicted.dropna()), bins=bins, alpha=0.5, label='Predicted')
    hist_data_i = pd.concat([
        pd.concat([pd.Series(n_hist),pd.Series(n_sim)],axis=1,keys=['Historical','Simulated']),
        pd.concat([pd.Series(bins_hist),pd.Series(bins_sim)],axis=1,keys=['Historical','Simulated']),
        ],keys=['Values','Edges'])
    ax.legend()
    ax.set_xlabel(xlabel)
    ax.set(title='Distribution comparison')
    return hist_data_i

def plot_lin_predict_scatter(lin_predict_, ax, exponentiate=False):
    lin_predict = lin_predict_.copy()
    if exponentiate:
        lin_predict[['Actual','Predicted']] = np.exp(lin_predict[['Actual','Predicted']].astype(float) )
    do_a_regress(lin_predict.Actual.astype(float),lin_predict.Predicted.astype(float),ax=ax,loc='lower right',)
    lin_predict = lin_predict.rename(columns={'Steel':'Fe'}).replace({'Steel':'Fe'}).rename({'Steel':'Fe'})
    sns.scatterplot(data=lin_predict, x='Actual',y='Predicted',hue='Commodity',ax=ax)
    ax.legend(title=None, labelspacing=0.2,markerscale=1,fontsize=18,loc='upper left')
    ax.set(title='Unlogged' if exponentiate else 'Logged')
    low = min([ax.get_xlim()[0], ax.get_ylim()[0]])
    high = max([ax.get_xlim()[1], ax.get_ylim()[1]])
    ax.set_xlim([low,high])
    ax.set_ylim([low,high])

def AIC(panel_model):
    """ 
    For use with linearmodels panel regression models.
    
    Eqn from https://www.statology.org/aic-in-python/
    """
    L=panel_model.loglik
    K=panel_model.df_model+2
    return 2*K - 2*L

def hausman(fe, re):
    """
    Compute hausman test for fixed effects/random effects models
    b = beta_fe
    B = beta_re
    From theory we have that b is always consistent, but B is consistent
    under the alternative hypothesis and efficient under the null.
    The test statistic is computed as
    z = (b - B)' [V_b - v_B^{-1}](b - B)
    The statistic is distributed z \sim \chi^2(k), where k is the number
    of regressors in the model.
    Parameters
    ==========
    fe : statsmodels.regression.linear_panel.PanelLMWithinResults
        The results obtained by using sm.PanelLM with the
        method='within' option.
    re : statsmodels.regression.linear_panel.PanelLMRandomResults
        The results obtained by using sm.PanelLM with the
        method='swar' option.
    Returns
    =======
    chi2 : float
        The test statistic
    df : int
        The number of degrees of freedom for the distribution of the
        test statistic
    pval : float
        The p-value associated with the null hypothesis
    
    Notes
    =====
    The null hypothesis supports the claim that the random effects
    estimator is "better". If we reject this hypothesis it is the same
    as saying we should be using fixed effects because there are
    systematic differences in the coefficients.
    
    Tests whether random effects estimator can be used, since it is 
    more efficient but can be biased, so if the fixed and random 
    effects estimators are not equal, the fixed effects estimator 
    is the correct/consistent one.
    
    If p<0.05, should use fixed effects
    """
    
    import numpy.linalg as la
    
    # Pull data out
    b = fe.params
    B = re.params
    v_b = fe.cov
    v_B = re.cov

    # NOTE: find df. fe should toss time-invariant variables, but it
    #       doesn't. It does return garbage so we use that to filter
    df = b[np.abs(b) < 1e8].size

    # compute test statistic and associated p-value
    chi2 = np.dot((b - B).T, la.inv(v_b - v_B).dot(b - B))
    pval = stats.chi2.sf(chi2, df)

    return chi2, df, pval

def panel_regression_categorical(primary_only, pri_and_co_ot, independent_string, dependent_string, second_independent_string=0, 
                                 inflation_adjust=True, add_constant=True, take_log=False, take_diff=True, 
                                 add_categorical = True, rank_check = True, add_oil = True, add_time=True, 
                                 payable_basis=True, metal=0):
    '''input strings for independent and dependent variables, 
    returns a dictionary of models. View model details using 
    e.g. models[name].summary, or compare all the models using 
    linearmodels.panel.compare. Performs the Hausman test on each 
    set of fixed effects and random effects models to determine 
    if the random effects model can be used, since it has lower 
    standard deviation but can be biased.
    
    Only the dependent/response variable is log-transformed:
        Exponentiate the coefficient, subtract one from this number, 
        and multiply by 100. This gives the percent increase (or 
        decrease) in the response for every one-unit increase in the 
        independent variable. Example: the coefficient is 0.198. 
        (exp(0.198) – 1) * 100 = 21.9. For every one-unit increase 
        in the independent variable, our dependent variable increases
        by about 22%.
    Only independent/predictor variable(s) is log-transformed:
        Divide the coefficient by 100. This tells us that a 1% increase
        in the independent variable increases (or decreases) the 
        dependent variable by (coefficient/100) units. Example: the
        coefficient is 0.198. 0.198/100 = 0.00198. For every 1% 
        increase in the independent variable, our dependent variable 
        increases by about 0.002. For x percent increase, multiply the 
        coefficient by log(1.x). Example: For every 10% increase in the 
        independent variable, our dependent variable increases by about
        0.198 * log(1.10) = 0.02.
    Both dependent/response variable and independent/predictor 
    variable(s) are log-transformed:
        Interpret the coefficient as the percent increase in the 
        dependent variable for every 1% increase in the independent 
        variable. Example: the coefficient is 0.198. For every 1% 
        increase in the independent variable, our dependent variable 
        increases by about 0.20%. For x percent increase, calculate 
        1.x to the power of the coefficient, subtract 1, and multiply
        by 100. Example: For every 20% increase in the independent 
        variable, our dependent variable increases by about 
        (1.20 0.198 – 1) * 100 = 3.7 percent.
    ^ https://data.library.virginia.edu/interpreting-log-transformations-in-a-linear-model/#:~:text=Rules%20for%20interpretation%201%20Only%20the%20dependent%2Fresponse%20variable,variable%20and%20independent%2Fpredictor%20variable%20%28s%29%20are%20log-transformed.%20
    '''
    
    models = {}
    fe_models = {}
    
    strings = [independent_string,dependent_string]
    independent_strings = [independent_string]
    if independent_string is None or independent_string=='':
        strings = [dependent_string]
        independent_strings = []
    if type(second_independent_string)!=int: 
        strings += [second_independent_string]
    
    if payable_basis:
        df = primary_only.loc[:,idx[strings,:]].stack().sort_index()
    else:
        df = pri_and_co_ot.copy().loc[[i for i in pri_and_co_ot.index if i[1]==i[2]]]
        indy = [i for i in primary_only.columns.get_level_values(0).unique() 
                if i not in df.columns.get_level_values(0).unique()]
        df = pd.concat([df, 
                        primary_only.loc[:,idx[indy,:]]
                       ],axis=1)
        df = df.loc[:,idx[strings,:]].stack().sort_index()
    df = df.loc[idx[:,:,:,:2018],:]
    if add_time:
        df['Year'] = df.index.get_level_values(-1)
#     if already_open_only:
#         ph = df['Commodity Price (USD/t)'].unstack()[1991].notna()
#         ph = ph[ph]
#         indy = ph.index.get_level_values(0)
    for data_folder in ['../generalization-cannot-share/Other data','generalization-cannot-share/Other data']:
            if os.path.exists(data_folder):
                break
    if add_oil:
        oil = pd.read_excel(f'{data_folder}/Oil.xls', index_col=0, header=0).rename(columns={'price':'Oil price'}).resample('AS').mean().loc['19910101':'20180101']
        oil = oil.rename(dict(zip(oil.index,[int(str(i).split('-')[0]) for i in oil.index])))
        df = df.unstack(0).unstack(0).unstack(0).sort_index()
        ph_ind = df.columns.get_level_values(0)[0]
        dfph = df.loc[:,idx[ph_ind,:]]
        oil_df = pd.DataFrame(np.tile(oil,dfph.shape[1]), dfph.index,dfph.columns).rename(columns={ph_ind:'Oil price'})
        df = pd.concat([df,oil_df],axis=1).unstack().unstack(0).dropna()
    
    if inflation_adjust:
        deflator = pd.read_excel(f'{data_folder}/Deflator.xls',index_col=0)['CPI']
        deflator.index = [int(str(i).split('-')[0]) for i in deflator.index]
        deflator /= deflator[2018]
        deflator = 1/deflator
        for j in [i for i in df.columns if 'price' in i.lower() or 'cost' in i.lower()]:
            df.loc[:,j] = df[j].unstack().apply(lambda x: x*deflator.loc[x.index],axis=1).stack()
            print(j,'adjusted for inflation')
        
    if take_log:
        cols = [k for k in df.columns if 'Year' not in k]
        df.loc[:,cols] = np.log(df[cols].astype(float))
        df.rename(columns=dict(zip(cols,[f'log({k})' for k in cols])),inplace=True)
        dependent_string = dependent_string if dependent_string=='Year' else f'log({dependent_string})'
        
    if add_categorical:
        df['Year-Categorical'] = pd.Categorical(df.index.get_level_values(-1))
        
    if add_constant:
        df = sm.add_constant(df)

    independent_strings = [k for k in df.columns if k!=dependent_string]
    
    df.index = pd.MultiIndex.from_tuples([('_'.join([str(i[0]),i[1],i[2]]),i[3]) for i in df.index])
    df_dep = df[dependent_string]
    df_dep.name = dependent_string
    df_indep = df[independent_strings]
    
    # Random vs fixed effects for each individual metal, below does for base metals and precious metals
    # since their units differ. Weirdness happening when performing these aggregate tests on Head Grade and 
    # Total Minesite Cost, where positive correlation appears. Hausman test determines whether we can use
    # random effects since it is more efficient. Hausman function is defined in this notebook. For more info, see:
    # https://www4.eco.unicamp.br/docentes/gori/images/arquivos/PanelData/HO235_Lesson5_RandomHausman.pdf
    metal_names = np.unique([i.split('_')[1] for i in df.index.get_level_values(0)])
    for j in metal_names:
        i = [i for i in df.index.get_level_values(0) if j == i.split('_')[2]]
        try:
            re_cat = RandomEffects(df_dep.loc[idx[i,:]],(df_indep.loc[idx[i,:],:])).fit(cov_type='robust') #choose random if studying effect of time-invariant variables
            models[j] = re_cat
            fe_cat = PanelOLS(df_dep.loc[idx[i,:]],(df_indep.loc[idx[i,:],:])).fit(cov_type='robust') #choose random if studying effect of time-invariant variables
            fe_models[j] = fe_cat
            print(j)
            
        except Exception as e:
            print(j+ ' did not work')
            print((df_indep.loc[idx[i,:],:]).head(15))
#             print((df3.loc[idx[i,:],:]).dtypes)
            print(e)
            if not rank_check:
                df_dep_ph = pd.concat([df_dep.loc[idx[i,:]],df_indep.loc[idx[i,:],:]],axis=1).dropna(how='any')
                df_indep_ph = df_dep_ph.loc[:,df_indep.columns]
                try:
                    df_dep_ph = df_dep_ph.loc[:,df_dep.columns]
                except:
                    df_dep_ph = df_dep_ph.loc[:,df_dep.name]
                print(df_dep_ph.isna().sum().sum(),(df_dep_ph==np.inf).sum().sum(),(df_dep_ph==-np.inf).sum().sum())
                print(df_indep_ph.isna().sum().sum(),(df_indep_ph==np.inf).sum().sum(),(df_indep_ph==-np.inf).sum().sum())
                try:
                    re_cat = RandomEffects(df_dep_ph,df_indep_ph,check_rank=False).fit(cov_type='robust') #choose random if studying effect of time-invariant variables
                    models[j+' rank_check'] = re_cat
                    print('Seems to have worked')
                except:
                    pass
#                     if 'Cobalt' not in j:
#                         return df1_ph, df3_ph,0

    re_cat = RandomEffects(df_dep,df_indep).fit(cov_type='robust') #choose random if studying effect of time-invariant variables
    models['All metals'] = re_cat
    fe_cat = PanelOLS(df_dep,df_indep).fit(cov_type='robust') #choose random if studying effect of time-invariant variables
    fe_models['All metals'] = fe_cat

    j = ['Copper','Lead','Molybdenum','Zinc','Nickel']
    i = [i for i in df_dep.index.get_level_values(0) if i.split('_')[2] in j]
    re_cat = RandomEffects(df_dep.loc[idx[i,:]],(df_indep.loc[idx[i,:],:])).fit(cov_type='robust') #choose random if studying effect of time-invariant variables
    models['Base metals'] = re_cat
    fe_cat = PanelOLS(df_dep.loc[idx[i,:]],(df_indep.loc[idx[i,:],:])).fit(cov_type='robust') #choose random if studying effect of time-invariant variables
    fe_models['Base metals'] = fe_cat

    j = ['Gold','Silver','Platinum']
    i = [i for i in df_dep.index.get_level_values(0) if i.split('_')[2] in j]
    re_cat = RandomEffects(df_dep.loc[idx[i,:]],(df_indep.loc[idx[i,:],:])).fit(cov_type='robust') #choose random if studying effect of time-invariant variables
    models['Precious metals'] = re_cat
    fe_cat = PanelOLS(df_dep.loc[idx[i,:]],(df_indep.loc[idx[i,:],:])).fit(cov_type='robust') #choose random if studying effect of time-invariant variables
    fe_models['Precious metals'] = fe_cat

    result = pd.concat([models[i].params for i in models.keys()],axis=1,keys=models.keys())
    pval = pd.DataFrame()
    fe_result = pd.concat([fe_models[i].params for i in fe_models.keys()],axis=1,keys=fe_models.keys())
    fe_pval = pd.DataFrame()
    best_models = {}
    for i in models.keys():
        try:
            x = models[i].pvalues
            x.name = i
            fe_x = fe_models[i].pvalues
            fe_x.name = i
        except:
            print(i)
            x = models[i].params
            x.loc[x.index] = 1
            x.name = i
        pval = pd.concat([pval,x],axis=1)
        fe_pval = pd.concat([fe_pval,fe_x],axis=1)
        try:
            hausman_p = hausman(fe_models[i], models[i])[2]
            if hausman_p < 0.05:
                print(i, 'Hausman says use fixed effects')
                best_models[i] = fe_models[i]
            else:
                print(i, 'Hausman ok to use random effects')
                best_models[i] = models[i]
        except Exception as e:
            print(i, 'Hausman failed', e)
            best_models[i] = fe_models[i]
#     pval = pd.concat([models[i].pvalues for i in models.keys()],axis=1,keys=models.keys())
#     return result,pval
    best_result = pd.concat([best_models[i].params for i in best_models.keys()],axis=1,keys=best_models.keys())
    best_pvals = pd.concat([best_models[i].pvalues for i in best_models.keys()],axis=1,keys=best_models.keys())
    
    results_cat_nan = result.copy()
    results_cat_nan[pval>0.1] = np.nan
    results_cat_nan.rename(dict(zip([i for i in results_cat_nan.index if 'Year-Categorical' in str(i)],[int(i.split('.')[1]) for i in results_cat_nan.index if 'Year-Categorical' in str(i)])),inplace=True)
    fe_results_cat_nan = fe_result.copy()
    fe_results_cat_nan[fe_pval>0.1] = np.nan
    fe_results_cat_nan.rename(dict(zip([i for i in fe_results_cat_nan.index if 'Year-Categorical' in str(i)],[int(i.split('.')[1]) for i in fe_results_cat_nan.index if 'Year-Categorical' in str(i)])),inplace=True)

    for i in results_cat_nan.columns:
        results_cat_nan.loc['f stat p-value',i] = float(str(models[i].f_statistic).split('P-value: ')[1].split('\n')[0])
        fe_results_cat_nan.loc['f stat p-value',i] = float(str(fe_models[i].f_statistic).split('P-value: ')[1].split('\n')[0])
    
    if take_log and add_time:
        results_cat_nan.loc['Year interpretation'] = np.exp(results_cat_nan.loc['Year'])-1
        fe_results_cat_nan.loc['Year interpretation'] = np.exp(fe_results_cat_nan.loc['Year'])-1
        best_result.loc['Year interpretation'] = np.exp(best_result.loc['Year'])-1
    for i in best_models.keys():
        best_result.loc['AIC',i] = AIC(best_models[i])
    return models, results_cat_nan, pval, fe_models, fe_results_cat_nan, fe_pval, best_models, best_result, best_pvals

def add_axis_labels(fig, option=None, xloc=-0.1, yloc=1.03):
    chars = character_list[10:36]
    ax = fig.axes
    for label,a in zip(chars, ax):
        xticks = len(a.get_xticks())
        if option=='price_middle_column':
            xloc = -0.21 if a in ax[1::3] else -0.17
            yloc = 1.02
        elif option in ['subset1','all']:
            xloc = -0.035 if xticks>10 else -0.07 if xticks>7 else -0.08
            yloc = 1.03
        elif option=='subset2':
            xloc = -0.05 if xticks>6 else -0.08 if xticks>4 else -0.13 if xticks>3 else -0.18
            yloc = 1.03
        
        a.text(xloc,yloc,label+')', transform=a.transAxes)
    
def lin_predict_to_hist(lin_predict):
    hist_data = pd.DataFrame()
    for comm in lin_predict.Commodity.unique():
        n_hist, bins_hist = np.histogram(lin_predict.loc[idx[:,comm,:],'Actual'].dropna(),bins=50)
        n_sim, bins_sim = np.histogram(lin_predict.loc[idx[:,comm,:],'Predicted'].dropna(),bins=50)
        hist_data_i = pd.concat([
            pd.concat([pd.Series(n_hist),pd.Series(n_sim)],axis=1,keys=['Actual','Predicted']),
            pd.concat([pd.Series(bins_hist),pd.Series(bins_sim)],axis=1,keys=['Actual','Predicted']),
            ],keys=['Values','Edges'])
        hist_data_i = pd.concat([hist_data_i],keys=[comm])
        hist_data = pd.concat([hist_data,hist_data_i])
    return hist_data

def plotting_grades_over_time(primary_only, axes=None):
    ore_treated_d = discont_to_cont(primary_only['Ore Treated (kt)'].droplevel(2))
    head_grade_d = discont_to_cont(primary_only['Head Grade (%)'].droplevel(2))
    head_grade_d = head_grade_d.stack()
    ore_treated_d = ore_treated_d.stack()
    ind = np.intersect1d(ore_treated_d.index, head_grade_d.index)
    head_grade_d, ore_treated_d = head_grade_d.loc[ind], ore_treated_d.loc[ind]
    hgd = head_grade_d.unstack()
    otd = ore_treated_d.unstack()
    otd = otd.astype(float)
    hgd = hgd.astype(float)

    def divide_by_first(ph):
        first = ph.notna().idxmax()
        return ph/ph[first]

    def first_greater_than_last(ph):
        first = ph.notna().idxmax()
        last = ph.iloc[::-1].notna().idxmax()
        return ph[first]>ph[last]

    grade_ph = hgd.loc[(hgd.notna().sum(axis=1)>2)].T.sort_index().T
    grade_ph = grade_ph.loc[idx[:,:,0],:]
    grade_ph = grade_ph.loc[:,:2018]
    grade_ph = grade_ph.apply(lambda x: divide_by_first(x), axis=1)
    first_greater = grade_ph.apply(lambda x: first_greater_than_last(x),axis=1)
    print('Fraction of mines with decreasing ore grades: {:.3f}'.format(first_greater.sum()/len(first_greater)))
    deyeared = pd.DataFrame()
    for i in grade_ph.index:
        ph = grade_ph.loc[i]
        first = ph.notna().idxmax()
        ph = ph.loc[first:].reset_index(drop=True)
        deyeared = pd.concat([deyeared, ph], axis=1)
    deyeared = deyeared.T.reset_index(drop=True).T

    if axes is None:
        fig,axes = easy_subplots(2)
    elif len(axes) != 2:
        axes = axes[-2:]
    ax=axes[0]
    deyeared = deyeared.T.reset_index(drop=True).T
    deyeared = deyeared.dropna(how='all', axis=1)
    deyeared.plot(legend=False, linewidth=1, alpha=0.3, logy=True, ax=ax, )
    deyeared.mean(axis=1).plot(color='k', linestyle=':', logy=True, ax=ax).grid(axis='x')
    ax.set(title=f'Ore grade over time, n={deyeared.shape[1]},'+'\nall mines with >2 years\ncontinuous data', 
        xlabel='Year (indexed to first operating year=0)',
        ylabel='Relative grade (grade in first year=1)')

    ax=axes[1]
    nd = deyeared.loc[:,deyeared.apply(lambda x: (x.dropna()<=1).all())]
    nd = deyeared.loc[:,deyeared.apply(lambda x: (x.dropna().iloc[-1])<1)]
    deyeared.apply(lambda x: (x.dropna()<=1).all())
    nd.plot(legend=False, linewidth=1, alpha=0.3, logy=True, ax=ax)
    nd.mean(axis=1).plot(color='k', linestyle=':', logy=True, ax=ax).grid(axis='x')
    ax.set(title=f'Ore grade over time, n={nd.shape[1]},'+'\nwhere grade in last\nyear < grade in first year', 
        xlabel='Year (indexed to first operating year=0)',
        ylabel='Relative grade (grade in first year=1)')
    return deyeared

def get_min_pvals_across_tests(many, n_best=25):
    """
    Used by figures S22-S24
    """
    tests = ['uniform-min','uniform-min-normality','uniform-ttest','uniform-shapiro','uniform-lilliefors-both','uniform-dagostino','uniform-anderson-both']
    nice_test_names = {'uniform-min':'All tests, min.\np-value','uniform-min-normality':'All normality\ntests','uniform-shapiro':'Shapiro-Wilk\ntest',
                    'uniform-lilliefors-both':'Lilliefors-corr.\nK-S test','uniform-dagostino':'D\'Ag.-Pea.\ntest',
                    'uniform-anderson-both':'And.-Dar.\ntest', 'uniform-ttest':'One-sample\nT-test'}
    nice_names_list = [nice_test_names[i] for i in tests]
    pvals_list = []
    means_list = []
    for test in tests:
        table,means,pvals = make_parameter_mean_std_table(many, n_best, stars=test, 
                                                        value_in_parentheses=None, rand_size=100,
                                                        how_many_tests=100)
        pvals_list += [pvals]
        means_list += [means]
    min_pvals = pd.concat(pvals_list,keys=range(len(pvals_list))).groupby(level=1).min()
    return min_pvals




def figures_2_and_3(many_sg, show=False):
    fig, data = plot_best_fits(many_sg, show_all_lines=False, plot_actual_price=True, commodities=['Cu','Ni','Pb','Zn','Au']);
    fig.savefig(f'{many_sg.folder_path}/figures/figure_2_tuning_results1.pdf')
    fig.savefig(f'{many_sg.folder_path}/figures/figure_2_tuning_results1.png')
    data.to_csv(f'{many_sg.folder_path}/figures/figure_2_tuning_results1.csv')
    if show: 
        plt.show()
    plt.close()
    fig, data = plot_best_fits(many_sg, show_all_lines=False, plot_actual_price=True, commodities=['Ag','Sn','Al','Steel']);
    fig.savefig(f'{many_sg.folder_path}/figures/figure_3_tuning_results2.pdf')
    fig.savefig(f'{many_sg.folder_path}/figures/figure_3_tuning_results2.png')
    data.to_csv(f'{many_sg.folder_path}/figures/figure_3_tuning_results2.csv')
    if show: 
        plt.show()
    plt.close()

def figure_4(many_sg, show=False):
    

    rand_size=25
    table, means, pvals, alt_means, fig = plot_colorful_table2(many_sg, stars='uniform-min', dpi=250, 
                                            value_in_parentheses='None',#'standard error'
                                            rand_size=rand_size, how_many_tests=100);
    print('number of statistically-significant commodities per parameter:')
    print((pvals<=0.1).sum().sort_values())
    many_sg.colorful_means = means.copy()
    # if rand_size is None:
    #     fig.axes[0].set_title('single test, size=10,000')
    # else:
    #     fig.axes[0].set(title=f'pi0, size={rand_size}')
    # fig.savefig('figures/compare_elasticities_table.pdf')
    pd.concat([
        table, means, pvals
    ], axis=1, keys=['Table reproduction','Mean parameter values','P-values']).\
    to_csv(f'{many_sg.folder_path}/figures/figure_4_compare_elas_table.csv')
    fig.savefig(f'{many_sg.folder_path}/figures/figure_4_compare_elas_table.pdf')
    fig.savefig(f'{many_sg.folder_path}/figures/figure_4_compare_elas_table.png')
    if show:
        plt.show()
    plt.close()

def figures_5_and_s31(many_sg, include_n=True):
    note_columns = ['Degrees of freedom','Years','Notes','Paper notes']

    def cleanup_whole(source):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for i in np.arange(0,source.shape[0]):
                for j in np.arange(0,source.shape[1]):
                    col = source.columns[j]
                    if col in note_columns:
                        continue
                    ix = source[col].iloc[i]
                    if type(ix)==str:
                        ind = source.index[i]
                        if ' ' in ix:
                            source.loc[(ind[0],ind[1]+', a',ind[2]),col]=float(ix.split(' ')[0])
                            try:
                                source.loc[(ind[0],ind[1]+', b',ind[2]),col]=float(ix.split(' ')[-1])
                            except:
                                print(ix)
                                raise ValueError
                        elif '-' in ix:
                            source.loc[(ind[0],ind[1]+', lower bound',ind[2]),col]=float(ix.split('-')[0])
                            source.loc[(ind[0],ind[1]+', upper bound',ind[2]),col]=float(ix.split('-')[-1])
                        source.iloc[i,j]=np.nan
            source = source.dropna(how='all')
            return source               

    def load_sources(many):
        sources = pd.read_excel('Sources for elasticities.xlsx',index_col=[0,1,2,3])
        if 'Li' in sources.columns:
            sources.drop(columns=['Li'],inplace=True)
        sources = sources.dropna(how='all')
        sources.rename(dict(zip(sources.index.get_level_values(0).unique(),
                                [i.replace('supply-demand imbalance','SD') for i in sources.index.get_level_values(0).unique()])),level=0,inplace=True)
        
        sources_to_rmse = {
            'Scrap supply elasticity to secondary price':'Collection elasticity to scrap price',
            'Scrap demand elasticity to scrap spread':'Direct melt fraction elasticity to scrap spread',
            'Price elasticity to supply-demand imbalance':'Refined price elasticity to SD',
            'Ore treated elasticity to total cash margin':'Mine CU elasticity to TCM',
            'Primary supply elasticity to price':'Mine CU elasticity to TCM',
            'Primary demand elasticity to price':'Demand elasticity to price',
            'Ore grade decline per year':'Ore grade elasticity to COT distribution mean',
                        }
        sources.rename(sources_to_rmse,level=0,inplace=True)
    #     sources = sources.loc[idx[:,:,['N','From Dahl'],:],:]
        sources = sources.droplevel(2)
        sources = cleanup_whole(sources)
        many.sources_info_orig = sources.copy()
        sources_ph = sources.loc[:,'Steel':'Zn']
        new_ind = []
        for e,i in enumerate(sources_ph.index):
            if 'Demand' in i[0]:
                new_ind += [(i[0].replace('Demand','Intensity'), i[1]+' - from demand', i[2])]
                if 'GDP' in i[0]:
                    sources_ph.iloc[e] = sources_ph.iloc[e] - 1
            elif i[0]=='Scrap price elasticity to primary price':
                new_ind += [('Scrap spread elasticity to price', i[1]+' - from scrap price', i[2])]
                sources_ph.iloc[e] = 1-sources_ph.iloc[e]
            else:
                new_ind += [i]
        sources_ph.index = pd.MultiIndex.from_tuples(new_ind)
        sources.index = pd.MultiIndex.from_tuples(new_ind)
        sources = pd.concat([sources_ph, sources[note_columns]], axis=1)
        many.sources_info = sources.copy()
        sources = sources.loc[:,'Steel':'Zn'].dropna(how='all')
        many.sources_methods = sources.copy()
        many.sources = sources.copy().droplevel(2)

    def get_distribution_differences(sources, this_study):
        """
        Returns an n by 2 dataframe with a 2-level index 
        (parameter name, commodity) and columns being the
        pvalues from the Kruskal-Wallis H-test and 
        Kolmogorov-Smirnov tests.
        """
        dist_test_pvals = pd.DataFrame()
        for i in sources.index.get_level_values(0).unique():
            dist_test_ph = pd.DataFrame()
            for c in sources.columns:
                if c in sources.columns:
                    sources_vals = sources.loc[i,c].dropna()
                else: 
                    sources_vals = pd.DataFrame()
                if c in this_study.columns:
                    model_vals = this_study.sort_index().loc[i,c].dropna()
                else: 
                    model_vals = pd.DataFrame()

                if model_vals.shape[0]>0 and sources_vals.shape[0]>0:
                    dist_test_ph.loc[c,'KW test pval'] = stats.kruskal(model_vals, sources_vals)[1]
                    dist_test_ph.loc[c,'KS test pval'] = stats.kstest(model_vals, sources_vals)[1]
            dist_test_ph = pd.concat([dist_test_ph],keys=[i])
            dist_test_pvals = pd.concat([dist_test_pvals,dist_test_ph])
        return dist_test_pvals
        
    def get_sources_this_study(many):
        load_sources(many)
        sources = many.sources.copy()
        many.rmse_df_nice = many.rmse_df_sorted.iloc[:,:25].rename(
            make_parameter_names_nice(many.rmse_df_sorted.index.get_level_values(1)), level=1
        )
        this_study = many.rmse_df_nice.stack().unstack(0)
        
        
        this_study['Source'] = this_study.index.get_level_values(1)
        sources['Source'] = sources.index.get_level_values(1)
        ind = np.intersect1d(sources.index.get_level_values(0).unique(), 
                            this_study.index.get_level_values(0).unique())
        sources = sources.loc[ind]
        this_study_full = this_study.copy()
        this_study = this_study.loc[ind]
        this_study = this_study.reset_index().set_index(['level_0','level_1','Source'])
        this_study_full = this_study_full.reset_index().set_index(['level_0','level_1','Source'])
        sources = sources.reset_index().set_index(['level_0','level_1','Source'])
        this_study = this_study.rename(
            columns=dict(zip(this_study.columns,[i.capitalize() for i in this_study.columns])))
        this_study_full = this_study_full.rename(
            columns=dict(zip(this_study_full.columns,[i.capitalize() for i in this_study_full.columns])))
        this_study = this_study.rename(
            dict(zip(np.arange(0,26),[f'This study {i}' for i in np.arange(0,26)]),level=1))
        this_study_full = this_study_full.rename(
            dict(zip(np.arange(0,26),[f'This study {i}' for i in np.arange(0,26)]),level=1))
        many.this_study_full = this_study_full.copy()
        sources = sources.rename(dict(zip(sources.index.get_level_values(1),
                                ['Literature' if '- demand' not in i else 'Literature' for i in
                                sources.index.get_level_values(1)
                                ]
                            )),level=1)
        sources_both = pd.concat([
            this_study,
            sources]).dropna(how='all',axis=1)
        sources_plt = sources_both.stack().reset_index().rename(columns={'level_0':'Parameter',
                                                                'level_1':'General source',
                                                                'level_3':'Commodity',
                                                                0: 'Value'
                                                            })
        sources_plt.loc[['This study' in i for i in sources_plt['General source']],'General source'] = 'This study'
        sources_plt['Commodity, source'] = [', '.join([i,j.lower()]) for i,j in zip(sources_plt['Commodity'],
                                                                        sources_plt['General source'])]
        sources_plt = sources_plt.astype({'Value':float})
        return sources, this_study, sources_plt

    def update_ore_grade_annual_to_COT(many, sources_plt):
        # setting per year changes in ore grade to be in terms of cumu. ore treated
        # done by multiplying per year value by the average percent change in cumu. OT per year
        intermediate = sources_plt.loc[
            (sources_plt['Parameter']=='Ore grade elasticity to COT distribution mean') &
            (sources_plt['General source']!='This study') &
            (sources_plt['Source']!='This study, c')
        ]
        initial_ind = intermediate.index
        intermediate = intermediate.set_index(['Source','Commodity'])

        mean_pct_change = many.sources.loc['Mean percent change per year in cumulative ore treated']
        intermediate.loc[:,'New value'] = intermediate.apply(lambda x: mean_pct_change[x.name[1]]*x['Value']/100,axis=1)
        intermediate['Value'] = intermediate['New value']
        intermediate = intermediate.drop(columns='New value')
        intermediate = intermediate.reset_index()
        intermediate.index = initial_ind

        sources_plt.loc[
            (sources_plt['Parameter']=='Ore grade elasticity to COT distribution mean') &
            (sources_plt['General source']!='This study') &
            (sources_plt['Source']!='This study, c')
        ] = intermediate

        sources_plt.loc[
            (sources_plt['Parameter']=='Ore grade elasticity to COT distribution mean') &
            (sources_plt['General source']!='This study') &
            (sources_plt['Source']!='This study, c')
        ]
        return sources_plt
        
    def update_sources_sub_parameters(sources_sub):
        sources_sub.loc[sources_sub.Parameter=='Mine cost change per year','Value']/=10
        sources_sub.loc[sources_sub.Parameter==
                        'Mine cost change per year','Parameter'
                    ] = r'$\frac{1}{10} x$ Mine cost change per year'
        sources_sub.loc[sources_sub.Parameter=='Intensity elasticity to time','Value']*=10
        sources_sub.loc[sources_sub.Parameter==
                        'Intensity elasticity to time','Parameter'
                    ] = '10 x Intensity elasticity to time'
        sources_sub.loc[sources_sub.Parameter=='Ore grade elasticity to COT distribution mean','Value']*=2
        sources_sub.loc[sources_sub.Parameter==
                        'Ore grade elasticity to COT distribution mean','Parameter'
                    ] = '2 x Ore grade elasticity to COT distribution mean'
    #         sources_sub.loc[sources_sub.Parameter=='Ore grade elasticity to COT distribution mean','Value']/=-5
    #         sources_sub.loc[sources_sub.Parameter==
    #                         'Ore grade elasticity to COT distribution mean','Parameter'
    #                        ] = r'$\frac{1}{5} x$ Ore grade elasticity to COT'

        param_list = []
        for i in sources_sub['Parameter'].unique():
            if len(sources_sub.loc[sources_sub['Parameter']==i]['General source'].unique())>1:
                param_list += [i]
        sources_sub = sources_sub.loc[sources_sub['Parameter'].apply(lambda x: x in param_list)]
        return sources_sub, param_list

    def update_sources_sub_stars(comm, sources_sub, dist_test_pvals, param_list, include_stars,
                                stars_overlapping, stars_overlapping90, stars_not_overlapping, include_n=True):
        if include_stars:
            for x in param_list:
                dist_param = [i for i in dist_test_pvals.index.get_level_values(0).unique()
                            if i in x.replace('\n',' ')][0]
                star = '***' if (dist_test_pvals.loc[dist_param].loc[comm]<0.001).any() else '**' if \
                                (dist_test_pvals.loc[dist_param].loc[comm]<0.01).any() else '*' if \
                                (dist_test_pvals.loc[dist_param].loc[comm]<0.05).any() else '.' if \
                                (dist_test_pvals.loc[dist_param].loc[comm]<0.1).any() else ' '
                if (dist_test_pvals.loc[dist_param].loc[comm]<0.05).any():
                    sub_sub = sources_sub.copy().loc[sources_sub['Parameter']==x]
                    lit_values = sub_sub.loc[sub_sub['General source']=='Literature','Value']
                    our_values = sub_sub.loc[sub_sub['General source']=='This study','Value']
                    lit_values75 = [lit_values.quantile(0.25), lit_values.quantile(0.75)]
                    our_values75 = [our_values.quantile(0.25), our_values.quantile(0.75)]
                    if (np.min(lit_values75)<np.max(our_values75) and np.min(lit_values75)>np.min(our_values75)):
                        stars_overlapping += [(comm,dist_param)]
                    elif (np.min(our_values75)<np.max(lit_values75) and np.min(our_values75)>np.min(lit_values75)):
                        stars_overlapping += [(comm,dist_param)]
                    else:
                        stars_not_overlapping += [(comm,dist_param)]
                    lit_values90 = [lit_values.quantile(0.1), lit_values.quantile(0.9)]
                    our_values90 = [our_values.quantile(0.1), our_values.quantile(0.9)]
                    if (np.min(lit_values90)<np.max(our_values90) and np.min(lit_values90)>np.min(our_values90)):
                        stars_overlapping90 += [(comm,dist_param)]
                    elif (np.min(our_values90)<np.max(lit_values90) and np.min(our_values90)>np.min(lit_values90)):
                        stars_overlapping90 += [(comm,dist_param)]  
                star = f' ({star})'
                if include_n:
                    n_lit_sources = sources_sub.loc[(sources_sub['Parameter']==x)&(sources_sub['General source']!='This study')].shape[0]
                    star = f'{star}, n={n_lit_sources}'
                sources_sub.loc[sources_sub['Parameter']==x,'Parameter'] = x+star
            param_list = sources_sub['Parameter'].unique()
        return sources_sub, param_list, stars_overlapping, stars_overlapping90, stars_not_overlapping
                
    def update_sources_sub_linelength(sources_sub, param_list):
        for x in [i for i in param_list if len(i)>len('Mine cost elasticity to ')]:
            split = x.split(' ')
            num_words = 7 if 'COT dist' in x else 4 if 'Direct melt fraction' in x else 3 if ' x$ ' not in x else 5
            sources_sub.loc[sources_sub.Parameter==
                        x,'Parameter'
                    ] = ' '.join(split[:num_words])+'\n'+' '.join(split[num_words:])
        return sources_sub

    def initialize_big_plot(commodity_subset):
        if commodity_subset == 'all':
            shapex, shapey = 20,3
            cu_span = 15
            al_span = 8
            ni_span = 7
            pb_span = 6
            au_span = 4
            ag_span = 3
            fig, ax = plt.subplot_mosaic('A'*cu_span + 'B'*(shapex-cu_span)+';'+
                                        'C'*al_span + 'D'*ni_span + 'E'*(shapex-al_span-ni_span)+';'+
                                        'F'*pb_span + 'G'*au_span + 'H'*ag_span + 'I'*(shapex-pb_span-au_span-ag_span),
                                        figsize=(30,28)
                                        )
            axes = {}
            axes['Cu'] = ax['A']
            axes['Steel'] = ax['B']
            axes['Al'] = ax['C']
            axes['Ni'] = ax['D']
            axes['Sn'] = ax['E']
            axes['Pb'] = ax['F']
            axes['Au'] = ax['G']
            axes['Ag'] = ax['H']
            axes['Zn'] = ax['I']
        elif commodity_subset=='subset1':
            # ['Cu','Al','Ni','Pb','Zn']
            shapex, shapey = 20,3
            al_span = 11
            pb_span = 10
            fig, ax = plt.subplot_mosaic('A'*shapex + ';'+
                                        'C'*al_span + 'D'*(shapex-al_span) +';'+
                                        'F'*pb_span + 'I'*(shapex-pb_span),
                                        figsize=(26,30)
                                        )
            axes = {}
            axes['Cu'] = ax['A']
            axes['Al'] = ax['C']
            axes['Ni'] = ax['D']
            axes['Pb'] = ax['F']
            axes['Zn'] = ax['I']
        elif commodity_subset=='subset2':
            shapex, shapey = 20,3
            steel_span = 12
            sn_span = 14
            fig, ax = plt.subplot_mosaic('A'*steel_span + 'B'*(shapex-steel_span)+';'+
                                        'C'*sn_span + 'E'*(shapex-sn_span),
                                        figsize=(22,18)
                                        )
            axes = {}
            axes['Steel'] = ax['A']
            axes['Au'] = ax['B']
            axes['Sn'] = ax['C']
            axes['Ag'] = ax['E']
            
        return fig, axes
                
    def plot_violin_ax(sources_sub, ax, comm, commodity_subset='all'):
        sns.violinplot(ax=ax, data=sources_sub, x='Parameter', y='Value', hue='General source',
                    linewidth=4, cut=0, width=0.95, inner='box', palette=['#666666','#66a61e'], 
                    )
        fontsize=26 if commodity_subset=='subset1' else 22
        if fontsize is None:
            ax.tick_params(axis='x',rotation=90)
            ax.tick_params(axis='y')
            ax.set_title(comm.replace('Steel','Fe'))
            ax.set_ylabel('Value')
            ax.legend(title=None, loc='lower right')
        else:
            ax.tick_params(axis='x',rotation=90,labelsize=fontsize)
            ax.tick_params(axis='y',labelsize=fontsize)
            ax.set_title(comm.replace('Steel','Fe'),fontsize=fontsize*1.1)
            ax.set_ylabel('Value',fontsize=fontsize)
            ax.set_xlabel('Parameter',fontsize=fontsize)
            ax.legend(title=None, fontsize=fontsize, loc='lower right')
        if comm not in ['Copper','Zinc']:
            ax.get_legend().remove()
        return ax
                
    def print_paper_info(many, sources_plt, dist_test_pvals,
                        stars_overlapping, stars_overlapping90, stars_not_overlapping):
        print('Number of unique publications (excluding our own work):', 
            np.unique([i.split(',')[0].split(' -')[0].split(')b')[0].split(')a')[0] 
                for i in many.sources.index.get_level_values(1) if 'This study' not in i]).shape[0])
        print('Number of unique regressions (excluding our own work):', 
            np.unique([i
                for i in many.sources.index.get_level_values(1) if 'This study' not in i]).shape[0])
        print('Number of unique methods (excluding our own work and treating long- and short-run as same):',
            np.unique([i[2].split(',')[0].split(' short')[0].split(
                ' long')[0].replace('OLS','Ordinary least squares')
            for i in many.sources_methods.index if 'This study' not in i[1]]).shape[0]
            )
        print('Number of unique values (excluding our own work):',
            many_sg.sources.stack().loc[idx[:,
                    [i for i in many.sources.index.get_level_values(1) if 'This study' not in i],
                    :]].shape[0]
            )
        print('Number of unique parameters:', sources_plt['Parameter'].unique().shape[0])
        print('Percent of distributions significantly different by either test (KW or KS), 95% confidence:',
            round(100*(dist_test_pvals<0.05).any(axis=1).sum()/dist_test_pvals.shape[0],3),
            f'({(dist_test_pvals<0.05).any(axis=1).sum()}/{dist_test_pvals.shape[0]})'
            )
        print('Percent of distributions significantly different by either test (KW or KS), 99% confidence:',
            round(100*(dist_test_pvals<0.01).any(axis=1).sum()/dist_test_pvals.shape[0],3),
            f'({(dist_test_pvals<0.01).any(axis=1).sum()}/{dist_test_pvals.shape[0]})'
            )
        print('Number of significantly different (95%) dists that overlap their 25-75% percentiles (their boxes):',
            f'{len(stars_overlapping)}/{len(stars_overlapping)+len(stars_not_overlapping)}'
            )
        print('Number of significantly different (95%) dists that overlap their 10-90% percentiles:',
            f'{len(stars_overlapping90)}/{len(stars_overlapping)+len(stars_not_overlapping)}'
            )

    def plot_comparative_violins(many, dpi=100, commodity_subset='all', show=False):
        with warnings.catch_warnings():
            include_stars = True
            warnings.simplefilter('error')

            sources, this_study, sources_plt = get_sources_this_study(many)

            sources_plt = update_ore_grade_annual_to_COT(many, sources_plt)

            dist_test_pvals = get_distribution_differences(sources, this_study)

            fig, axes = initialize_big_plot(commodity_subset)

            stars_overlapping = []
            stars_overlapping90 = []
            stars_not_overlapping = []
            sources_all = pd.DataFrame()
            subset1 = ['Cu','Al','Ni','Pb','Zn']
            if commodity_subset=='all':
                commodities = sources_plt.Commodity.unique()
            elif commodity_subset=='subset1':
                commodities = subset1
                init_plot2(fontsize=24)
            elif commodity_subset=='subset2':
                commodities = [i for i in sources_plt.Commodity.unique() if i not in subset1]
            for comm in sources_plt.Commodity.unique():
                sources_sub = sources_plt.loc[sources_plt['Commodity']==comm]
                sources_sub, param_list = \
                    update_sources_sub_parameters(sources_sub)

                sources_sub, param_list, stars_overlapping, stars_overlapping90, stars_not_overlapping = \
                    update_sources_sub_stars(comm, sources_sub, dist_test_pvals,
                                            param_list, include_stars,
                                            stars_overlapping, stars_overlapping90, stars_not_overlapping, include_n=include_n)

                sources_sub = \
                    update_sources_sub_linelength(sources_sub, param_list)
                sources_all = pd.concat([sources_all, sources_sub])
                if comm in commodities:
                    ax = axes[comm]
                    ax = plot_violin_ax(sources_sub, ax, comm, commodity_subset)
                    if comm=='Ag':
                        ax.set(yticks=[-1,-0.5,0,0.5])
                # if comm == sources_plt.Commodity.unique()[0]:
                #     ax.legend()

            fig.tight_layout(pad=0.3)
            add_axis_labels(fig, commodity_subset)
            fig.set_dpi(dpi)
            fig.axes[0].legend()
            if show:
                plt.show()
            sources_all = sources_all.reset_index(drop=True)
            sources.index = sources.index.set_names(['Parameter','General source','Source'])
            sources = sources.rename(columns={'Steel':'Fe'})
            many.sources_info_orig.to_csv(f'{many.folder_path}/tables/table_s28_literature_parameter_database_original.csv')
            many.sources_info.to_csv(f'{many.folder_path}/tables/table_s29_literature_parameter_database_updated.csv')
            many.rmse_df_sorted.rename(
                make_parameter_names_nice(many.rmse_df_sorted.index.get_level_values(1)), level=1
                ).to_csv(f'{many.folder_path}/tables/table_s30_parameter_results_this_study.csv')
            sources_all.to_csv(f'{many.folder_path}/figures/figure_5_literature_comparison.csv')
            sources_all.to_csv(f'{many.folder_path}/figures/figure_s31_literature_comparison.csv')
            if commodity_subset=='all':
                fig.savefig(f'{many.folder_path}/figures/figure_none_literature_comparison.pdf')
                fig.savefig(f'{many.folder_path}/figures/figure_none_literature_comparison.png')
            elif commodity_subset=='subset1':
                fig.savefig(f'{many.folder_path}/figures/figure_5_literature_comparison.pdf')
                fig.savefig(f'{many.folder_path}/figures/figure_5_literature_comparison.png')
            elif commodity_subset=='subset2':
                fig.savefig(f'{many.folder_path}/figures/figure_s31_literature_comparison.pdf')
                fig.savefig(f'{many.folder_path}/figures/figure_s31_literature_comparison.png')
            plt.close()
            print_paper_info(many, sources_plt, dist_test_pvals,
                            stars_overlapping, stars_overlapping90, stars_not_overlapping)
            init_plot2()
            return sources, sources_all, this_study, sources_plt, fig

    sources, sources_all, this_study, sources_plt, fig = plot_comparative_violins(many_sg, dpi=200, commodity_subset='subset1')
    sources, sources_all, this_study, sources_plt, fig = plot_comparative_violins(many_sg, dpi=200, commodity_subset='subset2')

def figure_s4(many_sg):
    split_on_china = True
    pct_change = False
    include_global = True
    norm = True
    norm_year = 2019
    end_year = 2019
    dpi = 250

    xl = 'input_files/static/Demand prediction data-copper.xlsx'
    volumes = pd.read_excel(xl, sheet_name='All sectors', header=[0,1], 
                index_col=0).sort_index().sort_index(axis=1).stack(0).unstack()

    def get_jewelry_bar_coin():
        gold_rolling_window=5
        static_data_folder = 'input_files/static'
        gold_vols = pd.read_excel(f'{static_data_folder}/Gold demand volume indicators.xlsx',sheet_name='Volume drivers',index_col=0).loc[2001:]
        gold_vols1 = gold_vols.loc[2001:2019].rolling(gold_rolling_window,min_periods=1,center=True).mean()
        gold_vols1 = pd.concat([gold_vols1,gold_vols.rolling(5,min_periods=1,center=True).mean().loc[2020:]])
        gold_vols2 = gold_vols.rolling(gold_rolling_window+2,min_periods=1,center=True).mean()
        global_cash_reserves = volumes.loc[:,idx[:,'Industrial']].apply(lambda x: x/x.sum(), axis=1).apply(lambda x: x*gold_vols2['Global cash reserves (USD$2021)'])#['US circulating coin production (million coins)'])
        diamond_demand = volumes.loc[:,idx[:,'Transport']].apply(lambda x: x/x.sum(), axis=1).apply(lambda x: x*gold_vols1['Diamond demand ($B)'])
        diamond_demand = diamond_demand.rename(columns={'Transport':'Jewelry'},level=1)
        global_cash_reserves = global_cash_reserves.rename(columns={'Industrial':'Bar and coin'},level=1)
        return diamond_demand, global_cash_reserves

    jewelry, bar_and_coin = get_jewelry_bar_coin()
    volumes = pd.concat([volumes, jewelry, bar_and_coin], axis=1)
    sector_units = {
        'Transport':'Vehicle sales (million vehicles/year)',
        'Construction':'Value added in construction (2010 USD)',
        'Electrical':'Total grid power demand (GW)',
        'Industrial':'Value added in manufacturing (2010 USD)',
        'Other':'Proxy: GDP (2010 USD)',
        'Jewelry':'Diamond demand (billion USD)',
        'Bar and coin':'Global cash reserves (2021 USD)'
    }
    sectors = volumes.columns.get_level_values(1).unique()
    non_china = ['EU', 'Japan', 'NAM', 'ROW']
    if split_on_china:
        ph = pd.concat([volumes.loc[:,idx[non_china,:]].groupby(axis=1,level=1).sum()],keys=['RoW'],axis=1)
        volumes = pd.concat([volumes, ph],axis=1).drop(non_china,axis=1,level=0)
    if include_global:
        ph = pd.concat([volumes.groupby(axis=1,level=1).sum()],keys=['Global'],axis=1)
        volumes = pd.concat([volumes, ph], axis=1)
        volumes = volumes.replace(0,np.nan)
    if pct_change:
        volumes = volumes.pct_change().dropna()
    if norm:
        volumes = volumes.apply(lambda x: x/volumes.loc[norm_year], axis=1)
    volumes = volumes.loc[:end_year]
    fig,ax = easy_subplots(sectors, 4)
    for sector, a in zip(sectors,ax):
        volumes.loc[:,idx[:,sector]].droplevel(1,axis=1).plot(
            ax=a, title=sector, xlabel='Year', ylabel=sector_units[sector]
        )
        a.legend(title=None)

    which = 'Percent per year' if pct_change else 'Actual'
    cols_to_use = {'Percent per year':'A:F', 'Actual':'H:M'}
    if split_on_china: 
        cols_to_use['Actual'] = 'W:AB'
    cols = cols_to_use[which]
    gdp = pd.read_excel(xl, sheet_name='GDP growth', header=[0], 
                index_col=0, usecols=cols).sort_index().sort_index(axis=1).dropna()
    if include_global:
        ph = pd.DataFrame(gdp.sum(axis=1))
        gdp = pd.concat([gdp, ph.rename(columns={ph.columns[0]:'Global'})], axis=1)
    if split_on_china:
        pop = pd.read_excel(xl, sheet_name='GDP growth', header=[0], 
                index_col=0, usecols='O:T').sort_index().sort_index(axis=1).dropna()
        if include_global:
            ph = pd.DataFrame(pop.sum(axis=1))
            pop = pd.concat([pop, ph.rename(columns={ph.columns[0]:'Global'})], axis=1)
    if which=='Percent per year':
        gdp = gdp.mul(100)
        sector = 'GDP change per year'
        ylabel = 'GDP change per year (%)'
    else:
        sector = 'GDP per capita'
        ylabel = 'GDP (2018 USD) per capita'
        gdp = gdp.rename(columns=dict(zip(gdp.columns, [i.split('.')[0] for i in gdp.columns])))
        if split_on_china:
            pop = pop.rename(columns=dict(zip(pop.columns, [i.split('.')[0] for i in pop.columns])))
            gdp.loc[:,'RoW'] = gdp[non_china].sum(axis=1)
            pop.loc[:,'RoW'] = pop[non_china].sum(axis=1)
            gdp = gdp/pop/1000
            gdp.drop(columns=non_china, inplace=True)
        if norm:
            gdp = gdp.apply(lambda x: x/gdp.loc[norm_year], axis=1)
    gdp = gdp.loc[:end_year]
    gdp.plot(ax=ax[-1], title=sector, ylabel=ylabel, xlabel='Year')

    fig.tight_layout()
    fig.set_dpi(dpi)

    data = pd.concat([
        volumes,
        pd.DataFrame(gdp.stack()).rename(columns={0:'GDP'}).stack().unstack(1).unstack()],
        axis=1).T.sort_index().T
    fig.savefig(f'{many_sg.folder_path}/figures/figure_s4_sectoral_volume_indicators.pdf')
    fig.savefig(f'{many_sg.folder_path}/figures/figure_s4_sectoral_volume_indicators.png')
    data.to_csv(f'{many_sg.folder_path}/figures/figure_s4_sectoral_volume_indicators.csv')
    plt.close()

def figure_s5(many_sg, show=False):
    import matplotlib as mpl
    target_rir = 0.55
    with warnings.catch_warnings(): 
        warnings.simplefilter('error')
        demands = pd.DataFrame(np.nan, np.arange(1912,2041),[-.03,-.02,-.01,0])
        models = []
        for d in demands.columns:
            dm = demandModel(verbosity=0)
            dm.hyperparam.loc['historical_growth_rate','Value'] = 0.03
            dm.hyperparam.loc['china_fraction_demand','Value'] = 0.7
            dm.hyperparam.loc['sector_specific_dematerialization_tech_growth','Value'] = d
            dm.hyperparam.loc['recycling_input_rate_china'] = target_rir
            dm.hyperparam.loc['recycling_input_rate_row'] = target_rir
            dm.run()
            if d==-1: d='Copper'
            else: d = round(d,2)
            demands.loc[:,d] = dm.demand.sum(axis=1)
            models += [dm]
        demands = demands.dropna(axis=1)

    fig,ax = easy_subplots(2)
    demands.plot(title='Demand, varying dematerialization',xlabel='Year',ylabel='Demand (kt)', ax=ax[0], color=list(mpl.color_sequences['Dark2'][:5])+['k']).grid(axis='x')
    ax[0].legend(title='Dematerialization rate',fontsize=20, title_fontsize=20, labelspacing=0.4, alignment='left', borderaxespad=0.1)
    d_a = demands.copy()

    with warnings.catch_warnings(): 
        warnings.simplefilter('error')
        demands = pd.DataFrame(np.nan, np.arange(1912,2041),[-3,-1,1,3])
        models = []
        for d in demands.columns:
            dm = demandModel(verbosity=0)
            dm.hyperparam.loc['historical_growth_rate','Value'] = 0.03
            dm.hyperparam.loc['china_fraction_demand','Value'] = 0.7
            dm.commodity_price_series.loc[2019:] = [dm.commodity_price_series.loc[2019]*(1+d/100)**(i-2019) for i in np.arange(2019,2041)]
            dm.hyperparam.loc['recycling_input_rate_china'] = target_rir
            dm.hyperparam.loc['recycling_input_rate_row'] = target_rir
            dm.run()
            # if d==-1: d='Copper'
            # else: d = round(d,2)
            demands.loc[:,d] = dm.demand.sum(axis=1)
            models += [dm]
        demands = demands.dropna(axis=1)

    demands.plot(title='Demand, varying price',xlabel='Year',ylabel='Demand (kt)', ax=ax[1], color=list(mpl.color_sequences['Dark2'][:5])+['k']).grid(axis='x')
    ax[1].legend(title='Price change per year (%)',fontsize=20, title_fontsize=20, labelspacing=0.4, alignment='left', borderaxespad=0.1)
    fig.tight_layout()
    d_b = demands.copy()

    data = pd.concat([d_a,d_b], axis=1, keys=['Demand, varying dematerialization','Demand, varying price'])
    save_quick_hist(many_sg, fig, data, None, figure_description='s5_demand_vary_demat_price',)
    if show:
        plt.show()
    plt.close()

def figure_s6(many_sg, show=False):
    target_rir = 0.55
    with warnings.catch_warnings(): 
        warnings.simplefilter('error')
        demands = pd.DataFrame(np.nan, np.arange(1912,2041),list(np.arange(0.01,0.051,0.01))+list([-1]))
        models = []
        for d in demands.columns:
            dm = demandModel(verbosity=0)
            dm.hyperparam.loc['historical_growth_rate','Value'] = 0.03
            dm.hyperparam.loc['china_fraction_demand','Value'] = 0.7
            dm.hyperparam.loc['historical_growth_rate','Value'] = d
            dm.hyperparam.loc['recycling_input_rate_china'] = target_rir
            dm.hyperparam.loc['recycling_input_rate_row'] = target_rir
            dm.run()
            if d==-1: d='Copper'
            else: d = round(d,2)
            demands.loc[:,d] = dm.demand.sum(axis=1)
            models += [dm]
        demands = demands.dropna(axis=1)

    fig,ax = easy_subplots(2)
    demands.plot(title='Varying historical growth rate',xlabel='Year',ylabel='Demand (kt)', ax=ax[0], color=list(mpl.color_sequences['Dark2'][:5])+['k']).grid(axis='x')
    ax[0].legend(title='Historical growth rate',fontsize=20, title_fontsize=20, labelspacing=0.4, alignment='left', borderaxespad=0.1)


    collection = pd.concat([i.collection_rate.loc[idx[:,'China'],['Other','Transport']] for i in models],axis=1,keys=demands.columns).loc[2019].loc['China'].unstack().rename(columns={'Transport':'All others','Other':'Transport'})
    collection.plot.bar( 
        title='Collection rate for varying historical growth',xlabel='Historical growth rate',ylabel='Collection rate, recycling input rate', ax=ax[1],
    ).grid(axis='x')
    pd.Series(target_rir,demands.columns).plot(color='k',linestyle='--',ax=ax[1])
    rirs = pd.Series([i.hyperparam['Value']['recycling_input_rate_china'] for i in models],demands.columns)
    rirs.plot(linewidth=0,marker='s',markerfacecolor='k',markersize='15').grid(axis='x')
    ax[1].legend(['Target recycling input rate','Recycling input rate','Collection rate, transport','Collection rate, other sectors'],fontsize=20, labelspacing=0.25, borderaxespad=0.1)
    ax[1].set_ylim(0,1.39)
    fig.tight_layout()

    data = pd.concat([
        collection.rename(columns={'Transport':'Collection rate, transport','All others':'Collection rate, other sectors'}),
        pd.DataFrame(rirs).rename(columns={0:'Recycling input rate'})
        ],axis=1)
    demands
    save_quick_hist(many_sg, fig, data, demands, figure_description='s6_demand_growth_rate_collection', alt_statistics_name='leftside')
    if show:
        plt.show()
    plt.close()

def figure_s7(many_sg, show=False):
    filename = 'input_files/user_defined/price adjustment results.csv'
    prices = pd.read_csv(filename, index_col=0).sort_index()
    commodities = sorted(['Aluminum','Steel','Gold','Tin','Copper','Nickel','Silver','Zinc','Lead'])
    fig,ax=easy_subplots(commodities)
    col_list = []
    for com,a in zip(commodities,ax):
        cols = [com+' original',f'log({com})',f'Rolling {com}']
        col_list += cols
        prices[f'Rolling {com}'] = prices[cols[1]].rolling(min_periods=1, window=5, center=True).mean()
        prices[cols].plot(ax=a, style=[':','--','-']).grid(axis='x')
        a.legend(['Unadjusted','Inflation and\nregression adjusted','Rolling mean'],loc='upper left')
        a.set_ylim(a.get_ylim()[0],a.get_ylim()[1]*1.3)
        a.set(title=f'Annual historical {com.lower()} price',
            xlabel='Year',
            ylabel=f'{com} price (USD/t)')
    fig.tight_layout()
    data = prices[col_list].rename(columns={f'{c} original':f'{c}, unadjusted' for c in commodities})
    data = data.rename(columns={f'log({c})':f'{c}, inflation and regression adjusted' for c in commodities})
    data = data.rename(columns={f'Rolling {c}':f'{c}, rolling mean' for c in commodities})
    save_quick_hist(many_sg, fig, data, None, figure_description='s7_prices_plus_adjust')
    if show:
        plt.show()
    plt.close()

def figure_s8_to_s20_table_s2_to_s7(many_sg, show=False):
    """
    All the plots done in this function are using proprietary data that cannot be shared. 
    To maximize transparency, we include the histogram data for each of the data sets used,
    including where scatter plots are provided. Histogram plots can be recreated from the 
    values and edges data (number of items per bin, edges of bins) using the
    matplotlib.pyplot.stairs() function.
    """
    for data_folder in ['../generalization-cannot-share/SP Global data','generalization-cannot-share/SP Global data']:
        if os.path.exists(data_folder):
            break
    if 'pri_and_co_pm.pkl' not in os.listdir(data_folder) and 'pri_and_co_pm.zip' in os.listdir(data_folder):
        with zipfile.ZipFile(f'{data_folder}/pri_and_co_pm.zip', 'r') as zip_ref:
            zip_ref.extractall(data_folder)

    primary_only = pd.read_pickle(f'{data_folder}/primary_only.pkl')
    coproduct_only = pd.read_pickle(f'{data_folder}/coproduct_only.pkl')
    pri_and_co = pd.read_pickle(f'{data_folder}/pri_and_co.pkl')
    pri_and_co_ot = pd.read_pickle(f'{data_folder}/pri_and_co_ot.pkl')
    pri_and_co_pm = pd.read_pickle(f'{data_folder}/pri_and_co_pm.pkl')
    tcrc_ot = pd.read_pickle(f'{data_folder}/tcrc_converted_ore_treated_basis.pkl')
    concentrates = pd.read_pickle(f'{data_folder}/concentrates.pkl')
    opening = pd.read_pickle(f'{data_folder}/opening.pkl')
    mine_parameters = pd.read_pickle(f'{data_folder}/mine_parameters.pkl')

    # Figure S8
    production = primary_only[[i for i in primary_only.columns if 'Prod' in i[0]]].droplevel(2).droplevel(0,axis=1).stack().replace(0,np.nan).dropna().astype(float)
    data, fig, hist_data = quick_hist(production,height_scale=0.5, xlabel='Production (kt)', width_scale=0.6)
    save_quick_hist(many_sg, fig, hist_data, data, 's8_production_histogram')

    # Figure S9
    hg = (primary_only.loc[:,idx['Head Grade (%)',:]].astype(float).groupby(level=[0,1]).sum().stack().replace({0:np.nan})).dropna()
    data, fig, hist_data = quick_hist(hg,width_scale=0.7,height_scale=0.5, xlabel='Head grade (%)')
    save_quick_hist(many_sg, fig, hist_data, data, 's9_ore_grade_histogram')

    # Figure S10
    data, fig, hist_data = plot_best_dists(primary_only, 'Metal Payable Percent (%)', xlabel='100-Payable percent (%)', height_scale=0.5, width_scale=0.7, show=False)
    save_quick_hist(many_sg, fig, hist_data, None, 's10_payable_percent_histogram')

    # Figure S11
    tcrc_primary, tcrc_pri = get_tcrc_primary(concentrates, opening)
    regress_yearso = get_regress_yearso(primary_only, pri_and_co, tcrc_primary)
    stat, fig, hist_data = quick_hist(regress_yearso['Cumulative Ore Treated (kt)'],height_scale=0.5,)
    data = regress_yearso['Cumulative Ore Treated (kt)'].copy()
    data = pd.concat([data],keys=[0]).unstack(1).stack()
    save_quick_hist(many_sg, fig, hist_data, stat, 's11_cumu_ot_ratio_ot')

    # Figure S12
    fig, ax = easy_subplots(1,height_scale=0.8, width_scale=0.8)
    a = ax[0]
    reserves = primary_only.copy().rename(columns={'Reserves: Ore Tonnage (tonnes)':'Reserves (kt)','Total Resources: Ore Tonnage Excl Reserves (tonnes)':'Resources excl reserves (kt)','Mill Capacity - tonnes/year':'Capacity (kt)'})
    reserves.loc[:,idx[['Reserves (kt)','Resources excl reserves (kt)','Capacity (kt)'],:]] = reserves.loc[:,idx[['Reserves (kt)','Resources excl reserves (kt)','Capacity (kt)'],:]].astype(float)/1e3
    ratio = reserves.loc[:,'Reserves (kt)'].replace(0,np.nan)/reserves.loc[:,'Ore Treated (kt)'].replace(0,np.nan)
    ratio = ratio.stack().dropna()
    ratio_sub = ratio[ratio<ratio.quantile(.95)]
    r, hist_data = find_best_dist(ratio_sub.astype(float), ax=a)
    fit = stats.lognorm.fit(ratio_sub)
    coef, pval = [], []
    for n in np.arange(0,100):
        sim = stats.lognorm.rvs(fit[0], fit[1], fit[2], size=len(ratio_sub), random_state=n)
        result = stats.kstest(ratio_sub,sim)
        coef += [result[0]]
        pval += [result[1]]
    a.set(title='Lognorm\n'+'K-S test p-value: {:.3f}'.format(np.mean(pval)))
    if show:
        plt.show()
    plt.close()
    save_quick_hist(many_sg, fig, hist_data, None, 's12_reserve_ratio_ot')

    # Figure S13
    regress = get_regress(primary_only, pri_and_co)
    stat, fig, hist_data = quick_hist(regress['Numerical Risk'],log=False,rounded=True,height_scale=0.5)
    save_quick_hist(many_sg, fig, hist_data, stat, 's13_numerical_risk')

    # Figure S14
    oge_results, regress_oge = get_oge(primary_only, pri_and_co, tcrc_primary)
    data = 1-pd.concat([oge_results.slope],keys=[0]).unstack(1).stack()
    fit = stats.lognorm.fit(data)
    pvals = []
    ind = 0
    for n in range(0,100):
        sim = stats.lognorm.rvs(fit[0],fit[1],fit[2], size=len(data), random_state=n)
        pval = stats.kstest(data, sim)[1]
        if n>1 and pval>np.max(pvals):
            ind = n
        pvals += [pval]

    sim = stats.lognorm.rvs(fit[0],fit[1],fit[2], size=len(data), random_state=ind)
    fig,axes = easy_subplots(3, ncol=3)
    ax = axes[0]
    data = 1-data
    sim = 1-sim
    bins = np.linspace(np.min(data), np.max(data), 40)
    (n_hist, bins_hist, patches_hist) = ax.hist(data, bins=bins, alpha=0.5, label='Real', color='tab:orange')
    (n_sim, bins_sim, patches_sim) = ax.hist(sim, bins=bins, alpha=0.5, label='Simulated', color='tab:blue');
    ax.set(title='Ore grade elasticity to\ncumulative production', ylabel='Count', xlabel='Ore grade elasticity')
    ax.legend()
    hist_data_i = pd.concat([
        pd.concat([pd.Series(n_hist),pd.Series(n_sim)],axis=1,keys=['Historical','Simulated']),
        pd.concat([pd.Series(bins_hist),pd.Series(bins_sim)],axis=1,keys=['Historical','Simulated']),
    ],keys=['Values','Edges'])

    deyeared_grades = plotting_grades_over_time(primary_only=primary_only, axes=axes)
    fig.tight_layout()
    save_quick_hist(many_sg, fig, hist_data_i, deyeared_grades, 's14_ore_grade_elasticity', alt_statistics_name='grade_data')

    # Figure S15, Table S2
    regress_rr = get_regress_rr(primary_only, pri_and_co, tcrc_primary)
    x = sm.add_constant(regress_rr[[i for i in regress_rr.columns if ('Head Grade' in i or 'Price' in i or 'SX' in i
                                                                  or 'Mine Type' in i)]]).astype(float)
    m = sm.GLS(regress_rr[[i for i in regress_rr.columns if 'Recovery Rate (%)' in i]],x,missing='drop').fit(cov_type='HC3')
    save_regression_table(many_sg, m, 's2_recovery_rate')

    fig,ax = easy_subplots(2)
    lin_predict = pd.concat([
        (regress_rr[[i for i in regress_rr.columns if 'Recovery Rate (%)' in i]]),
        (m.predict(x)),
        regress_rr.Commodity],
        axis=1,keys=['Actual','Predicted','Commodity']).droplevel(1,axis=1)
    do_a_regress(lin_predict.Actual,lin_predict.Predicted,ax=ax[0],loc='lower right')
    sns.scatterplot(data=lin_predict, x='Actual',y='Predicted',hue='Commodity',ax=ax[0])
    ax[0].legend(title=None, labelspacing=0.2,markerscale=1,fontsize=16,loc='lower left', ncol=2)
    ax[0].set(xlim=(0,110),ylim=(0,110), title='Predicted vs actual')

    hist_data = plot_lin_predict(lin_predict, ax[1], 'Recovery rate (%)')
    fig.tight_layout()
    if show:
        plt.show()
    plt.close()
    save_quick_hist(many_sg, fig, hist_data, None, 's15_recovery_rate')

    # Figure S16
    tot_reclamation = mine_parameters['Reclamation cost ($M)'].dropna().groupby(level=[0,1]).sum()
    capacity = mine_parameters['Capacity (kt)'].dropna().groupby(level=[0,1]).mean()/1e6 
    # ^ note that capacity is actually in tonnes, is mislabeled
    capacity.name = 'Capacity (Mt)'
    ind = np.intersect1d(tot_reclamation.index, capacity.index)
    capacity, tot_reclamation = capacity.loc[ind].astype(float), tot_reclamation.loc[ind].astype(float)
    fig,a = plt.subplots()
    do_a_regress(capacity, tot_reclamation, ax=a, log=True)
    a.set(title='Total reclamation cost vs. mill capacity')
    data = pd.concat([pd.concat([tot_reclamation, capacity], axis=1)],keys=[0]).unstack(1).stack()
    n_rec, bins_rec = np.histogram(tot_reclamation)
    n_cap, bins_cap = np.histogram(capacity)
    hist_data_i = pd.concat([
        pd.concat([pd.Series(n_rec),pd.Series(n_cap)],axis=1,keys=['Total reclamation cost','Capacity']),
        pd.concat([pd.Series(bins_rec),pd.Series(bins_cap)],axis=1,keys=['Total reclamation cost','Capacity']),
        ],keys=['Values','Edges'])
    save_quick_hist(many_sg, fig, hist_data_i, None, 's16_reclamation_costs')

    # Table S3
    regress_tcrc_conc = get_regress_tcrc_conc(tcrc_pri, pri_and_co)
    x = sm.add_constant(regress_tcrc_conc[[i for i in regress_tcrc_conc.columns if ('Head Grade' in i or 'Price' in i or 'SX' in i
                                        or 'Mine Type' in i or 'Refining charge' in i) and 'Powder' not in i and 'Matte' not in i 
                                        and '_nan' not in i and 'Dore' not in i and 'Stock Pile' not in i and 'Placer' not in i]])
    m = sm.GLS(regress_tcrc_conc[[i for i in regress_tcrc_conc.columns if 'TCRC (USD/t ore)' in i]],
            x.astype(float),missing='drop').fit(cov_type='HC3')
    save_regression_table(many_sg, m, 's3_tcrc_regression')

    # Figure S17
    fig,ax = easy_subplots(2)
    lin_predict = pd.concat([
            regress_tcrc_conc[[i for i in regress_tcrc_conc.columns if 'TCRC (USD/t ore)' in i]],
            m.predict(x),
            regress_tcrc_conc.Commodity],
            axis=1,keys=['Actual','Predicted','Commodity']).droplevel(1,axis=1)    
    plot_lin_predict_scatter(lin_predict, ax[0], False)
    plot_lin_predict_scatter(lin_predict, ax[1], True)
    hist_data = lin_predict_to_hist(lin_predict)
    save_quick_hist(many_sg, fig, hist_data, None, 's17_tcrc_regression')

    # Table S4
    regress_scapex = get_regress_scapex(primary_only, pri_and_co, tcrc_primary)
    summ = pd.DataFrame()
    for j in ['Capacity','none']:
        for k in ['Grade','none']:
            for n in ['Price','none']:
                for o in ['SXEW','none']:
                    for p in ['Numerical','none']:
                        columnar = [i for i in regress_scapex.columns if (j in i or k in i or n in i or o in i or p in i
                                                                                or 'Mine Type' in i) and 'Placer' not in i]
                        x = sm.add_constant(regress_scapex[columnar]).astype(float)
                        m = sm.GLS(regress_scapex[[i for i in regress_scapex.columns if 'sCAPEX norm' in i]].astype(float),
                                x,missing='drop').fit(cov_type='HC3')
                        summ.loc['-'.join([j,k,n,o,p]),'AIC'] = m.aic
                        summ.loc['-'.join([j,k,n,o,p]),'BIC'] = m.bic
                        summ.loc['-'.join([j,k,n,o,p]),'rsq'] = m.rsquared
                        summ.loc['-'.join([j,k,n,o,p]),'model'] = m
                        
                        x = regress_scapex[columnar].astype(float).values
                        summ.loc['-'.join([j,k,n,o,p]),'max_VIF'] = np.max([variance_inflation_factor(x,j) for j in np.arange(0,x.shape[1])])
    m = summ.loc[summ.BIC.idxmin(),'model']
    save_regression_table(many_sg, m, 's4_sustaining_capex_regression')

    # Figure S18
    j,k,n,o,p = summ.BIC.idxmin().split('-')
    columnar = [i for i in regress_scapex.columns if (j in i or k in i or n in i or o in i or p in i
                                                            or 'Mine Type' in i)]
    x = sm.add_constant(regress_scapex[columnar]).astype(float)
    y = regress_scapex[[i for i in regress_scapex.columns if 'sCAPEX norm' in i]].astype(float)
    m = sm.GLS(y,x,missing='drop').fit(cov_type='HC3')

    if 'Commodity' not in regress_scapex.columns:
        regress_scapex['Commodity'] = regress_scapex.index.get_level_values(1)
    lin_predict = pd.concat([
                regress_scapex[[i for i in regress_scapex.columns if 'sCAPEX norm' in i]],
                m.predict(x),
                regress_scapex.Commodity],
                axis=1,keys=['Actual','Predicted','Commodity']).droplevel(1,axis=1) 
    fig,ax=easy_subplots(2)
    lin_predict = pd.concat([lin_predict],keys=[0]).unstack(1).stack()
    plot_lin_predict_scatter(lin_predict, ax[0], False)
    lin_predict = lin_predict.loc[lin_predict.Actual<lin_predict.Actual.quantile(0.99)]
    plot_lin_predict_scatter(lin_predict, ax[1], True)
    hist_data = lin_predict_to_hist(lin_predict)
    save_quick_hist(many_sg, fig, hist_data, None, 's18_sustaining_capex_regression')

    # Table S5
    regress_tcm = get_regress_tcm(primary_only, pri_and_co)
    # j,k,n,o,p,q = summ_tcm.BIC.idxmin().split('-')
    j,k,n,o,p,q = 'none-Grade-Price-SXEW-none-none'.split('-')
    # j,k,n,o,p,q = 'none-Grade-Price-SXEW-none-Global Region'.split('-')
    # j,k,n,o,p,q = 'Capacity-Grade-Price-SXEW-Numerical-Global Region'.split('-')

    columnar = [i for i in regress_tcm.columns if (j in i or k in i or n in i or o in i or p in i
                                                            or 'Mine Type' in i)]
    x = sm.add_constant(regress_tcm[columnar]).astype(float)
    y = regress_tcm[[i for i in regress_tcm.columns if 'Total Cash Margin' in i]].astype(float)
    m = sm.GLS(y,x,missing='drop').fit(cov_type='HC3')
    save_regression_table(many_sg, m, 's5_total_cash_margin_regression')
    
    # Figure S19
    fig,ax = easy_subplots(2)
    lin_predict = pd.concat([
        y,
        m.predict(x),
        regress_tcm.Commodity],
        axis=1,keys=['Actual','Predicted','Commodity']).droplevel(1,axis=1)
    plot_lin_predict_scatter(lin_predict, ax[0], False)
    plot_lin_predict_scatter(lin_predict, ax[1], True)
    hist_data = lin_predict_to_hist(lin_predict)
    save_quick_hist(many_sg, fig, hist_data, None, 's19_total_cash_margin_regression')

    # Table S6
    regress_grade = get_regress_grade(primary_only, pri_and_co, tcrc_primary)
    x = sm.add_constant(regress_grade[[i for i in regress_grade.columns if ('Price' in i
                                                                    or 'Mine Type' in i)]]).astype(float)
    m = sm.GLS(regress_grade[[i for i in regress_grade.columns if 'Head Grade (%)' in i]],
            x,missing='drop').fit(cov_type='HC3')
    save_regression_table(many_sg, m, 's6_head_grade')

    # Table S7
    models_cat, results_cat, pval_cat, fe_models_cat, fe_results_cat, fe_pval_cat, best_models_cat, best_result, best_pvals = panel_regression_categorical(
        primary_only,
        pri_and_co_ot,
        dependent_string='Total Minesite Cost (USD/t)',
        independent_string='Head Grade (%)',
    #     second_independent_string='Commodity Price (USD/t)',
        take_diff=False,
        add_oil=True,
        add_constant=True,
        add_categorical=False,
        add_time=True, 
        take_log=True,
        inflation_adjust=True,
        payable_basis=True,
        rank_check=False)
    print('Done')
    try:
        comp = compare(best_models_cat,precision='pvalues')
    except:
        print('Some models\' pvalues cannot be calculated:')
        models_cat_drop = {}
        for i in best_models_cat.keys():
            try:
                best_models_cat[i].pvalues
                models_cat_drop[i] = best_models_cat[i]
            except:
                print('\t'+i)
        comp = compare(models_cat_drop, precision='pvalues')
    comp.dep_var = pd.Series('log(Total Minesite Cost (USD/t))', comp.cov_estimator.index)
    concat_list = ['dep_var','estimator_method','nobs','cov_estimator','rsquared','rsquared_within','rsquared_between','rsquared_overall',]
    ['f_statistic',]
    comb_param = pd.DataFrame(columns=comp.params.columns)
    for i in comp.params.index:
        comb_param.loc[i] = comp.params.loc[i]
        comb_param.loc[i+' p-value'] = comp.pvalues.loc[i]
    pd.concat([
        pd.concat([getattr(comp,i) for i in concat_list], axis=1).T,
        comp.f_statistic.T,
        comb_param
    ], ).to_csv(f'{many_sg.folder_path}/tables/table_s7_panel_regression_minesite_cost.csv')

    # Figure S20:
    inds = [i for i in results_cat.index if np.any([j in i for j in ['Price','Grade','price']]) or i=='Year']
    fig,ax=easy_subplots(inds,len(inds))
    # with warnings.catch_warnings():
    warnings.filterwarnings('ignore', '.*do not.*', )
    np.seterr(all='ignore')
    for i,a in zip(inds, ax):
        best_result[best_pvals<0.05].loc[i].plot.bar(ax=a,title=i,color='tab:blue',alpha=1).grid(axis='x')
        best_result[best_pvals>0.05].loc[i].plot.bar(ax=a,title=i,color='tab:blue',alpha=0.25).grid(axis='x')
    title_str = 'Commodity price excluded' if len(inds)==3 else 'Commodity price included'
    title_str += ' (light blue = insignificant at 95% confidence level)'
    if len(inds)==3:
        best_result_excl = best_result.copy()
    else:
        best_result_incl = best_result.copy()
    fig.suptitle(title_str, 
                weight='bold', y=0.92)
    fig.tight_layout()
    fig.set_dpi(200)
    # init_plot2(font='Arial',font_family='sans-serif')
    data = pd.concat([best_result, best_pvals],keys=['Parameter values','p-values'])
    label = 's20_panel_regression_coef_minesite_cost'
    fig.savefig(f'{many_sg.folder_path}/figures/figure_{label}.png')
    fig.savefig(f'{many_sg.folder_path}/figures/figure_{label}.pdf')
    data.to_csv(f'{many_sg.folder_path}/figures/figure_{label}.csv')
    if show:
        plt.show()
    plt.close()


    print_grade_change_per_year(primary_only)

def figure_s21(many, show=False):
    many_sg = many
    fig,ax = easy_subplots(1, width_scale=2.5)
    ax = ax[0]
    data = many_sg.rmse_df_sorted.loc[idx[:,'score'],:].droplevel(1).T
    ax.set_prop_cycle(plt.cycler('color',mpl.color_sequences['Dark2']+[mpl.color_sequences['Dark2'][0]])+plt.cycler('linestyle',['-','--',':','-.']*2+['--']))
    data.plot(ax=ax)
    ax.set(ylim=[ax.get_ylim()[0]*0.7,4], xlabel='Scenario number, sorted', ylabel='Score value', title='Sorted score values for each commodity')
    ax.legend(ncol=2)
    fig.set_dpi(150)
    fig.savefig(f'{many_sg.folder_path}/figures/figure_s21_sorted_score_values.pdf')
    fig.savefig(f'{many_sg.folder_path}/figures/figure_s21_sorted_score_values.png')
    data.to_csv(f'{many_sg.folder_path}/figures/figure_s21_sorted_score_values.csv')
    if show:
        plt.show()
    plt.close()

def figure_s22_to_s24(many, show=False):
    many_sg = many
    min_pvals = []
    index = np.arange(8,50)
    for n in index:
        min_pvals += [get_min_pvals_across_tests(many, n)]
    pv = pd.concat(min_pvals, keys=index)
    commodities = many.rmse_df.index.get_level_values(0).unique()
    n_per_plot = 5
    n_subplots = int(np.ceil(pv.shape[1]/n_per_plot))

    n_per_figure = 3
    n_figures = int(np.ceil(len(commodities)/n_per_figure))
    for k in range(n_figures):
        commodities_sub = commodities[k*n_per_figure:(k+1)*n_per_figure]
        fig,axes = easy_subplots(n_subplots*n_per_figure, n_per_figure, height_scale=0.8)
        for e,com in enumerate(commodities_sub):
            pv_ag = pv.loc[idx[:,com],:].droplevel(0,axis=1).droplevel(1)
            # pv_ag = pv_ag.div(pv_ag.max())

            ax = axes[e::n_per_figure]
            for i,a in enumerate(ax):
                if i*n_per_plot>=pv_ag.shape[1]:
                    continue
                pv_ag.iloc[:,i*n_per_plot:(i+1)*n_per_plot].plot(ax=a)
                a.legend(fontsize=14)
                a.set_ylim(-0.05,a.get_ylim()[1]*1.25)
                com2 = com.replace('Steel','Fe')
                a.set(title=f'{com2}, parameter set {i+1}', xlabel='Number of scenarios selected', ylabel='P-value')
        # fig.set_dpi(150)
        fig.tight_layout()
        fig.savefig(f'{many_sg.folder_path}/figures/figure_s{22+k}_uniformity_pvalues.pdf')
        fig.savefig(f'{many_sg.folder_path}/figures/figure_s{22+k}_uniformity_pvalues.png')
        if show:
            plt.show()
        plt.close()
    pv.to_csv(f'{many_sg.folder_path}/figures/figure_s22-s24_uniformity_pvalues.csv')
    
def figure_s25(many_sg, show=False):
    X1 = many_sg.rmse_df_sorted.loc[idx['Ag','primary_oge_scale'],:24]
    loc, scale = get_correct_loc_scale(X1.name[1])
    X = (X1.values-loc)/scale
    Y = ndtri(X)

    lowest_pval = test_normality_from_uniform_min(X1, normality_only=True)
    ttest_pval = stats.ttest_1samp(Y, 0)[1]
    fig,ax=easy_subplots(2)
    ax[0].hist(X)
    ax[0].set(xlim=(-0.05,1.05), title='Original distribution', ylabel='Frequency', xlabel='Parameter value')
    ax[1].hist(ndtri(X1))
    ax[1].set(title='Normal-transformed distribution', ylabel='Frequency', xlabel='Transformed parameter value', xlim=(-2.65,2.65))
    ax[1].text(0.3,0.4, 'Min. norm. p-val.: {:.3f}\nT-test p-value: {:.3f}'.format(lowest_pval,ttest_pval), transform=ax[1].transAxes)
    add_axis_labels(fig)
    pd.concat([X1,ndtri(X1)], axis=1, keys=['Original','Normal-transformed']).to_csv(f'{many_sg.folder_path}/figures/figure_s25_normality_failure.csv')
    fig.savefig(f'{many_sg.folder_path}/figures/figure_s25_normality_failure.pdf')
    fig.savefig(f'{many_sg.folder_path}/figures/figure_s25_normality_failure.png')
    if show:
        plt.show()
    plt.close()

def figure_s26(many_sg, show=False):
    tests = ['uniform-min','uniform-min-normality','uniform-ttest','uniform-shapiro','uniform-lilliefors-both','uniform-dagostino','uniform-anderson-both']
    nice_test_names = {'uniform-min':'All tests, min.\np-value','uniform-min-normality':'All normality\ntests','uniform-shapiro':'Shapiro-Wilk\ntest',
                    'uniform-lilliefors-both':'Lilliefors-corr.\nK-S test','uniform-dagostino':'D\'Ag.-Pea.\ntest',
                    'uniform-anderson-both':'And.-Dar.\ntest', 'uniform-ttest':'One-sample\nT-test'}
    nice_names_list = [nice_test_names[i] for i in tests]
    alt_means = pd.DataFrame()
    alt_means_list = []
    alt_means_dict = {}
    pvals_list = []
    means_list = []
    for test in tests:
        table, means, pvals, alt_means1, fig = plot_colorful_table2(many_sg, stars=test, dpi=250, 
                                                value_in_parentheses='None',#'standard error'
                                                rand_size=100, how_many_tests=100, show=False)
        blankie = pd.DataFrame(np.nan, index=alt_means1.index, columns=[''])
        if test == tests[0]:
            alt_means = pd.concat([alt_means, alt_means1], axis=1)
        else:
            alt_means = pd.concat([alt_means, blankie, alt_means1], axis=1)
        alt_means_list += [alt_means1]
        pvals_list += [pvals]
        means_list += [means]
        alt_means_dict[test] = alt_means1


    plots = alt_means1.index.get_level_values(0).unique()
    height_ratios = [alt_means1.loc[i].shape[0] for i in plots]
    ncol = len(alt_means_list)
    fig,axes = easy_subplots(len(plots)*ncol,ncol, height_ratios=height_ratios, use_subplots=True, sharex=True)
    axes = np.array(axes)
    axes = axes.reshape(len(plots), ncol)
    for e, test in enumerate(alt_means_dict):
        alt_means = alt_means_dict[test]
        alt_means[alt_means==0] = np.nan
        ax = axes[:,e]
        for i,a in zip(plots, ax):
            sub_means = alt_means.loc[i]
            sub_means= sub_means.rename({'Steel':'Fe'}).rename(columns={'Steel':'Fe'})
            sns.heatmap(sub_means,
                        ax=a,
                        fmt='s',
                        annot_kws={'fontsize':18},
                        xticklabels=True,
                        yticklabels=True,
                        cmap='vlag',
                        cbar=False,
                        vmin=-1,
                        vmax=1,
                    )
            # ax.set_yticks(np.arange(ax.get_yticks()[0],ax.get_yticks()[-1]+1,1))
            a.tick_params(labelbottom= i==plots[-1], labeltop= i==plots[0], labelsize=20, color='white', rotation=90)
            
            if e==0:
                a.tick_params(axis='y',rotation=0)
                
                if a in ax[-1:]:
                    a.set_ylabel(i,rotation='horizontal',horizontalalignment='left', labelpad=280, fontsize=22)
                elif a in ax[-2:]:
                    a.set_ylabel(i,rotation='horizontal',horizontalalignment='left', labelpad=260, fontsize=22)
                elif i=='Price formation':
                    a.set_ylabel(i,rotation='horizontal',horizontalalignment='left', labelpad=300, fontsize=22)
                else:
                    a.set_ylabel(i,rotation='horizontal',horizontalalignment='left', labelpad=170, fontsize=22)
            else:
                a.set_ylabel(None)
                a.tick_params(labelleft=False, left=False)
            
            if i=='Demand response':
                a.set_xlabel(nice_test_names[test])
                a.xaxis.set_label_position('top')
            a.grid(True)
    fig.set_size_inches((30,15))
    fig.tight_layout()
    if show:
        plt.show()
    plt.close()
    pd.concat([
            pd.concat(alt_means_list, keys=nice_names_list, axis=1), 
            pd.concat([i.T for i in means_list], keys=nice_names_list, axis=1), 
            pd.concat([i.T for i in pvals_list], keys=nice_names_list, axis=1), 
    ], axis=1, keys=['Table reproduction','Mean parameter values','P-values']).\
    to_csv(f'{many_sg.folder_path}/figures/figure_s26_compare_significance_table.csv')
    fig.savefig(f'{many_sg.folder_path}/figures/figure_s26_compare_significance_table.pdf')
    fig.savefig(f'{many_sg.folder_path}/figures/figure_s26_compare_significance_table.png')
    
def figure_s27(many_sg, many_15, many_16, many_17, use_r2_instead_of_mape=False, show=False):
    r2 = get_r2(many_sg, many_15, many_16, many_17, use_r2_instead_of_mape=use_r2_instead_of_mape)
    r2_difference = r2.mul(-1).add(r2['Full'],axis=0)

    # R2 differences
    # largest_diff = r2_difference.apply(lambda x: x.loc[abs(x).idxmax()], axis=1)
    largest_diff = r2_difference.drop(columns='Full')
    # if 'To 2015' in largest_diff.columns:
    #     largest_diff.drop(columns='To 2015',inplace=True)
    largest_diff = largest_diff.stack()
    many_sg.largest_diff = largest_diff.copy()

    fig,axes = easy_subplots(2)
    ax = axes[0]
    data = abs(largest_diff)
    d1 = data.copy()
    v = ax.hist(data, cumulative=True, histtype='step', density=True, linewidth=6, bins=50)
    n = v[0]

    if use_r2_instead_of_mape:
        change_list = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.4]
    else:
        change_list = [0.2,  0.5,  1,    1.5,  2,    3,   4,   6]
    for x in change_list:
        if use_r2_instead_of_mape:
            print('Percent of R2 changes less than {:.02f}: {:.1f}'.format(x,(data<x).sum()/len(data)*100))
        else:
            print('Percent of MAPE changes less than {:.02f}: {:.1f}'.format(x,(data<x).sum()/len(data)*100))
        err = 0.005
    #     val = []
    #     while len(val)==0:
    #         val = v[0][abs(v[1][:-1]-x)<err]
    #         err *= 1.2
    #     new_ticks += [np.mean(val)]
        
    ax.set(xlim=(ax.get_xlim()[0], data.max()),
        title=r'$R^2$ difference from full tuning'+'\ncumulative density' if use_r2_instead_of_mape else 'MAPE difference from full tuning\ncumulative density',
        xlabel=r'Absolute difference in $R^2$' if use_r2_instead_of_mape else 'Absolute difference in MAPE',
        ylabel='Cumulative density',
    #        yticks=new_ticks,
        yticks=[0, n[0],n[1],n[4], n[8], n[14], 1] if use_r2_instead_of_mape else [0,n[0],n[1],n[2],n[4],1]
        );
    ax.grid(True, axis='both')
    ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%0.3f'))

    if not hasattr(many_sg, 'means_coef_of_var'):
        figure_s30(many_sg, many_17, many_16, many_15, use_r2_instead_of_mape=use_r2_instead_of_mape, show=False)    
    means_coef_of_var = many_sg.means_coef_of_var.copy()

    # Coef of variation
    ax = axes[1]
    data = abs(means_coef_of_var.stack()).mul(100)
    many_sg.data_coef = data.copy()
    d2 = data.copy()
    v = ax.hist(data,
                bins=50, cumulative=True, density=True, histtype='step',
                linewidth=6,
    )
    n = v[0]
    ax.set(
        title='Coefficient of variation cumulative denstiy,\n all parameters & commodities, full tuning',
        xlabel='Coefficient of variation (%)',
        ylabel='Cumulative fraction',
        yticks=[0,n[10], n[15], n[23], n[30], 1],
        xlim=(ax.get_xlim()[0],data.max()),
    )
    ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%0.3f'))
    ax.grid(True, axis='both')

    if show:
        print('')
        for x in [25,50,75,100]:
            print('Percent of coefficients of variation less than {:.0f}: {:.1f}'.format(x,(data<x).sum()/len(data)*100))

        if use_r2_instead_of_mape:
            print('\nLargest 10% of R2 differences')
        else:
            print('\nLargest 10% of MAPE differences')
        print(largest_diff[(largest_diff>largest_diff.quantile(0.9))])

        if use_r2_instead_of_mape:
            print('\nAdditional next 10% of R2 differences')
        else:
            print('\nAdditional next 10% of MAPE differences')
        print(largest_diff[(largest_diff>largest_diff.quantile(0.8))&(largest_diff<largest_diff.quantile(0.9))])

    data = pd.concat([
        d1,
        pd.concat([d2],keys=['b'])],keys='ab')
    
    fig.tight_layout()
    add_axis_labels(fig, xloc=-0.19, yloc=1.12)
    r2_or_mape = 'r2' if use_r2_instead_of_mape else 'mape'
    fig.savefig(f'{many_sg.folder_path}/figures/figure_s27_cumulative_densities_{r2_or_mape}_cov.pdf')
    fig.savefig(f'{many_sg.folder_path}/figures/figure_s27_cumulative_densities_{r2_or_mape}_cov.png')
    data.to_csv(f'{many_sg.folder_path}/figures/figure_s27_cumulative_densities_{r2_or_mape}_cov.csv')
    if show:
        plt.show()
    plt.close()

def figures_s28_and_s29(many_sg, many_17, many_16, many_15, show=False, all_only=False):
    if not all_only:
        fig, data = plot_best_fits_many({'Full':many_sg, 
                        'To 2016':many_17, 
                        'To 2015':many_16, 
                        'To 2014':many_15}, 
                    commodities='subset1',
                    show_all_lines=False, dpi=150)
        fig.savefig(f'{many_sg.folder_path}/figures/figure_s28_tuning_results_training1.pdf')
        fig.savefig(f'{many_sg.folder_path}/figures/figure_s28_tuning_results_training1.png')
        data.to_csv(f'{many_sg.folder_path}/figures/figure_s28_tuning_results_training1.csv')
        if show:
            plt.show()
        plt.close()

        fig, data = plot_best_fits_many({'Full':many_sg, 
                        'To 2016':many_17, 
                        'To 2015':many_16, 
                        'To 2014':many_15}, 
                    commodities='subset2',
                    show_all_lines=False, dpi=150)
        fig.savefig(f'{many_sg.folder_path}/figures/figure_s29_tuning_results_training2.pdf')
        fig.savefig(f'{many_sg.folder_path}/figures/figure_s29_tuning_results_training2.png')
        data.to_csv(f'{many_sg.folder_path}/figures/figure_s29_tuning_results_training2.csv')
        if show:
            plt.show()
        plt.close()

    fig_fits, data = plot_best_fits_many({'Full':many_sg, 
                    'To 2016':many_17, 
                    'To 2015':many_16, 
                    'To 2014':many_15}, 
                commodities='all',
                show_all_lines=False, dpi=10)
    plt.close()
    many_sg.fig_fits = fig_fits
    
def figure_s30(many_sg, many_17, many_16, many_15, use_r2_instead_of_mape=False, show=False):
    table, means, pvals_loc_loc, alt_means, fig = plot_colorful_table2(many_sg, stars='uniform-min', dpi=50, rand_size=None);
    plt.close()
    table15, means15, pvals15, alt_means15, fig = plot_colorful_table2(many_15, stars='uniform-min', dpi=50, rand_size=None);
    plt.close()
    table16, means16, pvals16, alt_means16, fig = plot_colorful_table2(many_16, stars='uniform-min', dpi=50, rand_size=None);
    plt.close()
    table17, means17, pvals17, alt_means17, fig = plot_colorful_table2(many_17, stars='uniform-min', dpi=50, rand_size=None);
    plt.close()

    rmse_df = many_sg.rmse_df_sorted.loc[:,:24]
    rmse_df = rmse_df.loc[idx[:,[i for i in rmse_df.index.levels[1] if np.all([j not in i for j in ['R2','RMSE','score']])]],:]
    scales = rmse_df.apply(lambda x: get_correct_loc_scale(x.name[1])[1], axis=1)
    scales_nice = scales.rename(make_parameter_names_nice(scales.index.levels[1]), level=1)

    cov_within = (rmse_df.std(axis=1)/scales).unstack()
    cov_within = cov_within.rename(make_parameter_names_nice(cov_within.columns), axis=1).T

    if not hasattr(many_sg, 'fig_fits'):
        figures_s28_and_s29(many_sg, many_17, many_16, many_15, show=False, all_only=True)
    fig_fits = many_sg.fig_fits
    means_all = pd.concat([means, means15, means16, means17], 
                        keys=['Full','Split, 2014','Split, 2015', 'Split, 2016']).unstack(0).droplevel(0,axis=1)
    means_all = means_all.T
    means_relative = means_all.apply(lambda x: x/means_all.loc[x.name[0]].loc['Full'], axis=1)
    means_relative[abs(means_relative)>3] = np.nan

    means_all_sub = means_all.copy().drop('Split, 2015',level=1)
    means_coef_of_full = means_all_sub.groupby(level=0).std()/means_all_sub.loc[idx[:,'Full'],:].droplevel(1)
    # means_coef_of_var = means_all_sub.groupby(level=0).std()/means_all_sub.groupby(level=0).mean()
    means_coef_of_var = means_all_sub.groupby(level=0).std()/scales_nice.unstack(0)
    means_coef_of_var = abs(means_coef_of_var)
    means_coef_of_var_nan = means_coef_of_var.copy()

    many_sg.means_coef_of_var = means_coef_of_var.copy()
    many_sg.means_all = means_all.copy()
    # means_coef_of_var_nan[abs(means_coef_of_var_nan)>7] = np.nan
    # fig, ax = plt.subplots(figsize=(14,14))
    # sns.heatmap(means_coef_of_var_nan, ax=ax, yticklabels=True,
    #             annot = means_coef_of_var, vmin=0, vmax=2,
    #            )
    covs = many_sg.means_coef_of_var.stack()
    print('CoV across training sets: ')
    for ii in [0.05,0.1,0.15,0.2]:
        print('\tfraction less than {:.2f}: {:.3f}'.format(ii, (covs<ii).sum()/len(covs)))

    r2_df = pd.DataFrame()
    for a in fig_fits.get_axes():
        title = a.get_title()
        lines = a.get_lines()
        labels = [i.get_label() for i in lines]
        if use_r2_instead_of_mape:
            labels = [i.replace('Simulated','Full') for i in labels if i!='Historical']
            keys = [i.split(':')[0] for i in labels]
            r2 = [float(i.split('=')[1]) for i in labels]
            r2_series = pd.Series(r2,keys)
            r2_series.name = title.split(', ')[1]
            r2_series = pd.concat([r2_series], keys=[title.split(',')[0]])
        else:
            labels = [i.replace('Simulated','Full') for i in labels]
            line_data = {l.split(':')[0]:pd.Series(d.get_xydata()[:,1], d.get_xydata()[:,0]) for l,d in zip(labels, lines)}
            line_data = {l: line_data[l][line_data[l]>=0] for l in line_data}
            r2_dict = {i: np.mean(abs((line_data[i]-line_data['Historical'])/line_data['Historical'])) for i in line_data if i!='Historical'}
            r2_series = pd.concat([pd.Series(r2_dict)], keys=[title.split(',')[0]])
            r2_series.name = title.split(', ')[1]
        r2_df = pd.concat([r2_df, r2_series],axis=1)
    r2 = r2_df.stack().unstack(level=1)
    r2_difference = r2.mul(-1).add(r2['Full'],axis=0)
    # fig,ax=plt.subplots(figsize=(7,10))
    # sns.heatmap(r2_difference, yticklabels=True, ax=ax)

    cov = cov_within.copy()
    # cov = means_coef_of_var.copy()#.rename(columns=many_sg.element_commodity_map)
    # cov.columns = [i.lower() for i in cov.columns]
    r2_diff_max = r2_difference.std(axis=1)
    new_corr = pd.DataFrame()
    for k in ['Mine production', 'Refined metal price', 'Total demand']:
        r2_diff_indiv = r2_diff_max.loc[k] # Mine production, Refined metal price, Total demand
        # many_sg.element_commodity_map
        correlation_matrix = pd.concat([
            cov,
            pd.DataFrame(r2_diff_indiv).T]).T.corr()
        new_corr_ph = correlation_matrix[0].drop(0)
        new_corr_ph.name = k
        new_corr = pd.concat([new_corr, new_corr_ph],axis=1)
        

    feature_importance(many_sg,plot=False)
    plt.close()
    ascending_importance = many_sg.importances_df['Best test R2'].sort_values().rename(
        make_parameter_names_nice(many_sg.importances_df.index))
    new_corr = new_corr.loc[ascending_importance.index]
    # fig, ax = plt.subplots(figsize=(8,12))
    # sns.heatmap(new_corr, ax=ax, annot=True, cmap='bwr')
    # ax.set(title='Correlation between R2 variance\nand CoV for parameters')

    annotate=True
    fig, ax = plt.subplots(figsize=(17,12))
    blanky = pd.DataFrame(means_coef_of_var['Ag']).rename(columns={'Ag':''})
    blanky.loc[:] = np.nan

    # adding R2 difference
    # renamer = dict([(i,many_sg.commodity_element_map[i.capitalize()]) for i in r2_diff_max.index.levels[1]])
    for_r2_diff = r2_diff_max.unstack(1)
    for_r2_diff = pd.concat([
        for_r2_diff,
        pd.DataFrame(np.nan, for_r2_diff.index,['']),
        for_r2_diff
    ],axis=1,keys=['CoV-train','','CoV-parameters'])
    quantile = for_r2_diff.droplevel(0,axis=1).stack().quantile(0.9)
    if use_r2_instead_of_mape:
        print(f'R2 difference divided by 90th percentile: {quantile}')
    else:
        print(f'MAPE difference divided by 90th percentile: {quantile}')
    for_r2_diff /= quantile
    for_r2_diff.loc[:,'All'] = np.nan
    for_r2_diff = pd.concat([
        pd.DataFrame(np.nan, [''], for_r2_diff.columns),
        for_r2_diff,
    ])
    for_r2_diff_nn = for_r2_diff*quantile

    # making CoV+Importance dataframe
    cov_importance_df = pd.concat([
            (means_coef_of_var-means_coef_of_var.min().min())/(means_coef_of_var-means_coef_of_var.min().min()).stack().quantile(0.95),
            blanky,
            (cov-cov.min().min())/(cov-cov.min().min()).stack().quantile(0.95),
        ],axis=1,keys=['CoV-train','','CoV-parameters'])

    cov_importance_df_nn = pd.concat([
            means_coef_of_var,
            blanky,
            cov,
        ],axis=1,keys=['CoV-train','','CoV-parameters'])

    cov_importance_df = cov_importance_df.rename({'Steel':'Fe'}).rename(columns={'Steel':'Fe'})
    cov_importance_df_nn = cov_importance_df_nn.rename({'Steel':'Fe'}).rename(columns={'Steel':'Fe'})
    for_r2_diff = for_r2_diff.rename({'Steel':'Fe'}).rename(columns={'Steel':'Fe'})
    for_r2_diff_nn = for_r2_diff_nn.rename({'Steel':'Fe'}).rename(columns={'Steel':'Fe'})

    # combining cov_importance with for_r2_diff
    cov_importance = pd.concat([
        cov_importance_df,
        for_r2_diff,
    ])
    cov_importance_nn = pd.concat([
        cov_importance_df_nn,
        for_r2_diff_nn,
    ])

    if 'Region specific intensity elasticity to price' in cov_importance.index:
        cov_importance = cov_importance.drop('Region specific intensity elasticity to price')
    if 'Region specific intensity elasticity to price' in cov_importance_nn.index:
        cov_importance_nn = cov_importance_nn.drop('Region specific intensity elasticity to price')
    
    
    sns.heatmap(
        cov_importance.drop(columns='All').droplevel(0,axis=1), 
        ax = ax,
        xticklabels=True,
        yticklabels=True,
        vmax=1,
        annot = None if not annotate else cov_importance_nn.drop(columns='All'),
        annot_kws = {'fontsize':14, },
        fmt='.2f',
        cmap='Reds',
        cbar_kws={'ticks':[], 'label':'Increasing value →'}
    )
    xlab = r'Relative $R^2$ change                                      Relative $R^2$ change   ' if use_r2_instead_of_mape else \
            'Relative MAPE change                                    Relative MAPE change'
    ax.set(title ='CoV - means across training sets              CoV - full tuning parameters    ',
        xlabel=xlab,
        )
    fig.set(dpi=150)
    if show:
        plt.show()
    plt.close()
    data = cov_importance.copy()
    r2_or_mape = 'r2' if use_r2_instead_of_mape else 'mape'
    fig.savefig(f'{many_sg.folder_path}/figures/figure_s30_cov_rel_{r2_or_mape}.pdf')
    fig.savefig(f'{many_sg.folder_path}/figures/figure_s30_cov_rel_{r2_or_mape}.png')
    data.to_csv(f'{many_sg.folder_path}/figures/figure_s30_cov_rel_{r2_or_mape}.csv')

def figure_s32(manies, show=False):
    """must be [many_sg, many_17, many_16, many_15] in that order"""
    many_sg, many_17, many_16, many_15 = manies
    many_names = ['2001-2019', '2001-2016', '2001-2015', '2001-2014']
    fig, ax=easy_subplots(manies, 2, use_subplots=True, sharey=True, height_scale=0.9)
    data = pd.DataFrame()
    for many,a,name in zip(manies, ax, many_names):
        cummin_score = many.rmse_df.loc[idx[:,'score'],:].cummin(axis=1).droplevel(1).T
        # cummin_score = many.rmse_df.loc[idx[:,'score'],:].droplevel(1).T.rolling(20).mean()
        # ind = many.rmse_df.loc[idx[:,'score'],:].min(axis=1).droplevel(1).sort_values(ascending=False).index
        # cummin_score = cummin_score.loc[:,ind]
        cummin_score = cummin_score.T.sort_index().T
        cummin_score = cummin_score.rename({'Steel':'Fe'}).rename(columns={'Steel':'Fe'})
        cummin_score.plot(ax=a, style=['-','--','-.',':','-','--','-.',':','--']).grid(axis='x')
        # ax.legend(loc=(1.05,0))
        a.legend(ncols=2)
        a.set(title=f'Learning curves\ntuning {name}', xlabel='Number of iterations', 
            ylabel='Cumulative min. training score')
        score = pd.concat([cummin_score],keys=[name])
        data = pd.concat([data,score])
    fig.tight_layout()
    fig.set_dpi(150)
    fig.savefig(f'{many_sg.folder_path}/figures/figure_s32_bo_learning_curves.pdf')
    fig.savefig(f'{many_sg.folder_path}/figures/figure_s32_bo_learning_curves.png')
    data.to_csv(f'{many_sg.folder_path}/figures/figure_s32_bo_learning_curves.csv')
    if show:
        plt.show()
    plt.close()
    
def figure_s33(many_sg, show=False):
    log = True
    static_lifetimes=pd.read_excel('input_files/static/Generalization drive file.xlsx',
                                sheet_name='Static lifetimes', index_col=0,header=[0,1]).loc[2019].unstack()
    static_lifetimes.loc[:,'Static lifetime'] = static_lifetimes['Reserves']/static_lifetimes['Mine production']
    static_lifetime = static_lifetimes['Static lifetime']
    opening_probs = many_sg.rmse_df_sorted.loc[idx[:,'incentive_opening_probability'],
                                            :24].astype(float).mean(axis=1).droplevel(1)
    price = many_sg.historical_data.loc[idx[:,2019],'Primary commodity price'].droplevel(1)
    both = pd.concat([static_lifetime, opening_probs, price],axis=1,
                    keys=['Static lifetime','Opening probability','Price'])

    def plot_two_params_commodity(both, x, y, ax=0, log=True, show=False, **plt_kwargs):
        if x=='Price':
            x = np.log10(both[x])
        else:
            x = both[x]
        if log:
            y = np.log10(both[y].astype(float))
        else:
            y = both[y]
        z = both.index
        
        colors = dict(zip(z,
            list(sns.color_palette('deep',n_colors=9))))

        if type(ax)==int:
            fig,a = plt.subplots()
        else:
            a = ax
        a.scatter(x,y,alpha=0)
    #     m = sm.GLS(y.astype(float), (x.astype(float))).fit(cov_type='HC3')
    #     a.plot(a.get_xlim(), [a.get_ylim()[0], a.get_xlim()[1]*m.coef[x.name]], color='k', linestyle='--')
    #     a.plot(a.get_xlim(), [a.get_ylim()[0], a.get_xlim()[1]*2.6], color='k', linestyle='--')
    #     a.plot(a.get_xlim(), a.get_xlim(), color='k', linestyle='--')
        
        for i,j,s in zip(x,y,z):
            a.annotate(str(s.replace('Steel','Fe')),  xy=(i, j), color=colors[s],
                        fontsize="large", weight='heavy',
                        horizontalalignment='center',
                        verticalalignment='center')
        initial_xlim, initial_ylim = a.get_xlim(), a.get_ylim()
        offset_fraction = 10
        offset_x = (initial_xlim[1]-initial_xlim[0])/offset_fraction
        offset_y = (initial_ylim[1]-initial_ylim[0])/offset_fraction
        a.set_xlim(initial_xlim[0]-offset_x, initial_xlim[1]+offset_x)
        a.set_ylim(initial_ylim[0]-offset_y, initial_ylim[1]+offset_y)
        a.set(**plt_kwargs)
        if log and y.name=='Static lifetime':
            yticks = a.get_yticks()
            unlogged_ticks = np.arange(15,91,15)
            a.set_yticks([np.log10(i) for i in unlogged_ticks])
            a.set_yticklabels(unlogged_ticks)
        if x.name=='Price':
            logged_ticks = np.arange(3,9,1)
            a.set_xticks(logged_ticks)
            a.set_xticklabels([f'$10^{{{i}}}$' for i in logged_ticks])
        if y.name=='Price':
            logged_ticks = np.arange(3,9,1)
            a.set_yticks(logged_ticks)
            a.set_yticklabels([f'$10^{{{i}}}$' for i in logged_ticks])
        
        
    fig,ax = easy_subplots(3,3)
    plt_kwargs = {'title':'Static lifetime vs\nfraction of viable mines that open', 
                'xlabel':'Fraction of viable mines that open', 'ylabel':'Static lifetime (years)'}
    plot_two_params_commodity(both, 'Opening probability', 'Static lifetime', ax[0], False, **plt_kwargs)

    plt_kwargs = {'title':'Price vs\nfraction of viable mines that open', 
                'xlabel':'Fraction of viable mines that open', 'ylabel':'Price (USD/t)'}
    plot_two_params_commodity(both, 'Opening probability', 'Price', ax[1], **plt_kwargs)

    plt_kwargs = {'title':'Static lifetime vs\nprice', 
                'xlabel':'Price (USD/t)', 'ylabel':'Static lifetime (years)'}
    plot_two_params_commodity(both, 'Price', 'Static lifetime', ax[2], False, **plt_kwargs)

    fig.tight_layout()
    fig.set_dpi(150)
    fig.savefig(f'{many_sg.folder_path}/figures/figure_s33_static_lifetime_fraction_price.pdf')
    fig.savefig(f'{many_sg.folder_path}/figures/figure_s33_static_lifetime_fraction_price.png')
    both.to_csv(f'{many_sg.folder_path}/figures/figure_s33_static_lifetime_fraction_price.csv')
    if show:
        plt.show()
    plt.close()

def figures_s34_and_s36(many, show=False):
    for which in ['s34','s36']:
        demand_bool=which=='s34' # demand_bool=true means plot demand, false means plot scrap collected
        years = np.arange(2001,2020)
        if demand_bool:
            demand_cols = [i for i in many.results.columns if 'demand, global' in i]
            demand = many.results_sorted.loc[idx[:,:24,:],demand_cols].groupby(level=[0,2]).mean()
            demand = demand.rename(columns=dict(zip(demand.columns,[i.split(' ')[0] for i in demand.columns])))
        else:
            demand_cols = [i for i in many.results.columns if 'Old scrap' in i and 'collection' not in i]
            demand = many.results_sorted.loc[idx[:,:24,:],demand_cols].groupby(level=[0,2]).mean()
            demand = demand.rename(columns=dict(zip(demand.columns,[i.split(' ')[-1].capitalize() for i in demand.columns])))

        comms = demand.index.get_level_values(0).unique()
        fig,ax=easy_subplots(comms)
        colors = dict(zip(
            ['Bar and coin','Construction','Electrical','Industrial','Jewelry','Transport','Other'],
            list(sns.color_palette('deep',n_colors=7))))

        # fig,a = plt.subplots()
        data = pd.DataFrame()
        for comm,a in zip(comms,ax):
            demand_ph = demand.copy().loc[comm].loc[years]
            if comm in ['Ag','Au']:
                demand_ph.rename(columns={'Transport':'Bar and coin','Industrial':'Jewelry'},inplace=True)
            dicty = get_unit(demand_ph,demand_ph,'Demand (kt)')
            demand_ph = dicty['simulated']
            unit = dicty['unit']
            stacks = a.stackplot(demand_ph.index, demand_ph.T, labels=demand_ph.columns,
                    colors=[colors[q] for q in demand_ph.columns])

            hatches=["\\", "//","+",'x','o']
            for stack, hatch in zip(stacks, hatches):
                stack.set_hatch(hatch)
            label = 'Demand' if demand_bool else 'Scrap collected'
            a.set(title=comm.replace('Steel','Fe'), xlabel='Year', ylabel=f'{label} ({unit})',
                )
            a.legend()
            h,l=a.get_legend_handles_labels()
            a.legend(h[::-1],l[::-1], loc=(0.1,0.45), frameon=True, framealpha=0.5)
            data = pd.concat([data, pd.concat([demand_ph],keys=[comm])])
        data = data.reset_index().rename(columns={'level_0':'Commodity', 'level_1':'Year'})
        # fig.set_dpi(150)
        fig.tight_layout()
        which_descriptor = 'sectoral_demand' if demand_bool else 'sectoral_old_scrap_collected'
        fig.savefig(f'{many.folder_path}/figures/figure_{which}_{which_descriptor}.pdf')
        fig.savefig(f'{many.folder_path}/figures/figure_{which}_{which_descriptor}.png')
        data.to_csv(f'{many.folder_path}/figures/figure_{which}_{which_descriptor}.csv')
        if show:
            plt.show()
        plt.close()

def figure_s35(many_sg, show=False):
    years = np.arange(2001,2020)
    demand_cols = [i for i in many_sg.results.columns if 'demand, global' in i]
    demand = many_sg.results_sorted.loc[idx[:,:24,:],demand_cols].groupby(level=[0,2]).mean()
    demand = demand.rename(columns=dict(zip(demand.columns,[i.split(' ')[0] for i in demand.columns])))

    demand_cols_cn = [i for i in many_sg.results.columns if 'demand, China' in i]
    demand_cn = many_sg.results_sorted.loc[idx[:,:24,:],demand_cols_cn].groupby(level=[0,2]).mean()
    demand_cn = demand_cn.rename(columns=dict(zip(demand_cn.columns,[i.split(' ')[0] for i in demand_cn.columns])))

    china_demand_fraction = demand_cn.sum(axis=1)/demand.sum(axis=1)
    china_demand_fraction = pd.DataFrame(china_demand_fraction)
    china_demand_fraction.loc[:,'Rest of World'] = 1-china_demand_fraction.iloc[:,0]
    china_demand_fraction.rename(columns={china_demand_fraction.columns[0]:'China'},inplace=True)
    comms = china_demand_fraction.index.get_level_values(0).unique()
    fig,ax=easy_subplots(comms)
    colors = dict(zip(
        ['China','Rest of World'],
        list(sns.color_palette('deep',n_colors=7))))

    # fig,a = plt.subplots()
    data = pd.DataFrame()
    for comm,a in zip(comms,ax):
        china_demand_fraction_ph = china_demand_fraction.copy().loc[comm].loc[years]
        stacks = a.stackplot(china_demand_fraction_ph.index, china_demand_fraction_ph.T, 
                            labels=china_demand_fraction_ph.columns,
                            colors=[colors[q] for q in china_demand_fraction_ph.columns])

        hatches=["\\", "//","+",'x','o']
        for stack, hatch in zip(stacks, hatches):
            stack.set_hatch(hatch)
        label = 'Demand distribution'
        a.set(title=comm.replace('Steel','Fe'), xlabel='Year', ylabel=label)
        a.legend()
        h,l=a.get_legend_handles_labels()
        a.legend(h[::-1],l[::-1], loc='upper left', frameon=True, framealpha=0.5)
        data = pd.concat([data, pd.concat([china_demand_fraction_ph],keys=[comm])])
    data = data.reset_index().rename(columns={'level_0':'Commodity', 'level_1':'Year'})
    # fig.set_dpi(150)
    fig.tight_layout()
    fig.savefig(f'{many_sg.folder_path}/figures/figure_s35_china_fraction_demand.pdf')
    fig.savefig(f'{many_sg.folder_path}/figures/figure_s35_china_fraction_demand.png')
    data.to_csv(f'{many_sg.folder_path}/figures/figure_s35_china_fraction_demand.csv')
    if show: 
        plt.show()
    plt.close()
    
def figures_s37_to_s40(many_sg, show=False):
    def plot_future_for_variable(many, commodity='copper', var='Mean total cash margin', n=50, 
                    color='#1b9e77', dpi=50, end_year=2040, ax=0):
        if type(ax)==int:
            fig, ax = plt.subplots()
        else: fig=0
        data = many.results_sorted[var].loc[commodity].loc[idx[:n,2001:end_year]]
        if 'cost' not in var and 'margin' not in var and 'grade' not in var and 'TCRC' not in var:
            _ = get_unit(data,data,var)
            data = _['simulated']
            label= _['unit']
        elif 'grade' in var:
            label = '%'
        else:
            label = 'USD/t'
        data = data.reset_index()\
            .rename(columns={'level_1':'Year'})
        sns.lineplot(data=data, x='Year',y=var, color=color, ax=ax, label='Simulated')
        title=var.replace('Spread','Scrap spread').replace('Refined price','Cathode price')\
                .replace('Mean ','').replace('mine grade','ore grade')
        if title!='TCRC': title=title.capitalize()
        ax.set(ylabel=var+f' ({label})', title=commodity.capitalize().replace('Steel','Fe'))
        return fig, ax, data

    def plot_actual_variables(many, variable='Mean mine grade', dpi=50, use_r2_instead_of_mape=False, show=False):
        """
        varible: str, options are: `Mean mine grade`, 
        `Mean total minesite cost`, `Mean total cash margin`, 
        `TCRC`
        """
        colors = ['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e','#e6ab02','#a6761d','#666666']
        commodities = many.rmse_df.index.get_level_values(0).unique()
        # mean_grade_actual = pd.read_excel('generalization/input_files/user_defined/case study data.xlsx',
        #                                   sheet_name='Ore grade', index_col=0)
        mean_grade_actual = pd.read_excel(
            'input_files/static/SP Global weighted mean mine parameters update.xlsx',
            index_col=0, header=[0,1])
        variable_map = {'Mean mine grade':'Grade (%)', 
                    'TCRC':'Inflation-adjusted TCRC (USD/t)',
                    'Mean total minesite cost':'Inflation-adjusted minesite cost (USD/t)', 
                    'Mean total cash margin':'Inflation-adjusted total cash margin (USD/t)'
                    }
        actual_variable = variable_map[variable]
        mean_grade_actual = mean_grade_actual.loc[:2019,idx[:,actual_variable]].droplevel(1,axis=1).dropna(how='all',axis=1)
        all_sim_data = pd.DataFrame()
        fig,ax = easy_subplots(commodities, dpi=dpi, height_scale=0.8)
        for comm,color,a in zip(commodities, np.tile(colors,(2)),ax):
            # comm='gold'
            ax, sim_data = plot_future_for_variable(many, var=variable, ax=a,
                                            color=color, dpi=50, n=25, commodity=comm, end_year=2019)[1:]
            if comm.capitalize() in mean_grade_actual.columns:
                ax.plot(mean_grade_actual.loc[2001:, comm.capitalize()], color='k', label='Historical')
                lines = a.get_lines()
                labels = [i.get_label() for i in lines]
                line_data = {l:pd.Series(d.get_xydata()[:,1], d.get_xydata()[:,0]) for l,d in zip(labels, lines)}
                if use_r2_instead_of_mape:
                    r2_or_mape = sm.GLS(line_data['Historical'], sm.add_constant(line_data['Simulated'])).fit(cov_type='HC3').rsquared
                    r2_or_mape_str = r'$R^2$'+'={:.2f}'.format(r2_or_mape)
                else:
                    s = line_data['Simulated']
                    h = line_data['Historical']
                    s,h = s[h>0], h[h>0]
                    r2_or_mape = np.median(abs((s-h)/h))*100
                    r2_or_mape_str = 'MAPE={:.1f}%'.format(r2_or_mape)
                h,l = ax.get_legend_handles_labels()
                h += [Line2D([0],[0], color='w', linewidth=0)]
                l += [r2_or_mape_str]
                ax.legend(handles=h, labels=l)
            else:
                ax.get_legend().remove()
            sim_data['Commodity'] = comm
            all_sim_data = pd.concat([all_sim_data, sim_data])
        all_sim_data.rename(columns={'level_0':'Scenario'}, inplace=True)
        all_sim_data['Historical or simulated'] = 'Simulated'
        hist_data = mean_grade_actual.stack().reset_index().rename(columns={'level_0':'Year','level_1':'Commodity',0:variable})
        hist_data['Historical or simulated'] = 'Historical'
        hist_data['Scenario'] = 'Historical'
        all_data = pd.concat([hist_data,all_sim_data])[['Year','Commodity','Historical or simulated','Scenario',variable]]\
            .reset_index(drop=True)
        fig.tight_layout()
        fig.set_dpi(150)
        if show: 
            plt.show()
        plt.close()
        return fig, all_data

    fig, data = plot_actual_variables(many_sg, variable='Mean mine grade')
    fig.savefig(f'{many_sg.folder_path}/figures/figure_s37_mean_ore_grade.pdf')
    fig.savefig(f'{many_sg.folder_path}/figures/figure_s37_mean_ore_grade.png')
    data.to_csv(f'{many_sg.folder_path}/figures/figure_s37_mean_ore_grade.csv')
    if show: 
        plt.show()
    plt.close()
    fig, data = plot_actual_variables(many_sg, variable='TCRC')
    fig.savefig(f'{many_sg.folder_path}/figures/figure_s38_mean_tcrc.pdf')
    fig.savefig(f'{many_sg.folder_path}/figures/figure_s38_mean_tcrc.png')
    data.to_csv(f'{many_sg.folder_path}/figures/figure_s38_mean_tcrc.csv')
    if show: 
        plt.show()
    plt.close()
    fig, data = plot_actual_variables(many_sg, variable='Mean total cash margin')
    fig.savefig(f'{many_sg.folder_path}/figures/figure_s39_mean_total_cash_margin.pdf')
    fig.savefig(f'{many_sg.folder_path}/figures/figure_s39_mean_total_cash_margin.png')
    data.to_csv(f'{many_sg.folder_path}/figures/figure_s39_mean_total_cash_margin.csv')
    if show: 
        plt.show()
    plt.close()
    fig, data = plot_actual_variables(many_sg, variable='Mean total minesite cost')
    fig.savefig(f'{many_sg.folder_path}/figures/figure_s40_mean_total_minesite_cost.pdf')
    fig.savefig(f'{many_sg.folder_path}/figures/figure_s40_mean_total_minesite_cost.png')
    data.to_csv(f'{many_sg.folder_path}/figures/figure_s40_mean_total_minesite_cost.csv')
    if show: 
        plt.show()
    plt.close()

def figure_s41(many_sg, show=False):
    if not hasattr(many_sg, 'colorful_means'):
        figure_4(many_sg)
    means = many_sg.colorful_means.copy()
    bhu_param = pd.DataFrame([0.343, 0.17, 0.262, 0.362, 0.244, 0.73, 0.035, 0.338, 0.39],
                             ['Al', 'Ag', 'Au', 'Cu', 'Ni', 'Pb', 'Sn', 'Steel', 'Zn'],
                             ['Recycling input rate'])
    bhu_param['Mine cost change per year'] = means['Mining production']['Mine cost change per year']
    bhu_param['Fraction of viable mines that open'] = means[
        'Reserve development']['Fraction of viable mines that open']

    bhu_param['Recycling input rate'] = pd.Series([0.343, 0.17, 0.262, 0.362, 0.244, 0.73, 0.035, 0.338, 0.39],
                                ['Al', 'Ag', 'Au', 'Cu', 'Ni', 'Pb', 'Sn', 'Steel', 'Zn'])
    bhu_param['Intensity elasticity to price'] = means['Demand response']['Intensity elasticity to price']
    bhu_param['Mean mine production growth/year'] = many_sg.historical_data.sort_index().loc[
        idx[:,2001:2019],'Primary supply'].groupby(level=0).pct_change().groupby(level=0).mean()
    # bhu_param['CAC slope eq.'] = means['Reserve development']['Incentive ore grade elasticity to COT']
    bhu_param_color = bhu_param.copy().subtract(bhu_param.min())
    bhu_param_color = bhu_param_color.div(bhu_param_color.max())
    fig,ax=plt.subplots(figsize=(14,4))

    bhu_param = bhu_param.rename({'Steel':'Fe'}).rename(columns={'Steel':'Fe'})
    bhu_param_color = bhu_param_color.rename({'Steel':'Fe'}).rename(columns={'Steel':'Fe'})
    sns.heatmap(bhu_param_color.T, annot=bhu_param.T, fmt='.2f',
                cbar_kws={'label':'Low            Med           High','ticks':[]},)
    ax.set_title(' ')
    fig.set_dpi(200)
    fig.tight_layout()
    fig.savefig(f'{many_sg.folder_path}/figures/figure_s41_decision_tree_assessment.pdf')
    fig.savefig(f'{many_sg.folder_path}/figures/figure_s41_decision_tree_assessment.png')
    bhu_param_color.T.to_csv(f'{many_sg.folder_path}/figures/figure_s41_decision_tree_assessment.csv')
    if show:
        plt.show()
    plt.close()

def table_s27(many_sg,  many_17, many_16, many_15, use_r2_instead_of_mape=False):
    # Regression table:
    if not hasattr(many_sg,'results_all') or True:
        results_all = run_r2_parameter_change_regressions(many_sg, many_17, many_16, many_15, use_r2_instead_of_mape=use_r2_instead_of_mape)
    else:
        results_all = many_sg.results_all
    results_all_3 = results_all.copy()
    table_stats = pd.DataFrame()
    for p in ['Total demand', 'Refined price','Mine production']:
        results = results_all_3.loc[p]
        min_frac = 0.4 if p=='Mine production' else 0.6
        if use_r2_instead_of_mape:
            m = results.loc[(results.frac_positive>0.4)&(results.f_pvalue<0.1)].sort_values(by='AIC').m.iloc[0]
        else:
            if (results.f_pvalue<0.1).sum()>0:
                m = results.loc[(results.frac_positive>0)&(results.f_pvalue<0.2)].sort_values(by='AIC').m.iloc[0]
            else:
                m = results.sort_values(by='AIC').m.iloc[0]
        out = pd.DataFrame('', index=[], columns=[0,1])
        out.loc['R-squared:',1] = round(m.rsquared,3)
        out.loc['Adj. R-squared:',1] = round(m.rsquared_adj,3)
        out.loc['F-statistic:',1] = round(m.fvalue,3)
        out.loc['Prob(F-statistic):',1] = round(m.f_pvalue,3)
        out.loc['No. Observations:',1] = m.nobs
        out.loc['Df Residuals:',1] = m.df_resid
        out.loc['Df Model:',1] = m.df_model
        out.loc['Covariance Type:',1] = m.cov_type
        out.loc['Parameter',0] = 'Value'
        out.loc['Parameter',1] = 'P>|z|'
        for x in m.params.index:
            out.loc[x.replace('const','Constant'),0]=round(m.params[x],3)
            out.loc[x.replace('const','Constant'),1]=round(m.pvalues[x],3)
        out = out.fillna('').reset_index(drop=False)
        out = pd.concat([out],keys=[p],axis=1)
        table_stats = pd.concat([table_stats,out],axis=1)
    table_stats = table_stats.fillna('').droplevel(1,axis=1)
    many_sg.table_stats = table_stats.copy()
    table_stats.to_csv(f'{many_sg.folder_path}/tables/table_s27_regression_results_table.csv')

def print_grade_change_per_year(primary_only):
    df = primary_only.loc[:,
                      idx[['Head Grade (%)','Ore Treated (kt)','Recovered Metal (kt)'],:]].stack().astype(float)
    cumu_ot = primary_only.loc[:,idx['Ore Treated (kt)',:]]
    initial_cumu_ot = ((1991 - primary_only['Actual Start Up Year'][
        primary_only['Actual Start Up Year'].astype(float)<1991][1991].astype(float)) * \
            primary_only['Ore Treated (kt)'][1991]).fillna(0)
    initial_cumu_ot = pd.concat([pd.DataFrame(initial_cumu_ot)],keys=['Ore Treated (kt)'],axis=1)
    cumu_ot.loc[:,initial_cumu_ot.columns] += initial_cumu_ot
    cumu_ot = cumu_ot.stack().astype(float).groupby(level=[0,1,2]).cumsum()
    df = pd.concat([df, cumu_ot.rename(columns={'Ore Treated (kt)':'Cumulative Ore Treated (kt)'})],axis=1)
    df['Year'] = df.index.get_level_values(3)
    df = df.loc[idx[:,:,:,:2019],:]
    df['grade-metal product'] = df['Head Grade (%)']*df['Recovered Metal (kt)']

    mean_grade = df['grade-metal product'].groupby(level=[1,3]).sum()/df[
        'Recovered Metal (kt)'].groupby(level=[1,3]).sum()
    mean_grade = mean_grade.unstack(level=0)
    mean_grade_norm = mean_grade.div(mean_grade.loc[2019])
    print('Ore grade change per year, weighted mean, simple:')
    display(mean_grade.pct_change().mean()*100)

    log=True
    regressors_list = [['Year'], ['Cumulative Ore Treated (kt)'], ['Year','Cumulative Ore Treated (kt)']]
    dep_var = 'Head Grade (%)'
    if log:
        log_vars = ['Head Grade (%)','Ore Treated (kt)','Recovered Metal (kt)','Cumulative Ore Treated (kt)']
        df[log_vars] = np.log(df[log_vars])
        df = df.rename(columns=dict(zip(log_vars,[f'log({k})' for k in log_vars])))
        dep_var = f'log({dep_var})'
        regressors_list = [ [f'log({j})' if j in log_vars else j for j in i] for i in regressors_list]
    re_models = {}
    fe_models = {}
    models = {}
    results = pd.DataFrame()
    regressors = regressors_list[0]
    commodity = 'Gold'
    for commodity in df.index.get_level_values(1).unique():
        if commodity != 'Palladium':
            df_ph = df.copy().loc[idx[:,commodity,:],:]
            df_ph.index = pd.MultiIndex.from_tuples([('-'.join(i[:3]),i[3]) for i in df_ph.index])
            for regr,regressors in enumerate(regressors_list):
                ph = df_ph.copy()
                ph = ph.dropna()
                re_cat = RandomEffects(ph[dep_var], sm.add_constant(ph[regressors])).fit(cov_type='robust')
                re_models[commodity+f' ({regr})'] = re_cat
                fe_cat = PanelOLS(ph[dep_var], sm.add_constant(ph[regressors])).fit(cov_type='robust') 
                fe_models[commodity+f' ({regr})'] = fe_cat

                hausman_p = hausman(fe_cat, re_cat)[2]
                if hausman_p < 0.05:
                    hausman_choice = fe_cat
                else:
                    hausman_choice = re_cat
                models[commodity+f' ({regr})'] = hausman_choice
                results_ph = pd.concat([hausman_choice.params, hausman_choice.pvalues],axis=1,
                                    keys=['Parameters','P-values'])
                results_ph['Degrees of Freedom'] = hausman_choice.df_resid
                if 'Year' in results_ph.index:
                    results_ph.loc['Year (% change correction)'] = results_ph.loc['Year']
                    results_ph.loc['Year (% change correction)','Parameters'] = \
                        np.exp(results_ph['Parameters']['Year'])-1
                if hausman_choice==re_cat:
                    results_ph['Hausman choice'] = 'Random Effects'
                else:
                    results_ph['Hausman choice'] = 'Fixed Effects'
                results_ph = pd.concat([results_ph],keys=[commodity+f' ({regr})'])
                results = pd.concat([results, results_ph])
    sub_models0 = dict([(i.split(' (')[0],models[i]) for i in sorted(models) if '0' in i])
    sub_models1 = dict([(i.split(' (')[0],models[i]) for i in sorted(models) if '1' in i])
    sub_models2 = dict([(i.split(' (')[0],models[i]) for i in sorted(models) if '2' in i])
    sub_results0 = results.loc[idx[[i for i in results.index.get_level_values(0) if '0' in i],:],:]\
        .rename(dict(zip(results.index.get_level_values(0),
                        [i.split(' (')[0] for i in results.index.get_level_values(0)]))).sort_index()
    sub_results1 = results.loc[idx[[i for i in results.index.get_level_values(0) if '1' in i],:],:]\
        .rename(dict(zip(results.index.get_level_values(0),
                        [i.split(' (')[0] for i in results.index.get_level_values(0)]))).sort_index()
    sub_results2 = results.loc[idx[[i for i in results.index.get_level_values(0) if '2' in i],:],:]\
        .rename(dict(zip(results.index.get_level_values(0),
                        [i.split(' (')[0] for i in results.index.get_level_values(0)]))).sort_index()

    compare(sub_models1, precision='pvalues')
    print('Ore grade change per year, panel model:')
    display(sub_results0.loc[idx[:,'Year (% change correction)'],:])
    print('Ore grade elasticity to COT:')
    display(sub_results1.loc[idx[:,'log(Cumulative Ore Treated (kt))'],:])

    print('Mean percent change per year for cumulative ore treated:')
    mean_change_cumu_ot = cumu_ot.groupby(level=[1,3]).sum().loc[sub_results0.index.levels[0]][
        'Ore Treated (kt)'].unstack(0).pct_change().mean()
    display(mean_change_cumu_ot)
