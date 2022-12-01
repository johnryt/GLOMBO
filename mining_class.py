from mining_functions import *
from useful_functions import *
import numpy as np
import pandas as pd

idx = pd.IndexSlice
from scipy import stats
from datetime import datetime
from copy import deepcopy, copy
from IPython.display import display
from random import sample, choices, seed

from sys import exit

class AllMines():
    def __init__(self):
        self.loc = dict()

    def add_mines(self, OneMine_instance):
        self.loc[OneMine_instance.name] = OneMine_instance

    def get_ml_df(self):
        pass

    def copy(self):
        return deepcopy([self])[0]

    def generate_df(self, columns=None, redo_strings=False):
        return pd.concat([self.loc[q].generate_df(columns=columns,redo_strings=redo_strings) for q in self.loc], keys=self.loc)


class OneMine():
    def __init__(self, name, df=None):
        self.name = name
        self.index_max = 0
        if type(df) != type(None):
            new_strs = [
                str(i).lower().replace(' ', '_').replace('(', '').replace(')', '').replace('$', 'usd').replace('/',
                                                                                                               'p').replace(
                    '%', 'pct') for i in df.columns]
            for i, j in zip(df.columns, new_strs):
                setattr(self, j, df[i].values)
            self.columns = new_strs
            self.index = np.array(df.index)
            self.index_max = np.max(self.index)
            self.columns = np.append(self.columns, ['index'])
        else:
            self.columns = np.array([])
            self.index = np.array([])

    def add_var(self, name, array):
        setattr(self, name, array)
        self.columns = np.append(self.columns, [name])

    def copy(self, use_deepcopy=False):
        if use_deepcopy:
            return deepcopy([self])[0]
        else:
            copy_ph = OneMine(name=self.name)
            copy_ph.columns = self.columns
            for k in self.columns:
                q = getattr(self,k)
                setattr(copy_ph, k, q.copy())
            return copy_ph

    def shape(self):
        return len(self.index), len(self.columns)

    def sample(self, size=1, resample=False, random_state=0):
        seed(random_state)
        if resample:
            return choices(self.index, k=size)
        return sample(self.index, k=size)

    def drop(self, ind, to_na=True):
        for i in self.columns:
            if len(np.shape(getattr(self, i))) > 0:
                if to_na:
                    ph = getattr(self, i)
                    try:
                        if i != 'index':
                            ph[ind] = np.nan
                    except ValueError:
                        ph = ph.astype(float)
                        ph[[i in ind for i in self.index]] = np.nan
                    setattr(self, i, ph)
                else:
                    setattr(self, i, np.delete(getattr(self, i), np.where([i in ind for i in self.index])))

    def concat(self, new_OM_instance):
        if self.index_max==0:
            self.index_max = np.max(self.index)
        len_new = len(new_OM_instance.index)
        nan_array = np.repeat(np.nan, len_new)
        new_OM_instance.index = np.arange(self.index_max + 1, self.index_max + 1 + len(new_OM_instance.index))
        for i in self.columns:
            ph = getattr(self, i)
            # v = len(ph)
            ph_type = ph.dtype
            if type(ph) == np.ndarray:
                if i in new_OM_instance.columns:
                    ph = np.append(ph, getattr(new_OM_instance, i)).astype(ph_type)
                else:
                    print(85, i)
                    ph = np.append(ph, nan_array).astype(ph_type)
                setattr(self, i, ph)
                # print(97, i, v, len(ph))
        self.index_max = np.max(self.index)

    def generate_df(self, columns=None, redo_strings=False):
        if columns is None: columns = self.columns
        out = pd.DataFrame(
            [getattr(self, q) if type(getattr(self, q)) == np.ndarray else np.repeat(getattr(self, q), len(self.index))
             for q in columns], columns=self.index, index=columns).T
        if redo_strings:
            new_strs = [
                str(i).capitalize().replace('_usdpt', ' (USD/t)').replace('_usdm',' ($M)').replace(
                    '_pct',' (%)').replace('_kt',' (kt)').replace('Oge','OGE').replace(
                    'Tcrc','TCRC').replace('tcrc','TCRC').replace('Npv','NPV').replace(
                    'npv','NPV').replace('treat_','treated_').replace('capex','CAPEX').replace(
                    'Capex','CAPEX').replace('Cu_','CU_').replace('_', ' ') for i in out.columns]
            out.rename(columns=dict(zip(out.columns,new_strs)),inplace=True)
        return out


class miningModel:
    """
    docstring
    """

    def __init__(self, simulation_time=np.arange(2019, 2041), hyperparam=0, byproduct=False, verbosity=0, price_change_yoy=0):
        """"""
        self.simulation_time = simulation_time
        self.byproduct = byproduct
        self.verbosity = verbosity
        self.initialize_hyperparams(hyperparam)
        self.hyperparam['primary_recovery_rate_shuffle_param'] = 1
        self.price_change_yoy = price_change_yoy
        self.i = self.simulation_time[0]
        self.rs = self.hyperparam['random_state']

        self.concentrate_supply_series = pd.Series(np.nan, self.simulation_time)
        self.sxew_supply_series = pd.Series(np.nan, self.simulation_time)

    def load_variables_from_hyperparam(self):
        #         self.mine_cu_margin_elas, self.mine_cost_og_elas, self.mine_cost_price_elas, self.mine_cu0, self.mine_tcm0, self.discount_rate, self.ramp_down_cu, self.ramp_up_cu, self.ramp_up_years = \
        #             self.hyperparam[['mine_cu_margin_elas','mine_cost_og_elas','mine_cost_price_elas',
        #                                  'mine_cu0','mine_tcm0','discount_rate','ramp_down_cu','ramp_up_cu','ramp_up_years']]
        #         self.rs = self.hyperparam['random_state']
        #         self.minesite_cost_response_to_grade_price = self.hyperparam['minesite_cost_response_to_grade_price']
        #         self.resources_contained_series = self.hyperparam['incentive_resources_contained_series']
        self.subsample_series = self.hyperparam['incentive_subsample_series']
        self.initial_subsample_series = self.subsample_series.copy()

    #         self.simulate_history_bool = self.hyperparam['simulate_history_bool']
    #         self.incentive_tuning_option = self.hyperparam['incentive_tuning_option']

    def plot_relevant_params(self, include=0, exclude=[], plot_recovery_grade_correlation=True, plot_minesite_supply_curve=True, plot_margin_supply_curve=True, plot_primary_minesite_supply_curve=False, plot_primary_margin_supply_curve=False, log_scale=False, dontplot=False, byproduct=False):
        """
        docstring
        """
        mines = self.mines.copy()
        if type(include) == int:
            cols = [i for i in mines.columns if i not in exclude and mines.dtypes[i] not in [object, str]]
        else:
            cols = [i for i in include if mines.dtypes[i] not in [object, str]]

        byprod_labels = {0: 'Primary', 1: 'Host 1', 2: 'Host 2', 3: 'Host 3'}
        if self.byproduct:
            mines['byprod int'] = mines.copy()['Byproduct ID']
            mines['Byproduct ID'] = mines['Byproduct ID'].replace(byprod_labels)

        fig, ax = easy_subplots(len(cols) + plot_recovery_grade_correlation + plot_minesite_supply_curve +
                                plot_margin_supply_curve + plot_primary_minesite_supply_curve + plot_primary_margin_supply_curve)
        for i, a in zip(cols, ax):
            colors = ['#d7191c', '#fdae61', '#abd9e9', '#2c7bb6']
            if self.byproduct and i not in ['Byproduct ID', 'Byproduct cash flow ($M)']:
                colors = [colors[int(i)] for i in mines['byprod int'].unique()]
                sns.histplot(mines, x=i, hue='Byproduct ID', palette=colors, bins=50, log_scale=log_scale, ax=a)
                a.set(title=i.replace('Primary', 'Host'))
            elif self.byproduct and i == 'Byproduct cash flow ($M)':
                colors = [colors[int(i)] for i in mines.dropna()['byprod int'].unique()]
                sns.histplot(mines.dropna(), x=i, hue='Byproduct ID', palette=colors, bins=50, log_scale=log_scale,
                             ax=a)
                a.set(title=i)
            else:
                mines[i].replace({'Primary': 0, 'Host 1': 1, 'Host 2': 2, 'Host 3': 3}).plot.hist(ax=a, title=i.replace(
                    'Primary', 'Host'), bins=50)
            if i == 'Recovery rate (%)' and self.hyperparam['primary_rr_negative']:
                a.text(0.05, 0.95, 'Reset to default,\nnegative values found.\nPrice and grade too low.',
                       va='top', ha='left', transform=a.transAxes)

        if plot_recovery_grade_correlation:
            a = ax[-(plot_recovery_grade_correlation + plot_minesite_supply_curve + plot_margin_supply_curve)]
            do_a_regress(mines['Head grade (%)'], mines['Recovery rate (%)'], ax=a)
            a.set(xlabel='Head grade (%)', ylabel='Recovery rate (%)',
                  title='Correlation with partial shuffle param\nValue: {:.2f}'.format(
                      self.hyperparam['primary_recovery_rate_shuffle_param']))

        if plot_minesite_supply_curve:
            a = ax[-(plot_minesite_supply_curve + plot_margin_supply_curve)]
            minn, maxx = mines['Minesite cost (USD/t)'].min(), mines['Minesite cost (USD/t)'].max()
            minn, maxx = min(minn, mines['Commodity price (USD/t)'].min()), max(maxx,
                                                                                mines['Commodity price (USD/t)'].max())
            ylim = (minn - (maxx - minn) / 10, maxx + (maxx - minn) / 10)
            supply_curve_plot(mines, 'Production (kt)', ['Total cash cost (USD/t)'],
                              price_col='Commodity price (USD/t)', width_scale=1.2, line_only=True, ax=a,
                              xlabel='Cumulative production (kt)',
                              ylim=ylim, ylabel='Total cash cost (USD/t)',
                              title='Total cash cost supply curve\nMean: {:.2f}'.format(
                                  mines['Total cash cost (USD/t)'].mean()),
                              byproduct=byproduct)

        if plot_margin_supply_curve:
            a = ax[-(plot_margin_supply_curve)]
            minn, maxx = mines['Total cash margin (USD/t)'].min(), mines['Total cash margin (USD/t)'].max()
            ylim = (minn - (maxx - minn) / 10, maxx + (maxx - minn) / 10)
            supply_curve_plot(mines, 'Production (kt)', ['Total cash margin (USD/t)'],
                              width_scale=1.2, line_only=True, ax=a, xlabel='Cumulative production (kt)',
                              ylim=ylim, ylabel='Total cash margin (USD/t)',
                              title='Cash margin supply curve\nMean: {:.2f}'.format(
                                  mines['Total cash margin (USD/t)'].mean()),
                              byproduct=byproduct)
            a.plot([0, mines['Production (kt)'].sum()], [0, 0], 'k')

        if plot_primary_minesite_supply_curve and self.byproduct:
            a = ax[-(plot_minesite_supply_curve + plot_margin_supply_curve + plot_primary_minesite_supply_curve)]
            minn, maxx = mines['Primary Minesite cost (USD/t)'].min(), mines['Primary Minesite cost (USD/t)'].max()
            minn, maxx = min(minn, mines['Primary Commodity price (USD/t)'].min()), max(maxx, mines[
                'Primary Commodity price (USD/t)'].max())
            supply_curve_plot(mines, 'Primary Production (kt)', ['Primary Total cash cost (USD/t)'],
                              price_col='Primary Commodity price (USD/t)', width_scale=1.2, line_only=True, ax=a,
                              xlabel='Cumulative production (kt)',
                              ylim=(minn - (maxx - minn) / 10, maxx + (maxx - minn) / 10),
                              ylabel='Total cash cost (USD/t)',
                              title='Host total cash cost supply curve\nMean: {:.2f}'.format(
                                  mines['Primary Total cash cost (USD/t)'].mean()),
                              byproduct=byproduct)

        if plot_primary_margin_supply_curve and self.byproduct:
            a = ax[-(
                        plot_minesite_supply_curve + plot_margin_supply_curve + plot_primary_minesite_supply_curve + plot_primary_margin_supply_curve)]
            minn, maxx = mines['Primary Total cash margin (USD/t)'].min(), mines[
                'Primary Total cash margin (USD/t)'].max()
            supply_curve_plot(mines, 'Primary Production (kt)', ['Primary Total cash margin (USD/t)'],
                              width_scale=1.2, line_only=True, ax=a, xlabel='Cumulative production (kt)',
                              ylim=(minn - (maxx - minn) / 10, maxx + (maxx - minn) / 10),
                              ylabel='Total cash margin (USD/t)',
                              title='Host cash margin supply curve\nMean: {:.2f}'.format(
                                  mines['Primary Total cash margin (USD/t)'].mean()),
                              byproduct=byproduct)
            a.plot([0, mines['Primary Production (kt)'].sum()], [0, 0], 'k')

        fig.tight_layout()
        if dontplot:
            plt.close(fig)
        return fig

    def initialize_hyperparams(self, hyperparam):
        '''
        for many of the default parameters and their origins, see
        https://countertop.mit.edu:3048/notebooks/SQL/Bauxite-aluminum%20mines.ipynb
        and/or Displacement/04 Presentations/John/Weekly Updates/20210825 Generalization.pptx
        '''
        if type(hyperparam) == int:
            hyperparameters = {}
            hyperparameter_notes = {}

            if 'parameters for operating mine pool generation, mass':
                hyperparameters['verbosity'] = self.verbosity
                hyperparameters['byproduct'] = self.byproduct
                hyperparameters['primary_production'] = 1  # kt
                hyperparameters['primary_production_mean'] = 0.001  # kt
                hyperparameters['primary_production_var'] = 1
                hyperparameters['primary_production_distribution'] = 'lognorm'
                hyperparameters['primary_production_fraction'] = 1
                hyperparameters['primary_ore_grade_mean'] = 0.01
                hyperparameters['primary_ore_grade_var'] = 0.3
                hyperparameters['primary_ore_grade_distribution'] = 'lognorm'
                hyperparameters['primary_cu_mean'] = 0.85
                hyperparameters['primary_cu_var'] = 0.06
                hyperparameters['primary_cu_distribution'] = 'lognorm'
                hyperparameters['primary_payable_percent_mean'] = 0.63
                hyperparameters['primary_payable_percent_var'] = 1.83
                hyperparameters['primary_payable_percent_distribution'] = 'weibull_min'
                hyperparameters['primary_rr_default_mean'] = 13.996
                hyperparameters['primary_rr_default_var'] = 0.675
                hyperparameters['primary_rr_default_distribution'] = 'lognorm'
                hyperparameters['primary_ot_cumu_mean'] = 14.0018
                hyperparameters['primary_ot_cumu_var'] = 0.661
                hyperparameters['primary_ot_cumu_distribution'] = 'lognorm'

                hyperparameters['primary_rr_alpha'] = 39.2887
                hyperparameters['primary_rr_beta'] = 5.0898
                hyperparameters['primary_rr_gamma'] = 4.9559
                hyperparameters['primary_rr_delta'] = 21.8916
                hyperparameters['primary_rr_epsilon'] = -2.4569
                hyperparameters['primary_rr_theta'] = -17.4767
                hyperparameters['primary_rr_eta'] = 0.6320
                hyperparameters['primary_rr_rho'] = -10.7094
                hyperparameters['primary_rr_negative'] = False

                hyperparameters['primary_recovery_rate_var'] = 0.6056  # default value of 0.6056 comes from the mean across all materials in snl
                hyperparameters['primary_recovery_rate_distribution'] = 'lognorm'
                hyperparameters['primary_recovery_rate_shuffle_param'] = 0.4
                hyperparameters['primary_reserves_mean'] = 11.04953  # these values are from https://countertop.mit.edu:3048/notebooks/SQL/Second%20round%20generalization%20mine%20parameters.ipynb
                hyperparameters['primary_reserves_var'] = 0.902357  # use the ratio between reserves and ore treated in each year, finding lognormal distribution
                hyperparameters['primary_reserves_distribution'] = 'lognorm'
                hyperparameters['primary_reserves_reported'] = 30
                hyperparameters['primary_reserves_reported_basis'] = 'none'  # ore, metal, or none basis - ore: mass of ore reported as reserves (SNL style), metal: metal content of reserves reported, none: use the generated values without adjustment

                hyperparameters['production_frac_region1'] = 0.2
                hyperparameters['production_frac_region2'] = 0.2
                hyperparameters['production_frac_region3'] = 0.2
                hyperparameters['production_frac_region4'] = 0.2
                hyperparameters['production_frac_region5'] = 1 - np.sum([hyperparameters[i] for i in ['production_frac_region1','production_frac_region2','production_frac_region3','production_frac_region4']])

            if 'parameters for operating mine pool generation, cost':
                hyperparameters['primary_commodity_price'] = 6000  # USD/t
                hyperparameters['primary_minesite_cost_mean'] = 0
                hyperparameters['primary_minesite_cost_var'] = 1
                hyperparameters['primary_minesite_cost_distribution'] = 'lognorm'

                hyperparameters['minetype_prod_frac_underground'] = 0.3
                hyperparameters['minetype_prod_frac_openpit'] = 0.7
                hyperparameters['minetype_prod_frac_tailings'] = 0
                hyperparameters['minetype_prod_frac_stockpile'] = 0
                hyperparameters['minetype_prod_frac_placer'] = 1 - np.sum([hyperparameters[i] for i in ['minetype_prod_frac_underground','minetype_prod_frac_openpit','minetype_prod_frac_tailings','minetype_prod_frac_stockpile']])

                hyperparameters['primary_minerisk_mean'] = 9.4  # values for copper → ranges from 4 to 20
                hyperparameters['primary_minerisk_var'] = 1.35
                hyperparameters['primary_minerisk_distribution'] = 'norm'

                hyperparameters['primary_minesite_cost_regression2use'] = 'linear_113_price_tcm_sx'  # options: linear_107, bayesian_108, linear_110_price, linear_111_price_tcm, linear_112_price_tcm_sx, linear_112_price_tcm_sx; not used if primary_minesite_cost_mean>0
                hyperparameters['primary_tcm_flag'] = 'tcm' in hyperparameters['primary_minesite_cost_regression2use']
                hyperparameters['primary_tcrc_regression2use'] = 'linear_114'  # options: linear_114, linear_114_reftype
                hyperparameters['primary_tcrc_dore_flag'] = False  # determines whether the refining process is that for dore or for concentrate
                hyperparameters['primary_sxew_fraction'] = 0  # fraction of primary production coming from sxew mines
                hyperparameters['primary_sxew_fraction_change'] = 0  # change per year in the fraction of primary production coming from sxew mines
                hyperparameters['primary_sxew_fraction_series'] = pd.Series(0, self.simulation_time)
                hyperparameter_notes['primary_sxew_fraction_series'] = 'default primary sxew_fraction_series series'

                hyperparameters['primary_scapex_regression2use'] = 'linear_123_norm'  # options: linear_119_cap_sx, linear_117_price_cap_sx, linear_123_norm
                hyperparameters['primary_dcapex_regression2use'] = 'linear_124_norm'
                hyperparameter_notes['primary_dcapex_regression2use'] = 'options: linear_124_norm. See 20210825 Generalization.pptx, slide 124'

                hyperparameters['primary_reclamation_constant'] = 1.321
                hyperparameter_notes['primary_reclamation_constant'] = 'for use in np.exp(1.321+0.671*np.log(mines_cor_adj[Capacity (kt)]))'
                hyperparameters['primary_reclamation_slope'] = 0.671
                hyperparameter_notes['primary_reclamation_slope'] = 'for use in np.exp(1.321+0.671*np.log(mines_cor_adj[Capacity (kt)]))'
                hyperparameters['primary_overhead_regression2use'] = 'linear_194'
                hyperparameter_notes['primary_overhead_regression2use'] = 'options are linear_194, which uses values from 20210825 Generalization.pptx slide 194 or None, which gives a constant value for overhead at all mines of 0.1, in $M'
                hyperparameters = self.add_minesite_cost_regression_params(hyperparameters)

            if 'parameters for mine life simulation':
                hyperparameters['commodity'] = 'notAu'
                hyperparameters['primary_oge_s'] = 0.3320346
                hyperparameters['primary_oge_loc'] = 0
                hyperparameters['primary_oge_scale'] = 0.399365

                hyperparameters['mine_cu_margin_elas'] = 0.01
                hyperparameters['mine_cost_og_elas'] = -0.113
                hyperparameters['mine_cost_change_per_year'] = 0.5
                hyperparameter_notes['mine_cost_change_per_year'] = 'Percent (%) change in mine cost reductions per year, default 0.5%'
                hyperparameters['mine_cost_price_elas'] = 0
                hyperparameters['mine_cu0'] = 0.7688729808870376
                hyperparameters['mine_tcm0'] = 14.575211987093567
                hyperparameters['ramp_up_fraction'] = 0.02
                hyperparameter_notes['ramp_up_fraction'] = 'fraction of mines in the initial mine generation step that are in any of the ramp up stages (e.g. if ramp_up_year is 3 and ramp_up_fraction is 0.1, then 10% of the mines will have ramp up flag=1, 10% ramp up flag=2, etc.). Value is currently 0.02 based on an initial guess.'

                hyperparameters['discount_rate'] = 0.10
                hyperparameters['cu_cutoff'] = 1.1
                hyperparameter_notes['cu_cutoff'] = 'highest allowable capacity utilization'
                hyperparameters['ramp_down_cu'] = 0.4
                hyperparameters['ramp_up_cu'] = 0.4  # currently replacing this s.t. linear ramp up to 100% instead
                hyperparameters['ramp_up_years'] = 3
                hyperparameters['ramp_up_exponent'] = 1
                hyperparameters['byproduct_ramp_down_rr'] = 0.4
                hyperparameters['byproduct_ramp_up_rr'] = 0.4
                hyperparameters['byproduct_ramp_up_years'] = 1
                hyperparameters['close_price_method'] = 'probabilistic'
                hyperparameters['close_years_back'] = 3
                hyperparameters['close_probability_split_max'] = 0.3
                hyperparameters['close_probability_split_mean'] = 0.5
                hyperparameters['close_probability_split_min'] = 0.2

                hyperparameters['reinitialize'] = True
                hyperparameter_notes['reinitialize'] = 'bool, True runs the setup fn initialize_mine_life instead of pulling from init_mine_life.pkl'
                hyperparameters['load_mine_life_init_from_pkl'] = False
                hyperparameter_notes['load_mine_life_init_from_pkl'] = 'self explanatory'
                hyperparameters['minesite_cost_response_to_grade_price'] = True
                hyperparameter_notes['minesite_cost_response_to_grade_price'] = 'bool, True,minesite costs respond to ore grade decline as per slide 10 here: Group Research Folder_Olivetti/Displacement/04 Presentations/John/Weekly Updates/20210825 Generalization.pptx'
                hyperparameters['use_reserves_for_closure'] = False
                hyperparameter_notes['use_reserves_for_closure'] = 'bool, True forces mines to close when they run out of reserves, False allows otherwise. Should always use False'
                hyperparameters['forever_sim'] = False
                hyperparameter_notes['forever_sim'] = 'bool, if True allows the simulation to run until all mines have closed (or bauxite price series runs out of values), False only goes until set point'
                hyperparameters['simulate_closure'] = True
                hyperparameter_notes['simulate_closure'] = 'bool, whether to simulate 2019 operating mines and their closure, default Truebut can be set to False to test mine opening.'
                hyperparameters['simulate_opening'] = True
                hyperparameter_notes['simulate_opening'] = 'bool, whether to simulate new mine opening, default True but set False during mine opening evaluation so we dont end up in an infinite loop'
                hyperparameters['reinitialize_incentive_mines'] = False
                hyperparameter_notes['reinitialize_incentive_mines'] = 'bool, default False and True is not set up yet. Whether to create a new  incentive pool of mines or to use the pre-generated one, passing True requires supplying incentive_mine_hyperparameters, which can be accessed by calling self.output_incentive_mine_hyperparameters()'
                hyperparameters['continuous_incentive'] = False
                hyperparameter_notes['continuous_incentive'] = 'bool, if True, maintains the same set of incentive pool mines the entire time, dropping them from the incentive pool and adding them to the operating pool as they open. Hopefully this will eventually also include adding new mines to the incentive as reserves expand. If False, does not drop & samples from incentive pool each time; recommend changing incentive_mine_hyperparameters and setting reinitialize_incentive_mines=True if that is the case. Current default is False'
                hyperparameters['follow_copper_opening_method'] = True
                hyperparameter_notes['follow_copper_opening_method'] = 'bool, if True, generates an incentive pool for each year of the simulation, creates alterable subsample_series to track how many from the pool are sampled in each year'
                hyperparameters['calibrate_copper_opening_method'] = True
                hyperparameter_notes['calibrate_copper_opening_method'] = 'bool, should be False once the subsample_series is set up (sets up the subsample series for mine opening evaluation.'

                hyperparameters['primary_commodity_price_option'] = 'constant'
                hyperparameter_notes['primary_commodity_price_option'] = 'str, how commodity prices are meant to evolve. Options: constant, yoy, step, input. Input requires setting the variable primary_price_series after model initialization'
                hyperparameters['byproduct_commodity_price_option'] = 'constant'
                hyperparameter_notes['byproduct_commodity_price_option'] = 'str, how commodity prices are meant to evolve. Options: constant, yoy, step, input. Input requires setting the variable byproduct_price_series after model initialization'
                hyperparameters['primary_commodity_price_change'] = 10
                hyperparameter_notes['primary_commodity_price_change'] = 'percentage value, percent change in commodity price year-over-year (yoy) or in its one-year step.'
                hyperparameters['byproduct_commodity_price_change'] = 10
                hyperparameter_notes['byproduct_commodity_price_change'] = 'percentage value, percent change in commodity price year-over-year (yoy) or in its one-year step.'
                #             hyperparameters['',['Value','Notes']] = np.array([],dtype='object')
                #             hyperparameters['',['Value','Notes']] = np.array([],dtype='object')
                #             hyperparameters['',['Value','Notes']] = np.array([],dtype='object')

                hyperparameters['random_state'] = 20220208

            if 'parameters for byproducts' and self.byproduct:
                if 'parameters for byproduct production and grade':
                    hyperparameters['byproduct_pri_production_fraction'] = 0.1
                    hyperparameters['byproduct_host3_production_fraction'] = 0
                    hyperparameters['byproduct_host2_production_fraction'] = 0.4
                    hyperparameters['byproduct_host1_production_fraction'] = 1 - np.sum([hyperparameters[i] for i in ['byproduct_pri_production_fraction','byproduct_host3_production_fraction','byproduct_host2_production_fraction']])

                    hyperparameters['byproduct0_rr0'] = 80
                    hyperparameters['byproduct1_rr0'] = 80
                    hyperparameters['byproduct1_mine_rrmax'] = 90
                    hyperparameters['byproduct1_mine_tcm0'] = 25
                    hyperparameters['byproduct2_rr0'] = 85
                    hyperparameters['byproduct2_mine_rrmax'] = 95
                    hyperparameters['byproduct2_mine_tcm0'] = 25
                    hyperparameters['byproduct3_rr0'] = 80
                    hyperparameters['byproduct3_mine_rrmax'] = 95
                    hyperparameters['byproduct3_mine_tcm0'] = 25
                    hyperparameters['byproduct_rr_margin_elas'] = 0.01
                    hyperparameter_notes['byproduct0_rr0'] = 'byproduct median recovery rate at simulation start'
                    hyperparameter_notes['byproduct1_rr0'] = 'byproduct median recovery rate at simulation start'
                    hyperparameter_notes['byproduct1_mine_rrmax'] = 'byproduct median recovery rate at simulation start'
                    hyperparameter_notes['byproduct1_mine_tcm0'] = 'byproduct median total cash margin at simulation start'
                    hyperparameter_notes['byproduct2_rr0'] = 'byproduct median recovery rate at simulation start'
                    hyperparameter_notes['byproduct2_mine_rrmax'] = 'byproduct median recovery rate at simulation start'
                    hyperparameter_notes['byproduct2_mine_tcm0'] = 'byproduct median total cash margin at simulation start'
                    hyperparameter_notes['byproduct3_rr0'] = 'byproduct median recovery rate at simulation start'
                    hyperparameter_notes['byproduct3_mine_rrmax'] = 'byproduct median recovery rate at simulation start'
                    hyperparameter_notes['byproduct3_mine_tcm0'] = 'byproduct median total cash margin at simulation start'
                    hyperparameter_notes['byproduct_rr_margin_elas'] = 'byproduct recovery rate elasticity to total cash margin'

                    hyperparameters['byproduct_production'] = 4  # kt
                    hyperparameters['byproduct_production_mean'] = 0.03  # kt
                    hyperparameters['byproduct_production_var'] = 0.5
                    hyperparameters['byproduct_production_distribution'] = 'lognorm'

                    hyperparameters['byproduct_host1_grade_ratio_mean'] = 20
                    hyperparameters['byproduct_host2_grade_ratio_mean'] = 2
                    hyperparameters['byproduct_host3_grade_ratio_mean'] = 10
                    hyperparameters['byproduct_host1_grade_ratio_var'] = 1
                    hyperparameters['byproduct_host2_grade_ratio_var'] = 1
                    hyperparameters['byproduct_host3_grade_ratio_var'] = 1
                    hyperparameters['byproduct_host1_grade_ratio_distribution'] = 'norm'
                    hyperparameters['byproduct_host2_grade_ratio_distribution'] = 'norm'
                    hyperparameters['byproduct_host3_grade_ratio_distribution'] = 'norm'

                    hyperparameters['byproduct_pri_ore_grade_mean'] = 0.1
                    hyperparameters['byproduct_host1_ore_grade_mean'] = 0.1
                    hyperparameters['byproduct_host2_ore_grade_mean'] = 0.1
                    hyperparameters['byproduct_host3_ore_grade_mean'] = 0.1
                    hyperparameters['byproduct_pri_ore_grade_var'] = 0.3
                    hyperparameters['byproduct_host1_ore_grade_var'] = 0.3
                    hyperparameters['byproduct_host2_ore_grade_var'] = 0.3
                    hyperparameters['byproduct_host3_ore_grade_var'] = 0.3
                    hyperparameters['byproduct_pri_ore_grade_distribution'] = 'lognorm'
                    hyperparameters['byproduct_host1_ore_grade_distribution'] = 'lognorm'
                    hyperparameters['byproduct_host2_ore_grade_distribution'] = 'lognorm'
                    hyperparameters['byproduct_host3_ore_grade_distribution'] = 'lognorm'

                    hyperparameters['byproduct_pri_sxew_fraction'] = 0.5
                    hyperparameters['byproduct_host1_sxew_fraction'] = 0.2
                    hyperparameters['byproduct_host2_sxew_fraction'] = 0.5
                    hyperparameters['byproduct_host3_sxew_fraction'] = 0.5

                    hyperparameters['byproduct_pri_cu_mean'] = 0.85
                    hyperparameters['byproduct_host1_cu_mean'] = 0.85
                    hyperparameters['byproduct_host2_cu_mean'] = 0.85
                    hyperparameters['byproduct_host3_cu_mean'] = 0.85
                    hyperparameters['byproduct_pri_cu_var'] = 0.06
                    hyperparameters['byproduct_host1_cu_var'] = 0.06
                    hyperparameters['byproduct_host2_cu_var'] = 0.06
                    hyperparameters['byproduct_host3_cu_var'] = 0.06
                    hyperparameters['byproduct_pri_cu_distribution'] = 'lognorm'
                    hyperparameters['byproduct_host1_cu_distribution'] = 'lognorm'
                    hyperparameters['byproduct_host2_cu_distribution'] = 'lognorm'
                    hyperparameters['byproduct_host3_cu_distribution'] = 'lognorm'

                    hyperparameters['byproduct_pri_payable_percent_mean'] = 0.63
                    hyperparameters['byproduct_host1_payable_percent_mean'] = 0.63
                    hyperparameters['byproduct_host2_payable_percent_mean'] = 0.63
                    hyperparameters['byproduct_host3_payable_percent_mean'] = 0.63
                    hyperparameters['byproduct_pri_payable_percent_var'] = 1.83
                    hyperparameters['byproduct_host1_payable_percent_var'] = 1.83
                    hyperparameters['byproduct_host2_payable_percent_var'] = 1.83
                    hyperparameters['byproduct_host3_payable_percent_var'] = 1.83
                    hyperparameters['byproduct_pri_payable_percent_distribution'] = 'weibull_min'
                    hyperparameters['byproduct_host1_payable_percent_distribution'] = 'weibull_min'
                    hyperparameters['byproduct_host2_payable_percent_distribution'] = 'weibull_min'
                    hyperparameters['byproduct_host3_payable_percent_distribution'] = 'weibull_min'

                    hyperparameters['byproduct_pri_rr_default_mean'] = 13.996
                    hyperparameters['byproduct_host1_rr_default_mean'] = 13.996
                    hyperparameters['byproduct_host2_rr_default_mean'] = 13.996
                    hyperparameters['byproduct_host3_rr_default_mean'] = 13.996
                    hyperparameters['byproduct_pri_rr_default_var'] = 0.675
                    hyperparameters['byproduct_host1_rr_default_var'] = 0.675
                    hyperparameters['byproduct_host2_rr_default_var'] = 0.675
                    hyperparameters['byproduct_host3_rr_default_var'] = 0.675
                    hyperparameters['byproduct_pri_rr_default_distribution'] = 'lognorm'
                    hyperparameters['byproduct_host1_rr_default_distribution'] = 'lognorm'
                    hyperparameters['byproduct_host2_rr_default_distribution'] = 'lognorm'
                    hyperparameters['byproduct_host3_rr_default_distribution'] = 'lognorm'

                if 'byproduct costs':
                    hyperparameters['byproduct_commodity_price'] = 1000  # USD/t
                    hyperparameters['byproduct_minesite_cost_mean'] = 0  # USD/t
                    hyperparameters['byproduct_minesite_cost_var'] = 1
                    hyperparameters['byproduct_minesite_cost_distribution'] = 'lognorm'

                    hyperparameters['byproduct_host1_commodity_price'] = 2000
                    hyperparameters['byproduct_host2_commodity_price'] = 3000
                    hyperparameters['byproduct_host3_commodity_price'] = 1000

                    hyperparameters['byproduct_host1_minesite_cost_ratio_mean'] = 20
                    hyperparameters['byproduct_host2_minesite_cost_ratio_mean'] = 2
                    hyperparameters['byproduct_host3_minesite_cost_ratio_mean'] = 10
                    hyperparameters['byproduct_host1_minesite_cost_ratio_var'] = 1
                    hyperparameters['byproduct_host2_minesite_cost_ratio_var'] = 1
                    hyperparameters['byproduct_host3_minesite_cost_ratio_var'] = 1
                    hyperparameters['byproduct_host1_minesite_cost_ratio_distribution'] = 'norm'
                    hyperparameters['byproduct_host2_minesite_cost_ratio_distribution'] = 'norm'
                    hyperparameters['byproduct_host3_minesite_cost_ratio_distribution'] = 'norm'

                    hyperparameters['byproduct_host1_sus_capex_ratio_mean'] = 20
                    hyperparameters['byproduct_host2_sus_capex_ratio_mean'] = 2
                    hyperparameters['byproduct_host3_sus_capex_ratio_mean'] = 10
                    hyperparameters['byproduct_host1_sus_capex_ratio_var'] = 1
                    hyperparameters['byproduct_host2_sus_capex_ratio_var'] = 1
                    hyperparameters['byproduct_host3_sus_capex_ratio_var'] = 1
                    hyperparameters['byproduct_host1_sus_capex_ratio_distribution'] = 'norm'
                    hyperparameters['byproduct_host2_sus_capex_ratio_distribution'] = 'norm'
                    hyperparameters['byproduct_host3_sus_capex_ratio_distribution'] = 'norm'

                    hyperparameters['byproduct_host1_tcrc_ratio_mean'] = 20
                    hyperparameters['byproduct_host2_tcrc_ratio_mean'] = 2
                    hyperparameters['byproduct_host3_tcrc_ratio_mean'] = 10
                    hyperparameters['byproduct_host1_tcrc_ratio_var'] = 1
                    hyperparameters['byproduct_host2_tcrc_ratio_var'] = 1
                    hyperparameters['byproduct_host3_tcrc_ratio_var'] = 1
                    hyperparameters['byproduct_host1_tcrc_ratio_distribution'] = 'norm'
                    hyperparameters['byproduct_host2_tcrc_ratio_distribution'] = 'norm'
                    hyperparameters['byproduct_host3_tcrc_ratio_distribution'] = 'norm'

                    hyperparameters['byproduct_pri_minerisk_mean'] = 9.4  # value for copper → ranges from 4 to 20, cutoffs are enforced
                    hyperparameters['byproduct_host1_minerisk_mean'] = 9.4
                    hyperparameters['byproduct_host2_minerisk_mean'] = 9.4
                    hyperparameters['byproduct_host3_minerisk_mean'] = 9.4
                    hyperparameters['byproduct_pri_minerisk_var'] = 1.35
                    hyperparameters['byproduct_host1_minerisk_var'] = 1.35
                    hyperparameters['byproduct_host2_minerisk_var'] = 1.35
                    hyperparameters['byproduct_host3_minerisk_var'] = 1.35
                    hyperparameters['byproduct_pri_minerisk_distribution'] = 'norm'
                    hyperparameters['byproduct_host1_minerisk_distribution'] = 'norm'
                    hyperparameters['byproduct_host2_minerisk_distribution'] = 'norm'
                    hyperparameters['byproduct_host3_minerisk_distribution'] = 'norm'

            if 'parameters for the incentive pool':
                hyperparameters['simulate_opening'] = True
                hyperparameter_notes['simulate_opening'] = 'whether or not to simulate mine opening'
                hyperparameters['opening_flag_for_cu0'] = False
                hyperparameter_notes['opening_flag_for_cu0'] = 'whether or not opening is currently happening and we have to suppress the recalculation of cu0'
                hyperparameters['incentive_subsample_init'] = 1000
                hyperparameter_notes['incentive_subsample_init'] = 'initial value for incentive_subsample_series'

                hyperparameters['annual_reserves_ratio_with_initial_production_const'] = 1.1
                hyperparameter_notes['annual_reserves_ratio_with_initial_production_const'] = 'multiplies by annual demand to determine initial reserves used for incentive pool size'
                hyperparameters['annual_reserves_ratio_with_initial_production_slope'] = 0
                hyperparameter_notes['annual_reserves_ratio_with_initial_production_slope'] = 'linear change per year of annual demand to determine initial reserves used for incentive pool size, in fraction form (would input 0.1 for 10% increase per year)'
                hyperparameters['incentive_resources_contained_series'] = pd.Series(hyperparameters['primary_production'] * hyperparameters['annual_reserves_ratio_with_initial_production_const'], self.simulation_time)
                hyperparameter_notes['incentive_resources_contained_series'] = 'the contained metal in the assumed resources, currently set to be a series with each year having contained metal in resources equal to one year of production'

                hyperparameters['incentive_use_resources_contained_series'] = True
                hyperparameter_notes['incentive_use_resources_contained_series'] = 'whether to use the incentive_resources_contained_series series for determining incentive pool size or to use the incentive_subsample_series series for the number of mines in each year'
                hyperparameters['primary_price_resources_contained_elas'] = 0.5
                hyperparameter_notes['primary_price_resources_contained_elas'] = 'percent increase in the resources conatined/incentive pool size when price rises by 1%'
                hyperparameters['byproduct_price_resources_contained_elas'] = 0.05
                hyperparameter_notes['byproduct_price_resources_contained_elas'] = 'percent increase in the resources conatined/incentive pool size when price rises by 1%'
                hyperparameters['reserves_ratio_price_lag'] = 5
                hyperparameter_notes['reserves_ratio_price_lag'] = 'lag on price change price(t-lag)/price(t-lag-1) used for informing incentive pool size change, paired with resources_contained_elas_primary_price (and byproduct if byproduct==True)'
                hyperparameters['incentive_subsample_series'] = pd.Series(hyperparameters['incentive_subsample_init'], self.simulation_time)
                hyperparameter_notes['incentive_subsample_series'] = 'series with number of mines to select for subsample used for the incentive pool'
                hyperparameters['incentive_perturbation_percent'] = 10
                hyperparameter_notes['incentive_perturbation_percent'] = 'percent perturbation for the incentive pool parameters, on resampling from the operating mine pool'

                hyperparameters['incentive_roi_years'] = 10
                hyperparameter_notes['incentive_roi_years'] = 'Number of years of mine life to simulate to determine profitability for opening mines'
                hyperparameters['incentive_opening_method'] = 'unconstrained'
                hyperparameter_notes['incentive_opening_method'] = 'options: xinkai_thesis, karan_generalization, unconstrained; xinkai_thesis is the tuning of the incentive subsample size to match demand, while karan_generalization is tuning of price to fill the supply-demand gap. unconstrained allows mine opening to occur unimpeded and TCRC evolves based on TCRC elasticity to SD imbalance to close the gap.'
                hyperparameters['calibrate_incentive_opening_method'] = False
                hyperparameter_notes['calibrate_incentive_opening_method'] = 'whether to calibrate to minimize the demand gap if using the xinkai_thesis incentive_opening_method'
                hyperparameters['demand_series_method'] = 'yoy'
                hyperparameter_notes['demand_series_method'] = 'Options: yoy or target; yoy is year-over-year percent change in demand, target is the percent change in demand by simulation end.'
                hyperparameters['demand_series_pct_change'] = 0
                hyperparameter_notes['demand_series_pct_change'] =5, 'Percent change per year if demand_series_method is yoy, or percent change from initial to last simulation year if demand_series_method is target'
                hyperparameters['demand_series'] = 1
                hyperparameter_notes['demand_series'] = 'generated from demand_series_method and demand_series_pct_change in the update_operation_hyperparams function'
                hyperparameters['initial_ore_grade_decline'] = -0.05
                hyperparameter_notes['initial_ore_grade_decline'] = 'Initial ore grade for new mines, elasticity to cumulative ore treated'
                hyperparameters['incentive_mine_cost_change_per_year'] = hyperparameters['mine_cost_change_per_year']
                hyperparameter_notes['incentive_mine_cost_change_per_year'] = 'rate of cost decline year-over-year for the incentive pool'

                hyperparameters['incentive_tune_tcrc'] = True
                hyperparameter_notes['incentive_tune_tcrc'] = 'True means that in the karan_generalization method of tuning opening, we tune TCRC. False means we tune Commodity price.'
                hyperparameters['incentive_discount_rate'] = 0.1
                hyperparameter_notes['incentive_discount_rate'] = 'fraction from 0-1, discount rate applied to opening mines for NPV calculation, effectively return on investment minimum required for opening.'
                hyperparameters['incentive_opening_probability'] = 0
                hyperparameter_notes['incentive_opening_probability'] = 'fraction from 0-1, fraction of profitable mines that are capable of opening that actually undergo opening. Used to reduce TCRC values in incentive_open_karan_generalization(). Also used in unconstrained opening; if set to zero, allows this value to be set by the first n years of operation, otherwise it will be this value.'
                hyperparameters['incentive_require_tune_years'] = 0
                hyperparameter_notes['incentive_require_tune_years'] = 'requires incentive tuning for however many years such that supply=demand, with no requirements on incentive_opening_probability and allowing the given incentive_opening_probability to be used'
                hyperparameters['end_calibrate_years'] = 10
                hyperparameter_notes['end_calibrate_years'] = 'how many years after simulation start time to perform frac calibration in incentive_open_xinkai_thesis, unconstrained'
                hyperparameters['start_calibrate_years'] = 4
                hyperparameter_notes['start_calibrate_years'] = 'how many years after simulation start time to perform frac calibration in incentive_open_xinkai_thesis, unconstrained'

                hyperparameters['reserve_frac_region1'] = 0.19743337
                hyperparameters['reserve_frac_region2'] = 0.08555446
                hyperparameters['reserve_frac_region3'] = 0.03290556
                hyperparameters['reserve_frac_region4'] = 0.24350115
                hyperparameters['reserve_frac_region5'] = 0.44060546

                hyperparameters['internal_price_formation'] = False
                hyperparameter_notes['internal_price_formation'] = 'whether to use the supply-demand balance to determine price evolution internally or to use external inputs'
                hyperparameters['tcrc_sd_elas'] = 0.1
                hyperparameter_notes['tcrc_sd_elas'] = 'TCRC elasticity to supply divided by demand'
                hyperparameters['price_sd_elas'] = -0.1
                hyperparameter_notes['price_sd_elas'] = 'Commodity price elasticity to supply divided by demand'
                hyperparameters['simulate_history_bool'] = False
                hyperparameter_notes['simulate_history_bool'] = 'Simulate some years beforehand, for particular use with incentive_opening_method==unconstrained, such that TCRC or commodity price can have a chance to equilibrate. The number of years needed to be simulated beforehand is determined by the demand_series_pct_change, as values close to zero need more years'
                hyperparameters['incentive_tuning_option'] = 'pid'
                hyperparameter_notes['incentive_tuning_option'] = 'options: pid or elas. PID uses proportion-integral-derivative tuning to reach out initial TCRC/commodity price value, primarily for simulating history. elas uses the simple elasticity approach from tcrc_sd_elas or price_sd_elas'

                hyperparameters['use_ml_to_accelerate'] = False
                hyperparameter_notes['use_ml_to_accelerate'] = 'if True, tries using a machine learning model to predict which mines will open, using the real opening method once every several years so the model gets updated'
                hyperparameters['ml_accelerate_initialize_years'] = 20
                hyperparameter_notes['ml_accelerate_initialize_years'] = 'Begin using the ML model after this many years have run, remembering that the first 10 years may be used for tuning to get the average fraction of profitable mines that open each year'
                hyperparameters['ml_accelerate_every_n_years'] = 2
                hyperparameter_notes['ml_accelerate_every_n_years'] = 'Use the actual opening process every n years; use the ML process for the rest. Begins after the initial_years are finished'

            if 'adding notes':
                # operating pool mass values
                hyperparameter_notes['primary_production'] = 'Total mine production in whatever units we are using, metal content of the primary (host) commodity'
                hyperparameter_notes['primary_production_mean'] = 'Mean mine production in whatever units we are using, metal content of the primary (host) commodity'
                hyperparameter_notes['primary_production_var'] = 'Variance of the mine production in whatever units we are using, metal content of the primary (host) commodity'
                hyperparameter_notes['primary_production_distribution'] = 'valid stats.distribution distribution name, e.g. lognorm, norm, etc. Should be something with loc and scale parameters'
                hyperparameter_notes['primary_ore_grade_mean'] = 'mean ore grade (%) for the primary commodity'
                hyperparameter_notes['primary_ore_grade_var'] = 'ore grade variance (%) for the primary commodity'
                hyperparameter_notes['primary_ore_grade_distribution'] = 'distribution used for primary commodity ore grade, default lognormal'
                hyperparameter_notes['primary_cu_mean'] = 'mean capacity utilization of the primary mine (likely not necessary to consider this as just the primary, or at least secondary total CU would be the product of this and the secondary CU)'
                hyperparameter_notes['primary_cu_var'] = 'capacity utilization of primary mine variance'
                hyperparameter_notes['primary_cu_distribution'] = 'distirbution for primary mine capacity utilization, default lognormal'
                hyperparameter_notes['primary_payable_percent_mean'] = 'mean for 100-(payable percent): 0.63 value from https://countertop.mit.edu:3048/notebooks/SQL/Mining%20database%20read.ipynb, section 12.3 Payable Percent, for weibull minimum distribution of 100-(payable percent)'
                hyperparameter_notes['primary_payable_percent_var'] = 'variance for 100-(payable percent): 1.83 value from https://countertop.mit.edu:3048/notebooks/SQL/Mining%20database%20read.ipynb, section 12.3 Payable Percent, for weibull minimum distribution of 100-(payable percent)'
                hyperparameter_notes['primary_payable_percent_distribution'] = 'distribution for 100-(payable percent): from https://countertop.mit.edu:3048/notebooks/SQL/Mining%20database%20read.ipynb, section 12.3 Payable Percent, for weibull minimum distribution of 100-(payable percent)'
                hyperparameter_notes['primary_rr_grade_corr_slope'] = 'slope used for calculating log(100-recovery rate) from log(head grade), referencing slide 41 of 20210825 Generalization.pptx and section 10 Recovery rate, https://countertop.mit.edu:3048/notebooks/SQL/Second%20round%20generalization%20mine%20parameters.ipynb'
                hyperparameter_notes['primary_rr_grade_corr_slope'] = 'constant used for calculating log(100-recovery rate) from log(head grade), referencing slide 41 of 20210825 Generalization.pptx and section 10 Recovery rate, https://countertop.mit.edu:3048/notebooks/SQL/Second%20round%20generalization%20mine%20parameters.ipynb'
                hyperparameter_notes['primary_recovery_rate_mean'] = 'mean for 100-(recovery rate): lognormal distribution, using the mean of head grade to calculate the mean recovery rate value with constant standard deviation (average from each material), referencing slide 41 of 20210825 Generalization.pptx and section 10 Recovery rate (https://countertop.mit.edu:3048/notebooks/SQL/Mining%20database%20read.ipynb)as a function of ore grade. '
                hyperparameter_notes['primary_recovery_rate_var'] = 'variance for 100-(recovery rate): lognormal distribution, using the mean of head grade to calculate the mean recovery rate value with constant standard deviation (average from each material), referencing slide 41 of 20210825 Generalization.pptx and section 10 Recovery rate (https://countertop.mit.edu:3048/notebooks/SQL/Mining%20database%20read.ipynb) as a function of ore grade. '
                hyperparameter_notes['primary_recovery_rate_distribution'] = 'distribution for (100-recovery rate): lognormal distribution, using the mean of head grade to calculate the mean recovery rate value with constant standard deviation (average from each material), referencing slide 41 of 20210825 Generalization.pptx and section 10 Recovery rate (https://countertop.mit.edu:3048/notebooks/SQL/Mining%20database%20read.ipynb) as a function of ore grade. '
                hyperparameter_notes['primary_recovery_rate_shuffle_param'] = 'recovery rates are ordered to match the order of grade to retain correlation, then shuffled so the correlation is not perfect. This parameter (called part) passed to the partial shuffle function for the correlation between recovery rate and head grade; higher value = more shuffling'
                hyperparameter_notes['primary_reserves_mean'] = 'mean for primary reserves divided by ore treated. 11.04953 value from https://countertop.mit.edu:3048/notebooks/SQL/Second%20round%20generalization%20mine%20parameters.ipynb, use the ratio between reserves and ore treated in each year, finding lognormal distribution'
                hyperparameter_notes['primary_reserves_var'] = 'variance for primary reserves divided by ore treated. 0.902357 value from https://countertop.mit.edu:3048/notebooks/SQL/Second%20round%20generalization%20mine%20parameters.ipynb, use the ratio between reserves and ore treated in each year, finding lognormal distribution'
                hyperparameter_notes['primary_reserves_distribution'] = 'distribution for primary reserves divided by ore treated. lognormal values from https://countertop.mit.edu:3048/notebooks/SQL/Second%20round%20generalization%20mine%20parameters.ipynb, use the ratio between reserves and ore treated in each year, finding lognormal distribution'
                hyperparameter_notes['primary_reserves_reported'] = 'reserves reported for that year; tunes the operating mine pool such that its total reserves matches this value. Can be given in terms of metal content or total ore available, just have to adjust the primary_reserves_reported_basis variable in this df. Setting it to zero allows the generated value to be used.'
                hyperparameter_notes['primary_reserves_reported_basis'] = 'ore, metal, or none basis - ore: mass of ore reported as reserves (SNL style), metal: metal content of reserves reported, none: use the generated values without adjustment'

                hyperparameter_notes['production_frac_region1'] = 'region fraction of global production in 2019'
                hyperparameter_notes['production_frac_region2'] = 'region fraction of global production in 2019'
                hyperparameter_notes['production_frac_region3'] = 'region fraction of global production in 2019'
                hyperparameter_notes['production_frac_region4'] = 'region fraction of global production in 2019'
                hyperparameter_notes[
                    'production_frac_region5'] = 'region fraction of global production in 2019, calculated from the remainder of the other 4'

                # Prices
                hyperparameter_notes['primary_minesite_cost_mean'] = 'mean minesite cost for the primary commodity. Set to zero to use mine type, risk, and ore grade to generate the cost distribution instead'
                hyperparameter_notes['primary_minesite_cost_var'] = 'minesite cost variance for the primary commodity. Is not used if the mean value is set to zero'
                hyperparameter_notes['primary_minesite_cost_distribution'] = 'minesite cost distribution type (e.g. lognormal) for the primary commodity. Is not used if the mean value is set to zero'

                hyperparameter_notes['minetype_prod_frac_underground'] = 'fraction of mines using mine type underground; not used if primary_minesite_cost_mean is nonzero'
                hyperparameter_notes['minetype_prod_frac_openpit'] = 'fraction of mines using mine type openpit; not used if primary_minesite_cost_mean is nonzero'
                hyperparameter_notes['minetype_prod_frac_tailings'] = 'fraction of mines using mine type tailings; not used if primary_minesite_cost_mean is nonzero'
                hyperparameter_notes['minetype_prod_frac_stockpile'] = 'fraction of mines using mine type stockpile; not used if primary_minesite_cost_mean is nonzero'
                hyperparameter_notes['minetype_prod_frac_placer'] = 'fraction of mines using mine type placer; not used if primary_minesite_cost_mean is nonzero'

                hyperparameter_notes['minerisk_mean'] = 'mean value. Risk indicator is the sum of political, operational, terrorism, and security risks, which range from insignificant (1) to extreme (5). Therefore the minimum value is 4 and the maximum value is 20. Not used if primary_minesite_cost_mean is nonzero'
                hyperparameter_notes['minerisk_var'] = 'variance. Risk indicator is the sum of political, operational, terrorism, and security risks, which range from insignificant (1) to extreme (5). Therefore the minimum value is 4 and the maximum value is 20. Not used if primary_minesite_cost_mean is nonzero'
                hyperparameter_notes['minerisk_distribution'] = 'distribution, assumed normal. Risk indicator is the sum of political, operational, terrorism, and security risks, which range from insignificant (1) to extreme (5). Therefore the minimum value is 4 and the maximum value is 20. Not used if primary_minesite_cost_mean is nonzero'

                hyperparameter_notes['primary_minesite_cost_regression2use'] = 'options: linear_107, bayesian_108, linear_110_price, linear_111_price_tcm; not used if primary_minesite_cost_mean>0. First component (linear/bayesian) references regression type, number references slide in Displacement\04 Presentations\John\Weekly Updates\20210825 Generalization.pptx. Inclusion of \'price\' at the end indicates to use the regression that included commodity price. Inclusion of \'tcm\' indicates the regression was performed on total cash margin excl tcrc rather than minesite cost, and determines the primary_minesite_cost_flag'
                hyperparameter_notes['primary_minesite_cost_flag'] = 'True sets to generate minesite costs and total cash margin using the regressions on minesite cost; False sets to generate the same using the regressions on total cash margin (excl TCRC). Based on whether primary_minesite_cost_regression2use contains str tcm.'
                hyperparameter_notes['primary_scapex_slope_capacity'] = 'slope from regression of sustaining CAPEX on capacity, from 04 Presentations\John\Weekly Updates\20210825 Generalization.pptx, slide 34'
                hyperparameter_notes['primary_scapex_constant_capacity'] = 'constant from regression of sustaining CAPEX on capacity, from 04 Presentations\John\Weekly Updates\20210825 Generalization.pptx, slide 34'

                # Simulation
                hyperparameter_notes['primary_oge_s'] = 'parameters to generate (1-OGE) for lognormal distribution, found from looking at all mines in https://countertop.mit.edu:3048/notebooks/SQL/Mining%20database%20read.ipynb, section 12.2, where it says \'parameters used for generalization\'.'
                hyperparameter_notes['primary_oge_loc'] = 'parameters to generate (1-OGE) for lognormal distribution, found from looking at all mines in https://countertop.mit.edu:3048/notebooks/SQL/Mining%20database%20read.ipynb, section 12.2, where it says \'parameters used for generalization\'.'
                hyperparameter_notes['primary_oge_scale'] = 'parameters to generate (1-OGE) for lognormal distribution, found from looking at all mines in https://countertop.mit.edu:3048/notebooks/SQL/Mining%20database%20read.ipynb, section 12.2, where it says \'parameters used for generalization\'.'

                hyperparameter_notes['mine_cu_margin_elas'] = 'capacity utlization elasticity to total cash margin, current value is approximate result of regression attempts; see slide 42 in Group Research Folder_Olivetti\Displacement\04 Presentations\John\Weekly Updates\20210825 Generalization.pptx',
                hyperparameter_notes['mine_cost_og_elas'] = 'minesite cost elasticity to ore grade decline'
                hyperparameter_notes['mine_cost_price_elas'] = 'minesite cost elasticity to commodity price'
                hyperparameter_notes['mine_cu0'] = 'median capacity utlization in 2019, used to determine how mines change CU due to TCM'
                hyperparameter_notes['mine_tcm0'] = 'median total cash margin in 2019, used to determine how mines change CU due to TCM'
                hyperparameter_notes['discount_rate'] = 'discount rate (fraction, not percent - 0.1=10%), used for NPV/IRR calculation in mine opening decision'
                hyperparameter_notes['ramp_down_cu'] = 'capacity utilization during ramp down; float less than 1, default 0.4'
                hyperparameter_notes['ramp_up_cu'] = 'capacity utilization during ramp up, float less than 1, default 0.4'
                hyperparameter_notes['ramp_up_years'] = 'number of years allotted for ramp up (currently use total dCAPEX distributed among those years, so shortening would make each year of dCAPEX more expensive), int default 3'
                hyperparameter_notes['close_price_method'] = 'method used for price expected used for mine closing - mean, max, alonso-ayuso, or probabilistic are supported - if using probabilistic you can adjust the close_probability_split variables'
                hyperparameter_notes['close_years_back'] = 'number of years to use for rolling mean/max/min values when evaluating mine closing'
                hyperparameter_notes['close_probability_split_max'] = 'for the probabilistic closing method, probability given to the rolling close_years_back max'
                hyperparameter_notes['close_probability_split_mean'] = 'for the probabilistic closing method, probability given to the rolling close_years_back mean'
                hyperparameter_notes['close_probability_split_min'] = 'for the probabilistic closing method, probability given to the rolling close_years_back min - make sure these three sum to 1'
                hyperparameter_notes['random_state'] = 'random state int for sampling'
                hyperparameter_notes['reserve_frac_region1'] = 'region reserve fraction of global total in 2019'
                hyperparameter_notes['reserve_frac_region2'] = 'region reserve fraction of global total in 2019'
                hyperparameter_notes['reserve_frac_region3'] = 'region reserve fraction of global total in 2019'
                hyperparameter_notes['reserve_frac_region4'] = 'region reserve fraction of global total in 2019'
                hyperparameter_notes['reserve_frac_region5'] = 'region reserve fraction of global total in 2019'

            initial_demand = hyperparameters['primary_production']
            change = hyperparameters['demand_series_pct_change'] / 100 + 1
            sim_time = self.simulation_time
            if hyperparameters['demand_series_method'] == 'yoy':
                hyperparameters['demand_series'] = pd.Series([initial_demand * change ** (j - sim_time[0]) for j in sim_time], sim_time)
            elif hyperparameters['demand_series_method'] == 'target':
                hyperparameters['demand_series'] = pd.Series(np.linspace(initial_demand, initial_demand * change, len(sim_time)), sim_time)
            self.demand_series = hyperparameters['demand_series']

            incent_series = [i * (hyperparameters['annual_reserves_ratio_with_initial_production_const'] + j)
                             for i, j in zip(self.demand_series.values,
                                             [hyperparameters[
                                                  'annual_reserves_ratio_with_initial_production_slope'] * (
                                                          y - self.simulation_time[0]) for y in self.simulation_time])]
            hyperparameters['incentive_resources_contained_series'] = pd.Series(incent_series, self.simulation_time)
            hyperparameter_notes['incentive_resources_contained_series'] = 'the contained metal in the assumed resources, currently set to be a series with each year having contained metal in resources equal to one year of production'
            hyperparameters['incentive_subsample_series'] = pd.Series(hyperparameters['incentive_subsample_init'], self.simulation_time)
            hyperparameter_notes['incentive_subsample_series'] = 'series with number of mines to select for subsample used for the incentive pool'
            self.resources_contained_series = hyperparameters['incentive_resources_contained_series']
            self.subsample_series = hyperparameters['incentive_subsample_series']

            self.hyperparam = hyperparameters
            self.hyperparam_notes = hyperparameter_notes

        elif type(hyperparam) == str:
            if hyperparam.split('.')[-1] == 'pkl':
                self.hyperparam = pd.read_pickle(hyperparam)
            elif hyperparam.split('.')[-1] in ['xlsx', 'xls']:
                self.hyperparam = pd.read_excel(hyperparam, index_col=0)
        elif type(hyperparam) == pd.core.frame.DataFrame:
            self.hyperparam = hyperparam
        else:
            raise ValueError(
                'Mining module initialization failed, incorrect hyperparam input (should be integer/zero to allow self-generation, string for filepath (excel or pickle file), or dataframe if it already exists)')
        self.mine_cu_margin_elas, self.mine_cost_og_elas, self.mine_cost_price_elas, self.mine_cu0, self.mine_tcm0, self.discount_rate, self.ramp_down_cu, self.ramp_up_cu, self.ramp_up_years = \
            [self.hyperparam[i] for i in ['mine_cu_margin_elas', 'mine_cost_og_elas', 'mine_cost_price_elas',
                                      'mine_cu0', 'mine_tcm0', 'discount_rate', 'ramp_down_cu', 'ramp_up_cu',
                                      'ramp_up_years']]

    def update_operation_hyperparams(self, innie=0):
        hyperparameters = self.hyperparam
        if type(innie) == int:
            mines = self.mines.copy()
            if not hyperparameters['opening_flag_for_cu0']:
                hyperparameters['mine_cu0'] = mines['Capacity utilization'].median()
                hyperparameters['mine_tcm0'] = mines['Total cash margin (USD/t)'].median()
            if self.byproduct:
                byp = 'Byproduct ' if 'Byproduct Total cash margin (USD/t)' in mines.columns else ''
                for i in np.arange(1, 4):
                    hyperparameters['byproduct' + str(i) + '_rr0'] = mines.loc[
                        (mines['Byproduct ID'] == i) & (
                                    mines[byp + 'Total cash margin (USD/t)'] > 0), 'Recovery rate (%)'].median()
                    hyperparameters['byproduct' + str(i) + '_mine_tcm0'] = mines.loc[
                        (mines['Byproduct ID'] == i) & (mines[
                                                            byp + 'Total cash margin (USD/t)'] > 0), byp + 'Total cash margin (USD/t)'].median()
        else:
            mines = innie.copy()
            if not hyperparameters['opening_flag_for_cu0']:
                hyperparameters['mine_cu0'] = np.median(mines.capacity_utilization)
                hyperparameters['mine_tcm0'] = np.median(mines.total_cash_margin_usdpt)
            if self.byproduct:
                byp = 'byproduct_' if 'byproduct_total_cash_margin_usdpt' in mines.columns else ''
                for i in np.arange(1, 4):
                    hyperparameters['byproduct' + str(i) + '_rr0'] = np.median(mines.recovery_rate_pct[(
                                                                                                                                mines.byproduct_id == i) & (
                                                                                                                                getattr(
                                                                                                                                    mines,
                                                                                                                                    byp + 'total_cash_margin_usdpt') > 0)])
                    hyperparameters['byproduct' + str(i) + '_mine_tcm0'] = np.median(
                        getattr(mines, byp + 'total_cash_margin_usdpt')[
                            (mines['Byproduct ID'] == i) & (getattr(mines, byp + 'total_cash_margin_usdpt') > 0)])

        # if hyperparameters['commodity']!='Au':
        #     hyperparameters['annual_reserves_ratio_with_initial_production_const','Value'] = 8
        self.hyperparam = hyperparameters.copy()
        if (hyperparameters['primary_sxew_fraction_series'] == 0).all():
            self.sxew_fraction_series = pd.Series(
                hyperparameters['primary_sxew_fraction'] + hyperparameters[
                    'primary_sxew_fraction_change'] * (self.simulation_time - self.i), self.simulation_time)
            self.sxew_fraction_series.loc[self.sxew_fraction_series < 0] = 0
            self.sxew_fraction_series.loc[self.sxew_fraction_series > 1] = 1

    def add_minesite_cost_regression_params(self, hyperparameters_):
        hyperparameters = hyperparameters_.copy()
        reg2use = hyperparameters['primary_minesite_cost_regression2use']
        #             log(minesite cost) = alpha + beta*log(commodity price) + gamma*log(head grade)
        #                + delta*(numerical risk) + epsilon*placer (mine type)
        #                + theta*stockpile + eta*tailings + rho*underground + zeta*sxew

        if reg2use == 'linear_107':  # see slide 107 or 110 left-hand-side table in C:\Users\ryter\Dropbox (MIT)\Group Research Folder_Olivetti\Displacement\04 Presentations\John\Weekly Updates\20210825 Generalization.pptx
            hyperparameters['primary_minesite_cost_alpha'] = 7.4083
            hyperparameters['primary_minesite_cost_beta'] = 0
            hyperparameters['primary_minesite_cost_gamma'] = -1.033
            hyperparameters['primary_minesite_cost_delta'] = 0.0173
            hyperparameters['primary_minesite_cost_epsilon'] = -1.5532
            hyperparameters['primary_minesite_cost_theta'] = 0.5164
            hyperparameters['primary_minesite_cost_eta'] = -0.8997
            hyperparameters['primary_minesite_cost_rho'] = 0.7629
            hyperparameters['primary_minesite_cost_zeta'] = 0
        elif reg2use == 'bayesian_108':  # see slide 108 in C:\Users\ryter\Dropbox (MIT)\Group Research Folder_Olivetti\Displacement\04 Presentations\John\Weekly Updates\20210825 Generalization.pptx
            hyperparameters['primary_minesite_cost_alpha'] = 10.4893
            hyperparameters['primary_minesite_cost_beta'] = 0
            hyperparameters['primary_minesite_cost_gamma'] = -0.547
            hyperparameters['primary_minesite_cost_delta'] = 0.121
            hyperparameters['primary_minesite_cost_epsilon'] = -0.5466
            hyperparameters['primary_minesite_cost_theta'] = -0.5837
            hyperparameters['primary_minesite_cost_eta'] = -0.9168
            hyperparameters['primary_minesite_cost_rho'] = 1.4692
            hyperparameters['primary_minesite_cost_zeta'] = 0
        elif reg2use == 'linear_110_price':  # see slide 110 right-hand-side table in C:\Users\ryter\Dropbox (MIT)\Group Research Folder_Olivetti\Displacement\04 Presentations\John\Weekly Updates\20210825 Generalization.pptx
            hyperparameters['primary_minesite_cost_alpha'] = 0.5236
            hyperparameters['primary_minesite_cost_beta'] = 0.8453
            hyperparameters['primary_minesite_cost_gamma'] = -0.1932
            hyperparameters['primary_minesite_cost_delta'] = -0.015
            hyperparameters['primary_minesite_cost_epsilon'] = 0
            hyperparameters['primary_minesite_cost_theta'] = 0.2122
            hyperparameters['primary_minesite_cost_eta'] = -0.3076
            hyperparameters['primary_minesite_cost_rho'] = 0.1097
            hyperparameters['primary_minesite_cost_zeta'] = 0
        elif reg2use == 'linear_111_price_tcm':  # see slide 111 right-hand-side table in C:\Users\ryter\Dropbox (MIT)\Group Research Folder_Olivetti\Displacement\04 Presentations\John\Weekly Updates\20210825 Generalization.pptx
            hyperparameters['primary_minesite_cost_alpha'] = -2.0374
            hyperparameters['primary_minesite_cost_beta'] = 1.1396
            hyperparameters['primary_minesite_cost_gamma'] = 0.1615
            hyperparameters['primary_minesite_cost_delta'] = 0.0039
            hyperparameters['primary_minesite_cost_epsilon'] = 0.1717
            hyperparameters['primary_minesite_cost_theta'] = -0.2465
            hyperparameters['primary_minesite_cost_eta'] = 0.2974
            hyperparameters['primary_minesite_cost_rho'] = -0.0934
            hyperparameters['primary_minesite_cost_zeta'] = 0
        elif reg2use == 'linear_112_price_sx':  # updated total minesite cost see slide 112 left-hand-side table in C:\Users\ryter\Dropbox (MIT)\Group Research Folder_Olivetti\Displacement\04 Presentations\John\Weekly Updates\20210825 Generalization.pptx
            hyperparameters['primary_minesite_cost_alpha'] = 0.4683
            hyperparameters['primary_minesite_cost_beta'] = 0.8456
            hyperparameters['primary_minesite_cost_gamma'] = -0.1924
            hyperparameters['primary_minesite_cost_delta'] = -0.0125
            hyperparameters['primary_minesite_cost_epsilon'] = 0.1004
            hyperparameters['primary_minesite_cost_theta'] = 0.1910
            hyperparameters['primary_minesite_cost_eta'] = -0.3044
            hyperparameters['primary_minesite_cost_rho'] = 0.1288
            hyperparameters['primary_minesite_cost_zeta'] = 0.1285
        elif reg2use == 'linear_112_price_tcm_sx':  # updated total cash margin, see slide 112 right-hand-side table in C:\Users\ryter\Dropbox (MIT)\Group Research Folder_Olivetti\Displacement\04 Presentations\John\Weekly Updates\20210825 Generalization.pptx
            hyperparameters['primary_minesite_cost_alpha'] = -2.0096
            hyperparameters['primary_minesite_cost_beta'] = 1.1392
            hyperparameters['primary_minesite_cost_gamma'] = 0.1608
            hyperparameters['primary_minesite_cost_delta'] = 0.0027
            hyperparameters['primary_minesite_cost_epsilon'] = 0.1579
            hyperparameters['primary_minesite_cost_theta'] = -0.2338
            hyperparameters['primary_minesite_cost_eta'] = 0.2975
            hyperparameters['primary_minesite_cost_rho'] = -0.1020
            hyperparameters['primary_minesite_cost_zeta'] = -0.0583
        elif reg2use == 'linear_113_price_tcm_sx':  # updated total cash margin, see slide 113 right-hand-side table in C:\Users\ryter\Dropbox (MIT)\Group Research Folder_Olivetti\Displacement\04 Presentations\John\Weekly Updates\20210825 Generalization.pptx
            hyperparameters['primary_minesite_cost_alpha'] = -1.9868
            hyperparameters['primary_minesite_cost_beta'] = 1.1396
            hyperparameters['primary_minesite_cost_gamma'] = 0.1611
            hyperparameters['primary_minesite_cost_delta'] = 0
            hyperparameters['primary_minesite_cost_epsilon'] = 0.1628
            hyperparameters['primary_minesite_cost_theta'] = -0.2338
            hyperparameters['primary_minesite_cost_eta'] = 0.2968
            hyperparameters['primary_minesite_cost_rho'] = -0.1032
            hyperparameters['primary_minesite_cost_zeta'] = -0.0596

        reg2use = hyperparameters['primary_tcrc_regression2use']
        #             log(tcrc) = alpha + beta*log(commodity price) + gamma*log(head grade)
        #                 + delta*risk + epsilon*sxew + theta*dore (refining type)
        if reg2use == 'linear_114_reftype':  # see slide 113 left-hand-side table in C:\Users\ryter\Dropbox (MIT)\Group Research Folder_Olivetti\Displacement\04 Presentations\John\Weekly Updates\20210825 Generalization.pptx
            hyperparameters['primary_tcrc_alpha'] = -2.4186
            hyperparameters['primary_tcrc_beta'] = 0.9314
            hyperparameters['primary_tcrc_gamma'] = -0.0316
            hyperparameters['primary_tcrc_delta'] = 0.0083
            hyperparameters['primary_tcrc_epsilon'] = -0.1199
            hyperparameters['primary_tcrc_theta'] = -2.3439
            hyperparameters['primary_tcrc_eta'] = 0
        elif reg2use == 'linear_114':  # see slide 113 upper-right table in C:\Users\ryter\Dropbox (MIT)\Group Research Folder_Olivetti\Displacement\04 Presentations\John\Weekly Updates\20210825 Generalization.pptx
            hyperparameters['primary_tcrc_alpha'] = -2.3840
            hyperparameters['primary_tcrc_beta'] = 0.9379
            hyperparameters['primary_tcrc_gamma'] = -0.0233
            hyperparameters['primary_tcrc_delta'] = 0
            hyperparameters['primary_tcrc_epsilon'] = -0.0820
            hyperparameters['primary_tcrc_theta'] = -2.3451
            hyperparameters['primary_tcrc_eta'] = 0

        reg2use = hyperparameters['primary_scapex_regression2use']
        #             log(sCAPEX) = alpha + beta*log(commodity price) + gamma*log(head grade)
        #                 + delta*log(capacity) + epsilon*placer + theta*stockpile + eta*tailings + rho*underground + zeta*sxew
        if reg2use == 'linear_117_price_cap_sx':  # see slide 116 right-hand-side table in C:\Users\ryter\Dropbox (MIT)\Group Research Folder_Olivetti\Displacement\04 Presentations\John\Weekly Updates\20210825 Generalization.pptx
            hyperparameters['primary_scapex_alpha'] = -12.5802
            hyperparameters['primary_scapex_beta'] = 0.7334
            hyperparameters['primary_scapex_gamma'] = 0.6660
            hyperparameters['primary_scapex_delta'] = 0.9773
            hyperparameters['primary_scapex_epsilon'] = 0
            hyperparameters['primary_scapex_theta'] = 0
            hyperparameters['primary_scapex_eta'] = 0
            hyperparameters['primary_scapex_rho'] = 0.7989
            hyperparameters['primary_scapex_zeta'] = 0.6115
        elif reg2use == 'linear_119_cap_sx':  # see slide 118 right-hand-side table in C:\Users\ryter\Dropbox (MIT)\Group Research Folder_Olivetti\Displacement\04 Presentations\John\Weekly Updates\20210825 Generalization.pptx
            hyperparameters['primary_scapex_alpha'] = -5.3354
            hyperparameters['primary_scapex_beta'] = 0
            hyperparameters['primary_scapex_gamma'] = -0.0761
            hyperparameters['primary_scapex_delta'] = 0.8564
            hyperparameters['primary_scapex_epsilon'] = 0
            hyperparameters['primary_scapex_theta'] = -0.2043
            hyperparameters['primary_scapex_eta'] = -0.6806
            hyperparameters['primary_scapex_rho'] = 1.2780
            hyperparameters['primary_scapex_zeta'] = 0.9657
        elif reg2use == 'linear_123_norm':  # see slide 123 right-hand-side table in C:\Users\ryter\Dropbox (MIT)\Group Research Folder_Olivetti\Displacement\04 Presentations\John\Weekly Updates\20210825 Generalization.pptx
            hyperparameters['primary_scapex_alpha'] = -12.8678
            hyperparameters['primary_scapex_beta'] = 0.7479
            hyperparameters['primary_scapex_gamma'] = 0.6828
            hyperparameters['primary_scapex_delta'] = 0
            hyperparameters['primary_scapex_epsilon'] = 0
            hyperparameters['primary_scapex_theta'] = -0.5241
            hyperparameters['primary_scapex_eta'] = -0.5690
            hyperparameters['primary_scapex_rho'] = 0.7977
            hyperparameters['primary_scapex_zeta'] = 0.6478

        reg2use = hyperparameters['primary_dcapex_regression2use']
        if reg2use == 'linear_124_norm':
            hyperparameters['primary_dcapex_alpha'] = -10.9607
            hyperparameters['primary_dcapex_beta'] = 0.7492
            hyperparameters['primary_dcapex_gamma'] = 0.7747
            hyperparameters['primary_dcapex_delta'] = 0
            hyperparameters['primary_dcapex_epsilon'] = 0
            hyperparameters['primary_dcapex_theta'] = -6.3580
            hyperparameters['primary_dcapex_eta'] = -0.9672
            hyperparameters['primary_dcapex_rho'] = -0.1271
            hyperparameters['primary_dcapex_zeta'] = 0.2337

        reg2use = hyperparameters['primary_overhead_regression2use']
        if reg2use == 'linear_194':
            #             log(Overhead ($M)) = alpha + beta*log(commodity price) + gamma*log(head grade)
            #                 + delta*log(ore treated) + epsilon*placer + theta*stockpile + eta*tailings + rho*underground + zeta*sxew
            hyperparameters['primary_overhead_alpha'] = -4.2003
            hyperparameters['primary_overhead_beta'] = 0
            hyperparameters['primary_overhead_gamma'] = -0.0519
            hyperparameters['primary_overhead_delta'] = 0.6501
            hyperparameters['primary_overhead_epsilon'] = -1.7948
            hyperparameters['primary_overhead_theta'] = -0.3298
            hyperparameters['primary_overhead_eta'] = -2.1639
            hyperparameters['primary_overhead_rho'] = 0.6356
            hyperparameters['primary_overhead_zeta'] = -0.0841
        elif reg2use == 'None':
            hyperparameters['primary_overhead_alpha'] = np.log(0.1)
            hyperparameters['primary_overhead_beta'] = 0
            hyperparameters['primary_overhead_gamma'] = 0
            hyperparameters['primary_overhead_delta'] = 0
            hyperparameters['primary_overhead_epsilon'] = 0
            hyperparameters['primary_overhead_theta'] = 0
            hyperparameters['primary_overhead_eta'] = 0
            hyperparameters['primary_overhead_rho'] = 0
            hyperparameters['primary_overhead_zeta'] = 0

        return hyperparameters

    def recalculate_hyperparams(self):
        '''
        With hyperparameters initialized, can then edit any hyperparameters
        you\'d like then call this function to update any hyperparameters
        that could have been altered by such changes.
        '''
        hyperparameters = self.hyperparam
        hyperparameters['production_frac_region5'] = 1 - np.sum([hyperparameters[i] for i in ['production_frac_region1','production_frac_region2','production_frac_region3','production_frac_region4']])
        hyperparameters['minetype_prod_frac_placer'] = 1 - np.sum([hyperparameters[i] for i in ['minetype_prod_frac_underground','minetype_prod_frac_openpit','minetype_prod_frac_tailings','minetype_prod_frac_stockpile']])
        if self.byproduct:
            hyperparameters['byproduct_host1_production_fraction'] = 1 - np.sum([hyperparameters[i] for i in ['byproduct_pri_production_fraction','byproduct_host3_production_fraction','byproduct_host2_production_fraction']])


        hyperparameters = self.add_minesite_cost_regression_params(hyperparameters)
        hyperparameters['primary_tcm_flag'] = 'tcm' in hyperparameters[
            'primary_minesite_cost_regression2use']
        hyperparameters['primary_rr_negative'] = False

        initial_demand = hyperparameters['primary_production']
        change = hyperparameters['demand_series_pct_change'] / 100 + 1
        sim_time = self.simulation_time
        if hyperparameters['demand_series_method'] == 'yoy':
            hyperparameters['demand_series'] = pd.Series([initial_demand * change ** (j - sim_time[0]) for j in sim_time], sim_time)
            self.demand_series = hyperparameters['demand_series']
        elif hyperparameters['demand_series_method'] == 'target':
            hyperparameters['demand_series'] = pd.Series(np.linspace(initial_demand, initial_demand * change, len(sim_time)), sim_time)
            self.demand_series = hyperparameters['demand_series']
        elif self.verbosity > 1:
            print('using whatever value has already been assigned to demand_series')

        incent_series = [i * (hyperparameters['annual_reserves_ratio_with_initial_production_const'] + j)
                         for i, j in zip(self.demand_series.values,
                                         [hyperparameters[
                                              'annual_reserves_ratio_with_initial_production_slope'] * (
                                                      y - self.simulation_time[0]) for y in self.simulation_time])]
        hyperparameters['incentive_resources_contained_series'] = pd.Series(incent_series, self.simulation_time)
        hyperparameters['incentive_subsample_series'] = pd.Series(hyperparameters['incentive_subsample_init'], self.simulation_time)
        self.resources_contained_series = hyperparameters['incentive_resources_contained_series']
        self.subsample_series = hyperparameters['incentive_subsample_series']

        self.hyperparam = hyperparameters

    def generate_production_region(self):
        '''updates self.mines, first function called.'''
        seed(self.rs)
        hyperparam = self.hyperparam
        h = self.hyperparam
        pri_dist = getattr(stats, h['primary_production_distribution'])
        pri_prod = h['primary_production']
        pri_prod_mean_frac = h['primary_production_mean']
        pri_prod_var_frac = h['primary_production_var']
        production_fraction = h['primary_production_fraction']

        pri_prod_frac_dist = pri_dist.rvs(
            loc=0,
            scale=pri_prod_mean_frac,
            s=pri_prod_var_frac,
            size=int(np.ceil(2 * pri_prod / pri_prod_mean_frac)),
            random_state=self.rs)

        mines = pd.DataFrame(
            pri_prod_frac_dist,
            index=np.arange(0, int(np.ceil(2 * pri_prod / pri_prod_mean_frac))),
            columns=['Production fraction'])
        mines['Production fraction'] /= pri_prod

        mines_not_ramping = mines.copy()
        mines['Ramp up flag'] = 0
        if h['ramp_up_fraction'] != 0:
            for i in np.arange(1, h['ramp_up_years'] + 1):
                num_ramping = int(round(h['ramp_up_fraction'] * mines.shape[0], 0))
                if num_ramping > mines_not_ramping.shape[0]: raise ValueError(
                    'ramp_up_fraction hyperparam input is too large, not enough mines in the pool to have any remaining non-ramping mines')
                ind = mines_not_ramping.sample(num_ramping, replace=False, random_state=self.rs).index
                mines_not_ramping.drop(ind, inplace=True)
                mines.loc[ind, 'Ramp up flag'] = i
                if h['ramp_up_exponent'] != 0:
                    ramp_up_exp = h['ramp_up_exponent']
                    mines.loc[ind, 'Production fraction'] *= (
                                h['ramp_up_cu'] + (1 - h['ramp_up_cu']) / (h['ramp_up_years']) ** ramp_up_exp * (
                                    mines.loc[ind, 'Ramp up flag'] - 1) ** ramp_up_exp)
                else:
                    mines.loc[ind, 'Production fraction'] *= h['ramp_up_cu']

        mines = mines.loc[mines['Production fraction'].cumsum() < production_fraction, :]
        mines.loc[mines.index[-1] + 1, 'Production fraction'] = production_fraction - mines['Production fraction'].sum()
        mines.loc[mines.index[-1], 'Ramp up flag'] = 0
        mines['Production (kt)'] = mines['Production fraction'] * pri_prod

        # mines = OneMine('init')
        #
        # mines.add_var('production_fraction', pri_prod_frac_dist)
        # mines.add_var('index', np.arange(0,len(pri_prod_frac_dist)))
        # mines.production_fraction = mines.production_fraction/pri_prod
        #
        # mines_not_ramping = mines.copy()
        # mines.add_var('ramp_up_flag', np.repeat(0,len(pri_prod_frac_dist)))
        # if h['ramp_up_fraction']!=0:
        #     for i in np.arange(1,h['ramp_up_years']+1):
        #         num_ramping = int(round(h['ramp_up_fraction']*mines.shape[0],0))
        #         if num_ramping>mines_not_ramping.shape[0]: raise ValueError('ramp_up_fraction hyperparam input is too large, not enough mines in the pool to have any remaining non-ramping mines')
        #
        #         ind = mines_not_ramping.sample(size=num_ramping,replace=False,random_state=self.rs)
        #         mines_not_ramping.drop(ind,inplace=True)
        #         mines.ramp_up_flag[ind] = i
        #         if h['ramp_up_exponent']!=0:
        #             ramp_up_exp = h['ramp_up_exponent']
        #             mines.production_fraction[ind] = mines.production_fraction[ind] * (h['ramp_up_cu']+(1-h['ramp_up_cu'])/(h['ramp_up_years'])**ramp_up_exp*(mines.ramp_up_flag[ind]-1)**ramp_up_exp)
        #         else:
        #             mines.production_fraction[ind] = mines.production_fraction[ind] * h['ramp_up_cu']
        #
        # mines.drop( mines.index[np.cumsum(mines.production_fraction)>production_fraction] )
        # mines.production_fraction = np.append(mines.production_fraction, [production_fraction-np.nansum(mines.production_fraction)])
        # mines.index = np.append(mines.index,[len(mines.index)])
        # mines.ramp_up_flag = np.append(mines.ramp_up_flag,[0])
        # mines.production_kt = mines.production_fraction*pri_prod

        regions = [i for i in hyperparam if 'production_frac_region' in i]
        region_fractions = [hyperparam[i] for i in hyperparam if 'production_frac_region' in i]
        mines['Region'] = np.nan
        for i in regions:
            int_version = int(i.replace('production_frac_region', ''))
            ind = mines.loc[(mines.Region.isna()), 'Production fraction'].cumsum()
            ind = ind.loc[ind < hyperparam[i] * production_fraction].index
            mines.loc[ind, 'Region'] = int_version
        mines.loc[mines.Region.isna(), 'Region'] = int(
            regions[np.argmax(region_fractions)].replace('production_frac_region', ''))

        mines['Simulation start ore treated (kt)'] = np.nan
        mines['Ramp down flag'] = False
        mines['Closed flag'] = False
        mines['Operate with negative cash flow'] = False
        mines['Total cash margin expect (USD/t)'] = np.nan
        mines['NPV ramp next ($M)'] = np.nan
        mines['NPV ramp following ($M)'] = np.nan
        mines['Close method'] = np.nan
        mines['Simulated closure'] = np.nan
        mines['Initial head grade (%)'] = np.nan
        mines['Discount'] = 1
        mines['Initial price (USD/t)'] = np.nan
        mines['Commodity price (USD/t)'] = h['primary_commodity_price']
        mines['Capacity utilization expect'] = np.nan
        mines['CU ramp following'] = np.nan
        mines['Generated TCRC (USD/t)'] = np.nan
        mines['Ore treat expect (kt)'] = np.nan
        mines['Ore treat ramp following (kt)'] = np.nan
        mines['Price expect (USD/t)'] = np.nan
        mines['Real index'] = np.nan
        mines['TCRC expect (USD/t)'] = np.nan
        self.mines = mines.copy()

    def generate_grade_and_masses(self):
        '''
        Note that the mean ore grade reported will be different than
        the cumulative ore grade of all the ore treated here since
        we are randomly assigning grades and don\'t have a good way
        to correct for this.
        '''
        h = self.hyperparam
        self.assign_mine_types()
        self.mines['Risk indicator'] = self.values_from_dist('primary_minerisk').round(0)
        self.mines['Head grade (%)'] = self.values_from_dist('primary_ore_grade')
        self.mines.loc[self.mines['Head grade (%)'] > 80, 'Head grade (%)'] = 80
        while (self.mines['Head grade (%)']>100*h['primary_ore_grade_mean']).any():
            self.mines.loc[self.mines['Head grade (%)']>100*h['primary_ore_grade_mean']] /= 100
        self.mines['Commodity price (USD/t)'] = float(h['primary_commodity_price'])

        mines = self.mines.copy()

        mines['Capacity utilization'] = self.values_from_dist('primary_cu')
        ind = mines['Ramp up flag'] != 0
        if h['ramp_up_exponent'] != 0:
            ramp_up_exp = h['ramp_up_exponent']
            mines.loc[ind, 'Capacity utilization'] = h['mine_cu0'] * (
                        h['ramp_up_cu'] + (1 - h['ramp_up_cu']) / (h['ramp_up_years']) ** ramp_up_exp * (
                            mines.loc[ind, 'Ramp up flag'] - 1) ** ramp_up_exp)
        else:
            mines.loc[ind, 'Capacity utilization'] = h['ramp_up_cu']
        mines['Production capacity (kt)'] = mines['Capacity utilization'] * mines['Production (kt)']
        mines['Production capacity fraction'] = mines['Production capacity (kt)'] / mines[
            'Production capacity (kt)'].sum()
        mines['Payable percent (%)'] = 100 - self.values_from_dist('primary_payable_percent')
        mines.loc[mines['Production fraction'].cumsum() <= self.hyperparam[
            'primary_sxew_fraction'], 'Payable percent (%)'] = 100
        if self.hyperparam['primary_sxew_fraction']==1: mines['Payable percent (%)']=100.
        self.mines = mines.copy()
        self.generate_costs_from_regression('Recovery rate (%)')
        mines = self.mines.copy()
        rec_rates = mines['Recovery rate (%)'].copy()
        if rec_rates.max() < 30:
            rec_rates.loc[:] = 30
            if self.verbosity > 1:
                print('Generated recovery rates too low, using 30% for all')
        if rec_rates.min() > 99:
            rec_rates.loc[:] = 99
            if self.verbosity > 1:
                print('Generated recovery rates too high, using 99% for all')
        if (rec_rates > 99).any():
            rr = rec_rates[rec_rates < 99].sample(n=(rec_rates > 99).sum(), replace=True, random_state=self.rs).reset_index(drop=True)
            rr = rr.rename(dict(zip(rr.index, rec_rates[rec_rates > 99].index)))
            rec_rates.loc[rec_rates > 99] = rr
        if (rec_rates < 0).any():
            rr = rec_rates[rec_rates > 0].sample(n=(rec_rates < 0).sum(), replace=True, random_state=self.rs).reset_index(drop=True)
            rr = rr.rename(dict(zip(rr.index, rec_rates[rec_rates < 0].index)))
            rec_rates.loc[rec_rates < 0] = rr
        mines['Recovery rate (%)'] = rec_rates

        #         if rec_rates.max()<30 or (rec_rates<0).any() or (rec_rates>100).any():
        #             rec_rates = 100 - self.values_from_dist('primary_rr_default')
        #             self.hyperparam['primary_rr_negative','Value'] = True
        #         mines.loc[mines.sort_values('Head grade (%)').index,'Recovery rate (%)'] = \
        #             partial_shuffle(np.sort(rec_rates),self.hyperparam['primary_recovery_rate_shuffle_param'])

        mines['Ore treated (kt)'] = mines['Production (kt)'] / (
                    mines['Recovery rate (%)'] * mines['Head grade (%)'] / 1e4)
        mines['Capacity (kt)'] = mines['Ore treated (kt)'] / mines['Capacity utilization']
        mines['Paid metal production (kt)'] = mines[
                                                         ['Capacity (kt)', 'Capacity utilization', 'Head grade (%)',
                                                          'Recovery rate (%)', 'Payable percent (%)']].product(
            axis=1) / 1e6
        mines['Reserves ratio with ore treated'] = self.values_from_dist('primary_reserves')

        mines['Reserves (kt)'] = mines[['Ore treated (kt)', 'Reserves ratio with ore treated']].product(axis=1)
        mines['Reserves potential metal content (kt)'] = mines[['Reserves (kt)', 'Head grade (%)']].product(
            axis=1) * 1e-2

        # calibrating reserves to input values if needed
        primary_reserves_reported_basis = self.hyperparam['primary_reserves_reported_basis']
        primary_reserves_reported = self.hyperparam['primary_reserves_reported']
        if primary_reserves_reported_basis == 'ore' and primary_reserves_reported > 0:
            ratio = primary_reserves_reported / mines['Reserves (kt)'].sum()
        elif primary_reserves_reported_basis == 'metal' and primary_reserves_reported > 0:
            ratio = primary_reserves_reported / mines['Reserves potential metal content (kt)'].sum()
        else:
            ratio = 1
        mines['Reserves (kt)'] *= ratio
        mines['Reserves potential metal content (kt)'] *= ratio

        # setting up cumulative ore treated for use with calculating initial grades
        mines['Cumulative ore treated ratio with ore treated'] = self.values_from_dist('primary_ot_cumu')
        mines['Cumulative ore treated (kt)'] = mines['Cumulative ore treated ratio with ore treated'] * mines[
            'Ore treated (kt)']
        mines['Opening'] = self.simulation_time[0] - mines[
            'Cumulative ore treated ratio with ore treated'].round(0)
        mines['Initial ore treated (kt)'] = mines['Ore treated (kt)'] / self.hyperparam['ramp_up_years']

        if h['ramp_up_exponent'] != 0:
            mines.loc[ind, 'Cumulative ore treated (kt)'] = 0
            ramp_up_exp = h['ramp_up_exponent']
            for i in np.arange(1, h['ramp_up_years'] + 1):
                ind2 = (mines['Ramp up flag'] <= i) & (mines['Ramp up flag'] != 0)
                mines.loc[ind2, 'Cumulative ore treated (kt)'] += mines.loc[ind2, 'Ore treated (kt)'] * (
                            h['ramp_up_cu'] + (1 - h['ramp_up_cu']) / (h['ramp_up_years']) ** ramp_up_exp * (
                                i - 1) ** ramp_up_exp) / (h['ramp_up_cu'] + (1 - h['ramp_up_cu']) / (
                h['ramp_up_years']) ** ramp_up_exp * (mines.loc[ind2, 'Ramp up flag'] - 1) ** ramp_up_exp)
            mines.loc[ind, 'Initial ore treated (kt)'] = mines.loc[ind, 'Ore treated (kt)'] * h['ramp_up_cu'] / (
                        h['ramp_up_cu'] + (1 - h['ramp_up_cu']) / (h['ramp_up_years']) ** ramp_up_exp * (
                            mines.loc[ind, 'Ramp up flag'] - 1) ** ramp_up_exp)
        else:
            mines.loc[ind, 'Cumulative ore treated (kt)'] = mines.loc[ind, 'Ore treated (kt)'] * mines.loc[
                ind, 'Ramp up flag']
            mines.loc[ind, 'Initial ore treated (kt)'] = mines.loc[ind, 'Ore treated (kt)']
        mines.loc[ind, 'Opening'] = self.simulation_time[0] - mines.loc[ind, 'Ramp up flag']
        mines.loc[ind, 'Cumulative ore treated ratio with ore treated'] = mines['Cumulative ore treated (kt)'][ind] / \
                                                                          mines['Ore treated (kt)'][ind]

        self.mines = mines.copy()

    def generate_costs_from_regression(self, param):
        '''Called inside generate_total_cash_margin'''
        mines = self.mines.copy()
        h = self.hyperparam

        if h['primary_minesite_cost_mean'] > 0 and param == 'Minesite cost (USD/t)':
            mines[param] = self.values_from_dist('primary_minesite_cost')
        elif param in ['Minesite cost (USD/t)', 'Total cash margin (USD/t)']:
            #             log(minesite cost) = alpha + beta*log(head grade) + gamma*(head grade) + delta*(numerical risk) + epsilon*placer (mine type)
            #                + theta*stockpile + eta*tailings + rho*underground + zeta*sxew
            log_minesite_cost = h['primary_minesite_cost_alpha'] + \
                                h['primary_minesite_cost_beta'] * np.log(mines['Commodity price (USD/t)']) + \
                                h['primary_minesite_cost_gamma'] * np.log(mines['Head grade (%)']) + \
                                h['primary_minesite_cost_delta'] * mines['Risk indicator'] + \
                                h['primary_minesite_cost_epsilon'] * (mines['Mine type string'] == 'placer') + \
                                h['primary_minesite_cost_theta'] * (mines['Mine type string'] == 'stockpile') + \
                                h['primary_minesite_cost_eta'] * (mines['Mine type string'] == 'tailings') + \
                                h['primary_minesite_cost_rho'] * (mines['Mine type string'] == 'underground') + \
                                h['primary_minesite_cost_zeta'] * (mines['Payable percent (%)'] == 100)
            mines[param] = np.exp(log_minesite_cost)
        elif param == 'Recovery rate (%)':
            #             log(minesite cost) = alpha + beta*log(price) + gamma*(head grade) + delta*placer (mine type)
            #                + epsilon*stockpile + theta*tailings + eta*underground + rho*sxew
            minesite_cost = h['primary_rr_alpha'] + \
                            h['primary_rr_beta'] * np.log(mines['Commodity price (USD/t)']) + \
                            h['primary_rr_gamma'] * np.log(mines['Head grade (%)']) + \
                            h['primary_rr_delta'] * (mines['Mine type string'] == 'placer') + \
                            h['primary_rr_epsilon'] * (mines['Mine type string'] == 'stockpile') + \
                            h['primary_rr_theta'] * (mines['Mine type string'] == 'tailings') + \
                            h['primary_rr_eta'] * (mines['Mine type string'] == 'underground') + \
                            h['primary_rr_rho'] * (mines['Payable percent (%)'] == 100)
            mines[param] = minesite_cost
        elif param == 'TCRC (USD/t)':
            #             log(tcrc) = alpha + beta*log(commodity price) + gamma*log(head grade)
            #                 + delta*risk + epsilon*sxew + theta*dore (refining type)
            log_minesite_cost = h['primary_tcrc_alpha'] + \
                                h['primary_tcrc_beta'] * np.log(mines['Commodity price (USD/t)']) + \
                                h['primary_tcrc_gamma'] * np.log(mines['Head grade (%)']) + \
                                h['primary_tcrc_delta'] * mines['Risk indicator'] + \
                                h['primary_tcrc_epsilon'] * (mines['Payable percent (%)'] == 100) + \
                                h['primary_tcrc_theta'] * h['primary_tcrc_dore_flag'] + \
                                h['primary_tcrc_eta'] * (mines['Mine type string'] == 'tailings')
            mines[param] = np.exp(log_minesite_cost)
            mines.loc[mines['Payable percent (%)']==100,param]=0
        elif param == 'Sustaining CAPEX ($M)':
            #         log(sCAPEX) = alpha + beta*log(commodity price) + gamma*log(head grade)
            #            + delta*log(capacity) + epsilon*placer + theta*stockpile + eta*tailings + rho*underground + zeta*sxew
            prefix = 'Primary ' if self.byproduct else ''
            log_minesite_cost = \
                h['primary_scapex_alpha'] + \
                h['primary_scapex_beta'] * np.log(mines[prefix + 'Commodity price (USD/t)']) + \
                h['primary_scapex_gamma'] * np.log(mines[prefix + 'Head grade (%)']) + \
                h['primary_scapex_delta'] * np.log(mines['Capacity (kt)']) + \
                h['primary_scapex_epsilon'] * (mines['Mine type string'] == 'placer') + \
                h['primary_scapex_theta'] * (mines['Mine type string'] == 'stockpile') + \
                h['primary_scapex_eta'] * (mines['Mine type string'] == 'tailings') + \
                h['primary_scapex_rho'] * (mines['Mine type string'] == 'underground') + \
                h['primary_scapex_zeta'] * (mines[prefix + 'Payable percent (%)'] == 100)
            if 'norm' in h['primary_scapex_regression2use']:
                mines[param] = np.exp(log_minesite_cost) * mines['Capacity (kt)']
            else:
                mines[param] = np.exp(log_minesite_cost)
        elif param == 'Development CAPEX ($M)':
            #         log(sCAPEX) = alpha + beta*log(commodity price) + gamma*log(head grade)
            #            + delta*log(capacity) + epsilon*placer + theta*stockpile + eta*tailings + rho*underground + zeta*sxew
            prefix = 'Primary ' if self.byproduct else ''
            if 'Primary Commodity price (USD/t)' in mines.columns:
                pri = mines.loc[mines['Byproduct ID'] != 0].index
            else:
                pri = mines.index
            log_minesite_cost = \
                h['primary_dcapex_alpha'] + \
                h['primary_dcapex_beta'] * np.log(mines.loc[pri, prefix + 'Commodity price (USD/t)'].astype(float)) + \
                h['primary_dcapex_gamma'] * np.log(mines.loc[pri, prefix + 'Head grade (%)'].astype(float)) + \
                h['primary_dcapex_delta'] * np.log(mines.loc[pri, 'Capacity (kt)'].astype(float)) + \
                h['primary_dcapex_epsilon'] * (mines.loc[pri, 'Mine type string'] == 'placer') + \
                h['primary_dcapex_theta'] * (mines.loc[pri, 'Mine type string'] == 'stockpile') + \
                h['primary_dcapex_eta'] * (mines.loc[pri, 'Mine type string'] == 'tailings') + \
                h['primary_dcapex_rho'] * (mines.loc[pri, 'Mine type string'] == 'underground') + \
                h['primary_dcapex_zeta'] * (mines.loc[pri, prefix + 'Payable percent (%)'] == 100)
            if 'norm' in h['primary_dcapex_regression2use']:
                mines.loc[pri, param] = np.exp(log_minesite_cost) * mines.loc[pri, 'Capacity (kt)'].astype(float)
            else:
                mines.loc[pri, param] = np.exp(log_minesite_cost)
        elif param == 'Overhead ($M)':
            #             log(Overhead ($M)) = alpha + beta*log(commodity price) + gamma*log(head grade)
            #                 + delta*log(ore treated) + epsilon*placer + theta*stockpile + eta*tailings + rho*underground + zeta*sxew
            prefix = 'Primary ' if self.byproduct else ''
            if 'Primary Commodity price (USD/t)' in mines.columns:
                pri = mines.loc[mines['Byproduct ID'] != 0].index
            else:
                pri = mines.index
            log_minesite_cost = \
                h['primary_overhead_alpha'] + \
                h['primary_overhead_beta'] * np.log(mines.loc[pri, prefix + 'Commodity price (USD/t)'].astype(float)) + \
                h['primary_overhead_gamma'] * np.log(mines.loc[pri, prefix + 'Head grade (%)'].astype(float)) + \
                h['primary_overhead_delta'] * np.log(mines.loc[pri, 'Ore treated (kt)'].astype(float)) + \
                h['primary_overhead_epsilon'] * (mines.loc[pri, 'Mine type string'] == 'placer') + \
                h['primary_overhead_theta'] * (mines.loc[pri, 'Mine type string'] == 'stockpile') + \
                h['primary_overhead_eta'] * (mines.loc[pri, 'Mine type string'] == 'tailings') + \
                h['primary_overhead_rho'] * (mines.loc[pri, 'Mine type string'] == 'underground') + \
                h['primary_overhead_zeta'] * (mines.loc[pri, prefix + 'Payable percent (%)'] == 100)
            if 'norm' in h['primary_overhead_regression2use']:
                mines.loc[pri, param] = np.exp(log_minesite_cost) * mines.loc[pri, 'Capacity (kt)'].astype(float)
            else:
                mines.loc[pri, param] = np.exp(log_minesite_cost)

        self.mines = mines.copy()

    def generate_total_cash_margin(self):
        h = self.hyperparam

        # Risk indicator is the sum of political, operational, terrorism, and security risks, which range from insignificant (1) to extreme (5)
        risk_upper_cutoff = 20
        risk_lower_cutoff = 4
        self.mines.loc[self.mines['Risk indicator'] > risk_upper_cutoff, 'Risk indicator'] = risk_upper_cutoff
        self.mines.loc[self.mines['Risk indicator'] < risk_lower_cutoff, 'Risk indicator'] = risk_lower_cutoff

        self.generate_costs_from_regression('TCRC (USD/t)')

        if h['primary_tcm_flag']:
            self.generate_costs_from_regression('Total cash margin (USD/t)')
            self.mines['Total cash margin (USD/t)'] -= self.mines['TCRC (USD/t)']
            self.mines['Minesite cost (USD/t)'] = self.mines['Commodity price (USD/t)'] - self.mines[
                ['TCRC (USD/t)', 'Total cash margin (USD/t)']].sum(axis=1)
            self.mines['Total cash cost (USD/t)'] = self.mines[['TCRC (USD/t)', 'Minesite cost (USD/t)']].sum(
                axis=1)
            if self.verbosity > 1:
                print('tcm')
        else:
            self.generate_costs_from_regression('Minesite cost (USD/t)')
            self.mines['Total cash cost (USD/t)'] = self.mines[['TCRC (USD/t)', 'Minesite cost (USD/t)']].sum(
                axis=1)
            self.mines['Total cash margin (USD/t)'] = self.mines['Commodity price (USD/t)'] - self.mines[
                'Total cash cost (USD/t)']
            if self.verbosity > 1:
                print('tmc')

    def assign_mine_types(self):
        mines = self.mines.copy()
        h = self.hyperparam
        params = [i for i in h if 'minetype_prod_frac' in i]
        param_vals = [h[i] for i in h if 'minetype_prod_frac' in i]
        self.mine_type_mapping = {0: 'openpit', 1: 'placer', 2: 'stockpile', 3: 'tailings', 4: 'underground'}
        self.mine_type_mapping_rev = {'openpit': 0, 'placer': 1, 'stockpile': 2, 'tailings': 3, 'underground': 4}
        mines['Mine type'] = np.nan
        mines['Mine type string'] = np.nan

        for i in params:
            map_param = i.split('_')[-1]
            ind = mines.loc[(mines['Mine type'].isna()), 'Production fraction'].cumsum()
            ind = ind.loc[ind < h[i] * h['primary_production_fraction']].index
            mines.loc[ind, 'Mine type'] = self.mine_type_mapping_rev[map_param]
            mines.loc[ind, 'Mine type string'] = map_param
        mines.loc[mines['Mine type'].isna(), 'Mine type'] = self.mine_type_mapping_rev[
            params[np.argmax(param_vals)].split('_')[-1]]
        mines.loc[mines['Mine type string'].isna(), 'Mine type string'] = params[np.argmax(param_vals)].split('_')[-1]
        self.mines = mines.copy()

    def generate_oges(self):
        s, loc, scale = (self.hyperparam[i] for i in ['primary_oge_s','primary_oge_loc','primary_oge_scale'])
        self.mines['OGE'] = 0 - stats.lognorm.rvs(s=s, loc=loc, scale=scale, size=self.mines.shape[0],
                                                         random_state=self.rs)
        i = 0
        if (self.mines['OGE'] > 0).all(): raise ValueError(
            'All OGEs greater than zero, reassess primary_oge_scale/s inputs')
        while (self.mines['OGE'] > 0).any():
            self.mines.loc[self.mines['OGE'] > 0, 'OGE'] = 1 - stats.lognorm.rvs(s, loc, scale,
                                                                                 size=(self.mines['OGE'] > 0).sum(),
                                                                                 random_state=self.rs + i)
            i += 1
            print('mines with OGE>0, this is a problem')
        if self.byproduct:
            self.mines['Primary OGE'] = self.mines['OGE']

    def generate_annual_costs(self):
        h = self.hyperparam
        self.generate_costs_from_regression('Sustaining CAPEX ($M)')
        self.generate_costs_from_regression('Overhead ($M)')
        #         self.mines['Sustaining CAPEX ($M)'] /= 2
        mines = self.mines.copy()

        mines['Paid metal profit ($M)'] = mines[['Paid metal production (kt)',
                                                        'Total cash margin (USD/t)']].product(axis=1) / 1e3
        mines['Cash flow ($M)'] = mines['Paid metal profit ($M)'] - mines[
            ['Sustaining CAPEX ($M)', 'Overhead ($M)']].sum(axis=1)
        mines['Total reclamation cost ($M)'] = np.exp(h['primary_reclamation_constant'] +
                                                             h['primary_reclamation_slope'] * np.log(
            mines['Capacity (kt)'] / 1e3))
        mines['Cash flow expect ($M)'] = np.nan
        mines['Development CAPEX ($M)'] = 0

        if h['byproduct']:
            mines['Byproduct Total cash margin (USD/t)'] = mines['Total cash margin (USD/t)']
            mines['Byproduct Cash flow ($M)'] = 1e-3 * mines['Paid metal production (kt)'] * mines[
                'Byproduct Total cash margin (USD/t)'] - mines['Sustaining CAPEX ($M)']
        self.mines = mines.copy()

    def generate_byproduct_mines(self):
        if self.byproduct:
            h = self.hyperparam
            mines = pd.DataFrame()
            pri = miningModel(byproduct=True)
            pri.hyperparam = self.hyperparam.copy()
            pri.update_hyperparams_from_byproducts('byproduct')
            pri.update_hyperparams_from_byproducts('byproduct_pri')
            pri.byproduct = False

            if pri.hyperparam['byproduct_pri_production_fraction'] > 0:
                pri.initialize_mines()
                pri.mines['Byproduct ID'] = 0
                self.pri = pri
                byproduct_mine_models = [pri]
                byproduct_mines = [pri.mines]
            else:
                byproduct_mine_models = []
                byproduct_mines = []
            for param in np.unique([i.split('_')[1] for i in h if 'host' in i]):
                if self.hyperparam['byproduct_' + param + '_production_fraction'] != 0:
                    byproduct_model = self.generate_byproduct_params(param)
                    byproduct_mine_models += [byproduct_model]
                    byproduct_mines += [byproduct_model.mines]
            self.byproduct_mine_models = byproduct_mine_models
            self.mines = pd.concat(byproduct_mines).reset_index(drop=True)
            self.update_operation_hyperparams()

    def update_hyperparams_from_byproducts(self, param):
        h = self.hyperparam
        if param == 'byproduct':
            replace_h = [i for i in h if 'byproduct' in i and 'host' not in i and i != 'byproduct']
        else:
            replace_h = [i for i in h if param in i]
        replace_h_split = [i.split(param)[1] for i in replace_h]
        to_replace_h = [i for i in h if 'primary' in i and i.split('primary')[1] in replace_h_split]
        if param == 'byproduct':
            matched = [(i, j) for i in replace_h for j in to_replace_h if
                            '_'.join(i.split('_')[1:]) == '_'.join(j.split('_')[1:])]
        else:
            matched = [(i, j) for i in replace_h for j in to_replace_h if
                            '_'.join(i.split('_')[2:]) == '_'.join(j.split('_')[1:])]
        for i,j in matched:
            if j in matched:
                self.hyperparam[j] = self.hyperparam[i]
        # self.hyperparam[to_replace_h] = self.hyperparam.drop(to_replace_h).rename(matched).loc[
        #     to_replace_h]
        if self.verbosity > 1:
            display(matched)

    def generate_byproduct_params(self, param):
        ''' '''
        self.generate_byproduct_production(param)
        self.generate_byproduct_costs(param)
        self.correct_byproduct_production(param)
        self.generate_byproduct_total_costs(param)
        return getattr(self, param)

    def generate_byproduct_production(self, param):
        by_param = 'byproduct_' + param
        host1 = miningModel(byproduct=True, simulation_time=self.simulation_time, verbosity=self.verbosity)
        h = self.hyperparam
        host1.hyperparam = h
        h = h
        host1_params = [i for i in h if param in i]
        host1.update_hyperparams_from_byproducts(by_param)

        production_fraction = h[by_param + '_production_fraction']

        by_dist = getattr(stats, h['byproduct_production_distribution'])
        prod_mean_frac = h['byproduct_production_mean'] / h['byproduct_production']
        prod_var_frac = h['byproduct_production_var'] / h['byproduct_production']

        host1_prod_frac_dist = by_dist.rvs(
            loc=0,
            scale=prod_mean_frac,
            s=prod_var_frac,
            size=int(np.ceil(2 / prod_mean_frac)),
            random_state=self.rs)
        mines = pd.DataFrame(
            host1_prod_frac_dist,
            index=np.arange(0, int(np.ceil(2 / prod_mean_frac))),
            columns=['Production fraction'])
        mines = mines.loc[mines['Production fraction'].cumsum() < production_fraction, :]
        mines.loc[mines.index[-1] + 1, 'Production fraction'] = production_fraction - mines['Production fraction'].sum()
        mines['Production (kt)'] = mines['Production fraction'] * h['byproduct_production']

        regions = [i for i in h if 'production_frac_region' in i]
        region_fractions = [h[i] for i in h if 'production_frac_region' in i]
        mines['Region'] = np.nan
        for i in regions:
            int_version = int(i.replace('production_frac_region', ''))
            ind = mines.loc[(mines.Region.isna()), 'Production fraction'].cumsum()
            ind = ind.loc[ind < h[i] * production_fraction].index
            mines.loc[ind, 'Region'] = int_version
        mines.loc[mines.Region.isna(), 'Region'] = int(
            h[regions].astype(float).idxmax().replace('production_frac_region', ''))

#         regions = [i for i in hyperparam if 'production_frac_region' in i]
#         region_fractions = [hyperparam[i] for i in hyperparam if 'production_frac_region' in i]
# int(
#             regions[np.argmax(region_fractions)].replace('production_frac_region', ''))

        host1.mines = mines.copy()
        host1.assign_mine_types()
        host1.mines['Risk indicator'] = host1.values_from_dist(by_param + '_minerisk').round(0)
        host1.mines['Head grade (%)'] = host1.values_from_dist(by_param + '_ore_grade')

        mines = host1.mines.copy()
        mines['Capacity utilization'] = host1.values_from_dist(by_param + '_cu')
        mines['Recovery rate (%)'] = 100. - host1.values_from_dist(by_param + '_rr_default')
        mines['Production capacity (kt)'] = mines[['Capacity utilization', 'Production (kt)']].product(axis=1)
        mines['Production capacity fraction'] = mines['Production capacity (kt)'] / mines[
            'Production capacity (kt)'].sum()
        mines['Payable percent (%)'] = 100. - host1.values_from_dist(by_param + '_payable_percent')
        mines.loc[mines['Production fraction'].cumsum() <= h[by_param + '_sxew_fraction'], 'Payable percent (%)'] = 100.
        if h[by_param+'_sxew_fraction']==1: mines['Payable percent (%)']=100.
        mines.loc[mines['Payable percent (%)']==100,'TCRC (USD/t)'] = 0
        mines['Commodity price (USD/t)'] = h[by_param + '_commodity_price']

        host1.mines = mines.copy()
        setattr(self, param, host1)

    def generate_byproduct_costs(self, param):
        by_param = 'byproduct_' + param
        host1 = getattr(self, param)
        h = host1.hyperparam

        host1.generate_total_cash_margin()
        mines = host1.mines.copy()

        mines['Byproduct ID'] = int(param.split('host')[1])
        host1.mines = mines.copy()

        pri_main = [i for i in h if 'byproduct' in i and 'host' not in i and 'pri' not in i and i != 'byproduct']
        setattr(self, param, host1)
        return host1

    def correct_byproduct_production(self, param):
        by_param = 'byproduct_' + param
        host1 = getattr(self, param)
        h = host1.hyperparam
        mines = host1.mines.copy()

        primary_relocate = ['Commodity price (USD/t)', 'Recovery rate (%)', 'Head grade (%)', 'Payable percent (%)',
                            'Minesite cost (USD/t)', 'Total cash margin (USD/t)', 'Total cash cost (USD/t)',
                            'TCRC (USD/t)']
        for i in primary_relocate:
            mines['Primary ' + i] = mines[i]
            if i == 'Commodity price (USD/t)':
                mines['Primary ' + i] = mines[i].replace(0, h['byproduct_commodity_price'])
        mines['Commodity price (USD/t)'] = h['byproduct_commodity_price']
        for j in np.arange(0, 4):
            mines.loc[mines['Byproduct ID'] == j, 'Recovery rate (%)'] = h['byproduct' + str(j) + '_rr0']
        mines['Byproduct grade ratio'] = host1.values_from_dist(by_param + '_grade_ratio')
        mines.loc[mines['Byproduct grade ratio'] < 0, 'Byproduct grade ratio'] = mines['Byproduct grade ratio'].sample(
            n=(mines['Byproduct grade ratio'] < 0).sum(), random_state=self.rs).values
        mines['Head grade (%)'] = mines['Primary Head grade (%)'] / mines['Byproduct grade ratio']
        mines['Payable percent (%)'] = 100.

        mines['Byproduct minesite cost ratio'] = host1.values_from_dist(by_param + '_minesite_cost_ratio')
        mines.loc[mines['Byproduct minesite cost ratio'] < 0, 'Byproduct minesite cost ratio'] = mines[
            'Byproduct minesite cost ratio'].sample(n=(mines['Byproduct minesite cost ratio'] < 0).sum(),
                                                    random_state=self.rs).values
        mines['Minesite cost (USD/t)'] = mines['Primary Minesite cost (USD/t)'] / mines[
            'Byproduct minesite cost ratio']

        mines['Byproduct TCRC ratio'] = host1.values_from_dist(by_param + '_tcrc_ratio')
        mines.loc[mines['Byproduct TCRC ratio'] < 0, 'Byproduct TCRC ratio'] = mines['Byproduct TCRC ratio'].sample(
            n=(mines['Byproduct TCRC ratio'] < 0).sum(), random_state=self.rs).values
        mines['TCRC (USD/t)'] = mines['Primary TCRC (USD/t)'] / mines['Byproduct TCRC ratio']

        #         mines['Total cash cost (USD/t)'] = mines[['TCRC (USD/t)','Minesite cost (USD/t)']].sum(axis=1)
        #         mines['Total cash margin (USD/t)'] = mines['Commodity price (USD/t)'] - mines['Total cash cost (USD/t)']

        if self.verbosity > 1:
            print(
                'Currently assuming 95% byproduct recovery rate and 100% byproduct payable percent. Byproduct recovery rate is multiplied by primary recovery rate in future calculations.')

        mines.loc[mines['Byproduct ID'] == 0, 'Primary Recovery rate (%)'] = 100.
        mines['Ore treated (kt)'] = mines['Production (kt)'] / (
                    mines[['Recovery rate (%)', 'Head grade (%)', 'Primary Recovery rate (%)']].product(axis=1) / 1e6)
        mines['Primary Production (kt)'] = mines[['Ore treated (kt)', 'Primary Recovery rate (%)',
                                                         'Primary Head grade (%)']].product(axis=1) / 1e4
        mines['Capacity (kt)'] = mines['Ore treated (kt)'] / mines['Capacity utilization']
        mines['Paid metal production (kt)'] = mines[
                                                         ['Capacity (kt)', 'Capacity utilization', 'Head grade (%)',
                                                          'Recovery rate (%)', 'Primary Recovery rate (%)',
                                                          'Payable percent (%)']].product(axis=1) / 1e8
        mines['Primary Paid metal production (kt)'] = mines[
                                                                 ['Capacity (kt)', 'Capacity utilization',
                                                                  'Primary Head grade (%)', 'Primary Recovery rate (%)',
                                                                  'Primary Payable percent (%)']].product(axis=1) / 1e6

        mines['Reserves ratio with ore treated'] = host1.values_from_dist('primary_reserves')

        mines['Reserves (kt)'] = mines[['Ore treated (kt)', 'Reserves ratio with ore treated']].product(axis=1)
        mines['Reserves potential metal content (kt)'] = mines[['Reserves (kt)', 'Head grade (%)']].product(
            axis=1) * 1e-2

        mines['Cumulative ore treated ratio with ore treated'] = host1.values_from_dist('primary_ot_cumu')
        mines['Cumulative ore treated (kt)'] = mines['Cumulative ore treated ratio with ore treated'] * mines[
            'Ore treated (kt)']
        mines['Initial ore treated (kt)'] = mines['Ore treated (kt)'] / h['ramp_up_years']
        mines['Opening'] = host1.simulation_time[0] - mines[
            'Cumulative ore treated ratio with ore treated'].round(0)

        # calibrating reserves to input values if needed
        primary_reserves_reported_basis = h['primary_reserves_reported_basis']
        primary_reserves_reported = h['primary_reserves_reported']
        if primary_reserves_reported_basis == 'ore' and primary_reserves_reported > 0:
            ratio = primary_reserves_reported / mines['Reserves (kt)'].sum()
        elif primary_reserves_reported_basis == 'metal' and primary_reserves_reported > 0:
            ratio = primary_reserves_reported / mines['Reserves potential metal content (kt)'].sum()
        else:
            ratio = 1
        mines['Reserves (kt)'] *= ratio
        mines['Reserves potential metal content (kt)'] *= ratio

        host1.mines = mines.copy()
        host1.generate_oges()

        pri_main = [i for i in h if 'byproduct' in i and 'host' not in i and 'pri' not in i and i != 'byproduct']
        setattr(self, param, host1)

    def generate_byproduct_total_costs(self, param):
        by_param = 'byproduct_' + param
        host1 = getattr(self, param)
        host1.generate_costs_from_regression('Sustaining CAPEX ($M)')
        h = host1.hyperparam
        mines = host1.mines.copy()

        mines['Development CAPEX ($M)'] = 0
        mines['Primary Sustaining CAPEX ($M)'] = mines['Sustaining CAPEX ($M)']
        mines['Byproduct sCAPEX ratio'] = host1.values_from_dist(by_param + '_sus_capex_ratio')
        mines.loc[mines['Byproduct sCAPEX ratio'] < 0, 'Byproduct sCAPEX ratio'] = mines[
            'Byproduct sCAPEX ratio'].sample(n=(mines['Byproduct sCAPEX ratio'] < 0).sum(), random_state=self.rs).values
        mines['Sustaining CAPEX ($M)'] = mines['Primary Sustaining CAPEX ($M)'] / mines['Byproduct sCAPEX ratio']

        mines['Byproduct Total cash margin (USD/t)'] = mines['Commodity price (USD/t)'] - mines[
            'Minesite cost (USD/t)'] - mines['TCRC (USD/t)']
        mines['Byproduct Cash flow ($M)'] = 1e-3 * mines['Paid metal production (kt)'] * mines[
            'Byproduct Total cash margin (USD/t)'] - mines['Sustaining CAPEX ($M)']
        mines['Primary Total cash margin (USD/t)'] = mines['Primary Commodity price (USD/t)'] - mines[
            'Primary Minesite cost (USD/t)'] - mines['Primary TCRC (USD/t)']
        mines['Total cash margin (USD/t)'] = (mines['Byproduct Total cash margin (USD/t)'] * mines[
            'Paid metal production (kt)'] + mines['Primary Total cash margin (USD/t)'] * mines[
                                                         'Primary Paid metal production (kt)']) / mines[
                                                        'Primary Paid metal production (kt)']
        by_only = mines['Byproduct ID'][mines['Byproduct ID'] == 0].index
        mines.loc[by_only, 'Total cash margin (USD/t)'] = mines['Byproduct Total cash margin (USD/t)']
        mines['Cash flow ($M)'] = mines['Byproduct Cash flow ($M)'] + 1e-3 * mines[
            'Primary Paid metal production (kt)'] * mines['Primary Total cash margin (USD/t)'] - mines[
                                             'Primary Sustaining CAPEX ($M)'] - mines['Overhead ($M)'] - mines[
                                             'Development CAPEX ($M)']
        mines.loc[by_only, 'Cash flow ($M)'] = mines['Byproduct Cash flow ($M)'] - mines['Overhead ($M)'] - mines[
            'Development CAPEX ($M)']

        mines['Total reclamation cost ($M)'] = np.exp(h['primary_reclamation_constant'] +
                                                             h['primary_reclamation_slope'] * np.log(
            mines['Capacity (kt)'] / 1e3))

        mines['Byproduct ID'] = int(param.split('host')[1])
        host1.mines = mines.copy()

        pri_main = [i for i in h.index if 'byproduct' in i and 'host' not in i and 'pri' not in i and i != 'byproduct']
        setattr(self, param, host1)
        return host1

    def values_from_dist(self, param):
        hyperparam = self.hyperparam
        params = [i for i in hyperparam if param in i]
        if len(params) == 0:
            raise Exception('invalid param value given in values_from_dist call')
        else:
            dist_name = [i for i in params if 'distribution' in i][0]
            if len([i for i in params if 'distribution' in i]) > 1:
                raise Exception('993' + param + str(params))
            mean_name = [i for i in params if 'mean' in i][0]
            var_name = [i for i in params if 'var' in i][0]
            pri_dist = getattr(stats, hyperparam[dist_name])
            pri_mean = hyperparam[mean_name]
            pri_var = hyperparam[var_name]

            np.random.seed(self.rs)
            if hyperparam[dist_name] == 'norm':
                dist_rvs = pri_dist.rvs(
                    loc=pri_mean,
                    scale=pri_var,
                    size=self.mines.shape[0],
                    random_state=self.rs)
                dist_rvs[dist_rvs < 0] = np.random.choice(dist_rvs[dist_rvs > 0], len(dist_rvs[dist_rvs < 0]))
            else:
                dist_rvs = pri_dist.rvs(
                    pri_var,
                    loc=0,
                    scale=pri_mean,
                    size=self.mines.shape[0],
                    random_state=self.rs)
                dist_rvs[dist_rvs < 0] = np.random.choice(dist_rvs[dist_rvs > 0], len(dist_rvs[dist_rvs < 0]))
            if 'ratio' in param:
                dist_rvs[dist_rvs < 1] = np.random.choice(dist_rvs[dist_rvs > 1], len(dist_rvs[dist_rvs < 1]))
            return dist_rvs

    def initialize_mines(self):
        '''out: self.mine_life_init
        (copy of self.mines).
        function of hyperparam[reinitialize]'''
        self.load_variables_from_hyperparam()
        if self.hyperparam['reinitialize']:
            if self.byproduct:
                self.generate_byproduct_mines()
                # out: self.mines
            else:
                self.recalculate_hyperparams()
                self.generate_production_region()
                self.generate_grade_and_masses()
                self.generate_total_cash_margin()
                self.generate_oges()
                self.generate_annual_costs()
                # out: self.mines

            self.mine_life_init = self.mines.copy()

        elif self.hyperparam['load_mine_life_init_from_pkl']:
            if self.byproduct:
                try:
                    self.mine_life_init = pd.read_pickle('data/mine_life_init_byproduct.pkl')
                    self.mines = self.mine_life_init.copy()
                except:
                    raise Exception(
                        'Save an initialized mine file as data/mine_life_init_byproduct.pkl or set hyperparam[\'reinitialize\',\'Value\'] to True')
            else:
                try:
                    self.mine_life_init = pd.read_pickle('data/mine_life_init_primary.pkl')
                    self.mines = self.mine_life_init.copy()
                except:
                    raise Exception(
                        'Save an initialized mine file as data/mine_life_init_primary.pkl or set hyperparam[\'reinitialize\',\'Value\'] to True')

        self.update_operation_hyperparams()

    def op_initialize_prices(self):
        '''
        relevant hyperparams:
        - primary_commodity_price/byproduct
        - primary_commodity_price_option/byproduct
        - primary_commodity_price_change/byproduct

        Can bypass price series generation for pri/by
        by assigning series to primary_price_series
        or byproduct_price_series.
        '''
        h = self.hyperparam
        for price_or_tcrc in ['price', 'tcrc']:
            if self.byproduct and not hasattr(self, f'primary_{price_or_tcrc}_series') and not hasattr(self,
                                                                                                       f'byproduct_{price_or_tcrc}_series'):
                strings = ['primary_', 'byproduct_']
            elif self.byproduct and not hasattr(self, f'byproduct_{price_or_tcrc}_series'):
                strings = ['byproduct_']
            elif not hasattr(self, f'primary_{price_or_tcrc}_series'):
                strings = ['primary_']
            else:
                strings = []
            for string in strings:
                if price_or_tcrc == 'price':
                    price = h[string + 'commodity_price']
                    price_option = h[string + 'commodity_price_option']
                    price_series = pd.Series(price, self.simulation_time)
                    price_change = h[string + 'commodity_price_change']

                    if price_option == 'yoy':
                        price_series.loc[:] = [price * (1 + price_change / 100) ** n for n in
                                               np.arange(0, len(self.simulation_time))]
                    elif price_option == 'step':
                        price_series.iloc[1:] = price * (1 + price_change / 100)
                else:
                    tcrc_str = 'Primary TCRC (USD/t)' if self.byproduct and string == 'primary_' else 'TCRC (USD/t)'
                    tcrc = self.mine_life_init['TCRC (USD/t)'].mean()
                    price_series = pd.Series(tcrc, self.simulation_time)
                setattr(self, f'{string}{price_or_tcrc}_series', price_series)

    def op_initialize_mine_life(self):
        self.initialize_mines()
        self.cumulative_ore_treated = pd.Series(np.nan, self.simulation_time)
        self.supply_series = pd.Series(np.nan, self.simulation_time)
        h = self.hyperparam
        if h['reinitialize']:
            mine_life_init = self.mine_life_init.copy()
            # mine_life_init['Ramp up flag'] = 0
            mine_life_init['Ramp down flag'] = False
            mine_life_init['Closed flag'] = False
            mine_life_init['Operate with negative cash flow'] = False
            mine_life_init['Total cash margin expect (USD/t)'] = np.nan
            mine_life_init['Cash flow expect ($M)'] = np.nan
            mine_life_init['NPV ramp next ($M)'] = np.nan
            mine_life_init['NPV ramp following ($M)'] = np.nan
            mine_life_init['Close method'] = np.nan
            mine_life_init['Simulated closure'] = np.nan
            mine_life_init['Initial head grade (%)'] = mine_life_init['Head grade (%)'] / (
                        mine_life_init['Cumulative ore treated (kt)'] / mine_life_init['Initial ore treated (kt)']) ** \
                                                              mine_life_init['OGE']
            mine_life_init['Discount'] = 1
            mine_life_init['Initial price (USD/t)'] = np.nan

            if self.byproduct:
                mine_life_init['Primary Total cash margin expect (USD/t)'] = np.nan
                mine_life_init['Byproduct Cash flow expect ($M)'] = np.nan
                mine_life_init['Byproduct production flag'] = True
                mine_life_init['Byproduct Ramp up flag'] = False
                mine_life_init['Byproduct Ramp down flag'] = False
                mine_life_init['Primary Initial head grade (%)'] = mine_life_init['Primary Head grade (%)'] / (
                            mine_life_init['Cumulative ore treated (kt)'] / mine_life_init[
                        'Initial ore treated (kt)']) ** mine_life_init['OGE']
                mine_life_init = mine_life_init.fillna(0)
                mine_life_init['Primary Recovery rate (%)'] = mine_life_init[
                    'Primary Recovery rate (%)'].replace(0, 100)
                mine_life_init.loc[mine_life_init['Primary OGE'] == 0, 'Primary OGE'] = mine_life_init['OGE']
                mine_life_init['Primary Initial price (USD/t)'] = np.nan
            to_drop = ['Byproduct TCRC ratio', 'Byproduct minesite cost ratio', 'Byproduct sCAPEX ratio',
                       'Cumulative ore treated ratio with ore treated', 'Reserves ratio with ore treated']
            to_drop = mine_life_init.columns[mine_life_init.columns.isin(to_drop)]
            for j in to_drop:
                mine_life_init.drop(columns=j, inplace=True)
            self.mine_life_init = mine_life_init.copy()

        self.ml_yr = OneMine(name=self.i, df=self.mine_life_init)
        self.ml = AllMines()
        self.ml.add_mines(self.ml_yr)

        if h['forever_sim']:
            self.simulation_end = self.primary_price_series.index[-1]
        else:
            self.simulation_end = self.simulation_time[-1]

    def op_simulate_mine_life(self):
        simulation_time = self.simulation_time
        h = self.hyperparam
        i = self.i

        self.hstrings = ['primary_', 'byproduct_'] if self.byproduct else ['primary_']
        self.istrings = ['Primary ', ''] if self.byproduct else ['Primary ']
        primary_price_series = self.primary_price_series
        byproduct_price_series = self.byproduct_price_series if self.byproduct else 0

        ml_yr = self.ml.loc[i].copy() if i == simulation_time[0] else self.ml.loc[i - 1].copy()
        ml_last = ml_yr.copy()
        ml_yr.name = i
        ml0 = self.ml.loc[simulation_time[0]]

        # No longer include closed mines in the calculations → they won't have any data available after closure
        if (ml_yr.closed_flag != True).any():
            try:
                idx = (ml_last.closed_flag)&(~np.isnan(ml_last.production_kt.astype(float)))
                closed_index = ml_last.index[idx.astype(bool)]
            except Exception as e:
                print(ml_last.index)
                print(ml_last.closed_flag)
                print(ml_last.production_kt)
                print(~np.isnan(ml_last.production_kt.astype(float)))
                print((ml_last.closed_flag)&(~np.isnan(ml_last.production_kt.astype(float))))
                print(((ml_last.closed_flag)&(~np.isnan(ml_last.production_kt.astype(float)))).dtype)

                raise e
            # print(1967,closed_index)
            if len(closed_index)>0:
                ml_yr.drop(closed_index)
                ml_last.drop(closed_index)
            ml_yr.real_index = ml_yr.index
            ml_last.real_index = ml_last.index
            ml_yr.index = np.arange(0, len(ml_yr.index))
            ml_last.index = np.arange(0, len(ml_last.index))
            self.ml_last = ml_last.copy()

            # print(1978,len(ml_yr.commodity_price_usdpt))
            # if h['internal_price_formation']==False:
            if self.byproduct:
                if h['incentive_tune_tcrc']:
                    ml_yr.primary_commodity_price_usdpt = ml_yr.primary_commodity_price_usdpt * \
                                                          (primary_price_series.pct_change().fillna(0) + 1)[i]
                    ml_yr.commodity_price_usdpt = ml_yr.commodity_price_usdpt * \
                                                  (byproduct_price_series.pct_change().fillna(0) + 1)[i]
                else:
                    ml_yr.primary_tcrc_usdpt = ml_yr.primary_tcrc_usdpt * \
                                               (self.primary_tcrc_series.pct_change().fillna(0) + 1)[i]
                    ml_yr.tcrc_usdpt = ml_yr.tcrc_usdpt * (self.byproduct_tcrc_series.pct_change().fillna(0) + 1)[i]
                if i == simulation_time[0]:
                    ml_yr.initial_price_usdpt = np.repeat(byproduct_price_series[i], len(ml_yr.index))
                    ml_yr.primary_initial_price_usdpt = np.repeat(primary_price_series[i], len(ml_yr.index))
            else:
                if h['incentive_tune_tcrc']:
                    ml_yr.commodity_price_usdpt = ml_yr.commodity_price_usdpt * \
                                                  (primary_price_series.pct_change().fillna(0) + 1)[i]
                else:
                    ml_yr.tcrc_usdpt = ml_yr.tcrc_usdpt * (self.primary_tcrc_series.pct_change().fillna(0) + 1)[i]
                if i == simulation_time[0]:
                    ml_yr.initial_price_usdpt = np.repeat(primary_price_series[i], len(ml_yr.index))

            self.sxew_mines = ml_yr.index[ml_yr.payable_percent_pct == 100]
            self.conc_mines = ml_yr.index[((ml_yr.payable_percent_pct != 100)&(~np.isnan(ml_yr.payable_percent_pct.astype(float)))).astype(bool)]
            ml_yr.tcrc_usdpt[self.sxew_mines] = 0
            if self.byproduct:
                ml_yr.primary_tcrc_usdpt[self.sxew_mines] = 0
            closing_mines = ml_last.index[ml_last.ramp_down_flag.astype(bool)]
            opening_mines = ml_yr.index[ml_yr.ramp_up_flag != 0]
            not_new_opening = ml_yr.index[ml_yr.ramp_up_flag!=2] if i>self.simulation_time[1] else ml_yr.index
            # print(2014,len(not_new_opening),len(ml_yr.index))
            govt_mines = ml_yr.index[ml_yr.operate_with_negative_cash_flow.astype(bool)]
            #             end_ramp_up = ml_yr.loc[(opening_mines)&(ml_yr['Opening']+h['ramp_up_years']<=i)&(ml_yr['Opening']>simulation_time[0]-1)].index
            end_ramp_up = ml_yr.index[ml_yr.ramp_up_flag == h['ramp_up_years'] + 1]
            if len(opening_mines) > 0:
                ml_yr.opening[np.intersect1d(opening_mines, ml_yr.index[np.isnan(ml_yr.opening.astype(float))])] = i
                if self.byproduct:
                    ml_yr.initial_price_usdpt[ml_yr.opening == i] = byproduct_price_series[i]
                    ml_yr.primary_initial_price_usdpt[ml_yr.opening == i] = primary_price_series[i]
                else:
                    ml_yr.initial_price_usdpt[ml_yr.opening == i] = primary_price_series[i]

                if h['ramp_up_exponent'] != 0:
                    ramp_up_exp = h['ramp_up_exponent']
                    if self.byproduct:
                        for b in ml_yr['Byproduct ID'].unique():
                            print(1470, 'could be a spot for error if ml0 does not have the right CU value')
                            id_zero = ml0.index[ml0.byproduct_id == b]
                            by_open = np.intersect1d(id_zero, opening_mines)
                            ml_yr.capacity_utilization[by_open] = np.nanmean(ml0.capacity_utilization[id_zero]) * (
                                        h['ramp_up_cu'] + (1 - h['ramp_up_cu']) / (
                                h['ramp_up_years']) ** ramp_up_exp * (ml_yr.ramp_up_flag[by_open] - 1) ** ramp_up_exp)
                    else:
                        ml_yr.capacity_utilization[opening_mines] = h['mine_cu0'] * (
                                    h['ramp_up_cu'] + (1 - h['ramp_up_cu']) / (h['ramp_up_years']) ** ramp_up_exp * (
                                        ml_yr.ramp_up_flag[opening_mines] - 1) ** ramp_up_exp)
                else:
                    ml_yr.capacity_utilization[opening_mines] = h['ramp_up_cu']
                ml_yr.ramp_up_flag[opening_mines] = ml_yr.ramp_up_flag[opening_mines] + 1

            if len(end_ramp_up) > 0:
                ml_yr.capacity_utilization[end_ramp_up] = self.calculate_cu(h['mine_cu0'],
                                                                            ml_last.total_cash_margin_usdpt[
                                                                                end_ramp_up], govt=False)
                ml_yr.ramp_up_flag[end_ramp_up] = 0
                ml_yr.development_capex_usdm[end_ramp_up] = 0
            # Correcting to deal with government mines → 'Operate with negative cash flow' mines. Sets to enter ramp down if reserves have become smaller than prior year's ore treated.
            reserves_less_than_ot = ml_last.index[ml_last.reserves_kt < ml_last.ore_treated_kt]
            closing_mines = [i for i in closing_mines if i not in govt_mines or i in reserves_less_than_ot]

            unio = np.union1d(closing_mines, opening_mines)
            unio = np.union1d(unio, end_ramp_up)
            normal_mines = [i for i in ml_yr.index if i not in unio]
            ml_yr.capacity_utilization[normal_mines] = self.calculate_cu(ml_last.capacity_utilization[normal_mines],
                                                                         ml_last.total_cash_margin_usdpt[normal_mines])
            ml_yr.capacity_utilization[closing_mines] = h['ramp_down_cu']
            ml_yr.capacity_utilization[ml_yr.capacity_utilization > h['cu_cutoff']] = h['cu_cutoff']
            ml_yr.ore_treated_kt = ml_yr.capacity_utilization * ml_yr.capacity_kt
            ml_yr.initial_ore_treated_kt[ml_yr.initial_ore_treated_kt == 0] = ml_yr.ore_treated_kt[
                ml_yr.initial_ore_treated_kt == 0]
            if i > simulation_time[0]:
                ml_yr.cumulative_ore_treated_kt = ml_yr.cumulative_ore_treated_kt + ml_yr.ore_treated_kt
                ml_yr.reserves_kt = ml_yr.reserves_kt - ml_yr.ore_treated_kt
            else:
                ml_yr.simulation_start_ore_treated_kt = ml_yr.ore_treated_kt
                ml_yr.generated_tcrc_usdpt = ml_yr.tcrc_usdpt

            # This section handles opening mines, since with the ore treated-based minesite cost calculation, prices were rising far too quickly if the actual simulation start ore treated was used. The normal-operation value is our target anyway.
            first_opening = ml_yr.ramp_up_flag == 2
            if np.nansum(first_opening) > 0 and h['minesite_cost_response_to_grade_price']:
                if self.byproduct:
                    for b in ml_yr['Byproduct ID'].unique():
                        id_zero = ml0.index[ml0.byproduct_id == b]
                        by_open = np.intersect1d(id_zero, opening_mines)
                        print(1501, 'could be a spot for error if ml0 does not have the right CU value')
                        ml_yr.simulation_start_ore_treated_kt[by_open] = ml_yr.ore_treated_kt[by_open] * np.nanmean(
                            ml0.capacity_utilization[ml0.byproduct_id == b]) / ml_yr.capacity_utilization
                else:
                    ml_yr.simulation_start_ore_treated_kt[opening_mines] = ml_yr.ore_treated_kt[opening_mines] * h[
                        'mine_cu0'] / ml_yr.capacity_utilization[opening_mines]

            ml_yr.head_grade_pct = self.calculate_grade(ml_yr.initial_head_grade_pct, ml_yr.cumulative_ore_treated_kt,
                                                        ml_yr.initial_ore_treated_kt, ml_yr.oge)

            if self.byproduct:
                if i != simulation_time[0]:
                    for j in np.arange(1, 4):
                        by_id = ml_yr.byproduct_id == j
                        ml_yr.recovery_rate_pct[by_id] = (ml_last.recovery_rate_pct[by_id] * (
                                    ml_last.byproduct_total_cash_margin_usdpt[by_id] / h[
                                'byproduct' + str(j) + '_mine_tcm0']) ** h['byproduct_rr_margin_elas'])
                        ml_yr.recovery_rate_pct[ml_yr.recovery_rate_pct == np.nan] = -1
                        ml_yr.recovery_rate_pct[
                            (by_id) & (ml_yr.recovery_rate_pct > h['byproduct' + str(j) + '_mine_rrmax'])] = h[
                            'byproduct' + str(j) + '_mine_rrmax']
                        problem = ml_yr.recovery_rate[by_id] == -1
                        problem = ml_yr.index[by_id][problem]
                        if len(problem) > 0:
                            ml_yr.recovery_rate_pct[problem] = ml_last.recovery_rate_pct[problem]
                #                         print('Last recovery')
                #                         display(ml_last.loc[problem,'Recovery rate (%)'])
                #                         print('last tcm')
                #                         display(ml_last.loc[problem,'Byproduct Total cash margin (USD/t)'])
                #                         print('other')
                #                         print(h['byproduct'+str(j)+'_mine_tcm0'],h['byproduct_rr_margin_elas'])
                ml_yr.primary_head_grade_pct = self.calculate_grade(ml_yr.primary_initial_head_grade_pct,
                                                                    ml_yr.cumulative_ore_treated_kt,
                                                                    ml_yr.initial_ore_treated_kt, ml_yr.oge)
                ml_yr.primary_head_grade_pct[ml_yr.primary_head_grade_pct == np.nan] = 0
                ml_yr.head_grade_pct[ml_yr.byproduct_id != 0] = ml_yr.primary_head_grade[ml_yr.byproduct_id != 0] / \
                                                                ml_yr.byproduct_grade_ratio[ml_yr.byproduct_id != 0]

            ml_yr.closed_flag[closing_mines] = True
            ml_yr.simulated_closure[closing_mines] = i
            # 2 == NPV following
            ml_yr.ramp_down_flag[[q == 2 for q in ml_yr.close_method]] = True

            self.ml_yr1 = ml_yr.copy()
            self.ml_last1 = ml_last.copy()
            # print(2113,len(ml_yr.commodity_price_usdpt))

            ml_yr.minesite_cost_usdpt = self.calculate_minesite_cost(
                ml_last.minesite_cost_usdpt, ml_yr.head_grade_pct, ml_last.head_grade_pct,
                ml_yr.commodity_price_usdpt, ml_last.commodity_price_usdpt,
                ml_yr.ore_treated_kt, ml_yr.simulation_start_ore_treated_kt, i)
            if self.byproduct:
                ml_yr.primary_minesite_cost_usdpt = self.calculate_minesite_cost(
                    ml_last.primary_minesite_cost_usdpt, ml_yr.primary_head_grade_pct, ml_last.primary_head_grade_pct,
                    ml_yr.primary_commodity_price_usdpt, ml_last.primary_commodity_price_usdpt,
                    ml_yr.ore_treated_kt, ml_yr.simulation_start_ore_treated_kt, i)

            ml_yr.production_kt = self.calculate_production(ml_yr)
            ml_yr.paid_metal_production_kt = self.calculate_paid_metal_prod(ml_yr)
            if self.byproduct:
                ml_yr.production_kt = ml_yr.production_kt * ml_yr.primary_recovery_rate_pct / 100
                ml_yr.paid_metal_production_kt = ml_yr.paid_metal_production_kt * ml_yr.primary_recovery_rate_pct / 100

                ml_yr.primary_production_kt = ml_yr.ore_treated_kt * ml_yr.primary_recovery_rate_pct * ml_yr.primary_head_grade_pct / 1e4
                ml_yr.primary_paid_metal_production_kt = ml_yr.ore_treated_kt * ml_yr.primary_recovery_rate_pct * ml_yr.primary_head_grade_pct * ml_yr.primary_payable_percent_pct / 1e6
                ml_yr.byproduct_total_cash_margin_usdpt = ml_yr.commodity_price_usdpt - ml_yr.minesite_cost_usdpt - ml_yr.tcrc_usdpt
                ml_yr.byproduct_cash_flow_usdm = 1e-3 * ml_yr.paid_metal_production_kt * ml_yr.byproduct_total_cash_margin_usdpt - ml_yr.sustaining_capex_usdm
                ml_yr.primary_total_cash_margin_usdpt = ml_yr.primary_commodity_price_usdpt - ml_yr.primary_minesite_cost_usdpt - ml_yr.primary_tcrc_usdpt
                ml_yr.total_cash_margin_usdpt = (
                                                            ml_yr.byproduct_total_cash_margin_usdpt * ml_yr.paid_metal_production_kt + ml_yr.primary_total_cash_margin_usdpt * ml_yr.primary_paid_metal_production_kt) / ml_yr.primary_paid_metal_production_kt
                by_only = ml_yr.index[ml_yr.byproduct_id == 0]
                ml_yr.total_cash_margin_usdpt[by_only] = ml_yr.byproduct_total_cash_margin_usdpt[by_only]
                ml_yr.cash_flow_usdm = ml_yr.byproduct_cash_flow_usdm + 1e-3 * ml_yr.primary_paid_metal_production_kt * ml_yr.primary_total_cash_margin_usdpt - ml_yr.primary_sustaining_capex_usdm - ml_yr.overhead_usdm - ml_yr.development_capex_usdm
                ml_yr.cash_flow_usdm[by_only] = ml_yr.byproduct_cash_flow_usdm[by_only] - ml_yr.overhead_usdm[by_only] - \
                                                ml_yr.development_capex_usdm[by_only]

                ml_yr.primary_revenue_usdm = ml_yr.primary_paid_metal_production_kt * ml_yr.primary_total_cash_margin_usdpt * 1e-3
                ml_yr.byproduct_revenue_usdm = 1e-3 * ml_yr.paid_metal_production_kt * ml_yr.byproduct_total_cash_margin_usdpt
                ml_yr.byproduct_revenue_fraction = ml_yr.byproduct_revenue_usdm / (
                            ml_yr.byproduct_revenue_usdm + ml_yr.primary_revenue_usdm)
                ml_yr.byproduct_revenue_fraction[
                    (ml_yr.byproduct_revenue_fraction > 1) | (ml_yr.byproduct_revenue_fraction < 0)] = np.nan
            else:
                ml_yr.total_cash_margin_usdpt = ml_yr.commodity_price_usdpt - ml_yr.minesite_cost_usdpt - ml_yr.tcrc_usdpt
                ml_yr.cash_flow_usdm = 1e-3 * ml_yr.paid_metal_production_kt * ml_yr.total_cash_margin_usdpt - ml_yr.overhead_usdm - ml_yr.sustaining_capex_usdm - ml_yr.development_capex_usdm

            self.govt_mines = ml_yr.index[ml_yr.operate_with_negative_cash_flow.astype(bool)]

            self.ml_yr = ml_yr.copy()
            if self.byproduct:
                price_df = byproduct_price_series.copy()
            else:
                price_df = primary_price_series.copy()
            # price_df = self.ml['Commodity price (USD/t)'].unstack().copy()
            # price_df.loc[i,:] = ml_yr['Commodity price (USD/t)']
            price_expect = self.calculate_price_expect(price_df, i)
            ml_yr.price_expect_usdpt = ml_yr.commodity_price_usdpt * price_expect / np.nanmean(ml_yr.commodity_price_usdpt)

            # Simplistic byproduct production approach: no byprod prod when byprod cash flow<0 for that year
            if i > simulation_time[0]:
                if self.byproduct: ml_yr = self.byproduct_closure(ml_yr)

            # Check for mines with negative cash flow that should ramp down next year
            ml_yr = self.check_ramp_down(ml_yr, price_df, ml_yr.price_expect_usdpt)
            # 2 is for NPV following
            ml_yr.ramp_down_flag[[q == 2 for q in ml_yr.close_method]] = True

        ml_yr.index = ml_yr.real_index
        ml_last.index = ml_last.real_index
        self.ml_yr = ml_yr.copy()
        self.ml.add_mines(ml_yr.copy())
        if i > self.simulation_time[0]:
            self.cumulative_ore_treated.loc[i] = self.cumulative_ore_treated.loc[i - 1] + np.nansum(ml_yr.ore_treated_kt)
        else:
            self.cumulative_ore_treated.loc[i] = np.nansum(ml_yr.cumulative_ore_treated_kt)

    def simulate_mine_life_one_year(self):
        h = self.hyperparam
        i = self.i
        simulation_time = self.simulation_time
        if i == self.simulation_time[0] and not h['simulate_history_bool']:
            self.op_initialize_mine_life()

        if i == self.simulation_time[0]:
            self.op_initialize_prices()

        if i == self.simulation_time[0] and not h['simulate_history_bool']:
            self.op_simulate_mine_life()
            self.update_operation_hyperparams(innie=self.ml_yr)
        elif h['simulate_history_bool']:
            self.simulate_history()
        else:
            self.op_simulate_mine_life()
        if h['simulate_opening'] and i > self.simulation_time[0]:
            self.simulate_mine_opening()
            opening = self.mines_to_open.copy()
            if h['incentive_opening_method'] in ['karan_generalization', 'unconstrained']:
                self.update_price_tcrc()
                opening = self.opening.copy()

            if type(opening) == OneMine:
                self.ml_yr.concat(opening)
                self.ml.add_mines(self.ml_yr)
        ml_yr_ph = self.ml.loc[i]
        if self.byproduct:
            self.primary_price_series.loc[i] = np.nanmean(ml_yr_ph.primary_commodity_price_usdpt)
            self.primary_tcrc_series.loc[i] = np.nanmean(ml_yr_ph.primary_tcrc_usdpt)
            self.byproduct_price_series.loc[i] = np.nanmean(ml_yr_ph.commodity_price_usdpt)
            self.byproduct_tcrc_series.loc[i] = np.nanmean(ml_yr_ph.tcrc_usdpt)
        else:
            self.primary_price_series.loc[i] = np.nanmean(ml_yr_ph.commodity_price_usdpt)
            # print(2225,ml_yr_ph.tcrc_usdpt)
            ml_yr_ph.tcrc_usdpt = ml_yr_ph.tcrc_usdpt.astype(float)
            self.primary_tcrc_series.loc[i] = np.nanmean(ml_yr_ph.tcrc_usdpt)

        self.supply_series.loc[i] = np.nansum(ml_yr_ph.production_kt)
        self.concentrate_supply_series.loc[i] = np.nansum(ml_yr_ph.production_kt[ml_yr_ph.payable_percent_pct != 100])
        self.sxew_supply_series.loc[i] = np.nansum(ml_yr_ph.production_kt[ml_yr_ph.payable_percent_pct == 100])

    def calculate_cu(self, cu_last, tcm_last, govt=True):
        neg_tcm = tcm_last < 0
        cu = cu_last * abs(tcm_last / self.hyperparam['mine_tcm0']) ** self.hyperparam[
            'mine_cu_margin_elas']
        cu[neg_tcm] = 0.7
        #         ind = np.intersect1d(cu.index,self.govt_mines)
        #         ind = np.intersect1d(ind, cu.loc[cu==0.7].index)

        #         if govt:
        #             cu.loc[ind] = cu_last.loc[ind]
        # government mines (operating under negative cash flow) may have negative tcm and therefore
        # get their CU set to 0.7. Decided it is better to set them to their previous value instead
        return cu

    def calculate_grade(self, initial_grade, cumu_ot, initial_ot, oge=0):
        ''' '''
        grade = initial_grade * (cumu_ot / initial_ot) ** oge
        grade[cumu_ot == 0] = initial_grade[cumu_ot == 0]
        grade[grade < 0] = 1e-6
        grade[grade > 80] = 80
        return grade

    def calculate_minesite_cost(self, minesite_cost_last, grade, initial_grade, price, initial_price, ore_treated, sim_start_ore_treated, year_i):

        h = self.hyperparam
        if h['minesite_cost_response_to_grade_price']:
            minesite_cost_expect = minesite_cost_last * (grade / initial_grade) ** h['mine_cost_og_elas'] \
                                   * (price / initial_price) ** h['mine_cost_price_elas'] \
                                   * (1 + h['mine_cost_change_per_year'] / 100) ** (year_i - self.simulation_time[0])
        else:
            minesite_cost_expect = minesite_cost_last
        return minesite_cost_expect

    def calculate_paid_metal_prod(self, ml_yr):
        return ml_yr.ore_treated_kt * ml_yr.recovery_rate_pct * ml_yr.head_grade_pct * ml_yr.payable_percent_pct / 1e6

    def calculate_production(self, ml_yr):
        return ml_yr.ore_treated_kt * ml_yr.recovery_rate_pct * ml_yr.head_grade_pct / 1e4

    def calculate_cash_flow(self, ml_yr):
        '''Intended to skip over some steps and just give cash flows
        for ramp down evaluation. Returns cash flow series.'''
        paid_metal = self.calculate_paid_metal_prod(ml_yr)
        tcm = ml_yr.commodity_price_usdpt - ml_yr.minesite_cost_usdpt
        return paid_metal * tcm

    def calculate_price_expect(self, ml, i):
        '''i is year index'''
        close_price_method = self.hyperparam['close_price_method']
        close_years_back = int(self.hyperparam['close_years_back'])
        close_probability_split_max = self.hyperparam['close_probability_split_max']
        close_probability_split_mean = self.hyperparam['close_probability_split_mean']
        close_probability_split_min = self.hyperparam['close_probability_split_min']

        if len(ml.shape) > 1 and ml.shape[1] > 1:
            ml = ml.apply(lambda x: x.replace(np.nan, x.mean()), axis=1)

        # Process the dataframe of mines to return mine-level price expectation info (series of expected prices for each mine)
        if close_price_method == 'mean':
            if len(ml.index) <= close_years_back:
                price_expect = ml.mean()
            else:
                price_expect = ml.loc[i - close_years_back:i].mean()
        elif close_price_method == 'max':
            if len(ml.index) <= close_years_back:
                price_expect = ml.max()
            else:
                price_expect = ml.loc[i - close_years_back:i].max()
        elif close_price_method == 'probabilistic':
            if len(ml.index) <= close_years_back:
                price_expect_min = ml.min()
                price_expect_mean = ml.mean()
                price_expect_max = ml.max()
            else:
                price_expect_min = ml.loc[i - close_years_back:i].min()
                price_expect_mean = ml.loc[i - close_years_back:i].mean()
                price_expect_max = ml.loc[i - close_years_back:i].max()
            price_expect = close_probability_split_max * price_expect_max + \
                           close_probability_split_mean * price_expect_mean + \
                           close_probability_split_min * price_expect_min
        elif close_price_method == 'alonso-ayuso':
            # from Alonso-Ayuso et al (2014). Medium range optimization of copper extraction planning under uncertainty in future copper prices
            # slide 67 in C:\Users\ryter\Dropbox (MIT)\Group Research Folder_Olivetti\Displacement\04 Presentations\John\Weekly Updates\20210825 Generalization.pptx
            price_base = ml.loc[i]
            price_expect = price_base * 1.35 * 0.5 + price_base * 0.65 * 1 / 6 + price_base * 1 / 3
        return price_expect

    def check_ramp_down(self, ml_yr_, price_df, price_expect):
        h = self.hyperparam.copy()
        ml_yr = ml_yr_.copy()
        discount_rate = h['discount_rate']
        use_reserves_for_closure = h['use_reserves_for_closure']
        first_yr = self.simulation_time[0]
        i = self.i

        overhead = ml_yr.overhead_usdm
        sustaining_capex = ml_yr.sustaining_capex_usdm
        development_capex = ml_yr.development_capex_usdm

        capacity = ml_yr.capacity_kt
        initial_grade = ml_yr.initial_head_grade_pct
        initial_ore_treated = ml_yr.initial_ore_treated_kt
        initial_price = ml_yr.initial_price_usdpt
        oge = ml_yr.oge

        cu_expect = self.calculate_cu(ml_yr.capacity_utilization, ml_yr.total_cash_margin_usdpt)
        ml_yr.capacity_utilization_expect = cu_expect
        ot_expect = cu_expect * capacity
        cumu_ot_expect = ml_yr.cumulative_ore_treated_kt + ot_expect

        # tcrc_df = self.ml['TCRC (USD/t)'].unstack()
        # tcrc_df.loc[i,:] = ml_yr['TCRC (USD/t)']
        tcrc_expect = self.calculate_price_expect(self.primary_tcrc_series.copy(), i)
        ml_yr.tcrc_expect_usdpt = np.repeat(0, len(ml_yr.index)) # SX-EW mines have tcrc of zero so only set conc mines below
        if len(self.conc_mines)>0:
            if tcrc_expect>1e12: tcrc_expect=1e12
            ml_yr.tcrc_usdpt[ml_yr.tcrc_usdpt>1e12] = 1e12
            ml_yr.tcrc_usdpt = ml_yr.tcrc_usdpt.astype(float)
            ml_yr.tcrc_expect_usdpt[self.conc_mines] = ml_yr.tcrc_usdpt[self.conc_mines]*tcrc_expect/np.nanmean(ml_yr.tcrc_usdpt[self.conc_mines])
        tcrc_expect = ml_yr.tcrc_expect_usdpt

        if self.byproduct:
            # pri_price_df = self.ml['Primary Commodity price (USD/t)'].unstack()
            # pri_price_df.loc[i,:] = ml_yr['Primary Commodity price (USD/t)']
            pri_price_expect = self.calculate_price_expect(self.primary_price_series.copy(), i)
            pri_initial_price = ml_yr.primary_initial_price_usdpt
            ml_yr.primary_price_expect_usdpt = pri_price_expect

            # pri_tcrc_df = self.ml['Primary TCRC (USD/t)'].unstack()
            # pri_tcrc_df.loc[i,:] = ml_yr['Primary TCRC (USD/t)']
            pri_tcrc_expect = self.calculate_price_expect(self.primary_tcrc_series.copy(), i)
        else:
            pri_initial_price, pri_price_expect, pri_tcrc_expect = 0, 0, 0

        cash_flow_expect, by_cash_flow_expect, tcm_expect, by_tcm_expect, ml_yr = self.get_cash_flow(ml_yr,
                                                                                                     cumu_ot_expect,
                                                                                                     ot_expect,
                                                                                                     initial_ore_treated,
                                                                                                     initial_grade,
                                                                                                     price_expect,
                                                                                                     tcrc_expect,
                                                                                                     initial_price,
                                                                                                     overhead,
                                                                                                     sustaining_capex,
                                                                                                     development_capex,
                                                                                                     pri_initial_price,
                                                                                                     pri_price_expect,
                                                                                                     pri_tcrc_expect,
                                                                                                     neg_cash_flow=0)

        ml_yr.cash_flow_expect_usdm = cash_flow_expect
        ml_yr.total_cash_margin_expect_usdpt = tcm_expect
        if self.byproduct:
            ml_yr.byproduct_total_cash_margin_expect_usdpt = by_tcm_expect
            ml_yr.byproduct_cash_flow_expect_usdm = by_cash_flow_expect

        if ml_yr.shape()[0] == 0 or np.nansum(np.isnan(ml_yr.reserves_kt.astype(float)) == len(ml_yr.reserves_kt)):
            return ml_yr

        exclude_this_yr_reserves = ml_yr.index[ml_yr.reserves_kt < ot_expect]
        exclude_already_ramping = ml_yr.index[ml_yr.ramp_down_flag.astype(bool)]
        exclude_ramp_up = ml_yr.index[ml_yr.ramp_up_flag != 0]
        neg_cash_flow = ml_yr.index[ml_yr.cash_flow_expect_usdm < 0]
        if use_reserves_for_closure:
            exclude = list(exclude_this_yr_reserves) + list(exclude_already_ramping) + list(exclude_ramp_up)
        else:
            exclude = list(exclude_already_ramping) + list(exclude_ramp_up)

        neg_cash_flow = [i for i in neg_cash_flow if i not in exclude]

        ml_yr.cu_ramp_following = np.repeat(np.nan, len(ml_yr.index))
        ml_yr.ore_treat_ramp_following_kt = np.repeat(np.nan, len(ml_yr.index))
        ml_yr.ore_treat_expect_kt = ot_expect
        ml_yr.npv_ramp_next_usdm = np.repeat(np.nan, len(ml_yr.index))
        ml_yr.npv_ramp_following_usdm = np.repeat(np.nan, len(ml_yr.index))

        if len(neg_cash_flow) > 0:
            if self.verbosity > 1:
                print('len neg cash flow >0', self.i)
            reclamation = ml_yr.total_reclamation_cost_usdm[neg_cash_flow]

            # those with reserves likely to be depleted in the year following
            if use_reserves_for_closure:
                reserve_violation = ml_yr.reserves_kt[neg_cash_flow] < ot_expect[neg_cash_flow] + capacity[
                    neg_cash_flow] * self.ramp_down_cu
                reserve_violation = ml_yr.index[reserve_violation]
                ml_yr.ore_treat_ramp_following_kt[reserve_violation] = ml_yr.reserves_kt[reserve_violation] - ot_expect[
                    reserve_violation]
                ml_yr.cu_ramp_following[reserve_violation] = ml_yr.ore_treat_ramp_following_kt[reserve_violation] / \
                                                             capacity[reserve_violation]
                neg_cash_flow = [i for i in neg_cash_flow if i not in reserve_violation]

            ot_ramp_following = h['ramp_down_cu'] * capacity[neg_cash_flow]
            cumu_ot_expect = cumu_ot_expect[neg_cash_flow]
            initial_ore_treated = initial_ore_treated[neg_cash_flow]
            initial_grade = initial_grade[neg_cash_flow]
            initial_price = initial_price[neg_cash_flow]
            price_expect = price_expect[neg_cash_flow]
            tcrc_expect = tcrc_expect[neg_cash_flow]
            overhead = overhead[neg_cash_flow]
            sustaining_capex = sustaining_capex[neg_cash_flow]
            development_capex = development_capex[neg_cash_flow]

            if self.byproduct:
                pri_initial_price = pri_initial_price[neg_cash_flow]

            # Back to all neg_cash_flow mines, evaluate cash flow for ramp following
            cumu_ot_ramp_following = cumu_ot_expect + ot_ramp_following
            # print(2432,
            #                                                                                                  len(cumu_ot_ramp_following),
            #                                                                                                  len(ot_ramp_following),
            #                                                                                                  len(initial_ore_treated),
            #                                                                                                  len(initial_grade),
            #                                                                                                  len(price_expect),
            #                                                                                                  len(tcrc_expect),
            #                                                                                                  len(initial_price),
            #                                                                                                  len(overhead),
            #                                                                                                  len(sustaining_capex),
            #                                                                                                  len(development_capex))

            cash_flow_ramp_following, by_cash_flow_ramp_following, tcm_rf, by_tcm_rf, _ = self.get_cash_flow(ml_yr,
                                                                                                             cumu_ot_ramp_following,
                                                                                                             ot_ramp_following,
                                                                                                             initial_ore_treated,
                                                                                                             initial_grade,
                                                                                                             price_expect,
                                                                                                             tcrc_expect,
                                                                                                             initial_price,
                                                                                                             overhead,
                                                                                                             sustaining_capex,
                                                                                                             development_capex,
                                                                                                             pri_initial_price,
                                                                                                             pri_price_expect,
                                                                                                             pri_tcrc_expect,
                                                                                                             neg_cash_flow=neg_cash_flow)

            # More all neg_cash_flow mines, evaluating cash flow for ramp down in the next year
            ot_ramp_next = h['ramp_down_cu'] * capacity[neg_cash_flow]
            cumu_ot_ramp_next = ml_yr.ore_treated_kt[neg_cash_flow] + ot_ramp_next
            cash_flow_ramp_next, by_cash_flow_ramp_next, tcm_rn, by_tcm_rn, _ = self.get_cash_flow(ml_yr,
                                                                                                   cumu_ot_ramp_next,
                                                                                                   ot_ramp_next,
                                                                                                   initial_ore_treated,
                                                                                                   initial_grade,
                                                                                                   price_expect,
                                                                                                   tcrc_expect,
                                                                                                   initial_price,
                                                                                                   overhead,
                                                                                                   sustaining_capex,
                                                                                                   development_capex,
                                                                                                   pri_initial_price,
                                                                                                   pri_price_expect,
                                                                                                   pri_tcrc_expect,
                                                                                                   neg_cash_flow=neg_cash_flow)
            npv_ramp_following = cash_flow_expect[neg_cash_flow] + cash_flow_ramp_following / (
                        1 + discount_rate) - reclamation / (1 + discount_rate) ** 2
            npv_ramp_next = cash_flow_ramp_next - reclamation / (1 + discount_rate)

            ml_yr.npv_ramp_next_usdm[neg_cash_flow] = npv_ramp_next
            ml_yr.npv_ramp_following_usdm[neg_cash_flow] = npv_ramp_following

            ramp_down_next = npv_ramp_next > npv_ramp_following
            ramp_down_next = ml_yr.index[neg_cash_flow][ramp_down_next]
            ramp_down_following = npv_ramp_next < npv_ramp_following
            ramp_down_following = ml_yr.index[neg_cash_flow][ramp_down_following]
            ml_yr.ramp_down_flag[ramp_down_next] = True
            # ml_yr.drop(columns=['CU ramp following','Ore treat ramp following','Ore treat expect'],inplace=True)

            # 1 is for 'NPV next', 2 is for 'NPV following'
            ml_yr.close_method[ramp_down_next] = 1
            ml_yr.close_method[ramp_down_following] = 2

        return ml_yr

    def byproduct_closure(self, ml_yr_):
        ml_yr = ml_yr_.copy()
        byp1 = ml_yr['Byproduct Cash flow ($M)'][ml_yr['Byproduct Cash flow ($M)'] < 0].index
        byp2 = ml_yr['Byproduct ID'][ml_yr['Byproduct ID'] != 0].index
        byp = np.intersect1d(byp1, byp2)

        if len(byp) != 0:
            ml_yr.loc[byp, 'Production (kt)'] = 0
            ml_yr.loc[byp, 'Paid metal production (kt)'] = 0

            ml_yr.loc[byp, 'Byproduct Cash flow ($M)'] = 0
            ml_yr.loc[byp, 'Total cash margin (USD/t)'] = ml_yr['Primary Total cash margin (USD/t)']
            ml_yr.loc[byp, 'Cash flow ($M)'] = ml_yr['Primary Paid metal production (kt)'] * ml_yr[
                'Primary Total cash margin (USD/t)'] - ml_yr['Overhead ($M)'] - ml_yr['Primary Sustaining CAPEX ($M)'] - \
                                               ml_yr['Development CAPEX ($M)']
        return ml_yr

    def get_cash_flow(self, ml_yr_, cumu_ot_expect, ot_expect, initial_ore_treated, initial_grade, price_expect, tcrc_expect, initial_price, overhead, sustaining_capex, development_capex, pri_initial_price, pri_price_expect, pri_tcrc_expect, neg_cash_flow):
        by_cash_flow_expect = 0
        by_tcm_expect = 0
        ml_yr = ml_yr_.copy()
        h = self.hyperparam
        grade_expect = self.calculate_grade(initial_grade, cumu_ot_expect, initial_ore_treated,
                                            ml_yr.oge[neg_cash_flow])
        sxew_index = ml_yr.index[ml_yr.payable_percent_pct == 100]

        if self.byproduct:
            if self.i != self.simulation_time[0]:
                for j in np.arange(1, 4):
                    byprod_id = ml_yr.index[ml_yr.byproduct_id == j]
                    ml_yr.recovery_rate_pct[byprod_id] = (ml_yr.recovery_rate_pct[byprod_id] * (
                                ml_yr.byproduct_total_cash_margin_usdpt[byprod_id] / h[
                            'byproduct' + str(j) + '_mine_tcm0']) ** h['byproduct_rr_margin_elas'])
                    rec_rate_exceed = ml_yr.index[
                        ml_yr.recovery_rate_pct[byprod_id] > h['byproduct' + str(j) + '_mine_rrmax']]
                    ml_yr.recovery_rate_pct[np.intersect1d(byprod_id, rec_rate_exceed)] = h[
                        'byproduct' + str(j) + '_mine_rrmax']
            pri_grade_expect = self.calculate_grade(ml_yr.primary_initial_head_grade_pct, cumu_ot_expect,
                                                    ml_yr.initial_ore_treated_kt, ml_yr.oge)
            grade_expect[ml_yr.byproduct_id != 0] = pri_grade_expect / ml_yr.byproduct_grade_ratio

        #         ml_yr['Head grade expect (%)'] = grade_expect
        # print(2529,neg_cash_flow)
        # print(2530, len(grade_expect), len(price_expect),
        #                                                     len(initial_price), len(ot_expect))
        minesite_cost_expect = self.calculate_minesite_cost(ml_yr.minesite_cost_usdpt[neg_cash_flow], grade_expect,
                                                            ml_yr.head_grade_pct[neg_cash_flow], price_expect,
                                                            initial_price, ot_expect,
                                                            ml_yr.simulation_start_ore_treated_kt[neg_cash_flow],
                                                            self.i + 1)
        #         ml_yr['Minesite cost expect (USD/t)'] = minesite_cost_expect
        #         ml_yr['Price expect (USD/t)'] = price_expect
        paid_metal_expect = ot_expect * grade_expect * ml_yr.recovery_rate_pct[neg_cash_flow] * \
                            ml_yr.payable_percent_pct[neg_cash_flow] * 1e-6
        tcm_expect = price_expect - minesite_cost_expect - tcrc_expect
        cash_flow_expect = 1e-3 * paid_metal_expect * tcm_expect - overhead - sustaining_capex - development_capex

        if self.byproduct:
            paid_metal_expect = paid_metal_expect * ml_yr.primary_recovery_rate_pct[neg_cash_flow] / 100
            by_cash_flow_expect = 1e-3 * paid_metal_expect * tcm_expect - sustaining_capex

            pri_paid_metal_expect = ot_expect * pri_grade_expect * ml_yr.primary_recovery_rate_pct[neg_cash_flow] * \
                                    ml_yr.primary_payable_percent_pct[neg_cash_flow] / 1e6
            pri_minesite_cost_expect = self.calculate_minesite_cost(ml_yr.primary_minesite_cost_usdpt[neg_cash_flow],
                                                                    pri_grade_expect,
                                                                    ml_yr.primary_head_grade_pct[neg_cash_flow],
                                                                    pri_price_expect, pri_initial_price, ot_expect,
                                                                    ml_yr.simulation_start_ore_treated_kt, self.i + 1)

            pri_tcm_expect = pri_price_expect - pri_minesite_cost_expect - pri_tcrc_expect
            by_tcm_expect = tcm_expect.copy()
            tcm_expect = (
                                     by_tcm_expect * paid_metal_expect + pri_tcm_expect * pri_paid_metal_expect) / pri_paid_metal_expect
            by_only = ml_yr.index[neg_cash_flow][ml_yr.byproduct_id[neg_cash_flow] == 0]
            tcm_expect[by_only] = by_tcm_expect
            cash_flow_expect = by_cash_flow_expect + 1e-3 * pri_paid_metal_expect * pri_tcm_expect - \
                               ml_yr.primary_sustaining_capex_usdm[neg_cash_flow] - overhead - development_capex
            cash_flow_expect[by_only] = by_cash_flow_expect - overhead - development_capex

        return cash_flow_expect, by_cash_flow_expect, tcm_expect, by_tcm_expect, ml_yr

    def initialize_incentive_mines(self):
        h = self.hyperparam.copy()
        incentive_mines = self.mine_life_init.copy()
        i = self.i

        inc = miningModel(simulation_time=np.arange(self.i, self.i + self.hyperparam['incentive_roi_years']),
                          byproduct=self.byproduct)
        inc.hyperparam = self.hyperparam.copy()
        inc.simulation_time = np.arange(self.i, self.i + inc.hyperparam['incentive_roi_years'])
        inc.hyperparam['simulate_opening'] = False
        inc.hyperparam['reinitialize'] = False
        inc.hyperparam['opening_flag_for_cu0'] = True

        # price_df = self.ml.copy()['Commodity price (USD/t)'].unstack()
        price_expect = self.calculate_price_expect(self.primary_price_series, i)
        # price_expect = price_expect.fillna(price_expect.mean())
        inc.price_expect = price_expect

        # tcrc_df = self.ml.copy()['TCRC (USD/t)'].unstack()
        tcrc_expect = self.calculate_price_expect(self.primary_tcrc_series, i)
        # tcrc_expect = tcrc_expect.fillna(tcrc_expect.mean())

        inc.hyperparam['primary_commodity_price_option'] = 'constant'
        if self.byproduct:
            # pri_price_df = self.ml['Primary Commodity price (USD/t)'].unstack()
            pri_price_expect = self.calculate_price_expect(primary_price_series, i)
            incentive_mines['Primary Commodity price (USD/t)'] = pri_price_expect
            incentive_mines['Commodity price (USD/t)'] = price_expect
            inc.hyperparam['byproduct_commodity_price'] = np.nanmean(price_expect)
            inc.hyperparam['primary_commodity_price'] = np.nanmean(pri_price_expect)
            inc.pri_price_expect = pri_price_expect
            inc.hyperparam['byproduct_commodity_price_option'] = 'constant'
            # pri_tcrc_df = self.ml.copy()['Primary TCRC (USD/t)'].unstack()
            pri_tcrc_expect = self.calculate_price_expect(primary_tcrc_series, i)
            pri_tcrc_expect = pri_tcrc_expect.fillna(pri_tcrc_expect.mean())
        else:
            incentive_mines['Commodity price (USD/t)'] *= price_expect / incentive_mines[
                'Commodity price (USD/t)'].mean()
            inc.hyperparam['primary_commodity_price'] = float(price_expect)

        grade_decline = (self.cumulative_ore_treated[i] / self.cumulative_ore_treated[self.simulation_time[0]]) ** h[
            'initial_ore_grade_decline']
        cost_improve = (1 + h['incentive_mine_cost_change_per_year'] / 100) ** (i - self.simulation_time[0])
        incentive_mines['Ramp up flag'] = 1
        incentive_mines['Opening'] = i
        incentive_mines['Head grade (%)'] = incentive_mines['Initial head grade (%)'] * grade_decline
        incentive_mines.loc[incentive_mines['Head grade (%)'] > 80, 'Head grade (%)'] = 80
        incentive_mines['Cumulative ore treated (kt)'] = incentive_mines['Initial ore treated (kt)']
        incentive_mines['Total cash margin (USD/t)'] *= h['ramp_up_cu'] / incentive_mines[
            'Capacity utilization'] / cost_improve
        incentive_mines['Minesite cost (USD/t)'] *= h['ramp_up_cu'] / incentive_mines[
            'Capacity utilization'] * cost_improve
        incentive_mines['Capacity utilization'] = h['ramp_up_cu']
        if self.verbosity > 4:
            print(1951, 'incentive tcrc mean', incentive_mines['TCRC (USD/t)'].mean())
        if incentive_mines['TCRC (USD/t)'].mean() != 0:
            incentive_mines['TCRC (USD/t)'] *= tcrc_expect / incentive_mines['TCRC (USD/t)'].mean()
        #         incentive_mines['TCRC (USD/t)'] *= self.ml['TCRC (USD/t)'].unstack(0)[i].mean()/incentive_mines['TCRC (USD/t)'].mean()
        incentive_mines['Commodity price (USD/t)'] *= price_expect / incentive_mines[
            'Commodity price (USD/t)'].mean()
        #         incentive_mines['Commodity price (USD/t)'] *= self.ml['Commodity price (USD/t)'].unstack(0)[i].mean()/incentive_mines['Commodity price (USD/t)'].mean()
        if self.verbosity > 4:
            print(1958, 'incentive tcrc mean', incentive_mines['TCRC (USD/t)'].mean())
        if self.byproduct:
            incentive_mines['Primary Head grade (%)'] = incentive_mines[
                                                                   'Primary Initial head grade (%)'] * grade_decline
            incentive_mines.loc[incentive_mines['Primary Head grade (%)'] > 80, 'Primary Head grade (%)'] = 80
            incentive_mines['Primary Total cash margin (USD/t)'] *= h['ramp_up_cu'] / incentive_mines[
                'Capacity utilization'] / cost_improve
            incentive_mines['Byproduct Total cash margin (USD/t)'] *= h['ramp_up_cu'] / incentive_mines[
                'Capacity utilization'] / cost_improve
            incentive_mines['Primary Minesite cost (USD/t)'] *= h['ramp_up_cu'] / incentive_mines[
                'Capacity utilization'] * cost_improve
            incentive_mines['Primary Commodity price (USD/t)'] *= pri_price_expect.mean() / incentive_mines[
                'Primary Commodity price (USD/t)'].mean()
            if incentive_mines['Primary TCRC (USD/t)'].mean() != 0:
                incentive_mines['Primary TCRC (USD/t)'] *= pri_tcrc_expect.mean() / incentive_mines[
                    'Primary TCRC (USD/t)'].mean()

        inc.mine_life_init = incentive_mines.copy()
        inc.op_initialize_prices()
        inc.incentive_mines = incentive_mines.copy()
        self.incentive_mines = incentive_mines.copy()

        if i not in self.demand_series.index:
            initial_demand = inc.hyperparam['primary_production']
            change = inc.hyperparam['demand_series_pct_change'] / 100 + 1
            sim_time = inc.simulation_time
            if h['demand_series_method'] == 'yoy':
                self.demand_series = pd.Series([initial_demand * change ** (j - sim_time[0]) for j in sim_time],
                                               sim_time)
                self.demand_series.loc[i] = initial_demand * change ** (i - sim_time[0])
                if i - 1 not in self.demand_series.index:
                    self.demand_series.loc[i - 1] = initial_demand * change ** (i - 1 - sim_time[0])
            elif h['demand_series_method'] == 'target':
                self.demand_series = pd.Series(np.linspace(initial_demand, initial_demand * change, len(sim_time)),
                                               sim_time)
                self.demand_series.loc[i] = initial_demand - change * (i - sim_time[0])
                if i - 1 not in self.demand_series.index:
                    self.demand_series.loc[i - 1] = initial_demand - change * (i - 1 - sim_time[0])
        self.resources_contained_series.loc[i] = self.demand_series[i-1] * \
                                                 h['annual_reserves_ratio_with_initial_production_const']
        if i - 2 in self.demand_series.index and (
                i - 1 not in self.resources_contained_series.index or self.resources_contained_series.isna()[i - 1]):
            self.resources_contained_series.loc[i - 1] = self.demand_series[i - 2] * h[
                'annual_reserves_ratio_with_initial_production_const']
        if i > self.simulation_time[0] and h['reserves_ratio_price_lag'] > i - self.simulation_time[0] - 1:
            self.resources_contained_series.loc[i] = self.resources_contained_series[i - 1] * \
                                                     self.demand_series[i] / self.demand_series[i - 1] * \
                                                     (1 + h['annual_reserves_ratio_with_initial_production_slope'])
        elif i > self.simulation_time[0]:
            lag = min([h['reserves_ratio_price_lag'], i - self.simulation_time[0] - 1])
            lag = lag if lag > 0 else 0
            self.resources_contained_series.loc[i] = self.resources_contained_series[i - 1] * \
                                                     self.demand_series[i-1] / self.demand_series[i - 2] * \
                                                     (1 + h['annual_reserves_ratio_with_initial_production_slope']) * \
                                                     (self.primary_price_series[i - lag] / self.primary_price_series[
                                                         i - lag - 1]) ** h['primary_price_resources_contained_elas']
            if self.byproduct:
                self.resources_contained_series.loc[i] = self.resources_contained_series[i - 1] * \
                                                         self.demand_series[i] / self.demand_series[i - 1] * \
                                                         (1 + h[
                                                             'annual_reserves_ratio_with_initial_production_slope']) * \
                                                         (self.byproduct_price_series[i - lag] /
                                                          self.byproduct_price_series[i - lag - 1]) ** h[
                                                             'byproduct_price_resources_contained_elas']

        max_res = self.demand_series[self.simulation_time[0]] * h[
            'annual_reserves_ratio_with_initial_production_const'] * 100
        if self.verbosity>4:
            print(2673, max_res)
            print(2674, self.demand_series)
            print(2675, self.resources_contained_series)
        self.resources_contained_series.loc[self.resources_contained_series > max_res] = max_res
        self.reserves_ratio_with_demand_series = self.resources_contained_series / self.demand_series

        self.inc = inc

    def select_incentive_mines(self):
        h = self.hyperparam.copy()
        inc = self.inc
        incentive_mines = inc.incentive_mines.copy()
        i = self.i

        if h['incentive_use_resources_contained_series']:
            resources_contained = self.resources_contained_series[i]
            sxew_fraction = self.sxew_fraction_series[i]
            if self.sxew_fraction_series[self.simulation_time[0]] <= 0.1 and (
                    self.sxew_fraction_series > self.sxew_fraction_series[self.simulation_time[0]]).any():
                print(
                    'WARNING: small SX-EW fraction coupled with increasing SX-EW fraction may lead to drawing from a very small incentive pool. If SX-EW starts at zero and increases, it will produce an error.')
            try:
                n = int(resources_contained / incentive_mines['Production (kt)'].sum() * incentive_mines.shape[0] * 2)
            except Exception as e:
                print(2694,resources_contained)
                print(2695,incentive_mines['Production (kt)'])
                print(2696,incentive_mines['Production (kt)'].sum())
                print(2697,incentive_mines.shape[0])
                import pickle
                file = open('self.pkl', 'wb')
                pickle.dump(self, file)
                file.close()
                pd.Series(self.hyperparam).to_pickle('the worst.pkl')
                raise e
            n = 1 if n < 1 else n
            incentive_mines = incentive_mines.sample(n=n, replace=True, random_state=self.rs).reset_index(drop=True)
            sxew = incentive_mines.loc[incentive_mines['Payable percent (%)'] == 100]
            conc = incentive_mines.loc[incentive_mines['Payable percent (%)'] != 100]
            if sxew_fraction > 0 and sxew.shape[0] > 0:
                if sxew['Production (kt)'].iloc[0] < resources_contained * sxew_fraction:
                    sxew = sxew.loc[sxew['Production (kt)'].cumsum() < resources_contained * sxew_fraction]
                else:
                    sxew = pd.DataFrame(sxew.iloc[0]).T.rename({0: i})
            if sxew_fraction < 1 and conc.shape[0] > 0:
                if conc['Production (kt)'].iloc[0] < resources_contained * (1 - sxew_fraction):
                    conc = conc.loc[conc['Production (kt)'].cumsum() < resources_contained * (1 - sxew_fraction)]
                else:
                    conc = pd.DataFrame(conc.iloc[0]).T.rename({0: i})
            #             if incentive_mines['Production (kt)'].iloc[0]<resources_contained:
            #                 incentive_mines = incentive_mines.loc[incentive_mines['Production (kt)'].cumsum()<resources_contained]
            #             else:
            #                 incentive_mines = pd.DataFrame(incentive_mines.iloc[0]).T.rename({0:i})
            incentive_mines = pd.concat([conc, sxew]).sample(frac=1, random_state=self.rs, replace=False).reset_index(
                drop=True)
            self.subsample_series.loc[i] = incentive_mines.shape[0]
            self.initial_subsample_series.loc[i] = incentive_mines.shape[0]
        else:
            n = self.subsample_series.loc[i]
            incentive_mines = incentive_mines.sample(n=n, replace=True, random_state=self.rs).reset_index(drop=True)
        sxew = incentive_mines['Payable percent (%)']==100.
        self.perturb_cols = [i for i in incentive_mines.dtypes[incentive_mines.dtypes == float].index if
                             'price' not in i and i not in ['Opening', 'Closure', 'Known opening', 'Region',
                                                            'Mine type', 'Risk indicator']]
        perturbation_size = h['incentive_perturbation_percent'] / 100
        jitter_df = pd.DataFrame(np.reshape(stats.uniform.rvs(1 - perturbation_size, perturbation_size * 2,
                                                              size=incentive_mines[self.perturb_cols].size,
                                                              random_state=self.rs),
                                            (incentive_mines.shape[0], len(self.perturb_cols))), incentive_mines.index,
                                 self.perturb_cols)
        incentive_mines[self.perturb_cols] = incentive_mines[self.perturb_cols] * jitter_df
        incentive_mines.loc[sxew,'Payable percent (%)'] = 100.
        incentive_mines.loc[incentive_mines['Payable percent (%)']==100,'TCRC (USD/t)'] = 0.

        inc.mines = incentive_mines.copy()
        inc.generate_costs_from_regression('Development CAPEX ($M)')
        incentive_mines = inc.mines.copy()

        inc.mine_life_init = incentive_mines.copy()
        self.inc = inc

    def add_development_capex(self):
        h = self.hyperparam
        inc_mines = self.incentive_mines.copy()
        dcapex_method = h['incentive_dcapex_method']
        capacities = inc_mines.copy()['Capacity (kt)']
        if dcapex_method == 'fitch_lin':
            log = False
            capacities /= 1e3
        elif dcapex_method == 'fitch_log':
            log = True
            capacities /= 1e3
        elif dcapex_method == 'all_commodities':
            log = True
            capacities *= 1e3
        const = h['incentive_dcapex_' + dcapex_method + '_constant']
        slope = h['incentive_dcapex_' + dcapex_method + '_slope']

        if log:
            dcapex = np.exp(const + slope * np.log(capacities))
        else:
            dcapex = const + slope * capacities
        inc_mines['Development CAPEX ($M)'] = dcapex / h['ramp_up_years']
        inc_mines.loc[inc_mines['Payable percent (%)'] > 100] = 99.9
        self.incentive_mines = inc_mines.copy()

    def op_sim_mine_opening(self):
        h = self.inc.hyperparam
        if h['incentive_opening_method'] in ['xinkai_thesis', 'unconstrained']:
            self.incentive_open_xinkai_thesis()

        elif h['incentive_opening_method'] == 'karan_generalization':
            self.incentive_open_karan_generalization()

        mines_to_open = self.inc.mines_to_open.copy()
        if mines_to_open.shape[0] > 0:
            mines_to_open['Ramp up flag'] = 2
            mines_to_open['Initial price (USD/t)'] = self.primary_price_series[self.i]
            mines_to_open['Commodity price (USD/t)'] = self.primary_price_series[self.i]
            mines_to_open = OneMine(name=self.i, df=mines_to_open)

        #         mines_to_open = mines_to_open.rename(
        #                     dict(zip(mines_to_open.index.get_level_values(1),
        #                              np.arange(self.ml.index.get_level_values(1).max()+self.i*1e4,
        #                                        self.ml.index.get_level_values(1).max()+self.i*1e4+mines_to_open.shape[0]))),level=1)

        self.mines_to_open = mines_to_open

    def incentive_open_xinkai_thesis(self):
        nowish = datetime.now()
        inc = self.inc
        h = inc.hyperparam.copy()
        simulation_time = inc.simulation_time
        i = self.i

        condition_for_ml = i > self.simulation_time[0] + h['ml_accelerate_initialize_years'] and i % h[
            'ml_accelerate_every_n_years'] != 0


        inc.mine_life_init['Initial price (USD/t)'] = self.primary_price_series[self.i]
        inc.mine_life_init['Commodity price (USD/t)'] = self.primary_price_series[self.i]
        inc.mines = OneMine(name=i, df=inc.mine_life_init)
        if h['use_ml_to_accelerate']:
            inc2 = deepcopy([inc])[0]
            if condition_for_ml:
                inc.i = i
                inc.simulate_mine_life_one_year()
        else:
            inc.simulate_mine_life_all_years()
        opening_sim = inc.ml.copy()
        outside_ml = self.ml.loc[i]
        current_prod = np.nansum(outside_ml.production_kt)
        inc_mines = opening_sim.loc[i].generate_df()

        incentive_end = inc.simulation_time[-1]
        incentive_start = inc.simulation_time[0]
        max_len = len(opening_sim.loc[incentive_end].cash_flow_usdm)
        discount_rate_ph = inc.hyperparam['incentive_discount_rate']
        def make_big_array(j):
            ph = opening_sim.loc[j].cash_flow_usdm / (1 + discount_rate_ph) ** (j - incentive_start)
            return np.append(ph, np.repeat(np.nan,max_len-len(ph)))
        result = map(make_big_array, np.arange(incentive_start,incentive_end+1))
        npv = np.nansum(np.array(list(result)),axis=0)
        # opsim['Discount rate (divisor)'] = [
        #     (1 + h['incentive_discount_rate']) ** (j[0] - simulation_time[0]) for j in opsim.index]
        inc_mines['NPV ($M)'] = npv

        self.inc_mines = inc_mines
        self.opening_sim = opening_sim

        run_ml_selected_mines = False  # this variable lets us try to use the ml model to decrease the size of inc_mines and therefore the time required to simulate. Didn't seem super worthwhile.
        if h['use_ml_to_accelerate']:
            if not hasattr(self, 'all_inc_mines'):
                self.all_inc_mines = pd.DataFrame()
                self.accuracy_knc = pd.Series(dtype=float)
                self.accuracy_lda = pd.Series(dtype=float)
                self.accuracy_gnb = pd.Series(dtype=float)
                self.accuracy_dtc = pd.Series(dtype=float)
            else:
                #                 opener_predict_knc, self.accuracy_knc.loc[i] = run_mine_ml_model(KNeighborsClassifier, self.all_inc_mines, inc_mines, verbosity=0)
                self.inc_mines_ph = inc_mines.copy()
                opener_predict_lda, self.accuracy_lda.loc[i] = run_mine_ml_model(LinearDiscriminantAnalysis,
                                                                                 self.all_inc_mines, inc_mines,
                                                                                 verbosity=0)
            #                 opener_predict_gnb, self.accuracy_gnb.loc[i] = run_mine_ml_model(GaussianNB, self.all_inc_mines, inc_mines, verbosity=0)
            #                 opener_predict_dtc, self.accuracy_dtc.loc[i] = run_mine_ml_model(DecisionTreeClassifier, self.all_inc_mines, inc_mines, verbosity=0)
            if run_ml_selected_mines and condition_for_ml:
                inc2.mine_life_init = inc2.mine_life_init.loc[opener_predict_lda]
                inc2.mines = inc2.mine_life_init.copy()
                inc2.simulate_mine_life_all_years()
                opening_sim2 = inc2.ml.copy()
                inc2_mines = opening_sim2.loc[i].copy()
                opening_sim2['Discount rate (divisor)'] = [
                    (1 + h['incentive_discount_rate']) ** (j[0] - simulation_time[0]) for j in opening_sim2.index]
                inc2_mines['NPV ($M)'] = (
                            opening_sim2['cash_flow_usdm'] / opening_sim2['Discount rate (divisor)']).groupby(
                    level=1).sum()
                self.all_inc_mines = pd.concat([self.all_inc_mines, pd.concat([inc2_mines], keys=[i])])
            if not condition_for_ml:
                self.all_inc_mines = pd.concat([self.all_inc_mines, pd.concat([inc_mines], keys=[i])])

        frac = 1
        self.end_calibrate = self.simulation_time[0] + h['end_calibrate_years']
        self.start_frac = self.simulation_time[0] + h['start_calibrate_years']
        self.frac_series = self.subsample_series / self.initial_subsample_series

        if h['calibrate_incentive_opening_method'] or (h['incentive_opening_method'] == 'unconstrained' and i <= self.end_calibrate and h['incentive_opening_probability'] == 0) or self.i <= h['incentive_require_tune_years'] + self.simulation_time[0]:
            target = self.demand_series.loc[i]
            subset = inc_mines.sample(frac=frac, random_state=self.rs).reset_index(drop=True)
            subset.loc[subset['NPV ($M)'] < 0,'production_kt'] = 0
            current_sum = subset['production_kt'].sum()
            if current_sum < (target - current_prod) and current_sum != 0:
                frac = (target - current_prod) / current_sum + 1
                subset = subset.sample(frac=frac, random_state=self.rs, replace=True).reset_index(drop=True)
            error = abs(subset['production_kt'].cumsum() + current_prod - target).astype(float)
            if h['ramp_up_exponent'] != 0:
                error /= (h['ramp_up_years'] + 1) ** (h['ramp_up_exponent'])
            self.error = error
            self.subset = subset
            if current_sum == 0:
                if self.verbosity > -1:
                    print(
                        'perhaps runaway demand or mines are incapable of opening (low ore grade likely culprit)\n\tSet verbosity=(-1) to remove this message')
                error_idx = 1000
            else:
                error_idx = error.idxmin()
            error_series = error.copy()

            if self.verbosity > 2:
                print('Tuning to demand: {:.3f}'.format(target))

            self.error_series = error_series
            self.subsample_series.loc[i] = error_idx
            n = int(self.subsample_series.loc[i])
            inc_mines = inc_mines.sample(n=n,
                                         random_state=self.rs, replace=n >= inc_mines.shape[0])
        else:
            if hasattr(self, 'hist_mod'):
                frac = self.hist_mod.frac
                self.frac = frac
            else:
                if h['incentive_opening_probability'] == 0:
                    frac = (self.subsample_series / self.initial_subsample_series).loc[
                           self.start_frac:self.end_calibrate].mean()
                    while frac == 0:
                        self.start_frac -= 1
                        frac = (self.subsample_series / self.initial_subsample_series).loc[
                               self.start_frac:self.end_calibrate].mean()
                        if frac == 0 and self.start_frac == simulation_time[0]:
                            raise ValueError(
                                'frac==0, mines are not sufficiently viable to run this scenario. Try changing ore grade distribution?')
                        elif self.start_frac < 2000:
                            raise ValueError('fraction is not working correctly')
                else:
                    frac = h['incentive_opening_probability']
                self.frac = frac
                inc_mines = inc_mines.sample(frac=frac, random_state=self.rs, replace=frac >= 1)
                self.subsample_series.loc[i] = inc_mines.shape[0]
            if self.verbosity > 2:
                print('fraction:', 2120, frac)

        inc.opening_sim = opening_sim.copy()
        inc.inc_mines = inc_mines.copy()
        if h['use_ml_to_accelerate'] and condition_for_ml and not run_ml_selected_mines:
            opening_ind = np.intersect1d(opener_predict_lda, inc_mines.index)
            if self.verbosity > 3:
                print('USING ML')
        elif run_ml_selected_mines and condition_for_ml:
            profitable_ml_sim = inc2_mines.loc[inc2_mines['NPV ($M)'] > 0].index
            opening_ind = np.intersect1d(profitable_ml_sim, inc_mines.index)
            if self.verbosity > 3:
                print('USING ML, SIM SELECTED')
        else:
            opening_ind = inc_mines['NPV ($M)'] > 0
        frac = 1 if h['incentive_opening_method'] == 'unconstrained' else 1
        inc.mines_to_open = inc_mines.loc[opening_ind].sample(frac=frac, random_state=self.rs, replace=frac > 1)

        if self.verbosity > 1:
            print(f'Number of mines opening {inc.mines_to_open.shape[0]}')
            print('Number of mines closing:', np.nansum(self.ml_yr.simulated_closure==i))
        self.inc = inc
        if not hasattr(self, 'inc_list'):
            self.inc_list = []
        self.inc_list += [inc]
        if self.verbosity > 2:
            print('incentive_open_xinkai_thesis takes:', str(datetime.now() - nowish))

    def incentive_open_karan_generalization(self):
        inc = self.inc
        h = inc.hyperparam
        simulation_time = inc.simulation_time
        i = self.i

        inc.mines = inc.mine_life_init.copy()
        s_ph = deepcopy([inc])[0]
        self.s_ph = s_ph
        inc.simulate_mine_life_all_years()
        opening_sim = inc.ml.copy()
        inc_mines = opening_sim.loc[i].copy()
        inc_mines['Ramp up flag'] = 1
        opening_sim['Discount rate (divisor)'] = [
            (1 + h['incentive_discount_rate']) ** (j[0] - simulation_time[0]) for j in opening_sim.index]
        inc_mines['NPV ($M)'] = (opening_sim['Cash flow ($M)'] / opening_sim['Discount rate (divisor)']).groupby(
            level=1).sum()
        error_series = pd.Series(dtype=float)
        n_series = pd.Series(dtype=float)

        if h['calibrate_incentive_opening_method']:
            print('Last time I touched this, was trying to use ML models to predict the price that would give' +
                  ' the desired amount of opening. Seemed like it was promising, just trying a different approach.')
            target = self.demand_series[i]
            operating_production = self.ml.loc[i, 'Production (kt)'].sum()
            target_new = target - operating_production
            target_new = 0 if inc_mines['Production (kt)'].min() / 2 > target_new else target_new
            # the production_min/2>target_new clause is to handle the case where operating_production is smaller than
            # target, but by such a small amount as to still justify no mines opening. With a nonzero target_new
            # in that case, the program will try to find the largest TCRC or smallest price that can be charged
            # to produce the result, but when no mines are opening we need to find the smallest TCRC and largest price
            # that accomplishes that goal. Otherwise, we get runaway TCRC increases and price decreases.

            incentive_tune_tcrc = h['incentive_tune_tcrc']
            if incentive_tune_tcrc:
                price_select = 'TCRC (USD/t)'
            else:
                price_select = 'Commodity price (USD/t)'

            n = int(self.subsample_series[i] * h['incentive_opening_probability'])
            subset = inc_mines.sample(n=n, random_state=self.rs, replace=n > inc_mines.shape[0])
            error = subset.loc[subset['NPV ($M)'] > 0, 'Production (kt)'].sum() - target_new
            error_series.loc[inc_mines[price_select].mean()] = error
            n_series.loc[inc_mines[price_select].mean()] = n

            sign = error > 0
            not_sign = not sign
            price_change = 1000
            mpc_multiplier = 0.01  # min price change multiplier
            while price_change > subset[price_select].mean():
                price_change /= 3
            min_error = np.inf
            min_price_change = inc_mines[price_select].mean() * mpc_multiplier
            counter = 0
            price = subset[price_select].mean()
            multiplier = -1 if sign or error == min_error else 1

            if self.i == simulation_time[0]:
                self.error_df = pd.DataFrame(dtype=float)
            if self.verbosity > 3:
                print(price,
                      '\n\tError:', error, '\n\tMultiplier:', multiplier, '\n\tSign, not sign:', sign, not_sign,
                      '\n\tPrice change:', price_change, '\n\tTarget:', target, '\n\tProduction:', error + target,
                      '\n\tNumber of mines:', n, '\n\tNumber opening:', (subset['NPV ($M)'] > 0).sum())

            if not hasattr(self, 'all_yrs_inc_mines'):
                self.all_yrs_inc_mines = pd.DataFrame()
            self.all_inc_mines = pd.DataFrame()
            count = 0
            while (abs(error) > target * mpc_multiplier or target_new == 0) and (
                    price_change > min_price_change or error == min_error):
                if target_new < subset['Production (kt)'].sum():
                    inc_mines[price_select] *= (price + multiplier * price_change) / inc_mines[
                        price_select].mean()
                elif incentive_tune_tcrc:
                    inc_mines[price_select] = 10
                    print('incentive_tune_tcrc')
                else:
                    inc_mines[price_select] *= 10
                    print('not incentive_tune_tcrc')

                s = deepcopy([s_ph])[0]
                s.mine_life_init = inc_mines.copy()
                s.mines = inc_mines.copy()
                s.simulate_mine_life_all_years()
                opening_sim = s.ml.copy()

                opening_sim['Discount rate (divisor)'] = [
                    (1 + h['incentive_discount_rate']) ** (j[0] - simulation_time[0]) for j in opening_sim.index]
                inc_mines['NPV ($M)'] = (
                            opening_sim['Cash flow ($M)'] / opening_sim['Discount rate (divisor)']).groupby(
                    level=1).sum()

                self.opening_sim = opening_sim.copy()
                self.inc_mines1 = inc_mines.copy()

                subset = inc_mines.sample(n=n, random_state=self.rs, replace=n > inc_mines.shape[0])
                self.subset1 = subset.copy()

                if count >= 2:
                    opener_predict_knc, accuracy = run_mine_ml_model(KNeighborsClassifier, self.all_inc_mines, subset,
                                                                     verbosity=3)
                    opener_predict_lda, accuracy = run_mine_ml_model(LinearDiscriminantAnalysis, self.all_inc_mines,
                                                                     subset, verbosity=3)
                    opener_predict_gnb, accuracy = run_mine_ml_model(GaussianNB, self.all_inc_mines, subset,
                                                                     verbosity=3)
                    opener_predict_dtc, accuracy = run_mine_ml_model(DecisionTreeClassifier, self.all_inc_mines, subset,
                                                                     verbosity=3)

                error = subset.loc[subset['NPV ($M)'] > 0, 'Production (kt)'].sum() - target_new
                price = inc_mines[price_select].mean()
                error_series.loc[price] = error
                n_series.loc[price] = (subset['NPV ($M)'] > 0).sum()

                self.all_inc_mines = pd.concat([self.all_inc_mines,
                                                pd.concat([subset], keys=[count])
                                                ])
                count += 1

                sign = error > 0
                multiplier = -1 if sign or error == min_error else 1
                if incentive_tune_tcrc: multiplier *= -1

                if self.verbosity > 3:
                    print(price,
                          '\n\tError:', error, '\n\tMultiplier:', multiplier, '\n\tSign, not sign:', sign, not_sign,
                          '\n\tPrice change:', price_change, '\n\tTarget:', target, '\n\tProduction:', error + target,
                          '\n\tNumber of mines:', n, '\n\tNumber opening:', (subset['NPV ($M)'] > 0).sum())

                no_mines = (subset['NPV ($M)'] > 0).sum() == 0
                if sign == not_sign and price_change != min_price_change:
                    sign = error > 0
                    not_sign = not sign
                    price_change /= 3
                    if price_change < min_price_change:
                        price_change = min_price_change
                        multiplier = 1 if incentive_tune_tcrc else -1
                        min_error_series = error_series.loc[abs(error_series) == abs(error_series).min()]
                        if (error == min_error_series.iloc[0] or target - operating_production == min_error_series.iloc[
                            0]) and no_mines:
                            error_series.loc[min_error_series.index] = 0
                            print('line 2098 executed')
                            min_error_series.loc[:] = 0
                        if (incentive_tune_tcrc and target_new != 0 and no_mines == False) or (
                                target_new == 0 and no_mines and incentive_tune_tcrc == False):
                            price = max(min_error_series.index)
                        else:
                            price = min(min_error_series.index)
                        err_ser = error_series.loc[~error_series.index.duplicated()]
                        min_error = err_ser.loc[price]
                        error = min_error if target_new > 0 else np.inf
                elif inc_mines['Commodity price (USD/t)'].mean() > 1e6:
                    print('perhaps runaway demand or mines are incapable of opening')
                    break
                elif price_change == min_price_change:
                    counter += 1
                    if counter > 3:
                        if no_mines:
                            min_error_series = error_series.loc[abs(error_series) == abs(error_series).min()]
                            if error == min_error_series.iloc[0]:
                                error_series.loc[min_error_series.index] = 0
                            target_new = 0
                            price_change *= 3
                        else:
                            price_change *= 3
                            min_price_change = price_change
                        counter = 0

                if target_new > subset['Production (kt)'].sum():
                    break

            self.error_series = error_series
            error_ph = pd.DataFrame(error_series)
            error_ph = error_ph.rename(columns={error_ph.columns[0]: i})
            self.error_df = pd.concat([self.error_df, error_ph], axis=1)
            ph = error_series.loc[(n_series == 0)]
            if (ph == 0).any():
                error_series.loc[n_series == 0] = 0
                no_mines = True
            else:
                no_mines = False
            error_series = error_series.loc[abs(error_series) == abs(error_series).min()]
            if target_new > subset['Production (kt)'].sum():
                multiplier_price = inc_mines[price_select].mean()
            elif (incentive_tune_tcrc and target_new != 0 and no_mines == False) or (
                    no_mines and incentive_tune_tcrc == False):
                multiplier_price = max(error_series.index)
            else:
                multiplier_price = min(error_series.index)
            self.new_price_select = multiplier_price

            inc_mines[price_select] *= multiplier_price / inc_mines[price_select].mean()
            inc_mines = inc_mines.sample(n=self.subsample_series.loc[i],
                                         random_state=self.rs, replace=n > inc_mines.shape[0])
            s = deepcopy([s_ph])[0]
            s.mine_life_init = inc_mines.copy()
            s.mines = inc_mines.copy()
            s.simulate_mine_life_all_years()
            opening_sim = s.ml.copy()

            opening_sim['Discount rate (divisor)'] = [
                (1 + h['incentive_discount_rate']) ** (j[0] - simulation_time[0]) for j in opening_sim.index]
            inc_mines['NPV ($M)'] = (
                        opening_sim['Cash flow ($M)'] / opening_sim['Discount rate (divisor)']).groupby(level=1).sum()

            inc.opening_sim = opening_sim.copy()
            inc.inc_mines = inc_mines.copy()
            inc.mines_to_open = inc_mines.loc[inc_mines['NPV ($M)'] > 0]
            self.inc = inc

        else:
            self.bayesian_tune()

    def run_karan_generalization(self, price):
        inc = self.inc
        h = inc.hyperparam
        simulation_time = inc.simulation_time
        i = self.i

        if not hasattr(self, 'error_series'):
            self.error_series = pd.Series(dtype=float)
        if not hasattr(self, 'n_series'):
            self.n_series = pd.Series(dtype=float)

        s = deepcopy([self.s_ph])[0]

        incentive_tune_tcrc = h['incentive_tune_tcrc']
        if incentive_tune_tcrc:
            price_select = 'TCRC (USD/t)'
        else:
            price_select = 'Commodity price (USD/t)'

        inc_mines = s.mine_life_init.copy()
        if inc_mines[price_select].mean()!=0:
            inc_mines[price_select] *= price / inc_mines[price_select].mean()

        target = self.demand_series[i]
        operating_production = self.ml.copy().loc[i, 'Production (kt)'].sum()
        target_new = target - operating_production
        target_new = 0 if inc_mines['Production (kt)'].min() / 2 > target_new else target_new

        s.mine_life_init = inc_mines.copy()
        s.mines = inc_mines.copy()
        s.simulate_mine_life_all_years()
        opening_sim = s.ml.copy()
        opening_sim['Discount rate (divisor)'] = [(1 + h['discount_rate']) ** (j[0] - simulation_time[0]) for j
                                                         in opening_sim.index]
        inc_mines['NPV ($M)'] = (opening_sim['Cash flow ($M)'] / opening_sim['Discount rate (divisor)']).groupby(
            level=1).sum()

        n = self.subsample_series[i]
        subset = inc_mines.sample(n=n, random_state=self.rs, replace=n > inc_mines.shape[0])
        error = subset.loc[subset['NPV ($M)'] > 0, 'Production (kt)'].sum() - target_new
        price = inc_mines[price_select].mean()
        self.error_series.loc[price] = error
        n_mines = (subset['NPV ($M)'] > 0).sum()
        self.n_series.loc[price] = n_mines

        print(price,
              '\n\tError:', error, '\n\tTarget:', target, '\n\tTotal Production:', error + target_new,
              '\n\tTarget new:', target_new, '\n\tOperating production:', operating_production,
              '\n\tNumber of mines:', n, '\n\tNumber opening:', (subset['NPV ($M)'] > 0).sum())

        return error, n_mines, opening_sim

    def bayesian_tune(self):
        init_samples = 4
        scale = 2500
        slope = 0.00001
        prices = np.linspace(0, scale, init_samples)
        epo = [self.run_karan_generalization(p) for p in prices]
        errors, n_opening, opening_sim_list = [o[0] for o in epo], [o[1] for o in epo], [o[2] for o in epo]
        errors = np.array([e + p * slope for e, p in zip(errors, prices)])

        prices_scaled = prices / scale

        X, y = [x.reshape(len(x), 1) for x in [prices_scaled, errors]]

        model = GaussianProcessRegressor()
        model.fit(X, y)

        if self.verbosity > 3:
            plot(X, y, model)

        for i in np.arange(0, 20):
            if min([i - p * slope for i, p in zip(y, X * scale)]) > 0:
                x = max(X) * 2
                print('49')
            else:
                x = optimize_acquisition(X, y, model)
            while x in X and i < 15:
                x = x - 0.05
            if x in X:
                break
            x_rescaled = x * scale
            actual, n_open, os_ph = self.run_karan_generalization(x_rescaled)
            self.opening_sim_list = opening_sim_list
            opening_sim_list = opening_sim_list + [os_ph]
            actual += slope * x_rescaled
            est, _ = surrogate(model, [[x]])

            # add the data to the dataset
            X = np.vstack((X, [[x]]))
            y = np.vstack((y, [[actual]]))

            if self.verbosity > 3:
                print(
                    '  x = {:.3f}, predicted = {:.3f}, actual = {:.3f}, rescaled price: {:.1f}\nCurrent best: {:.1f}:'.format(
                        x, est[0][0], actual, x_rescaled, X[:, 0][np.argmin(abs(y))] * scale))

            if n_open == 0:
                argminy = np.argmin(y)
                y = y[X <= X[argminy]]
                X = X[X <= X[argminy]]
                X, y = [x.reshape(len(x), 1) for x in [X, y]]

            # update the model
            model.fit(X, y)

            if self.verbosity > 3:
                plt.figure()
                plt.title(i)
                plot(X, y, model)

        self.new_price_select = X[:, 0][np.argmin(abs(y))] * scale
        opening_sim = opening_sim_list[np.argmin(abs(y))]

        opening_sim['Discount rate (divisor)'] = [
            (1 + self.inc.hyperparam['discount_rate']) ** (j[0] - self.simulation_time[0]) for j in
            opening_sim.index]
        inc_mines = opening_sim.copy().loc[self.i]
        inc_mines['NPV ($M)'] = (opening_sim['Cash flow ($M)'] / opening_sim['Discount rate (divisor)']).groupby(
            level=1).sum()

        self.inc.opening_sim = opening_sim.copy()
        self.inc.inc_mines = inc_mines.copy()
        self.inc.mines_to_open = inc_mines.loc[inc_mines['NPV ($M)'] > 0]

        if not hasattr(self, 'error_df'):
            self.error_df = pd.DataFrame(dtype=float)
        self.error_df = pd.concat([self.error_df, pd.concat([self.error_series], keys=[self.i])])
        delattr(self, 'error_series')

    def update_price_tcrc(self, byproduct=False):
        h = self.hyperparam
        opening = self.mines_to_open.copy()
        i = self.i

        if h['internal_price_formation']:
            if h['incentive_tune_tcrc']:
                price_select = 'tcrc_usdpt'
                elas = h['tcrc_sd_elas']
            else:
                price_select = 'commodity_price_usdpt'
                elas = h['price_sd_elas']
            if byproduct:
                price_select = 'primary_' + price_select

            if h['incentive_opening_method'] == 'karan_generalization':
                ml_ot = self.ml.loc[i].ore_treated_kt
                ml_price = getattr(self.ml.loc[i],price_select)
                open_ot = opening.ore_treated_kt
                open_price = getattr(opening,price_select) * self.new_price_select / np.nanmean(getattr(opening,price_select))
                new_price_select = (np.nansum(ml_ot * ml_price) + np.nansum(open_ot * open_price)) / (
                            np.nansum(ml_ot) + np.nansum(open_ot))
                #             new_price_select = open_price.mean()
                setattr(self.ml.loc[i], price_select, getattr(ml.loc[i],price_select) * new_price_select / np.nanmean(getattr(ml.loc[i],price_select)))
                setattr(opening, price_select,
                    getattr(opening, price_select) * new_price_select / np.nanmean(getattr(opening, price_select).astype(float)))

            elif h['incentive_opening_method'] == 'unconstrained':
                self.supply_series.loc[i] = np.nansum(self.ml.loc[i].production_kt) + np.nansum(opening.production_kt)
                incentive_tuning_option = h['incentive_tuning_option']
                if 'pid-' in incentive_tuning_option:
                    if i < int(incentive_tuning_option.split('-')[1]):
                        incentive_tuning_option = 'pid'
                    else:
                        incentive_tuning_option = 'elas'
                if (i < self.end_calibrate and h[
                    'incentive_opening_probability'] == 0) or incentive_tuning_option == 'elas':
                    sd_ratio = self.supply_series.loc[i] / self.demand_series.loc[i]
                    price_change_ratio = sd_ratio ** elas
                    setattr(self.ml.loc[i], price_select, getattr(ml.loc[i],price_select) * price_change_ratio)

                elif i > self.simulation_time[0]:
                    # Trying PID controller style for initial TCRC tune
                    e_t = (self.supply_series / self.demand_series - 1)[i]
                    e_t_l1 = (self.supply_series / self.demand_series - 1)[i - 1]
                    integral = (self.supply_series / self.demand_series - 1).loc[:i].sum()
                    derivative = (e_t - e_t_l1) / 1

                    k_u = 1
                    t_u = 20
                    k_p = k_u * 0.6  # = 0.6 for k_u=1
                    k_i = 1.2 * k_u / t_u  # 0.06
                    k_d = 3 * k_u * t_u / 40  # 1.5

                    u_t = k_p * (e_t + k_i * integral + k_d * derivative)
                    price_change_ratio = 1 + u_t if u_t > -1 else 0.1
                    if not hasattr(self, 'pid_params'):
                        self.pid_params = pd.DataFrame(np.nan, self.simulation_time,
                                                       ['error', 'integral', 'derivative'])
                    self.pid_params.loc[i, 'error'] = e_t
                    self.pid_params.loc[i, 'integral'] = integral
                    self.pid_params.loc[i, 'derivative'] = derivative
                    if self.verbosity > 3:
                        print(3379,
                              'price change ratio: {:.3f}\n\tu_t: {:.3f}\n\te_t: {:.3f}\n\tintegral: {:.3f}\n\tderviative: {:.3f}'.format(
                                  price_change_ratio, u_t, e_t, integral, derivative))
                setattr(self.ml.loc[i], price_select, getattr(ml.loc[i],price_select) * price_change_ratio)
                setattr(opening, price_select,
                    getattr(opening, price_select) * price_change_ratio)

                if self.verbosity > 3:
                    fig, ax = easy_subplots(3, 3, dpi=40)
                    # plot SD evolution
                    self.supply_series.plot(ax=ax[0], marker='o', label='Supply')
                    self.demand_series.plot(ax=ax[0], marker='v', label='Demand')
                    ax[0].legend()
                    ax[0].set(title='SD evolution')

                    # plot SD ratio
                    (self.supply_series / self.demand_series).plot(ax=ax[1], marker='o')
                    ax[1].set(title='SD ratio evolution')

                    # plot price/tcrc evolution
                    pd.concat([self.ml[price_select], opening[price_select]]).unstack(0).mean().plot(ax=ax[2],
                                                                                                            marker='o')
                    ax[2].set(title=price_select)

                    plt.show()
                    plt.close()

        else:
            for price_select, alt_select in zip(['TCRC (USD/t)', 'Commodity price (USD/t)'],
                                                ['tcrc_usdpt', 'commodity_price_usdpt']):
                if byproduct:
                    new_price = self.byproduct_price_series[i] if price_select == 'Commodity price (USD/t)' else \
                    self.byproduct_tcrc_series[i]
                    self.ml.loc[idx[self.i, :], price_select] *= new_price / self.ml.loc[
                        idx[self.i, :], price_select].mean()
                    opening[price_select] *= new_price / opening[price_select].mean()

                    price_select = 'Primary ' + price_select
                    new_price = self.primary_price_series[i] if price_select == 'Primary Commodity price (USD/t)' else \
                    self.primary_tcrc_series[i]
                    self.ml.loc[idx[self.i, :], price_select] *= new_price / self.ml.loc[
                        idx[self.i, :], price_select].mean()
                    opening[price_select] *= new_price / opening[price_select].mean()
                else:
                    new_price = self.primary_price_series[i] if price_select == 'Commodity price (USD/t)' else \
                    self.primary_tcrc_series[i]
                    if True:#'TCRC' in price_select:
                        year_i_mines = self.ml.loc[i]
                        conc_mines = (year_i_mines.payable_percent_pct!=100)&(~np.isnan(year_i_mines.payable_percent_pct.astype(float)))
                        if np.sum(conc_mines)>0:
                            setattr(self.ml.loc[i], alt_select, getattr(year_i_mines, alt_select) * new_price / np.nanmean(
                                getattr(year_i_mines, alt_select)[conc_mines]))
                        if len(opening.index)>0:
                            conc_mines = (opening.payable_percent_pct!=100)&(~np.isnan(opening.payable_percent_pct.astype(float)))
                            if np.sum(conc_mines)>0:
                                setattr(opening, alt_select,
                                    getattr(opening, alt_select) * new_price / np.nanmean(getattr(opening, alt_select).astype(float)[conc_mines]))
                    else:
                        setattr(self.ml.loc[i], alt_select, new_price)
                        setattr(opening, alt_select, new_price)
                        # print(3437,opening.commodity_price_usdpt)

        self.opening = opening.copy()
        if self.byproduct and byproduct == False:
            self.update_price_tcrc(byproduct=True)

    def simulate_history(self):
        self.simulate_history_bool = False
        sh = deepcopy([self])[0]
        h = sh.hyperparam
        h.loc['simulate_history_bool'] = False
        demand_growth = h['demand_series_pct_change'] / 100 + 1
        final_demand = h['byproduct_production'] if self.byproduct else h['primary_production']
        years_back = 20 if demand_growth < 0.1 else 50

        sh.simulation_time = np.arange(self.simulation_time[0] - years_back, self.simulation_time[0] + 1)
        h.loc['simulation_time'] = sh.simulation_time

        h['demand_series_method'] = 'yoy'
        initial_demand = final_demand * demand_growth ** (-1 * years_back)
        print(2536, initial_demand, final_demand)
        if self.byproduct:
            h.loc['byproduct_production'] = initial_demand
        else:
            h.loc['primary_production'] = initial_demand;

        h.loc['initial_ore_grade_decline'] = 0
        h.loc['mine_cost_change_per_year'] = 0
        h.loc['incentive_tuning_option'] = self.incentive_tuning_option
        sh.incentive_tuning_option = self.incentive_tuning_option

        sh.hyperparam = h
        sh.recalculate_hyperparams()
        sh.primary_price_series = pd.Series(float(h['primary_commodity_price']), sh.simulation_time)
        if self.byproduct: sh.byproduct_price_series = pd.Series(float(h['byproduct_commodity_price']),
                                                                 sh.simulation_time)

        sh.simulate_mine_life_all_years()
        self.mine_history = sh.ml.copy()
        self.ml = sh.ml.copy().loc[idx[self.simulation_time[0], :], :]
        self.ml_yr = sh.ml.copy().loc[self.simulation_time[0]]
        self.mine_life_init = sh.mine_life_init.copy()
        self.mines = sh.mine_life_init.copy()
        self.hist_mod = sh
        self.update_operation_hyperparams(innie=self.ml_yr)

        #         self.incentive_tuning_option = 'elas'
        self.cumulative_ore_treated = pd.Series(np.nan, self.simulation_time)
        self.cumulative_ore_treated.loc[self.simulation_time[0]] = self.ml['Ore treated (kt)'].sum()
        self.supply_series = sh.supply_series.copy()

    def simulate_mine_life_all_years(self):
        for i in self.simulation_time:
            self.i = i
            self.simulate_mine_life_one_year()

    def simulate_mine_opening(self):
        self.initialize_incentive_mines()
        self.select_incentive_mines()
        self.op_sim_mine_opening()

    def price_vs_subsample_size(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            m = miningModel(byproduct=False, verbosity=0)
            m.hyperparam['minesite_cost_response_to_grade_price'] = True
            m.hyperparam['primary_commodity_price'] = 7000
            m.hyperparam['primary_ore_grade_mean'] = 0.1
            m.hyperparam['calibrate_incentive_opening_method'] = True
            m.hyperparam['incentive_opening_method'] = 'karan_generalization'
            m.hyperparam['demand_series_pct_change', :] = 10
            m.hyperparam['incentive_use_resources_contained_series'] = False
            m.hyperparam['incentive_tune_tcrc'] = True

            m.simulation_time = np.arange(2019, 2041)
            m.op_initialize_prices()
            for i in m.simulation_time[:2]:
                m.i = i
                m.simulate_mine_life_one_year()

            incentive_tune_tcrc = m.hyperparam['incentive_tune_tcrc']
            if incentive_tune_tcrc:
                price_select = 'TCRC (USD/t)'
            else:
                price_select = 'Commodity price (USD/t)'

            priceVsample = pd.Series(np.nan, [int(p) for p in np.linspace(100, 10000, 50)])
            for s in priceVsample.index:
                m.subsample_series.loc[i] = s
                m.incentive_open_karan_generalization()
                price = m.inc.inc_mines[price_select].mean()
                priceVsample.loc[s] = price
                print(s, price)
            self.priceVsample = priceVsample.copy()

    def run(self):
        self.simulate_mine_life_one_year()


# Attempts to use Bayesian regression, these fit in with the bayesian_tune() and run_karan_generalization() functions
def surrogate(model, X):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return model.predict(X, return_std=True)


def acquisition(X, Xsamples, model):
    yhat, _ = surrogate(model, X)
    best = min(yhat)
    mu, std = surrogate(model, Xsamples)
    mu = mu[:, 0]
    # calculate probability of improvement
    probs = stats.norm.cdf((mu - best) / std + 1e-9)
    return probs


def optimize_acquisition(X, y, model):
    nsamp = 100
    Xsamples = np.linspace(0, max(X), nsamp).reshape(nsamp, 1)
    scores = acquisition(X, Xsamples, model)
    ix = np.argmin(scores)
    return (Xsamples[ix, 0])


def plot(X, y, model):
    plt.scatter(X, y)
    nsamp = 100
    Xsamples = np.linspace(0, max(X), nsamp).reshape(nsamp, 1)
    ysamples, _ = surrogate(model, Xsamples)
    plt.plot(Xsamples, ysamples)
    plt.show()
