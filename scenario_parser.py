import pandas as pd
import numpy as np
import warnings

scrap_parameters = ['scenario_type',
                    'collection_rate_price_response',
                    'direct_melt_price_response',
                    'secondary_refined_price_response',
                    'collection_rate_duration',
                    'collection_rate_pct_change_tot',
                    'collection_rate_pct_change_inc',
                    'scrap_demand_duration',
                    'scrap_demand_pct_change_tot',
                    'scrap_demand_pct_change_inc',
                    'direct_melt_duration',
                    'direct_melt_pct_change_tot',
                    'direct_melt_pct_change_inc',
                    'secondary_refined_duration',
                    'secondary_refined_pct_change_tot',
                    'secondary_refined_pct_change_inc',
                    ]

def get_scenario_ids(scen_cols):
    """
    Used to get scenario IDs and handle errors for the get_scenario_dataframe()
    function.

    - scen_cols: array of strings where the scenario name is separated from the
      leading word Scenario by a space. The scenario name should not have spaces.
    """
    try:
        scen_ids = [i.split(' ')[1] for i in scen_cols]
    except:
        raise ValueError(
            'Column format for scenario names is incorrect; requires a space between the `Scenario` label and the scenarioname')
    if len(scen_cols) == 0:
        raise ValueError('No columns were provided with the `Scenario` label')
    return scen_ids


def handle_scen_error(scen_col):
    """
    Used for error handling in get_scenario_dataframe funciton
    """
    if len(scen_col) - len(scen_col.replace(' ', '')) != 1:
        raise ValueError(f'Cannot have spaces in your scenarioname, see column: {scen_col}')


def get_scenario_dataframe(file_path_for_scenario_setup=None, default_year=None):
    """
    Takes a path to an excel file that is properly formatted for scenario inputs.
    In no year for the change is given, the default value for the year, 2019 will
    be used instead.

    - file_path_for_scenario_setup: str, relative or absolute path to excel file
      formatted to load scenarios
    - default_year: None or int, the default year used when no year is given for
      a scenario

    Returns a pandas series with a three-level index, in order: scenario name,
    variable name, year of change. The value given in the series is the new
    value to change to.
    """
    default_year = 2019 if default_year is None else default_year
    file_path_for_scenario_setup = 'generalization/Scenario setup.xlsx' if \
        file_path_for_scenario_setup is None else file_path_for_scenario_setup
    scen_data = pd.read_excel(file_path_for_scenario_setup)
    scen_data = scen_data.dropna(how='all').dropna(how='all', axis=1)
    scen_cols = [i for i in scen_data.columns if 'Scenario' in i]
    year_cols = [i for i in scen_data.columns if 'Year' in i]
    scen_ids = get_scenario_ids(scen_cols)
    scen_data = scen_data.loc[scen_data[scen_cols].notna().any(axis=1)]
    scenarios = pd.DataFrame()
    for scen_col, scen_id in zip(scen_cols, scen_ids):
        handle_scen_error(scen_col)
        year_col = [i for i in year_cols if i.split(' ')[1] == scen_id]
        if len(year_col) > 0:
            year_col = year_col[0]
            scen = scen_data.copy()[['Variable name', scen_col, year_col]]
        else:
            year_col = default_year
            scen = scen_data.copy()[['Variable name', scen_col]]
            scen['Year'] = default_year
        scen = scen.loc[scen[scen_col].notna()]
        scen = scen.rename(columns={scen_col: 'Scenario', year_col: 'Year'}).fillna(2019)
        scen = pd.concat([scen], keys=[scen_id])
        scenarios = pd.concat([scenarios, scen])
    scenarios = scenarios.reset_index(drop=False).set_index(['level_0', 'Variable name', 'Year'])['Scenario']
    scenarios.index = scenarios.index.set_names([file_path_for_scenario_setup, file_path_for_scenario_setup, 'Year'])
    return scenarios

# TODO ensure that demand volume growth rates get updated correctly, mine generation parameters too, since probably need to regenerate incentive pool depending on what we change

def scenario_update_scrap_handling(self):
    update = self.scenario_update_df.copy()
    self.scenario_type = ''
    self.collection_rate_price_response = self.hyperparam['Value']['collection_rate_price_response']
    self.direct_melt_price_response = self.hyperparam['Value']['direct_melt_price_response']
    self.secondary_refined_price_response = self.hyperparam['Value']['direct_melt_price_response']
    self.collection_rate_duration = 0
    self.collection_rate_pct_change_tot = 0
    self.collection_rate_pct_change_inc = 0
    self.direct_melt_duration = 0
    self.direct_melt_pct_change_tot = 0
    self.direct_melt_pct_change_inc = 0
    self.secondary_refined_duration = 0
    self.secondary_refined_pct_change_tot = 0
    self.secondary_refined_pct_change_inc = 0
    self.secondary_refined_alt = False
    self.direct_melt_alt = False

    update = check_year_consistency(self, update)
    update = check_duration_variables(self, update)
    update = check_scrap_demand_overlap(update)
    update = update_from_scrap_demand(self, update)
    update = update_percent_values(self, update)
    self.scenario_update_df = update.copy()
    update_scenario_type(self)

def check_year_consistency(self, update):
    filename = update.index.names[0]
    update_ind = update.index.get_level_values(0).unique()
    intersect = np.intersect1d(update_ind, scrap_parameters)
    scrap_param_in_update = update.loc[intersect].index.get_level_values(0)
    if len(scrap_param_in_update.unique()) != len(scrap_param_in_update):
        warnings.warn(
            f'\nCan only accept one row for each Scrap scenario in {filename}; other parameter categories are ok to have duplicates for different years. Only the first row will be implemented.')
    years = update.loc[intersect].index.get_level_values(1).unique()
    if len(years)>0 and np.any(years != years[0]):
        warnings.warn('''\nUsing 2019 for integ.scrap_shock_year since there are discrepancies on scrap year
coming from the scenario input file:\n''' + filename +
                      '''\nTo use a different year, ensure all values are matching in the Year column.
                      **This includes any values in the Year column left blank; please ensure that no 
                      Scrap scenario variables have blank Year columns if you want to change the year of the
                      scrap supply/demand change**''')
        self.scrap_shock_year = 2019
        update_ph = update.loc[intersect]
        update_ph.index = pd.MultiIndex.from_tuples([(k[0], self.scrap_shock_year) for k in update_ph.index])
        update = pd.concat([
            update.drop(intersect, level=0),
            update_ph
        ])
    else:
        self.scrap_shock_year = 2019 if len(years)==0 else int(years[0])
    return update


def check_duration_variables(self, update):
    """
    making sure the _duration variables are set correctly
    """
    update_ind = update.index.get_level_values(0).unique()
    intersect = np.intersect1d(update_ind, scrap_parameters)
    for q in ['collection_rate', 'scrap_demand']:
        if np.any([q in i for i in intersect]) \
                and q + '_duration' not in intersect:
            warnings.warn('\n' + q + '_duration not set, using default value 1')
            update = pd.concat([
                update,
                pd.Series(1, pd.MultiIndex.from_tuples([(q + '_duration', self.scrap_shock_year)])),
            ])
    return update


def check_scrap_demand_overlap(update):
    """
    checking that we don't have overlap between scrap_demand, direct_melt, and secondary_refined variables
    """
    update_ind = update.index.get_level_values(0).unique()
    intersect = np.intersect1d(update_ind, scrap_parameters)
    if np.any(['direct_melt' in k or 'secondary_refined' in k for k in intersect]) and \
            np.any(['scrap_demand' in k for k in intersect]):
        warnings.warn('''\nIncluding variables starting with direct_melt or secondary_refined
        alongside variables starting with scrap_demand means that the scrap_demand variable will
        supercede the others. If you want to specify the direct_melt and secondary_refined behavior 
        separately, set those variables and leave scrap_demand variables blank.
        ''')
        for k in [k for k in intersect if 'direct_melt' in k or 'secondary_refined' in k]:
            update = update.drop(k, level=0)
    return update


def update_from_scrap_demand(self, update):
    """
    updating the secondary refining and direct melt variables from scrap demand input
    """
    update_ind = update.index.get_level_values(0).unique()
    intersect = np.intersect1d(update_ind, scrap_parameters)

    for k in [i for i in intersect if 'scrap_demand' in i]:
        value = update.loc[k].iloc[0]
        if 'duration' in k:
            refine_val, direct_val = value, value
        else:
            refine_val = value * 0.4
            direct_val = value * 0.6
        refine_str = 'secondary_refined' + k.split('scrap_demand')[1]
        direct_str = 'direct_melt' + k.split('scrap_demand')[1]
        update = pd.concat([
            update,
            pd.Series(refine_val, pd.MultiIndex.from_tuples([(refine_str, self.scrap_shock_year)])),
            pd.Series(direct_val, pd.MultiIndex.from_tuples([(direct_str, self.scrap_shock_year)]))
        ])
    return update


def update_percent_values(self, update):
    """
    updating percentage values
    """
    update_ind = update.index.get_level_values(0).unique()
    intersect = np.intersect1d(update_ind, scrap_parameters)

    for k in intersect:
        value = update.loc[k].iloc[0]
        if '_pct_' in k:
            update.loc[k] = update.loc[[k]] / 100 + 1
            value = value / 100 + 1
        setattr(self, k, value)
    return update


def update_scenario_type(self):
    update = self.scenario_update_df.copy()
    update_ind = update.index.get_level_values(0).unique()
    intersect = np.intersect1d(update_ind, scrap_parameters)
    initial_scenario_type = self.scenario_type
    if np.any(['collection' in j for j in intersect]) and \
            np.any(['scrap_demand' in j for j in intersect]):
        self.scenario_type = 'both-alt'
    elif np.any(['collection' in j for j in intersect]):
        self.scenario_type = 'scrap supply'
    elif np.any(['scrap_demand' in j for j in intersect]):
        self.scenario_type = 'scrap demand-alt'
    if 'scenario_type' in intersect and initial_scenario_type != self.scenario_type:
        warnings.warn("""
        Giving a scenario_type input means there can be discrepancy between the
        other variables given and the scenario_type; ensure this variable is one of
        `scrap supply`, `scrap demand`, `scrap demand-alt`, `both`, or `both-alt`.

        """)
