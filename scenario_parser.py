import pandas as pd


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


# TODO move this function to integration where it's commented out
def scenario_update_scrap_handling(self):
    update = self.scenario_update_df.copy()
    filename = update.index.names[0]
    scrap_parameters = ['scenario_type',
                        'collection_rate_price_response',
                        'direct_melt_price_response',
                        'secondary_refined_price_response',
                        'collection_rate_duration',
                        'collection_rate_pct_change_tot',
                        'collection_rate_pct_change_inc',
                        'direct_melt_duration',
                        'direct_melt_pct_change_tot',
                        'direct_melt_pct_change_inc',
                        'secondary_refined_duration',
                        'secondary_refined_pct_change_tot',
                        'secondary_refined_pct_change_inc',
                        ]
    update_index_level0 = update.index.get_level_values(0).unique()
    intersect = np.intersect1d(update_index_level0, scrap_parameters)
    scrap_param_in_update = update.loc[intersect].index.get_level_values(0)
    if len(scrap_param_in_update.unique()) != len(scrap_param_in_update):
        warnings.warn(
            f'Can only accept one row for each Scrap scenario in {filename}; other parameter categories are ok to have duplicates for different years. Only the first row will be implemented.')
    years = update.loc[intersect].index.get_level_values(1).unique()
    if np.any(years != years[0]):
        warnings.warn(
            'using 2019 for integ.scrap_shock_year since there are discrepancies on scrap year coming from the scenario input file: ' + filename)
        self.scrap_shock_year = 2019
    else:
        self.scrap_shock_year = int(years[0])

    for k in intersect:
        value = update.loc[k].iloc[0]
        setattr(self, k, value)
        self.hyperparam.loc[k, 'Value'] = value
