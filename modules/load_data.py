import pandas as pd
import numpy as np
import os
import shutil
from datetime import datetime

import warnings
idx = pd.IndexSlice

class LoadFolderContents:
    """"
    Used to load all the csv files from within a given scenario folder.
    If you only want a single commodity, load the entire folder and down-select
    from there, as single commodity (or subsets) are not supported.

    Will look for the folder in the current working directory, the path given,
    and in:
        generalization/output_files/Historical tuning
        generalization/output_files/Simulation
        output_files/Historical tuning
        output_files/Simulation

    ----------------------------------------
    OUTPUTS:
        No direct outputs, but saves the following variables as attributes of self
        - rmse_df
        - hyperparam
        - results
        - historical_data
        - rmse_df_sorted
        - hyperparam_sorted
        - results_sorted
    """

    def __init__(self, folder_path):
        if not os.path.exists(folder_path):
            potential_paths = ['generalization/output_files/Historical tuning',
                               'generalization/output_files/Simulation',
                               'output_files/Historical tuning',
                               'output_files/Simulation',
                               ]
            for path in potential_paths:
                if os.path.exists(path):
                    if folder_path in os.listdir(path):
                        self.folder_path = f'{path}/{folder_path}'
        else:
            self.folder_path = folder_path

        if not hasattr(self, 'folder_path'):
            raise ValueError('The folder path given has to give the relative or absolute path to your folder, '
                             'including the folder name, or give the folder name and we will check for it '
                             'inside the following paths:\n' + str(potential_paths + [os.getcwd()])
                             )

    def load_scenario_data(self):
        folder_path = self.folder_path
        if os.path.exists(folder_path):
            subfolders = [f'{self.folder_path}/{i}' for i in os.listdir(folder_path) if
                          i != 'input_files' and '.' not in i]
            self.subfolders = subfolders
            commodities = [i.split('_')[-1] for i in subfolders]
            for subfolder in subfolders:
                commodity = subfolder.split('_')[-1]
                for file in os.listdir(subfolder):
                    if file[-4:] == '.csv':
                        df_name = file.split('.csv')[0]
                        file = f'{subfolder}/{file}'
                        var_name = df_name + '_ph'
                        if 'historical_data' in file:
                            variable = pd.read_csv(file, index_col=[0])
                        else:
                            variable = pd.read_csv(file, index_col=[0, 1])
                            if 'rmse_df' in file:
                                variable = variable.iloc[:, 0].unstack(0)
                        setattr(self, var_name, variable)

                        if not hasattr(self, df_name):
                            setattr(self, df_name, pd.DataFrame())
                        so_far = getattr(self, df_name)
                        to_add = pd.concat([variable], keys=[commodity])
                        new = pd.concat([so_far, to_add])
                        setattr(self, df_name, new)

    def get_sorted_dataframes(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            if not hasattr(self, 'rmse_df'):
                return None
            self.rmse_df_sorted = pd.DataFrame()
            self.hyperparam_sorted = pd.DataFrame()
            self.results_sorted = pd.DataFrame()
            for commodity in self.rmse_df.index.get_level_values(0).unique():
                rmse = self.rmse_df.loc[commodity].copy()
                if 'score' not in rmse.index:
                    return None
                sorted_index = rmse.sort_values(by='score', axis=1).columns
                rmse_sorted = pd.concat([rmse.loc[:, sorted_index].T.reset_index(drop=True).T], keys=[commodity])
                self.rmse_df_sorted = pd.concat([self.rmse_df_sorted, rmse_sorted])
                for df_name in ['hyperparam', 'results']:
                    df = getattr(self, df_name).loc[commodity]
                    df = df.loc[idx[sorted_index, :], :]
                    ind = df.copy().index
                    level_0 = df.index.get_level_values(0).unique().sort_values()
                    level_1 = df.index.get_level_values(1).unique()
                    df = df.rename(dict(zip(df.index.get_level_values(0).unique(),
                                            np.arange(0, len(df.index.get_level_values(0).unique())))), level=0)
                    previous = getattr(self, df_name + '_sorted')
                    df = pd.concat([df], keys=[commodity])
                    setattr(self, df_name + '_sorted', pd.concat([previous, df]))


class ResavePklAsCsv:
    """
    Takes an entire folder and converts all pkl files to csv files, just initialize with
    the folder name and then use the run() method.
    """

    def __init__(self, output_data_folder='generalization/output_files/Historical tuning', many=None):
        self.many = many
        self.output_data_folder = output_data_folder
        self.time_strs = {}

        self.element_commodity_map = {'Steel': 'Steel', 'Al': 'Aluminum', 'Au': 'Gold', 'Cu': 'Copper',
                                      'Steel': 'Steel', 'Co': 'Cobalt', 'REEs': 'REEs', 'W': 'Tungsten', 'Sn': 'Tin',
                                      'Ta': 'Tantalum', 'Ni': 'Nickel', 'Ag': 'Silver', 'Zn': 'Zinc', 'Pb': 'Lead',
                                      'Mo': 'Molybdenum', 'Pt': 'Platinum', 'Te': 'Telllurium', 'Li': 'Lithium'}
        self.commodity_element_map = dict(zip(self.element_commodity_map.values(), self.element_commodity_map.keys()))

    def run(self):
        for file in os.listdir(self.output_data_folder):
            self.run_one(file)

    def run_one(self, file):
        print(file)
        self.file = file
        if file[-4:] == '.pkl':
            self.get_scenario_name()
            self.create_folders_subfolders()
            self.copy_input_files()
            self.save_results()

    def get_scenario_name(self):
        # Get scenario name and make sure they all get saved in the same folder (same timestamp)
        if 'Simulation' in self.output_data_folder:
            self.commodity = self.file.split('_')[0]
            self.scenario_name = self.file.split(f'{self.commodity}_')[1].split('.pkl')[0]
            num_strings = [str(i) for i in np.arange(0, 10)]
            if self.scenario_name[-1] in num_strings:
                self.scenario_id = self.scenario_name[-1]
                self.scenario_name = self.scenario_name[:-1]
        elif 'Historical tuning' in self.output_data_folder:
            self.scenario_name = self.file.split('_all_')[1].split('.pkl')[0]
            self.commodity = self.file.split('_run_hist')[0]

        if self.scenario_name not in self.time_strs:
            self.time_str = str(datetime.now()).replace(':', '_').replace('.', '_')[:21]
            self.time_strs[self.scenario_name] = self.time_str
        self.time_str = self.time_strs[self.scenario_name]
        self.element = self.commodity_element_map[self.commodity.capitalize()]

    def create_folders_subfolders(self):
        # Create scenario folder
        self.scenario_folder = f'{self.output_data_folder}/{self.time_str}_{self.scenario_name}'
        if not os.path.exists(self.scenario_folder):
            os.mkdir(self.scenario_folder)

        # Create data_element subfolder
        new_folder = f'data_{self.element}'
        self.data_folder = f'{self.scenario_folder}/{new_folder}'
        if not os.path.exists(self.data_folder):
            os.mkdir(self.data_folder)

        # Create input_files subfolder
        new_folder = 'input_files'
        self.input_files_folder = f'{self.scenario_folder}/{new_folder}'
        if not os.path.exists(self.input_files_folder):
            os.mkdir(self.input_files_folder)

    def copy_input_files(self):
        # Copy over input files
        user_data_folder = 'generalization/input_files/user_defined'
        for file_name in os.listdir(user_data_folder):
            # construct full file path
            source = f'{user_data_folder}/{file_name}'
            destination = f'{self.input_files_folder}/{file_name}'
            # copy only excel and csv files
            if '.xls' in source or '.csv' in source:
                shutil.copy(source, destination)

    def save_results(self):
        # Load pickle file
        pkl_file = pd.read_pickle(f'{self.output_data_folder}/{self.file}')
        write_data = {}
        write_data['results'] = pd.concat([pkl_file[i]['results']['Global'][0] for i in pkl_file.columns],
                                          keys=pkl_file.columns)
        write_data['hyperparam'] = pd.concat([pkl_file[i]['hyperparam'] for i in pkl_file.columns],
                                             keys=pkl_file.columns)
        write_data['rmse_df'] = pkl_file.iloc[:, -1]['rmse_df']
        write_data['historical_data'] = self.many.integ.historical_data.loc[self.commodity]
        if write_data['rmse_df'].index.nlevels == 1:
            write_data['rmse_df'].index = pd.MultiIndex.from_tuples(write_data['rmse_df'].index)

        # Save results from pickle file
        for i in ['results', 'hyperparam', 'rmse_df', 'historical_data']:
            file_path_string = f'{self.data_folder}/{i}.csv'
            if os.path.exists(file_path_string):
                if i != 'historical_data':
                    ind_col = [0, 1]
                    current = pd.read_csv(file_path_string, index_col=ind_col)
                    self.current = current.copy()
                    self.write_data = write_data
                    if i != 'rmse_df':
                        write_data[i] = pd.concat([write_data[i], current]).sort_index()
                    else:
                        if current.index.get_level_values(0).max() > write_data[i].index.get_level_values(0).max():
                            write_data[i] = current.copy()
            write_data[i].to_csv(file_path_string)


def create_output_folder_level_one():
    time_str = str(datetime.now()).replace(':', '_').replace('.', '_')[:21]
    output_data_folder = 'generalization/output_files'
    scenario_name = 'split_grades'
    sim_or_hist = 'Simulation'
    main_folder_str = f'{time_str} {scenario_name}'
    output_data_folder_full = f'{output_data_folder}/{sim_or_hist}'
    n_iter = 0
    MAX_RUNS_PER_SECOND = 3
    if main_folder_str not in os.listdir(output_data_folder_full):
        print(main_folder_str)
        final_folder = f'{output_data_folder_full}/{main_folder_str}'
        os.mkdir(final_folder)
    else:
        raise ValueError(
            'Scenario folder name already taken (needs to be run at least 0.1 seconds after the other with the same name)')
    return final_folder


def create_output_folder_level_two():
    commodity = 'copper'
    scenario_name = '2023-01-16 13_53_30_6 split_grades'
    output_data_folder = 'generalization/output_files'
    sim_or_hist = 'Simulation'
    output_data_folder_full = f'{output_data_folder}/{sim_or_hist}'
    folder_level_one = f'{output_data_folder_full}/{scenario_name}'
    if scenario_name not in os.listdir(output_data_folder_full):
        folder_level_one = create_output_folder_level_one()

    new_folder = f'{commodity} data'
    data_folder = f'{folder_level_one}/{new_folder}'
    os.mkdir(data_folder)

    new_folder = 'input_files'
    input_files_folder = f'{folder_level_one}/{new_folder}'
    os.mkdir(input_files_folder)

    return data_folder, input_files_folder


def add_input_files_to_folder():
    data_folder, input_files_folder = create_output_folder_level_two()
    user_data_folder = 'generalization/input_files/user_defined'
    print(input_files_folder)
    for file_name in os.listdir(user_data_folder):
        # construct full file path
        source = f'{user_data_folder}/{file_name}'
        destination = f'{input_files_folder}/{file_name}'
        # copy only excel and csv files
        if '.xls' in source or '.csv' in source:
            shutil.copy(source, destination)
