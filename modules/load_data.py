import pandas as pd
import numpy as np
import os
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
