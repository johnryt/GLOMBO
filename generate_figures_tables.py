from modules.Many import Many
from modules.figures_tables_functions import *
import os
cwd = os.getcwd()
if '\\' in cwd:
    spliton = '\\'
elif '/' in cwd:
    spliton = '/'
else:
    spliton = None
    if cwd=='generalizationOutside':
        os.chdir('generalization')
if spliton is not None and cwd.split(spliton)[-1]=='generalizationOutside':
    os.chdir('generalization')

USER = 'jwr' # if you have the files available to plot Figures S4-S16, set to 'jwr', else can leave as 'guest' or whatever you want

# these folders correspond to an older version of the model but are included for completeness
OLD_PAPER_MAIN_FOLDER = '2023-01-17 15_28_13_0_split_grades'
OLD_PAPER_SPLIT_2015_FOLDER = '2023-01-17 15_29_01_3_split_2015'
OLD_PAPER_SPLIT_2016_FOLDER = '2023-01-17 15_29_04_4_split_2016'
OLD_PAPER_SPLIT_2017_FOLDER = '2023-01-17 15_28_21_4_split_2017'

# these folder names correspond with the data used to create all figures for the publication. Please do not change these, 
# and instead create your own variables or simply update the function call.
PAPER_MAIN_FOLDER = '2023-08-22 18_09_43_8_run_hist_all_main'
PAPER_SPLIT_2015_FOLDER = '2023-08-25 23_06_21_1_run_hist_all_split_2015'
PAPER_SPLIT_2016_FOLDER = '2023-08-25 09_55_03_0_run_hist_all_split_2016'
PAPER_SPLIT_2017_FOLDER = '2023-08-23 22_09_31_6_run_hist_all_split_2017'

def output_figures_tables(main_folder, split_15_folder=None, split_16_folder=None, split_17_folder=None, user=None):
    """
    Generates and saves all figures (and associated data) and tables associated with 
    the model run, saving them in the `figures` and `tables` directories within the 
    folder given in main_folder. Figures and tables are labeled according to their
    corresponding label in the publication.

    All the split_XX_folder variables are optional, but all three are required to 
    generate any outputs associated with the train-test split data. 

    The user variable is for the author`s use, as proprietary data used for generating
    some of the initial data and plots in the Supporting Information could not be 
    saved in a publicly-accessible space. The functions called when the user string
    is equal to the author`s original value rely on data outside accessible file paths,
    and will produce errors if a run is attempted. Please contact the author for data
    questions related to these figures (Supplementary Figures 8-20 and 
    Supplementary Tables 2-7). The proprietary data source is S&P Global Capital IQ Pro.
    """
    many_sg = Many()
    many_sg.load_data(main_folder)


    if split_15_folder is not None:
        many_15 = Many()
        many_15.load_data(split_15_folder)
    else:
        print('no folder given for the train-test 2001-2015 split')

    if split_16_folder is not None:
        many_16 = Many()
        many_16.load_data(split_16_folder)
    else:
        print('no folder given for the train-test 2001-2016 split')

    if split_17_folder is not None:
        many_17 = Many()
        many_17.load_data(split_17_folder)
    else:
        print('no folder given for the train-test 2001-2017 split')

    figures_2_and_3(many_sg)
    figure_4(many_sg)
    figures_5_and_s31(many_sg)
    
    figure_s4(many_sg)
    figure_s5(many_sg)
    figure_s6(many_sg)
    figure_s7(many_sg)
    figure_s21(many_sg)
    figure_s22_to_s24(many_sg)
    figure_s25(many_sg)
    figure_s26(many_sg)
    figure_s33(many_sg)
    figures_s34_and_s36(many_sg)
    figure_s35(many_sg)
    figures_s37_to_s40(many_sg)
    figure_s41(many_sg)
    
    if np.all([i is not None for i in [split_15_folder, split_16_folder, split_17_folder]]):
        table_s27(many_sg,  many_17, many_16, many_15)
        figure_s27(many_sg, many_15, many_16, many_17, show=False)
        figures_s28_and_s29(many_sg, many_17, many_16, many_15, show=False, all_only=False)
        figure_s30(many_sg, many_17, many_16, many_15, show=False)
        figure_s32([many_sg, many_17, many_16, many_15], show=False)
        

    if user=='jwr':
        figure_s8_to_s20_table_s2_to_s7(many_sg)
    return

output_figures_tables(PAPER_MAIN_FOLDER, PAPER_SPLIT_2015_FOLDER, PAPER_SPLIT_2016_FOLDER, PAPER_SPLIT_2017_FOLDER, USER)
# output_figures_tables(PAPER_MAIN_FOLDER, None, None, None, USER)

