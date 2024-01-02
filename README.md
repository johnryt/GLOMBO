# GLOMBO
## Data locations
 This repository serves as the source of code, data, figures, and tables associated with the publication _Understanding key mineral supply chain dynamics using economics-informed material flow analysis and Bayesian optimization_. The data associated with Table 1 of the main text and Supplementary Tables 8-26 can be found at `input_files/user_defined/case study data.xlsx`. Other input data can be found in the `input_files` folder as well, but data, figures, and tables corresponding to those in the text are saved in the folder of the corresponding main model run, in this case `output_files/Historical tuning/2023-08-22 18_09_43_8_run_hist_all_main`, in the `figures` and `tables` folders contained within it. Note that this folder also contains the folder `input_files`, which contains a copy of each file within the `main/input_files` folder from when the code was executed (as record in case input files are modified).

 The database of literature values may be found in the `Sources for elasticities.xlsx` file in the main folder, or the updated version post-processing in `output_files/Historical tuning/2023-08-22 18_09_43_8_run_hist_all_main/tables/table_s28_literature_parameter_database.csv`. 

## Creating GLOMBO working environment
If intending to run any code from this repository, it is recommended to create a working environment such that all python packages are consistent between your machine and the author's. With this repository cloned to your machine (either using command line git, GitHub Desktop, or similar), the following code can be executed in the terminal (typically Anaconda Prompt on Windows), ensuring your working directory is the location of the cloned repository (the folder with glombo.yml in it):
```
conda env create -f glombo.yml
```

The same can be accomplished by installing this package to your machine, which can be done using pip in the command line via:
```
pip install git+https://github.com/johnryt/GLOMBO
```
You can check the glombo working environment was created by running `conda env list` in your terminal. With this accomplished, you can begin running the model or reproducing its figures (or reproducing figures using alternative model tuning runs). 

## Generating figures
 To save space, only the data comprising each figure is saved on github and not the figures themselves (note *.png and *.pdf in .gitignore), but these can be produced by running the python file `generate_figures_tables.py` in the main folder. This function will also print out many of the data points referenced in the text, such as the number of unique publications in the database, the fraction of MAPE changes associated with varying train-test splits less than 1%, etc. 

 ## Running the full model
 The full historical tuning presented in the paper can be run using the `run.py` python file. It took several days on a PC with a 13th generation Intel i9 2200 MHz, 24 core processor and 64 GB of RAM, so understand that this is a substantial commitment. 

 All the remaining code is within the `modules` folder, with `Many.py` the most useful for running several iterations of the model (recommended in all circumstances, as there are instances of instability).
 
