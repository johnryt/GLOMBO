import numpy as np
import pandas as pd
import statsmodels.api as sm
idx = pd.IndexSlice
from matplotlib import pyplot as plt
import seaborn as sns
from cycler import cycler
from datetime import datetime

def get_sheet_details(file_path):
    sheets = []
    file_name = os.path.splitext(os.path.split(file_path)[-1])[0]
    # Make a temporary directory with the file name
    directory_to_extract_to = os.path.join(os.getcwd(), file_name)
#     print(file_path)
    try:
        os.mkdir(directory_to_extract_to)
    except FileExistsError:
        shutil.rmtree(directory_to_extract_to)
        os.mkdir(directory_to_extract_to)


    # Extract the xlsx file as it is just a zip file
    zip_ref = zipfile.ZipFile(file_path, 'r')
    zip_ref.extractall(directory_to_extract_to)
    zip_ref.close()

    # Open the workbook.xml which is very light and only has meta data, get sheets from it
    path_to_workbook = os.path.join(directory_to_extract_to, 'xl', 'workbook.xml')
    with open(path_to_workbook, 'r') as f:
        xml = f.read()
        dictionary = xmltodict.parse(xml)
        for sheet in dictionary['workbook']['sheets']['sheet']:
            sheet_details = {
                'id': sheet['@sheetId'], # can be @sheetId for some versions
                'name': sheet['@name'] # can be @name
            }
            sheets.append(sheet_details)

    # Delete the extracted files directory
    shutil.rmtree(directory_to_extract_to)
    return sheets

def get_sheet_details(file_path):
    sheet_names = []
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        xml = zip_ref.open(r'xl/workbook.xml').read()
        dictionary = xmltodict.parse(xml)

        if not isinstance(dictionary['workbook']['sheets']['sheet'], list):
            sheet_names.append(dictionary['workbook']['sheets']['sheet']['@name'])
        else:
            for sheet in dictionary['workbook']['sheets']['sheet']:
                sheet_names.append(sheet['@name'])
    return sheet_names

def init_plot2(fontsize=20,figsize=(8,5.5),font='Arial',font_family='sans-serif',linewidth=4,font_style='bold',have_axes=True,dpi=50,marker=None,markersize=12,markeredgewidth=1.0,markeredgecolor=None,markerfacecolor=None, markercycler=False, cmap='Dark2', n_colors=8, **kwargs):
    '''Sets default plot formats.
    Potential inputs: fontsize, figsize, font,
    font_family, font_style, linewidth, have_axes,
    dpi, marker, markersize, markeredgewidth,
    markeredgecolor, markerfacecolor.
    have_axes: determines whether there is a border
    on the plot. Also has **kwargs so that any other
    arguments that can be passed to mpl.rcParams.update
    that were not listed above.

    cmap can take any matplotlib colormap string.'''
    import matplotlib as mpl
    params = {
        'axes.labelsize': fontsize,
        'font.size': fontsize,
        'axes.titlesize':fontsize+1,
        'figure.titlesize':fontsize+1,
        'legend.fontsize': fontsize,
        'xtick.labelsize': fontsize,
        'ytick.labelsize': fontsize,
        'axes.titlesize': fontsize+2,
        'figure.titlesize': fontsize+2,
        'text.usetex': False,
        'figure.figsize': figsize,
        'lines.linewidth': linewidth,
        'lines.solid_capstyle': 'round',
        'legend.framealpha': 1,
        'legend.frameon': False,
        'mathtext.default': 'regular',
        'axes.linewidth': 2/3*linewidth,
        'xtick.direction': 'in', # in, out, inout
        'ytick.direction': 'in', # in, out, inout
        'xtick.major.size': 7,
        'xtick.major.width': 2,
        'xtick.major.pad': 3.5,
        'ytick.major.size': 7,
        'ytick.major.width': 2,
        'ytick.major.pad': 3.5,
        'font.family': font_family,
        'font.'+font_family: font,
        'figure.dpi': dpi,
        'lines.marker': marker,
        'lines.markersize':markersize,
        'lines.markeredgewidth':markeredgewidth,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'savefig.transparent': True,
        'savefig.bbox': 'tight',
        }

    mpl.rcParams.update(params)
    mpl.rcParams.update(**kwargs)
    mpl.rcParams['axes.spines.left'] = have_axes
    mpl.rcParams['axes.spines.right'] = have_axes
    mpl.rcParams['axes.spines.top'] = have_axes
    mpl.rcParams['axes.spines.bottom'] = have_axes
    mpl.rcParams['axes.axisbelow'] = True
    mpl.rcParams['axes.grid'] = True
    mpl.rcParams['axes.grid.axis'] = 'y'
    mpl.rcParams['grid.color'] = '0.9'
    mpl.rcParams['grid.linewidth'] = 1
    mpl.rcParams['grid.linestyle'] = '-'
    mpl.rcParams['axes.labelweight'] = font_style
    mpl.rcParams['font.weight'] = font_style

    sns.set_palette(cmap)

    if markeredgecolor != None:
        mpl.rcParams['lines.markeredgecolor'] = markeredgecolor
    if markerfacecolor != None:
        mpl.rcParams['lines.markerfacecolor'] = markerfacecolor
    if markercycler:
        cmap_colors = mpl.cm.get_cmap(cmap)
        colors = [cmap_colors(i) for i in np.linspace(0,1,n_colors)]
        markers = ["o","s","^","v","p","8","P","X"]
        if n_colors>8: markers = np.tile(markers,int(np.floor(n_colors/8)+1))
        markers = markers[:n_colors]
        mpl.rcParams['axes.prop_cycle'] =  cycler('color',colors) + cycler('marker',markers)

def reduce_mem_usage(df,inplace=False):
    '''Returns dataframe with columns changed to have dtypes of minimum
    size for the values contained within. Does not adjust object dtypes.
    From https://www.kaggle.com/gemartin/load-data-reduce-memory-usage'''
    if inplace:
        props = df
    else:
        props = df.copy()
    props = props.drop_duplicates().T.drop_duplicates().T
    start_mem_usg = props.memory_usage().sum() / 1024**2
    print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    NAlist = [] # Keeps track of columns that have missing values filled in.
    progress = int(np.floor(len(props.columns)/10))
    n = 1
    for num, col in enumerate(props.columns):
        prop_col = props[col]
        try:
            col_type = prop_col.dtypes.values[0]
        except Exception as e:
#             print(e)
            if 'numpy.dtype[' in str(e):
                col_type = prop_col.dtypes


        if col_type != object:  # Exclude strings

            # Print current column type
#             print("******************************")
#             print("Column: ",col)
#             print("dtype before: ",col_type)

            # make variables for Int, max and min
            IsInt = False
            try:
                mx = prop_col.max().max()
                mn = prop_col.min().min()
            except:
                mx = prop_col.max()
                mn = prop_col.min()


            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(prop_col).all().all():
                NAlist.append(col)
                props[col].fillna(mn-1,inplace=True)

            # test if column can be converted to an integer
            asint = prop_col.fillna(0).astype(np.int64)
            result = (prop_col - asint)
            result = result.sum().sum()
            if result > -0.01 and result < 0.01:
                IsInt = True


            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mx < 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mx < 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)

            # Make float datatypes 32 bit
            else:
                props[col] = props[col].astype(np.float32)

            # Print new column type
            try:
                col_type = props[col].dtypes.values[0]
            except Exception as e:
#                 print(e)
                if 'numpy.dtype[float64]' in str(e):
                    col_type = props[col].dtypes
#             print("dtype after: ",col_type)
#             print("******************************")
        if num == progress*n:
            print('{}0% complete'.format(n))
            n+=1

    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024**2
    print("Memory usage is: ",mem_usg," MB")
    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return props

def twinx2(ax1,tw,n=2):
    '''Primary axis, secondary axis, number of digits to round to (default 2), does not return anything.
    Sets up secondary y-axis to be aligned with the primary axis ticks.'''
    l = ax1.get_ylim()
    l2 = tw.get_ylim()
    f = lambda x : l2[0]+(x-l[0])/(l[1]-l[0])*(l2[1]-l2[0])
    ticks = f(ax1.get_yticks())
    tw.yaxis.set_major_locator(mpl.ticker.FixedLocator(ticks))
    tw.grid(None)
    tw.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.'+str(n)+'f'))

def do_a_regress(x,y,ax=0,intercept=True,scatter_color='tab:blue',line_color='k',
                 xlabel='independent var',ylabel='dependent var',log=False,print_log=False,
                 plot=True,loc='upper left',add_labels_bool=False,force_predict_v_actual=False):
    '''Performs regression between two pandas series or dataframes.
    x: pd.Series() or pd.DataFrame() of the independent variable(s)
    y: pd.Series() or pd.DataFrame() of the dependent variable
    ax: matplotlib axes to plot on, defaults to creating new figure
    intercept: boolean, whether or not to give the regression a
      constant, default True
    scatter_color: matplotlib interpretable color, default tab:blue
    line_color: matplotlib interpretable color, default black (k)
    xlabel: string, overwrite series or column name for independent
      variable
    ylabel: string, overwrite series or column name for dependent
      variable
    plot: boolean, whether or not to plot the regression result
    log: bool, whether to log transform both x and y
    print_log: bool, whether to print x/y values lost during log
    loc: text location of regression equation if plotting. Options:
      bottom right, lower right, upper left (default), or upper right
    add_labels: boolean, adds index values to each scatterplot point,
      not recommended for large datasets, default False
    force_predict_v_actual: boolean, whether to force the plot to
      plot the regression predicted values vs the actual values
      instead of y vs x. Default False but force True if there is
      more than one independent variable given.
    Returns tuple of (series with model parameters, fitted model) if
      force_predict_v_actual==False.
    If force_predict_v_actual is True, returns tuple of
      (series with model parameters, fitted predicted vs actual model,
      fitted y vs x model)'''
    if type(x) == pd.core.frame.DataFrame:
        if x.shape[1]>1:
            force_predict_v_actual = True
        elif x.shape[1]==1:
            x = x[x.columns[0]]
    try:
        x.name == None
    except:
        x.name = xlabel
    try:
        y.name == None
    except:
        y.name = ylabel
    if x.name == None:
        x.name = xlabel
    if xlabel != 'independent var':
        x.name = xlabel
    if ylabel != 'dependent var':
        y.name = ylabel
    if log:
        if len(x[x>0]) != len(x) and print_log:
            print(f'{len(x[x>0])} negative/zero/nan x values lost')
        x = x[x>0]
        if len(y[y>0]) != len(y) and print_log:
            print(f'{len(y[y>0])} negative/zero/nan y values lost')
        y = y[y>0]
        ind = np.intersect1d(x.index,y.index)
        if (len(ind) != len(x.index) or len(ind) != len(y.index)) and print_log:
            print(f'{len(x.index)-len(ind)} x values lost, {len(y.index)-len(ind)} y values lost to unaligned indices')
        x,y = np.log(x.loc[ind]), np.log(y.loc[ind])
        x.name, y.name = 'log('+str(x.name)+')', 'log('+str(y.name)+')'

    if intercept:
        x_i = sm.add_constant(x)
    else:
        x_i = x.copy()
    m = sm.GLS(y,x_i,missing='drop').fit(cov_type='HC3')

    if plot and not force_predict_v_actual:
        if type(ax) == int:
            fig,ax = plt.subplots()
        ax.scatter(x,y,color=scatter_color)
        if add_labels_bool:
            add_labels(x,y,ax)
        try:
            x = x.loc[[x.idxmax(),x.idxmin()]]
        except:
            x = x.loc[[x.idxmax()[0],x.idxmin()[0]]]
        if intercept:
            ax.plot(x, m.params['const'] + m.params[x.name]*x,label='Best-fit line',color=line_color)
        else:
            ax.plot(x, m.params[x.name]*x,label='Best-fit line',color=line_color)

        add_regression_text(m,x,y,loc=loc,ax=ax)

        ax.set(xlabel=x.name, ylabel=y.name)

    elif force_predict_v_actual:
        y_predicted = m.predict(x_i)
        if plot:
            if type(ax)==int:
                fig,ax = plt.subplots()
            y.name = 'Actual'
            y_predicted.name = 'Predicted'
            m_predict_v_actual = do_a_regress(y,y_predicted,ax=ax,intercept=intercept,
                                              scatter_color=scatter_color,line_color=line_color,
                                              xlabel='Actual',ylabel='Predicted',plot=plot,loc=loc,
                                              add_labels_bool=add_labels_bool,force_predict_v_actual=False)[1]


    if force_predict_v_actual:
        return pd.Series(m.params).rename({x.name:'slope','x1':'slope'}),m_predict_v_actual,m
    else:
        return pd.Series(m.params).rename({x.name:'slope','x1':'slope'}),m

def easy_subplots(nplots, ncol=4, height_scale=1,width_scale=1,use_subplots=False,**kwargs):
    '''sets up plt.subplots with the correct number of rows and columns and figsize,
    given number of plots (nplots) and the number of columns (ncol).
    Option to make figures taller or shorter by changing the height_scale.
    Can also give additional arguments to either plt.figure or plt.subplots.
    use_subplots: True to use plt.subplots to create axes, False to use
      plt.figure, which then allows dpi to be specified.'''
    if type(nplots) != int:
        nplots = len(nplots)
    if nplots <= ncol:
        ncol = nplots
    if nplots%3==0 and ncol==4:
        ncol=3

    nrows = int(np.ceil(nplots/ncol))
    figsize = (7*ncol*width_scale,height_scale*6*int(np.ceil(nplots/ncol)))
    regular_version = False
    if use_subplots:
        fig, ax = plt.subplots(nrows,ncol,**kwargs,
                          figsize=figsize)
    else:
        fig = plt.figure(**kwargs, figsize = figsize)
        ax = []
        for i in np.arange(1,int(np.ceil(nrows*ncol))+1):
            ax += [fig.add_subplot(nrows, ncol, i)]
    return fig,np.array(ax).flatten()

def add_labels(x,y,ax):
    '''Add labels to scatter plot for each common index in series x and y.'''
    for i in np.intersect1d(x.index,y.index):
        ax.text(x.loc[i],y.loc[i],i,ha='center',va='top')

def twinx2(ax1,tw,n=2):
    '''Primary axis, secondary axis, number of digits to round to (default 2),
    does not return anything.'''
    l = ax1.get_ylim()
    l2 = tw.get_ylim()
    f = lambda x : l2[0]+(x-l[0])/(l[1]-l[0])*(l2[1]-l2[0])
    ticks = f(ax1.get_yticks())
    tw.yaxis.set_major_locator(mpl.ticker.FixedLocator(ticks))
    tw.grid(None)
    tw.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.'+str(n)+'f'))

def kstest100(x):
    '''Takes in series, creates 100 simulated normal
    distributions from the series mean, std, and length,
    and returns the mean coefficient and p-value of the
    Kolmogorov-Smirnov test of the series x and its
    simulated counterpart.'''
    coef, pval = [], []
    for n in np.arange(0,100):
        x_sim = stats.norm.rvs(loc=x.mean(),scale=x.std(),size=len(x),random_state=n)
        result = stats.kstest(x,x_sim)
        coef += [result[0]]
        pval += [result[1]]
    return np.mean(coef),np.mean(pval)

def add_regression_text(m,x,y,loc='bottom right',ax=0):
    '''
    adds the regression line equation to a plot of y vs x for model m.
    m = sm.GLS() model,
    x = pd.Series() or pd.DataFrame() of the independent variable(s)
    y = pd.Series() or pd.DataFrame() of the dependent variable
    loc = text location: bottom right, lower right, upper left, or upper right
    ax = axes on which to add the regression text
    '''
    n0 = 'e' if abs(m.params[0]) < 1e-2 else 'f'
    if len(m.params.index) > 1:
        n1 = 'e' if abs(m.params[1]) < 1e-2 else 'f'
        plus_or_minus = '+' if m.params['const'] > 0 else '-'

    if type(ax)==int:
        if 'const' in m.params.index:
            if loc=='bottom right':
                plt.text(x.min(),y.min(),
                    ('y = {:.3'+n1+'}x '+plus_or_minus+' {:.3'+n0+'}\nR-squared: {:.3f}\nStd dev equiv: {:.3f}').format(m.params[1],abs(m.params[0]),m.rsquared,m.mse_resid**0.5),
                     ha='left',va='bottom')
            elif loc=='lower right':
                plt.text(x.max(),y.min(),
                    ('y = {:.3'+n1+'}x '+plus_or_minus+' {:.3'+n0+'}\nR-squared: {:.3f}\nStd dev equiv: {:.3f}').format(m.params[1],abs(m.params[0]),m.rsquared,m.mse_resid**0.5),
                     ha='right',va='bottom')
            elif loc=='upper left':
                plt.text(x.min(),y.max(),
                    ('y = {:.3'+n1+'}x '+plus_or_minus+' {:.3'+n0+'}\nR-squared: {:.3f}\nStd dev equiv: {:.3f}').format(m.params[1],abs(m.params[0]),m.rsquared,m.mse_resid**0.5),
                     ha='left',va='top')
            elif loc=='upper right':
                plt.text(x.max(),y.max(),
                    ('y = {:.3'+n1+'}x '+plus_or_minus+' {:.3'+n0+'}\nR-squared: {:.3f}\nStd dev equiv: {:.3f}').format(m.params[1],abs(m.params[0]),m.rsquared,m.mse_resid**0.5),
                     ha='right',va='top')
        else:
            if loc=='bottom right':
                plt.text(x.min(),y.min(),
                    ('y = {:.3'+n1+'}x\nR-squared: {:.3f}\nStd dev equiv: {:.3f}').format(m.params[1],m.rsquared,m.mse_resid**0.5),
                     ha='left',va='bottom')
            elif loc=='lower right':
                plt.text(x.max(),y.min(),
                    ('y = {:.3'+n1+'}x\nR-squared: {:.3f}\nStd dev equiv: {:.3f}').format(m.params[1],m.rsquared,m.mse_resid**0.5),
                     ha='right',va='bottom')
            elif loc=='upper left':
                plt.text(x.min(),y.max(),
                    ('y = {:.3'+n1+'}x\nR-squared: {:.3f}\nStd dev equiv: {:.3f}').format(m.params[1],m.rsquared,m.mse_resid**0.5),
                     ha='left',va='top')
            elif loc=='upper right':
                plt.text(x.max(),y.max(),
                    ('y = {:.3'+n1+'}x\nR-squared: {:.3f}\nStd dev equiv: {:.3f}').format(m.params[1],m.rsquared,m.mse_resid**0.5),
                     ha='right',va='top')
    else:
        if 'const' in m.params.index:
            if loc=='bottom right':
                ax.text(x.min(),y.min(),
                    ('y = {:.3'+n1+'}x '+plus_or_minus+' {:.3'+n0+'}\nR-squared: {:.3f}\nStd dev equiv: {:.3f}').format(m.params[1],abs(m.params[0]),m.rsquared,m.mse_resid**0.5),
                     ha='left',va='bottom')
            elif loc=='lower right':
                ax.text(x.max(),y.min(),
                    ('y = {:.3'+n1+'}x '+plus_or_minus+' {:.3'+n0+'}\nR-squared: {:.3f}\nStd dev equiv: {:.3f}').format(m.params[1],abs(m.params[0]),m.rsquared,m.mse_resid**0.5),
                     ha='right',va='bottom')
            elif loc=='upper left':
                ax.text(x.min(),y.max(),
                    ('y = {:.3'+n1+'}x '+plus_or_minus+' {:.3'+n0+'}\nR-squared: {:.3f}\nStd dev equiv: {:.3f}').format(m.params[1],abs(m.params[0]),m.rsquared,m.mse_resid**0.5),
                     ha='left',va='top')
            elif loc=='upper right':
                ax.text(x.max(),y.max(),
                    ('y = {:.3'+n1+'}x '+plus_or_minus+' {:.3'+n0+'}\nR-squared: {:.3f}\nStd dev equiv: {:.3f}').format(m.params[1],abs(m.params[0]),m.rsquared,m.mse_resid**0.5),
                     ha='right',va='top')
        else:
            if loc=='bottom right':
                ax.text(x.min(),y.min(),
                    ('y = {:.3'+n1+'}x\nR-squared: {:.3f}\nStd dev equiv: {:.3f}').format(m.params[1],m.rsquared,m.mse_resid**0.5),
                     ha='left',va='bottom')
            elif loc=='lower right':
                ax.text(x.max(),y.min(),
                    ('y = {:.3'+n1+'}x\nR-squared: {:.3f}\nStd dev equiv: {:.3f}').format(m.params[1],m.rsquared,m.mse_resid**0.5),
                     ha='right',va='bottom')
            elif loc=='upper left':
                ax.text(x.min(),y.max(),
                    ('y = {:.3'+n1+'}x\nR-squared: {:.3f}\nStd dev equiv: {:.3f}').format(m.params[1],m.rsquared,m.mse_resid**0.5),
                     ha='left',va='top')
            elif loc=='upper right':
                ax.text(x.max(),y.max(),
                    ('y = {:.3'+n1+'}x\nR-squared: {:.3f}\nStd dev equiv: {:.3f}').format(m.params[1],m.rsquared,m.mse_resid**0.5),
                     ha='right',va='top')

def find_best_dist(stacked_df, plot=True, print_chi_squared=False, bins=40, ax=0, density=False):
    '''takes a stacked dataframe and outputs a list
    of distributions that best fits that data (descending).
    stacked_df: pandas dataframe or series
    plot: bool, whether to plot the given data and
       the simulated data from best dist.
    print_chi_squared: bool, whether to print
       distribution results (minimize chi sq)
    bins: int, number of bins in histogram
    ax: matplotlib axis
    density: bool, whether to use length of data
       to form simulated distribution or to use
       1000 points and a normalized histogram
    '''
    y = stacked_df.copy()
    x = stacked_df.copy()
#     y = y[y>0]
    dist_names = ['weibull_min','norm','weibull_max','beta','invgauss','uniform','gamma','expon','lognorm','pearson3','triang','powerlognorm','logistic']
    chi_square_statistics = []
    # 11 equi-distant bins of observed Data
    percentile_bins = np.linspace(0,100,11)
    percentile_cutoffs = np.percentile(y, percentile_bins)
    observed_frequency, hist_bins = (np.histogram(y, bins=percentile_cutoffs))
    cum_observed_frequency = np.cumsum(observed_frequency)

    # Loop through candidate distributions
    for distribution in dist_names:
        # Set up distribution and get fitted distribution parameters
        dist = getattr(stats, distribution)
        try:
            param = dist.fit(y)
        except:
            param = [0,1]
#         print("{}\n{}\n".format(dist, param))


        # Get expected counts in percentile bins
        # cdf of fitted sistrinution across bins
        cdf_fitted = dist.cdf(percentile_cutoffs, *param)
        expected_frequency = []
        for bin in range(len(percentile_bins)-1):
            expected_cdf_area = cdf_fitted[bin+1] - cdf_fitted[bin]
            expected_frequency.append(expected_cdf_area)

        # Chi-square Statistics
        expected_frequency = np.array(expected_frequency) * y.size
        cum_expected_frequency = np.cumsum(expected_frequency)
        ss = sum (((cum_expected_frequency - cum_observed_frequency) ** 2) / cum_observed_frequency)
        chi_square_statistics.append(ss)


    #Sort by minimum chi-square statistics
    results = pd.Series(chi_square_statistics,dist_names).sort_values()

    if print_chi_squared:
        print ('\nDistributions listed by goodness of fit:')
        print ('.'*40)
        print (results)

    if results.notna().any():
        best_dist = getattr(stats,results.idxmin())
    else:
        best_dist = getattr(stats,'anglit')
#     print(results.idxmin())
    best_params = best_dist.fit(y)
#     print(best_params)
    best_sim_size = 1000 if density else len(y)
    if len(best_params)==2:
        best_sim = best_dist.rvs(best_params[0],best_params[1],size=len(y),random_state=0)
    elif len(best_params)==3:
        best_sim = best_dist.rvs(best_params[0],best_params[1],best_params[2],size=len(y),random_state=0)
    elif len(best_params)==4:
        best_sim = best_dist.rvs(best_params[0],best_params[1],best_params[2],best_params[3],size=len(y),random_state=0)

    if plot:
        if type(ax)==int:
            fig,ax = easy_subplots(1,1)
            ax = ax[0]
        ax.hist(x.values.flatten(),bins=np.linspace(y.min(),y.max(),bins),color='tab:blue',alpha=0.5,density=density)
        ax.hist(best_sim,bins=np.linspace(y.min(),y.max(),bins),color='tab:orange',alpha=0.5,density=density)
        ax.set(title=results.index[0])
#         ax[1].plot(y,best_dist.pdf(y,))
    return results.index

def year_decimal_to_datetime(year_decimal):
    '''
    Takes in any year-as-decimal value and returns
    the equivalent (to microsecond) datetime form.
    Can be used to convert an entire index, i.e.
    df.index = [year_decimal_to_datetime(j) for j in df.index]
    '''
    n_years = np.floor(year_decimal)
    months = (year_decimal-n_years)*12+1
    n_months = np.floor(months)
    days_in_month = 31 if (n_months%2!=0 and n_months<8) or (n_months%2==0 and n_months>=8) else 30 if n_months!=2 else 28 if n_years%4!=0 else 29
    days = (months-n_months)*days_in_month+1
    n_days = np.floor(days)
    hours = (days-n_days)*24
    n_hours = np.floor(hours)
    minutes = (hours-n_hours)*60
    n_minutes = np.floor(minutes)
    seconds = (minutes-n_minutes)*60
    n_seconds = np.floor(seconds)
    microseconds = (seconds-n_seconds)*1000
    n_microseconds = np.floor(microseconds)
    return datetime(int(n_years), int(n_months), int(n_days), int(n_hours), int(n_minutes), int(n_seconds), int(n_microseconds))

init_plot2()
