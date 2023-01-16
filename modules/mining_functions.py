from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
idx = pd.IndexSlice

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier

def run_mine_ml_model(model, all_inc_mines, to_predict, verbosity=0, accuracy=False):
    '''
    takes model function name in, e.g. KNeightborsClassifier
      or sklearn.naive_bayes.GaussianNB:
      sklearn.discriminant_analysis.LinearDiscriminantAnalysis
      sklearn.neighbors.KNeighborsClassifier
      sklearn.naive_bayes.GaussianNB
      sklearn.tree.DecisionTreeClassifier
      sklearn.svm.SVC
      ↑ can import these as e.g.:
      from sklearn.svm import SVC
    all_inc_mines is the collection of dataframes from previous
      year simulations that serve as our prediction basis.
      Should probably have at least 2 so we can assess effect
      of price change.
    to_predict is dataframe of mines we want to know whether
      they close
    Returns the index corresponding with mines that are
      predicted to open.
    '''
    input_df = all_inc_mines.copy().reset_index(drop=True)
    opening = input_df.copy()[['Production (kt)','Head grade (%)','Commodity price (USD/t)','TCRC (USD/t)',
                                 'Recovery rate (%)','OGE','Development CAPEX ($M)','Total cash margin (USD/t)',
                                 'Mine type','Sustaining CAPEX ($M)','Payable percent (%)']].dropna()
    opening.loc[:,'1=should open'] = (input_df['NPV ($M)']>0).astype(int)
    opening = opening
    exog = [i for i in opening.columns if i!='1=should open']
    X = opening[exog]
    Y = opening['1=should open']
    
    opening = to_predict.copy()[exog]
    if accuracy:
        opening.loc[:,'1=should open'] = (to_predict.copy()['NPV ($M)']>0).astype(int)
        Y_test = opening['1=should open']
    exog = [i for i in opening.columns if i!='1=should open']
    X_test = opening[exog]
    
    
    if model == KNeighborsClassifier:
        clf = model(n_neighbors=5)
    else:
        clf = model()
    
    x_train = X.values
    if (Y==1).all():
        Y.loc[Y.idxmin()] = 0
    elif (Y==0).all():
        Y.loc[Y.idxmax()] = 1
    y_train = Y.ravel()
    x_index = X_test.index
    x_test = X_test.values
    if accuracy:
        y_test = Y_test.values
        
    clf.fit(x_train,y_train)
    prediction = clf.predict(x_test)
    if verbosity>2:
        if accuracy:
            print(f'Accuracy for {model}:',round(accuracy_score(prediction, y_test),4))
        print('   Number of mines opening (actual, predicted):',np.sum(y_test),',',np.sum(clf.predict(x_test)))
    
    predict_prod = X_test.loc[x_index[prediction==1],'Production (kt)'].sum()
    if accuracy:
        actual_prod = X_test.loc[x_index[y_test],'Production (kt)'].sum()
        if verbosity>2:
            print('   actual production:',actual_prod,'\n   predicted production:',predict_prod)
        return x_index[prediction==1], accuracy_score(prediction, y_test)
    else:
        return x_index[prediction==1], np.nan

def all_non_consecutive(arr):
    ans = []
    start = arr[0]
    index = arr[0]
    end = arr[0]
    for number in arr:
        if index == number:
            index += 1
            end = number
        else:
            ans.append({'start': start, 'end': end})
            start = number
            index = number + 1
    ans.append({'start':start, 'end':arr[-1]})

    return ans

def partial_shuffle(a, part=0.5):
    '''input array and the fraction you want partially shuffled.
    from eumiro\'s answer at https://stackoverflow.com/questions/8340584/how-to-make-a-random-but-partial-shuffle-in-python'''
    seed(120121)
    # which characters are to be shuffled:
    idx_todo = sample(list(np.arange(0,len(a))), int(len(a) * part))

    # what are the new positions of these to-be-shuffled characters:
    idx_target = idx_todo[:]
    shuffle(idx_target)

    # map all "normal" character positions {0:0, 1:1, 2:2, ...}
    mapper = dict((i, i) for i in np.arange(len(a)))

    # update with all shuffles in the string: {old_pos:new_pos, old_pos:new_pos, ...}
    mapper.update(zip(idx_todo, idx_target))

    # use mapper to modify the string:
    return [a[mapper[i]] for i in np.arange(len(a))]

def supply_curve_plot(df, x_col, stack_cols, ax=0, dividing_line_width=0.2, 
                      price_col='', price_line_width=4,legend=True,legend_fontsize=19,legend_cols=2,
                      title='Cost supply curve',xlabel='Cumulative bauxite production (kt)',
                      ylabel='Total minesite cost ($/t)',unit_split=True,line_only=False,ylim=(0,71.5),
                      byproduct=False, **kwargs):
    '''Creates a stacked supply curve
    df: dataframe with time index
    x_col: str, name of column in dataframe to use as
      the value along x-axis (not cumulative, the
      function does the cumsum)
    stack_cols: list, the columns comprising the stack,
      their sum creates the supply curve shape
    ax: axis on which to plot
    dividing_line_width: float, linewidth for lines 
      separating the stacks, acts somewhat like shading
    price_col: str, name of column to plot as an additional
      line on the axes
    price_line_width: float, width of the line for price
    legend: bool
    legend_fontsize: float,
    legend_cols: int, number of columns for legend
    title: str
    xlabel: str
    ylabel: str
    unit_split: bool, split to remove units
    line_only: bool, whether to only plot the line
    ylim: tuple, default 0, 71.5
    **kwargs: arguments passed to easy_subplots'''
    if type(ax)==int:
        fig, ax = easy_subplots(1,1,**kwargs)
        ax = ax[0]
    # plt.figure(dpi=300)
    ph = df.copy()
    if len(stack_cols)>1:
        ph.loc[:,'Sort by'] = ph[stack_cols].sum(axis=1)
    else:
        ph.loc[:,'Sort by'] = ph[stack_cols[0]]
    ph = ph.sort_values('Sort by')
    ph_prod = ph[x_col].cumsum()
    ph.loc[:,'x plot'] = ph_prod
    ph.index=np.arange(2,int(max(ph.index)*2+3),2)
    ph1 = ph.copy().rename(dict(zip(ph.index,[i-1 for i in ph.index])))
    ph1.loc[:,'x plot'] = ph1['x plot'].shift(1).fillna(0)
    
    ph2 = pd.concat([ph,ph1]).sort_index()
    if line_only:
        if byproduct:
            n = 0
            for i in ph2['Byproduct ID'].unique():
                ph4 = ph2.loc[ph2['Byproduct ID']==i]
                ii = all_non_consecutive(ph4.index)
                for j in ii:
                    ph2.loc[j['start']:j['end'],'Cat'] = n
                    n+=1
            index_ph = ['Primary','Host 1','Host 2','Host 3'][:len(ph2['Byproduct ID'].unique())]
            colors = dict(zip(index_ph,['#d7191c','#fdae61','#abd9e9','#2c7bb6']))
            custom_lines = []
            for i in ph2.Cat.unique():
                if i!=ph2.Cat.iloc[-1]:
                    ind = ph2.loc[ph2.Cat==i].index
                    ax.vlines(ph2.loc[ind[-1],'x plot'], ph2.loc[ind[-1],'Sort by'], ph2.loc[ind[-1]+1,'Sort by'],color='k',linewidth=1)
                ax.step(ph2.loc[ph2.Cat==i,'x plot'],ph2.loc[ph2.Cat==i,'Sort by'],color = colors[ph2.loc[ph2.Cat==i,'Byproduct ID'].iloc[0]], label=ph2.loc[ph2.Cat==i,'Byproduct ID'].iloc[0])
            
            for i in index_ph:
                custom_lines += [Line2D([0],[0],color=colors[i])]
            ax.legend(custom_lines,index_ph)
        else:
            ax.plot(ph2['x plot'], ph2.loc[:,'Sort by'])
    else:
        ax.stackplot(ph2['x plot'],
                 ph2.loc[:,stack_cols].T,
                 labels=[i.split(' (')[0] if unit_split else i for i in stack_cols],
                 colors=['#d53e4f','#f46d43','#fdae61','#fee08b','#e6f598','#abdda4','#66c2a5','#3288bd'])
    if legend and not line_only:
        ax.legend(loc='upper left',fontsize=legend_fontsize,ncol=legend_cols)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if not line_only:
        ax.vlines(ph['x plot'],0,ph['Sort by'],color='k',linewidth=dividing_line_width)
    if len(price_col)>0:
        ax.step(ph1['x plot'],ph[price_col],label=price_col.split(' (')[0],linewidth=price_line_width)
    ax.set_ylim(ylim)
    return ph2