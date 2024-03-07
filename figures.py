import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter
from lifelines.plotting import add_at_risk_counts
from tableone import TableOne
import math
import statsmodels.api as sm
import warnings
from scipy.stats import chi2_contingency
from load_data import *

def baselineCharacteristics(df_input, pval=True, lite = False, csv_filename=None):
    """Generate baseline characteristics table, prints results into console

        Parameters
        ----------
        df_input - dataframe to use
        """
    
    df = df_input.copy()
    print("Median Follow-up: " + str(df['last_contact'].median()))
    print("25% IQR: " + str(df['last_contact'].quantile(0.25)))
    print("75% IQR: " + str(df['last_contact'].quantile(0.75)))
    
    columns = ['age', 'sex', 'race_parse', 'facility', 'cdcc', 'year', 'grade', 'histology', 't_stage', 'n_stage', 'stage', 'her2ratio', 'her2copies', 'er','pr', 'er_num', 'pr_num', 'ihc', 'ki67_num', 'recurrence_score_group', 'immuno', 'chemo', 'hormone', 'pcr', 'last_contact', 'alive']
    labels = {'age':'Age', 'age_group':'Age', 'sex':'Sex', 'race_parse':'Race / Ethnicity', 'cdcc':'Charlson/Dayo Score', 'facility':'Facility Type', 'year':'Diagnosis Year', 
              'grade':'Grade', 'histology':'Histologic Subtype', 't_stage':'T Stage', 'n_stage':'N Stage', 'stage':'Stage Group', 
            'her2ratio':'HER2/CEP17 Ratio', 'her2copies':'HER2 Copies', 'er':'ER status', 'pr':'PR status', 'er_num':'ER (% Positive)', 'pr_num':'PR (% Positive)', 'ihc':'HER2 IHC status', 'ki67_num':'Ki67 (% Positive)', 
              'recurrence_score_group':'OncotypeDx Score', 'immuno':'Immunotherapy', 'chemo':'Chemotherapy', 'hormone':'Hormonal Therapy', 'pcr':'Pathologic Complete Response', 'last_contact':'Last Contact Months', 'alive':'Vital Status'}
    categorical = [ 'sex', 'race_parse', 'facility', 'cdcc', 'grade', 'histology', 't_stage', 'n_stage', 'stage', 'er', 'pr', 'ihc', 'recurrence_score_group', 'immuno', 'chemo', 'hormone', 'pcr', 'alive']
    for c in ['sex', 'race_parse', 'facility', 'cdcc', 'histology', 'regional_nodes_positive', 'ihc', 'odx', 'chemo', 'hormone']:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    df['sex'] = df.sex.map({1:'Male', 2:'Female'})
    df['race_parse'] = df.race_parse.map({0:'Non-Hispanic White', 1:'Non-Hispanic Black', 2:'Hispanic', 3:'Other', 4:'Native American', 5:'Asian'})
    df['facility'] = df.facility.map({1:'Community Cancer Program', 2:'Comprehensive Comunity Cancer Program', 3:'Academic / Research', 4:'Integrated Network Cancer Program'})
    df['cdcc'] = df.cdcc.map({0: '0', 1: '>= 1', 2: '>= 1', 3: '>= 1'})
    df['histology'] = df.histology.map({0: 'Ductal', 1: 'Lobular', 2: 'Others', 
                                        3: 'Others', 4: 'Others', 5: 'Others', 6: 'Others',
                                        7: 'Others', 8: 'Others', 9: 'Others', 10: 'Others',
                                        11: 'Others'})

    df['ihc'] = df.ihc.map({0:'0',1:'1+',2:'2+',3:'3+'})
    df['alive'] = df.alive.map({0: 'Alive', 1: 'Deceased'})
    df['recurrence_score_group'] =  pd.NA
    df.loc[df.odx >= 0, 'recurrence_score_group'] = "Low (0 - 10)"
    df.loc[df.odx >= 11, 'recurrence_score_group'] = "Intermediate (11 - 25)"
    df.loc[df.odx >= 26, 'recurrence_score_group'] = "High (26+)"
    df['immuno'] = df.immuno.map({0: 'No Immuno', 1: 'Immunotherapy'}) 
    df['chemo'] = df.chemo.map({0: 'No Chemo', 1: 'Chemotherapy'})
    df['hormone'] = df.hormone.map({0: 'No Hormonal Therapy', 1: 'Hormonal Therapy'})

    if lite:
        mytable = TableOne(df, columns=columns, categorical=categorical, groupby='cat4', pval=pval, labels = labels)
    else:
        mytable = TableOne(df, columns=columns, categorical=categorical, groupby='cat', pval=pval, labels = labels)
    print(mytable.tabulate(tablefmt="github"))

    if csv_filename is not None:
        mytable.to_csv(csv_filename)



def bar_plot(cats, lite = False):
    rates = {}
    if lite:
        cat_values = ['p','n','pn','np']
        key_mapping = {
            'p':'HER2+',
            'n':'HER2-',
            'pn':'IHC-/ISH+',
            'np':'IHC+/ISH-'
        }
        col = 'cat4'
    else:
        cat_values = ['AA','A2','A3','A4','B','C','D','E']
        key_mapping = {
        'AA': 'HER2+',
        'A2': 'Group 2/IHC+',
        'A3': 'Group 3/IHC+',
        'A4': 'Group 4/IHC+',
        'B': 'ISH+/IHC-',
        'C': 'HER2-',
        'D': 'ISH-/IHC+',
        'E': 'Group 2,3,4/IHC-',
        }
        col = 'cat'

    for cat_value in cat_values:
        subset = cats[cats[col] == cat_value]
        if len(subset) == 0:
            rate = 0
        else:
            rate = sum(subset['pcr'] == 1) / len(subset)
        rates[cat_value] = rate

    standard_errors = {}
    for cat_value in cat_values:
        subset = cats[cats[col] == cat_value]
        n = len(subset)
        if n == 0:
            standard_error = 0
        else:
            p = rates[cat_value]
            standard_error = math.sqrt((p * (1 - p)) / n)
        standard_errors[cat_value] = standard_error
    rates = {key_mapping[key]: value for key, value in rates.items()}
    standard_errors = {key_mapping[key]: value for key, value in standard_errors.items()}
    return rates, standard_errors


blue_family = ['#2B5B84','#1f77b4', '#357abD', '#60a3d9', '#add8e6']
orange_family = [ '#ff9f40', '#ff7f0e', '#ff5500']
grey_family = ['#7f7f7f']
custom_colors = blue_family + orange_family + grey_family
custom_colors4 = ['#2B5B84','#357abD','#ff9f40', '#ff5500']

def plot_bars(rates, standard_errors, custom_colors, order, diagonal = False):
    fig, ax = plt.subplots()

    ordered_rates = {key: rates[key] for key in order}
    ordered_errors = {key: standard_errors[key] for key in order}
    ordered_colors = [custom_colors[order.index(key)] for key in order]

    bars = ax.bar(ordered_rates.keys(), ordered_rates.values(), color=ordered_colors)

    errors = list(ordered_errors.values())
    for i, bar in enumerate(bars):
        height = bar.get_height()
        x = bar.get_x() + bar.get_width() / 2.0
        error = errors[i]
        
        ax.errorbar(x, height, yerr=error, color='black', capsize=5)
        
        ax.hlines(y=height + error, xmin=x-0.1, xmax=x+0.1, color='black')
        ax.hlines(y=height - error, xmin=x-0.1, xmax=x+0.1, color='black')

    plt.ylabel('PCR')
    plt.ylim(0, 1)
    if diagonal:
        ax.set_xticklabels(order, rotation=45, ha="right", rotation_mode="anchor")
    plt.show()


def chi_squared_p_value(df1, df2):
    """Compute the p-value using Chi-squared test for two dataframes."""
    df1_count = df1['pcr'].value_counts()
    df2_count = df2['pcr'].value_counts()
    contingency_table = pd.DataFrame({'df1': df1_count, 'df2': df2_count}).fillna(0)
    _, p_value, _, _ = chi2_contingency(contingency_table)
    return p_value


def generate_p_value_table(dataframes):
    """Generate a table of p-values for pairwise comparison of dataframes."""
    df_names = list(dataframes.keys())
    p_value_table = pd.DataFrame(index=df_names, columns=df_names)
    
    for i, df1_name in enumerate(df_names):
        for j, df2_name in enumerate(df_names):
            p_value_table.loc[df1_name, df2_name] = chi_squared_p_value(dataframes[df1_name], dataframes[df2_name])
    
    return p_value_table


def significance_marker(p_value):
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return ''
    
def stringBeta(log_reg, v, rnd_fig = 2):
    return str(round(np.exp(log_reg.params[v]), rnd_fig)) + " (" + str(round(np.exp(log_reg.conf_int()[0][v]), rnd_fig)) + " - " + str(round(np.exp(log_reg.conf_int()[1][v]), rnd_fig)) + ")," + roundp(log_reg.pvalues[v]) + ","

def roundp(p, peq = False):
    """Return a formatted string for p values
    
    Parameters
    ----------
    p - p value float
    peq - True if need to include 'p' in the resulting string

    Returns
    -------
    Formatted p value string

    """
    if p < 0.01:
        if p < 0.001:
            if peq:
                return "p < 0.001"
            else:
                return "< 0.001"
        else:
            if peq:
                return "p = " + str(round(p,3))
            else:
                return str(round(p,3))            
    else:
        if peq:
            return "p = " + str(round(p,2))
        else:
            return str(round(p,2))

def logistic_regression(df_subset1, cols, all_histologies = True, her2_categories = False, missing_indicator = False, multiple_impute = True, receptors = None, csv_filename=None):
    df_subset2 = df_subset1.copy()
    
    df_subset, test_cols, label_cols, label_names = parse_columns(df_subset2, cols, ignore = ['pCR'], include_stage4 = False, all_histologies = all_histologies, her2_categories = her2_categories, missing_indicator = missing_indicator, multiple_impute = multiple_impute, receptors = receptors)
    
    if multiple_impute:
        df_subset['her2low'] = 0
        df_subset.loc[df_subset.her2.astype(float) > 0, 'her2low'] = 1
        df_subset['her2_1'] = 0
        df_subset['her2_2'] = 0
        df_subset.loc[df_subset.her2.astype(float) == 1, 'her2_1'] = 1        
        df_subset.loc[df_subset.her2.astype(float) == 2, 'her2_2'] = 1
        if receptors is not None:
            if receptors == 'tnbc':
                df_subset = df_subset[df.receptors == 0]
            else:
                df_subset = df_subset[df.receptors > 0]
        df_subset = df_subset[(df.neoadj_chemo == 1) & (df.stage < 4)]
        df_subset = df_subset.dropna(subset = ['pcr']).reset_index()
    
    df_subset['const'] = 1
    test_cols += ['const']

    Xtrain = df_subset[test_cols]
    ytrain = df_subset[['pcr']]
    log_reg = sm.Logit(ytrain, Xtrain).fit(disp = 0)
    print(log_reg.summary())
    '''
    for l, n in zip(label_cols, label_names):
        if len(n) == 1:
            print(n[0] + "," + stringBeta(log_reg, l[0]))
        if len(n) > 1:
            print(n[0] + ",1.00 (ref.),-")
            for l_sub, n_sub in zip(l, n[1:]):
                print(n_sub + "," + stringBeta(log_reg, l_sub))
    '''
    if csv_filename:
        summary_table = log_reg.summary().tables[1]
        summary_df = pd.DataFrame(summary_table.data)
        summary_df.columns = summary_df.iloc[0]
        summary_df = summary_df.drop(0)
        summary_df['coef'] = pd.to_numeric(summary_df['coef'])
        summary_df['odds_ratio'] = np.exp(summary_df['coef'])
        summary_df['[0.025'] = pd.to_numeric(summary_df['[0.025'], errors='coerce')
        summary_df['0.975]'] = pd.to_numeric(summary_df['0.975]'], errors='coerce')
        summary_df['odds_ratio_confidence_lower'] = np.exp(summary_df['[0.025'])
        summary_df['odds_ratio_confidence_upper'] = np.exp(summary_df['0.975]'])
        cols = summary_df.columns.tolist()
        cols.insert(cols.index('coef') + 1, cols.pop(cols.index('odds_ratio')))
        summary_df = summary_df[cols]
        summary_df.to_csv(csv_filename, index=False)


def stringHR(cph, value, rnd_fig = 2):
    cis = cph.confidence_intervals_[cph.confidence_intervals_.index == value].values.tolist()[0]
    return str(round(cph.hazard_ratios_[value],rnd_fig)) + " (" + str(round(math.exp(cis[0]),rnd_fig)) + " - " + str(round(math.exp(cis[1]),rnd_fig)) + ")," + roundp(cph.summary.p[value]) + ","


def cphModelMultivariate(df_subset, cols, all_histologies = True, her2_categories = False, stage = None, receptors = None, missing_indicator = False, multiple_impute = True, csv_filename=None):
    df_subset, test_cols, label_cols, label_names = parse_columns(df_subset, cols, all_histologies = all_histologies, her2_categories = her2_categories, missing_indicator = False, multiple_impute = multiple_impute, receptors = receptors)
    df_subset = df_subset.copy().dropna(subset=['alive', 'last_contact'])
    '''
    df_subset['her2low'] = 0
    df_subset.loc[df_subset.her2.astype(float) > 0, 'her2low'] = 1
    df_subset['her2_1'] = 0
    df_subset['her2_2'] = 0
    df_subset.loc[df_subset.her2.astype(float) == 1, 'her2_1'] = 1        
    df_subset.loc[df_subset.her2.astype(float) == 2, 'her2_2'] = 1
    '''
    if stage is not None:
        df_subset = df_subset[df_subset.stage == stage]
    if receptors is not None:
        if receptors == 'tnbc':
            df_subset = df_subset[df_subset.receptors.astype(float) == 0]
        else:
            df_subset = df_subset[df_subset.receptors.astype(float) > 0]


    print('n = ' + str(len(df_subset.index)))
    test_cols += ['alive', 'last_contact']

    cph = CoxPHFitter(penalizer=0.1)
    df_cph = df_subset[test_cols]
    cph.fit(df_cph, "last_contact", event_col = "alive")
    cph.print_summary()
    '''
    for l, n in zip(label_cols, label_names):
        if len(n) == 1:
            print(n[0] + "," + stringHR(cph, l[0]))
        if len(n) > 1:
            print(n[0] + ",-,-")
            for l_sub, n_sub in zip(l, n[1:]):
                print(n_sub + "," + stringHR(cph, l_sub)) 
    '''
    if csv_filename:
        summary_df = cph.summary
        summary_df.to_csv(csv_filename)



def impute_values(df, columns):
    for col in columns:
        if df[col].isna().sum() > 0:
            if df[col].dtype == 'object' or df[col].dtype == 'bool':
                mode_value = df[col].mode()[0]
                df[col].fillna(mode_value, inplace=True)
            else:
                median_value = df[col].median()
                df[col].fillna(median_value, inplace=True)
    return df        


def plot_survival_stage(df, ax, name, legend=False, col='cat4', color_dict=None, legend_map=None):
    
    kmfs = []
    for s in df[col].unique().tolist():
        df_subset = df[df[col].isin([s])]

        df_subset = df_subset[['last_contact', 'alive']]

        df_subset = df_subset.dropna()
    
        kmf = KaplanMeierFitter()

        plot_kwargs = {}
        if color_dict and s in color_dict:
            plot_kwargs['color'] = color_dict[s]
        
        label = legend_map[s] if legend_map and s in legend_map else s
        kmf.fit(df_subset['last_contact'], df_subset['alive'], label=label)
        kmf.plot_survival_function(ax=ax, ci_show=False, **plot_kwargs)

        kmfs += [kmf]
        
    add_at_risk_counts(*kmfs, ax=ax, xticks=[0, 20, 40, 60, 80, 100])

    ax.set_xlim([0, 100])
    ax.set_title(name)

    if legend:
        ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.66))
    else:
        ax.get_legend().set_visible(False)
    ax.set_xlabel("Months")
    ax.set_ylabel("Overall Survival")