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
warnings.filterwarnings('ignore')

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)


    
def load_data_her2(filename=None, savefile=None, loadfile=None, lower=True):
    """Load data from NCDB and print # excluded at each step

     Parameters
     ----------
     filename - name of csv NCDB PUF file
     savefile - Provide a string if you want to save the resultant dataframe as CSV (saves as 'saved_' + savefile) 
     loadfile - Provide a string if you want to load a cleaned NCDB dataframe (from 'saved_' + loadfile)
     lower - True if the column headers need to be lowercase (NCDB 2017 switched from uppercase to lowercase)

     Returns
     -------
     dataframe with loaded NCDB dataset
     """
    if loadfile:
        return pd.read_csv(loadfile, dtype=str)
    
    #This next line specifies the columns from NCDB to load. These need to be in the same order as they are listed in NCDB
    fields = ['facility_type_cd', 'AGE', 'sex', 'RACE', 'spanish_hispanic_origin', #0-4
              'insurance_status', 'cdcc_total_best', 'year_of_diagnosis', 'histology', 'behavior', #5-9
              'GRADE', 'TUMOR_SIZE', 'REGIONAL_NODES_POSITIVE', 'TNM_CLIN_T', 'TNM_CLIN_N', #10-14
              'TNM_CLIN_M', 'tnm_clin_stage_group', 'TNM_PATH_T', 'TNM_PATH_N', 'TNM_PATH_M', #15-19
              'tnm_path_stage_group', 'analytic_stage_group', 'cs_mets_dx_bone', 'cs_mets_dx_brain', 'cs_mets_dx_liver', #20-24
              'cs_mets_dx_lung', 'lymph_vascular_invasion', 'CS_SITESPECIFIC_FACTOR_8','CS_SITESPECIFIC_FACTOR_9', 'CS_SITESPECIFIC_FACTOR_10', #25-29 
              'CS_SITESPECIFIC_FACTOR_11','CS_SITESPECIFIC_FACTOR_12', 'CS_SITESPECIFIC_FACTOR_13', 'CS_SITESPECIFIC_FACTOR_15', 'CS_SITESPECIFIC_FACTOR_16', #30-34
              'CS_SITESPECIFIC_FACTOR_21', 'CS_SITESPECIFIC_FACTOR_22', 'CS_SITESPECIFIC_FACTOR_23', 'dx_defsurg_started_days', 'dx_chemo_started_days', #35-39
              'RX_SUMM_CHEMO', 'dx_hormone_started_days', 'RX_SUMM_HORMONE', 'dx_immuno_started_days', 'RX_SUMM_IMMUNOTHERAPY', #40-44
              'DX_LASTCONTACT_DEATH_MONTHS', 'PUF_VITAL_STATUS', 'tumor_size_summary_2016', 'mets_at_dx_bone', 'mets_at_dx_brain', #45-49
              'mets_at_dx_liver', 'mets_at_dx_lung',  'ajcc_tnm_clin_n', 'ajcc_tnm_clin_stg_grp', 'ajcc_tnm_clin_t', #50-54
              'ajcc_tnm_path_n', 'ajcc_tnm_path_stg_grp', 'ajcc_tnm_path_t', 'ajcc_tnm_post_path_n', 'ajcc_tnm_post_path_t', #55-59
              'er_percent_pos_or_rng', 'er_summary',  'grade_clin', 'grade_path',  'her2_ihc_summary', #60-64
              'her2_ish_dual_num', 'her2_ish_dual_ratio', 'her2_ish_single_num', 'her2_ish_summary','her2_overall_summ', #65-69
              'ki67', 'oncotype_risk_invas', 'oncotype_score_inv', 'pr_percent_pos_or_rng', 'pr_summary'] #70-74
    
    #Since the column names are confusing, we provide a list of shorter names to be used
    fieldname = ['facility', 'age', 'sex', 'race', 'hispanic', #0-4
                 'insurance', 'cdcc', 'year', 'histology', 'behavior', #5-9
                 'grade', 'tumor_size', 'regional_nodes_positive', 'tnm_clin_t', 'tnm_clin_n', #10-14
                 'tnm_clin_m', 'tnm_clin_stage_group', 'tnm_path_t', 'tnm_path_n', 'tnm_path_m', #15-19
                 'tnm_path_stage_group', 'stage', 'mets_bone', 'mets_brain', 'mets_liver', #20-24
                 'mets_lung', 'lvi', 'her2', 'her2ihc', 'her2ratio', #25-29
                 'her2ratio_summ', 'her2copies', 'her2copies_summ', 'her2sum', 'receptors', #30-34
                 'neoadj_response', 'recurrence_assay', 'recurrence_score', 'surg_days', 'chemo_days', #35-39
                 'rx_summ_chemo', 'hormone_days', 'rx_summ_hormone', 'immuno_days', 'rx_summ_immunotherapy', #40-44
                 'last_contact', 'alive',  'tumor_size_summary_2016', 'mets_at_dx_bone', 'mets_at_dx_brain', #45-49
                 'mets_at_dx_liver', 'mets_at_dx_lung', 'ajcc_tnm_clin_n', 'ajcc_tnm_clin_stg_grp', 'ajcc_tnm_clin_t', #50-54
                 'ajcc_tnm_path_n', 'ajcc_tnm_path_stg_grp', 'ajcc_tnm_path_t', 'ajcc_tnm_post_path_n', 'ajcc_tnm_post_path_t', #55-59
                 'er_percent', 'er_summary', 'grade_clin', 'grade_path', 'her2_ich_summary', #60-64
                 'her2_ish_dual_num', 'her2_ish_dual_ratio', 'her2_ish_single_num', 'her2_ish_summary', 'her2_overall_summ',  #65-69
                 'ki67', 'oncotype_risk_invas', 'oncotype_score_inv', 'pr_percent', 'pr_summary'] #70-74

    #This converts the words in the 'fields' variable to lowercase
    if lower:
        fields = [f.lower() for f in fields]
    
    #This actually reads the CSV file
    df = pd.read_csv(filename, usecols=fields, dtype=str)
    
    #We set the column names within the CSV file to the simpler names
    df.columns = fieldname

    print("Total Patients in NCDB: " + str(len(df.index)))
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df = df[df.year >= 2013]
    print("Excluding diagnoses before 2013: " + str(len(df.index)))

    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df.loc[df.age > 90, 'age'] = np.nan
    df['sex'] = pd.to_numeric(df['sex'], errors='coerce')
    df.loc[~df.sex.isin([1,2]), 'sex'] = np.nan
    
    df['race'] = pd.to_numeric(df['race'], errors='coerce')
    df['hispanic'] = pd.to_numeric(df['hispanic'], errors='coerce')
  
    df['race_parse'] = 0 #non-hispanic white
    df.loc[((df.race == 2) & (df.hispanic == 0)), 'race_parse'] = 1 #non-hispanic black
    df.loc[(df.hispanic > 0) & (df.hispanic < 9), 'race_parse'] = 2  # hispanic
    df.loc[((df.race == 98) & (df.hispanic == 0)), 'race_parse'] = 3 #other
    df.loc[((df.race == 3) & (df.hispanic == 0)), 'race_parse'] = 4 #Native american
    df.loc[((~df.race.isin([0,1,2,3,98,99])) & (df.hispanic == 0)), 'race_parse'] = 5 #asian / pacific islander
    df.loc[(~df.race.isin([0,1,2,3,98,99, np.nan])) & (df.hispanic == 0), 'race_parse'] = 5 #asian / pacific islander
    df.loc[df.race == 99, 'race_parse'] = np.nan #unknown race (we will ignore unknown ethnicity)
    
    df['histology'] = pd.to_numeric(df['histology'], errors='coerce')  
    df.loc[df.histology.isin([8500, 8501, 8502, 8503, 8504, 8505, 8506, 8507, 8508, 8523, 8230]), 'histology'] = 0 #ductal
    df.loc[df.histology.isin([8520, 8521, 8524, 8525]), 'histology'] = 1 #lobular
    df.loc[df.histology.isin([8522]), 'histology'] = 2 # mixed
    df.loc[df.histology.isin([8480, 8481]), 'histology'] = 3 # mucinous
    df.loc[df.histology.isin([8503, 8504, 8260, 8050, 8051, 8052]), 'histology'] = 4 # papillary
    df.loc[df.histology.isin([8211, 8210]), 'histology'] = 5 # tubular
    df.loc[df.histology.isin([8530]), 'histology'] = 6 # inflammatory
    df.loc[df.histology.isin([8510, 8512, 8513, 8514]), 'histology'] = 7 # medullary
    df.loc[df.histology.isin([8570, 8571, 8572, 8573, 8574, 8575, 8576]), 'histology'] = 8 # metaplsatic
    df.loc[df.histology.isin([8540, 8541, 8542, 8543]), 'histology'] = 9 # paget
    df.loc[(df.histology >= 8800) & (df.histology <= 9581), 'histology'] = 10 # sarcoma, etc    
    df.loc[~df.histology.isin([0,1,2,3,4,5,6,7,8,9,10,np.nan]), 'histology'] = 11 #other

    df = df[df.behavior.astype(float) == 3]
    print("Excluding non-invasive cases: " + str(len(df.index)))
    
    df['grade'] = pd.to_numeric(df['grade'], errors='coerce')  
    df.loc[df.grade == 4, 'grade'] = 3
    df.loc[~df.grade.astype(float).isin([1,2,3]), 'grade'] = np.nan
    
        
    #fix for later year of diagnosis:
    df.loc[(df.year.astype(float) >= 2018) & (df.grade_clin.isin(['H','3','C','D'])), 'grade'] = 3
    df.loc[(df.year.astype(float) >= 2018) & (df.grade_clin.isin(['M','2','B'])), 'grade'] = 2
    df.loc[(df.year.astype(float) >= 2018) & (df.grade_clin.isin(['L','1','A'])), 'grade'] = 1
    df.loc[(df.year.astype(float) >= 2018) & (df.grade_path.isin(['H','3','C','D'])), 'grade'] = 3
    df.loc[(df.year.astype(float) >= 2018) & (df.grade_path.isin(['M','2','B'])), 'grade'] = 2
    df.loc[(df.year.astype(float) >= 2018) & (df.grade_path.isin(['L','1','A'])), 'grade'] = 1
    
    #Tumor size likely obsolete
    df.loc[df.year.astype(float) >= 2018, 'tumor_size'] = df['tumor_size_summary_2016']
    df['tumor_size'] = pd.to_numeric(df['tumor_size'], errors='coerce')  
    df.loc[df.tumor_size.astype(float) == 990, 'tumor_size'] = 1
    df.loc[df.tumor_size.astype(float) == 991, 'tumor_size'] = 5 # < 1 cm
    df.loc[df.tumor_size.astype(float) == 992, 'tumor_size'] = 15 # 1-2 cm
    df.loc[df.tumor_size.astype(float) == 993, 'tumor_size'] = 25 # 2-3 cm
    df.loc[df.tumor_size.astype(float) == 994, 'tumor_size'] = 35 # 3-4 cm
    df.loc[df.tumor_size.astype(float) == 995, 'tumor_size'] = 45 # 4-5 cm
    df.loc[df.tumor_size.astype(float) > 995, 'tumor_size'] = np.nan
    
    
    #probably need to just use TNM stage
    df['regional_nodes_positive'] = pd.to_numeric(df['regional_nodes_positive'], errors='coerce')  
    df.loc[df.regional_nodes_positive.astype(float).isin([98,99]), 'regional_nodes_positive'] = np.nan
    
    df['pcr'] = 0
    df.loc[((df.tnm_path_n.str.contains("0") & (~df.tnm_path_n.str.contains("+", na = True, regex=False))) & ((df.tnm_path_t.str.contains("0")) | (df.tnm_path_t.str.contains("IS")))), 'pcr'] = 1
    df.loc[((df.neoadj_response.astype(float) > 30) | (df.neoadj_response.astype(float) == 25) | (df.neoadj_response.isna())) & (df.tnm_path_n.str.contains("88") | df.tnm_path_n.str.contains("X") | df.tnm_path_t.str.contains("88") | df.tnm_path_t.str.contains("X") | (df.tnm_path_t == "") | (df.tnm_path_n == "") | (df.tnm_path_t.isna()) | (df.tnm_path_n.isna())), 'pcr'] = np.nan
    df.loc[df.pcr.isna() & (df.neoadj_response.astype(float).isin([10]))] = 1
    df.loc[(df.year.astype(float) >= 2018), 'pcr'] = 0
    df.loc[(df.year.astype(float) >= 2018) & ((df.ajcc_tnm_post_path_n.str.contains('N0') & (~df.ajcc_tnm_post_path_n.str.contains('+',  na = True, regex=False))) & (df.ajcc_tnm_post_path_t.str.contains('T0') | df.ajcc_tnm_post_path_t.str.contains('Tis'))), 'pcr'] = 1
    df.loc[(df.year.astype(float) >= 2018) & (df.ajcc_tnm_post_path_n.str.contains('NX') | df.ajcc_tnm_post_path_t.str.contains('TX') | (df.ajcc_tnm_post_path_n == '') | (df.ajcc_tnm_post_path_t == '') | df.ajcc_tnm_post_path_n.isna() | df.ajcc_tnm_post_path_t.isna()), 'pcr'] = np.nan

    df.loc[(df.year.astype(float) >= 2018), 'mets_bone'] = df['mets_at_dx_bone']
    df.loc[(df.year.astype(float) >= 2018), 'mets_brain'] = df['mets_at_dx_brain']
    df.loc[(df.year.astype(float) >= 2018), 'mets_liver'] = df['mets_at_dx_liver']
    df.loc[(df.year.astype(float) >= 2018), 'mets_lung'] = df['mets_at_dx_lung']
    
    df['mets_bone'] = pd.to_numeric(df['mets_bone'], errors='coerce')  
    df['mets_brain'] = pd.to_numeric(df['mets_brain'], errors='coerce')  
    df['mets_liver'] = pd.to_numeric(df['mets_liver'], errors='coerce')  
    df['mets_lung'] = pd.to_numeric(df['mets_lung'], errors='coerce')  
    
    df.loc[(df.mets_bone.astype(float) > 2), 'mets_bone'] = np.nan
    df.loc[(df.mets_brain.astype(float) > 2), 'mets_brain'] = np.nan
    df.loc[(df.mets_liver.astype(float) > 2), 'mets_liver'] = np.nan
    df.loc[(df.mets_lung.astype(float) > 2), 'mets_lung'] = np.nan

    '''
    # Now the most important part - we can select which patients we want to look at!
    # First, lets only use patients who are characterized as HER2 positive by their testing:
    # The CS_SITESPECIFIC_FACTOR_15 column corresponds to the summary of her2 testing, which I renamed her2sum
    # If you review https://web2.facs.org/cstage0205/breast/Breastschema.html - you can see that HER2 positive is coded as 20
    
    # The following code selects all rows from our 'dataframe' where the her2sum variable is in the provided list of [20]
    # We say .astype(float) because some of the data from NCDB is loaded as sequences of characters or 'strings' by default
    # A float is a floating point number, so it allows us to comare the values to other numbers
    # Notice that the 20 in brackets [20] doesn't have quotations around it, this indicates it is a number and not a string
    # You can access columns in dataframes in two ways: df['column'] or df.column
    df = df[df.her2sum.astype(float).isin([20]) | df.her2_overall_summ.astype(float).isin([0])]
    print("Excluding cases classified as HER2+: " + str(len(df.index)))
    '''

    
    df['receptors'] = pd.to_numeric(df['receptors'], errors='coerce')
    df.loc[df.receptors.astype(float) > 111, 'receptors'] = np.nan   
    df.pr_summary = pd.to_numeric(df.pr_summary, errors = 'coerce')
    df.er_summary = pd.to_numeric(df.er_summary, errors = 'coerce')
    df.loc[(df.pr_summary.isin([0, 1])) & (df.er_summary.isin([0, 1])) & (df.her2_overall_summ.astype(float).isin([0,1])), 'receptors'] = df.er_summary.astype(float) * 100 + df.pr_summary.astype(float) * 10 + df.her2_overall_summ.astype(float)
    #df = df[df.receptors.astype(float).isin([0,10,100,110,np.nan])]

    df['er_num'] = pd.to_numeric(df['er_percent'], errors='coerce')
    df.loc[df.er_percent == 'R10', 'er_num'] = 5
    df.loc[df.er_percent == 'R20', 'er_num'] = 15
    df.loc[df.er_percent == 'R30', 'er_num'] = 25
    df.loc[df.er_percent == 'R40', 'er_num'] = 35
    df.loc[df.er_percent == 'R50', 'er_num'] = 45
    df.loc[df.er_percent == 'R60', 'er_num'] = 55
    df.loc[df.er_percent == 'R70', 'er_num'] = 65
    df.loc[df.er_percent == 'R80', 'er_num'] = 75
    df.loc[df.er_percent == 'R90', 'er_num'] = 85
    df.loc[df.er_percent == 'R99', 'er_num'] = 95

    df['pr_num'] = pd.to_numeric(df['pr_percent'], errors='coerce')
    df.loc[df.pr_percent == 'R10', 'pr_num'] = 5
    df.loc[df.pr_percent == 'R20', 'pr_num'] = 15
    df.loc[df.pr_percent == 'R30', 'pr_num'] = 25
    df.loc[df.pr_percent == 'R40', 'pr_num'] = 35
    df.loc[df.pr_percent == 'R50', 'pr_num'] = 45
    df.loc[df.pr_percent == 'R60', 'pr_num'] = 55
    df.loc[df.pr_percent == 'R70', 'pr_num'] = 65
    df.loc[df.pr_percent == 'R80', 'pr_num'] = 75
    df.loc[df.pr_percent == 'R90', 'pr_num'] = 85
    df.loc[df.pr_percent == 'R99', 'pr_num'] = 95
    
    df['ki67_num'] = pd.to_numeric(df['ki67'], errors='coerce')


    # Next, we want HER2 IHC to be 0+ to 2+ (3+ is positive). Technically this should be 0, 10, or 20, corresponding to
    # 0+, 1+, and 2+, if we again review the site specific factors (this time, site specific factor 8).
    # However, 1 and 2 used to signify 1+ and 2+, so I included these to catch a few patients who were miscoded

    df.loc[df.her2_ich_summary.astype(float).isin([0]), 'her2'] = 0
    df.loc[df.her2_ich_summary.astype(float).isin([1]), 'her2'] = 1
    df.loc[df.her2_ich_summary.astype(float).isin([2]), 'her2'] = 2
    df.loc[df.her2_ich_summary.astype(float).isin([3]), 'her2'] = 3

    df.loc[df.her2.astype(float) == 10, 'her2'] = 1
    df.loc[df.her2.astype(float) == 20, 'her2'] = 2
    df.loc[df.her2.astype(float) == 30, 'her2'] = 3
    #df = df[(df.her2.astype(float).isin([0, 1, 2]))]

    #print("Excluding cases where HER2 IHC isn't 0-2: " + str(len(df.index)))

    df['her2ratio'] = pd.to_numeric(df['her2ratio'], errors='coerce') 
    df.loc[df.her2ratio.astype(float).isin([981,982,983,984,985,986,987]), 'her2ratio'] = 980
    df.loc[df.her2ratio.astype(float).isin([991]), 'her2ratio'] = 100
    df.loc[df.her2ratio.astype(float).isin([988, 989, 990, 992, 993, 994, 995, 996, 997, 998, 999]), 'her2ratio'] = np.nan
    df.loc[df.her2ratio.astype(float) < 100, 'her2ratio'] = np.nan
    df['her2ratio'] = df['her2ratio'].astype(float)/100
    

    #her2ratio_binary takes the value 1 if one of her2ratio_summ, her2_ish_dual_ratio, her2_ish_summary is coded as positive, and 0 otherwise
    #her2ish_binary takes the value 1 if one of her2copies_summ, her2_ish_summary is coded as positive, and 0 otherwise

    df['her2ratio_binary'] = np.nan
    df.loc[df['her2_ish_dual_ratio']=='XX.2', 'her2ratio_binary'] = 0
    df.loc[df['her2_ish_dual_ratio']=='XX.3', 'her2ratio_binary'] = 1

    df['her2ish_binary'] = np.nan

    df['her2ratio_summ'] = pd.to_numeric(df['her2ratio_summ'], errors='coerce')
    df.loc[df['her2ratio_summ']==10, 'her2ratio_binary'] = 1
    df.loc[df['her2ratio_summ'] == 20, 'her2ratio_binary'] = 0

    df['her2copies_summ'] = pd.to_numeric(df['her2copies_summ'], errors='coerce')
    df.loc[df['her2copies_summ']== 10, 'her2ish_binary'] = 1
    df.loc[df['her2copies_summ'] == 20, 'her2ish_binary'] = 0

    df['her2_ish_summary'] = pd.to_numeric(df['her2_ish_summary'], errors='coerce')
    df.loc[df['her2_ish_summary'] == 3, 'her2ish_binary'] = 1
    df.loc[df['her2_ish_summary'] == 3, 'her2ratio_binary'] = 1
    df.loc[df['her2_ish_summary'] == 0, 'her2ish_binary'] = 0
    df.loc[df['her2_ish_summary'] == 0, 'her2ratio_binary'] = 0


    df['her2_ish_dual_ratio'] = pd.to_numeric(df['her2_ish_dual_ratio'], errors='coerce')
    df.loc[~df.her2_ish_dual_ratio.isna(), 'her2ratio'] = df['her2_ish_dual_ratio']
    
    df['her2copies'] = pd.to_numeric(df['her2copies'], errors='coerce')
    df.loc[df.her2copies.astype(float).isin([981,982,983,984,985,986,987]), 'her2copies'] = 980
    df.loc[df.her2copies.astype(float).isin([991]), 'her2copies'] = 100
    df.loc[df.her2copies.astype(float).isin([988, 989, 990, 992, 993, 994, 995, 996, 997, 998, 999]), 'her2copies'] = np.nan
    df.loc[df.her2copies.astype(float) < 100, 'her2copies'] = np.nan
    df['her2copies'] = df['her2copies'].astype(float)/100

    df.loc[df['her2_ish_dual_num'] == 'XX.1', 'her2_ish_dual_num'] = 100
    df.loc[df['her2_ish_single_num'] == 'XX.1', 'her2_ish_single_num'] = 100

    df['her2_ish_dual_num'] = pd.to_numeric(df['her2_ish_dual_num'], errors='coerce')
    df.loc[~df.her2_ish_dual_num.isna(), 'her2copies'] = df['her2_ish_dual_num']
    df['her2_ish_single_num'] = pd.to_numeric(df['her2_ish_single_num'], errors='coerce')
    df.loc[~df.her2_ish_single_num.isna(), 'her2copies'] = df['her2_ish_single_num']


    '''
    df = df[(~df.her2ratio.isna()) | (~df.her2copies.isna()) | (df.her2.astype(float) < 2)]
    print("Excluding cases where HER2 IHC is 2 and FISH is not performed: " + str(len(df.index)))
        
    df = df[(df.her2.astype(float) < 2) | (df.her2copies.astype(float) < 6) | (~df.her2ratio.isna())]
    print("Excluding cases where HER2 IHC is 2 and FISH CN >= 6: " + str(len(df.index)))

    df = df[(df.her2.astype(float) < 2) | (df.her2copies.astype(float) < 4) | (df.her2ratio.astype(float) < 2)]
    print("Excluding cases where HER2 IHC is 2 and FISH CN >= 4 and ratio >= 2: " + str(len(df.index)))
    '''


    df['odx'] = np.nan
    df.loc[(df.recurrence_assay.astype(float) == 10) & (df.recurrence_score.astype(float) < 101), 'odx'] = df.recurrence_score.astype(float)
    df['oncotype_risk_invas'] = pd.to_numeric(df.oncotype_risk_invas, errors = 'coerce')
    df['oncotype_score_inv'] = pd.to_numeric(df.oncotype_score_inv, errors = 'coerce')
    df.loc[(df.oncotype_risk_invas.astype(float).isin([0,1])), 'odx'] = df['oncotype_score_inv'].astype(float)
                                                           
    df['chemo'] = np.nan
    df.loc[df.rx_summ_chemo.astype(float).isin([1,2,3]), 'chemo'] = 1
    df.loc[df.rx_summ_chemo.astype(float).isin([0, 82, 85, 86, 87]), 'chemo'] = 0

    df['immuno'] = np.nan
    df.loc[df.rx_summ_immunotherapy.astype(float).isin([1]), 'immuno'] = 1
    df.loc[df.rx_summ_immunotherapy.astype(float).isin([0]), 'immuno'] = 0 #82, 85, 86, 87
    df.loc[df.rx_summ_immunotherapy.astype(float).isin([82, 85, 86, 87]), 'immuno'] = 2

                                                           
    df['hormone'] = np.nan
    df.loc[df.rx_summ_hormone.astype(float).isin([1]), 'hormone'] = 1
    df.loc[df.rx_summ_hormone.astype(float).isin([0, 82, 85, 86, 87]), 'hormone'] = 0

    df['surg_days'] = pd.to_numeric(df['surg_days'], errors='coerce')
    df['chemo_days'] = pd.to_numeric(df['chemo_days'], errors='coerce')
    df['hormone_days'] = pd.to_numeric(df['hormone_days'], errors='coerce')
    df['immuno_days'] = pd.to_numeric(df['immuno_days'], errors='coerce')

    df['neoadj_chemo'] = np.nan
    df.loc[df['surg_days'] > df['chemo_days'] + 30, 'neoadj_chemo'] = 1
    df.loc[df['surg_days'] <= df['chemo_days'] + 30, 'neoadj_chemo'] = 0

    df['neoadj_immuno'] = np.nan
    df.loc[df['surg_days'] > df['immuno_days'] + 30, 'neoadj_immuno'] = 1
    df.loc[df['surg_days'] <= df['immuno_days'] + 30, 'neoadj_immuno'] = 0

    df.loc[df.neoadj_response.astype(float).isin([10,20,25,30])& df['chemo_days'].isna(), 'neoadj_chemo'] = 1
    df.loc[df.neoadj_response.astype(float).isin([987, 998])& df['chemo_days'].isna(), 'neoadj_chemo'] = 0
    df.loc[(df.ajcc_tnm_post_path_t.str.contains('yp') | df.ajcc_tnm_post_path_n.str.contains('yp'))& df['chemo_days'].isna(), 'neoadj_chemo'] = 1
    df.loc[df.chemo == 0, 'neoadj_chemo'] = 0                                                         

    df['neoadj_endo'] = np.nan
    df.loc[df['surg_days'] > df['hormone_days'] + 30, 'neoadj_endo'] = 1
    df.loc[df['surg_days'] <= df['hormone_days'] + 30, 'neoadj_endo'] = 0
    df.loc[df.hormone == 0, 'neoadj_endo'] = 0

    df.loc[(df.neoadj_chemo == 1) & (df.neoadj_endo == 1), 'neoadj_endo'] = 0

    #df['stage'] = pd.to_numeric(df['stage'], errors='coerce')  
    #df.loc[df.stage.astype(float) > 4, 'stage'] = np.nan
    #df = df[df.stage > 0]
    
    df['clin_stage'] = np.nan
    df.loc[df.tnm_clin_stage_group.astype('string').str.contains('1'), 'clin_stage'] = 1
    df.loc[df.tnm_clin_stage_group.astype('string').str.contains('2'), 'clin_stage'] = 2
    df.loc[df.tnm_clin_stage_group.astype('string').str.contains('3'), 'clin_stage'] = 3
    df.loc[df.tnm_clin_stage_group.astype('string').str.contains('4'), 'clin_stage'] = 4
    df.loc[df.tnm_clin_stage_group.astype('string').str.contains('0'), 'clin_stage'] = 0

    df.loc[df.ajcc_tnm_clin_stg_grp.astype('string').str.contains('1'), 'clin_stage'] = 1
    df.loc[df.ajcc_tnm_clin_stg_grp.astype('string').str.contains('2'), 'clin_stage'] = 2
    df.loc[df.ajcc_tnm_clin_stg_grp.astype('string').str.contains('3'), 'clin_stage'] = 3
    df.loc[df.ajcc_tnm_clin_stg_grp.astype('string').str.contains('4'), 'clin_stage'] = 4
    df.loc[df.ajcc_tnm_clin_stg_grp.astype('string').str.contains('0'), 'clin_stage'] = 0
    
    df['path_stage'] = np.nan
    df.loc[df.tnm_path_stage_group.astype('string').str.contains('1'), 'path_stage'] = 1
    df.loc[df.tnm_path_stage_group.astype('string').str.contains('2'), 'path_stage'] = 2
    df.loc[df.tnm_path_stage_group.astype('string').str.contains('3'), 'path_stage'] = 3
    df.loc[df.tnm_path_stage_group.astype('string').str.contains('4'), 'path_stage'] = 4
    df.loc[df.tnm_path_stage_group.astype('string').str.contains('0'), 'path_stage'] = 0

    df.loc[df.ajcc_tnm_path_stg_grp.astype('string').str.contains('1'), 'path_stage'] = 1
    df.loc[df.ajcc_tnm_path_stg_grp.astype('string').str.contains('2'), 'path_stage'] = 2
    df.loc[df.ajcc_tnm_path_stg_grp.astype('string').str.contains('3'), 'path_stage'] = 3
    df.loc[df.ajcc_tnm_path_stg_grp.astype('string').str.contains('4'), 'path_stage'] = 4
    df.loc[df.ajcc_tnm_path_stg_grp.astype('string').str.contains('0'), 'path_stage'] = 0
    
    df['clin_t'] = np.nan
    df.loc[df.tnm_clin_t.astype('string').str.contains('0'), 'clin_t'] = 0
    df.loc[df.tnm_clin_t.astype('string').str.contains('1'), 'clin_t'] = 1
    df.loc[df.tnm_clin_t.astype('string').str.contains('2'), 'clin_t'] = 2
    df.loc[df.tnm_clin_t.astype('string').str.contains('3'), 'clin_t'] = 3
    df.loc[df.tnm_clin_t.astype('string').str.contains('4'), 'clin_t'] = 4
    df.loc[df.ajcc_tnm_clin_t.astype('string').str.contains('0'), 'clin_t'] = 0
    df.loc[df.ajcc_tnm_clin_t.astype('string').str.contains('1'), 'clin_t'] = 1
    df.loc[df.ajcc_tnm_clin_t.astype('string').str.contains('2'), 'clin_t'] = 2
    df.loc[df.ajcc_tnm_clin_t.astype('string').str.contains('3'), 'clin_t'] = 3
    df.loc[df.ajcc_tnm_clin_t.astype('string').str.contains('4'), 'clin_t'] = 4

    df['path_t'] = np.nan
    df.loc[df.tnm_path_t.astype('string').str.contains('0'), 'path_t'] = 0
    df.loc[df.tnm_path_t.astype('string').str.contains('1'), 'path_t'] = 1
    df.loc[df.tnm_path_t.astype('string').str.contains('2'), 'path_t'] = 2
    df.loc[df.tnm_path_t.astype('string').str.contains('3'), 'path_t'] = 3
    df.loc[df.tnm_path_t.astype('string').str.contains('4'), 'path_t'] = 4
    df.loc[df.ajcc_tnm_path_t.astype('string').str.contains('0'), 'path_t'] = 0
    df.loc[df.ajcc_tnm_path_t.astype('string').str.contains('1'), 'path_t'] = 1
    df.loc[df.ajcc_tnm_path_t.astype('string').str.contains('2'), 'path_t'] = 2
    df.loc[df.ajcc_tnm_path_t.astype('string').str.contains('3'), 'path_t'] = 3
    df.loc[df.ajcc_tnm_path_t.astype('string').str.contains('4'), 'path_t'] = 4

    df['clin_n'] = np.nan
    df.loc[df.tnm_clin_n.astype('string').str.contains('0'), 'clin_n'] = 0
    df.loc[df.tnm_clin_n.astype('string').str.contains('1'), 'clin_n'] = 1
    df.loc[df.tnm_clin_n.astype('string').str.contains('2'), 'clin_n'] = 2
    df.loc[df.tnm_clin_n.astype('string').str.contains('3'), 'clin_n'] = 3
    df.loc[df.ajcc_tnm_clin_n.astype('string').str.contains('0'), 'clin_n'] = 0
    df.loc[df.ajcc_tnm_clin_n.astype('string').str.contains('1'), 'clin_n'] = 1
    df.loc[df.ajcc_tnm_clin_n.astype('string').str.contains('2'), 'clin_n'] = 2
    df.loc[df.ajcc_tnm_clin_n.astype('string').str.contains('3'), 'clin_n'] = 3

    df['path_n'] = np.nan
    df.loc[df.tnm_path_n.astype('string').str.contains('0'), 'path_n'] = 0
    df.loc[df.tnm_path_n.astype('string').str.contains('1'), 'path_n'] = 1
    df.loc[df.tnm_path_n.astype('string').str.contains('2'), 'path_n'] = 2
    df.loc[df.tnm_path_n.astype('string').str.contains('3'), 'path_n'] = 3
    df.loc[df.ajcc_tnm_path_n.astype('string').str.contains('0'), 'path_n'] = 0
    df.loc[df.ajcc_tnm_path_n.astype('string').str.contains('1'), 'path_n'] = 1
    df.loc[df.ajcc_tnm_path_n.astype('string').str.contains('2'), 'path_n'] = 2
    df.loc[df.ajcc_tnm_path_n.astype('string').str.contains('3'), 'path_n'] = 3

    #df['stage'] = df[['stage', 'clin_stage', 'path_stage']].max(axis=1)
    
    #df['stage'] = df['path_stage']
    df.loc[df.neoadj_chemo == 1, 'stage'] = df['clin_stage']
    df.loc[df.stage.isna(), 'stage'] = df['path_stage']
    df.loc[df.stage.isna(), 'stage'] = df['clin_stage']
    
    df['t_stage'] = df['path_t']
    df.loc[df.neoadj_chemo == 1, 't_stage'] = df['clin_t']
    df['n_stage'] = df['path_n']
    df.loc[df.neoadj_chemo == 1, 'n_stage'] = df['clin_n']
    
    #Exclude stage 0 patients
    df.stage = pd.to_numeric(df.stage, errors='coerce')
    df = df[(df.stage > 0) & (df.stage <= 4)]
    print("Excluding Stage 0 cases: " + str(len(df.index)))

    
    df.loc[df.mets_bone == 1, 'stage'] = 4
    df.loc[df.mets_brain == 1, 'stage'] = 4
    df.loc[df.mets_liver == 1, 'stage'] = 4
    df.loc[df.mets_lung == 1, 'stage'] = 4

    df.loc[df.stage < 4, 'mets_bone'] = 0
    df.loc[df.stage < 4, 'mets_brain'] = 0
    df.loc[df.stage < 4, 'mets_liver'] = 0
    df.loc[df.stage < 4, 'mets_lung'] = 0
 
    df['alive'] = df['alive'].astype(float)
    df['alive'] = 1 - df['alive'] #To give death rate instead of survival


    df['her2'] = pd.to_numeric(df['her2'], errors='raise')
    df['facility'] = pd.to_numeric(df['facility'], errors='raise')
    df['cdcc'] = pd.to_numeric(df['cdcc'], errors='raise')
    df['last_contact'] = pd.to_numeric(df['last_contact'], errors='raise')

    '''
    conditions1 = [
    ((df['her2ratio'] >= 2)|(df['her2ratio_binary']==1)&(df['her2ratio'].isna())) & (df['her2copies'] >= 4), #1
    ((df['her2ratio'] >= 2)|(df['her2ratio_binary']==1)&(df['her2ratio'].isna())) & (df['her2copies'] < 4), #2
    ((df['her2ratio'] < 2)|(df['her2ratio_binary']==0)&(df['her2ratio'].isna())) & (df['her2copies'] >= 6), #3
    ((df['her2ratio'] < 2)|(df['her2ratio_binary']==0)&(df['her2ratio'].isna())) & (df['her2copies'] >= 4) & (df['her2copies'] < 6), #4
    ((df['her2ratio'] < 2)|(df['her2ratio_binary']==0)&(df['her2ratio'].isna())) & (df['her2copies'] < 4), #5

    # Group 0 contains all patients who are considered ISH positive, but do not fit into ASCO/CAP group 1. 
    # The rule for determining discordance for group 0 is identical to group 1.
    # The conditions are:
    # 1. ISH ratio test does not give sufficient information, so the ISH is determined by the copies test that is positive
    ((df['her2ratio'].isna()&(df['her2ratio_binary'].isna())) & ((df['her2copies'] >= 6)|(df['her2ish_binary']==1)&(df['her2copies'].isna())))\
        | (df['her2ratio_binary']==1)&((df['her2ratio'].isna())|(df['her2ratio']>=2))&(df['her2copies'].isna())&(df['her2ish_binary'].isna()), #0
    ((df['her2ratio'].isna()&(df['her2ratio_binary'].isna())) & ((df['her2copies'] < 4)|(df['her2ish_binary']==0)&(df['her2copies'].isna())))\
        | (df['her2ratio_binary']==0)&((df['her2ratio'].isna())|(df['her2ratio']<2))&(df['her2copies'].isna())&(df['her2ish_binary'].isna()), #6
    ]

    choices1 = [1, 2, 3, 4, 5, 0, 6]
    df['asco_group'] = np.select(conditions1, choices1, default=np.nan)
    '''
    # the issue with the above code is that patients with df['her2ratio_binary'].notna()&df['her2ish_binary'].notna() became nan


    # This code implements table a, b, c from the subplementary figures
    conditions_table = [
        (df['her2ratio_binary'].isna()) & (df['her2ish_binary'].isna()),
        (df['her2ratio_binary'].isna()) & (df['her2ish_binary'] == 0),
        (df['her2ratio_binary'].isna()) & (df['her2ish_binary'] == 1),
        
        (df['her2ratio_binary'] == 0) & (df['her2ish_binary'].isna()),
        (df['her2ratio_binary'] == 0) & (df['her2ish_binary'] == 0),
        (df['her2ratio_binary'] == 0) & (df['her2ish_binary'] == 1),

        (df['her2ratio_binary'] == 1) & (df['her2ish_binary'].isna()),
        (df['her2ratio_binary'] == 1) & (df['her2ish_binary'] == 0),
        (df['her2ratio_binary'] == 1) & (df['her2ish_binary'] == 1),
    ]

    choices_a = [
        7, 6, 0,
        6, 6, 7,
        0, 7, 0
    ]

    choices_b = [
        7, 5, 7,
        6, 5, 7,
        7, 7, 7
    ]

    choices_c = [
        7, 7, 1,
        7, 7, 7,
        0, 7, 1
    ]

    df['table_a'] = np.select(conditions_table, choices_a, default=np.nan)
    df['table_b'] = np.select(conditions_table, choices_b, default=np.nan)
    df['table_c'] = np.select(conditions_table, choices_c, default=np.nan)

    conditions_group = [
        (df['her2ratio'].isna()) & (df['her2copies'].isna()),
        (df['her2ratio'].isna()) & (df['her2copies'] < 4),
        (df['her2ratio'].isna()) & (df['her2copies'] >= 4) & (df['her2copies'] < 6),
        (df['her2ratio'].isna()) & (df['her2copies'] >= 6),

        (df['her2ratio'] < 2) & (df['her2copies'].isna()),
        (df['her2ratio'] < 2) & (df['her2copies'] < 4),
        (df['her2ratio'] < 2) & (df['her2copies'] >= 4) & (df['her2copies'] < 6),
        (df['her2ratio'] < 2) & (df['her2copies'] >= 6),

        (df['her2ratio'] >= 2) & (df['her2copies'].isna()),
        (df['her2ratio'] >= 2) & (df['her2copies'] < 4),
        (df['her2ratio'] >= 2) & (df['her2copies'] >= 4) & (df['her2copies'] < 6),
        (df['her2ratio'] >= 2) & (df['her2copies'] >= 6),
    ]

    choices_group = [
        10, 6, 10, 0,
        20, 5, 4, 3,
        30, 2, 1, 1
    ]

    df['asco_group'] = np.select(conditions_group, choices_group, default=np.nan)
    df['asco_group_sensitivity'] = df['asco_group']
    df.loc[df['asco_group_sensitivity']>=10, 'asco_group_sensitivity'] = 7

    df.loc[df['asco_group'] == 10, 'asco_group'] = df.loc[df['asco_group'] == 10, 'table_a']
    df.loc[df['asco_group'] == 20, 'asco_group'] = df.loc[df['asco_group'] == 20, 'table_b']
    df.loc[df['asco_group'] == 30, 'asco_group'] = df.loc[df['asco_group'] == 30, 'table_c']


    
    df['er'] = df.receptors.map({0:'-', 1:'-', 10:'-', 11:'-', 100:'+',101:'+', 110:'+',111:'+'})
    df['pr'] = df.receptors.map({0:'-', 1:'-', 10:'+', 11:'+', 100:'-', 101:'-', 110:'+', 111:'+'})
    df['ihc'] = df.her2.map({0: 0, 1: 1, 10: 1, 2: 2, 20: 2, 3: 3, 30: 3}).fillna(np.nan)

    conditions_cat = [
    (df['asco_group'] == 0) & (df['ihc'] <= 1),
    (df['asco_group'] == 0) & (df['ihc'] == 2),
    (df['asco_group'] == 0) & (df['ihc'] == 3),
    (df['asco_group'] == 0) & (df['ihc'].isna()),


    (df['asco_group'] == 1) & (df['ihc'] <= 1),
    (df['asco_group'] == 1) & (df['ihc'] == 2),
    (df['asco_group'] == 1) & (df['ihc'] == 3),
    (df['asco_group'] == 1) & (df['ihc'].isna()),

    (df['asco_group'] == 2) & (df['ihc'] <= 1),
    (df['asco_group'] == 2) & (df['ihc'] == 2),
    (df['asco_group'] == 2) & (df['ihc'] == 3),
    (df['asco_group'] == 2) & (df['ihc'].isna()),

    (df['asco_group'] == 3) & (df['ihc'] <= 1),
    (df['asco_group'] == 3) & (df['ihc'] == 2),
    (df['asco_group'] == 3) & (df['ihc'] == 3),
    (df['asco_group'] == 3) & (df['ihc'].isna()),

    (df['asco_group'] == 4) & (df['ihc'] <= 1),
    (df['asco_group'] == 4) & (df['ihc'] == 2),
    (df['asco_group'] == 4) & (df['ihc'] == 3),
    (df['asco_group'] == 4) & (df['ihc'].isna()),

    (df['asco_group'] == 5) & (df['ihc'] <= 1),
    (df['asco_group'] == 5) & (df['ihc'] == 2),
    (df['asco_group'] == 5) & (df['ihc'] == 3),
    (df['asco_group'] == 5) & (df['ihc'].isna()),

    (df['asco_group'] == 6) & (df['ihc'] <= 1),
    (df['asco_group'] == 6) & (df['ihc'] == 2),
    (df['asco_group'] == 6) & (df['ihc'] == 3),
    (df['asco_group'] == 6) & (df['ihc'].isna()),

    (df['asco_group'] == 7) & (df['ihc'] <= 1),
    (df['asco_group'] == 7) & (df['ihc'] == 2),
    (df['asco_group'] == 7) & (df['ihc'] == 3),
    (df['asco_group'] == 7) & (df['ihc'].isna()),
    ]

    choices_cat = [
    'B','AA','AA','AA',
    'B','AA','AA','AA',
    'E','E','A2','G',
    'E','A3-2','A3','G',
    'E','E','A4','G',
    'C','C','D','C',
    'C','C','D','C',
    'C','F','AA','F'
    ]

    df['cat'] = np.select(conditions_cat, choices_cat, default=np.nan)
    #override: cat is set OTHER/Invalid if (ASCO group is np.nan) AND (HER2 ISH is available)
    #that is, for cat to be determined solely by IHC values, all ISH values must be missing
    df.loc[(df['cat']!='F')&(df['asco_group']==7)&(~((df['her2ratio'].isna())&(df['her2ratio_binary'].isna())&(df['her2ish_binary'].isna())&(df['her2copies'].isna()))), 'cat'] = 'H'

    df['cat_pos'] = df['cat']


    df['cat4'] = df['cat'].replace({'A2':'o',
                                'A3':'o',
                                'A3-2':'o',
                                'A4':'o',
                                'B':'pn',
                                'C':'n',
                                'D':'np',
                                'E':'o',
                                'F':'o',
                                'G':'o',
                                'H':'o',
                                'AA':'p'})
    
    df['cat4_pos'] = df['cat4']
    
    
    #Exclude OTHER category patients (IHC/ISH information cannot be used to determine HER2 status under ASCO guidelines)


    #df = df[df['cat'] != 'F']
    #print("Excluding OTHER category: " + str(len(df.index)))

    # We can now save our more limited datafile if we wnat
    if savefile:
        df.to_csv(savefile, index=False)
    
    # This returns the dataframe as a result of the function, so it can be accessed for further processing
    return df


saved_impute = None
def setDummies(df, var, subset, missing_indicator = True, multiple_impute = False):
    ret_cols = []
    if missing_indicator:
        df['missing_' + str(var)] = 0
        df.loc[df[var].isna(), 'missing_' + str(var)] = 1
    for v in subset:
        df[var + "_" + str(v)] = 0
        ret_cols += [var  + "_" +  str(v)]
        df.loc[df[var] == v, var  + "_" + str(v)] = 1
        if missing_indicator:
            df.loc[df[var].isna(), var  + "_" + str(v)] = df[var  + "_" + str(v)].mean()
        if multiple_impute:
            df.loc[df[var].isna(), var  + "_" + str(v)] = np.nan            
    return ret_cols

def applyMissing(df, column, targets = None, missing_indicator = True, multiple_impute = False):
    if targets is None:
        targets = [column]
    if missing_indicator:
        df['missing_' + column] = 0
        df.loc[df[column].isna(), 'missing_' + column] = 1
        for t in targets:
            df.loc[df[column].isna(), t] = df[t].mean()
    if multiple_impute:
        for t in targets:
            df.loc[df[column].isna(), t] = np.nan

def parse_columns(df_subset, cols, ignore = [], include_stage4 = True, all_histologies = True, her2_categories = False, multiple_impute = True, missing_indicator = False, redo_impute = False, receptors = None):
    if (not multiple_impute) and (not missing_indicator):
        df_subset = df_subset[cols].dropna()
        print('Dropping all "NA" values.')

    cols = [c for c in cols if c not in ignore]
    test_cols = []
    label_cols = []
    label_names = []
    if missing_indicator:
        for c in cols:
            if c not in ['alive', 'last_contact','her2']:
                if len(df_subset[df_subset[c].isna()].index) > 0:
                    test_cols += ['missing_' + c]
                    label_cols += [['missing_' + c]]
                    label_names += [['Missing ' + c]]
            
    if 'age' in cols:
        df_subset['age'] = df_subset['age'] / 10
        test_cols += ['age']
        label_cols += [['age']]
        label_names += [['Age (per year)']]
        applyMissing(df_subset, 'age', missing_indicator = missing_indicator, multiple_impute = multiple_impute)

    if 'sex' in cols:
        t = setDummies(df_subset, 'sex', {2}, missing_indicator)
        test_cols += t
        label_cols += [t]
        label_names += [['Female', 'Male']]
        
    if 'cdcc' in cols:
        test_cols += ['cdcc']
        label_cols += [['cdcc']]
        label_names += [['CDCC 0', 'CDCC > 0']]
        df_subset['cdcc_high'] = 0
        df_subset.loc[df_subset.cdcc.astype(float) > 0, 'cdcc_high'] = 1
        applyMissing(df_subset, 'cdcc', ['cdcc_high'], missing_indicator = missing_indicator, multiple_impute = multiple_impute)

    if 'immuno' in cols:
        t = setDummies(df_subset, 'immuno', {1}, missing_indicator)
        test_cols += t
        label_cols += [t]
        label_names += [['No Immunotherapy', 'Immunotherapy']]


        
    if 'facility' in cols:
        t = setDummies(df_subset, 'facility', {2,3,4}, missing_indicator = missing_indicator, multiple_impute = multiple_impute)
        test_cols += t
        label_cols += [t]
        label_names += [['Community Cancer Program', 'Comprehensive Comunity Cancer Program', 'Academic / Research', 'Integrated Network Cancer Program']]

    if 'receptors' in cols:
        if len(df_subset[df_subset.receptors.astype(float)==0].index > 0) and receptors is None:
            test_cols += ['prneg', 'erneg', 'tnbc']
            label_cols += [['prneg', 'erneg', 'tnbc']]
            label_names += [['ER+/PR+', 'ER+/PR-', 'ER-/PR+', 'TNBC']]
            df_subset['prneg'] = 0
            df_subset['erneg'] = 0
            df_subset['tnbc'] = 0
            df_subset.loc[df_subset.receptors.astype(float) == 100, 'prneg'] = 1
            df_subset.loc[df_subset.receptors.astype(float) == 10, 'erneg'] = 1
            df_subset.loc[df_subset.receptors.astype(float) == 0, 'tnbc'] = 1
            applyMissing(df_subset, 'receptors', ['prneg', 'erneg', 'tnbc'], missing_indicator = missing_indicator, multiple_impute = multiple_impute)
        else:
            test_cols += ['prneg', 'erneg']
            label_cols += [['prneg', 'erneg']]
            label_names += [['ER+/PR+', 'ER+/PR-', 'ER-/PR+']]
            df_subset['prneg'] = 0
            df_subset['erneg'] = 0
            df_subset.loc[df_subset.receptors.astype(float) == 100, 'prneg'] = 1
            df_subset.loc[df_subset.receptors.astype(float) == 10, 'erneg'] = 1
            applyMissing(df_subset, 'receptors', ['prneg', 'erneg'], missing_indicator = missing_indicator, multiple_impute = multiple_impute)

            
    if 'er_num' in cols:
        df_subset['er_num'] = df_subset['er_num'] / 10
        test_cols += ['er_num']
        label_cols += [['er_num']]
        label_names += [['ER (per 10% expression)']]
        applyMissing(df_subset, 'er_num', missing_indicator = missing_indicator, multiple_impute = multiple_impute)

    if 'pr_num' in cols:
        df_subset['pr_num'] = df_subset['pr_num'] / 10
        test_cols += ['pr_num']
        label_cols += [['pr_num']]
        label_names += [['PR (per 10% expression)']]
        applyMissing(df_subset, 'pr_num', missing_indicator = missing_indicator, multiple_impute = multiple_impute)

    if 'ki67_num' in cols:
        df_subset['ki67_num'] = df_subset['ki67_num'] / 10
        test_cols += ['ki67_num']
        label_cols += [['ki67_num']]
        label_names += [['Ki67 (per 10% expression)']]
        applyMissing(df_subset, 'ki67_num', missing_indicator = missing_indicator, multiple_impute = multiple_impute)

    if 'her2' in cols:
        if her2_categories:
            test_cols += ['her2_1', 'her2_2']
            label_cols += [['her2_1', 'her2_2']]
            label_names += [['HER2 0', 'HER2 1+', 'HER2 2+']]
            df_subset['her2_1'] = 0
            df_subset['her2_2'] = 0
            df_subset.loc[df_subset.her2.astype(float) == 1, 'her2_1'] = 1        
            df_subset.loc[df_subset.her2.astype(float) == 2, 'her2_2'] = 1
        else:
            test_cols += ['her2low']
            label_cols += [['her2low']]
            label_names += [['HER2 0', 'HER2 Low']]
            df_subset['her2low'] = 0
            df_subset.loc[df_subset.her2.astype(float) > 0, 'her2low'] = 1

    if 'her2ratio' in cols:
        test_cols += ['her2ratio']
        label_cols += [['her2ratio']]
        label_names += [['HER2 Ratio (per 1.0 increase)']]
        applyMissing(df_subset, 'her2ratio', missing_indicator = missing_indicator, multiple_impute = multiple_impute)

    if 'her2copies' in cols:
        test_cols += ['her2copies']
        label_cols += [['her2copies']]
        label_names += [['HER2 Count (per 1.0 increase)']]
        applyMissing(df_subset, 'her2copies', missing_indicator = missing_indicator, multiple_impute = multiple_impute)
    
    if 'odx' in cols:
        test_cols += ['odx']
        label_cols += [['odx']]
        label_names += [['OncotypeDx (per 1 point increase)']]
        applyMissing(df_subset, 'odx', missing_indicator = missing_indicator, multiple_impute = multiple_impute)

    if 'race_parse' in cols:
        t = setDummies(df_subset, 'race_parse', {1,2,3,4,5}, missing_indicator = missing_indicator, multiple_impute = multiple_impute)
        test_cols += t
        label_cols += [t]
        label_names += [['Non-Hispanic White', 'Non-Hispanic Black', 'Hispanic', 'Other', 'Native American', 'Asian / Pacific Islander']]

    if 'tumor_size' in cols:
        test_cols += ['tumor_size']
        label_cols += [['tumor_size']]
        label_names += [['Tumor Size (per mm increase)']]
        applyMissing(df_subset, 'tumor_size', missing_indicator = missing_indicator, multiple_impute = multiple_impute)

    if 'regional_nodes_positive' in cols:
        test_cols += ['regional_nodes_positive']
        label_cols += [['regional_nodes_positive']]
        label_names += [['Nodes Positive (per # of involved nodes)']]
        applyMissing(df_subset, 'regional_nodes_positive', missing_indicator = missing_indicator, multiple_impute = multiple_impute)
        
    if 'tnm_clin_t' in cols:
        test_cols += ['t2', 't3', 't4']
        label_cols += [['t2', 't3', 't4']]
        label_names += [['T1', 'T2', 'T3', 'T4']]
        if not missing_indicator and not multiple_impute:
            df_subset.loc[df_subset.tnm_clin_t.str.contains('1') | df_subset.tnm_clin_t.str.contains('2') | df_subset.tnm_clin_t.str.contains('3') | df_subset.tnm_clin_t.str.contains('4')].reset_index()
        
        df_subset['t2'] = 0
        df_subset['t3'] = 0
        df_subset['t4'] = 0        
        df_subset.loc[df_subset.tnm_clin_t.str.contains('2'), 't2'] = 1
        df_subset.loc[df_subset.tnm_clin_t.str.contains('3'), 't3'] = 1
        df_subset.loc[df_subset.tnm_clin_t.str.contains('4'), 't4'] = 1
        if missing_indicator or multiple_impute:
            df_subset.loc[~(df_subset.tnm_clin_t.str.contains('1') | df_subset.tnm_clin_t.str.contains('2') | df_subset.tnm_clin_t.str.contains('3') | df_subset.tnm_clin_t.str.contains('4')), 'tnm_clin_t'] = np.nan
            applyMissing(df_subset, 'tnm_clin_t', ['t2', 't3', 't4'], missing_indicator = missing_indicator, multiple_impute = multiple_impute)

    if 'tnm_clin_n' in cols:
        test_cols += ['n1', 'n2', 'n3']
        label_cols += [['n1', 'n2', 'n3']]
        label_names += [['N0', 'N1', 'N2', 'N3']]
        if not missing_indicator and not multiple_impute:
            df_subset = df_subset[df_subset.tnm_clin_n.str.contains('0') | df_subset.tnm_clin_n.str.contains('1') | df_subset.tnm_clin_n.str.contains('2') | df_subset.tnm_clin_n.str.contains('3')].reset_index()
        df_subset['n1'] = 0
        df_subset['n2'] = 0
        df_subset['n3'] = 0
        df_subset.loc[df_subset.tnm_clin_n.str.contains('1'), 'n1'] = 1
        df_subset.loc[df_subset.tnm_clin_n.str.contains('2'), 'n2'] = 1
        df_subset.loc[df_subset.tnm_clin_n.str.contains('3'), 'n3'] = 1
        if missing_indicator or multiple_impute:
            df_subset.loc[~(df_subset.tnm_clin_n.str.contains('1') | df_subset.tnm_clin_n.str.contains('2') | df_subset.tnm_clin_n.str.contains('3') | df_subset.tnm_clin_n.str.contains('0')), 'tnm_clin_n'] = np.nan
            applyMissing(df_subset, 'tnm_clin_n', ['n1', 'n2', 'n3'], missing_indicator = missing_indicator, multiple_impute = multiple_impute)

    if 't_stage' in cols:
        if not missing_indicator and not multiple_impute:
            df_subset = df_subset[df_subset.t_stage.isin([1,2,3,4])].reset_index()
        test_cols += ['t2', 't3', 't4']
        label_cols += [['t2', 't3', 't4']]
        label_names += [['T1', 'T2', 'T3', 'T4']]
        df_subset['t2'] = 0
        df_subset['t3'] = 0
        df_subset['t4'] = 0
        df_subset.loc[df_subset.t_stage == 2, 't2'] = 1
        df_subset.loc[df_subset.t_stage == 3, 't3'] = 1
        df_subset.loc[df_subset.t_stage == 4, 't4'] = 1
        if missing_indicator or multiple_impute:
            df_subset.loc[~(df_subset.t_stage.isin([1,2,3,4])), 't_stage'] = np.nan
            applyMissing(df_subset, 't_stage', ['t2', 't3', 't4'], missing_indicator = missing_indicator, multiple_impute = multiple_impute)

    if 'n_stage' in cols:
        test_cols += ['n1', 'n2', 'n3']
        label_cols += [['n1', 'n2', 'n3']]
        label_names += [['N0', 'N1', 'N2', 'N3']]
        if not missing_indicator and not multiple_impute:
            df_subset = df_subset[df_subset.n_stage.isin([0,1,2,3])].reset_index()
        df_subset['n1'] = 0
        df_subset['n2'] = 0
        df_subset['n3'] = 0
        df_subset.loc[df_subset.n_stage == 1, 'n1'] = 1
        df_subset.loc[df_subset.n_stage == 2, 'n2'] = 1
        df_subset.loc[df_subset.n_stage == 3, 'n3'] = 1
        if missing_indicator or multiple_impute:
            df_subset.loc[~(df_subset.n_stage.isin([0,1,2,3])), 'n_stage'] = np.nan
            applyMissing(df_subset, 'n_stage', ['n1', 'n2', 'n3'], missing_indicator = missing_indicator, multiple_impute = multiple_impute)

    if 'stage' in cols:
        if include_stage4:
            t = setDummies(df_subset, 'stage', {2,3,4},  missing_indicator = missing_indicator, multiple_impute = multiple_impute)
            test_cols += t
            label_cols += [t]
            label_names += [['Stage 1', 'Stage 2', 'Stage 3', 'Stage 4']]
        else:
            t = setDummies(df_subset, 'stage', {2,3}, missing_indicator = missing_indicator, multiple_impute = multiple_impute)
            test_cols += t
            label_cols += [t]
            label_names += [['Stage 1', 'Stage 2', 'Stage 3']]
            
    if 'grade' in cols:
        t = setDummies(df_subset, 'grade', {2,3},  missing_indicator = missing_indicator, multiple_impute = multiple_impute)
        test_cols += t
        label_cols += [t]
        label_names += [['Grade 1', 'Grade 2', 'Grade 3']]
        
    if 'histology' in cols:
        if all_histologies:
            t = setDummies(df_subset, 'histology', {1,2,3,4,5,6,7,8,9,10}, missing_indicator = missing_indicator, multiple_impute = multiple_impute)
            test_cols += t + ['histology_others']
            label_cols += [t + ['histology_others']]
            df_subset['histology_others'] = 0
            df_subset.loc[~df_subset.histology.isin([0,1,2,3,4,5,6,7,8,9,10]), 'histology_others'] = 1
            label_names += [['Ductal', 'Lobular', 'Ductal and Lobular', 'Mucinous', 'Papillary', 'Tubular', 'Inflammatory', 'Medullary', 'Metaplastic', 'Paget Disease', 'Sarcoma', 'Other']]

        else:
            t = setDummies(df_subset, 'histology', {1,2,3}, missing_indicator = missing_indicator, multiple_impute = multiple_impute)
            test_cols += t + ['histology_others']
            label_cols += [t + ['histology_others']]
            df_subset['histology_others'] = 0
            df_subset.loc[~df_subset.histology.isin([0,1,2,3]), 'histology_others'] = 1
            label_names += [['Ductal', 'Lobular', 'Ductal and Lobular', 'Mucinous']]

    if 'er' in cols:
        t = setDummies(df_subset, 'er', {'-'},  missing_indicator = missing_indicator, multiple_impute = multiple_impute)
        test_cols += t
        label_cols += [t]
        label_names += [['ER +','ER -']]

    if 'pr' in cols:
        t = setDummies(df_subset, 'pr', {'-'},  missing_indicator = missing_indicator, multiple_impute = multiple_impute)
        test_cols += t
        label_cols += [t]
        label_names += [['PR +','PR -']]

    if 'cat' in cols:
        t = setDummies(df_subset, 'cat', {'AA','A2','A3','A4','B','D','E'}, missing_indicator = missing_indicator, multiple_impute = multiple_impute)
        test_cols += t
        label_cols += [t]
        label_names += [['HER2-','HER2+','Group 2/IHC+','Group 3/IHC+','Group 4/IHC+','ISH+/IHC-','ISH-/IHC+','Group 2,3,4/IHC-']]

    if 'cat_pos' in cols:
        t = setDummies(df_subset, 'cat_pos', {'C','A2','A3','A4','B','D','E'}, missing_indicator = missing_indicator, multiple_impute = multiple_impute)
        test_cols += t
        label_cols += [t]
        label_names += [['HER2+','HER2-','Group 2/IHC+','Group 3/IHC+','Group 4/IHC+','ISH+/IHC-','ISH-/IHC+','Group 2,3,4/IHC-']]


    if 'cat4' in cols:
        t = setDummies(df_subset, 'cat4', {'p','pn','np'}, missing_indicator = missing_indicator, multiple_impute = multiple_impute)
        test_cols += t
        label_cols += [t]
        label_names += [['HER2-','HER2+','IHC-/ISH+','IHC+/ISH-']]

    if 'cat4_pos' in cols:
        t = setDummies(df_subset, 'cat4_pos', {'n','pn','np'}, missing_indicator = missing_indicator, multiple_impute = multiple_impute)
        test_cols += t
        label_cols += [t]
        label_names += [['HER2+','HER2-','IHC-/ISH+','IHC+/ISH-']]


    if multiple_impute:
        global saved_impute
        if (saved_impute is not None) and (not redo_impute):
            print("Using saved imputation")
            df_subset = saved_impute.copy()
        else:
            print("Redoing imputation")
            from sklearn.experimental import enable_iterative_imputer
            from sklearn.impute import IterativeImputer
            imp = IterativeImputer(max_iter=100, random_state=0, verbose = 2, skip_complete=True)
            saved_impute = df_subset.copy()
            print(len(df_subset.index))
            saved_impute[test_cols] = imp.fit_transform(saved_impute[test_cols])
            print(len(saved_impute.index))

    return df_subset, test_cols, label_cols, label_names