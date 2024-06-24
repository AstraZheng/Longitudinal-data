#!/usr/bin/env python
# coding: utf-8

# ### RA - Job History Coding 
# Created a long dataset to record the individuals' occupation history including the jobs and the gaps using UKB online follow-up data. For every individual, all his/her job histories are recorded in sequential years, where each row represents one specific year in his/her career life. The years for the gaps are not included in the timeline, but are recorded independently using additional columns, categorized by the gap reasons.

# In[38]:

import os 
import pandas as pd
import numpy as np
import datetime
import fnmatch
import warnings
warnings.filterwarnings('ignore')
#os.chdir('C:\\Users\\chuyi\\Desktop\\NUS RA\\Recode the job history')


# #### 1. Loaded the data 
# Cut from subset_online_follow_up.txt

# In[39]:


#large = pd.read_table('full_dataset_cut4.txt')
large = pd.read_table('subset_online_follow_up.txt')
large = large.rename(columns={'f.eid': 'ID',
                              'f.22200.0.0' : 'Year_of_Birth',
                              'f.22599.0.0' : 'Number_of_Jobs_Held'})
large[large['Year_of_Birth'] > 0].shape


# #### 2. Create the dataframe for job history with job coding

# In[40]:


items = ['f.22602.0.*', 'f.22603.0.*', 'f.22617.0.*'] 

# The function to create the long data with selected columns 
# input: items (using wildcard * to match the field index

def create_long_data(items) :

    info = ['ID','Year_of_Birth', 'Number_of_Jobs_Held']
    col_list = large.columns.tolist()
    select = fnmatch.filter(col_list, 'f.22601.0.*')
    df_job = large.loc[:, info + select]
    df_ind = pd.wide_to_long(df_job, 
                             stubnames = 'f.22601.0.', 
                             i = info, 
                             j = 'Index').reset_index().dropna()
 
    to_match = items
    to_match2 = to_match.copy()
    for i in list(range(0, len(to_match))) :
        to_match2[i] = to_match[i][:-1]
    
    for i in list(range(0, len(to_match))) :
        select = fnmatch.filter(col_list, to_match[i])
        df = large.loc[:, info + select]
        df_step = pd.wide_to_long(df, stubnames = to_match2[i], i = info, j = 'Index').reset_index().dropna()
        df_ind = df_ind.merge(df_step, on = info + ['Index'], how = 'left')


    df_ind = df_ind.rename(columns={'f.22601.0.' : 'Job_Code',
                                    'f.22602.0.' : 'Job_Start_Year',
                                    'f.22603.0.' : 'Job_End_Year',
                                    'f.22617.0.' : 'SOC2000'})
    df_ind['Job_End_Year'] = df_ind['Job_End_Year'].replace(-313, 2017)
    df_ind['Job_Hold_Year'] = df_ind['Job_End_Year'] - df_ind['Job_Start_Year'] + 1
    df_ind['Year'] = df_ind['Job_Start_Year']

    # duplicate the rows by their working years 
    df_long = df_ind.loc[df_ind.index.repeat(df_ind['Job_Hold_Year'])].reset_index(drop=True)
    

    # change the years and create the long data with sequential years
    id_list = df_long['ID'].unique()
    accumulate = pd.DataFrame(columns=df_ind.columns)
    for i in list(range(0, len(id_list))):
        sub1 = df_ind[df_ind['ID'] == id_list[i]]
        index_list = list(range(0,len(sub1)))
        for j in index_list:
            sub2 = df_long[(df_long['ID'] == id_list[i]) & (df_long['Index'] == j)]  
            sub2.loc[:,'Year'] = np.arange(sub2.iloc[0,5], sub2.iloc[0,5] + len(sub2))
            accumulate = pd.concat([accumulate, sub2], ignore_index=True)

    accumulate['Age'] = accumulate['Year'] - accumulate['Year_of_Birth']
    accumulate['Job_End_Year'] = accumulate['Job_End_Year'].replace(2017, -313) 
    accumulate['Index'] = accumulate['Index'] + 1
    accumulate = accumulate.drop_duplicates(['ID','Year','Age'],keep='last')
    return accumulate


# run the function
df_accumulate = create_long_data(items)


# #### 3. Create the dataframe that collects all the gap records for individuals

# In[41]:


# create the dataframe for gap years

df_gaps = large.loc[:, ['ID', 'f.22661.0.0']].rename(columns={'f.22661.0.0':'Number_of_Gaps'})
df_gaps['Number_of_Gaps'] = df_gaps['Number_of_Gaps'].replace(np.nan, 0)

col_list = large.columns.tolist()
gap_start_col = fnmatch.filter(col_list, 'f.22663.0.*')
gap_end_col = fnmatch.filter(col_list, 'f.22664.0.*')
gap_coding_col = fnmatch.filter(col_list, 'f.22660.0.*')

df_gap_start =  large.loc[:, ['ID','Year_of_Birth'] + gap_start_col]
df_step1 = pd.wide_to_long(df_gap_start, stubnames = 'f.22663.0.', i = ['ID', 'Year_of_Birth'], j = 'Index').reset_index().dropna()
df_gap_end =  large.loc[:, ['ID','Year_of_Birth'] + gap_end_col]
df_step2 = pd.wide_to_long(df_gap_end, stubnames = 'f.22664.0.', i = ['ID', 'Year_of_Birth'], j = 'Index').reset_index().dropna()
df_gap_coding =  large.loc[:, ['ID','Year_of_Birth'] + gap_coding_col]
df_step3 = pd.wide_to_long(df_gap_coding, stubnames = 'f.22660.0.', i = ['ID', 'Year_of_Birth'], j = 'Index').reset_index().dropna() 

df_full_gap = df_step1.merge(df_step2, on = ['ID', 'Year_of_Birth', 'Index'], how='outer').merge(df_step3, on = ['ID', 'Year_of_Birth', 'Index'], how='outer')

df_full_gap = df_full_gap.rename(columns={'f.22663.0.': 'Gap_Start_Year',
                                          'f.22664.0.': 'Gap_End_Year',
                                          'f.22660.0.' : 'Reason_of_Gap'})

df_full_gap2 = df_full_gap.drop_duplicates(['ID', 'Gap_Start_Year'], keep='last')
df_full_gap['Length_of_Gap'] = df_full_gap['Gap_End_Year'] - df_full_gap['Gap_Start_Year'] + 1
df_full_gap.loc[df_full_gap['Length_of_Gap'] < 0, 'Length_of_Gap'] = np.nan
df_full_gap['Gap_Before_This_Job'] = "Yes"   


# #### 4. Reconstruct the gap information and merge it to the job history dataframe 

# ##### 4.1 Identify the retirement year and add a column 'Retired Year'

# In[42]:


# identify the retired year 
df_retired = df_full_gap[(df_full_gap['Gap_End_Year'] == -313) & (df_full_gap['Reason_of_Gap'] == 108)].drop_duplicates(['ID'], keep = 'last')
df_retired = df_retired[['ID', 'Year_of_Birth', 'Gap_Start_Year']].rename(columns={'Gap_Start_Year' : 'Retired_Year'})
df_add_gap1 = df_accumulate.merge(df_retired, on = ['ID', 'Year_of_Birth'], how = 'left')


# ##### 4.2 Find the gaps that before and closest to the job that recorded in the current row

# In[43]:


# match job start year with gap end year 
df_full_gap['Gap_End_Year2'] = df_full_gap['Gap_End_Year']
df_full_gap2 = df_full_gap.drop(['Index'], axis=1).rename(columns={'Gap_End_Year2' : 'Job_Start_Year'})    # match gap end year with job start year 
df_add_gap2 = df_add_gap1.merge(df_full_gap2, on = ['ID','Year_of_Birth', 'Job_Start_Year'], how = 'left')

# some additional cases: match job start year with gap end year + 1
df_full_gap['Gap_End_Year3'] = df_full_gap['Gap_End_Year'] + 1
df_full_gap_adj = df_full_gap.drop(['Index'], axis=1).rename(columns={'Gap_End_Year3' : 'Job_Start_Year'})    # match gap end year with job start year 
df_add_gap_adj = df_add_gap1.merge(df_full_gap_adj, on = ['ID','Year_of_Birth', 'Job_Start_Year'], how = 'left')

# merge two dataframes 
combine_col_list = ['Gap_Before_This_Job', 'Gap_Start_Year', 'Gap_End_Year', 'Reason_of_Gap', 'Length_of_Gap']
for col_name in combine_col_list :
    df_add_gap2[col_name] = df_add_gap2[col_name].fillna(df_add_gap_adj.pop(col_name))   ########## adjust

# merge df with number of gaps column
df_add_gap2 = df_add_gap2.merge(df_gaps, on = ['ID'], how = 'left')


# ##### 4.3 Check if the job held after the retirement, and add the column 'Is After Retirement'.

# In[44]:


# add the column "the job is after retirement"
df_after_retirement = df_add_gap2[(df_add_gap2['Gap_Before_This_Job'] == "Yes") & (df_add_gap2['Reason_of_Gap'] == 108)].drop_duplicates(['ID','Index'])
df_after_retirement['Is_After_Retirement'] = "Yes"
df_after_retirement = df_after_retirement[['ID', 'Job_Code', 'Index', 'Gap_Start_Year', 'Gap_End_Year', 'Reason_of_Gap', 'Is_After_Retirement']]
df_add_gap3 = df_add_gap2.merge(df_after_retirement, on = ['ID', 'Job_Code', 'Index', 'Gap_Start_Year', 'Gap_End_Year', 'Reason_of_Gap'], how = 'left')

col_order = ['ID', 'Year_of_Birth', 'Number_of_Jobs_Held', 'Number_of_Gaps', 'Retired_Year',
             'Job_Code', 'SOC2000', 'Year', 'Age',  'Job_Start_Year', 'Job_End_Year', 'Index', 
             'Is_After_Retirement', 'Gap_Before_This_Job', 'Gap_Start_Year', 'Gap_End_Year', 'Reason_of_Gap', 'Length_of_Gap']
df_add_gap3 = df_add_gap3[col_order]


# ##### 4.4 Detected the gaps that can't be captured by matching the column (no job followed)
# Usually the case when someone reported several different gaps (with different reasons) between two jobs. 

# In[45]:


# get the undetected gaps: 
detected_gap = df_add_gap3[df_add_gap3['Gap_Start_Year'].notna()].drop(['Year', 'Age'], axis = 1).drop_duplicates()
full_gap = df_full_gap[['ID', 'Gap_Start_Year', 'Gap_End_Year', 'Reason_of_Gap']]

merge = full_gap.merge(detected_gap, on = ['ID', 'Gap_Start_Year', 'Gap_End_Year', 'Reason_of_Gap'], how = 'left')
undetected = merge[merge['Year_of_Birth'].isnull()]

undetected = undetected[(undetected['Gap_End_Year'] != -313) | (undetected['Reason_of_Gap'] != 108)][['ID', 'Gap_Start_Year', 'Gap_End_Year', 'Reason_of_Gap']]
undetected_gap = undetected[undetected['Gap_End_Year'] != -313]


# get the last recorded job end year (except -313: currently working on)
df_last_job_end = df_accumulate[df_accumulate['Job_End_Year'] != -313].groupby(['ID'])['Job_End_Year'].max().reset_index()
mid = undetected_gap.merge(df_last_job_end, on = ['ID'], how = 'left')

# the finalized undetected gap 
undetected_ava_gap = mid[mid['Gap_Start_Year'] < mid['Job_End_Year']].drop(['Job_End_Year'], axis=1)


# ##### 4.5 Code these undetected gaps using additional columns for each gap reasons 

# In[46]:


df_add_gap4 = df_add_gap3.copy()

# make a function to code the gaps below
reason_list = [101, 102, 103, 105, 106, 107, -717]
reason_str_list = ['Part-time_Gap','Voluntary_Work_Gap', 'Education_Gap', 'Family_Reason_Gap', 'Sickness_Gap', 'Unemployment_Gap', 'Other_Reason_Gap']
start_col_list = ['P_Start', 'V_Start', 'E_Start', 'F_Start', 'S_Start','U_Start', 'O_Start']
end_col_list = ['P_End', 'V_End', 'E_End', 'F_End', 'S_End', 'U_End', 'O_End']

for i in range(0,len(reason_list)):

    for_spec = undetected_ava_gap[undetected_ava_gap['Reason_of_Gap'] == reason_list[i]]
    id_list = for_spec['ID']
    current = df_accumulate.loc[df_accumulate['ID'].isin(id_list)].drop(['Year', 'Age'], axis = 1).drop_duplicates()
    current2 = current.merge(for_spec , on = ['ID'], how = 'left')
    keep_spec = current2[current2['Gap_End_Year'] <= current2['Job_Start_Year']].drop_duplicates(subset=['ID', 'Gap_Start_Year', 'Gap_End_Year'], keep = 'first')
    keep_spec = keep_spec.drop_duplicates(subset = ['ID','Job_Start_Year', 'Job_End_Year'], keep = 'last').rename(columns={'Gap_Start_Year' : start_col_list[i],
                                                                                                                             'Gap_End_Year' : end_col_list[i]})
    keep_spec[reason_str_list[i]] = "Yes"
    keep_spec = keep_spec[['ID','Job_Start_Year', 'Job_End_Year', start_col_list[i], end_col_list[i], reason_str_list[i]]]

    df_add_gap4 = df_add_gap4.merge(keep_spec, on = ['ID','Job_Start_Year', 'Job_End_Year'], how = 'left')


for i in (list(range(0,len(df_add_gap4)))) :

    for j in range(0,len(reason_list)) : 

        if df_add_gap4['Reason_of_Gap'][i] == reason_list[j] :
            df_add_gap4[reason_str_list[j]][i] = "Yes"
            df_add_gap4[start_col_list[j]][i] = df_add_gap4['Gap_Start_Year'][i]
            df_add_gap4[end_col_list[j]][i] = df_add_gap4['Gap_End_Year'][i]


# ##### 4.6 For each individual, create the column 'Have Gap During Career' 
# To identify if the individual had the gaps in his/her entire career life (except the retirement).

# In[47]:


df_add_gap4['Retired_Year'] = df_add_gap4['Retired_Year'].replace(np.nan, -313)
df_add_gap4['Have_Gap_During_Career'] = df_add_gap4['Is_After_Retirement'].replace("Yes", np.nan)

for i in range(0, len(df_add_gap4)) :
    if df_add_gap4['Number_of_Gaps'][i] == 0:
        df_add_gap4['Have_Gap_During_Career'][i] = "No"
    if (df_add_gap4['Number_of_Gaps'][i] == 1) & (df_add_gap4['Retired_Year'][i] != -313):
        df_add_gap4['Have_Gap_During_Career'][i] = "No"

df_add_gap4['Have_Gap_During_Career'] = df_add_gap4['Have_Gap_During_Career'].replace(np.nan,"Yes")

# update the column 'Gap Before This Job'
for i in range(0, len(df_add_gap4)):
    if (df_add_gap4['Part-time_Gap'][i] == "Yes") | (df_add_gap4['Voluntary_Work_Gap'][i] == "Yes") | (df_add_gap4['Education_Gap'][i] == "Yes") | (df_add_gap4['Family_Reason_Gap'][i] == "Yes") | (df_add_gap4['Sickness_Gap'][i] == "Yes") | (df_add_gap4['Unemployment_Gap'][i] == "Yes") |(df_add_gap4['Other_Reason_Gap'][i] == "Yes"):
        df_add_gap4['Gap_Before_This_Job'][i] = "Yes"


# ##### 4.7 Organize the columns and finalize the job history dataframe 

# In[48]:


col_order2 = ['ID', 'Year_of_Birth', 'Number_of_Jobs_Held', 'Have_Gap_During_Career', 'Number_of_Gaps', 'Retired_Year',
             'Job_Code', 'SOC2000', 'Year', 'Age',  'Job_Start_Year', 'Job_End_Year', 'Index', 
             'Is_After_Retirement', 'Gap_Before_This_Job', 'Gap_Start_Year', 'Gap_End_Year', 'Reason_of_Gap', 'Length_of_Gap',
             'Part-time_Gap', 'P_Start', 'P_End', 'Voluntary_Work_Gap', 'V_Start', 'V_End', 
             'Education_Gap', 'E_Start', 'E_End', 'Family_Reason_Gap', 'F_Start', 'F_End',
             'Sickness_Gap', 'S_Start', 'S_End', 'Unemployment_Gap', 'U_Start', 'U_End',
             'Other_Reason_Gap', 'O_Start', 'O_End']
df_add_gap4 = df_add_gap4[col_order2]


# ### Link SOC2000 with ONET coding

# In[49]:


onet = pd.read_csv('uksoc_onet_all.csv', encoding='windows-1252')


# In[50]:


tolink = onet[['jrecord', 'leadership_d','jdemand_manage_LV']]
tolink = tolink.rename(columns = {'jrecord' : 'SOC2000'})


# In[51]:


df_linked_leadership = df_add_gap4.merge(tolink, on = ['SOC2000'], how = 'left')


# In[52]:


# col_list = onet.columns.tolist()
# select = fnmatch.filter(col_list, '*resp*')
# select


# In[53]:


df_linked_leadership.to_csv('UKB_follow-up_job_leadership2.txt', sep='\t', index=False, header=True)

