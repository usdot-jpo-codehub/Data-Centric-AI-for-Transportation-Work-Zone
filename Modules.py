import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics as sta
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import time
import random
import pickle
import os
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import math


def data_cleaning(data):

    data = data.rename(columns={'description':'severity'})

    data.loc[(data['Standardized Type']=='Collision')&(data['severity'].isna()),'severity'] = 'Collision'
    data.loc[(data['Standardized Type']=='Injuries Involved')&(data['severity'].isna()),'severity'] = 'Injuries Involved'
    data.loc[data['severity']=='Collision','severity'] = 'Collision, Property Damage' 
    data.loc[data['severity']=='Injuries Involved','severity'] = 'Collision, Personal Injury'
    data = data.drop(columns=['Standardized Type'])

    data['inci_exist'] = 0
    # Create a new column B with values based on conditions in column A
    data['inci_exist'] = data['severity'].notna().astype(int)
    data['start_time'] = pd.to_datetime(data['start_time'], format='%Y/%m/%d %H:%M')
    data['end_time'] = pd.to_datetime(data['end_time'])
    data['measurement_tstamp'] = pd.to_datetime(data['measurement_tstamp'])
    data['t2srt'] = (data['measurement_tstamp'] - data['start_time']).dt.total_seconds() / 3600
    data['t2end'] = (data['measurement_tstamp'] - data['end_time']).dt.total_seconds() / 3600
    data['TimeOfDay'] = data['measurement_tstamp'].dt.hour
    data['duration'] = (data['end_time'] - data['start_time']).dt.total_seconds() / 3600

    # remove very short and very long work zones
    data = data[(data['duration']>1)&(data['duration']<12)]
    # add progress feature to indicate the time position during a work zone
    data['progress'] = ((data['measurement_tstamp'] - data['start_time']).dt.total_seconds() / 3600)/((data['end_time'] - data['start_time']).dt.total_seconds() / 3600)
    data['wz_during'] = 0
    data.loc[(data['progress']>=0)&(data['progress']<=1),'wz_during'] = 1
    data = data[data['t2srt']>=-12]

    # find odds ids (same set of tmc and start time but appears in different sequences) 
    data_uniIdOrd = data[['tmc_code','start_time','order']].drop_duplicates().reset_index(drop=True)
    data_uniIdOrd = data_uniIdOrd.sort_values(by=['tmc_code', 'start_time','order']).reset_index(drop=True)
    check_dup_order = data_uniIdOrd.groupby(['tmc_code', 'start_time']).size().reset_index(name='group_size')
    odd_id_strt = check_dup_order[check_dup_order['group_size']>1].reset_index(drop=True)
    odd_ids = odd_id_strt['tmc_code'].unique()
    clean_data = data[~data['tmc_code'].isin(odd_ids)]
    clean_data_id_strt_ord = clean_data[['tmc_code','start_time','order']].drop_duplicates().reset_index(drop=True)
    clean_data_id_strt_ord['sampleID'] = clean_data_id_strt_ord.index
    data_w_sampleid = pd.merge(data, clean_data_id_strt_ord[['tmc_code','start_time','sampleID']], on=['tmc_code', 'start_time'], how='left')
    data_clean = data_w_sampleid[data_w_sampleid['sampleID']>=0].reset_index(drop=True)
    data_clean['temp_avg_spd'] = data_clean.groupby('sampleID')['speed'].transform('mean')
    data_clean['sampleID'] = data_clean['sampleID'].astype(int)
    print(len(data_clean))
    # assign work zone numerical ids based on the string ids
    wz_id_df = pd.DataFrame({'id':data_clean['id'].unique(),'wz_id':range(len(data_clean['id'].unique()))})
    data_clean = pd.merge(data_clean,wz_id_df,on='id')
    print(len(data_clean))
    # pick out samples with order 0 or -1, i.e., workzone onlink or workzone is one link downstream
    clean_sampleid_nearWZ = clean_data_id_strt_ord[clean_data_id_strt_ord['order'].isin([0,-1,-2,-3,-4,-5])]['sampleID'].values

    # remove very short and very long work zones
    data = data[(data['duration']>1)&(data['duration']<12)]
    # add progress feature to indicate the time position during a work zone
    data['progress'] = ((data['measurement_tstamp'] - data['start_time']).dt.total_seconds() / 3600)/((data['end_time'] - data['start_time']).dt.total_seconds() / 3600)
    data['wz_during'] = 0
    data.loc[(data['progress']>=0)&(data['progress']<=1),'wz_during'] = 1
    data = data[data['t2srt']>=-12]

    # select sample with congestion during work zone
    data_clean['congestion'] = 0
    data_clean['inci_wz'] = 0
    data_clean.loc[(data_clean['speed']<(0.75*data_clean['average_speed']))&(data_clean['speed']<(0.75*data_clean['temp_avg_spd']))&(data_clean['wz_during']==1)&(data_clean['average_speed']>65),'congestion'] = 1
    data_clean.loc[(data_clean['speed']<(0.65*data_clean['average_speed']))&(data_clean['speed']<(0.65*data_clean['temp_avg_spd']))&(data_clean['wz_during']==1)&(data_clean['average_speed']>30)&(data_clean['average_speed']<=65),'congestion'] = 1
    data_clean.loc[(data_clean['wz_during']==1)&(data_clean['inci_exist']==1),'inci_wz'] = 1

    ## Ensure all samples have work zone timeframes
    # make sure all data has at leas one wz_during = 1
    wz_during_cnt = data_clean.groupby('sampleID')['wz_during'].sum()
    id_no_wz_during = wz_during_cnt[wz_during_cnt == 0].index.values
    # make sure clean data doesn't have nan values in volume
    nan_rows_df = data_clean[data_clean['volume'].isna()]
    # Extract values of column A from the rows where B has NaN values
    sampleID_nanvol = nan_rows_df['sampleID'].unique()

    data_clean = data_clean[~data_clean['sampleID'].isin(sampleID_nanvol)].reset_index(drop=True)
    data_clean = data_clean[~data_clean['sampleID'].isin(id_no_wz_during)].reset_index(drop=True)
    data_clean_nearWZ = data_clean[data_clean['sampleID'].isin(clean_sampleid_nearWZ)].reset_index(drop=True)
    
    return data_clean_nearWZ






def encode_categorical_features(data_1):
    # re-encode order
    data_1.loc[data_1['order']==-5, 'order'] = 6
    data_1.loc[data_1['order']==-4, 'order'] = 5
    data_1.loc[data_1['order']==-3, 'order'] = 4
    data_1.loc[data_1['order']==-2, 'order'] = 3
    data_1.loc[data_1['order']==-1, 'order'] = 2
    data_1.loc[data_1['order']==0, 'order'] = 1
    # adjust progress values
    data_1.loc[(data_1['progress']<0),'progress'] = 0
    data_1 = data_1[data_1['progress']<=1].reset_index(drop=True)
    data_1 = data_1.drop(['confidence_score', 'cvalue','end_time','time_start_record',
                                'time_end_record','id','t2srt','t2end',  'travel_time_minutes',
                                'incident_id'], axis=1)

    direction_labels = ['North', 'South', 'West', 'East', 'Inner Loop', 'Outer Loop']
    dayofweek_labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    roadtype_labels = ['State Road', 'US Highways', '(Intercontinental Highway) Expressway']
    inci_labels = ['Collision, Property Damage','Collision, Personal Injury','Collision, Fatality']
    # encode direction labels
    for i in range(len(direction_labels)):
        data_1.loc[data_1['direction']==direction_labels[i],'direction'] = i+1
    # encode day of week labels
    for i in range(len(dayofweek_labels)):
        if i <= 4:
            data_1.loc[data_1['week']==dayofweek_labels[i],'week'] = 1
        else:

            data_1.loc[data_1['week']==dayofweek_labels[i],'week'] = 2
    # encode sholder_closure_count and traffic_lane_closure_count labels
    for i in range(2,-1,-1):
        if i<3:
            data_1.loc[data_1['traffic_lane_closure_count']==i,'traffic_lane_closure_count'] = i+1
        else:
            data_1.loc[data_1['traffic_lane_closure_count']>=i,'traffic_lane_closure_count'] = 4
    # encode sholder_closure_count and traffic_lane_closure_count labels
    for i in range(1,-1,-1):
        if i<1:
            data_1.loc[data_1['sholder_closure_count']==i,'sholder_closure_count'] = 1
        else:
            data_1.loc[data_1['sholder_closure_count']>=i,'sholder_closure_count'] = 2
    # encode wz_during and congestion labels
    for i in range(1,-1,-1):
        data_1.loc[data_1['wz_during']==i,'wz_during'] = i+1
        data_1.loc[data_1['congestion']==i,'congestion'] = i+1
        data_1.loc[data_1['inci_exist']==i,'inci_exist'] = i+1
        data_1.loc[data_1['inci_wz']==i,'inci_wz'] = i+1
        data_1.loc[data_1['on ramp']==i,'on ramp'] = i+1
        data_1.loc[data_1['off ramp']==i,'off ramp'] = i+1
    # encode road type labels
    for i in range(len(roadtype_labels)):
        data_1.loc[data_1['road_type'] == roadtype_labels[i],'road_type'] = i+1
    # encode TimeofDay labels
    for i in range(23,-1,-1):
        data_1.loc[data_1['TimeOfDay'] == i,'TimeOfDay'] = i+1
    # encode incident severity labels
    data_1['severity'] = data_1['severity'].fillna(1)
    for i in range(len(inci_labels)):
        if i == 0:
            data_1.loc[data_1['severity'] == inci_labels[i],'severity'] = 2
        else:
            data_1.loc[data_1['severity'] == inci_labels[i],'severity'] = 3
    # re-encode lanes
    data_1.loc[data_1['lanes']>=5, 'lanes'] = 5
    return data_1


def split_train_val_test_workzones(data_1,impact_wzid,high_impact_wzid,inci_wzID):
    wz_features = data_1[['wz_id','order','direction','week', 'congestion','road_type','on ramp','off ramp']].drop_duplicates().reset_index(drop=True)
    wz_features = wz_features[wz_features['order'] == 1].reset_index(drop=True)
    # Assuming df is your DataFrame
    wz_features = wz_features.drop_duplicates(subset=['wz_id'], keep='last').reset_index(drop=True)
    valid_wzid = wz_features['wz_id'].values

    # Set the seed value for reproducibility
    random_seed = 142  # Replace this with any seed value you prefer
    # Set the value of 'k' (number of values to be selected)
    train_size = int(len(valid_wzid)*0.7)
    val_size = int(len(valid_wzid)*0.15)
    id_all = pd.Series(valid_wzid)
    # Set the random seed and randomly select k values from the specified column
    id_train_wz = list(id_all.sample(n=train_size, replace=False, random_state=random_seed))
    id_val_wz = list(id_all[~id_all.isin(id_train_wz)].sample(n=val_size, replace=False, random_state=random_seed))
    id_test_wz = list(id_all[(~id_all.isin(id_train_wz))&(~id_all.isin(id_val_wz))])
    print('train:val:test WZ size: ',len(id_train_wz),len(id_val_wz),len(id_test_wz))

    train_high_impact_inter = list(set(id_train_wz).intersection(set(high_impact_wzid)))
    val_high_impact_inter = list(set(id_val_wz).intersection(set(high_impact_wzid)))
    test_high_impact_inter = list(set(id_test_wz).intersection(set(high_impact_wzid)))
    print('train:val:test &high_impact size: ',len(train_high_impact_inter),len(val_high_impact_inter),
         len(test_high_impact_inter))
    print(len(id_train_wz)/len(train_high_impact_inter),len(id_val_wz)/len(val_high_impact_inter),
          len(id_test_wz)/len(test_high_impact_inter))

    train_impact_inter = list(set(id_train_wz).intersection(set(impact_wzid)))
    val_impact_inter = list(set(id_val_wz).intersection(set(impact_wzid)))
    test_impact_inter = list(set(id_test_wz).intersection(set(impact_wzid)))
    print('train:val:test &impact size: ',len(train_impact_inter),len(val_impact_inter),
          len(test_impact_inter))
    print(len(id_train_wz)/len(train_impact_inter),len(id_val_wz)/len(val_impact_inter),
          len(id_test_wz)/len(test_impact_inter))

    # select 1000 unimpact samples for training
    # result = [x for x in A if x not in B and x not in C]
    id_train_wz_umimpact = [x for x in id_train_wz if x not in impact_wzid and x not in high_impact_wzid]
    # Set the random seed (replace 42 with any integer you want)
    random.seed(42)
    # Randomly select N values from the list
    selected_id_train_wz_umimpact = random.sample(id_train_wz_umimpact, len(id_train_wz_umimpact))
    selected_id_train_wz_combine = selected_id_train_wz_umimpact + list(set(id_train_wz).intersection(set(impact_wzid)))

    expand_rate_cong = len(selected_id_train_wz_combine)/len(list(set(selected_id_train_wz_combine).intersection(set(impact_wzid))))
    expand_rate_inci = len(selected_id_train_wz_combine)/len(list(set(selected_id_train_wz_combine).intersection(set(inci_wzID))))
    print('expand_rate_cong: '+str(expand_rate_cong)+', expand_rate_inci: '+str(expand_rate_inci))

    inci_impact_inter = list(set(inci_wzID).intersection(set(impact_wzid)))
    print('number of inci wz in all WZ:', len(inci_wzID))
    print('number of inci wz in impact wz:', len(inci_impact_inter))

return valid_wzid, id_train_wz, id_val_wz, id_test_wz, selected_id_train_wz_combine


def create_sub_sequence(wzids,expand_rate,dataset_type):  
    tic = time.perf_counter()
    for i in range(len(wzids)):
        wzid = wzids[i]
        data_thiswz = data_1[data_1['wz_id'] == wzid]
        sample_orders = np.sort(data_thiswz['order'].unique())
        for j in range(len(sample_orders)):
            if j == 0:
                df_wz_allsam = data_thiswz[data_thiswz['order']==sample_orders[j]][['measurement_tstamp','sampleID','wz_id','speed','average_speed','wz_during',
                                                                                    'sholder_closure_count','traffic_lane_closure_count','TimeOfDay',
                                                                                   'duration','volume','miles','distance_to_work_zone','inci_wz','lanes','lane_closure_rate']]
                df_wz_allsam = df_wz_allsam.rename(columns = {'speed':'speed_1','average_speed':'avgspd_1','miles':'length_1','distance_to_work_zone':'distance_1','inci_wz':'inci_1','volume':'volume_1'})
                df_wz_allsam['speed_2'],df_wz_allsam['speed_3'],df_wz_allsam['speed_4'],df_wz_allsam['speed_5'],df_wz_allsam['speed_6'] = 0,0,0,0,0
                df_wz_allsam['length_2'],df_wz_allsam['length_3'],df_wz_allsam['length_4'],df_wz_allsam['length_5'],df_wz_allsam['length_6'] = 0,0,0,0,0
                df_wz_allsam['distance_2'],df_wz_allsam['distance_3'],df_wz_allsam['distance_4'],df_wz_allsam['distance_5'],df_wz_allsam['distance_6'] = 0,0,0,0,0
                df_wz_allsam['inci_2'],df_wz_allsam['inci_3'],df_wz_allsam['inci_4'],df_wz_allsam['inci_5'],df_wz_allsam['inci_6'] = 0,0,0,0,0
                df_wz_allsam['avgspd_2'],df_wz_allsam['avgspd_3'],df_wz_allsam['avgspd_4'],df_wz_allsam['avgspd_5'],df_wz_allsam['avgspd_6'] = 0,0,0,0,0
                df_wz_allsam['volume_2'],df_wz_allsam['volume_3'],df_wz_allsam['volume_4'],df_wz_allsam['volume_5'],df_wz_allsam['volume_6'] = 0,0,0,0,0
            else:
                df_wz_thisorder = data_thiswz[data_thiswz['order']==sample_orders[j]][['measurement_tstamp','sampleID','wz_id','speed','average_speed','miles','distance_to_work_zone','inci_wz','volume']]
                df_wz_allsam = pd.merge(df_wz_allsam,df_wz_thisorder[['measurement_tstamp','speed','miles','distance_to_work_zone','inci_wz','average_speed','volume']],on='measurement_tstamp',how = 'left')
                df_wz_allsam['speed_'+str(sample_orders[j])] = df_wz_allsam['speed'].values
                df_wz_allsam['length_'+str(sample_orders[j])] = df_wz_allsam['miles'].values
                df_wz_allsam['distance_'+str(sample_orders[j])] = df_wz_allsam['distance_to_work_zone'].values
                df_wz_allsam['inci_'+str(sample_orders[j])] = df_wz_allsam['inci_wz'].values
                df_wz_allsam['avgspd_'+str(sample_orders[j])] = df_wz_allsam['average_speed'].values
                df_wz_allsam['volume_'+str(sample_orders[j])] = df_wz_allsam['volume'].values
                df_wz_allsam = df_wz_allsam.drop(columns = ['speed','miles','distance_to_work_zone','inci_wz','average_speed','volume'])
        df_wz_allsam = pd.merge(df_wz_allsam,wz_features[wz_features['wz_id']==wzid],on = 'wz_id', how = 'left')
        df_wz_allsam['inci_occ'] = 1
        df_wz_allsam.loc[(df_wz_allsam['inci_1']==2)|(df_wz_allsam['inci_2']==2)|(df_wz_allsam['inci_3']==2)|(df_wz_allsam['inci_4']==2)|
                         (df_wz_allsam['inci_5']==2)|(df_wz_allsam['inci_6']==2),'inci_occ'] = 2

        # create sub-seqeunces
        df = df_wz_allsam.drop(['order', 'measurement_tstamp', 'sampleID','wz_id','inci_2','inci_3','inci_4','inci_5','inci_6'], axis=1).copy()
        df = df[['speed_1', 'speed_2','speed_3', 'speed_4', 'speed_5','speed_6', 
                 'volume_1', 'volume_2', 'volume_3','volume_4','volume_5','volume_6',
                 'avgspd_1','avgspd_2','avgspd_3','avgspd_4','avgspd_5','avgspd_6',
                 'length_1','length_2','length_3','length_4','length_5','length_6',
                 'distance_1', 'distance_2', 'distance_3','distance_4','distance_5','distance_6',
                 'duration','TimeOfDay','lane_closure_rate',
                 'lanes','wz_during','direction', 'sholder_closure_count', 'traffic_lane_closure_count','week', 'congestion', 'road_type','on ramp','off ramp']]
        df = df.fillna(df.shift().add(df.shift(-1)).div(2))
        # select sequence from  hour before work zone to the end
        df = df[df['wz_during']==2].reset_index(drop=True)
        if df.isna().values.any():
            continue
        spd_sum = df.iloc[:,:6].sum()
        fill_df = df.copy()
        if (spd_sum == 0).any():
            if spd_sum['speed_1']==0 and spd_sum['speed_2']>0:
                fill_df[fill_df.columns[[0,6,12,18,24]]] = df.iloc[:,[1,7,13,19,25]].values
            elif spd_sum['speed_2']==0 and spd_sum['speed_1']>0 and spd_sum['speed_3']>0:
                fill_df[fill_df.columns[[1,7,13,19,25]]] = (df.iloc[:,[0,6,12,18,24]].values + df.iloc[:,[2,8,14,20,26]].values)/2
            elif spd_sum['speed_3']==0 and spd_sum['speed_2']>0 and spd_sum['speed_4']>0:
                fill_df[fill_df.columns[[2,8,14,20,26]]] = (df.iloc[:,[1,7,13,19,25]].values + df.iloc[:,[3,9,15,21,27]].values)/2
            elif spd_sum['speed_4']==0 and spd_sum['speed_3']>0 and spd_sum['speed_5']>0:
                fill_df[fill_df.columns[[3,9,15,21,27]]] = (df.iloc[:,[2,8,14,20,26]].values + df.iloc[:,[4,10,16,22,28]].values)/2
            elif spd_sum['speed_5']==0 and spd_sum['speed_4']>0 and spd_sum['speed_6']>0:
                fill_df[fill_df.columns[[4,10,16,22,28]]] = (df.iloc[:,[3,9,15,21,27]].values + df.iloc[:,[5,11,17,23,29]].values)/2
            elif spd_sum['speed_6']==0 and spd_sum['speed_5']>0:
                fill_df[fill_df.columns[[5,11,17,23,29]]] = df.iloc[:,[4,10,16,22,28]].values
        # print(fill_df.columns)
        if dataset_type=='test' or dataset_type=='val':
            if i == 0:
                df_forconcat = fill_df.copy()
                df_forconcat['wz_id'] = wzid
                df_all = df_forconcat.copy()
            else:
                df_forconcat = fill_df.copy()
                df_forconcat['wz_id'] = wzid
                df_all = pd.concat([df_all, df_forconcat], axis=0)
                df_all = df_all.reset_index(drop=True)
        input_array_1 = fill_df[['volume_1', 'volume_2', 'volume_3','volume_4','volume_5','volume_6',
                             'avgspd_1','avgspd_2','avgspd_3','avgspd_4','avgspd_5','avgspd_6',
                             'length_1','length_2','length_3','length_4','length_5','length_6',
                             'distance_1', 'distance_2', 'distance_3','distance_4','distance_5','distance_6']].values


        # Create a numpy array with these values
        input_array_2 = np.array([fill_df['duration'].min(),fill_df['TimeOfDay'].min(), fill_df['lane_closure_rate'].max(),
                                  fill_df['lanes'].max(), fill_df['direction'].min(),fill_df['week'].min(),fill_df['road_type'].min(),
                                 fill_df['sholder_closure_count'].max(),fill_df['traffic_lane_closure_count'].max(),fill_df['on ramp'].max(),fill_df['off ramp'].max(),fill_df['congestion'].max()])
        # print(input_array_2.shape[0])

        
        output_array_1 = fill_df[['speed_1','speed_2','speed_3','speed_4','speed_5','speed_6']].values
        # output_array_2 = np.array([fill_df['inci_occ'].max()])

        window_length = 4
        pic_length = 48
        if len(input_array_1)<= window_length or len(input_array_1)> pic_length:
            continue
        sequences = []
        targets_1 = []
        targets_2 = []

        #pad input
        pad_amount_in = ((0, pic_length-input_array_1.shape[0]), (0, 0))
        input_array_1 = np.pad(input_array_1, pad_amount_in, mode='constant', constant_values=0)
        # pad target
        pad_amount_out = ((0, pic_length-output_array_1.shape[0]), (0, 0))
        output_array_1 = np.pad(output_array_1, pad_amount_out, mode='constant', constant_values=0)
        
        input_array_1 = input_array_1.reshape(1, input_array_1.shape[0],input_array_1.shape[1])
        input_array_2 = input_array_2.reshape(1, len(input_array_2))
        output_array_1 = output_array_1.reshape(1, output_array_1.shape[0],output_array_1.shape[1])
        
        high_impact_expand_rate = 12
        if i == 0 or ('train_X1' not in locals()):
            train_X1, train_X2, train_Y1  = input_array_1.copy(), input_array_2.copy(), output_array_1.copy()
        else:
            if (dataset_type == 'train') and (wzid in impact_wzid) and (wzid not in high_impact_wzid):
                # Duplicate the array for 20 times along the first axis
                input_array_1 = np.tile(input_array_1, (expand_rate, 1, 1))
                # Reshape and concatenate along the first axis
                input_array_1 = input_array_1.reshape(expand_rate, 48, 24)
                # expand for input array 2 and target array
                input_array_2 = np.tile(input_array_2, (expand_rate, 1))
                # print(wzid,input_array_2.shape[0])
                input_array_2 = input_array_2.reshape(expand_rate, input_array_2.shape[1])
                output_array_1 = np.tile(output_array_1, (expand_rate, 1, 1))
                output_array_1 = output_array_1.reshape(expand_rate, 48, 6)
            if (dataset_type == 'train') and (wzid in high_impact_wzid):
                # Duplicate the array for 20 times along the first axis
                input_array_1 = np.tile(input_array_1, (expand_rate*high_impact_expand_rate, 1, 1))
                # Reshape and concatenate along the first axis
                input_array_1 = input_array_1.reshape(expand_rate*high_impact_expand_rate, 48, 24)
                # expand for input array 2 and target array
                input_array_2 = np.tile(input_array_2, (expand_rate*high_impact_expand_rate, 1))
                input_array_2 = input_array_2.reshape(expand_rate*high_impact_expand_rate, input_array_2.shape[1])
                output_array_1 = np.tile(output_array_1, (expand_rate*high_impact_expand_rate, 1, 1))
                output_array_1 = output_array_1.reshape(expand_rate*high_impact_expand_rate, 48, 6)

            train_X1, train_X2, train_Y1 = np.concatenate((train_X1, input_array_1), axis=0),np.concatenate((train_X2, input_array_2), axis=0), np.concatenate((train_Y1, output_array_1), axis=0)
            # print(train_X1.shape)
    # timer ends
    toc = time.perf_counter()
    
    if dataset_type == 'test' or  dataset_type == 'val':
        print(f'Execution time: {(toc-tic):2f}')
        print(train_X1.shape)
        return train_X1, train_X2, train_Y1, df_all
    elif dataset_type == 'single_df_only' :
        return df_all
    else:
        print(f'Execution time: {(toc-tic):2f}')
        print(train_X1.shape)
        return train_X1, train_X2, train_Y1