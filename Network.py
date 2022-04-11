#Calculate the number of communications between two countries by using the radiation model
import pandas as pd
import numpy as np

df_commuting_flow=pd.read_csv('/.../connected_vehicle/people_code/commuting_flow.csv')
df_commuting_flow=df_commuting_flow.iloc[:,[0,2]]
df_commuting_flow=df_commuting_flow.groupby(['c_id1']).sum()
df_commuting_flow.to_csv('/.../connected_vehicle/radiation_model/driver_number_middle_result.csv')



df_ppl_num=pd.read_csv('/.../connected_vehicle/radiation_model/driver_number_middle_result.csv')
df_county_coordinates=pd.read_csv('/.../connected_vehicle/shortest_path/centroid.csv')

df_ppl_num['lat']=df_ppl_num.apply(lambda z:list(df_county_coordinates.loc[\
               (df_county_coordinates['c_id']==z.c_id1)]['lat'])[0],axis=1)
df_ppl_num['lon']=df_ppl_num.apply(lambda z:list(df_county_coordinates.loc[\
               (df_county_coordinates['c_id']==z.c_id1)]['lon'])[0],axis=1)
df_ppl_num.columns=['c_id','driver_number','lat','lon']
df_ppl_num.to_csv('/.../connected_vehicle/radiation_model/driver_number.csv',index=None)



def haversine(lon1, lat1, lon2, lat2):
    """calculate the distance of two gps records, return unit: meter"""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371
    return c * r * 1000

driver_number=pd.read_csv('/.../connected_vehicle/radiation_model/driver_number.csv')

for pop_i,pop_r in driver_number.iterrows():
    df_cal_result=pd.DataFrame()

    Ti=int(pop_r[1]*14)

    df_cal_result['c_id1']=[int(pop_r[0])]*len(driver_number.index)
    df_cal_result['mi']=[pop_r[1]] * len(driver_number.index)
    df_cal_result['lat1'] = [pop_r[2]] * len(driver_number.index)
    df_cal_result['lon1'] = [pop_r[3]] * len(driver_number.index)

    df_cal_result['c_id2']=list(driver_number['c_id'])
    df_cal_result['nj']=list(driver_number['driver_number'])
    df_cal_result['lat2']=list(driver_number['lat'])
    df_cal_result['lon2']=list(driver_number['lon'])

    df_cal_result=df_cal_result.loc[(df_cal_result['c_id1']!=df_cal_result['c_id2'])]

    df_cal_result['distance']=haversine(df_cal_result['lon1'], df_cal_result['lat1'],\
                                        df_cal_result['lon2'], df_cal_result['lat2'])

    df_Sij=df_cal_result.loc[:,['nj','distance']].copy()
    df_Sij['Sij']=df_Sij.apply(\
               lambda z:int(df_cal_result.loc[df_cal_result['distance']<=z.distance]['nj'].sum()-z.nj),axis=1)
    df_cal_result['Sij']=list(df_Sij['Sij'].copy())
    df_Sij=0

    df_cal_result['Tij']=Ti*df_cal_result['mi']*df_cal_result['nj']/\
                      ((df_cal_result['mi']+df_cal_result['Sij'])*(df_cal_result['mi']+df_cal_result['nj']+df_cal_result['Sij']))


    df_cal_result=df_cal_result.loc[:,['c_id1','c_id2','Tij']]
    df_cal_result.to_csv('/.../connected_vehicle/radiation_model/result/'+\
                         str(int(pop_r[0]))+'.csv',index=None)





#Calculate the number of friend connections between two countries
import os
import math

ppl_code=pd.read_csv('/.../connected_vehicle/people_code/people_code.csv')
ppl_num_coordinate=pd.read_csv('/.../connected_vehicle/radiation_model/driver_number.csv')
cal_result_all=pd.DataFrame(columns=['c_id1','c_id2'])
h=1
for csc_i,csc_r in ppl_num_coordinate.iterrows():
    cal_result=pd.DataFrame()

    cal_result['c_id1']=[int(csc_r[0])]*len(ppl_num_coordinate.index)

    cal_result['c_id2']=list(ppl_num_coordinate['c_id'])

    cal_result=cal_result.loc[(cal_result['c_id1']<cal_result['c_id2'])]
    cal_result_all=pd.concat([cal_result_all,cal_result])
    h+=1
cal_result_all=cal_result_all.reset_index(drop=True)


files=os.listdir('/.../connected_vehicle/xin3/radiation_model/result')
z_flow_dic={}
for path in files:
    z_flow_dic[path]=pd.read_csv('/.../connected_vehicle/xin3/radiation_model/result/'+path)


len_cal_result_all=len(cal_result_all)
aver_flow=[]
for csc_i,csc_r in cal_result_all.iterrows():

    df_z_flow1=z_flow_dic[str(int(csc_r[0])) + '.csv']
    z_flow1=list(df_z_flow1.loc[(df_z_flow1['c_id2']==int(csc_r[1]))]['Tij'])[0]



    df_z_flow2=z_flow_dic[str(int(csc_r[1]))+ '.csv']
    z_flow2=list(df_z_flow2.loc[(df_z_flow2['c_id2']==int(csc_r[0]))]['Tij'])[0]


    aver_flow+=[(z_flow1+z_flow2)/2]
cal_result_all['aver_flow']=aver_flow


ppl_Num=int(ppl_code['id_end'].max)
all_flow=cal_result_all['aver_flow'].sum()
cal_result_all['fr_rate']=cal_result_all['aver_flow']/all_flow
cal_result_all['fr_num_t']=ppl_Num*7*cal_result_all['fr_rate']
cal_result_all['fr_num_y']=cal_result_all['fr_num_t'].apply(lambda z:\
                           int(np.random.choice([1, 0],p=[z,1-z])) if z<1 else\
                           int(math.floor(z)+np.random.choice([1, 0],p=[z-math.floor(z),1-(z-math.floor(z))])))
cal_result_all.to_csv('/.../connected_vehicle/friends_network/fr_num.csv',index=None)

h_df=cal_result_all[['c_id2','c_id1','aver_flow','fr_rate','fr_num_t','fr_num_y']].copy()
h_df.columns=['c_id1','c_id2','aver_flow','fr_rate','fr_num_t','fr_num_y']
h_df.to_csv('/.../connected_vehicle/friends_network/fr_num_reverse.csv',index=None)






import random
ppl_code=pd.read_csv('/.../connected_vehicle/people_code/people_code.csv')
con_county=ppl_code.copy()
fr_N=pd.read_csv('/.../connected_vehicle/friends_network/fr_num.csv')



ppl_code=pd.read_csv('/.../connected_vehicle/people_code/people_code.csv')
for c_id in set(ppl_code['c_id1']):
    os.mkdir('/.../connected_vehicle/friends_network/reverse_fr_net/' + str(c_id))


def find_ppl_num(c_id):
    c_df = ppl_code.loc[(ppl_code['c_id1'] == c_id)]
    c_ppl_all = c_df['flow'].sum()
    return  c_ppl_all

def find_start_id(c_id):
    start_id = int(ppl_code.loc[(ppl_code['c_id1'] == c_id)]['id_start'].min())
    return  start_id

def find_friends_outer(c_id1,c_id2,friends_num):

    ppl_num1=find_ppl_num(c_id1)
    ppl_num2=find_ppl_num(c_id2)

    start_id1=find_start_id(c_id1)
    start_id2=find_start_id(c_id2)

    fr_net=pd.DataFrame()
    fr_net['random']=random.sample(range(int(ppl_num1 * ppl_num2)), friends_num)
    fr_net['first']=(start_id1 + fr_net['random'] / ppl_num2).astype(int)
    fr_net['second']=(start_id2 + fr_net['random'] % ppl_num2).astype(int)

    fr_net = fr_net.iloc[:, 1:]

    fr_net_p = fr_net.copy()
    fr_net_r = fr_net.copy()
    fr_net_r.to_csv('/.../connected_vehicle/friends_network/reverse_fr_net/' + \
                      str(c_id2) + '/' + str(c_id2) + '_' + str(c_id1) + '.csv', index=None)
    return fr_net_p


def find_friends_inter(c_id):
    start_id=find_start_id(c_id)
    c_ppl_all=find_ppl_num(c_id)
    fr_net=pd.DataFrame(columns=['random','first','second'])
    friends_num=c_ppl_all*3
    al_num=0
    count=0
    while al_num!=friends_num:
        fr_net_temporary = pd.DataFrame(columns=['random', 'first', 'second'])
        fr_net_temporary['random']=random.sample(range(int(c_ppl_all*c_ppl_all)),int(friends_num-al_num))
        fr_net_temporary['first']=(start_id+fr_net_temporary['random']/c_ppl_all).astype(int)
        fr_net_temporary['second']=(start_id+fr_net_temporary['random']%c_ppl_all).astype(int)
        fr_net_temporary=fr_net_temporary.loc[fr_net_temporary['second']>fr_net_temporary['first']]
        fr_net=pd.concat([fr_net,fr_net_temporary])
        fr_net=fr_net.drop_duplicates()
        al_num=len(fr_net.index)
        count+=1
    fr_net = fr_net.iloc[:, 1:]
    h_df=fr_net[['second','first']].copy()
    h_df.columns=['first','second']
    fr_net=pd.concat([fr_net,h_df])
    return fr_net



h=1
for county in list(set(ppl_code['c_id1'])):
    county_df = fr_N.loc[(fr_N['c_id1']==county)]
    county_df=county_df.reset_index(drop=True)
    fr_net=pd.DataFrame(columns=['first','second'])
    len_county_df=len(county_df.index)

    for cf_i,cf_r in county_df.iterrows():
        one_fr_net = find_friends_outer(int(cf_r[0]), int(cf_r[1]),int(cf_r[5]))
        fr_net = pd.concat([fr_net, one_fr_net])


    one_fr_net = find_friends_inter(county)
    fr_net = pd.concat([fr_net, one_fr_net])

    fr_net.to_csv('/.../connected_vehicle/friends_network/random_fr_net/' + \
                       str(int(county)) + '.csv', index=None)
    h+=1





ppl_code=pd.read_csv('/.../connected_vehicle/people_code/people_code.csv')
fr_N=pd.read_csv('/.../connected_vehicle/friends_network/fr_num_reverse.csv')
h=1
for county in list(set(ppl_code['c_id1'])):
    county_df = fr_N.loc[(fr_N['c_id1']==county)]
    county_df=county_df.reset_index(drop=True)
    len_county_df=len(county_df.index)
    fr_net=pd.read_csv('/.../connected_vehicle/friends_network/random_fr_net/'+\
                       str(int(county)) + '.csv')
    for cf_i, cf_r in county_df.iterrows():
        fr_net_fz=pd.read_csv('/.../connected_vehicle/friends_network/reverse_fr_net/'+\
                             str(int(cf_r[0]))+'/'+str(int(cf_r[0]))+'_'+str(int(cf_r[1]))+'.csv')
        fr_net_h = pd.DataFrame()
        fr_net_h['first'] = list(fr_net_fz['second'])
        fr_net_h['second'] = list(fr_net_fz['first'])
        fr_net=pd.concat([fr_net,fr_net_h])
    fr_net['first']=fr_net['first'].astype(int)
    fr_net['second'] = fr_net['second'].astype(int)
    fr_net.to_csv('/.../connected_vehicle/friends_network/random_h/'+\
                       str(int(county)) + '.csv',index=None)
    h+=1





files=os.listdir('/.../connected_vehicle/friends_network/random_h')

chu_file=pd.DataFrame(columns=['first','second'])
h=1
for i in range(1,7):
    if i==6:
        file_List=files[500*(i-1):]
    else:
        file_List=files[500*(i-1):500*i]
    for file in file_List:
        pin_df = pd.read_csv('/.../connected_vehicle/friends_network/random_h/' + file)
        chu_file = pd.concat([chu_file, pin_df])
        h += 1
    chu_file.to_csv('/.../connected_vehicle/friends_network/friends_network/fr_net'+str(i)+'.csv', index=None)




#Calculate the synchronization friend relationship network for each time window
sigma = #The value of sigma needs to be entered (The value of sigma is equal to 0 or 10 or 30 or 120)

for n in range(1,7):
	df = pd.read_csv('/.../connected_vehicle/friends_network/friends_network/fr_net'+str(n)+'.csv')
	for tw in range(1,144*28+1):
		df_move_wai_all = pd.read_csv('/.../connected_vehicle/move_vehicle/external_move_id_sigma'+ str(sigma) +'/' + str(tw) + '.csv')
		df_move_nei_all = pd.read_csv('/.../connected_vehicle/move_vehicle/internal_move_id_sigma'+ str(sigma) +'/' + str(tw) + '.csv')
		df_move_all = pd.concat([df_move_wai_all,df_move_nei_all])[['id']]
		list_mid = list(df_move_all['id'])
		df_tw = df[(df['first'].isin(set(list_mid)))&(df['second'].isin(set(list_mid)))]
		df_tw.to_csv('/.../connected_vehicle/friends_network/connected_network/sigma='+ str(sigma) +'/'+str(n)+'/'+str(tw)+'.csv',index=False)



