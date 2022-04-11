import pandas as pd
import numpy as np
import networkx as nx
import csv

def haversine(lon1, lat1, lon2, lat2):
    """calculate the distance of two gps records, return unit: meter"""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371
    return c * r * 1000

#Files 'centroid.csv' and 'neighboring_county_gis.csv' are obtained by ArcGIS after processing file 'no_lake2.shp'
df_centroid = pd.read_csv('/.../connected_vehicle/shortest_path/centroid.csv')#Centroid of counties
df_near_county = pd.read_csv('/.../connected_vehicle/shortest_path/neighboring_county_gis.csv')#Adjacent counties


df_near_county = pd.merge(df_near_county,df_centroid,left_on='c_id1',right_on='c_id')
df_near_county = df_near_county[['c_id1','c_id2','lat','lon']]
df_near_county = df_near_county.rename(columns={'lat':'lat1','lon':'lon1'})
df_near_county = pd.merge(df_near_county,df_centroid,left_on='c_id2',right_on='c_id')
df_near_county = df_near_county[['c_id1','c_id2','lat1','lon1','lat','lon']]
df_near_county = df_near_county.rename(columns={'lat':'lat2','lon':'lon2'})
df_near_county['distance'] = df_near_county.apply(lambda z:haversine(z.lon1,z.lat1,z.lon2,z.lat2),axis=1)
df_near_county.to_csv('/.../connected_vehicle/shortest_path/neighboring_county_lat_lon.csv',index=False)#Coordinates and distance of adjacent counties



#Remove travel data of Alaska, Hawaii, Puerto Rico and from the United States to outside the United States
df_OD=pd.read_excel('/.../connected_vehicle/people_code/table3.xlsx')
#The data in file 'table3.xlsx' is from the U.S. Census Bureau 2011-2015 5-Year ASC Commuting Flows

df_OD.columns=['State FIPS Code','County FIPS Code','Minor Civil Division FIPS Code','State Name',\
               'County Name','Minor Civil Division Name','State FIPS Code2','County FIPS Code2',\
               'Minor Civil Division FIPS Code','State Name','County Name','Minor Civil Division Name',\
               'Workers in Commuting Flow','Margin of Error']
df_OD.dropna(axis=0,subset = ['State FIPS Code','State FIPS Code2',\
                              'County FIPS Code','County FIPS Code2'],inplace=True)
df_OD['State FIPS Code2']=df_OD['State FIPS Code2'].astype('int')
df_OD['County FIPS Code2']=df_OD['County FIPS Code2'].astype('int')
df_OD=df_OD.loc[(df_OD['State FIPS Code'] != 2) & \
                (df_OD['State FIPS Code'] != 15) & \
                (df_OD['State FIPS Code'] != 72) & \
                (df_OD['State FIPS Code2'] != 2) & \
                (df_OD['State FIPS Code2'] != 15)& \
                (df_OD['State FIPS Code2'] != 72)]
df_OD=df_OD.reset_index(drop=True)
all_ppl=df_OD['Workers in Commuting Flow'].sum()
all_nei_ppl=df_OD.loc[(df_OD['State FIPS Code']==df_OD['State FIPS Code2'])&\
                      (df_OD['County FIPS Code']==df_OD['County FIPS Code2'])]['Workers in Commuting Flow'].sum()



drop_list_index=[]
df_OD=df_OD.iloc[:,[0,1,6,7,12]]
df_OD=df_OD.groupby(['State FIPS Code','County FIPS Code',\
               'State FIPS Code2','County FIPS Code2']).sum()
df_OD.to_csv('/.../connected_vehicle/people_code/commuting_flow_middle_result.csv')


df_OD=pd.read_csv('/.../connected_vehicle/people_code/commuting_flow_middle_result.csv')
df_OD.columns = ['state1','county1','state2','county2','flow']
df_OD['c_id1'] = df_OD.apply(lambda z:z.state1*1000+z.county1,axis=1)
df_OD['c_id2'] = df_OD.apply(lambda z:z.state2*1000+z.county2,axis=1)
df_OD = df_OD[['c_id1','c_id2','flow']]

df_ppl_code=df_OD.copy()
id_start=[]
id_end=[]
for i,r in df_ppl_code.iterrows():
    if i==0:
        id_start+=[1]
        id_end+=[r[2]]
    else:
        id_start+=[id_end[-1]+1]
        id_end+=[id_start[-1]+r[2]-1]
df_ppl_code['id_start']=id_start
df_ppl_code['id_end']=id_end

#Number all drivers
with open('/.../connected_vehicle/people_code/all_people.csv', 'a') as schedule:
    writer = csv.writer(schedule)
    list_title = ['id', 'c_id1','c_id2']
    writer.writerow(list_title)
    for i,r in df_ppl_code.iloc[:100,:].iterrows():
        c_id1 = r[0]
        c_id2 = r[1]
        id_start = r[3]
        id_end = r[4]
        for ppl in range(id_start,id_end+1):
            w_row = [ppl,c_id1,c_id2]
            writer.writerow(w_row)





#Build the shortest path network for drivers to travel
df_near_county = pd.read_csv('/.../connected_vehicle/shortest_path/neighboring_county_lat_lon.csv')

df_edge = df_near_county[['c_id1','c_id2','distance']]
df_node = pd.DataFrame(data=list(set(df_edge['c_id1'])),columns=['node'])

list_edge = []
for i in range(0,len(df_edge)):
    list_edge.append([df_edge['c_id1'][i],df_edge['c_id2'][i],df_edge['distance'][i]])
list_node = list(df_node['node'])
G = nx.Graph()
G.add_nodes_from(list_node)
G.add_weighted_edges_from(list_edge)


all_people=pd.read_csv('/.../connected_vehicle/people_code/all_people.csv')
all_people['>200km']=all_people.apply(lambda z:1 if nx.dijkstra_path_length(G,z.c_id1,z.c_id2)>200000 \
                                      else 0,axis=1)
all_people=all_people.loc[all_people['>200km']==0]
all_people=all_people.iloc[:,:3]
all_people.to_csv('/.../connected_vehicle/people_code/all_people_200less.csv',index=None)



all_people_200less=pd.read_csv('/.../connected_vehicle/people_code/all_people_200less.csv')
list_index=[]
for i in range(len(all_people_200less.index)):
    if np.random.choice([0,1],p=[0.67,0.33])==1:
        list_index += [i]
all_people_200less=all_people_200less.loc[list_index,:]
all_people_200less=all_people_200less.iloc[:,:3]
all_people_200less.to_csv('/.../connected_vehicle/people_code/all_people_200less_random.csv',index=None)



all_people_200less_random=pd.read_csv('/.../connected_vehicle/people_code/all_people_200less_random.csv')
all_people_200less_random['flow']=[1]*len(all_people_200less_random.index)
all_people_200less_random=all_people_200less_random.iloc[:,1:]
all_people_200less_random=all_people_200less_random.groupby(['c_id1','c_id2']).sum()
all_people_200less_random.to_csv('/.../connected_vehicle/people_code/commuting_flow.csv')



all_people_200less_random=pd.read_csv('/.../connected_vehicle/people_code/commuting_flow.csv')
id_start=[]
id_end=[]
for i,r in all_people_200less_random.iterrows():
    if i==0:
        id_start+=[1]
        id_end+=[r[2]]
    else:
        id_start+=[id_end[-1]+1]
        id_end+=[id_start[-1]+r[2]-1]
all_people_200less_random['id_start']=id_start
all_people_200less_random['id_end']=id_end
all_people_200less_random.to_csv('/.../connected_vehicle/people_code/people_code.csv',index=None)



import random
ppl_code=pd.read_csv('/.../connected_vehicle/people_code/people_code.csv')
ppl_Num=int(ppl_code['id_end'].max())
for i in [c/100 for c in list(range(1,101))]:
	ppl_list = random.sample(range(1, ppl_Num + 1), round(ppl_Num * i))
	df_ppl = pd.DataFrame()
	df_ppl['ppl'] = ppl_list
	df_ppl.to_csv('/.../connected_vehicle/friends_network/m_id/m='+str(i)+'.csv', index=None)

