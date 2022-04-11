#Generate travel schedules for inner-county commuter
import pandas as pd
import numpy as np
import scipy.stats
import random
import math
import csv


sigma = #The value of sigma needs to be entered (The value of sigma is equal to 0 or 10 or 30 or 120)

df_commuting_flow=pd.read_csv('/.../connected_vehicle/people_code/commuting_flow.csv')
df_commuting_flow=df_commuting_flow.loc[df_commuting_flow['c_id1']==df_commuting_flow['c_id2']]
df_commuting_flow=df_commuting_flow.sort_values(by=['c_id1','c_id2']).reset_index(drop=True)

df_ppl_code=pd.read_csv('/.../connected_vehicle/people_code/people_code.csv')

#The data in file 'departure_time.csv' and 'trip_duration.csv' are from the U.S. Census Bureau
df_departure_time = pd.read_csv('/.../connected_vehicle/move_time/departure_time.csv')#Departure time distribution
df_trip_duration=pd.read_csv('/.../connected_vehicle/move_time/trip_duration.csv')#trip duration distribution

list_starttime_distribution=[]
for p_i,p_r in df_departure_time.iterrows():
    list_starttime_distribution+=[p_r[0]/p_r[1]]*int(p_r[1])
list_durationtime_distribution=[]
for p_i,p_r in df_trip_duration.iterrows():
    list_durationtime_distribution+=[p_r[1]/p_r[2]]*int(p_r[2])


#Generate Gaussian random number
rand_data = list(np.random.normal(0, sigma , 300000))
gossi_random_No=pd.DataFrame()
gossi_random_No['random_No']=rand_data
gossi_random_No=gossi_random_No.loc[(gossi_random_No['random_No']<sigma*3)&(gossi_random_No['random_No']>sigma*3*(-1))]
gossi_random_No.to_csv('/.../connected_vehicle/move_time/gossi_random_No_'+str(sigma)+'m.csv',index=None)
goss_No=list(pd.read_csv('/.../connected_vehicle/move_time/gossi_random_No_'+str(sigma)+'m.csv')['random_No'])



def write_in_data(ppl,list_starttime_distribution,list_durationtime_distribution,c_id):
    tw_m=np.random.choice(list(range(1440)),p=list_starttime_distribution)
    duration_tw=np.random.choice(list(range(1,90)),p=list_durationtime_distribution)
    w_row=[ppl,c_id]
    for week in range(4):
        for i in range(5):
            if i == 0 and week==0:
                dtw=-1
                while dtw<0:
                    randomNo_weekday = round(random.choice(goss_No))
                    dtw = int(tw_m + randomNo_weekday)
                atw = int(dtw + duration_tw)
            else:
                randomNo_weekday = round(random.choice(goss_No))
                dtw = int(tw_m + randomNo_weekday)
                atw = int(dtw + duration_tw)
            w_row += [int(dtw + 1440 * i+ week*10080), int(atw + 1440 * i+week*10080), \
                      int(atw + 1440 * i + week*10080+ 480),
                      int(atw + 1440 * i + week*10080 + 480 + duration_tw )]  # Weekdays

        for i in range(5, 7):
            randomNo_weekend = np.random.choice([0,1],p=[0.2,0.8])  # 80% commute on weekends
            if randomNo_weekend == 0:
                w_row += [-1, -1, -1, -1]
            if randomNo_weekend == 1:
                randomNo_weekday = round(random.choice(goss_No))
                dtw = int(tw_m + randomNo_weekday)
                atw = int(dtw + duration_tw)
                w_row += [int(dtw + 1440 * i + week * 10080), int(atw + 1440 * i + week * 10080), \
                          int(atw + 1440 * i + week * 10080 + 480),\
                          int(atw + 1440 * i + week * 10080 + 480 + duration_tw)]# Weekend
    writer.writerow(w_row)


def find_start_id(c1):
    start_id = int(df_ppl_code.loc[(df_ppl_code['c_id1']==c1)&\
                                 (df_ppl_code['c_id2']==c1),'id_start'].iloc[0])
    return(start_id)

def find_end_id(c1):
    end_id = int(df_ppl_code.loc[(df_ppl_code['c_id1']==c1)&\
                                 (df_ppl_code['c_id2']==c1),'id_end'].iloc[0])
    return(end_id)



list1 = list(set(df_commuting_flow['c_id1']))[:1500]
list2 = list(set(df_commuting_flow['c_id1']))[1500:]


day = 28
for n in [1,2]:
    if n==1:
        list_id = list1
    else:
        list_id = list2
    with open('/.../connected_vehicle/move_time/move_time_internal_sigma'+str(sigma)+'_'+str(n)+'.csv','a') as schedule1:
        writer = csv.writer(schedule1)
        list_title = ['id','c_id']
        for trip in range(1,day*2+1):
            list_title += ['s'+str(trip),'e'+str(trip)]
        writer.writerow(list_title)
        count=1
        for c_id1 in list_id:
            df_county_flow = df_commuting_flow[df_commuting_flow['c_id1']==c_id1]
            for cf_i, cf_r in df_county_flow.iterrows():
                c_id = int(cf_r[0])
                start_id = find_start_id(c_id)
                end_id = find_end_id(c_id)
                for ppl in range(start_id, end_id + 1):
                    write_in_data(ppl,list_starttime_distribution,list_durationtime_distribution,c_id)
                count+=1








#Generate travel schedules for cross-county commuter
import networkx as nx

df_commuting_flow=pd.read_csv('/.../connected_vehicle/people_code/commuting_flow.csv')
df_commuting_flow=df_commuting_flow.loc[df_commuting_flow['c_id1']!=df_commuting_flow['c_id2']]
df_commuting_flow=df_commuting_flow.reset_index(drop=True)

df_ppl_code=pd.read_csv('/.../connected_vehicle/people_code/people_code.csv')

#The data in file 'departure_time.csv' is from the U.S. Census Bureau
df_departure_time =pd.read_csv('/.../connected_vehicle/move_time/departure_time.csv')

#Generate Gaussian random number
goss_No=list(pd.read_csv('/.../connected_vehicle/move_time/gossi_random_No_'+str(sigma)+'m.csv')['random_No'])


list_starttime_distribution=[]
for p_i,p_r in df_departure_time.iterrows():
    list_starttime_distribution+=[p_r[0]/p_r[1]]*int(p_r[1])



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




def write_in_data(ppl,list_starttime_distribution,c_id1,c_id2,duration_tw):
    tw_m=np.random.choice(list(range(1440)),p=list_starttime_distribution)
    w_row=[ppl,c_id1,c_id2]
    for week in range(4):
        for i in range(5):
            if i == 0 and week==0:
                dtw=-1
                while dtw<0:
                    randomNo_weekday = round(random.choice(goss_No))
                    dtw = int(tw_m + randomNo_weekday)
                atw = int(dtw + duration_tw)
            else:
                randomNo_weekday = round(random.choice(goss_No))
                dtw = int(tw_m + randomNo_weekday)
                atw = int(dtw + duration_tw)
            w_row += [int(dtw + 1440 * i+ week*10080), int(atw + 1440 * i+week*10080), \
                      int(atw + 1440 * i + week*10080+ 480),\
                      int(atw + 1440 * i  + week*10080+ 480 + duration_tw)]  # Weekdays

        for i in range(5, 7):
            randomNo_weekend = np.random.choice([0,1],p=[0.2,0.8])  # 80% commute on weekends
            if randomNo_weekend == 0:
                w_row += [-1, -1, -1, -1]
            if randomNo_weekend == 1:
                randomNo_weekday = round(random.choice(goss_No))
                dtw = int(tw_m + randomNo_weekday)
                atw = int(dtw + duration_tw)
                w_row += [int(dtw + 1440 * i + week * 10080), int(atw + 1440 * i + week * 10080), \
                          int(atw + 1440 * i + week * 10080 + 480),\
                          int(atw + 1440 * i + week * 10080 + 480 + duration_tw)]# Weekend
    writer.writerow(w_row)



def find_start_id(c1,c2):
    start_id = int(df_ppl_code.loc[(df_ppl_code['c_id1']==c1)&\
                                 (df_ppl_code['c_id2']==c2),'id_start'].iloc[0])
    return(start_id)

def find_end_id(c1,c2):
    end_id = int(df_ppl_code.loc[(df_ppl_code['c_id1']==c1)&\
                                 (df_ppl_code['c_id2']==c2),'id_end'].iloc[0])
    return(end_id)



day = 28
with open('/.../connected_vehicle/move_time/move_time_external_sigma'+str(sigma)+'.csv','a') as schedule1:
    writer = csv.writer(schedule1)
    list_title = ['id', 'c_id1','c_id2']
    for trip in range(1, day * 2 + 1):
        list_title += ['s' + str(trip), 'e' + str(trip)]
    writer.writerow(list_title)


    for c_id1 in list(set(df_commuting_flow['c_id1'])):
        df_county_flow=df_commuting_flow.loc[df_commuting_flow['c_id1']==c_id1]
        for sf_i, sf_r in df_county_flow.iterrows():
            start_id = find_start_id(int(sf_r[0]),int(sf_r[1]))
            end_id = find_end_id(int(sf_r[0]),int(sf_r[1]))
            c_id2 = int(sf_r[1])
            duration_tw = math.ceil((nx.dijkstra_path_length(G, c_id1, c_id2) / 100000) * 60)
            for ppl in range(start_id,end_id+1):
                write_in_data(ppl,list_starttime_distribution,c_id1,c_id2,duration_tw)





#Find out which inner-county commuters are driving at each time window
import pandas as pd

for n in [1,2]:
	df = pd.read_csv('/.../connected_vehicle/move_time/move_time_internal_sigma'+str(sigma)+'_'+str(n)+'.csv')
	list_min = list()
	list_max = list()
	for trip in range(1,57):
		if ((trip>=11)&(trip<=14))|((trip>=25)&(trip<=28))|((trip>=39)&(trip<=42))|((trip>=53)&(trip<=56)):
			df_fz = df[df['s'+str(trip)]!=-1]
			list_min.append(df_fz['s'+str(trip)].min())
		else:
			list_min.append(df['s'+str(trip)].min())
		list_max.append(df['e'+str(trip)].max())
	df_se = pd.DataFrame()
	df_se['s'] = list_min
	df_se['e'] = list_max
	df_se['trip'] = list(range(1,57))

	def select_start_in(tw):
		list_trip = list()
		for i in range(len(df_se)):
			if ((((tw-1)*10)>=df_se['s'][i])&((tw*10)<=df_se['e'][i])):
				list_trip.append(df_se['trip'][i])
		return(list_trip)

	for tw in range(1,144*28+1):
		df_all =pd.DataFrame()
		for se in select_start_in(tw):
			df_tw = df[(df['s'+str(se)]<=(tw*10))&(df['e'+str(se)]>=((tw-1)*10))]
			df_tw = df_tw[['id','c_id']]
			df_all = pd.concat([df_all,df_tw])
			df_tw = 0
		df_all.to_csv('/../connected_vehicle/move_vehicle/internal_move_id_sigma'+str(sigma)+'_'+str(n)+'/'+str(tw)+'.csv',index=False)
		df_all=0

for tw in range(1,144*28+1):
	df1 = pd.read_csv('/.../connected_vehicle/move_vehicle/internal_move_id_sigma'+str(sigma)+'_1/'+str(tw)+'.csv')
	df2 = pd.read_csv('/.../connected_vehicle/move_vehicle/internal_move_id_sigma'+str(sigma)+'_2/'+str(tw)+'.csv')
	df1 = pd.concat([df1,df2])
	df2 = 0
	df1.to_csv('/.../connected_vehicle/move_vehicle/internal_move_id_sigma'+str(sigma)+'/'+str(tw)+'.csv',index=False)
	df1 = 0






#Find out which cross-county commuters are driving and where they are at each time window
import os

def calcu_azimuth(lat1, lon1, lat2, lon2):
    lat1_rad = lat1 * math.pi / 180
    lon1_rad = lon1 * math.pi / 180
    lat2_rad = lat2 * math.pi / 180
    lon2_rad = lon2 * math.pi / 180
    y = math.sin(lon2_rad - lon1_rad) * math.cos(lat2_rad)
    x = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(
        lon2_rad - lon1_rad)
    brng = math.atan2(y, x) / math.pi * 180
    return float((brng + 360.0) % 360.0)



def havecoordinate(X1, Y1, X2, Y2, a, b):
    R = 6371
    brng = math.radians(calcu_azimuth(X1, Y1, X2, Y2))
    d = b - a
    lat1 = math.radians(X1)
    lon1 = math.radians(Y1)
    lat2 = math.asin(math.sin(lat1) * math.cos(d / R) +
                     math.cos(lat1) * math.sin(d / R) * math.cos(brng))
    lon2 = lon1 + math.atan2(math.sin(brng) * math.sin(d / R) * math.cos(lat1),
                             math.cos(d / R) - math.sin(lat1) * math.sin(lat2))
    lat2 = math.degrees(lat2)
    lon2 = math.degrees(lon2)
    return {'lat': lat2, 'lon': lon2}




df_near_county = pd.read_csv('/.../connected_vehicle/shortest_path/neighboring_county_lat_lon.csv')

df_edge = df_near_county[['c_id1', 'c_id2', 'distance']]
df_node = pd.DataFrame(data=list(set(df_edge['c_id1'])), columns=['node'])

list_edge = []
for i in range(0, len(df_edge)):
    list_edge.append([df_edge['c_id1'][i], df_edge['c_id2'][i], df_edge['distance'][i]])
list_node = list(df_node['node'])
G = nx.Graph()
G.add_nodes_from(list_node)
G.add_weighted_edges_from(list_edge)


def path_mid(path_cid):
    path = [0]
    for k in range(1, len(path_cid) - 1):
        path += [nx.dijkstra_path_length(G, int(path_cid[k - 1]), int(path_cid[k])) + path[k - 1]]
    path = path[1:]
    return (path)


def path_one(od_distance, path_cid_mid):
    path = [0]
    for i in range(int(math.ceil(od_distance / one_minute_dis))):
        path += [one_minute_dis * (i + 1)]
    if od_distance < path[-1]:
        path = path[:-1] + [od_distance] + path_cid_mid
        path.sort()
    else:
        path = path + path_cid_mid
        path.sort()
    return (path)


vehicle_speed = 100000
one_minute_dis = vehicle_speed / 60


def find_coordinate(x1, x2):
    o_lat = list(df_centroid[df_centroid['c_id'] == x1]['lat'])[0]
    o_lon = list(df_centroid[df_centroid['c_id'] == x1]['lon'])[0]
    d_lat = list(df_centroid[df_centroid['c_id'] == x2]['lat'])[0]
    d_lon = list(df_centroid[df_centroid['c_id'] == x2]['lon'])[0]
    od_distance = nx.dijkstra_path_length(G, x1, x2)
    writer.writerow([x1, x2, o_lat, o_lon, 0])
    path_cid = nx.dijkstra_path(G, x1, x2)
    path_cid_mid = path_mid(path_cid)
    path_cid_all = [0] + path_cid_mid + [od_distance]

    if od_distance < one_minute_dis:
        writer.writerow([x1, x2, d_lat, d_lon, 1])
    else:
        tw = 1
        path_one_minute = path_one(od_distance, path_cid_mid)
        for m in range(1, len(path_cid_all)):
            path_inner = path_one_minute[
                         (path_one_minute.index(path_cid_all[m - 1])):(path_one_minute.index(path_cid_all[m]) + 1)]
            X1 = df_centroid.loc[(df_centroid['c_id'] == int(path_cid[m - 1]))]['lat'].values
            Y1 = df_centroid.loc[(df_centroid['c_id'] == int(path_cid[m - 1]))]['lon'].values
            X2 = df_centroid.loc[(df_centroid['c_id'] == int(path_cid[m]))]['lat'].values
            Y2 = df_centroid.loc[(df_centroid['c_id'] == int(path_cid[m]))]['lon'].values
            for n in range(1, len(path_inner) - 1):
                latlon_dict = havecoordinate(X1, Y1, X2, Y2, path_inner[0] / 1000, path_inner[n] / 1000)
                writer.writerow([x1, x2, float(latlon_dict['lat']), float(latlon_dict['lon']), tw])
                tw += 1
        writer.writerow([x1, x2, d_lat, d_lon, tw])


df_commuting_flow = pd.read_csv('/.../connected_vehicle/people_code/commuting_flow.csv')
df_commuting_flow = df_commuting_flow[df_commuting_flow['c_id1'] != df_commuting_flow['c_id2']]
df_commuting_flow = df_commuting_flow.reset_index(drop=True)

df_centroid = pd.read_csv('/.../connected_vehicle/shortest_path/centroid.csv')

with open('/.../connected_vehicle/coordinates_location/coordinate_for_each_timewindow.csv','a') as coordinate_for_each_timewindow:
    writer = csv.writer(coordinate_for_each_timewindow)
    writer.writerow(['c_id1', 'c_id2', 'lat', 'lon', 'tw'])
    sc = []
    for index, row in df_commuting_flow.iterrows():
        c_id1 = int(row[0])
        c_id2 = int(row[1])
        if [c_id1, c_id2] not in sc:
            find_coordinate(c_id1, c_id2)
            find_coordinate(c_id2, c_id1)
            sc += [[c_id1, c_id2], [c_id2, c_id1]]






#'location_complete.csv' is obtained by processing 'coordinate_for_each_timewindow.csv' with ArcGIS
#ArcGIS found the country where each location is located.
df_location = pd.read_csv('/.../connected_vehicle/coordinates_location/location_complete.csv')
df_location.columns = ['c_id','c_id1','c_id2','tw']
df_all = pd.read_csv('/.../connected_vehicle/move_time/move_time_external_sigma'+str(sigma)+'.csv')

for timewindow in range(0,1440*28+1):
	df_move_tw = pd.DataFrame()
	for trip in range(1,57):
		if (trip %2 == 1):
			goback = 'go'
		else:
			goback = 'back'
		df_move = df_all[(df_all['s'+str(trip)]<=timewindow)&(df_all['e'+str(trip)]>=timewindow)]
		if (len(df_move)>0):
			df_move = df_move[['id','c_id1','c_id2','s'+str(trip),'e'+str(trip)]]
			df_move.columns=['id','c_id1','c_id2','s','e']
			df_move['goback'] = goback
			df_move_tw = pd.concat([df_move,df_move_tw])
	df_move_tw['tw'] = timewindow - df_move_tw['s']

	df_same = df_move_tw[['c_id1','c_id2','s','e','goback','tw']].drop_duplicates()
	df_same_go = df_same[df_same['goback']=='go']
	df_same_go = pd.merge(df_same_go,df_location,how = 'left',on=['c_id1','c_id2','tw'])
	df_same_go.columns = ['c_id1','c_id2','s','e','goback','tw','c_id']
	df_same_back = df_same[df_same['goback']=='back']
	df_same_back.columns = ['c_id2','c_id1','s','e','goback','tw']
	df_same_back = pd.merge(df_same_back,df_location,how = 'left',on=['c_id2','c_id1','tw'])
	df_same_back.columns = ['c_id2','c_id1','s','e','goback','tw','c_id']
	df_same = pd.concat([df_same_go,df_same_back])
	df_move_tw = pd.merge(df_move_tw,df_same,on=['c_id1','c_id2','s','e','goback','tw'])
	df_move_tw = df_move_tw[['id','c_id']]
	df_move_tw['c_id'] = df_move_tw['c_id'].astype(int)
	df_move_tw.to_csv('/.../connected_vehicle/move_time/external_move_id_sigma'+str(sigma)+'_1min/'+str(timewindow)+'.csv',index=False)








for tw in range(1, 4033):
    df_ten = pd.DataFrame()
    for n in range(0, 11):
        t = (tw - 1) * 10 + n
        df = pd.read_csv('/.../connected_vehicle/move_time/external_move_id_sigma'+str(sigma)+'_1min/' + str(t) + '.csv')
        df_ten = pd.concat([df, df_ten])
        df_ten = df_ten.reset_index(drop=True)
        df = 0
    list_id = list(set(df_ten['id']))
    list_cid = list()

    c_dic = {}
    for i in list_id:
        c_dic[str(i)] = []
    for c_i, c_r in df_ten.iterrows():
        c_dic[str(c_r[0])] += [int(c_r[1])]

    for ppl in list_id:
        cid = max(c_dic[str(ppl)], key=c_dic[str(ppl)].count)
        list_cid.append(cid)
    c_dic = {}
    df_ten = 0
    df_trans = pd.DataFrame()
    df_trans['id'] = list_id
    df_trans['c_id'] = list_cid
    df_trans.to_csv('/.../connected_vehicle/move_vehicle/external_move_id_sigma'+str(sigma)+'/' + str(tw) + '.csv',index=False)

