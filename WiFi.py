#Simulate vehicles infected with WiFi virus in each time window
import numpy as np
import pandas as pd
import random
import math
from time import time
names = locals()

#'area.csv' is obtained through SHP file, including county number and county area
df_area = pd.read_csv('/.../connected_vehicle/shortest_path/area.csv')



def average_k(N,county):
	area = df_area[df_area['c_id']==county]['land'].reset_index(drop=True)[0]
	k = (N*math.pi*300*300)/area
	return(k)


def spread(N,S,I,k):
	list_new_i = list()
	for a in range(10):
		new_i = k * S * I / N 
		if (new_i>S):
			new_i = S
		integer = int(new_i)
		decimal = new_i - integer
		d_choice = np.random.choice([0,1],p=[1-decimal,decimal])
		new_i = integer + d_choice	
		I = I + new_i
		S = S - new_i
		list_new_i.append(new_i)
	return (list_new_i)


sigma = #The value of sigma needs to be entered
m = #The value of market share needs to be entered

seed_ppl = #Randomly select an initial infected vehicle
ill_num = 1
list_ill_people_num = list()
seed_No=#The serial number of the simulation needs to be entered
list_ill_ppl = list([seed_ppl])



for tw in range(1,144*28+1):
	start_time = time()
	TW = tw
	list_new_ill = list()	
	df_ill_new =pd.DataFrame()
	df_ill_new['ill'] = list([0])*10	

	df_move_ex_all = pd.read_csv('/.../connected_vehicle/move_vehicle/external_move_id_sigma' + str(sigma) + '/' + str(tw) + '.csv')
	df_move_ex_all=df_move_ex_all[df_move_ex_all['id'].isin(\
		set(pd.read_csv('/.../connected_vehicle/friends_network/m_id/m='+str(m)+'.csv')['ppl']))]
	df_move_in_all = pd.read_csv('/.../connected_vehicle/move_vehicle/internal_move_id_sigma' + str(sigma) + '/' + str(tw) + '.csv')
	df_move_in_all=df_move_in_all[df_move_in_all['id'].isin(\
		set(pd.read_csv('/.../connected_vehicle/friends_network/m_id/m='+str(m)+'.csv')['ppl']))]

	df_move_all = pd.concat([df_move_ex_all,df_move_in_all])
	df_ill = df_move_all[df_move_all['id'].isin(set(list_ill_ppl))]	
	df_noill = df_move_all[~df_move_all['id'].isin(set(list_ill_ppl))]
	list_ill_county = list(set(df_ill['c_id']))

	for county in list_ill_county:
		county = int(county)
		df_ill_tw = df_ill[df_ill['c_id']==county] 
		df_noill_tw = df_noill[df_noill['c_id']==county] 
		list_I = list(df_ill_tw['id'])
		list_S = list(df_noill_tw['id'])
		I = len(set(list_I))
		S = len(set(list_S))
		N = I + S

		if(I>0):
			k = average_k(N,county)
			list_new = spread(N,S,I,k)

			df_ill_new['new'] = list_new
			df_ill_new['ill'] = df_ill_new['ill'] + df_ill_new['new']
			new_i = df_ill_new['new'].sum()	
			if (new_i>0):
				for ill_id in random.sample(list_S,new_i):
					list_ill_ppl.append(ill_id)
					list_new_ill.append(ill_id)		
	for n in list(df_ill_new['ill']):
		ill_num = ill_num+n
		list_ill_people_num.append(ill_num)


	df_csv = pd.DataFrame()
	df_csv['ill_num'] = list_ill_people_num
	df_csv.to_csv('.../connected_vehicle/result_wifi/sigma='+str(sigma)+'/m='+str(m)+'/virus_ppl_'+str(seed_No)+'.csv',index=False)

	df_ill_ppl = pd.DataFrame()
	df_ill_ppl['id'] = list_new_ill
	df_ill_ppl.to_csv('.../connected_vehicle/result_wifi/sigma='+str(sigma)+'/m='+str(m)+'/simulation'+str(seed_No)+'/'+str(TW)+'.csv',index=False)
