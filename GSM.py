#Simulate vehicles infected with GSM virus in each time window
import pandas as pd
sigma=#The value of sigma needs to be entered
m_s=#The value of market share needs to be entered
con_ppl=set(pd.read_csv('/.../connected_vehicle/friends_network/m_id/m='+str(m_s)+'.csv')['ppl'])
sl=#The serial number of the simulation needs to be entered
virus_ppl_for_each_tw=[]#Randomly select an initial infected vehicle
df_start=pd.DataFrame()
df_start['ppl']=virus_ppl_for_each_tw
df_start.to_csv('/.../connected_vehicle/result_GSM/sigma='+str(sigma)+'/m='+str(m_s)+'/simulation'+str(sl)+'/'+str(0)+'.csv',index=None)
df_len_virus_ppl_for_each_tw=pd.DataFrame(columns=['I_No'])
df_len_virus_ppl_for_each_tw.to_csv('/.../connected_vehicle/result_GSM/sigma='+str(sigma)+'/m='+str(m_s)+'/virus_ppl_'+str(sl)+'.csv',index=None)
len_virus_ppl_for_each_tw=1


for i in range(1, 4033):
	virus_ppl_for_each_tw_fz = virus_ppl_for_each_tw.copy()
	fr_net=pd.concat([pd.read_csv('/.../connected_vehicle/friends_network/connected_network/sigma='+str(sigma)+'/1/'+str(i)+'.csv'),\
					  pd.read_csv('/.../connected_vehicle/friends_network/connected_network/sigma='+str(sigma)+'/2/'+str(i)+'.csv'),\
					  pd.read_csv('/.../connected_vehicle/friends_network/connected_network/sigma='+str(sigma)+'/3/'+str(i)+'.csv'),\
					  pd.read_csv('/.../connected_vehicle/friends_network/connected_network/sigma='+str(sigma)+'/4/'+str(i)+'.csv'),\
					  pd.read_csv('/.../connected_vehicle/friends_network/connected_network/sigma='+str(sigma)+'/5/'+str(i)+'.csv'),\
					  pd.read_csv('/.../connected_vehicle/friends_network/connected_network/sigma='+str(sigma)+'/6/'+str(i)+'.csv')])
	al_ppl_for_each_tw=[]
	move_ppl=set(fr_net['first'])

	for j in range(1,11):
		len_virus_ppl_for_each_tw_fz=len_virus_ppl_for_each_tw

		import_virus_ev = virus_ppl_for_each_tw.copy()
		import_virus_ev = set(import_virus_ev).difference(set(al_ppl_for_each_tw))
		import_virus_ev = list(import_virus_ev.intersection(move_ppl))
		al_ppl_for_each_tw+=import_virus_ev

		friends_list_r=[]

		if import_virus_ev!=[]:
			one_fr_net = fr_net.loc[fr_net['first'].isin(set(import_virus_ev))]
			friends_list_r = list(one_fr_net['second'].copy())
		else:
			friends_list_r = []

		if friends_list_r != []:
			virus_ppl_for_each_tw += list(set(friends_list_r).intersection(con_ppl))
			friends_list_r = ()
			virus_ppl_for_each_tw = list(set(virus_ppl_for_each_tw))
		len_virus_ppl_for_each_tw = len(virus_ppl_for_each_tw)

		if len_virus_ppl_for_each_tw_fz!=len_virus_ppl_for_each_tw:
			df_len_virus_ppl_for_each_tw = pd.read_csv(\
				'/.../connected_vehicle/result_GSM/sigma='+str(sigma)+'/m='+str(m_s)+'/virus_ppl_'+str(sl)+'.csv')
			df_len_virus_ppl_for_each_tw = df_len_virus_ppl_for_each_tw.append( \
				{'I_No': len_virus_ppl_for_each_tw}, ignore_index=True)
			df_len_virus_ppl_for_each_tw.to_csv( \
				'/.../connected_vehicle/result_GSM/sigma='+str(sigma)+'/m='+str(m_s)+'/virus_ppl_'+str(sl)+'.csv', index=None)
		else:
			for e in range(j,11):
				df_len_virus_ppl_for_each_tw = pd.read_csv( \
					'/.../connected_vehicle/result_GSM/sigma='+str(sigma)+'/m='+str(m_s)+'/virus_ppl_'+str(sl)+'.csv')
				df_len_virus_ppl_for_each_tw = df_len_virus_ppl_for_each_tw.append( \
					{'I_No': len_virus_ppl_for_each_tw}, ignore_index=True)
				df_len_virus_ppl_for_each_tw.to_csv( \
					'/.../connected_vehicle/result_GSM/sigma='+str(sigma)+'/m='+str(m_s)+'/virus_ppl_'+str(sl)+'.csv', index=None)
			break
	chu_df = pd.DataFrame(columns=['ppl'])
	chu_df['ppl'] = list(set(virus_ppl_for_each_tw).difference(set(virus_ppl_for_each_tw_fz)))
	chu_df.to_csv('/.../connected_vehicle/result_GSM/sigma='+str(sigma)+'/m='+str(m_s)+'/simulation'+str(sl)+'/'+str(i) + '.csv', index=None)
