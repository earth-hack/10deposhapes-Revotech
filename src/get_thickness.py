import pandas as pd
import numpy as np

data = pd.read_csv('CHEAL-A8-Pre-GR-shape.csv', header=0)
# df = pd.read_csv('GR_shape-2 (1).csv', header=0)
# zone = [None] + df['Zone'].values.tolist()
# top = df['Top_Depth'].values
# bottom = [np.min(top)] + df['Bottom_Depth'].values.tolist()

# data['Zone'] = -999
# data['Thickness'] = 0
data['Lithofacies'] = -999

bottom = [np.min(data['TVD'])-0.01] + data['TVD'][data['Unit_mark_final'] == 1].values.tolist()

for i in range(len(bottom)-1, 0, -1):
	# print(bottom[i], bottom[i-1])
	# mask = (data['DEPTH'] <= bottom[i]) & (data['DEPTH'] > bottom[i-1])
	mask = (data['TVD'] <= bottom[i]) & (data['TVD'] > bottom[i-1])
	tmp = data[mask]['TVD'].values
	# vmud_mean = np.mean(data['V_mud'][mask].values)
	vmud = data['V_mud'][mask].values
	litho = np.zeros(5)
	litho[1] = np.count_nonzero(vmud < 0.15)
	litho[2] = np.count_nonzero((vmud >= 0.15) & (vmud < 0.5))
	litho[3] = np.count_nonzero((vmud >= 0.5) & (vmud < 0.7))
	litho[4] = np.count_nonzero(vmud >= 0.7)
	# data.loc[mask, 'Zone'] = zone[i]
	# data.loc[mask, 'Thickness'] = np.max(tmp) - np.min(tmp)
	# if vmud_mean < 0.15:
	# 	data.loc[mask, 'Lithofacies'] = 1
	# elif vmud_mean < 0.5:
	# 	data.loc[mask, 'Lithofacies'] = 2
	# elif vmud_mean < 0.7:
	# 	data.loc[mask, 'Lithofacies'] = 3
	# else:
	# 	data.loc[mask, 'Lithofacies'] = 4
	data.loc[mask, 'Lithofacies'] = np.argmax(litho)

data.to_csv('CHEAL-A8.csv')
