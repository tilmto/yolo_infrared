import numpy as np
import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def get_ground_truth(fname):
	f = open(fname,'r')
	ground_truth_list = []
	for obj in f.readlines():
		ground_truth_list.append(list(map(float,(obj.strip().split()))))
	return np.array(ground_truth_list)

infrared_root = '../SuZhou_0/1_int8'
files = os.listdir(infrared_root)
gt_all = []

for fname in files:
	if fname.endswith('.txt'):
		gt_all.extend(get_ground_truth(os.path.join(infrared_root,fname)))

gt_all = np.array(gt_all)
wh = gt_all[:,3:5]

clusters = KMeans(n_clusters=8,random_state=10,n_init=5).fit(wh)
print('[w,h] cluster centers:\n',clusters.cluster_centers_)
plt.scatter(wh[:,0],wh[:,1],c=clusters.labels_)
plt.show()