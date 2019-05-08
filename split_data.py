import os
import shutil
import numpy as np

def get_obj(fname):
	try:
		f = open(fname,'r')
	except IOError:
		print('No such file:',fname)
	obj_list = []
	for obj in f.readlines():
		obj_list.append(list(map(float,(obj.strip().split()))))
	return np.array(obj_list)

root_dir = '../SuZhou_0/1_int8'
fname_list = []
for fname in os.listdir(root_dir):
	if fname.endswith('.txt'):
		fname_list.append(os.path.join(root_dir,fname))

fname_list = np.array(fname_list)
np.random.shuffle(fname_list)
data_size = fname_list.shape[0]
test_list = fname_list[:int(data_size/10)]
train_list = fname_list[int(data_size/10):]

repeat_list = []
for fname in train_list:
	obj_list = get_obj(fname)
	pnum = 0
	cnum = 0
	bnum = 0
	for obj in obj_list:
		if obj[0]==0:
			pnum += 1
		elif obj[0]==1:
			cnum += 1
		else:
			bnum += 1

	if pnum+bnum>=cnum:
		repeat_list.append(fname)

for train_data in train_list:
	shutil.copy(train_data,'../train_data/'+os.path.basename(train_data))
	shutil.copy(train_data[:-4]+'.jpg','../train_data/'+os.path.basename(train_data)[:-4]+'.jpg')

for train_data in repeat_list:
	shutil.copy(train_data,'../train_data/'+os.path.basename(train_data)[:-4]+'_1.txt')
	shutil.copy(train_data[:-4]+'.jpg','../train_data/'+os.path.basename(train_data)[:-4]+'_1.jpg')

for test_data in test_list:
	shutil.copy(test_data,'../test_data/'+os.path.basename(test_data))
	shutil.copy(test_data[:-4]+'.jpg','../test_data/'+os.path.basename(test_data)[:-4]+'.jpg')
