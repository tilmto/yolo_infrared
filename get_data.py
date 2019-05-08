import cv2 as cv
import numpy as np
import copy
import os

train_dir = '../train_data'
test_dir = '../test_data'
train_list = []
test_list = []

for fname in os.listdir(train_dir):
	if fname.endswith('.txt'):
		train_list.append(os.path.join(train_dir,fname))

for fname in os.listdir(test_dir):
	if fname.endswith('.txt'):
		test_list.append(os.path.join(test_dir,fname))

train_list = np.array(train_list)
test_list = np.array(test_list)

def gene_train():
	for i in range(train_list.shape[0]):
		yield train_list[i]

def gene_test():
	for i in range(test_list.shape[0]):
		yield test_list[i]

train_generator = gene_train()
test_generator = gene_test()


anchors = np.array([[0.06846971,0.09351028],
					[0.87282928,0.93134569],
					[0.21763393,0.63630022],
					[0.18499602,0.22887094],
					[0.41931916,0.3931352 ]],dtype=np.float32)
'''
anchors = np.array([[0.12336957,0.14655337],
                    [0.74363542,0.85898437],
                    [0.12821875,0.36161719],
                    [0.28491945,0.25482921],
                    [0.05236321,0.07885134],
                    [0.48402562,0.43126456],
                    [0.99181985,0.99311938],
                    [0.26013569,0.73224198]],dtype=np.float32)
'''

def get_obj(fname):
	try:
		f = open(fname,'r')
	except IOError:
		print('No such file:',fname)
		exit()

	obj_list = []
	for obj in f.readlines():
		obj_list.append(list(map(float,(obj.strip().split()))))
	return np.array(obj_list)


def draw_obj(img,obj_list):
	img = copy.deepcopy(img)
	for obj in obj_list:
		if obj[0]==0:
			color = [0,0,255]
		elif obj[0]==1:
			color = [0,255,0]
		else:
			color = [255,0,0]
		left_top = (int((obj[1]-obj[3]/2)*img.shape[1]),int((obj[2]-obj[4]/2)*img.shape[0]))
		right_bottom = (int((obj[1]+obj[3]/2)*img.shape[1]),int((obj[2]+obj[4]/2)*img.shape[0]))
		cv.rectangle(img,left_top,right_bottom,color,2)
	cv.imshow('img',img)
	cv.waitKey(0)


def compute_iou(bbox1,bbox2):
	left_top1 = [bbox1[0]-bbox1[2]/2,bbox1[1]-bbox1[3]/2]
	right_bottom1 = [bbox1[0]+bbox1[2]/2,bbox1[1]+bbox1[3]/2]
	left_top2 = [bbox2[0]-bbox2[2]/2,bbox2[1]-bbox2[3]/2]
	right_bottom2 = [bbox2[0]+bbox2[2]/2,bbox2[1]+bbox2[3]/2]

	left_top_inter = np.maximum(np.array(left_top1),np.array(left_top2))
	right_bottom_inter = np.minimum(np.array(right_bottom1),np.array(right_bottom2))

	area_sum = bbox1[2]*bbox1[3]+bbox2[2]*bbox2[3]

	w_inter,h_inter = np.maximum(right_bottom_inter-left_top_inter,[0,0])
	area_inter = w_inter*h_inter
	
	return area_inter/(area_sum-area_inter)


def flip_img(img,obj_list):
	img_flip = cv.flip(img,1)
	obj_list = copy.deepcopy(obj_list)  
	obj_list[:,1] = np.ones(obj_list.shape[0])-obj_list[:,1]
	return img_flip,obj_list


def gray(img):
	img = copy.deepcopy(img)
	img = img/255
	ret,new_img = cv.threshold(img,0.4,1,cv.THRESH_BINARY)
	weight = 0.2
	img = new_img*weight+(1-weight)*img
	return img*255


def salt_noise(img):
	for i in range(2000):
		x= np.random.randint(640)
		y = np.random.randint(512)
		if np.random.rand()<0.5:
			img[y,x] = 0
		else:
			img[y,x] = 255
	return img


def gaussian_blur(img):
	return cv.GaussianBlur(img,(5,5),10)


def get_train_data(anchors=anchors,batch_size=3):
	global train_generator
	end_of_epoch = False

	images = []
	labels = []

	for img_idx in range(batch_size):
		try:
			fname = train_generator.__next__()
		except StopIteration:
			train_generator = gene_train()
			end_of_epoch = True

			return np.array(images),np.array(labels),end_of_epoch

		obj_list = get_obj(fname)

		if not obj_list.shape[0]:
			continue

		img = cv.imread(fname[:-4]+'.jpg')

		if np.random.rand()<0.5:
			img,obj_list = flip_img(img,obj_list)

		random_seed = np.random.rand()
		if random_seed<0.3:
			img = gray(img)
		elif random_seed<0.5:
			img = salt_noise(img)
		elif random_seed<0.7:
			img = gaussian_blur(img)

		img = cv.resize(img,(512,512))
		img = (img-127)/128
		images.append(img)

		label = np.zeros([16*16,anchors.shape[0],8])
		for obj in obj_list:
			grid_x = int(obj[1]*16)
			grid_y = int(obj[2]*16)
			index = grid_y*16+grid_x
			for i in range(anchors.shape[0]):
				label[index,i,0] = obj[1]*16-grid_x
				label[index,i,1] = obj[2]*16-grid_y
				label[index,i,2] = np.sqrt(obj[3])
				label[index,i,3] = np.sqrt(obj[4])
				label[index,i,4] = compute_iou(obj[1:],np.array([grid_x/16+1/32,grid_y/16+1/32,anchors[i][0],anchors[i][1]]))
				label[index,i,5+int(obj[0])] = 1

		labels.append(label)

	return np.array(images),np.array(labels),end_of_epoch


def get_valid_data(anchors=anchors,batch_size=1,height_limit=0):
	global test_generator
	end_of_epoch = False

	images = []
	labels = []
	gt_bboxes = []
	fname_list = []
	orig_images = []

	for img_idx in range(batch_size):
		try:
			fname = test_generator.__next__()
		except StopIteration:
			test_generator = gene_test()
			end_of_epoch = True

			return np.array(images),np.array(labels),np.array(gt_bboxes),np.array(fname_list),np.array(orig_images),end_of_epoch

		fname_list.append(fname)
		obj_list = get_obj(fname)

		gt_bbox = obj_list.copy()
		gt_bbox = gt_bbox[gt_bbox[:,4]>height_limit]
		gt_bbox[:,1] *= 640
		gt_bbox[:,2] *= 512
		gt_bbox[:,3] *= 640
		gt_bbox[:,4] *= 512
		gt_bboxes.extend(gt_bbox)

		img = cv.imread(fname[:-4]+'.jpg')
		orig_images.append(img)

		img = cv.resize(img,(512,512))
		img = (img-127)/128
		images.append(img)

		label = np.zeros([16*16,anchors.shape[0],8])
		for obj in obj_list:
			grid_x = int(obj[1]*16)
			grid_y = int(obj[2]*16)
			index = grid_y*16+grid_x
			for i in range(anchors.shape[0]):
				label[index,i,0] = obj[1]*16-grid_x
				label[index,i,1] = obj[2]*16-grid_y
				label[index,i,2] = np.sqrt(obj[3])
				label[index,i,3] = np.sqrt(obj[4])
				label[index,i,4] = compute_iou(obj[1:],np.array([grid_x/16+1/32,grid_y/16+1/32,anchors[i][0],anchors[i][1]]))
				label[index,i,5+int(obj[0])] = 1

		labels.append(label)

	return np.array(images),np.array(labels),np.array(gt_bboxes),np.array(fname_list),np.array(orig_images),end_of_epoch


if __name__ == '__main__':
	images,labels,end_of_epoch = get_train_data(100)
	print(images.shape,labels.shape)

	images,labels,gt_bboxes,fname_list,orig_images,end_of_epoch = get_valid_data(3)
	print(images.shape,labels.shape,gt_bboxes.shape,orig_images.shape)
	print(gt_bboxes)