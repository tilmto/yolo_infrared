import cv2 as cv
import numpy as np
import copy
import os

def postprocess(pred,output_size=(16,16),img_shape=(512,640),threshold=0.1):
	bboxes,obj_probs,class_probs = decode(pred,output_size)

	bboxes = np.reshape(bboxes,[-1,4])
	bboxes[:,0] *= img_shape[1]
	bboxes[:,1] *= img_shape[0]
	bboxes[:,2] *= img_shape[1]
	bboxes[:,3] *= img_shape[0]
	bboxes = bboxes.astype(np.int32)

	obj_probs = np.reshape(obj_probs,[-1])

	class_probs = np.reshape(class_probs,[len(obj_probs),-1])
	classes = np.argmax(class_probs,axis=1)
	max_probs = class_probs[np.arange(len(obj_probs)),classes]

	scores = obj_probs*max_probs

	keep_index = scores > threshold
	classes = classes[keep_index]
	scores = scores[keep_index]
	bboxes = bboxes[keep_index]

	classes,scores,bboxes = bboxes_sort(classes,scores,bboxes)
	classes,scores,bboxes = nms(classes,scores,bboxes)

	return bboxes,scores,classes


def decode(pred,output_size=(16,16)):
	H,W = output_size

	xy_offset = pred[:,:,:,0:2]
	wh_offset = np.power(pred[:,:,:,2:4],2)
	obj_probs = pred[:,:,:,4]
	class_probs = pred[:,:,:,5:]

	width_index = np.arange(W)
	height_index = np.arange(H)

	x_cell,y_cell = np.meshgrid(width_index,height_index)
	x_cell = np.reshape(x_cell,[1,-1,1])
	y_cell = np.reshape(y_cell,[1,-1,1])

	bbox_x = (x_cell+xy_offset[:,:,:,0])/W
	bbox_y = (y_cell+xy_offset[:,:,:,1])/H
	bbox_w = wh_offset[:,:,:,0]
	bbox_h = wh_offset[:,:,:,1]
	
	bboxes = np.stack([bbox_x,bbox_y,bbox_w,bbox_h],axis=3)

	return bboxes,obj_probs,class_probs


def bboxes_sort(classes,scores,bboxes,top_k=30):
	index = np.argsort(-scores)
	classes = classes[index][:top_k]
	scores = scores[index][:top_k]
	bboxes = bboxes[index][:top_k]
	return classes,scores,bboxes


def nms(classes,scores,bboxes,nms_threshold=0.5):
	keep_bboxes = np.ones(scores.shape,dtype=np.bool)
	for i in range(scores.shape[0]-1):
		if keep_bboxes[i]:
			for j in range(i+1,scores.shape[0]):
				iou = compute_iou(bboxes[i], bboxes[j])
				if iou>nms_threshold and classes[i]==classes[j]:
					keep_bboxes[j] = False

	idxes = np.where(keep_bboxes)
	return classes[idxes],scores[idxes],bboxes[idxes]


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


def compute_map(classes,bboxes,gt_bboxes,threshold=0.5):
	person_pr = []
	car_pr = []
	bus_pr = []

	map_list = []

	gt_num = [0,0,0]
	for i in range(gt_bboxes.shape[0]):
		gt_num[int(gt_bboxes[i][0])] += 1

	for i in range(classes.shape[0]):
		is_bg = True

		if classes[i]==0:
			for j in range(gt_bboxes.shape[0]):
				if gt_bboxes[j][0]==0:
					iou = compute_iou(bboxes[i],gt_bboxes[j][1:])
					if iou>threshold:
						person_pr.append(1)
						is_bg = False
						break
			if is_bg:
				person_pr.append(0)
			
		elif classes[i]==1:
			for j in range(gt_bboxes.shape[0]):
				if gt_bboxes[j][0]==1:
					iou = compute_iou(bboxes[i],gt_bboxes[j][1:])
					if iou>threshold:
						car_pr.append(1)
						is_bg = False
						break
			if is_bg:
				car_pr.append(0)

		else:
			for j in range(gt_bboxes.shape[0]):
				if gt_bboxes[j][0]==2:
					iou = compute_iou(bboxes[i],gt_bboxes[j][1:])
					if iou>threshold:
						bus_pr.append(1)
						is_bg = False
						break
			if is_bg:
				bus_pr.append(0)

	if gt_num[0]:
		map_list.append(compute_ap(person_pr,gt_num[0]))

	if gt_num[1]:
		map_list.append(compute_ap(car_pr,gt_num[1]))

	if gt_num[2]:
		map_list.append(compute_ap(bus_pr,gt_num[2]))

	return np.mean(map_list)


def compute_ap(pr,gt_num):
	precision_list = []
	right_num = 0
	false_num = 0
	max_precision = 0

	if not len(pr):
		return 0

	for i in range(len(pr)):
		if pr[i]==1:
			right_num += 1
			max_precision = right_num/(right_num+false_num)
			precision_list.append(max_precision)
			if len(precision_list)>=gt_num:
				break
		else:
			false_num += 1

	if len(precision_list)<gt_num:
		for i in range(gt_num-len(precision_list)):
			right_num += 1
			max_precision = right_num/(right_num+false_num)
			precision_list.append(max_precision)

	return np.mean(precision_list)


def compute_precision(classes,bboxes,gt_bboxes,threshold=0.5):
	right_num = 0
	for i in range(classes.shape[0]):
		for j in range(gt_bboxes.shape[0]):
			if classes[i]==gt_bboxes[j][0]:
				iou = compute_iou(bboxes[i],gt_bboxes[j][1:])
				if iou>threshold:
					right_num += 1
					break

	if classes.shape[0]:
		return right_num/classes.shape[0]
	else:
		return 0
		

def compute_recall(classes,bboxes,gt_bboxes,threshold=0.5):
	right_num = 0
	for i in range(gt_bboxes.shape[0]):
		for j in range(classes.shape[0]):
			if classes[j]==gt_bboxes[i][0]:
				iou = compute_iou(bboxes[j],gt_bboxes[i][1:])
				if iou>threshold:
					right_num += 1
					break
	return right_num/gt_bboxes.shape[0]


def draw_obj(img,bboxes,classes):
	img = copy.deepcopy(img)
	for i,bbox in enumerate(bboxes):
		if classes[i]==0:
			color = [0,0,255]
		elif classes[i]==1:
			color = [0,255,0]
		else:
			color = [255,0,0]
		left_top = (int(bbox[0]-bbox[2]/2),int(bbox[1]-bbox[3]/2))
		right_bottom = (int(bbox[0]+bbox[2]/2),int(bbox[1]+bbox[3]/2))
		cv.rectangle(img,left_top,right_bottom,color,2)
	cv.imshow('img',img)
	cv.waitKey(0)


if __name__ == '__main__':
	pred = np.ones([1,256,5,8])
	gt_bboxes = np.ones([5,5])
	bboxes,scores,classes = postprocess(pred)
	print(bboxes.shape,scores.shape,classes.shape)
	print(compute_map(classes,bboxes,gt_bboxes))