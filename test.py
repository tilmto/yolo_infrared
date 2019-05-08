import tensorflow as tf
import numpy as np
import cv2 as cv
import sys
import os
import copy

input_dir = sys.argv[1]
output_dir = sys.argv[2]

test_images = []
img_tensor = []
fnames = os.listdir(input_dir)
print(fnames)
for fname in fnames:
    img = cv.imread(os.path.join(input_dir,fname))
    test_images.append(img)
    img = cv.resize(img,(512,512))
    img = (img-127)/128
    img_tensor.append(img)

test_images = np.array(test_images)
img_tensor = np.array(img_tensor)


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


def draw_obj(img,bboxes,classes,fname):
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
    cv.imwrite(fname,img)


if __name__ == '__main__':  
    config=tf.ConfigProto(inter_op_parallelism_threads=6,intra_op_parallelism_threads=6)
    config.gpu_options.allow_growth = True

    with tf.Graph().as_default(), tf.Session(config=config) as sess:
        load_graph='my_model_best.meta'
        saver = tf.train.import_meta_graph(load_graph)
        saver.restore(sess,'my_model_best')
        graph = tf.get_default_graph()
        images = graph.get_tensor_by_name("images:0")
        is_training = graph.get_tensor_by_name("is_training:0")
        predict = graph.get_tensor_by_name("predict:0")

        for i in range(test_images.shape[0]):
            feed_dict={images:np.expand_dims(img_tensor[i],axis=0),is_training:False}
            pred = sess.run(predict,feed_dict=feed_dict)
            bboxes,scores,classes = postprocess(pred,img_shape=(test_images[i].shape[0],test_images[i].shape[1]))
            draw_obj(test_images[i],bboxes,classes,os.path.join(output_dir,fnames[i]))