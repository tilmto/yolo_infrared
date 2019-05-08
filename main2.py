import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import cv2 as cv
import copy
import time
import get_data
import util
from MobileYolo_scale import MobileYolo_scale

def model_size():
    params = tf.trainable_variables()
    size = 0
    for x in params:
        sz = 1
        for dim in x.get_shape():
            sz *= dim.value
        size += sz
    return size

tf.app.flags.DEFINE_string('model','1','1:MobileYolo\n2:Yolo\n')
tf.app.flags.DEFINE_integer('batch_size',10,'Input the batch size')
tf.app.flags.DEFINE_float('cr',0.8,'Input the compress ratio for skip method')
tf.app.flags.DEFINE_float('height_limit',0,'Input the height limit for validaton set')
tf.app.flags.DEFINE_boolean('is_training',True,'Training mode or not')
tf.app.flags.DEFINE_boolean('load_model',False,'Load model or not')
tf.app.flags.DEFINE_boolean('load_backbone',False,'Load backbone network only or not')
tf.app.flags.DEFINE_boolean('draw',False,'Draw bboxes or not')
tf.app.flags.DEFINE_boolean('scale',False,'Use scale method or not')
flags = tf.app.flags.FLAGS

config = tf.ConfigProto(inter_op_parallelism_threads=6,intra_op_parallelism_threads=6)
config.gpu_options.allow_growth = True


anchors = np.array([[0.12336957,0.14655337],
                    [0.74363542,0.85898437],
                    [0.12821875,0.36161719],
                    [0.28491945,0.25482921],
                    [0.05236321,0.07885134],
                    [0.48402562,0.43126456],
                    [0.99181985,0.99311938],
                    [0.26013569,0.73224198]],dtype=np.float32)
'''
anchors = np.array([[0.06846971,0.09351028],
                    [0.87282928,0.93134569],
                    [0.21763393,0.63630022],
                    [0.18499602,0.22887094],
                    [0.41931916,0.3931352 ]],dtype=np.float32)
'''

scale_list = [(512,512),(450,450),(400,400),(350,350),(300,300),(256,256)]
batch_size = flags.batch_size if not flags.scale else 1
no_scale = np.zeros((batch_size))

if __name__ == '__main__':
    images_plh = tf.placeholder(tf.float32,[None,None,None,3],name='images')
    labels_plh = tf.placeholder(tf.float32,[None,256,anchors.shape[0],8],name='labels')
    best_scale_plh = tf.placeholder(tf.int32,[None],name='best_scale')
    is_training_plh = tf.placeholder(tf.bool,name='is_training')

    model = MobileYolo_scale(images_plh,labels_plh,anchors,best_scale_plh,is_training_plh)

    saver = tf.train.Saver(max_to_keep=10)

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        if not flags.is_training or flags.load_model:
            if flags.load_backbone:     
                variables = slim.get_variables_to_restore()
                variables_to_restore = [v for v in variables if v.name.find('scale')==-1] 
                saver2 = tf.train.Saver(variables_to_restore,max_to_keep=10)
                saver2.restore(sess,'my_model_best')
            else:
                saver.restore(sess,'my_model_best')
            print('Load model from my_model_best.')
        else:
            print('Create and initialize new model. Size:',model_size())

        best_map = 0
        best_precision = 0
        best_recall = 0

        best_ep = 0
        max_epoch = 500

        if flags.is_training:
            print('do train!!!!!!')
            for ep_idx in range(max_epoch):
                train_loss_sum = 0
                scale_loss_sum = 0
                batch_num = 0

                while True:
                    images,labels,end_of_epoch = get_data.get_train_data(anchors,batch_size)             

                    if images.shape[0]:
                        if not flags.scale:
                            feed_dict={
                                images_plh:images,
                                labels_plh:labels,
                                is_training_plh:True,
                                best_scale_plh:no_scale
                            }
                            _,loss=sess.run([model.optimizer,model.loss],feed_dict=feed_dict)
                            train_loss_sum += loss
                            batch_num += 1

                        else:
                            loss_list = []
                            for i in range(6):
                                new_images = []
                                for image in images:
                                    new_images.append(cv.resize(image,scale_list[i]))
                                feed_dict={images_plh:np.array(new_images),labels_plh:labels,is_training_plh:True,best_scale_plh:no_scale}
                                _,loss = sess.run([model.optimizer,model.loss],feed_dict=feed_dict)
                                loss_list.append(loss)
                            
                            best_scale = np.argmin(loss_list)
                            train_loss_sum += np.min(loss_list)

                            for i in range(6):
                                new_images = []
                                for image in images:
                                    new_images.append(cv.resize(image,scale_list[i]))
                                feed_dict={images_plh:np.array(new_images),labels_plh:labels,is_training_plh:True,best_scale_plh:[best_scale]}
                                _,scale_loss = sess.run([model.optimizer_scale,model.loss_scale],feed_dict=feed_dict)
                                scale_loss_sum += scale_loss 

                            batch_num += 1


                    if end_of_epoch:
                        break

                print('epoch:',ep_idx,' train_loss_avg:',train_loss_sum/batch_num,' scale_loss_avg:',scale_loss_sum/scale_loss)

                ######### do validaton #########
                map_list = []
                precision_list = []
                recall_list = []
                cr_list = []

                while True:
                    images,labels,gt_bboxes,fname_list,orig_images,end_of_epoch = get_data.get_valid_data(anchors,batch_size=1,height_limit=flags.height_limit)
                    
                    if gt_bboxes.shape[0]:
                        feed_dict={
                            images_plh:images,
                            labels_plh:labels,
                            is_training_plh:False,
                            best_scale_plh:no_scale
                        }

                        if images.shape[0]:
                            scales = sess.run(model.scale_pred,feed_dict=feed_dict)
                            best_scale = np.argmax(scales)

                            new_images = []
                            for image in images:
                                new_images.append(cv.resize(image,scale_list[best_scale]))

                            feed_dict={images_plh:np.array(new_images),labels_plh:labels,is_training_plh:True,best_scale_plh:no_scale}

                            pred = sess.run(model.predict,feed_dict=feed_dict)
                            bboxes,scores,classes = util.postprocess(pred)

                            map_list.append(util.compute_map(classes,bboxes,gt_bboxes))
                            precision_list.append(util.compute_precision(classes,bboxes,gt_bboxes))
                            recall_list.append(util.compute_recall(classes,bboxes,gt_bboxes))

                    if end_of_epoch:
                        break

                test_map = np.mean(map_list)
                test_precision = np.mean(precision_list)
                test_recall = np.mean(recall_list)

                if flags.model=='2':
                    cr_avg = np.mean(cr_list)

                if test_map>best_map and (flags.model!='2' or cr_avg<flags.cr+0.03):
                    best_ep = ep_idx
                    best_map = test_map
                    saver.save(sess,'./my_model_best')

                print('map:',test_map,' precision:',test_precision,' recall:',test_recall,'\nbest_epoch:',best_ep,' best_map:',best_map)

                if flags.model=='2':
                    print('compress ratio:',cr_avg)

        else:
            print('do test!!!!!!')

            map_list = []
            precision_list = []
            recall_list = []
            cr_list = []

            while True:
                images,labels,gt_bboxes,fname_list,orig_images,end_of_epoch = get_data.get_valid_data(anchors,batch_size=1,height_limit=flags.height_limit)

                if gt_bboxes.shape[0]:
                    feed_dict={
                        images_plh:images,
                        labels_plh:labels,
                        is_training_plh:False,
                        best_scale_plh:None
                    }

                    if images.shape[0]:
                        scales = sess.run(model.scale_pred,feed_dict=feed_dict)
                        best_scale = np.argmax(scales)

                        new_images = []
                        for image in images:
                            new_images.append(cv.resize(image,scale_list[best_scale]))

                        feed_dict={images_plh:np.array(new_images),labels_plh:labels,is_training_plh:True,best_scale_plh:None}

                        pred = sess.run(model.predict,feed_dict=feed_dict)
                        bboxes,scores,classes = util.postprocess(pred)

                        map_list.append(util.compute_map(classes,bboxes,gt_bboxes))
                        precision_list.append(util.compute_precision(classes,bboxes,gt_bboxes))
                        recall_list.append(util.compute_recall(classes,bboxes,gt_bboxes))
                
                if end_of_epoch:
                    break

            test_map = np.mean(map_list)
            test_precision = np.mean(precision_list)
            test_recall = np.mean(recall_list)

            print('map:',test_map,' precision:',test_precision,' recall:',test_recall)
            if flags.model=='2':
                print('compress ratio:',np.mean(cr_list))
