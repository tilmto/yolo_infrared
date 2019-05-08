import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

class MobileYolo_skip:
	def __init__(self,images,labels,anchors,cr=0.8,is_training=True,img_size=512,scope='Yolo_MobileNet_v2'):
		self.images = images
		self.labels = labels
		self.anchors = anchors
		self.cr = cr
		self.is_training = is_training
		self.img_size = img_size

		self.anchors_per_unit = self.anchors.shape[0]
		self.final_channels = self.anchors_per_unit*8
		self.output_size = [16,16]
		self.scales = [1,1,1,1]

		self.lstm_cell=tf.nn.rnn_cell.LSTMCell(num_units=10,num_proj=1,name='lstm_cell')
		self.layer_masks = []

		self.build_model(scope,is_training)

		self.loss_yolo = self.compute_loss()

		flops = {'conv1':154140672,'16':128974848,'24_1':76283904,'24':67239936,'32_1':40402944,'32':28704768,
				 '64_1':19759104,'64':26935296,'96_1':33226752,'96':59277312,'160_1':40771584,'160':40427520,
				 '320':60088320,'conv2':20971520,'conv3':5120}

		self.flops_fix = flops['conv1']+flops['16']+flops['24_1']+flops['32_1']+flops['64_1']+flops['96_1']+flops['160_1']+flops['320']+flops['conv2']+flops['conv3']
		self.flops_dyn = flops['24']+flops['32']*2+flops['64']*3+flops['96']*2+flops['160']*2
		self.flops_sum = self.flops_fix+self.flops_dyn

		self.computation_cost = tf.reduce_mean(flops['24']*self.layer_masks[0]+flops['32']*tf.reduce_sum(self.layer_masks[1:3],axis=0)
											+flops['64']*tf.reduce_sum(self.layer_masks[3:6],axis=0)
											+flops['96']*tf.reduce_sum(self.layer_masks[6:8],axis=0)
											+flops['160']*tf.reduce_sum(self.layer_masks[8:],axis=0))+self.flops_fix

		self.compress_ratio = self.computation_cost/self.flops_sum

		alpha = 1e-9
		sign = tf.cond(self.compress_ratio>self.cr,lambda:1.0,lambda:-1.0)
		self.loss = self.loss_yolo + alpha*sign*self.computation_cost

		self.optimizer = tf.train.AdamOptimizer(1e-4).minimize(self.loss)


	def build_model(self,scope,is_training=True):
		x = self.images

		with tf.variable_scope(scope):
			x = slim.conv2d(x,32,[7,7],2,padding='SAME',activation_fn=None,scope='conv1')
			x = tf.nn.relu6(self.batch_norm(x,is_training,scope='bn1'))

			x = self.residual(x,out_channels=16,multi=1,stride=1,is_training=self.is_training,scope='residual_16_1')

			x = self.residual(x,out_channels=24,multi=6,stride=2,is_training=self.is_training,scope='residual_24_1')
			next_x = self.residual(x,out_channels=24,multi=6,stride=1,is_training=self.is_training,scope='residual_24_2')
			mask = self.skip_layer(x,scope='skip_layer_24_1')
			one_tensor = tf.ones_like(mask)
			x = mask*next_x+(one_tensor-mask)*x
			self.layer_masks.append(tf.reshape(mask,[-1]))

			x = self.residual(x,out_channels=32,multi=6,stride=2,is_training=self.is_training,scope='residual_32_1')
			for i in range(2):
				next_x = self.residual(x,out_channels=32,multi=6,stride=1,is_training=self.is_training,scope='residual_32_'+str(i+2))
				mask = self.skip_layer(x,scope='skip_layer_32_'+str(i+1))
				x = mask*next_x+(one_tensor-mask)*x
				self.layer_masks.append(tf.reshape(mask,[-1]))

			x = self.residual(x,out_channels=64,multi=6,stride=2,is_training=self.is_training,scope='residual_64_1')
			for i in range(3):
				next_x = self.residual(x,out_channels=64,multi=6,stride=1,is_training=self.is_training,scope='residual_64_'+str(i+2))
				mask = self.skip_layer(x,scope='skip_layer_64_'+str(i+1))
				x = mask*next_x+(one_tensor-mask)*x
				self.layer_masks.append(tf.reshape(mask,[-1]))

			x = self.residual(x,out_channels=96,multi=6,stride=1,is_training=self.is_training,scope='residual_96_1')
			for i in range(2):
				next_x = self.residual(x,out_channels=96,multi=6,stride=1,is_training=self.is_training,scope='residual_96_'+str(i+2))
				mask = self.skip_layer(x,scope='skip_layer_96_'+str(i+1))
				x = mask*next_x+(one_tensor-mask)*x
				self.layer_masks.append(tf.reshape(mask,[-1]))

			x = self.residual(x,out_channels=160,multi=6,stride=2,is_training=self.is_training,scope='residual_160_1')
			for i in range(2):
				next_x = self.residual(x,out_channels=160,multi=6,stride=1,is_training=self.is_training,scope='residual_160_'+str(i+2))
				mask = self.skip_layer(x,scope='skip_layer_160_'+str(i+1))
				x = mask*next_x+(one_tensor-mask)*x
				self.layer_masks.append(tf.reshape(mask,[-1]))

			x = self.residual(x,out_channels=320,multi=6,stride=1,is_training=self.is_training,scope='residual_320_1')

			x = slim.conv2d(x,128,[1,1],1,padding='SAME',activation_fn=None,scope='conv2')
			x = tf.nn.relu6(self.batch_norm(x,is_training,scope='bn2'))

			x = slim.conv2d(x,self.final_channels,[1,1],1,padding='SAME',activation_fn=None,scope='conv3')

			self.model_output = tf.reshape(x,[-1,x.shape[1].value*x.shape[2].value,self.anchors_per_unit,8],name='model_output')


	def residual(self,x,out_channels,multi=6,stride=1,is_training=True,scope='residual'):
		in_channels = x.shape[-1].value

		if stride==1:
			orig_x = x

			with tf.variable_scope(scope):
				x = slim.conv2d(x,in_channels*multi,[1,1],1,padding='SAME',activation_fn=None,scope='rconv1')
				x = tf.nn.relu6(self.batch_norm(x,is_training,scope='rbn1'))

				with tf.variable_scope('depthwise_conv'):
					dw_filter = tf.get_variable('dw_filter',[3,3,in_channels*multi,1],initializer=tf.truncated_normal_initializer(stddev=0.1))
					x = tf.nn.depthwise_conv2d(x,dw_filter,strides=[1,1,1,1],padding='SAME')
					x = tf.nn.relu6(self.batch_norm(x,is_training,scope='rbn2'))

				x = slim.conv2d(x,out_channels,[1,1],1,padding='SAME',activation_fn=None,scope='rconv3')

				if in_channels != out_channels:
					orig_x = slim.conv2d(orig_x,out_channels,[1,1],1,padding='SAME',activation_fn=None,scope='orig_conv')

				x = x+orig_x

		else:
			with tf.variable_scope(scope):
				x = slim.conv2d(x,in_channels*multi,[1,1],1,padding='SAME',activation_fn=None,scope='rconv1')
				x = tf.nn.relu6(self.batch_norm(x,is_training,scope='rbn1'))

				with tf.variable_scope('depthwise_conv'):
					dw_filter = tf.get_variable('dw_filter',[3,3,in_channels*multi,1],initializer=tf.truncated_normal_initializer(stddev=0.1))
					x = tf.nn.depthwise_conv2d(x,dw_filter,strides=[1,stride,stride,1],padding='SAME')
					x = tf.nn.relu6(self.batch_norm(x,is_training,scope='rbn2'))

				x = slim.conv2d(x,out_channels,[1,1],1,padding='SAME',activation_fn=None,scope='rconv3')

		return x


	def skip_layer(self,feature_map,scope='skip_layer'):
		x = feature_map
		with tf.variable_scope(scope):
			x = slim.avg_pool2d(x,[x.shape[1].value, x.shape[2].value],stride=1,padding='VALID',scope='avg_pool')
			x = slim.conv2d(x,32,[1,1],1,padding='SAME',activation_fn=None,scope='conv')
			x = tf.reshape(x,[-1,1,32])

			x,state = tf.nn.dynamic_rnn(cell=self.lstm_cell,inputs=x,dtype=tf.float32,scope='rnn')

			x = tf.reshape(x,[-1,1,1,1])
			zero_tensor = tf.zeros_like(x)
			one_tensor = tf.ones_like(x)

			return tf.nn.sigmoid(x)+tf.stop_gradient(tf.where(x>0,one_tensor,zero_tensor)-tf.nn.sigmoid(x))


	def batch_norm(self,x,is_training=True,scope='bn',moving_decay=0.9,eps=1e-6):
		with tf.variable_scope(scope):
			gamma = tf.get_variable('gamma',x.shape[-1],initializer=tf.constant_initializer(1))
			beta  = tf.get_variable('beta', x.shape[-1],initializer=tf.constant_initializer(0))

			axes = list(range(len(x.shape)-1))
			batch_mean, batch_var = tf.nn.moments(x,axes,name='moments')

			ema = tf.train.ExponentialMovingAverage(moving_decay)

			def mean_var_with_update():
				ema_apply_op = ema.apply([batch_mean,batch_var])
				with tf.control_dependencies([ema_apply_op]):
					return tf.identity(batch_mean), tf.identity(batch_var)

			mean, var = tf.cond(tf.equal(is_training,True),mean_var_with_update,
					lambda:(ema.average(batch_mean),ema.average(batch_var)))

			return tf.nn.batch_normalization(x,mean,var,beta,gamma,eps)


	def compute_loss(self):
		anchors = tf.reshape(self.anchors,[1,1,self.anchors_per_unit,2])
		sprob,sconf,snoob,scoor = self.scales

		_coords = self.labels[:,:,:,0:4]
		_confs = self.labels[:,:,:,4]
		_wh = tf.pow(_coords[:,:,:,2:4],2)*np.reshape(self.output_size,[1,1,1,2])
		_area = _wh[:,:,:,0]*_wh[:,:,:,1]
		_centers = _coords[:,:,:,0:2]
		_left_top,_right_bottom = _centers-_wh/2,_centers+_wh/2

		coords = self.model_output[:,:,:,0:4]
		coords_xy = tf.nn.sigmoid(coords[:,:,:,0:2])
		coords_wh = tf.sqrt(tf.exp(coords[:,:,:,2:4])*anchors)
		coords = tf.concat([coords_xy,coords_wh],axis=3)
		
		confs = tf.nn.sigmoid(self.model_output[:,:,:,4])
		confs = tf.expand_dims(confs,-1)
		
		probs = tf.nn.softmax(self.model_output[:,:,:,5:])

		self.predict = tf.concat([coords,confs,probs],axis=3,name='predict')

		wh = tf.pow(coords[:,:,:,2:4],2)*np.reshape(self.output_size,[1,1,1,2])
		area = wh[:,:,:,0]*wh[:,:,:,1]
		centers = coords[:,:,:,0:2]
		left_top,right_bottom = centers-wh/2,centers+wh/2

		inter_left_top = tf.maximum(left_top,_left_top)
		inter_right_bottom = tf.minimum(right_bottom,_right_bottom)
		inter_wh = tf.maximum(inter_right_bottom-inter_left_top,0.0)
		inter_area = inter_wh[:,:,:,0]*inter_wh[:,:,:,1]
		ious = tf.truediv(inter_area,area+_area-inter_area)

		best_iou_mask = tf.equal(ious,tf.reduce_max(ious,axis=2,keepdims=True))
		best_iou_mask = tf.cast(best_iou_mask,tf.float32)
		mask = best_iou_mask*_confs
		mask = tf.expand_dims(mask,-1)

		confs_w = snoob*(1-mask)+sconf*mask
		coords_w = scoor*mask
		probs_w = sprob*mask
		weights = tf.concat([coords_w,coords_w,coords_w,coords_w,confs_w,probs_w,probs_w,probs_w],axis=3)

		loss = tf.pow(self.predict-self.labels,2)*weights
		loss = tf.reduce_sum(loss, axis=[1, 2, 3])
		loss = 0.5*tf.reduce_mean(loss)

		return loss

def model_size():
	params = tf.trainable_variables()
	size = 0
	for x in params:
		sz = 1
		for dim in x.get_shape():
			sz *= dim.value
		size += sz
	return size


if __name__ == '__main__':
	images = tf.placeholder(tf.float32,[None,512,512,3],name='images')
	labels = tf.placeholder(tf.float32,[None,16*16,5,8],name ='labels')
	anchors = np.ones([5,2])

	with tf.Session() as sess:
		mobile = MobileYolo_skip(images,labels,anchors)
		sess.run(tf.global_variables_initializer())
		print(mobile.predict.shape.as_list())
		print('Size:',model_size())