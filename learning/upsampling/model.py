from __future__ import division
from __future__ import print_function
import os
import time
import tensorflow.compat.v1 as tf
from ops import *
from utils import *
import numpy as np
from skimage.measure import compare_psnr, compare_ssim
import matplotlib.pyplot as plt

class UMGFNet(object):

    def __init__(self, sess, param_set):
        self.sess           = sess
        self.phase          = param_set['phase']
        self.batch_size     = param_set['batch_size']
        self.inputI_width_size    = param_set['inputI_width_size']
        self.inputI_height_size    = param_set['inputI_height_size']
        self.r     = param_set['r']
        self.factor     = param_set['factor']
        self.niters     = param_set['niters']
        self.inputI_chn     = param_set['inputI_chn']
        self.output_chn     = param_set['output_chn']
        self.ImagePath  = param_set['ImagePath']
        self.DepthPath  = param_set['DepthPath']
        self.chkpoint_dir   = param_set['chkpoint_dir']
        self.lr             = param_set['learning_rate']
        self.epoch          = param_set['epoch']
        self.labeling_dir   = param_set['labeling_dir']
        self.result_dir   = param_set['result_dir']

        self.best_rmse = 1000
        self.best_ssim = 0
        self.model_name = "umgnet_depth_upscale-%d"%self.factor
        self.inputI_size = [self.inputI_width_size,self.inputI_height_size, self.inputI_chn]
        # build model graph
        self.build_model()


    def l1_loss(self, prediction, ground_truth):
        """
        :param prediction: the current prediction of the ground truth.
        :param ground_truth: the measurement you are approximating with regression.
        :return: mean of the l1 loss across all voxels.
        """
        absolute_residuals = tf.abs(tf.subtract(prediction, ground_truth))
        return tf.reduce_sum(tf.sqrt(absolute_residuals**2+1e-6))

    # build model graph
    def build_model(self):
        # input
        self.input_guide = tf.placeholder(dtype=tf.float32, shape=[None, None, None, self.inputI_chn], name='input_guide')
        self.input_target = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 1], name='input_target')
        self.input_gt = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 1], name='input_gt')

        # for our model
        self.pred_depth_list = self.umgf_model(self.input_guide, self.input_target,self.niters)#
        self.total_loss = 0
        for pred_depth in self.pred_depth_list:
            self.total_loss += self.l1_loss(pred_depth, self.input_gt)
        
        #for compared models. Here, we use svlr_model as an example
        #self.pred_depth = self.svlr_model(self.input_guide, self.input_target)
        #self.total_loss = self.l1_loss(self.pred_depth, self.input_gt)

        # trainable variables
        self.u_vars = tf.trainable_variables()
        # create model saver
        self.saver = tf.train.Saver(max_to_keep=100)

    #our model
    def umgf_model(self, input_guide, input_target, niters):
        # smoothing the traget image and guided image
        img_box = box_filter(input_target, self.r)
        g_img_box = box_filter(input_guide, self.r*2)

        #generating the unsharp masks
        guide_res = input_guide-g_img_box
        target_res = input_target-img_box

        #the network for dynamicly processing guided image to obtain task-Specific guidance map, which can be replaced by traditional operator such as tf.image.rgb_to_grayscale
        guide_conv1 = conv_lrelu(guide_res, output_chn=16, kernel_size=3, name = 'guide_conv1')
        guide_conv2 = conv2d(guide_conv1, output_chn=1, kernel_size=1, name = 'guide_conv2')

        #successive guided filtering network
        pred_depth_list = []
        target_concat = tf.concat([tf.image.rgb_to_grayscale(guide_res),target_res], axis=3, name='input_concat')
        for i in range(niters):
            target_conv, target_concat = amount_block(target_concat, output_chn=24, kernel_size=3, name = 'target_conv_concat%d'%i)
            pred_depth = target_conv*guide_conv2 + img_box
            pred_depth_list.append(pred_depth)

        return pred_depth_list

    # the model: Joint Image Filtering with Deep Convolutional Networks
    def djf_model(self, input_guide, input_target):

        guide_conv1 = conv_relu(input_guide, output_chn=96, kernel_size=9, name = 'guide_conv1')
        guide_conv2 = conv_relu(guide_conv1, output_chn=48, kernel_size=1, name = 'guide_conv2')
        guide_conv3 = conv2d(guide_conv2, output_chn=1, kernel_size=5, use_bias=True, name = 'guide_conv3')

        target_conv1 = conv_relu(input_target, output_chn=96, kernel_size=9, name = 'target_conv1')
        target_conv2 = conv_relu(target_conv1, output_chn=48, kernel_size=1, name = 'target_conv2')
        target_conv3 = conv2d(target_conv2, output_chn=1, kernel_size=5, use_bias=True, name = 'target_conv3')

        fusion_concat = tf.concat([guide_conv3, target_conv3], axis=3, name='fusion_concat')
        fusion_conv1 = conv_relu(fusion_concat, output_chn=64, kernel_size=9, name = 'fusion_conv1')
        fusion_conv2 = conv_relu(fusion_conv1, output_chn=32, kernel_size=1, name = 'fusion_conv2')
        pred_depth = conv2d(fusion_conv2, output_chn=1, kernel_size=5, use_bias=True, name = 'pred_depth')

        return pred_depth

    #the model: Spatially Variant Linear Representation Models for Joint Filtering
    def svlr_model(self, input_guide, input_target):

        input_fusion = tf.concat([input_guide, input_target], axis=3, name='input_fusion')
        conv1 = conv_lrelu(input_fusion, output_chn=64, kernel_size=3, name = 'conv1')
        conv2 = conv_lrelu(conv1, output_chn=64, kernel_size=3, name = 'conv2')
        conv3 = conv_lrelu(conv2, output_chn=64, kernel_size=3, name = 'conv3')
        conv4 = conv_lrelu(conv3, output_chn=64, kernel_size=3, name = 'conv4')
        conv5 = conv_lrelu(conv4, output_chn=64, kernel_size=3, name = 'conv5')
        conv6 = conv_lrelu(conv5, output_chn=64, kernel_size=3, name = 'conv6')
        conv7 = conv_lrelu(conv6, output_chn=64, kernel_size=3, name = 'conv7')
        conv8 = conv_lrelu(conv7, output_chn=64, kernel_size=3, name = 'conv8')
        conv9 = conv_lrelu(conv8, output_chn=64, kernel_size=3, name = 'conv9')
        conv10 = conv_lrelu(conv9, output_chn=64, kernel_size=3, name = 'conv10')
        conv11 = conv_lrelu(conv10, output_chn=64, kernel_size=3, name = 'conv11')
        conv12 = conv2d(conv11, output_chn=2, kernel_size=3, name = 'conv12')


        pred_depth = conv12[:,:,:,0,tf.newaxis] * input_guide + conv12[:,:,:,1,tf.newaxis]


        return pred_depth

    #the model: fast end-to-end trainable guided filter
    def dgf_model(self, input_guide, input_target):

        target_conv1 = conv_lrelu(input_target, output_chn=24, kernel_size=3, dilation=(1,1), name = 'target_conv1')
        target_conv2 = conv_lrelu(target_conv1, output_chn=24, kernel_size=3, dilation=(1,1), name = 'target_conv2')
        target_conv3 = conv_lrelu(target_conv2, output_chn=24, kernel_size=3, dilation=(2,2), name = 'target_conv3')
        target_conv4 = conv_lrelu(target_conv3, output_chn=24, kernel_size=3, dilation=(4,4), name = 'target_conv4')
        target_conv5 = conv_lrelu(target_conv4, output_chn=24, kernel_size=3, dilation=(8,8), name = 'target_conv5')
        target_conv6 = conv_lrelu(target_conv5, output_chn=24, kernel_size=3, dilation=(16,16), name = 'target_conv6')
        target_conv7 = conv_lrelu(target_conv6, output_chn=24, kernel_size=3, dilation=(1,1), name = 'target_conv7')
        target_conv8 = conv2d(target_conv7,    output_chn=1,  kernel_size=1, dilation=(1,1), name = 'target_conv8')

        guide_conv1 = conv_lrelu(input_guide, output_chn=16, kernel_size=3, name = 'guide_conv1')
        guide_conv2 = conv2d(guide_conv1, output_chn=1, kernel_size=1, name = 'guide_conv2')


        pred_depth = guided_filter(guide_conv2, target_conv8, r=2, eps=1e-2)


        return pred_depth

    # train function
    def train(self):

        u_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.total_loss, var_list=self.u_vars)
        # initialization
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        log_file = open("%s/%s_log.txt"%(self.result_dir,self.model_name), "w")

        if self.load_chkpoint(self.chkpoint_dir):
            print(" [*] Load SUCCESS\n")
            log_file.write(" [*] Load SUCCESS\n")
        else:
            print(" [!] Load failed...\n")
            log_file.write(" [!] Load failed...\n")

        # load all volume files
        guide_img, input_depth, gt_depth = load_data_pairs(self.ImagePath, self.DepthPath, self.factor)
        
        train_guide_img = guide_img[:1000]
        train_input_depth = input_depth[:1000]
        train_gt_depth = gt_depth[:1000]
        test_guide_img = guide_img[1000:]
        test_input_depth = input_depth[1000:]
        test_gt_depth = gt_depth[1000:]

        self.test_training(test_guide_img, test_input_depth, test_gt_depth, 0, log_file)

        rand_idx = np.arange(1000)
        start_time = time.time()

        for epoch in np.arange(self.epoch):
            np.random.shuffle(rand_idx)
            epoch_total_loss = 0.0
            for i_dx in rand_idx:
                # train batch
                batch_guide, batch_input, batch_gt = get_batch_patches(train_guide_img[i_dx], train_input_depth[i_dx], train_gt_depth[i_dx], self.inputI_size, self.batch_size)
                _, cur_train_loss = self.sess.run([u_optimizer, self.total_loss], feed_dict={self.input_guide: batch_guide, self.input_target: batch_input, self.input_gt: batch_gt})

                epoch_total_loss += cur_train_loss

            print("Epoch: [%d] time: %4.4f, train_loss: %.8f" % (epoch+1, time.time() - start_time, epoch_total_loss/1000.0))
            log_file.write("Epoch: [%d] time: %4.4f, train_loss: %.8f\n" % (epoch+1, time.time() - start_time, epoch_total_loss/1000.0))
            log_file.flush()
            start_time = time.time()

            if epoch+1 > 0:
                self.test_training(test_guide_img, test_input_depth, test_gt_depth, epoch+1, log_file)

        log_file.close()

    # for our model
    def test_training(self, test_guide_img, test_input_depth, test_gt_depth, step, log_file):
        n,w,h,c = test_guide_img.shape
        all_rmse_list = np.zeros([self.niters,n])
        all_ssim_list = np.zeros([self.niters,n])
        for k in range(0, n):
            
            if self.inputI_chn<3:
                test_guide_img_ins = rgb2gray(test_guide_img[k]).reshape(1,w,h,1)
            else:
                test_guide_img_ins = test_guide_img[k].reshape(1,w,h,c)
            test_input_depth_ins = test_input_depth[k].reshape(1,w,h,1)

            run_objs = self.pred_depth_list
            feed_dicts = {self.input_guide: test_guide_img_ins, self.input_target: test_input_depth_ins}
            pred_depth_list  = self.sess.run(run_objs, feed_dict=feed_dicts)
            gt_depth_scaled = test_gt_depth[k,6:-6,6:-6]*1000.0
            data_range = np.max(gt_depth_scaled)-np.min(gt_depth_scaled)
            for pred_depth,_iter in zip(pred_depth_list,range(self.niters)):
                pre_depth_scaled = pred_depth[0,6:-6,6:-6,0]*1000.0
                all_rmse_list[_iter,k] = np.sqrt(np.mean(pow(pre_depth_scaled-gt_depth_scaled,2)))
                all_ssim_list[_iter,k] = compare_ssim(gt_depth_scaled,pre_depth_scaled,data_range=data_range)

        mean_rmse_list = np.mean(all_rmse_list, axis=1)
        mean_ssim_list = np.mean(all_ssim_list, axis=1)
        if mean_rmse_list[-1]<self.best_rmse:
            self.best_rmse = mean_rmse_list[-1]
            self.best_ssim = mean_ssim_list[-1]
            self.save_chkpoint(self.chkpoint_dir, self.model_name, 0)

        print('Epoch: [%d],rmses:'%step, end='')
        print(''.join('%0.2f,'%mean_rmse_list[k] for k in range(self.niters)))
        print('best_rmse: %0.2f'%self.best_rmse)

        print('Epoch: [%d],ssim:'%step, end='')
        print(''.join('%0.4f,'%mean_ssim_list[k] for k in range(self.niters)))
        print('best_ssim: %0.4f'%self.best_ssim)

        log_file.write('Epoch: [%d],rmses:'%step)
        log_file.write(''.join('%0.2f,'%mean_rmse_list[k] for k in range(self.niters)))
        log_file.write('best_rmse: %0.2f\n'%self.best_rmse)

        log_file.write('Epoch: [%d],ssim:'%step)
        log_file.write(''.join('%0.4f,'%mean_ssim_list[k] for k in range(self.niters)))
        log_file.write('best_ssim: %0.4f\n'%self.best_ssim)

        log_file.flush()
   
    # for compared model
    """def test_training(self, test_guide_img, test_input_depth, test_gt_depth, step, log_file):
        n,w,h,c = test_guide_img.shape
        all_rmse = np.zeros([n])
        all_ssim = np.zeros([n])
        for k in range(0, n):
            #print (k+1)
            if self.inputI_chn<3:
                test_guide_img_ins = rgb2gray(test_guide_img[k]).reshape(1,w,h,1)#test_guide_img[k].reshape(1,w,h,c)##
            else:
                test_guide_img_ins = test_guide_img[k].reshape(1,w,h,c)##
            test_input_depth_ins = test_input_depth[k].reshape(1,w,h,1)

            pred_depth  = self.sess.run(self.pred_depth, feed_dict={self.input_guide: test_guide_img_ins, self.input_target: test_input_depth_ins})

            pre_depth_scaled = pred_depth[0,6:-6,6:-6,0]*1000.0
            gt_depth_scaled = test_gt_depth[k,6:-6,6:-6]*1000.0
            all_rmse[k] = np.sqrt(np.mean(pow(pre_depth_scaled-gt_depth_scaled,2)))
            data_range = np.max(gt_depth_scaled)-np.min(gt_depth_scaled)
            all_ssim[k] = compare_ssim(gt_depth_scaled,pre_depth_scaled,data_range=data_range)

        mean_rmse = np.mean(all_rmse, axis=0)
        mean_ssim = np.mean(all_ssim, axis=0)
        print("Epoch: [%d], rmse:%0.2f, ssim:%0.4f\n"%(step, mean_rmse, mean_ssim))
        log_file.write("Epoch: [%d], rmse:%0.2f, ssim:%0.4f\n"%(step, mean_rmse, mean_ssim))
        log_file.flush()"""

    def test(self):

        print("Starting test Process:\n")

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        if self.load_chkpoint(self.chkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        guide_img, input_depth, gt_depth = load_data_pairs(self.ImagePath, self.DepthPath, self.factor)
        test_guide_img = guide_img[1000:]
        test_input_depth = input_depth[1000:]
        test_gt_depth = gt_depth[1000:]

        n,w,h,c = test_guide_img.shape
        all_rmse = np.zeros([n])
        all_ssim = np.zeros([n])
        for k in range(n):
            if self.inputI_chn<3:
                test_guide_img_ins = rgb2gray(test_guide_img[k]).reshape(1,w,h,1)
            else:
                test_guide_img_ins = test_guide_img[k].reshape(1,w,h,c)
            test_input_depth_ins = test_input_depth[k].reshape(1,w,h,1)

            pred_depth_list = self.sess.run(self.pred_depth_list, feed_dict={self.input_guide: test_guide_img_ins, self.input_target: test_input_depth_ins})
            pred_depth = pred_depth_list[-1]
            pre_depth_scaled = pred_depth[0,6:-6,6:-6,0]*1000.0
            gt_depth_scaled = test_gt_depth[k,6:-6,6:-6]*1000.0

            data_range = np.max(gt_depth_scaled)-np.min(gt_depth_scaled)
            all_rmse[k] = np.sqrt(np.mean(pow(pre_depth_scaled-gt_depth_scaled,2)))
            all_ssim_list[k] = compare_ssim(gt_depth_scaled,pre_depth_scaled,data_range=data_range)
            
            plt.imsave('./test_labeling/dgf/depth_%04d.png'%(k+1), pred_depth[0,:,:,0]*10.0, cmap = plt.cm.jet, vmin=np.min(test_gt_depth[k])*10.0, vmax=np.max(test_gt_depth[k])*10.0)         

        mean_rmse = np.mean(all_rmse, axis=0)
        mean_ssim = np.mean(all_ssim, axis=0)
        print("rmse:%0.2f, ssim:0.4f\n"%(mean_rmse,mean_ssim))


    # save checkpoint file
    def save_chkpoint(self, checkpoint_dir, model_name, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)

    # load checkpoint file
    def load_chkpoint(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        checkpoint_dir = os.path.join(checkpoint_dir, self.model_name)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False
