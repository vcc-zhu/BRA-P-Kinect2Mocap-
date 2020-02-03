# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 15:37:25 2019

@author: WINDOWS10-3
"""

import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import time
start = time.time()
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
tf.reset_default_graph()
lr = 0.00001
#training_iters = 100000
length_of_clip = 120
rng = np.random.RandomState(20000)#20000是伪随机数的种子
#rng = np.random.RandomState(23455)z

batch_size = 16



guass_mean1_path='./Kinect2mocap_BRA-p_keepprob80/b0.0005s0.0002h1'
#读取数据
X_origi= np.load('./train_data/rotate_mocap_120_train_August_shuffle_RootFix.npz')['X']#(3n,120,73)
X_noise = np.load('./train_data/rotate_kinect_120_train_August_shuffle_RootFix.npz')['X']

Xmean_noise = X_noise.mean(axis=1).mean(axis=0)[np.newaxis,np.newaxis,:]
Xstd_noise = np.array([[[X_noise.std()]]]).repeat(X_noise.shape[2], axis=2)
#print('数量',I)
np.savez_compressed('preprocess_core_kinect_high_noise_August_RootFix.npz', Xmean=Xmean_noise, Xstd=Xstd_noise)
X_noise=(X_noise-Xmean_noise)/Xstd_noise#(3n,73,120)


Xmean_origi = X_origi.mean(axis=1).mean(axis=0)[np.newaxis,np.newaxis,:]
Xstd_origi = np.array([[[X_origi.std()]]]).repeat(X_origi.shape[2], axis=2)
#print('数量',I)
np.savez_compressed('preprocess_core_dongbu_high_noise_August_RootFix.npz', Xmean=Xmean_origi, Xstd=Xstd_origi)
X_origi=(X_origi-Xmean_origi)/Xstd_origi#(3n,73,120)
  
     
print('X_origi.shape[0]',X_origi.shape[0])                                      
data_file_number=X_origi.shape[0]#3n


n_steps = length_of_clip 

n_inputs = 75
n_encoder_fc1 = 128
n_encoder_fc2 = 256
n_encoder_fc3 = 512

n_decoder_fc1 = 512
n_decoder_fc2 = 256
n_outputs= 177

n_hidden_units = 512 # neurons in hidden layer

number_of_layers = 1

x = tf.placeholder(tf.float32, [None, n_steps, n_inputs],name='x')
y = tf.placeholder(tf.float32, [None, n_steps, n_outputs],name='y')
recn = tf.placeholder(tf.float32, [None, n_steps, n_outputs],name='out')

# Define weights
def fc(input_motion,w1_shape,w2_shape,name='fc'):
    with tf.variable_scope(name):
        w=tf.get_variable('weight',initializer=tf.random_normal([w1_shape,w2_shape],stddev=0.1))
        b=tf.get_variable('bias',initializer=tf.constant(0.001, shape=[w2_shape, ]))
        return tf.matmul(input_motion, w) + b

def cmu_AED_encoder(X,reuse=False,name="recon_encoder"):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        
# =============================================================================
#         X = (X-cmu_DataMean)/cmu_DataStd
# 
# =============================================================================
        X = tf.reshape(X, [-1, n_inputs])#(?, 66)
        e1 = fc(X,n_inputs, n_encoder_fc1,name='e1')#(?, 128)
        e2 = fc(e1,n_encoder_fc1, n_encoder_fc2,name='e2') #(?, 256) 
        e3 = fc(e2,n_encoder_fc2, n_encoder_fc3,name='e3') #(?, 512) 
        e3 = tf.reshape(e3, [-1, n_steps, n_hidden_units])#(?, 120, 256)
        # 定义双向LSTM——————————————————————————————————————————basic LSTM Cell.

        with tf.variable_scope('RNN1'):
            cell_fw = tf.contrib.rnn.BasicLSTMCell(n_hidden_units)
            dropout_cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob=0.8,output_keep_prob=0.8)

            cell_bw = tf.contrib.rnn.BasicLSTMCell(n_hidden_units)
            dropout_cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, input_keep_prob=1)
        # lstm cell is divided into two parts (c_state, h_state)
        with tf.variable_scope('RNN1'):
            stacked_cell_fw = tf.contrib.rnn.MultiRNNCell([dropout_cell_fw for _ in range(number_of_layers)])
            stacked_cell_bw = tf.contrib.rnn.MultiRNNCell([dropout_cell_bw  for _ in range(number_of_layers)])
        with tf.variable_scope('RNN1'):
            (outputs, states) = tf.nn.bidirectional_dynamic_rnn(stacked_cell_fw, stacked_cell_bw, e3, dtype=tf.float32)
        outputs_1 = tf.concat(outputs, 2)  # (?, 120, 1024)
        # 定义双向LSTM——————————————————结束—————————————basic LSTM Cell.
        result = outputs_1[:,:,n_hidden_units:2*n_hidden_units]  # (?, 120, 512)
        return result

def cmu_AED_decoder(X,reuse=False,name="recon_decoder"):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        # hidden layer for input to cell  
        #X=(?, 120, 66)
        
        # 定义双向LSTM——————————————————————————————————————————basic LSTM Cell.

        with tf.variable_scope('RNN1'):
            cell_fw = tf.contrib.rnn.BasicLSTMCell(n_hidden_units)
            dropout_cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob=0.8,output_keep_prob=0.8)

            cell_bw = tf.contrib.rnn.BasicLSTMCell(n_hidden_units)
            dropout_cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, input_keep_prob=1)
        # lstm cell is divided into two parts (c_state, h_state)
        with tf.variable_scope('RNN1'):
            stacked_cell_fw = tf.contrib.rnn.MultiRNNCell([dropout_cell_fw for _ in range(number_of_layers)])
            stacked_cell_bw = tf.contrib.rnn.MultiRNNCell([dropout_cell_bw  for _ in range(number_of_layers)])
        with tf.variable_scope('RNN1'):
            (outputs, states) = tf.nn.bidirectional_dynamic_rnn(stacked_cell_fw, stacked_cell_bw, X, dtype=tf.float32)
        outputs_1 = tf.concat(outputs, 2)  # (?, 120, 1024)
        # 定义双向LSTM——————————————————结束—————————————basic LSTM Cell.
        hidden = outputs_1[:,:,n_hidden_units:2*n_hidden_units]  # (?, 120, 512)
        print("hidden",hidden.shape)
        decoder_1 = tf.reshape(hidden, [-1, n_hidden_units])# (?, 512)
        print("decoder_1",decoder_1.shape)
        d1 = fc(decoder_1,n_decoder_fc1, n_decoder_fc2,name='d1')
        print("d1",d1.shape)
        d2 = fc(d1,n_decoder_fc2, n_outputs,name='d2')# (?, 128)
#        d3 = fc(d2,n_decoder_fc3, n_outputs,name='d3')#(?, 66)
        d3 = tf.reshape(d2, [-1, n_steps, n_outputs])#(?, 120, 63)
#        d3=d3*cmu_DataStd+cmu_DataMean
        return d3


hidden=cmu_AED_encoder(x)
recon=cmu_AED_decoder(hidden)
loss_position = 0.5 * tf.reduce_mean(tf.pow(tf.subtract(recon,y),2.0))

def cmu_AED(X,reuse=False,name="Jingque_AutoEncoder"):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        X = tf.reshape(X, [-1, n_outputs])#(?, 66)
#        e1 = fc(X,n_inputs, n_encoder_fc1,name='e1')#(?, 128)
        e2 = fc(X,n_outputs, n_encoder_fc2,name='e2') #(?, 256) 
        e3 = fc(e2,n_encoder_fc2, n_encoder_fc3,name='e3') #(?, 512) 
        e3 = tf.reshape(e3, [-1, n_steps, n_hidden_units])#(?, 120, 256)
        # 定义双向LSTM——————————————————————————————————————————basic LSTM Cell.

        with tf.variable_scope('RNN1'):
            cell_fw_e = tf.contrib.rnn.BasicLSTMCell(n_hidden_units)
            dropout_cell_fw_e = tf.contrib.rnn.DropoutWrapper(cell_fw_e, input_keep_prob=0.8,output_keep_prob=0.8)

            cell_bw_e = tf.contrib.rnn.BasicLSTMCell(n_hidden_units)
            dropout_cell_bw_e = tf.contrib.rnn.DropoutWrapper(cell_bw_e, input_keep_prob=1)
        # lstm cell is divided into two parts (c_state, h_state)
        with tf.variable_scope('RNN1'):
            stacked_cell_fw_e = tf.contrib.rnn.MultiRNNCell([dropout_cell_fw_e for _ in range(number_of_layers)])
            stacked_cell_bw_e = tf.contrib.rnn.MultiRNNCell([dropout_cell_bw_e  for _ in range(number_of_layers)])
        with tf.variable_scope('RNN1'):
            (outputs_e, states_e) = tf.nn.bidirectional_dynamic_rnn(stacked_cell_fw_e, stacked_cell_bw_e, e3, dtype=tf.float32)
        outputs_1 = tf.concat(outputs_e, 2)  # (?, 120, 1024)
        # 定义双向LSTM——————————————————结束—————————————basic LSTM Cell.
        hidden_recon = outputs_1[:,:,n_hidden_units:2*n_hidden_units]  # (?, 120, 512)
       
 
        # 定义双向LSTM——————————————————————————————————————————basic LSTM Cell.

        with tf.variable_scope('RNN2'):
            cell_fw = tf.contrib.rnn.BasicLSTMCell(n_hidden_units)
            dropout_cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob=0.8,output_keep_prob=0.8)

            cell_bw = tf.contrib.rnn.BasicLSTMCell(n_hidden_units)
            dropout_cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, input_keep_prob=1)
        # lstm cell is divided into two parts (c_state, h_state)
        with tf.variable_scope('RNN2'):
            stacked_cell_fw_d = tf.contrib.rnn.MultiRNNCell([dropout_cell_fw for _ in range(number_of_layers)])
            stacked_cell_bw_d = tf.contrib.rnn.MultiRNNCell([dropout_cell_bw  for _ in range(number_of_layers)])
        with tf.variable_scope('RNN2'):
            (outputs_d, states_d) = tf.nn.bidirectional_dynamic_rnn(stacked_cell_fw_d, stacked_cell_bw_d, hidden_recon, dtype=tf.float32)
        outputs_2= tf.concat(outputs_d, 2)  # (?, 120, 1024)
        # 定义双向LSTM——————————————————结束—————————————basic LSTM Cell.
        hidden_d = outputs_2[:,:,n_hidden_units:2*n_hidden_units]  # (?, 120, 512)
        print("hidden",hidden_d.shape)
        decoder_1 = tf.reshape(hidden_d, [-1, n_hidden_units])# (?, 512)
        print("decoder_1",decoder_1.shape)
#        d1 = fc(decoder_1,n_decoder_fc1, n_decoder_fc2,name='d1')
#        print("d1",d1.shape)
        d2 = fc(decoder_1,n_decoder_fc1, n_decoder_fc2,name='d2')# (?, 128)
        d3 = fc(d2,n_decoder_fc2, n_outputs,name='d3')#(?, 66)
        d3 = tf.reshape(d3, [-1, n_steps, n_outputs])#(?, 120, 63)
#        d3=d3*cmu_DataStd+cmu_DataMean
        return d3,hidden_recon


recon1,hidden_recon=cmu_AED(recon,reuse=False)
_,hidden2_recon=cmu_AED(recon1,reuse=True)
recon_y,hidden_y=cmu_AED(y,reuse=True)

print('recon1.shape:',recon1.shape)
print('hidden_recon.shape:',hidden_recon.shape)

position2_loss=0.5 * tf.reduce_mean(tf.pow(tf.subtract(recon1,y),2.0))


hidden_loss=0.5 * tf.reduce_mean(tf.pow(tf.subtract(hidden_recon,hidden_y),2.0))
hidden2_loss=0.5 * tf.reduce_mean(tf.pow(tf.subtract(hidden2_recon,hidden_y),2.0))
def constrain_length(recon,Xmean_origi,Xstd_origi,length_of_clip,batch_size):

    print('recon.shape',recon.shape)
    
    recon = recon*Xstd_origi+Xmean_origi
    recn_1 = tf.reshape(recon,[-1,length_of_clip,59,3])
    recn_f = tf.concat([recn_1[:,:,0],recn_1[:,:,0],recn_1[:,:,1],recn_1[:,:,2],recn_1[:,:,0],recn_1[:,:,4],recn_1[:,:,5],recn_1[:,:,0],
                        recn_1[:,:,7],recn_1[:,:,8],recn_1[:,:,9],recn_1[:,:,10],recn_1[:,:,11],recn_1[:,:,10],recn_1[:,:,13],recn_1[:,:,14],
                        recn_1[:,:,15],recn_1[:,:,16],recn_1[:,:,17],recn_1[:,:,18],recn_1[:,:,16],recn_1[:,:,20],recn_1[:,:,21],recn_1[:,:,22],
                        recn_1[:,:,16],recn_1[:,:,24],recn_1[:,:,25],recn_1[:,:,26],recn_1[:,:,16],recn_1[:,:,28],recn_1[:,:,29],recn_1[:,:,30],
                        recn_1[:,:,16],recn_1[:,:,32],recn_1[:,:,33],recn_1[:,:,34],recn_1[:,:,10],recn_1[:,:,36],recn_1[:,:,37],
                        recn_1[:,:,38],recn_1[:,:,39],recn_1[:,:,40],recn_1[:,:,41],recn_1[:,:,39],recn_1[:,:,43],recn_1[:,:,44],
                        recn_1[:,:,45],recn_1[:,:,39],recn_1[:,:,47],recn_1[:,:,48],recn_1[:,:,49],recn_1[:,:,39],recn_1[:,:,51],
                        recn_1[:,:,52],recn_1[:,:,53],recn_1[:,:,39],recn_1[:,:,55],recn_1[:,:,56],recn_1[:,:,57]],2)
    recn_f = tf.reshape(recn_f,[-1,length_of_clip,59,3])
    length1=tf.sqrt(tf.reduce_sum((recn_1[:,:,1:]-recn_f[:,:,1:])**2,axis=3))
    print("length1.shape",length1.shape)    
    
    bvh_bone_length = [9.76355191515874,43.12,43.12,9.76355191515874,43.12,43.12,14.755,10.023,10.437,10.023,10.713,9.72,7.85708527381497,
                       13.1,27.25,26.75,4.11048853544199,3.788,2.631,3.92563498048405,5.46439164042988,3.723,2.111,3.60440688602161,
                       5.33248619313731,4.062,2.547,3.50742341327648,4.79248025556705,3.541,2.456,3.51140797971412,4.40437214594771,
                       2.835,1.792,7.85708527381497,13.1,27.25,26.75,4.11048853544199,3.788,2.631,3.92563498048405,5.46439164042988,
                       3.723,2.111,3.60440688602161,5.33248619313731,4.062,2.547,3.50742341327648,4.79248025556705,3.541,2.456,
                       3.51140797971412,4.40437214594771,2.835,1.792]
    bvh_bone_length = np.array(bvh_bone_length,dtype=np.float32)
    print('label_bone:',bvh_bone_length.shape)
    length2 = np.tile(bvh_bone_length,(batch_size,length_of_clip,1))
    
    print("length2.shape",length2.shape)
    loss_bone_longeth=tf.reduce_mean(tf.abs(length1-length2))

#    loss_bone_longeth=tf.reduce_mean((tf.sqrt(tf.reduce_sum((recn_1[:,:,1:]-recn_f[:,:,1:])**2,axis=3))-tf.sqrt(tf.reduce_sum((label_1[:,:,1:]-label_f[:,:,1:])**2,axis=3)))**2)
    return loss_bone_longeth


def smooth(n_steps,in_,inputs,batch_size):
#    in_=[128,120,63/84]=[batch_size,n_steps,inputs]    
    smooth_matrix1 = np.eye(n_steps+4,k = 1,dtype=np.float32)
    smooth_matrix2 =3*np.eye(n_steps+4, k = -1,dtype=np.float32)
    smooth_matrix3 = np.eye(n_steps+4, k = -2,dtype=np.float32)
    smooth_matrix4 = 3*np.eye(n_steps+4, dtype=np.float32)
    smooth_matrix = smooth_matrix4-smooth_matrix1-smooth_matrix2+smooth_matrix3    
    smooth_matrix[0,0]=1
    smooth_matrix[1,1]=-2
    smooth_matrix[1,0]=1
    smooth_matrix[1,2]=1
    smooth_matrix[n_steps+2,n_steps]=0
    smooth_matrix[n_steps+2,n_steps+1]=1
    smooth_matrix[n_steps+2,n_steps+2]=-2
    smooth_matrix[n_steps+2,n_steps+3]=1
    smooth_matrix[n_steps+3,n_steps+1]=0
    smooth_matrix[n_steps+3,n_steps+2]=1
    smooth_matrix[n_steps+3,n_steps+3]=-1
    smooth_matrix=-1*smooth_matrix
#    print('M:',smooth_matrix.shape)
#    for i in range(len(smooth_matrix)):
#        for j in range(len(smooth_matrix)):
#            print(smooth_matrix[i][j],end='/t')
#        print('/n')

    smooth_matrix_all3=np.tile(smooth_matrix,(batch_size,1))
    smooth_matrix_all3=smooth_matrix_all3.reshape([batch_size,smooth_matrix.shape[0],smooth_matrix.shape[1]])
    smooth_matrix_tensor= tf.convert_to_tensor(smooth_matrix_all3)#(batch,124,124)
    in_1 = tf.concat((tf.reshape(in_[:,0],shape=[batch_size,1,inputs]),in_),1)#(batch,121, 63)
    print(in_1.shape)
    in_1 = tf.concat((tf.reshape(in_[:,0],shape=[batch_size,1,inputs]),in_1),1)#(batch,122, 63)
    print(in_1.shape)
    in_1 = tf.concat((in_1,tf.reshape(in_[:,n_steps-1],shape=[batch_size,1,inputs])),1)#(batch,123, 63)
    print(in_1.shape)
    in_1 = tf.concat((in_1,tf.reshape(in_[:,n_steps-1],shape=[batch_size,1,inputs])),1)#(batch,124, 63)
    print(in_1.shape)

    matmul_result= tf.matmul(smooth_matrix_tensor,in_1)
    norm_all=0
    for i in range(matmul_result.shape[0]):
        each_norm=tf.norm(matmul_result[i],ord=2)
        norm_all+=each_norm
    smooth_loss=norm_all/batch_size
#    smooth_loss = tf.norm (tf.matmul(smooth_matrix_tensor,in_1),ord=2)   
#    smooth_loss=smooth_loss/batch_size
    return smooth_loss 



loss_bone_longeth=constrain_length(recon,Xmean_origi,Xstd_origi,length_of_clip,batch_size)
loss_bone_longeth2=constrain_length(recon1,Xmean_origi,Xstd_origi,length_of_clip,batch_size)
loss_smooth=smooth(length_of_clip,recon,n_outputs,batch_size)
loss_smooth_2 = smooth(length_of_clip,recon1,n_outputs,batch_size)

loss=loss_position+0.0005*loss_bone_longeth+0.0002*loss_smooth+1*hidden_loss
#+0*loss_smooth+0*loss_bone_longeth
train_op = tf.train.AdamOptimizer(lr).minimize(loss)

ps_out=guass_mean1_path+'/out'
ps2=os.path.exists(ps_out)
if(ps2==False):
    os.makedirs(ps_out)  

    
    
    
def savedata(arr,each,epoch,num):
    str0 = '{}'.format(str(each).zfill(3))
    str1 = '{}'.format(str(epoch).zfill(5))
    str2 = '{}'.format(str(num).zfill(5))
    str3 = ps_out+ "/"  +str0+'_'+str1 +'_'+str2 +'.txt'
    np.savetxt(str3, arr, delimiter=' ')
  
checkpoint_steps =5
#checkpoint_dir = './model/' 

checkpoint_dir = guass_mean1_path+'/model/' 
ps1=os.path.exists(checkpoint_dir )
if(ps1==False):
    os.makedirs(checkpoint_dir )  
# =============================================================================
t_vars = tf.trainable_variables() 
print("t_vars.len",len(t_vars))   
for var in t_vars: print(var.name)
    
recon_vars = [var for var in t_vars if 'recon' in var.name]
print("recon_vars.len",len(recon_vars))   
for var in recon_vars: print(var.name)  
recon_optim = tf.train.AdamOptimizer(lr).minimize(loss, var_list=recon_vars)

                
                
saver = tf.train.Saver(max_to_keep=800)
gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
config=tf.ConfigProto(gpu_options=gpu_options)
with tf.Session(config=config) as sess:
# with tf.Session( ) as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess,  './Kinect2mocap_BRA-P_keepprob80/b0.0005s0.0002h1/model/model.ckpt-500')
       
#    init_op = tf.global_variables_initializer() 
#    sess.run(init_op)
    #加载训练好的参数parameter
    variables = slim.get_variables_to_restore()
    variables_to_restore_CMU_parameter = [v for v in variables if 'Jingque_AutoEncoder' in v.name]
    saver_parameter_CMU = tf.train.Saver(variables_to_restore_CMU_parameter)
#    tf.get_variable(variables_to_restore_CMU_parameter, trainable=False)

   
    saver_parameter_CMU.restore(sess, './Kinect2mocap_perceptualautoencoder_layer512_keepprob80/b0.0005s0.0002/model/model.ckpt-500')
    print('load Done!')
    last_mean_loss = 0
    last_mean_loss_positon = 0
    last_mean_loss_smooth = 0
    last_mean_loss_bone = 0
    last_mean_loss_position2 = 0
    last_mean_loss_hidden = 0
    last_mean_loss_smooth2 = 0
    
    orgin_top = np.zeros([length_of_clip,3])
    orgin_last = np.zeros([length_of_clip,7])
    for epoch in range(500,600):
        epochtime1=time.time()
        last_mean_loss_list = []
        last_mean_loss_positon_list = []
        last_mean_loss_smooth_list = []
        last_mean_loss_bone_list = []
        last_mean_loss_bone2_list = []
        last_mean_loss_position2_list = []
        last_mean_loss_hidden_list = []
        last_mean_loss_hidden2_list = []
        last_mean_loss_smooth2_list = []
        
        for each in range(int(data_file_number/batch_size)):#每个epoch
            ls,_,Xnoise,Xrecon,Xrecon1,Xlabel,loss_position_value,loss_bone_longeth_value,loss_bone_longeth2_value,loss_smooth_value,position2_loss_value,hidden_loss_value,hidden2_loss_value,smooth2_value = sess.run([loss,recon_optim,x,recon,recon1,y,loss_position,loss_bone_longeth,loss_bone_longeth2,loss_smooth,position2_loss,hidden_loss,hidden2_loss,loss_smooth_2], feed_dict={x:X_noise[each*batch_size:(each+1)*batch_size,:,:],y:X_origi[each*batch_size:(each+1)*batch_size,:,:]})
            
            
            last_mean_loss_list.append([ls])            
            last_mean_loss_positon_list.append([loss_position_value])
            last_mean_loss_smooth_list.append([loss_smooth_value])
            last_mean_loss_bone_list.append([loss_bone_longeth_value]) 
            last_mean_loss_bone2_list.append([loss_bone_longeth2_value])
            last_mean_loss_position2_list.append([position2_loss_value])
            last_mean_loss_hidden_list.append([hidden_loss_value])
            last_mean_loss_hidden2_list.append([hidden2_loss_value])
            last_mean_loss_smooth2_list.append([smooth2_value])

            
            if (each%200==0):
                print('epoch:',epoch,'each:',each,'loss:',ls,'loss_p:',loss_position_value,'loss_b',loss_bone_longeth_value,'loss_s',loss_smooth_value,'loss_p2',position2_loss_value,'hidden_loss:',hidden_loss_value,'smooth2:',smooth2_value,'loss_bone_longet2:',loss_bone_longeth2_value)
                Xnoise = (Xnoise*Xstd_noise) + Xmean_noise
                
                Xrecon = (Xrecon*Xstd_origi) + Xmean_origi

                Xlabel = (Xlabel*Xstd_origi) + Xmean_origi
                Xrecon1 = (Xrecon1*Xstd_origi) + Xmean_origi
                    
                savedata(Xnoise[0],epoch,each,1)
                savedata(Xrecon[0],epoch,each,2)
                savedata(Xlabel[0],epoch,each,4)
                savedata(Xrecon1[0],epoch,each,3)
#                
        #        savedata_loss(losslist,epoch)
        
        curr_mean = np.mean(last_mean_loss_list)
        curr_mean_position = np.mean(last_mean_loss_positon_list)
        curr_mean_smooth = np.mean(last_mean_loss_smooth_list)
        curr_mean_bone = np.mean(last_mean_loss_bone_list)
        curr_mean_position2 = np.mean(last_mean_loss_position2_list)
        curr_mean_hidden = np.mean(last_mean_loss_hidden_list)
        curr_mean_hidden2 = np.mean(last_mean_loss_hidden2_list)
        curr_mean_smooth2 = np.mean(last_mean_loss_smooth2_list)
        curr_mean_bone2 = np.mean(last_mean_loss_bone2_list)
        
        diff_mean, last_mean_loss = curr_mean-last_mean_loss, curr_mean
        diff_mean_positon, last_mean_loss_positon = curr_mean_position-last_mean_loss_positon, curr_mean_position
        diff_mean_smooth, last_mean_loss_smooth = curr_mean_smooth-last_mean_loss_smooth, curr_mean_smooth
        diff_mean_bone, last_mean_loss_bone = curr_mean_bone-last_mean_loss_bone, curr_mean_bone
        diff_mean_position2, last_mean_loss_position2 = curr_mean_position2-last_mean_loss_position2, curr_mean_position2
        diff_mean_hidden, last_mean_loss_hidden = curr_mean_hidden-last_mean_loss_hidden, curr_mean_hidden
        diff_mean_smooth2, last_mean_loss_smooth2 = curr_mean_smooth2-last_mean_loss_smooth2, curr_mean_smooth2
    
        
        print('curr_mean:',curr_mean,'diff_mean：',diff_mean)
        print('curr_mean_position:',curr_mean_position,'diff_mean_positon',diff_mean_positon)
        print('curr_mean_smooth:',curr_mean_smooth,'diff_mean_smooth',diff_mean_smooth)
        print('curr_mean_bone:',curr_mean_bone,'diff_mean_bone',diff_mean_bone)
        print('curr_mean_position2:',curr_mean_position2,'diff_mean_position2',diff_mean_position2)
        print('curr_mean_hidden:',curr_mean_hidden,'diff_mean_hidden',diff_mean_hidden)
        print('curr_mean_smooth2:',curr_mean_smooth2,'diff_mean_smooth2',diff_mean_smooth2)

    
        f1 = open(guass_mean1_path+'/mean_loss.txt','a')
        f2 = open(guass_mean1_path+'/mean_loss_position.txt','a')
        f3 = open(guass_mean1_path+'/mean_loss_smooth.txt','a')
        f4 = open(guass_mean1_path+'/mean_loss_bone.txt','a')
        f5 = open(guass_mean1_path+'/mean_loss_position2.txt','a')
        f6 = open(guass_mean1_path+'/mean_loss_hidden.txt','a')
        f7 = open(guass_mean1_path+'/mean_loss_smooth2.txt','a')
        f8 = open(guass_mean1_path+'/mean_loss_bone2.txt','a')
        f9 = open(guass_mean1_path+'/mean_loss_hidden2.txt','a')

        curr_mean_w=str(curr_mean)
        curr_mean_position_w = str(curr_mean_position)
        curr_mean_smooth_w = str(curr_mean_smooth)
        curr_mean_bone_w = str(curr_mean_bone)
        curr_mean_position_w2 = str(curr_mean_position2)
        curr_mean_hidden_w = str(curr_mean_hidden)
        curr_mean_hidden2_w = str(curr_mean_hidden2)
        curr_mean_smooth2_w = str(curr_mean_smooth2)
        curr_mean_bone2_w = str(curr_mean_bone2)
    
        f1.write(curr_mean_w+'\n')
        f2.write(curr_mean_position_w+'\n')
        f3.write(curr_mean_smooth_w+'\n')
        f4.write(curr_mean_bone_w+'\n')
        f5.write(curr_mean_position_w2+'\n')
        f6.write(curr_mean_hidden_w+'\n')
        f7.write(curr_mean_smooth2_w+'\n')
        f8.write(curr_mean_bone2_w+'\n')
        f9.write(curr_mean_hidden2_w+'\n')
        
        f1.close()
        f2.close()
        f3.close()
        f4.close()
        f5.close()
        f6.close()
        f7.close()
        f8.close()
        f9.close()
        epochtime2=time.time()
        print('epoch_time: ',epochtime2-epochtime1)
        if (epoch + 1) % checkpoint_steps == 0:
            saver.save(sess, checkpoint_dir + 'model.ckpt', global_step=epoch+1)  
        end = time.time()
        print("程序运行时间是（秒）", (end - start)) 
    
       
                
                

