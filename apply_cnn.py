# Hank on Jul 2, 2021
# adapt to predict by Dong from Jul 20, 2021
# final version for processing Landsat 8 image
# mean_name and model_path add os.path.dirname(os.path.abspath(__file__))
# import apply_evaluation
# import importlib
# importlib.reload (apply_evaluation)
# apply_evaluation.predict_and_evaluate('000')

import os
import sys
from math import pi
import json
import importlib
import numpy as np

import rasterio
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
import landsat_metadata

PATCH_SIZE = 512
IMG_BANDS = 8

## load mean and stardard varition 
mean_name=os.path.dirname(os.path.abspath(__file__))+'/mean.std.no.fill.csv'
x_mean = 0
x_std = 1
if os.path.exists(mean_name):
    dat_temp = np.loadtxt(open(mean_name, "rb"), dtype='<U30', delimiter=",", skiprows=1)
    arr = dat_temp.astype(np.float64)
    # x_dim = arr.shape[0]-1
    x_mean,x_std = arr[:,0],arr[:,1]
else:
    print("Error !!!! mean file not exists " + mean_name)
##******************************************************************************************************************
## generate cloud percentage json 
def get_json (cloud_arr):
    """landsat8 and sentinel-2 has three classes: cloud, cloud shadow, clear, get json from qa result"""
    fill_num = np.count_nonzero(np.bitwise_and(np.right_shift(cloud_arr,8),1)==1)
    total_num = cloud_arr.size
    cloud_num = np.count_nonzero(np.bitwise_and(np.right_shift(cloud_arr,1),1)==1) 
    shadow_num = np.count_nonzero(np.bitwise_and(np.right_shift(cloud_arr,3),1)==1)
    edge_num = np.count_nonzero(np.bitwise_and(np.right_shift(cloud_arr,2),1)==1)
    water_num = np.count_nonzero(np.bitwise_and(np.right_shift(cloud_arr,5),1)==1)
    
    p_fill   = np.divide(fill_num, total_num) *100
    p_cloud  = np.divide(cloud_num,(total_num-fill_num)) *100
    p_shadow = np.divide(shadow_num,(total_num-fill_num)) *100
    p_edge   = np.divide(edge_num,(total_num-fill_num)) *100
    p_contm  = np.divide((cloud_num+shadow_num+edge_num), (total_num-fill_num)) *100
    p_water  = np.divide(water_num,(total_num-fill_num)) *100
    
    ojt = {
    "filled_percentage" : f"{p_fill:.2f}",
    "contaminated_percentage": f"{p_contm:.2f}",
    "cloud_percentage": f"{p_cloud:.2f}",
    "cloud_shadow_percentage": f"{p_shadow:.2f}",
    "cloud_shadow_edge_percentage": f"{p_edge:.2f}",
    "water_percentage": f"{p_water:.2f}"
    }    
    json_ojt = json.dumps(ojt, indent=5)
    return json_ojt
##******************************************************************************************************************
# custom_objects from model
class SpectralNorm(tf.keras.constraints.Constraint):
    def __init__(self, n_iter=5):
        self.n_iter = n_iter
    def call(self, input_weights):
        w = tf.reshape(input_weights, (-1, input_weights.shape[-1]))
        u = tf.random.normal((w.shape[0], 1))
        for _ in range(self.n_iter):
            v = tf.matmul(w, u, transpose_a=True)
            v /= tf.norm(v)
            u = tf.matmul(w, v)
            u /= tf.norm(u)
            spec_norm = tf.matmul(u, tf.matmul(w, v),    transpose_a=True)
        return input_weights/spec_norm
        
class SelfAttention(Layer):
    ## this name and kwargs are important for reload the model 
    ## see https://stackoverflow.com/questions/62280161/saving-keras-models-with-custom-layers
    def __init__(self, reduced_factor_x=1, reduced_factor=1, c_factor=8, bn=False, name=None, **kwargs):
        super(SelfAttention, self).__init__(name=name)
        self.reduced_factor = reduced_factor
        self.reduced_factor_x = reduced_factor_x
        self.c_factor = c_factor
        self.bn = bn
        super(SelfAttention, self).__init__(**kwargs)
    
    ## this config turns to important for saving and realoding the model 
    ## see https://stackoverflow.com/questions/62280161/saving-keras-models-with-custom-layers
    def get_config(self):
        config = super(SelfAttention, self).get_config()
        config.update({"reduced_factor_x": self.reduced_factor_x})
        config.update({"reduced_factor": self.reduced_factor})
        config.update({"c_factor": self.c_factor})
        return config
    
    def build(self, input_shape):
        n,h,w,c = input_shape
        self.conv_fy =  Conv2D(c//self.c_factor, 1, padding='same', name='Conv_fy', kernel_constraint=SpectralNorm(), kernel_initializer='he_normal')#  
        self.conv_gx =  Conv2D(c//self.c_factor, 1, padding='same', name='Conv_gx'  , kernel_constraint=SpectralNorm(), kernel_initializer='he_normal')#        
        if self.reduced_factor_x>1:
            self.conv_after_attn = Conv2DTranspose(c   , kernel_size=[self.reduced_factor_x, self.reduced_factor_x],strides=[self.reduced_factor_x, self.reduced_factor_x], padding='same', \
                name='Conv_After_Attn', kernel_constraint=SpectralNorm(), kernel_initializer='he_normal')# 
        
        else:
            self.conv_after_attn = Conv2D(c   , 1, padding='same', name='Conv_After_Attn', kernel_constraint=SpectralNorm(), kernel_initializer='he_normal')# 
        
        self.sigma = self.add_weight(shape=[1], initializer='zeros', trainable=True, name='sigma')
    
    def call(self, x, y):
        n, h, w, c = x.shape
        self.n_feats = h*w
        fy = self.conv_fy(y)        
        if self.reduced_factor_x>1:
            fy = tf.nn.max_pool2d(fy, ksize=self.reduced_factor_x, strides=self.reduced_factor_x, padding='VALID')
                    
        fy = tf.reshape(fy, (-1, self.n_feats//(self.reduced_factor_x**2), fy.shape[-1])) ## reduce this not good
        
        gx = self.conv_gx(x)        
        gx = tf.nn.max_pool2d(gx, ksize=self.reduced_factor, strides=self.reduced_factor, padding='VALID')
        gx = tf.reshape(gx, (-1, self.n_feats//(self.reduced_factor**2), gx.shape[-1]))
        
        hx = x
        hx = tf.nn.max_pool2d(hx, ksize=self.reduced_factor, strides=self.reduced_factor, padding='VALID')
        hx = tf.reshape(hx, (-1, self.n_feats//(self.reduced_factor**2), hx.shape[-1]))
        
        attn = tf.matmul(fy, gx, transpose_b=True)        
        attn = tf.nn.softmax(attn) 
        
        attn_g = tf.matmul(attn, hx)
        attn_g = tf.reshape(attn_g, (-1, (h//self.reduced_factor_x), (w//self.reduced_factor_x), attn_g.shape[-1]))
        if self.reduced_factor_x>1:
            attn_g = tf.image.resize(attn_g, [h,w])
        
        output = x + self.sigma * attn_g
        return output

class SelfAttention_reduce(Layer):
    ## this name and kwargs are important for reload the model 
    ## see https://stackoverflow.com/questions/62280161/saving-keras-models-with-custom-layers
    def __init__(self, reduced_factor_x=1, reduced_factor=1, c_factor=8, bn=False, name=None, **kwargs):
        super(SelfAttention_reduce, self).__init__(name=name)
        self.reduced_factor = reduced_factor
        self.reduced_factor_x = reduced_factor_x
        self.c_factor = c_factor
        self.bn = bn
        super(SelfAttention_reduce, self).__init__(**kwargs)
    
    ## this config turns to important for saving and realoding the model 
    ## see https://stackoverflow.com/questions/62280161/saving-keras-models-with-custom-layers
    def get_config(self):
        config = super(SelfAttention_reduce, self).get_config()
        config.update({"reduced_factor_x": self.reduced_factor_x})
        config.update({"reduced_factor": self.reduced_factor})
        config.update({"c_factor": self.c_factor})
        return config
    
    def build(self, input_shape):
        n,h,w,c = input_shape
        self.conv_fy =  Conv2D(c//self.c_factor, 1, padding='same', name='Conv_fy', kernel_constraint=SpectralNorm(), kernel_initializer='he_normal')#  
        self.conv_gx =  Conv2D(c//self.c_factor, 1, padding='same', name='Conv_gx'  , kernel_constraint=SpectralNorm(), kernel_initializer='he_normal')#  
        # self.conv_hx =  Conv2D(c               , 1, padding='same', name='Conv_hx'   , kernel_constraint=SpectralNorm(), kernel_initializer='he_normal')# v27_6  
        self.conv_hx =  Conv2D(c//self.c_factor, 1, padding='same', name='Conv_hx'   , kernel_constraint=SpectralNorm(), kernel_initializer='he_normal')# v27_7   
        if self.reduced_factor_x>1:
            self.conv_after_attn = Conv2DTranspose(c   , kernel_size=[self.reduced_factor_x, self.reduced_factor_x],strides=[self.reduced_factor_x, self.reduced_factor_x], padding='same', \
                name='Conv_After_Attn', kernel_constraint=SpectralNorm(), kernel_initializer='he_normal')# 
        else:
            self.conv_after_attn = Conv2D(c   , 1, padding='same', name='Conv_After_Attn', kernel_constraint=SpectralNorm(), kernel_initializer='he_normal')# 
        
        self.sigma = self.add_weight(shape=[1], initializer='zeros', trainable=True, name='sigma')
    
    def call(self, x, y):
        n, h, w, c = x.shape
        self.n_feats = h*w
        fy = self.conv_fy(y)
        # fy = tf.reshape(fy, (-1, self.n_feats, fy.shape[-1]))
        if self.reduced_factor_x>1:
            fy = tf.nn.max_pool2d(fy, ksize=self.reduced_factor_x, strides=self.reduced_factor_x, padding='VALID')
        
        fy = tf.reshape(fy, (-1, self.n_feats//(self.reduced_factor_x**2), fy.shape[-1])) ## reduce this not good
        
        gx = self.conv_gx(x)
        
        gx = tf.nn.max_pool2d(gx, ksize=self.reduced_factor, strides=self.reduced_factor, padding='VALID')
        gx = tf.reshape(gx, (-1, self.n_feats//(self.reduced_factor**2), gx.shape[-1]))
        
        hx = self.conv_hx(x)
        hx = tf.nn.max_pool2d(hx, ksize=self.reduced_factor, strides=self.reduced_factor, padding='VALID')
        hx = tf.reshape(hx, (-1, self.n_feats//(self.reduced_factor**2), hx.shape[-1]))
        
        attn = tf.matmul(fy, gx, transpose_b=True)
        # attn = tf.matmul(gx, fy, transpose_b=True) # do not change
        attn = tf.nn.softmax(attn) 
        
        attn_g = tf.matmul(attn, hx)
        attn_g = tf.reshape(attn_g, (-1, (h//self.reduced_factor_x), (w//self.reduced_factor_x), attn_g.shape[-1]))
        attn_g = self.conv_after_attn(attn_g)
        output = x + self.sigma * attn_g
        return output

## copied from https://chowdera.com/2021/08/20210803043343640U.html#Selfattention_3
class SelfAttention_reduce_sumsampling(Layer):
    ## this name and kwargs are important for reload the model 
    ## see https://stackoverflow.com/questions/62280161/saving-keras-models-with-custom-layers
    def __init__(self, reduced_factor_x=1, reduced_factor=1, c_factor=8, bn=False, name=None, **kwargs):
        super(SelfAttention_reduce_sumsampling, self).__init__(name=name)
        self.reduced_factor = reduced_factor
        self.reduced_factor_x = reduced_factor_x
        self.c_factor = c_factor
        self.bn = bn
        super(SelfAttention_reduce_sumsampling, self).__init__(**kwargs)
    
    ## this config turns to important for saving and realoding the model 
    ## see https://stackoverflow.com/questions/62280161/saving-keras-models-with-custom-layers
    def get_config(self):
        config = super(SelfAttention_reduce_sumsampling, self).get_config()
        config.update({"reduced_factor_x": self.reduced_factor_x})
        config.update({"reduced_factor": self.reduced_factor})
        config.update({"c_factor": self.c_factor})
        return config
    
    def build(self, input_shape):
        L2 = 1e-6
        reg = tf.keras.regularizers.l2(l=L2)
        n,h,w,c = input_shape
        self.conv_fy =  Conv2D(c//self.c_factor, self.reduced_factor, strides=(self.reduced_factor, self.reduced_factor), padding='same', name='Conv_fy', kernel_constraint=SpectralNorm(), kernel_initializer='he_normal', use_bias=False, kernel_regularizer=reg)#  
        self.conv_gx =  Conv2D(c//self.c_factor, self.reduced_factor, strides=(self.reduced_factor, self.reduced_factor), padding='same', name='Conv_gx', kernel_constraint=SpectralNorm(), kernel_initializer='he_normal', use_bias=False, kernel_regularizer=reg)#  
        self.conv_hx =  Conv2D(c//self.c_factor, self.reduced_factor, strides=(self.reduced_factor, self.reduced_factor), padding='same', name='Conv_hx', kernel_constraint=SpectralNorm(), kernel_initializer='he_normal', use_bias=False, kernel_regularizer=reg)# v27_7   
        if self.reduced_factor_x>1:
            self.conv_after_attn = Conv2DTranspose(c   , kernel_size=[self.reduced_factor_x, self.reduced_factor_x], strides=[self.reduced_factor_x, self.reduced_factor_x], padding='same', \
                name='Conv_After_Attn', kernel_constraint=SpectralNorm(), kernel_initializer='he_normal', use_bias=False, kernel_regularizer=reg)# 
        else:
            self.conv_after_attn = Conv2D(c   , 1, padding='same', name='Conv_After_Attn', kernel_constraint=SpectralNorm(), kernel_initializer='he_normal', use_bias=False, kernel_regularizer=reg)# 
        
        self.sigma = self.add_weight(shape=[1], initializer='zeros', trainable=True, name='sigma') # ,regularizer=reg
    
    def call(self, x, y):
        n, h, w, c = x.shape
        self.n_feats = h*w
        fy = self.conv_fy(y)
        # fy = tf.reshape(fy, (-1, self.n_feats, fy.shape[-1]))
        # if self.reduced_factor_x>1:
            # fy = tf.nn.max_pool2d(fy, ksize=self.reduced_factor_x, strides=self.reduced_factor_x, padding='VALID')
        
        fy = tf.reshape(fy, (-1, self.n_feats//(self.reduced_factor_x**2), fy.shape[-1])) ## reduce this not good
        
        gx = self.conv_gx(x)
        
        # gx = tf.nn.max_pool2d(gx, ksize=self.reduced_factor, strides=self.reduced_factor, padding='VALID')
        gx = tf.reshape(gx, (-1, self.n_feats//(self.reduced_factor**2), gx.shape[-1]))
        
        hx = self.conv_hx(x)
        # hx = tf.nn.max_pool2d(hx, ksize=self.reduced_factor, strides=self.reduced_factor, padding='VALID')
        hx = tf.reshape(hx, (-1, self.n_feats//(self.reduced_factor**2), hx.shape[-1]))
        
        attn = tf.matmul(fy, gx, transpose_b=True)
        # attn = tf.matmul(gx, fy, transpose_b=True) # do not change
        attn = tf.nn.softmax(attn) 
        
        attn_g = tf.matmul(attn, hx)
        attn_g = tf.reshape(attn_g, (-1, (h//self.reduced_factor_x), (w//self.reduced_factor_x), attn_g.shape[-1]))
        attn_g = self.conv_after_attn(attn_g)
        output = x + self.sigma * attn_g
        return output
##******************************************************************************************************************
# dn to toa step
def find_one_file (dir1, pattern=''):
    find_file = ''
    for root, dirs, files in os.walk(dir1):
        for file in files:            
            if pattern in file: 
                print(file)           
                find_file = os.path.join(root, file)    
    return find_file 

def as_int16(input, scale=10000):
    input2 = input*scale
    index=input2>=0
    not_index = np.logical_not(index)
    input2[index] = input2[index]+0.5
    input2[not_index] = input2[not_index]-0.5
    return input2.astype(np.int16)

SCALE_REFLECTANCE = 10000
SCALE_TEMPERATURE = 100

# https://www.usgs.gov/core-science-systems/nli/landsat/using-usgs-landsat-level-1-data-product
# toa reflectence  
def calculate_toa(dn8_bands,sz,l1_mtl,toa8_bands,is_mask,dn_filled=0):
    # sz[0,3000:3010,3000:3010]
    # toa10_bands[0,3000:3010,3000:3010] 
    # toa10_bands[:,3000:3010,3000:3010] 
    # dn10_bands[bandi,3000:3010,3000:3010] 
    # dn10_bands[bandi,1:10,0:10] 
    # temprad
    
    # for bandi in range(10):
        # is_mask[:,:] = np.logical_or(dn10_bands[bandi,:,:]!=dn_filled, is_mask[:,:])
    
    sub_index = is_mask
    # https://www.runoob.com/numpy/numpy-mathematical-functions.html
    cossz = np.cos(sz[0,sub_index]/100/180*pi)
    # band 1 
    bandi = 0
    # sub_index = dn10_bands[bandi,:,:]!=dn_filled
    # tempdn = dn10_bands[bandi,sub_index]
    # tmptoa = as_int16((l1_mtl.REFLECTANCE_MULT_BAND_1*tempdn+l1_mtl.REFLECTANCE_ADD_BAND_1)/cossz)
    # toa10_bands[bandi, sub_index] = as_int16((l1_mtl.REFLECTANCE_MULT_BAND_1*tempdn+l1_mtl.REFLECTANCE_ADD_BAND_1)/cossz)
    toa8_bands[bandi, sub_index] = as_int16((l1_mtl.REFLECTANCE_MULT_BAND_1*dn8_bands[bandi,sub_index]+l1_mtl.REFLECTANCE_ADD_BAND_1)/cossz)
    
    # band 2
    bandi = 1
    # sub_index = dn10_bands[bandi,:,:]!=dn_filled
    # cossz = np.cos(sz[0,sub_index]/100/180*pi)
    toa8_bands[bandi, sub_index] = as_int16((l1_mtl.REFLECTANCE_MULT_BAND_2*dn8_bands[bandi,sub_index]+l1_mtl.REFLECTANCE_ADD_BAND_2)/cossz)
    
    # band 3
    bandi = 2
    # sub_index = dn10_bands[bandi,:,:]!=dn_filled
    # cossz = np.cos(sz[0,sub_index]/100/180*pi)
    toa8_bands[bandi, sub_index] = as_int16((l1_mtl.REFLECTANCE_MULT_BAND_3*dn8_bands[bandi,sub_index]+l1_mtl.REFLECTANCE_ADD_BAND_3)/cossz)
    
    # band 4
    bandi = 3
    # sub_index = dn10_bands[bandi,:,:]!=dn_filled
    # cossz = np.cos(sz[0,sub_index]/100/180*pi)
    toa8_bands[bandi, sub_index] = as_int16((l1_mtl.REFLECTANCE_MULT_BAND_4*dn8_bands[bandi,sub_index]+l1_mtl.REFLECTANCE_ADD_BAND_4)/cossz)
    
    # band 5
    bandi = 4
    # sub_index = dn10_bands[bandi,:,:]!=dn_filled
    # cossz = np.cos(sz[0,sub_index]/100/180*pi)
    toa8_bands[bandi, sub_index] = as_int16((l1_mtl.REFLECTANCE_MULT_BAND_5*dn8_bands[bandi,sub_index]+l1_mtl.REFLECTANCE_ADD_BAND_5)/cossz)
    
    # band 6
    bandi = 5
    # sub_index = dn10_bands[bandi,:,:]!=dn_filled
    # cossz = np.cos(sz[0,sub_index]/100/180*pi)
    toa8_bands[bandi, sub_index] = as_int16((l1_mtl.REFLECTANCE_MULT_BAND_6*dn8_bands[bandi,sub_index]+l1_mtl.REFLECTANCE_ADD_BAND_6)/cossz)
    
    # band 7
    bandi = 6
    # sub_index = dn10_bands[bandi,:,:]!=dn_filled
    # cossz = np.cos(sz[0,sub_index]/100/180*pi)
    toa8_bands[bandi, sub_index] = as_int16((l1_mtl.REFLECTANCE_MULT_BAND_7*dn8_bands[bandi,sub_index]+l1_mtl.REFLECTANCE_ADD_BAND_7)/cossz)
    
    # band 8
    bandi = 7
    # sub_index = dn10_bands[bandi,:,:]!=dn_filled
    # cossz = np.cos(sz[0,sub_index]/100/180*pi)
    toa8_bands[bandi, sub_index] = as_int16((l1_mtl.REFLECTANCE_MULT_BAND_9*dn8_bands[bandi,sub_index]+l1_mtl.REFLECTANCE_ADD_BAND_9)/cossz)

def l8c2l1_dn2toa (ddir, BAND_N = 8, DN_FILLED = 0, TOA_FILLED = -32767):
    ##************************************************************************************
    QA_file = find_one_file (ddir, pattern='_QA_PIXEL.TIF')
    base_name = os.path.basename(QA_file)[0:40]
    qa_dataset = rasterio.open(QA_file)
    qa_band = qa_dataset.read()
    no_fill_mask = np.bitwise_and(qa_band,1)==0

    ##*********************************************
    ## TOA process 
    l1_mtl_file = find_one_file (ddir, pattern='_MTL.txt')
    importlib.reload(landsat_metadata)
    l1_mtl = landsat_metadata.landsat_metadata(l1_mtl_file)

    sz_file = find_one_file (ddir, pattern='_SZA.TIF') 
    sz_dataset = rasterio.open(sz_file)
    sz = sz_dataset.read()   #(band, height, width)

    ## 8 reflective bands & 2 thermal bands 

    dn8_bands = np.full([BAND_N, sz.shape[1], sz.shape[2]], fill_value=DN_FILLED, dtype=np.uint16) ## must use fill
    for bandi in range(BAND_N):
        if bandi<7:
            dn_file = find_one_file (ddir, pattern='_B'+str(bandi+1)+'.TIF')
        else:     
            dn_file = find_one_file (ddir, pattern='_B'+str(bandi+2)+'.TIF')
    
        dn_dataset = rasterio.open(dn_file)
        # dn = dn_dataset.read()
        dn8_bands[bandi,:,:] = dn_dataset.read()   # output of rasterio is (X,X,X)

    toa8_bands = np.full([BAND_N, sz.shape[1], sz.shape[2]], fill_value=TOA_FILLED, dtype=np.int16) ## must use fill
    # is_mask = np.full([sz.shape[1], sz.shape[2]], fill_value=False, dtype=bool) ## must use fill
    is_mask = no_fill_mask.reshape(sz.shape[1], sz.shape[2])
    calculate_toa(dn8_bands,sz,l1_mtl,toa8_bands,is_mask,dn_filled=DN_FILLED)

    ## Save TOA with combining all 10 bands 
    image_meta = dn_dataset.profile.copy()
    image_meta['count'] = BAND_N
    image_meta['dtype'] = 'int16'
    image_meta['compress'] ='lzw'
    toa_file = './' + base_name +'.TIF'
    with rasterio.open(toa_file, 'w', **image_meta) as dst:
        dst.write(toa8_bands)
    return toa8_bands

##*******************************************************************************************************************
# perpare label functions
UNIQUE_LABELS = [64, 128, 192, 255] 
def combine_thin_thick_123(fms_image):
    fms_image_out = fms_image.copy()
    # for ci in range(len(UNIQUE_LABELS)):
    fms_image_out[fms_image==2] = 3
    return fms_image_out
    
def covert_123_to_label(fms_image):
    fms_image_out = fms_image.copy()
    for ci in range(len(UNIQUE_LABELS)):
        fms_image_out[fms_image==ci] = UNIQUE_LABELS[ci]
    return fms_image_out

def set_bit (value, bit_index):
    return value | (1 << bit_index)
##*********************************************************************************************************************
# cloud mask to qa step    
def img_to_qa (parr):    
    cldmsk = parr == 255
    csmsk = parr == 64
    fillmsk = parr == 10

    addc_arr = np.zeros(parr.shape[0], dtype = np.int16)
    addr_arr = np.zeros(parr.shape[1], dtype = np.int16)

    left_arr1 = np.insert(np.delete(parr, 0, axis=0), -1, addr_arr, axis=0)
    
    left_up_arr1 = np.insert(np.delete(left_arr1, 0, axis=1), -1, addc_arr, axis=1)
    
    left_down_arr1 = np.insert(np.delete(left_arr1, -1, axis=1), 0, addc_arr, axis=1)

    rigt_arr1 = np.insert(np.delete(parr,-1, axis=0), 0, addr_arr, axis=0)
    
    rigt_up_arr1 = np.insert(np.delete(rigt_arr1, 0, axis=1), -1, addc_arr, axis=1)
    
    rigt_down_arr1 = np.insert(np.delete(rigt_arr1, -1, axis=1), 0, addc_arr, axis=1)

    up_arr1 = np.insert(np.delete(parr, 0, axis=1), -1, addc_arr, axis=1)

    down_arr1 = np.insert(np.delete(parr,-1, axis=1), 0, addc_arr, axis=1)

    edge_cmsk = (left_arr1==255)|(rigt_arr1==255)|(up_arr1==255)|(down_arr1==255)|(left_up_arr1==255)|(left_down_arr1==255)|(rigt_up_arr1==255)|(rigt_down_arr1==255)
    edge_cmsk[cldmsk] = 0
    
    edge_clds = (left_arr1==64)|(rigt_arr1==64)|(up_arr1==64)|(down_arr1==64)|(left_up_arr1==64)|(left_down_arr1==64)|(rigt_up_arr1==64)|(rigt_down_arr1==64) 
    edge_clds[csmsk] = 0
    
    clearmsk = (np.logical_not(cldmsk)) & (np.logical_not(csmsk)) & (np.logical_not(fillmsk)) & (np.logical_not(edge_cmsk)) & (np.logical_not(edge_clds)) 

    qaarr = np.zeros((parr.shape[0], parr.shape[1]), dtype = np.int16)

    qaarr[cldmsk]                                                  = set_bit(0b0000000000000000, 1) 
    qaarr[csmsk]                                                   = set_bit(0b0000000000000000, 3) 
    qaarr[clearmsk]                                                = set_bit(0b0000000000000000, 0)     
    qaarr[edge_cmsk & np.logical_not(np.logical_or(cldmsk,csmsk))] = set_bit(0b0000000000000000, 2)           
    qaarr[edge_clds & np.logical_not(np.logical_or(cldmsk,csmsk))] = set_bit(0b0000000000000000, 2)
    qaarr[fillmsk]                                                 = set_bit(0b0000000000000000, 8)

    return qaarr   

def img_to_qas (parr, fmask_dir):  
    """
    cloud, cloud shadow, and add cirrus, snow, water,  
    parr: cloud mask result: cloud=255, cloud shadow=64, clear=128, fill=10
    fmask: l8c2l1 qa band directory
    """
    qa_band = np.squeeze(rasterio.open(fmask_dir).read())
    
    watermsk = np.bitwise_and(np.right_shift(qa_band,7),1)==1
    
    snow_msk  = np.bitwise_and(np.right_shift(qa_band,5),1)==1
    snow_conf = np.bitwise_and(np.right_shift(qa_band,12),3)>=3
    snowmsk = np.logical_or(snow_msk,snow_conf)
    
    cirr_msk = np.bitwise_and(np.right_shift(qa_band,2),1)==1
    cirr_conf = np.bitwise_and(np.right_shift(qa_band,14),3)>=3
    cirrmsk = np.logical_or(cirr_msk,cirr_conf)
    
    ##******************************************************************        
    cldmsk = parr == 255
    csmsk = parr == 64
    fillmsk = parr == 10

    addc_arr = np.zeros(parr.shape[0], dtype = np.int16)
    addr_arr = np.zeros(parr.shape[1], dtype = np.int16)

    left_arr1 = np.insert(np.delete(parr, 0, axis=0), -1, addr_arr, axis=0)
    
    left_up_arr1 = np.insert(np.delete(left_arr1, 0, axis=1), -1, addc_arr, axis=1)
    
    left_down_arr1 = np.insert(np.delete(left_arr1, -1, axis=1), 0, addc_arr, axis=1)

    rigt_arr1 = np.insert(np.delete(parr,-1, axis=0), 0, addr_arr, axis=0)
    
    rigt_up_arr1 = np.insert(np.delete(rigt_arr1, 0, axis=1), -1, addc_arr, axis=1)
    
    rigt_down_arr1 = np.insert(np.delete(rigt_arr1, -1, axis=1), 0, addc_arr, axis=1)

    up_arr1 = np.insert(np.delete(parr, 0, axis=1), -1, addc_arr, axis=1)

    down_arr1 = np.insert(np.delete(parr,-1, axis=1), 0, addc_arr, axis=1)

    edge_cmsk = (left_arr1==255)|(rigt_arr1==255)|(up_arr1==255)|(down_arr1==255)|(left_up_arr1==255)|(left_down_arr1==255)|(rigt_up_arr1==255)|(rigt_down_arr1==255)
    edge_cmsk[fillmsk] = 0
    edge_cmsk[cldmsk] = 0
    
    edge_clds = (left_arr1==64)|(rigt_arr1==64)|(up_arr1==64)|(down_arr1==64)|(left_up_arr1==64)|(left_down_arr1==64)|(rigt_up_arr1==64)|(rigt_down_arr1==64) 
    edge_clds[fillmsk] = 0
    edge_clds[csmsk] = 0
    
    clearmsk = (np.logical_not(cldmsk)) & (np.logical_not(csmsk)) & (np.logical_not(fillmsk)) & (np.logical_not(cirrmsk)) & (np.logical_not(edge_cmsk)) & (np.logical_not(edge_clds)) 

    qaarr = np.zeros((parr.shape[0], parr.shape[1]), dtype = np.int16)

    qaarr[cirrmsk]   = np.bitwise_or(qaarr[cirrmsk] , 1<<6)
    qaarr[snowmsk]   = np.bitwise_or(qaarr[snowmsk] , 1<<4)
    qaarr[watermsk]  = np.bitwise_or(qaarr[watermsk], 1<<5)
    qaarr[cldmsk]    = np.bitwise_or(qaarr[cldmsk]  , 1<<1)   
    qaarr[csmsk]     = np.bitwise_or(qaarr[csmsk]   , 1<<3)
    qaarr[clearmsk]  = np.bitwise_or(qaarr[clearmsk], 1<<0)
    maskc = edge_cmsk & np.logical_not(np.logical_or(cldmsk,csmsk))
    qaarr[maskc] = np.bitwise_or(qaarr[maskc],        1<<2) 
    masks = edge_clds & np.logical_not(np.logical_or(cldmsk,csmsk))
    qaarr[masks]     = np.bitwise_or(qaarr[masks],    1<<2) 
    qaarr[fillmsk]   = np.bitwise_or(qaarr[fillmsk] , 1<<8)  
    return qaarr    
##********************************************************************************************************************
## This function is for generating patches 
## split_images function 
## toa_all: 10 bands TOA, 7 reflective bands, 1 cirrus and 2 thermal bands 
## bqa_toa: toa quality band
## samples_toa, lines_toa: dimensions of the toa image
## xoffset, yoffset: start points on the TOA images
## line_min_no_fill: minimized pixels = of no-filled values in patches (PATCH_SIZE indicates no fill)
## patch_step: PATCH overlapping 
## file_prefix: used to save data
## ref_profile: profile for geometry 

# line_min_no_fill=PATCH_SIZE
# patch_step=PATCH_SIZE
# ref_profile = rasterio.open(dnn_cld_file).profile.copy()
def split_toa_for_prediction (toa_all, bqa_toa, samples_toa, lines_toa, xoffset=0, yoffset=0, patch_size=512, is_norm=True,
                              line_min_no_fill=1, patch_step=PATCH_SIZE):                                 
    
    no_filled_toa = np.bitwise_and(bqa_toa,1)==0
    x_toa_BEGIN = 0
    y_toa_BEGIN = 0
    x_toa_END   = samples_toa
    y_toa_END   = lines_toa  
    # break;
    ## find y start with at least {line_min_no_fill} pixels non-filled both cloud and collection-1 
    y_toa_BEGIN_inner = 0; y_toa_END_inner = 0; is_begin_set = False; is_end_set = False 
    for iy_toa_start in range(y_toa_BEGIN,y_toa_END):
        filled_sta = no_filled_toa[0,iy_toa_start,x_toa_BEGIN:x_toa_END]
        if filled_sta.sum()>=line_min_no_fill and not is_begin_set:
            y_toa_BEGIN_inner = iy_toa_start
            is_begin_set = True 
            break
    
    ## find y end with at least {line_min_no_fill} pixels non-filled both cloud and collection-1 
    for iy_toa_start in range(y_toa_BEGIN_inner,y_toa_END):
        filled_sta = no_filled_toa[0,iy_toa_start,x_toa_BEGIN:x_toa_END]
        if filled_sta.sum()<line_min_no_fill and not is_end_set:
            y_toa_END_inner = iy_toa_start
            is_end_set = True 
            break
        elif line_min_no_fill==1 and filled_sta.sum()>0: ## fill TOA 
            for bi in range(IMG_BANDS):
                # fixed a bug on Jul 24 as each band has different widths (thermal & reflective)
                index_sta = np.argwhere(np.logical_and(filled_sta, toa_all[bi,iy_toa_start,x_toa_BEGIN:x_toa_END]>-30000) ); 
                if index_sta.sum() == 0:
                    break
                index_sta_left=index_sta[0][0]; index_sta_right=index_sta[-1][0]+1; 
                # valid_len = (index_sta_right-index_sta_left)
                toa_all[bi,iy_toa_start, np.logical_not(filled_sta)] = toa_all[bi,iy_toa_start,filled_sta].mean()

                if index_sta_left>0:
                    toa_all[bi,iy_toa_start,0:(index_sta_left)] = toa_all[bi,iy_toa_start,index_sta_left]
                if index_sta_right<x_toa_END:
                    toa_all[bi,iy_toa_start,index_sta_right:x_toa_END] = toa_all[bi,iy_toa_start,(index_sta_right-1)]
            
    if not is_end_set: ## end of all lines 
        y_toa_END_inner = y_toa_END
    
    ## *************************************************************************
    ## define return variables 
    sum_local = 0
    # tse_local = 0
    n_local = 0
    MAX_PATCHES = 400
    TEST_x = np.full([MAX_PATCHES, patch_size, patch_size, IMG_BANDS], fill_value=-9999, dtype=np.float32)   
    START_x = list()
    START_y = list()
    is_top  = list()
    is_left = list()
    ## *************************************************************************
    # start to process for each 512 * 512 block
    # iy_toa_start = y_toa_BEGIN_inner
    for iy_toa_start in range(y_toa_BEGIN_inner,y_toa_END_inner,patch_step):
        if (iy_toa_start+patch_size)>y_toa_END_inner: ## last block move back to fit patch size
            # break
            iy_toa_start = y_toa_END_inner-patch_size
        
        iy_toa_end = iy_toa_start+patch_size
        
        ## find x start and x end with at least 512 pixels non-filled both cloud and collection-1 TOA
        # fix a bug on Jul 23, 2021
        x_toa_BEGIN_inner = 0; x_toa_END_inner = 0; is_begin_set = False; is_end_set = False 
        for ix_toa_start in range(x_toa_BEGIN,x_toa_END):
            filled_sta = no_filled_toa[0, iy_toa_start:iy_toa_end, ix_toa_start]
            if filled_sta.sum()>=line_min_no_fill and not is_begin_set:
                x_toa_BEGIN_inner = ix_toa_start
                is_begin_set = True 
                break
        
        ## find x end with at least {line_min_no_fill} pixels non-filled both cloud and collection-1 
        for ix_toa_start in range(x_toa_BEGIN_inner,x_toa_END):
            filled_sta = no_filled_toa[0, iy_toa_start:iy_toa_end, ix_toa_start]
            if filled_sta.sum()<line_min_no_fill and not is_end_set:
                x_toa_END_inner = ix_toa_start
                is_end_set = True 
                break        
        if not is_end_set: ## end of all lines 
            x_toa_END_inner = x_toa_END
            
        print ("\tiy_toa_start" + str(iy_toa_start) )
        print ("\t\tx_toa_BEGIN_inner" + str(x_toa_BEGIN_inner)+"\tx_toa_END_inner" + str(x_toa_END_inner) )
        # break
        # ix_toa_start = x_toa_BEGIN_inner
        for ix_toa_start in range(x_toa_BEGIN_inner,x_toa_END_inner,patch_step):
            if (ix_toa_start+patch_size)>x_toa_END_inner: ## last block move back to fit patch size
                ix_toa_start = x_toa_END_inner-patch_size
                
            startx_toa = ix_toa_start; starty_toa = iy_toa_start
            startx_str = '{:04d}'.format(starty_toa)+'.'+'{:04d}'.format(startx_toa)                               
            
            patch_toa1 = toa_all[:,starty_toa:(starty_toa+patch_size),startx_toa:(startx_toa+patch_size)]    # in for loop that shape is (10, 512, 512)
            for bi in range(IMG_BANDS):
                if is_norm:
                    TEST_x[n_local,:,:,bi] = (patch_toa1[bi,:,:].astype(np.float32)-x_mean[bi])/x_std[bi]
                else:
                    TEST_x[n_local,:,:,bi] = patch_toa1[bi,:,:].astype(np.float32)
                           
            ## ********************************************
            ## find out the intersection            
            
            combined_no_filled = no_filled_toa[:,starty_toa:(starty_toa+patch_size),startx_toa:(startx_toa+patch_size)].reshape(patch_size,patch_size)   # shape (512, 512)
            b1_mean = patch_toa1[0,combined_no_filled].mean()
            b1_std =  patch_toa1[0,combined_no_filled].std() 
            
            ## for registration demonstration only
            print ('\t\tB1 mean '+'{:4.1f}'.format(b1_mean) + 'B1 std '+'{:4.1f}'.format(b1_std))
            
            # rasterio.open(dnn_cld_file).profile.copy()
            if combined_no_filled.sum()<patch_size**2 and line_min_no_fill==patch_size:
                print ('!!!!!combined_no_filled.sum()<PATCH_SIZE**2 ' + str(combined_no_filled.sum())+'  \t'+ startx_str)
                        
            sum_local = sum_local+patch_toa1.sum(axis=(1,2))/patch_size**2
            n_local = n_local+1      # n_local = n_test
            START_x.append(startx_toa)
            START_y.append(starty_toa)
            is_top .append(iy_toa_start==y_toa_BEGIN_inner)
            is_left.append(ix_toa_start==x_toa_BEGIN_inner)
            # break;
        # break 
    print(sum_local)
    print(n_local)
    return TEST_x, n_local, START_x, START_y, is_top, is_left
  
def get_prediction(model, TEST_x, samples_toa, lines_toa, n_test, START_x, START_y, is_top, is_left, BATCH_SIZE=24, patch_step=512, IMG_HEIGHT=512, IMG_WIDTH=512):
    classes = np.full([lines_toa, samples_toa], fill_value=10, dtype=np.uint8)
    ## 10 is filled in 123 and in label 
        
    # index_testi = np.full(n_test, fill_value=False, dtype=np.bool)
    tempx = np.full([BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, IMG_BANDS], fill_value=-9999, dtype=np.float32)    
    
    ## doing prediction
    tempi = 0
    starti = 0
    # endi = 0
    print(n_test)
    for i in range(n_test):
        tempx[tempi,:,:,:] = TEST_x[i,:,:,:]
        tempi=tempi+1
        if tempi>=(BATCH_SIZE) or i >=(n_test -1):
            print ("process patches "+str(starti+1)+"..."+str(i+1) + "\ttempi = " + str(tempi) )
            # if tempi==1:
                # tempx[tempi,:,:,:] = TEST_x[i,:,:,:]
                # print ("! number is 1 is invoked") 
                        
            logits = model.predict(tempx)
            prop = tf.nn.softmax(logits).numpy()
            classesi = np.argmax(prop,axis=3).astype(np.uint8)
            for j in range(tempi):
                jin = j+starti
                if jin>(n_test-1):
                    print ("! number is 1 is invoked") 
                    jin = n_test-1
                
                starty_at_j =  START_y[jin] if is_top [jin] else START_y[jin]+int((IMG_HEIGHT-patch_step)/2)
                startx_at_j =  START_x[jin] if is_left[jin] else START_x[jin]+int((IMG_HEIGHT-patch_step)/2)
                
                classes[starty_at_j:(START_y[jin]+IMG_HEIGHT),startx_at_j:(START_x[jin]+IMG_WIDTH)] = classesi[j,(starty_at_j-START_y[jin]):IMG_HEIGHT,(startx_at_j-START_x[jin]):IMG_WIDTH]               
                                    
            starti = i+1
            tempi = 0
            print (i)
            # break
        # print (i)
        # break
    return classes
    
# fucntion of predicting new data with trained model 
def predict_to_use (toa_all_file, toa_bqa_file, xoffset, yoffset, model, BATCH_SIZE=14, IMG_HEIGHT=512, IMG_WIDTH=512):                                                  
    toa_all = toa_all_file
    print(toa_all.shape)
        
    with rasterio.open(toa_bqa_file) as dataset:
        bqa_toa = dataset.read()    
    
    ##*********************************************
    ## split x and y 
    samples_toa = toa_all.shape[2]
    lines_toa   = toa_all.shape[1]
    patch_step = 512 - 52*2
    ## without filled pixels 
    # function 'split_toa_for_prediction' dealing with data and generate patches
    TEST_x, n_test, START_x, START_y, is_top, is_left = split_toa_for_prediction (toa_all, bqa_toa, samples_toa, lines_toa, line_min_no_fill=1, patch_step=patch_step)       
    
    no_filled_toa = (np.bitwise_and(bqa_toa,1)==0).reshape(lines_toa, samples_toa)

    ##*********************************************
    ## get and save prediction 
    classes = get_prediction(model, TEST_x, samples_toa, lines_toa, n_test, START_x, START_y, is_top, is_left, BATCH_SIZE=BATCH_SIZE, patch_step=patch_step)    # function 'get_prediction'   ,temp_band1
    classes123 = combine_thin_thick_123(classes)
    predicted = covert_123_to_label(classes123.astype(np.uint8))
    
    predicted[np.logical_not(no_filled_toa)] = 10      # this step is for filling the edge patches as "10"
    print(predicted.shape)
    return predicted     
        
#*******************************END OF PREDICTION**************************************************

    
    

