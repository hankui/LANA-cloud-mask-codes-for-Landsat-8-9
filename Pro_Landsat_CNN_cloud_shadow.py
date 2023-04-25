"""
updated by Hankui Zhang and Dong Luo (SDSU) on Mar 01, 2023 
updated by Hankui Zhang and Dong Luo (SDSU) on Nov 07, 2022 
written by Hankui Zhang and Dong Luo (SDSU) on Jul 28, 2021 

This script is used to generate Landsat-8 cloud/cloud shadow mask with 16-bit qa format from digital number (DN value). 
    bit 0: combined QA mask.           1 is use pixel, 0 is ignore pixel (if cloud ==1 or shadow == 1 or adjacent to cloud == 1 or filled ==1)
    bit 1: cloud:                      1 is yes, 0 is no 
    bit 2: adjacent to cloud/shadow:   1 is yes, 0 is no
    bit 3: cloud shadow:               1 is yes, 0 is no
    bit 4: snow/ice                    1 is yes, 0 is no
    bit 5: water                       1 is yes, 0 is no
    bit 6: cirrus                      1 is yes, 0 is no
    bit 8: filled:                     1 is yes, 0 is no

How to use it:
	This application has three arguments.
	argv1 is the dn folder name (unzip folder)
	argv2 is the quality band file name 
	argv3 is the output directory to store the cloud mask
	Command line example:
	python Pro_Landsat_CNN_cloud_shadow.py /weld/gsce_weld_1/gpfs/data2/workspace/zhangh/L7_L8_cloud/Cmsk_test/toal8/LC08_L1GT_016029_20210301_20210311_02_T2/ /weld/gsce_weld_1/gpfs/data2/workspace/zhangh/L7_L8_cloud/Cmsk_test/cloudl8/LC08_L1GT_016029_20210301_20210311_02_T2_QA_PIXEL.TIF /weld/gsce_weld_1/gpfs/data2/workspace/zhangh/L7_L8_cloud/Cmsk_test/predicted_fulll8

"""
import os
import sys
import rasterio
import apply_cnn
import tensorflow as tf 

from apply_cnn import SelfAttention, SelfAttention_reduce, SelfAttention_reduce_sumsampling

if __name__ == "__main__":
    if len(sys.argv)<4 or len(sys.argv)>5:
        this_file = 'Pro_Landsat_CNN_cloud_shadow.py'
        if '__file__' in globals():
            this_file = os.path.basename(__file__)

        usage = this_file + " DN_folder QA_file output_cloud_path model_path (optional)"
        print('\n\n! The correct usage should be ' + usage + "\n")
        exit()
    
    dn_folder_path    = sys.argv[1]    # '/gpfs/data2/workspace/zhangh/L7_L8_cloud/Cmsk_test/toa/LC08_L1TP_115034_20161104_20200905_02_T1/'
    toa_bqa_file      = sys.argv[2]    # '/gpfs/data2/workspace/zhangh/L7_L8_cloud/Cmsk_test/cloud/LC08_L1TP_115034_20161104_20200905_02_T1_QA_PIXEL.TIF'
    OUTPUT_DIR     = sys.argv[3]    # '/gpfs/data2/workspace/zhangh/L7_L8_cloud/Cmsk_test/predicted_full/'
    
    model_path = os.path.dirname(os.path.abspath(__file__))+"/v31_4.epoch100.batch64.M1.test1.model.h5"
    if len(sys.argv)==5:
        model_path    = sys.argv[4]    # '/gpfs/data2/workspace/zhangh/L7_L8_cloud/Cmsk_test/v7_2.ID029.epoch50.batch14.M0.model'
        print("warning the default trained model will been replaced by "+model_path)
    
    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # toa_all_file = toa_path
    # toa_bqa_file = bqa_path
    xoffset = 0
    yoffset = 0
    toa_all_file = apply_cnn.l8c2l1_dn2toa (dn_folder_path)
    model = tf.keras.models.load_model(model_path, compile= False, custom_objects={'SelfAttention': SelfAttention, "SelfAttention_reduce": SelfAttention_reduce, "SelfAttention_reduce_sumsampling": SelfAttention_reduce_sumsampling})
    cmask = apply_cnn.predict_to_use(toa_all_file, toa_bqa_file, xoffset, yoffset, model, BATCH_SIZE=14, IMG_HEIGHT=512, IMG_WIDTH=512)    
    qaimg = apply_cnn.img_to_qas(cmask, toa_bqa_file)
    
    base_name = os.path.basename(toa_bqa_file)[0:40]
    
    json_ojt = apply_cnn.get_json(qaimg)
    out_json = OUTPUT_DIR + '/' + base_name +'.CLD.percentage.json'
    with open(out_json, "w") as outfile:
        outfile.write(json_ojt)
           
    cnn_qa_filename = OUTPUT_DIR+'/'+base_name+".CNN.QA.tif"     
    rst = rasterio.open(toa_bqa_file)
    naip_meta = rst.profile.copy()
    naip_meta['count'] = 1
    naip_meta['width' ] = rst.shape[1]   
    naip_meta['height'] = rst.shape[0]   
    naip_meta['dtype'] = 'uint16'
    with rasterio.open(cnn_qa_filename, 'w', **naip_meta) as dst:
        dst.write(qaimg, 1)

