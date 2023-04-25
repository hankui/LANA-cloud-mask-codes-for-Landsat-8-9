***************************************************************************************************************
Software requirement:
	python 3.7+
	Numpy >= 1.19.5
	Rasterio >= 1.2.6
	Tensorflow >= 2.6.0

***************************************************************************************************************
Included files:
	* Pro_Landsat_CNN_cloud_shadow.py
		-The main python function
	* apply_cnn.py
		-python functions needed and invoked by the main function
	* mean.std.no.fill.csv
		-csv file storing the mean and standard deviation values for each band used for normalization
	* v31_4.epoch100.batch64.M1.test1.model.h5
		-trained CNN model with h5 format
		-note this file could be in https://zenodo.org/record/7786456#.ZEf693aZNaQ due to the file size limitation in github

**************************************************************************************************************
Perparation:
	Download Landsat 8 Collection 2 level-1 image
	unzip the folder.
	The model needs following files (all of them should be in one folder):
		Aerosols  band image (*B1.tif) 
		Blue      band image (*B2.tif) 
		Green     band image (*B3.tif) 
		Red       band image (*B4.tif) 
		NIR       band image (*B5.tif) 
		SWIR1     band image (*B6.tif) 
		SWIR2     band image (*B7.tif) 
		Cirrus    band image (*B9.tif) 
		*_QA_PIXEL.TIF  quality assessment layer
		*_SZA.TIF       solar zenith angle layer
		*_MTL.txt       metadta file	
***************************************************************************************************************
INPUT:
	dn folder (need to unzip)
	quality band (file ended with *QA_PIXEL)
OUTPUT:
	Cloud mask tif file with QA format (16-bit):
    bit 0: combined QA mask.          1 is use pixel, 0 is ignore pixel (if cloud ==1 or shadow == 1 or adjacent to cloud == 1 or cirrus = 1 or filled ==1)
    bit 1: cloud:                     1 is yes, 0 is no 
    bit 2: adjacent to cloud/shadow:  1 is yes, 0 is no
    bit 3: cloud shadow:              1 is yes, 0 is no
    bit 4: snow/ice:                  1 is yes, 0 is no
    bit 5: water:                     1 is yes, 0 is no
    bit 6: cirrus:                    1 is yes, 0 is no
    bit 8: filled:                    1 is yes, 0 is no

*****************************************************************************************************************
How to use it:
	This application has three arguments.
	argv1 is the image folder name (filled value set as -36767)
	argv2 is the quality band file name 
	argv3 is the output directory to store the cloud mask
	Command line example:
	python Pro_Landsat_CNN_cloud_shadow.py /weld/gsce_weld_1/gpfs/data2/workspace/zhangh/L7_L8_cloud/Cmsk_test/dnl8/LC08_L1GT_016029_20210301_20210311_02_T2/ /weld/gsce_weld_1/gpfs/data2/workspace/zhangh/L7_L8_cloud/Cmsk_test/dnl8/LC08_L1GT_016029_20210301_20210311_02_T2/LC08_L1GT_016029_20210301_20210311_02_T2_QA_PIXEL.TIF /weld/gsce_weld_1/gpfs/data2/workspace/zhangh/L7_L8_cloud/Cmsk_test/predicted_fulll8

*****************************************************************************************************************
The paper below provides the algorithm and evaluation:
Hankui Zhang, Dong Luo, David Roy, A learning attention network algorithm (LANA) for accurate Landsat-8 cloud and shadow masking, Remote Sensing of Environment, in preparation 

The training data (512*512 30m pixel patches) for the current model is publicly available at: https://zenodo.org/record/7786456#.ZEf693aZNaQ

*****************************************************************************************************************
Acknowledgements 
The US Government’s rights to these data are detailed in FAR 52.227-14 and IA 52.204-713b.
The USGS is thanked to provide the USGS Landsat 8 Cloud Cover Assessment Validation Data: https://landsat.usgs.gov/landsat-8-cloud-cover-assessment-validation-data 
The SPARCS data source: https://www.usgs.gov/landsat-missions/spatial-procedures-automated-removal-cloud-and-shadow-sparcs-validation-data 
