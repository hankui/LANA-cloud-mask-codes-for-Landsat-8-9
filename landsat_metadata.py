""""
Landsat 8 TOA correction code
orginal: Landsat 8 collection 2 with level 1 and level 2 code
goal: adapt collection 2 to collection 1
Step 1: read metadata
"""
# Hank adapted on Mar 17, 2021 from 
# https://github.com/NASA-DEVELOP/dnppy/blob/master/dnppy/landsat/landsat_metadata.py
# import landsat_metadata

__author__ = 'jwely'
__all__ = ["landsat_metadata"]

# standard imports
from datetime import datetime
# from dnppy.solar import solar
import inspect


class landsat_metadata:
    """
    A landsat metadata object. This class builds is attributes
    from the names of each tag in the xml formatted .MTL files that
    come with landsat data. So, any tag that appears in the MTL file
    will populate as an attribute of landsat_metadata.
    You can access explore these attributes by using, for example
    .. code-block:: python
        from dnppy import landsat
        meta = landsat.landsat_metadata(my_filepath) # create object
        from pprint import pprint                    # import pprint
        pprint(vars(m))                              # pretty print output
        scene_id = meta.LANDSAT_SCENE_ID             # access specific attribute
    :param filename: the filepath to an MTL file.
    """

    def __init__(self, filename):
        """
        There are several critical attributes that keep a common
        naming convention between all landsat versions, so they are
        initialized in this class for good record keeping and reference
        """

        # custom attribute additions
        self.FILEPATH           = filename
        self.DATETIME_OBJ       = None

        # product metadata attributes
        self.LANDSAT_SCENE_ID   = None
        self.DATA_TYPE          = None
        self.ELEVATION_SOURCE   = None
        self.OUTPUT_FORMAT      = None
        self.SPACECRAFT_ID      = None
        self.SENSOR_ID          = None
        self.WRS_PATH           = None
        self.WRS_ROW            = None
        self.NADIR_OFFNADIR     = None
        self.TARGET_WRS_PATH    = None
        self.TARGET_WRS_ROW     = None
        self.DATE_ACQUIRED      = None
        self.SCENE_CENTER_TIME  = None

        # image attributes
        self.CLOUD_COVER        = None
        self.IMAGE_QUALITY_OLI  = None
        self.IMAGE_QUALITY_TIRS = None
        self.ROLL_ANGLE         = None
        self.SUN_AZIMUTH        = None
        self.SUN_ELEVATION      = None
        self.EARTH_SUN_DISTANCE = None    # calculated for Landsats before 8.
        
        ## projection parameters added by Hankui on Mar 17 2021
        self.MAP_PROJECTION     = None
        self.DATUM              = None   
        self.ELLIPSOID          = None
        self.UTM_ZONE           = None
        self.ZONE_NUMBER        = None
        self.REFLECTIVE_LINES   = None
        self.REFLECTIVE_SAMPLES = None 
        self.PRODUCT_SAMPLES_REF= None
        self.PRODUCT_LINES_REF  = None 
        self.ACQUISITION_DATE   = None
        self.CORNER_UL_PROJECTION_X_PRODUCT = None 
        self.CORNER_UL_PROJECTION_Y_PRODUCT = None
        self.CORNER_UL_LAT_PRODUCT  = None 
        self.CORNER_UL_LON_PRODUCT  = None
        self.CORNER_LR_LAT_PRODUCT  = None
        self.CORNER_LR_LON_PRODUCT  = None

        ## projection parameters added by Hankui on Mar 21 2021 for TOA calculation 
        self.REFLECTANCE_MULT_BAND_1= None
        self.REFLECTANCE_MULT_BAND_2= None   
        self.REFLECTANCE_MULT_BAND_3= None
        self.REFLECTANCE_MULT_BAND_4= None
        self.REFLECTANCE_MULT_BAND_5= None
        self.REFLECTANCE_MULT_BAND_6= None
        self.REFLECTANCE_MULT_BAND_7= None 
        self.REFLECTANCE_MULT_BAND_8= None
        self.REFLECTANCE_MULT_BAND_9= None 
        self.RADIANCE_MULT_BAND_10  = None
        self.RADIANCE_MULT_BAND_11  = None 
        self.REFLECTANCE_ADD_BAND_1 = None
        self.REFLECTANCE_ADD_BAND_2 = None   
        self.REFLECTANCE_ADD_BAND_3 = None
        self.REFLECTANCE_ADD_BAND_4 = None
        self.REFLECTANCE_ADD_BAND_5 = None
        self.REFLECTANCE_ADD_BAND_6 = None
        self.REFLECTANCE_ADD_BAND_7 = None 
        self.REFLECTANCE_ADD_BAND_8 = None
        self.REFLECTANCE_ADD_BAND_9 = None 
        self.RADIANCE_ADD_BAND_10   = None
        self.RADIANCE_ADD_BAND_11   = None
        self.K1_CONSTANT_BAND_10    = None
        self.K2_CONSTANT_BAND_10    = None 
        self.K1_CONSTANT_BAND_11    = None
        self.K2_CONSTANT_BAND_11    = None 
        
        self.TEMPERATURE_MULT_BAND_ST_B10 = None 
        self.TEMPERATURE_ADD_BAND_ST_B10  = None 
        
        # read the file and populate the MTL attributes
        self._read(filename)

    def _read(self, filename):
        """ reads the contents of an MTL file """
        
        # if the "filename" input is actually already a metadata class object, return it back.
        # https://www.cnblogs.com/yaohong/p/8874154.html
        if inspect.isclass(filename):
            return filename

        fields = []
        values = []

        metafile = open(filename, 'r')
        metadata = metafile.readlines()

        for line in metadata:
            # skips lines that contain "bad flags" denoting useless data AND lines
            # greater than 1000 characters. 1000 character limit works around an odd LC5
            # issue where the metadata has 40,000+ characters of whitespace
            bad_flags = ["END", "GROUP"]
            if not any(x in line for x in bad_flags) and len(line) <= 1000:
                try:
                    line = line.replace("  ", "")
                    line = line.replace("\n", "")      # "\n" seperate as a new line
                    field_name, field_value = line.split(' = ')
                    fields.append(field_name)
                    values.append(field_value)
                except:
                    pass
        # https://www.runoob.com/python/python-func-hasattr.html
        # https://www.runoob.com/python/python-func-setattr.html
        for i in range(len(fields)):
            if hasattr(self, fields[i]) and getattr(self, fields[i])!=None: # Hank on Mar 21 2021 to include surface reflectance MTL 
                continue 
            
            # format fields without quotes,dates, or times in them as floats
            if not any(['"' in values[i], 'DATE' in fields[i], 'TIME' in fields[i]]):
                setattr(self, fields[i], float(values[i]))
            else:
                values[i] = values[i].replace('"', '')
                setattr(self, fields[i], values[i])

        # create datetime_obj attribute (drop decimal seconds)
        if self.DATE_ACQUIRED!=None:
            dto_string          = self.DATE_ACQUIRED    + self.SCENE_CENTER_TIME
            self.DATETIME_OBJ   = datetime.strptime(dto_string.split(".")[0], "%Y-%m-%d%H:%M:%S")
        else: 
            dto_string          = self.ACQUISITION_DATE
            self.DATETIME_OBJ   = datetime.strptime(dto_string.split(".")[0], "%Y-%m-%d")
        
        # only landsat 8 includes sun-earth-distance in MTL file, so calculate it
        # for the Landsats 4,5,7 using solar module.
        # if not self.SPACECRAFT_ID == "LANDSAT_8":

            # use 0s for lat and lon, sun_earth_distance is not a function of any one location on earth.
            # s = solar(0, 0, self.DATETIME_OBJ, 0)
            # self.EARTH_SUN_DISTANCE = s.get_rad_vector()

        # print("Scene {0} center time is {1}".format(self.LANDSAT_SCENE_ID, self.DATETIME_OBJ))