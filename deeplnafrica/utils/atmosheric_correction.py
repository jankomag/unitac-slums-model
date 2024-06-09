# Adapted original code by Sam Murphy: https://github.com/samsammurphy/gee-atmcorr-S2
import os
import ee
import datetime
import math
import multiprocessing
from tqdm import tqdm
from Py6S import SixS, AtmosProfile, Geometry, AeroProfile, PredefinedWavelengths, Wavelength

class Atmospheric():

  def round_date(date,xhour):
    """
    rounds a date of to the closest 'x' hours
    """
    y = date.get('year')
    m = date.get('month')
    d = date.get('day')
    H = date.get('hour')
    HH = H.divide(xhour).round().multiply(xhour)
    return date.fromYMD(y,m,d).advance(HH,'hour')
  
  def round_month(date):
    """
    round date to closest month
    """
    # start of THIS month
    m1 = date.fromYMD(date.get('year'),date.get('month'),ee.Number(1))
    
    # start of NEXT month
    m2 = m1.advance(1,'month')
      
    # difference from date
    d1 = ee.Number(date.difference(m1,'day')).abs()
    d2 = ee.Number(date.difference(m2,'day')).abs()
    
    # return closest start of month
    return ee.Date(ee.Algorithms.If(d2.gt(d1),m1,m2))
  
  
  
  def water(geom,date):
    """
    Water vapour column above target at time of image aquisition.
    
    (Kalnay et al., 1996, The NCEP/NCAR 40-Year Reanalysis Project. Bull. 
    Amer. Meteor. Soc., 77, 437-471)
    """
    
    # Point geometry required
    centroid = geom.centroid()
    
    # H2O datetime is in 6 hour intervals
    H2O_date = Atmospheric.round_date(date,6)
    
    # filtered water collection
    water_ic = ee.ImageCollection('NCEP_RE/surface_wv').filterDate(H2O_date, H2O_date.advance(1,'month'))
    
    # water image
    water_img = ee.Image(water_ic.first())
    
    # water_vapour at target
    water = water_img.reduceRegion(reducer=ee.Reducer.mean(), geometry=centroid).get('pr_wtr')
                                        
    # convert to Py6S units (Google = kg/m^2, Py6S = g/cm^2)
    water_Py6S_units = ee.Number(water).divide(10)                                   
    
    return water_Py6S_units
  
  
  
  def ozone(geom,date):
    """
    returns ozone measurement from merged TOMS/OMI dataset
    
    OR
    
    uses our fill value (which is mean value for that latlon and day-of-year)
  
    """
    
    # Point geometry required
    centroid = geom.centroid()
       
    def ozone_measurement(centroid,O3_date):
        
        # filtered ozone collection
        ozone_ic = ee.ImageCollection('TOMS/MERGED').filterDate(O3_date, O3_date.advance(1,'month'))
        
        # ozone image
        ozone_img = ee.Image(ozone_ic.first())
        
        # ozone value IF TOMS/OMI image exists ELSE use fill value
        ozone = ee.Algorithms.If(ozone_img,\
        ozone_img.reduceRegion(reducer=ee.Reducer.mean(), geometry=centroid).get('ozone'),\
        ozone_fill(centroid,O3_date))
        
        return ozone
      
    def ozone_fill(centroid,O3_date):
        """
        Gets our ozone fill value (i.e. mean value for that doy and latlon)
        
        you can see it
        1) compared to LEDAPS: https://code.earthengine.google.com/8e62a5a66e4920e701813e43c0ecb83e
        2) as a video: https://www.youtube.com/watch?v=rgqwvMRVguI&feature=youtu.be
        
        """
        
        # ozone fills (i.e. one band per doy)
        ozone_fills = ee.ImageCollection('users/samsammurphy/public/ozone_fill').toList(366)
        
        # day of year index
        jan01 = ee.Date.fromYMD(O3_date.get('year'),1,1)
        doy_index = date.difference(jan01,'day').toInt()# (NB. index is one less than doy, so no need to +1)
        
        # day of year image
        fill_image = ee.Image(ozone_fills.get(doy_index))
        
        # return scalar fill value
        return fill_image.reduceRegion(reducer=ee.Reducer.mean(), geometry=centroid).get('ozone')
     
    # O3 datetime in 24 hour intervals
    O3_date = Atmospheric.round_date(date,24)
    
    # TOMS temporal gap
    TOMS_gap = ee.DateRange('1994-11-01','1996-08-01')  
    
    # avoid TOMS gap entirely
    ozone = ee.Algorithms.If(TOMS_gap.contains(O3_date),ozone_fill(centroid,O3_date),ozone_measurement(centroid,O3_date))
    
    # fix other data gaps (e.g. spatial, missing images, etc..)
    ozone = ee.Algorithms.If(ozone,ozone,ozone_fill(centroid,O3_date))
    
    #convert to Py6S units 
    ozone_Py6S_units = ee.Number(ozone).divide(1000)# (i.e. Dobson units are milli-atm-cm )                             
    
    return ozone_Py6S_units
 

  def aerosol(geom,date):
    """
    Aerosol Optical Thickness.
    
    try:
      MODIS Aerosol Product (monthly)
    except:
      fill value
    """
    
    def aerosol_fill(date):
        """
        MODIS AOT fill value for this month (i.e. no data gaps)
        """
        return ee.Image('users/samsammurphy/public/AOT_stack')\
                .select([ee.String('AOT_').cat(date.format('M'))])\
                .rename(['AOT_550'])
               
               
    def aerosol_this_month(date):
        """
        MODIS AOT original data product for this month (i.e. some data gaps)
        """
        # image for this month
        img =  ee.Image(\
                        ee.ImageCollection('MODIS/006/MOD08_M3')\
                            .filterDate(Atmospheric.round_month(date))\
                            .first()\
                        )
        
        # fill missing month (?)
        img = ee.Algorithms.If(img,\
                                # all good
                                img\
                                .select(['Aerosol_Optical_Depth_Land_Mean_Mean_550'])\
                                .divide(1000)\
                                .rename(['AOT_550']),\
                                # missing month
                                    aerosol_fill(date))
                        
        return img    
        
  
    def get_AOT(AOT_band,geom):
        """
        AOT scalar value for target
        """  
        return ee.Image(AOT_band).reduceRegion(reducer=ee.Reducer.mean(),\
                                    geometry=geom.centroid())\
                                    .get('AOT_550')
                                

    after_modis_start = date.difference(ee.Date('2000-03-01'),'month').gt(0)
    
    AOT_band = ee.Algorithms.If(after_modis_start, aerosol_this_month(date), aerosol_fill(date))
    
    AOT = get_AOT(AOT_band,geom)
    
    AOT = ee.Algorithms.If(AOT,AOT,get_AOT(aerosol_fill(date),geom))
    # i.e. check reduce region worked (else force fill value)
    
    return AOT


def apply_atmospheric_correction(img_collection, geometry):
    img_collection_list = img_collection.toList(img_collection.size())
    num_imgs = img_collection_list.size().getInfo()

    with multiprocessing.Pool(processes=2*os.cpu_count()) as pool:
        parallel_args_list = [(img_collection_list, i, geometry) for i in range(num_imgs)]
        images = []
        with tqdm(total=num_imgs) as progress_bar:
            for corr_image in pool.imap(_atmospheric_correction_parallel, parallel_args_list):
                images.append(corr_image)
                progress_bar.update()
        
        return ee.ImageCollection.fromImages(ee.List(images))


def _atmospheric_correction_parallel(args):
    img_collection_list, index, geometry = args

    img = ee.Image(img_collection_list.get(index))
    toa = img.divide(10000)
    info = img.getInfo()['properties']
    date = datetime.datetime.utcfromtimestamp(info['system:time_start']/1000)# i.e. Python uses seconds, EE uses milliseconds
    solar_z = info['MEAN_SOLAR_ZENITH_ANGLE']

    ee_date = ee.Date(date)
    h2o = Atmospheric.water(geometry,ee_date).getInfo()
    o3 = Atmospheric.ozone(geometry,ee_date).getInfo()
    aot = Atmospheric.aerosol(geometry,ee_date).getInfo()

    SRTM = ee.Image('CGIAR/SRTM90_V4')# Shuttle Radar Topography mission covers *most* of the Earth
    try:
        alt = SRTM.reduceRegion(reducer = ee.Reducer.mean(),geometry = geometry.centroid()).get('elevation').getInfo()
        km = alt/1000 # i.e. Py6S uses units of kilometers
    except:
        return img

    # Instantiate
    s = SixS()

    # Atmospheric constituents
    s.atmos_profile = AtmosProfile.UserWaterAndOzone(h2o,o3)
    s.aero_profile = AeroProfile.Continental
    s.aot550 = aot

    # Earth-Sun-satellite geometry
    s.geometry = Geometry.User()
    s.geometry.view_z = 0               # always NADIR (I think..)
    s.geometry.solar_z = solar_z        # solar zenith angle
    s.geometry.month = date.month # month and day used for Earth-Sun distance
    s.geometry.day = date.day     # month and day used for Earth-Sun distance
    s.altitudes.set_sensor_satellite_level()
    s.altitudes.set_target_custom_altitude(km)

    def spectralResponseFunction(bandname):
        """
        Extract spectral response function for given band name
        """
        bandSelect = {
            'B1':PredefinedWavelengths.S2A_MSI_01,
            'B2':PredefinedWavelengths.S2A_MSI_02,
            'B3':PredefinedWavelengths.S2A_MSI_03,
            'B4':PredefinedWavelengths.S2A_MSI_04,
            'B5':PredefinedWavelengths.S2A_MSI_05,
            'B6':PredefinedWavelengths.S2A_MSI_06,
            'B7':PredefinedWavelengths.S2A_MSI_07,
            'B8':PredefinedWavelengths.S2A_MSI_08,
            'B8A':PredefinedWavelengths.S2A_MSI_8A,
            'B9':PredefinedWavelengths.S2A_MSI_09,
            'B10':PredefinedWavelengths.S2A_MSI_10,
            'B11':PredefinedWavelengths.S2A_MSI_11,
            'B12':PredefinedWavelengths.S2A_MSI_12,
            }
        return Wavelength(bandSelect[bandname])

    def toa_to_rad(bandname):
        """
        Converts top of atmosphere reflectance to at-sensor radiance
        """
        
        # solar exoatmospheric spectral irradiance
        ESUN = info['SOLAR_IRRADIANCE_'+bandname]
        solar_angle_correction = math.cos(math.radians(solar_z))
        
        # Earth-Sun distance (from day of year)
        doy = date.timetuple().tm_yday
        d = 1 - 0.01672 * math.cos(0.9856 * (doy-4))# http://physics.stackexchange.com/questions/177949/earth-sun-distance-on-a-given-day-of-the-year
    
        # conversion factor
        multiplier = ESUN*solar_angle_correction/(math.pi*d**2)

        # at-sensor radiance
        rad = toa.select(bandname).multiply(multiplier)
        
        return rad

    def surface_reflectance(bandname):
        """
        Calculate surface reflectance from at-sensor radiance given waveband name
        """
        
        # run 6S for this waveband
        s.wavelength = spectralResponseFunction(bandname)
        s.run()
        
        # extract 6S outputs
        Edir = s.outputs.direct_solar_irradiance             #direct solar irradiance
        Edif = s.outputs.diffuse_solar_irradiance            #diffuse solar irradiance
        Lp   = s.outputs.atmospheric_intrinsic_radiance      #path radiance
        absorb  = s.outputs.trans['global_gas'].upward       #absorption transmissivity
        scatter = s.outputs.trans['total_scattering'].upward #scattering transmissivity
        tau2 = absorb*scatter                                #total transmissivity
        
        # radiance to surface reflectance
        rad = toa_to_rad(bandname)
        ref = rad.subtract(Lp).multiply(math.pi).divide(tau2*(Edir+Edif))
        
        return ref

    b = surface_reflectance('B2')
    g = surface_reflectance('B3')
    r = surface_reflectance('B4')
    nir = surface_reflectance('B8')
    corr_image = b.addBands(g).addBands(r).addBands(nir)
    corr_image = corr_image.multiply(10000).toFloat()

    return corr_image