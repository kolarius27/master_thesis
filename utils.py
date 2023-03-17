import pdal
from osgeo import ogr
from shapely import Polygon, Point, LineString
import geopandas as gpd
import pandas as pd
import glob
import os
import laspy as lp
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from PIL import Image
import rasterio as rio
import scipy
import math as m


def directional_vector_angles(yaw, pitch):
    x = m.cos(yaw)*m.cos(pitch)
    y = m.sin(yaw)*m.cos(pitch)
    z = m.sin(pitch)
    return unit_vector(np.array(x, y, z))


def directional_vector_points(pointA, pointB):
    return unit_vector(np.array(pointB-pointA))


def unit_vector(vector):
    return vector / np.linalg.norm(vector)


def angle_between(pointA, pointB, yaw, pitch):
    v1 = directional_vector_angles(yaw, pitch)
    v2 = directional_vector_points(pointA, pointB)
    return np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))


def get_files(path):
    return glob.glob(path)


def create_shapefile(shp_path, filenames, geometries):
    d = {'filename': filenames, 'geometry': geometries}
    gdf = gpd.GeoDataFrame(d, crs="EPSG:32632")
    gdf.to_file(shp_path)


def path2str(path, end):
    if end != None:
        return str(path).replace('\\', '/')[:-4] + '{}'.format(end)
    else:
        return str(path).replace('\\', '/')
    

def get_geometry_from_shp(path):
    data = gpd.read_file(path)
    return data['geometry'][0]

