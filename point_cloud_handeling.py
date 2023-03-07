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


def main():
    # path_raster = r"E:/NATUR_CUNI/_DP/data/LAZ/raster/ahk_uls_pdalmerge_aoi_first_raster.tif"
    # path_raster2 = r"E:/NATUR_CUNI/_DP/data/LAZ/raster/ahk_uls_pdalmerge_aoi_raster.tif"
    path_las = r'E:/NATUR_CUNI/_DP/data/LAZ/ahk_uls_pdalmerge_aoi.laz'
    profil_shp = r'E:/NATUR_CUNI/_DP/data/profil.shp'
    boulder1 = r'E:/NATUR_CUNI/_DP/data/LAZ/boulder/first/boulder1.las'
    
    output_path = r'E:/NATUR_CUNI/_DP/data/LAZ'

    # raster_to_histo(path_raster, output_path)
    # raster_to_histo(path_raster2, output_path)
    # pc_clip(path_las, profil_shp)
    # pc_raster(path_las)
    # las_to_histo(path_las, output_path)
    thinning(boulder1, _, _)


def scan_angle():
    pass


def thinning(path, output, method):
    orig_las = lp.read(path, )
    print(list(orig_las.point_format.dimension_names))
    orig_las.gps_time.sort()
    print(orig_las.gps_time)
    


def las_to_histo(path, output_path):
    las = lp.read(path)
    dim_list = list(las.point_format.dimension_names)
    split_path = os.path.split(path)
    print(dim_list[:4])
    for name in dim_list[:4]:
        if name in ['X', 'Y', 'Z']:
            dim = np.array(las[name])/100
        else:
            dim = np.array(las[name])
        output = os.path.join(output_path, 'histo', split_path[1][:-4], name + '.jpg')
        histogram(dim, name, output)



def raster_to_histo(path, output_path):
    raster = rio.open(path, mode="r+")
    print(type(raster.descriptions))
    # masked_band = np.ma.masked_array(raster.read(1), raster.nodata)
    bands = raster.read(masked=True)
    mask = bands[0].mask
    split_path = os.path.split(path)
    for band, desc in zip(bands, raster.descriptions):
        band.mask = mask
        list_band = []
        for row in band:
            for value in row:
                if type(value) != np.ma.core.MaskedConstant:
                    list_band.append(value)
        output = os.path.join(output_path, 'histo', split_path[1][:-4], desc + '.jpg')
        histogram(np.array(list_band), desc, output)


        
def histogram_old(band):
    summary_stats(np.array(band))


    n, bins, patches = plt.hist(x=band, bins='auto',
                            alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('My Very Own Histogram')
    plt.text(1, 1, r'$\mu=15, b=3$', transform=ax.transAxes)
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 1000) * 1000 if maxfreq % 1000 else maxfreq + 1000)
    plt.show()


def histogram(band, desc, output):
    stats = summary_stats(np.array(band))
    fig, ax = plt.subplots()
    n, bins, patches = ax.hist(x=band, bins = 20)
    plt.title(desc)
    i=0.95
    for stat in stats.iteritems():
        text = '{}: {}'.format(stat[0], round(stat[1],3))
        ax.text(1.05,i, text, transform=ax.transAxes)
        i-=0.05
    plt.savefig(output,  bbox_inches='tight')
    plt.show()

def summary_stats(band):
    band_series = pd.Series(band)
    stats1 = band_series.describe()
    stats2 = scipy.stats.describe(band)
    stats1['median'] = np.median(band)
    stats1['mode'] = scipy.stats.mode(band).mode[0]
    stats1['mad'] = np.median(abs(band-np.median(band)))
    stats1['variance'] = stats2.variance
    stats1['range'] = np.ptp(band)
    stats1['rms'] = np.sqrt(np.mean(band**2))
    stats1['skewness'] = stats2.skewness
    stats1['kurtosis'] = stats2.kurtosis
    return stats1


def pc_raster(path):
    file = path2str(path, None)
    output = path2str(path, '_first_raster_01.tif')

    x="""
    {{
        "pipeline": [
            {{
                "filename": "{}",
                "spatialreference": "EPSG:32632"
            }},
            {{
                "filename": "{}",
                "resolution": 0.25,
                "where": "ReturnNumber == 1"
            }}
        ]
    }}""".format(file, output)

    print(x)

    pipeline = pdal.Pipeline(x)
    execute = pipeline.execute()
    print('success')


def point_density(path, mode, radius):
    file = path2str(path, None)
    output_pc = path2str(path, '_pd.las')
    output_raster = path2str(path, '_pd.tif')

    x="""
    {{
        "pipeline": [
            {{
                "filename": "{}",
                "spatialreference": "EPSG:32632"
            }},
            {{
                "type": "filters.radialdensity",
                "radius": "{}"
            }},
            {{
                "type": "writers.las",
                "filename": {output_pc},
                "output_dims":
            }}
        ]
    }}""".format(file, output_pc)


def pc_clip(path, shapefile):
    print('start')
    input = path2str(path, None)
    output = path2str(path, '_profil.laz')
    poly = path2str(get_geometry_from_shp(shapefile), None)
    pdal_type = """"filters.crop",
                "polygon": "{}" """.format(poly)
    print('inputs prepared')

    pipeline = get_pipeline(input, pdal_type, output)
    execute = pipeline.execute()
    print("success")
    


def pc_merge(path):
    output = os.path.join(os.path.split(path)[0], "ahk_uls_pdalmerge.laz")
    print(output)
    pipeline = get_pipeline(path, '"filters.merge"', output)
    execute = pipeline.execute()
    print("success")


def pc_visualize(path):
    point_cloud = lp.read(path)
    print("point cloud read")
    points = np.vstack((point_cloud.x, point_cloud.y, point_cloud.z)).transpose()
    colors = np.vstack((point_cloud.red, point_cloud.green, point_cloud.blue)).transpose()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors/65535)
    print("let's visualize")

    o3d.visualization.draw_geometries([pcd])


def pc_trajectory(path):
    file_list = get_files(path)

    d_lines = {'start': [], 'end': [], 'geometry': []}

    for file in file_list:
        split_path = os.path.split(file)
        shp_point_path = os.path.join(split_path[0], 'points', split_path[1][:-4] + '_p.shp')
        print(shp_point_path)
        
        df = pd.read_csv(file)
        df['geometry'] = df.apply(lambda row: Point(row[['Easting[m]', 'Northing[m]', 'Height[m]']]), axis=1)
        geodf_points = gpd.GeoDataFrame(df, crs="EPSG:32632")
        geodf_points.to_file(shp_point_path)

        start = df['Time[s]'].iat[0]
        end = df['Time[s]'].iat[-1]
        line = LineString(geodf_points['geometry'])
        d_lines['start'] += [start]
        d_lines['end'] += [end]
        d_lines['geometry'] += [line]

    shp_line_path = os.path.join(split_path[0], 'lines', 'trajectory.shp')

    geodf_lines = gpd.GeoDataFrame(d_lines, crs="EPSG:32632")
    geodf_lines.to_file(shp_line_path)


def create_boundary_shp(path):
    print(os.path.split(path)[0])
    shp_path = os.path.join(os.path.split(path)[0], 'boundary/boundary.shp')
    print(shp_path)
    
    file_list = get_files(path)

    filenames, geometries = create_lists(file_list)

    create_shapefile(shp_path, filenames, geometries)



def get_files(path):
    return glob.glob(path)


def get_pipeline(file, type, output):
    file = path2str(file, None)
    output = path2str(output, None)
    x = """
    {{
        "pipeline": [
            "{}",
            {{
                "type": {}
            }},
            "{}"
        ]
    }}""".format(file, str(type), output)
    print(x)
    return pdal.Pipeline(x)


def boundary(metadata):
    coord = metadata['metadata']['filters.stats']['bbox']['native']['boundary']['coordinates'][0]
    return Polygon(coord)


def create_lists(file_list):

    filenames = []
    geometries = []

    for file in file_list:
        filename = os.path.split(file)[-1]
        filenames.append(filename)
        pipeline = pdal.Pipeline(get_pipeline(file))
        execution = pipeline.execute()
        metadata = pipeline.metadata
        geom = boundary(metadata)
        geometries.append(geom)

    print(filenames)
    print(geometries)

    return filenames, geometries


def create_shapefile(shp_path, filenames, geometries):
    d = {'filename': filenames, 'geometry': geometries}
    gdf = gpd.GeoDataFrame(d, crs="EPSG:32632")
    gdf.to_file(shp_path)


def get_geometry_from_shp(path):
    data = gpd.read_file(path)
    return data['geometry'][0]


def path2str(path, end):
    if end != None:
        return str(path).replace('\\', '/')[:-4] + '{}'.format(end)
    else:
        return str(path).replace('\\', '/')


def test():
    print('start')
    # pipeline = pdal.Reader(filename=r'E:/NATUR_CUNI/_DP/data/LAZ/tile_652783_5188541.laz') | pdal.Filters.info()
    pipeline = pdal.Pipeline(x)
    print('success')
    count = pipeline.execute()
    print('megasuccess')
    arrays = pipeline.arrays
    metadata = pipeline.metadata
    log = pipeline.log
    print(metadata['metadata']['filters.stats']['bbox']['native']['boundary'])


if __name__ == '__main__':
    main()