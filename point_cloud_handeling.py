import pdal
from osgeo import ogr
from shapely import Polygon, Point, LineString
import geopandas as gpd
import pandas as pd
import glob
import os


def main():
    path = r"E:/NATUR_CUNI/_DP/data/Trajectory/*.txt"
    
    trajectory(path)


def trajectory(path):
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


def get_pipeline(file):
    file = '"' + str(file).replace('\\', '/') + '"'
    x = """
    {{
        "pipeline": [
            {},
            {{
                "type": "filters.stats"
            }}
        ]
    }}""".format(file)
    print(x)
    return x


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