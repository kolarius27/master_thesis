import pdal
from osgeo import ogr
from shapely import Polygon
import geopandas as gpd
import glob

def main():
    path = r'E:/NATUR_CUNI/_DP/data/LAZ/*.laz'
    
    file_list01 = get_files(path)

    create_dataframe(file_list01)



def get_files(path):
    return glob.glob(path)


def get_pipeline(file):
    x = """
    {
        "pipeline": [
            {},
            {
                "type": "filters.stats"
            },
            "info.json"
        ]
    }""".format(file)
    return x


def boundary(metadata):
    coord = metadata['metadata']['filters.stats']['bbox']['native']['boundary']['coordinates'][0]
    return Polygon(coord)


def create_dataframe(file_list):

    attributes = []
    geometries = []

    for file in file_list:
        pipeline = pdal.Pipeline(get_pipeline(file))
        execution = pipeline.execute()
        metadata = pipeline.metadata
        geom = boundary(metadata)
        geometries.append(geom)


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