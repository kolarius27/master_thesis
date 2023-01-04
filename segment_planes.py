#!/bin/python
# -*- coding: iso-8859-15 -*-
import os
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
from functools import wraps, partial
from time import time
from multiprocessing import Pool

from pykdtree.kdtree import KDTree      # Install: https://github.com/storpipfugl/pykdtree

# path to pre-compiled outcrop segmentation tool (https://doi.org/10.5194/isprs-annals-III-5-105-2016)
OUTCROPPATH = r"..\Outcrop_segmentation_tool\outcrop.exe"


def main(inFiles, outFile, tempDir='temp'):
    """
    Input: Path to xyz point cloud of one epoch
    TempDir: Path to temporary directory
    Output: Path to segmented point cloud of resp. epoch
    """
    segPointsPerTile = []
    rft = partial(run_for_tile, tempDir=tempDir) # multithread prosessing
    with Pool(processes=4) as pool:
        segPointsPerTile = pool.map(rft, inFiles)
    planeCntPerTile = [np.max(tile[:, 12]) for tile in segPointsPerTile if tile is not None] #  get number of planes per tile
    pointCntPerTile = [np.shape(tile)[0] for tile in segPointsPerTile if tile is not None] # get number of poins per tile
    output_arr = np.zeros((sum(pointCntPerTile), 15)) # create array containing all points of all points per timestamp with correctec segid
    for idx, segPoints in enumerate(segPointsPerTile):
        if segPoints is None:
            continue
        segPoints[:, 12] += sum(planeCntPerTile[:idx]) # correct segid by number of segments of preious tiles
        start_idx = sum(pointCntPerTile[:idx])
        end_idx = sum(pointCntPerTile[:idx+1]) if idx < len(pointCntPerTile) else -1
        output_arr[start_idx:end_idx, :] = segPoints
    np.savetxt(outFile, output_arr[:, [0,1,2,12]], fmt=["%1.3f","%1.3f", "%1.3f", "%1.3f"], delimiter=" ") # write segmented point cloud to file


# for each tile, segments points in two iterations and merge segmented point clouds, return segmented cloud as numpy array
def run_for_tile(inFile, tempDir):
    # read input
    inPoints = pd.read_csv(inFile, sep='\t').to_numpy()
    if not os.path.exists(tempDir):
        os.makedirs(tempDir)
    segPoints1 = segment(inFile, tempDir, ['50', '99.9', '5.0', '0.1', '0.05', '0', '400', '100000000', '60.0']) # 1st iteration
    if segPoints1 is None:
        return None
    unsegPoints = get_unsegmented_points(inPoints, segPoints1, search_radius=0.001) # get unsegmented points
    iteration2File = write_unseg_points(unsegPoints, tempDir, inFile) # write unsegmented points to file
    segPoints2 = segment(iteration2File, tempDir, ['50', '99.9', '5.0', '0.2', '0.05', '0', '50', '100000000', '60.0']) # 2nd iteration
    if segPoints2 is not None:
        segPoints2[:, 12] += np.max(segPoints1[:, 12])  # update segment id for second iteration to start with max(1st iteration)
        segPoints = np.concatenate((segPoints1, segPoints2)) # merge segmented points of both iterations
    return segPoints


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('%r took: %2.4f sec' % (f.__name__, te-ts))
        return result
    return wrap

@timing
# segments point cloud based on input parameters for outcrop segmentation tool; return segmented points as numpy array
def segment(inFile, tempdir, args, delTemp=False):
    file_name_out = Path(tempdir) / Path(inFile).name
    asc_file_out = str(file_name_out) + "_" + "_".join(args) + ".asc"
    if not os.path.exists(asc_file_out):
        subprocess.call([OUTCROPPATH, str(inFile), *args[:6], str(file_name_out), *args[6:]])
    # read temp file
    try:
        segPoints = pd.read_csv(asc_file_out, sep='\t').to_numpy()
    except pd.errors.EmptyDataError:
        segPoints = None
    # delete temp file
    if delTemp:
        os.remove(asc_file_out)  # output file is deleted
    return segPoints

@timing
# returns unsegmented points by comparing original with segmented point cloud
def get_unsegmented_points(inPoints, segPoints, search_radius):
    # gets number of points
    n_target = inPoints.shape[0]
    n_source = segPoints.shape[0]
    print("%i points in target cloud read." % (n_target))
    print("%i points in source cloud read." % (n_source))

    # takes only XYZ columns of input ascii for 3D KDTree
    kd_tree = KDTree(segPoints[:, :3])

    # dist - stores the Euclidean linear distances to all neighbours
    # idx  - stores the array index of the neigbhour in  the original data array: needed to access the point data itself
    dist, idx = kd_tree.query(inPoints[:, :3], k=1, sqr_dists=False, eps=0.0) # search rad = 0.001 by default
    search_radius = 0.001

    # selects those who have a corresponding match (target -> source) in max. search radius
    #idx_target_found_match = np.where(dist <= search_radius)[0]

    # selects those who have no corresponding match (target -> source) in max. search radius (= non-segmented points)
    idx_target_found_no_match = (dist >= search_radius) # true where distance is geq
    # Number of points without a match
    n_nomatch = np.count_nonzero(idx_target_found_no_match)
    # Number of points with a match
    # n_match = n_target - np.count_nonzero(idx_target_found_no_match)
    if n_nomatch > 0:
        print("%i point(s) could not be matched with given search_radius." % (n_nomatch))

    # selects those who have no corresponding match and write them to output file
    outputcloud2 = inPoints[idx_target_found_no_match]
    return outputcloud2

@timing
# writes unsegmented point cloud to output file
def write_unseg_points(input, tempdir, inFile):
    file_name_out = str(Path(tempdir) / Path(inFile).name) + "_not_seg.xyz"
    np.savetxt(str(file_name_out), input, delimiter="\t",
            fmt=["%.3f", "%.3f", "%.3f", "%.9f", "%.9f", "%.9f", "%.9f", "%.0f", "%.0f", "%.9f", "%.9f", "%.9f"])
    return file_name_out


if __name__ == '__main__':
    inFiles = [r"J:\01_Projekte\AHK-4D\Paper\00_heidata\ASCII\input\20190803\20190803_region_1a_normals.xyz",
               r"J:\01_Projekte\AHK-4D\Paper\00_heidata\ASCII\input\20190803\20190803_region_1b_normals.xyz",
               r"J:\01_Projekte\AHK-4D\Paper\00_heidata\ASCII\input\20190803\20190803_region_2a_normals.xyz",
               r"J:\01_Projekte\AHK-4D\Paper\00_heidata\ASCII\input\20190803\20190803_region_2b_normals.xyz",
               r"J:\01_Projekte\AHK-4D\Paper\00_heidata\ASCII\input\20190803\20190803_region_3a_normals.xyz",
               r"J:\01_Projekte\AHK-4D\Paper\00_heidata\ASCII\input\20190803\20190803_region_3b_normals.xyz",
               r"J:\01_Projekte\AHK-4D\Paper\00_heidata\ASCII\input\20190803\20190803_region_4a_normals.xyz",
               r"J:\01_Projekte\AHK-4D\Paper\00_heidata\ASCII\input\20190803\20190803_region_4b_normals.xyz",
               r"J:\01_Projekte\AHK-4D\Paper\00_heidata\ASCII\input\20190803\20190803_region_5a_normals.xyz",
               r"J:\01_Projekte\AHK-4D\Paper\00_heidata\ASCII\input\20190803\20190803_region_5b_normals.xyz"]
    main(inFiles, r'J:\01_Projekte\AHK-4D\Paper\00_heidata\ASCII\output\20190803_segm_merged.xyz',
         r"J:\01_Projekte\AHK-4D\Paper\00_heidata\ASCII\temp_files")