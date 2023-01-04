#!/bin/python
# -*- coding: iso-8859-15 -*-

from math import *
from time import time
from pathlib import Path
import numpy as np
from tqdm import tqdm
from functools import wraps, partial
from sklearn.neighbors import KDTree
import pickle
from numpy import linalg
import pandas as pd



def main(pc_t1, pc_t2, clf, outdir, t1, t2, reg_error):
    """
    Input: Path to xyz point clouds of two epochs, path to pre-trained classifier (clf), registration error between epochs, timestamps t1 and t2
    Output: Path to input point cloud of 1st epoch extended by M3C2 distances between corresponding planes and uncertainty
    """
    ## counts number of segments for each epoch
    print("Count number of segments per timestamp...")
    # get number of planes for each timestamp which will be used as parameters for following function calls
    cnt_segments_t1 = int(get_number_of_planes(pc_t1))
    cnt_segments_t2 = int(get_number_of_planes(pc_t2))

    cnt_pts_t1 = get_number_of_pts(pc_t1)
    cnt_pts_t2 = get_number_of_pts(pc_t2)

    ## derives segment metrics for epoch 1
    print("Create segment metrics list for epoch 1...")
    # creates 2D array with segment metrics of for each segment in t1
    t1_segment_metrics_array = np.zeros((cnt_segments_t1, 13), dtype=float)

    # defines 2D array of fix length (total number of points in dim 1 and number of values stored for each point in dim 2)
    t1_segment_pts_array = np.zeros((cnt_pts_t1, 4), dtype=float)

    # creates segmemnt metrics and plane points for epoch 2
    t1_segment_metrics_array_final, t1_plane_pts_array_final, sorted_out_t1 = create_segment_metrics_array \
            (t1_segment_metrics_array, t1_segment_pts_array, pc_t1, cnt_pts_t1)

    ## derives segment metrics for epoch 2
    print("Create segment metrics list for epoch 2...")
    # creates 2D array with segment metrics of for each segment in t2
    t2_segment_metrics_array = np.zeros((cnt_segments_t2, 13), dtype=float)

    # defines 2D array of fix length (total number of points in dim 1 and number of values stored for each point in dim 2)
    t2_segment_pts_array = np.zeros((cnt_pts_t2, 4), dtype=float)

    # creates segmemnt metrics and plane points for epoch 2
    t2_segment_metrics_array_final, t2_plane_pts_array_final, sorted_out_t2 = create_segment_metrics_array \
            (t2_segment_metrics_array, t2_segment_pts_array, pc_t2, cnt_pts_t2)

    ## creates 2D list with CoGs of all segments of epoch 1
    print("Creating list of CoGs for segments of epoch 1 and 2...")
    CoG_t1 = create_CoG_array(t1_segment_metrics_array_final, cnt_segments_t1)

    ## creates 2D list with CoGs of all segments of epoch 2
    CoG_t2 = create_CoG_array(t2_segment_metrics_array_final, cnt_segments_t2)

    ## builds kd search tree
    t2_kd_tree_apply_rf = KDTree(CoG_t2[:,1:], leaf_size=2)

    # loads random forest classifier
    clf = pickle.load(open(clf, 'rb'))

    # starts classification and returns list of corresponding plane pairs in two epochs
    print("Starting correspondence search...")
    corr_planes_list = apply_rf(cnt_segments_t1, CoG_t1, CoG_t2, t2_kd_tree_apply_rf,
                                t1_segment_metrics_array_final, t2_segment_metrics_array_final, clf)

    ## applies M3C2 distance calculation between corresponding planes
    print("Calculating M3C2 distances between corresponding planes...")
    calc_CD_PB_M3C2(outdir, t1, t2, corr_planes_list, t1_plane_pts_array_final,t1_segment_metrics_array_final,
                    t2_segment_metrics_array_final,reg_error)

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('%r took: %2.4f sec' % (f.__name__, te-ts))
        return result
    return wrap

# returns covariance matrix from an array of points
def get_covariance_matrix(point_array):
    cov_mat = np.cov(point_array.T)
    assert cov_mat.size == 9
    return cov_mat


# returns eigenvalue info from a covariance matrix
def get_eigen_infos(cov_mat):
	# get eigenvalues and eigenvectors
	eigenvalues_unsorted,eigenvectors_unsorted = np.linalg.eig(cov_mat)

	# sort eigenvalues and eigenvectors
	idx = eigenvalues_unsorted.argsort()[::-1]
	eigenvalues = eigenvalues_unsorted[idx]
	eigenvectors = eigenvectors_unsorted[:,idx]

	# write out and reformat values
	eL = eigenvalues[0]
	eI = eigenvalues[1]
	eS = eigenvalues[2]
	assert eL >= eI
	assert eI >= eS
	evecL = np.array(eigenvectors[:,0].T)
	evecI = np.array(eigenvectors[:,1].T)
	evecS = np.array(eigenvectors[:,2].T)
	return eL,eI,eS,evecL,evecI,evecS


# returns the unit vector of a vector (https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python/13849249#13849249)
def unit_vector(vector):
    return vector / np.linalg.norm(vector)


# returns the angle in deg between vectors 'v1' and 'v2'
def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) * 180./np.pi


# returns the smaller angle in deg between vectors 'v1' and 'v2'
def angle_between_mirr(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return min(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) * 180./np.pi,
               np.arccos(np.clip(np.dot(v1_u, -v2_u), -1.0, 1.0)) * 180./np.pi)


# returns the number of segments (planes) in a point cloud based on segment id
def get_number_of_planes(file):
    inPoints = pd.read_csv(file, sep=' ', header=None, usecols=[3])
    segment_cnt = int(inPoints.max())
    print(segment_cnt)
    return segment_cnt


def get_number_of_pts(file):
    inPoints = pd.read_csv(file, sep=' ', header=None, usecols=[3])
    pt_cnt = int(len(inPoints.index))
    print(pt_cnt)
    return pt_cnt


# returns segments of a point cloud in a 2D array
def create_segment_metrics_array(metrics_array,segment_pts_array,pc_t1,number_pts_timestamp):
    segment_array = np.zeros((600000,4),dtype=float)
    sorted_out = 0
    curr_seg_id = 1
    cnt_segments = 0
    curr_point_idx = 0
    cnt_pt = 0
    print("Processing file: ",pc_t1)
    fileobj_in = open(pc_t1,'r')
    lines = fileobj_in.readlines()
    # append not-used line to also get last point
    lines.append("0 0 0 1")

    # for every point:
    # write xyz + segment id in new 2D array. First point is written at index 0
    for point in tqdm(lines):
        point = [float(element) for element in point.split(" ")]
        x = point[0]
        y = point[1]
        z = point[2]
        seg_id = point[3]

        # add point to array based on current point index
        if cnt_pt<number_pts_timestamp:
            pts_array = np.array([x,y,z,seg_id])
            segment_pts_array[cnt_pt, :] = pts_array
        cnt_pt+=1

        # if segment id of current point is different from previous point, create entry for new segment
        if curr_seg_id!=point[3]:
            # store points of current segment in new variable
            curr_segment = segment_array[:curr_point_idx,:]
            seg_id = curr_segment[0][3]
            seg_id = int(seg_id)
            list_index = int(seg_id-1)

            # create 2D array with points of current segment
            plane_array = np.array(curr_segment)[:, 0:3]

            # calculate covariance matrix for each point based on all points of this segment
            cov_mat = get_covariance_matrix(plane_array)

            # get eigenvalue info
            eL,eI,eS,evecL,evecI,evecS = get_eigen_infos(cov_mat)

            # get plane sigma
            sigma_cm = sqrt(eS)*100

            # grab number of points per plane
            cnt_segment_pts = plane_array.shape[0]

            # get normal vector
            normal_vector = evecS

            # get CoG of segment
            CoG = np.mean(plane_array, axis=0)

            # largest eigenvalue
            eigenvalue_large = eL

            # middle eigenvalue
            eigenvalue_middle = eI

            # pdens calculated by dividing the number of points by the area which can be approximated by the extents
            # of a plane along (eL and eI)
            pdens = cnt_segment_pts/(eL*eI)

            # create array of points of current segment and additional segment attributes
            arr = np.array([seg_id,sigma_cm,cnt_segment_pts,normal_vector[0], normal_vector[1], normal_vector[2],
                                    CoG[0],CoG[1],CoG[2],eigenvalue_large,eigenvalue_middle,pdens,sigma_cm])

            # insert array of current segment in array containing all segments at index of segment id
            metrics_array[list_index, :] = arr

            cnt_segments+=1

            curr_seg_id = point[3]
            curr_point_idx = 0

        # else, point is added to the array of the current segment
        else:
            segment_array[curr_point_idx, :]  = point
            curr_point_idx +=1

    print("Number of planes: ", cnt_segments)
    fileobj_in.close()
    return(metrics_array, segment_pts_array, sorted_out)


# returns array of center of gravities for each segment
def create_CoG_array(metrics_array,cnt_segments_per_timestamp):
    CoG_array = np.zeros((cnt_segments_per_timestamp,4))

    for segment in tqdm(metrics_array):
        seg_id = segment[0]
        seg_id_idx = int(seg_id-1)
        CoG = segment[6:9]
        arr = np.array([seg_id,CoG[0],CoG[1],CoG[2]])
        CoG_array[seg_id_idx] = arr      # for each plane add a list containing CoG
    return CoG_array

"""
# create 2D list with sublists containing CoGs and segid per plane excluding the planes which were used for validation
def clean_CoG_seg_id_list(number_planes_timestamp,segment_metrics_array_final_timestamp):

    number_planes_clean = int(number_planes_timestamp-160)
    #print(validation_data_corr)
    CoG_segid_clean = np.zeros((number_planes_clean,4))

    number_planes = int(number_planes_timestamp)
    #print(validation_data_corr)
    CoG_segid_clean = np.zeros((number_planes,4))

    segid_validation = 0
    validation_segids = np.zeros((160,1))

    # timestamp determines which element of validation_data_corr has to be used in each list
    idx_cnt = 0
    for element in validation_data_corr:
        if timestamp == 1:
            segid_validation_t1 = int(element[0])
            validation_segids[idx_cnt] = np.array(segid_validation_t1)
            idx_cnt +=1
        if timestamp == 2:
            segid_validation_t2 = int(element[1])
            validation_segids[idx_cnt] = np.array(segid_validation_t2)
            idx_cnt += 1

    idx_cnt_2 = 0
    for element in tqdm(segment_metrics_array_final_timestamp):
        segid = int(element[0])
        #print(segid)
        CoG_x = element[6]
        CoG_y = element[7]
        CoG_z = element[8]
        # all planes whose segids are not part of the validation list shall be added to the clean list

        if segid not in validation_segids:
            #print(segid)
            CoG_segid_clean[idx_cnt_2] = np.array([CoG_x,CoG_y,CoG_z,segid])
            idx_cnt_2 += 1

        CoG_segid_clean[idx_cnt_2] = np.array([CoG_x,CoG_y,CoG_z,segid])
        idx_cnt_2 += 1
    #print(len(CoG_segid_clean))
    #print(idx_cnt_2)
    return CoG_segid_clean


# create 1D list with centroids of t1 or t2 excluding centroids of validation planes
def clean_centroid_list(CoG_segid_clean):
    length_CoG = len(CoG_segid_clean)
    CoG_clean = np.zeros((length_CoG,3))
    idx_cnt = 0
    for element in tqdm(CoG_segid_clean):
        # grab first element of list, which is segid and substract 1 to obtain right index
        #index = element[1]-1
        #CoG = timestamp_metrics_list[index][4]
        CoG_x = element[0]
        CoG_y = element[1]
        CoG_z = element[2]
        CoG_clean[idx_cnt] = np.array([CoG_x,CoG_y,CoG_z])
        idx_cnt += 1
    return CoG_clean
"""

# returns list of segment ids of corresponding segments of two timestamps t1 and t2
def apply_rf(number_of_planes_t1,CoG_array_t1,CoG_array_t2,t2_kd_tree,
             t1_segment_metrics_array_final,t2_segment_metrics_array_final,clf):
    non_corr_planes_found = np.zeros((number_of_planes_t1,2))
    corr_planes_list = np.zeros((number_of_planes_t1,2))

    # create counters
    corr_plane_idx = 0
    non_corr_plane_idx = 0
    only_1_plane_but_proba_too_low = 0
    more_than_1_plane_but_proba_too_low = 0
    no_planes_in_neigh = 0

    # for every segment in t1
    for segment in tqdm(CoG_array_t1):

       # creates list storing differences of segments metrics (as measure of plane simularity)
       candidate_diff_metrics_list = np.zeros((50000,6))

       # creates list storing only segment ids where indices of segids corresponds to indices in candidate_diff_metrics_list
       seg_id_candidate_diff_metrics = np.zeros((50000,1))

       # creates list containing probability of each plane for class 1
       probas_class1_list = np.zeros((50000,1))

       # creates list containing segids of candidate plane in t2 which are in the same order
       seg_id_probas_class1_list = np.zeros((50000,1))

       # gets segid of current segment
       seg_id_t1 = int(segment[0])

       # gets index to access correct element in timestamp1_metrics list
       seg_id_index_t1 = seg_id_t1-1
       segment_metrics_t1 = t1_segment_metrics_array_final[seg_id_index_t1]

       # gets normal vector and CoG
       vec_t1 = segment_metrics_t1[3:6]
       vec_t1 /= np.linalg.norm(vec_t1)
       CoG = segment[1:4]
       CoG_array_t1 = CoG[np.newaxis, :]   # adds second dimension as this required for kd tree search

       # stores potentially corresponding segments (candidates) in t2 derived from a neighborhood search in a list
       # list contains indices of candidates according to CoG_sed_id_list
       candidate_list = t2_kd_tree.query_radius(CoG_array_t1, r=2.0, return_distance=False)[0]

       # creates new array storing metrics of candidates
       segment_metrics = np.zeros((len(candidate_list), t2_segment_metrics_array_final.shape[1]))

       # creates empty list containing segment ids of candicates only
       seg_id_cand_test_list = []

       # for each candidate
       for loopindex, index_cog in enumerate(candidate_list):
            seg_id_t2 = int(CoG_array_t2[index_cog, 0])

            # gets correct index to access segment metrics array
            seg_id_idx = seg_id_t2-1

            # gets segment metrics of current candidate and adds them to an array of segment metrics of all candidates
            segment_metrics[loopindex, :] = t2_segment_metrics_array_final[seg_id_idx]

            seg_id_cand_test_list += [seg_id_t2]

       # for each candidate
       for loopindex, segid_cand_t2 in zip(range(segment_metrics.shape[0]), seg_id_cand_test_list):

            # stores segment metrics in new variable
            candidate_metrics_t2 = segment_metrics[loopindex, :]

            # gets normal vector of candidate
            vec_t2 = candidate_metrics_t2[3:6]
            vec_t2 /= np.linalg.norm(vec_t2)

            # compute angle in degree
            angle = np.arccos(np.dot(vec_t1, vec_t2)) * np.pi/180

            # calculate absolute difference between number of points of segment of t1 and of candidate of t2
            diffs = segment_metrics_t1 - candidate_metrics_t2
            absdiffs = np.abs(diffs[[1,2,9,10]])
            diff_std, diff_pts, diff_eL, diff_eI = absdiffs
            diff_pdens = diffs[11]

            # extents array of differences in segment metrics (as measure of plane similarity)
            candidate_diff_metrics_list[loopindex] = np.array([angle,diff_pts,diff_std,diff_eL,diff_eI,diff_pdens])

            # extents array of segment ids
            seg_id_candidate_diff_metrics[loopindex] = np.array(segid_cand_t2)


       candidate_diff_metrics_array = candidate_diff_metrics_list[:segment_metrics.shape[0],:]   # 2D
       seg_id_candidate_diff_metrics_array = seg_id_candidate_diff_metrics[:segment_metrics.shape[0]] # 1D

       # starts binary classification if at least one candidate was found in neighbourhood of segment of t1
       if len(candidate_diff_metrics_array)>0:
            y_pred = clf.predict_proba(candidate_diff_metrics_array)

            y_pred_list = np.array(y_pred).tolist()

            proba_idx = 0
            for c, element in enumerate(y_pred_list):

                # gets segment id of candidate
                seg_id_t2 = seg_id_candidate_diff_metrics[c]

                # gets class probability for class "1" (= corresponding)
                proba_class1 = element[1]

                # extents list of probabilities of candidates for class "1"  (= corresponding)
                probas_class1_list[proba_idx] = np.array(proba_class1)

                # create a list with segment id of candidates
                # segment id at index 0 corresponds to class probability of candidate in probas_class1_list at index 0
                seg_id_probas_class1_list[proba_idx] = np.array(seg_id_t2)
                proba_idx += 1

            # converts lists to arrays
            probas_class1_array = probas_class1_list[:proba_idx]
            seg_id_probas_class1_array = seg_id_probas_class1_list[:proba_idx]

            # finds plane with highest class probability for class 1
            # if more than 1 plane candidate was found
            if len(probas_class1_array)>1:

                # get index of candidate with highest probability for class 1 (most similar candidate)
                proba_max_idx = np.argmax(probas_class1_array)

                # get actual probability value
                proba_max = np.max(probas_class1_array)

                # get segment id of candidate using the index
                seg_id_proba_max_t2 = int(seg_id_probas_class1_array[proba_max_idx])

                # remove highest probability value
                probas_class1_array = np.delete(probas_class1_array, proba_max_idx)

                # get index of candidate with second highest probability for class 1 (second most similar candidate)
                proba_max_idx_2nd = np.argmax(probas_class1_array)

                # get actual probability value
                proba_max_2nd = np.max(probas_class1_array)

                # get segment id of candidate using the index
                seg_id_proba_2nd_max_t2 = seg_id_probas_class1_array[proba_max_idx_2nd]

                # if class probability of most similar plane is > 80%
                if proba_max > 0.80:

                    # computes difference in class probability between most similar and second most similar candidate
                    diff_proba_max_2nd = proba_max - proba_max_2nd

                    # if difference in class probability is > 10%
                    if diff_proba_max_2nd > 10.0:

                        # most similar plane candidate of t2 is accepted as corresponding to current segment of t1
                        # segment ids of corresponding planes are added to list
                        corr_planes_list[corr_plane_idx] = np.array([seg_id_t1,seg_id_proba_max_t2])
                        corr_plane_idx +=1

                    # if difference in class probability is < 10%
                    else:

                        # most similar plane candidate of t2 is not accepted as corresponding to current segment of t1
                        non_corr_planes_found[non_corr_plane_idx] = np.array(seg_id_t1)
                        non_corr_plane_idx+=1

                # if class probability of most similar plane is < 80%
                else:
                    # most similar plane candidate of t2 is not accepted as corresponding to current segment of t1
                    non_corr_planes_found[non_corr_plane_idx] = np.array(seg_id_t1,proba_max)
                    non_corr_plane_idx+=1
                    more_than_1_plane_but_proba_too_low+=1

            # if only 1 candidate was found in neighbourhood of current segment of t1
            else:
                probas_class1_array = np.array(probas_class1_list)

                # get index of element with highest proba value
                proba_max_idx = np.argmax(probas_class1_array)

                # get actual proba value
                proba_max = np.max(probas_class1_array)

                # if class probability of most similar plane is > 80%
                if proba_max > 0.80:
                    # most similar plane candidate of t2 is accepted as corresponding to current segment of t1
                    # segment ids of corresponding planes are added to list
                    seg_id_proba_max_t2 = int(seg_id_probas_class1_list[proba_max_idx])
                    corr_planes_list[corr_plane_idx] = np.array([seg_id_t1,seg_id_proba_max_t2])
                    corr_plane_idx+=1

                # if class probability of most similar plane is < 80%
                else:
                    # most similar plane candidate of t2 is not accepted as corresponding to current segment of t1
                    non_corr_planes_found[non_corr_plane_idx] = np.array(seg_id_t1)
                    non_corr_plane_idx+=1
                    only_1_plane_but_proba_too_low +=1

        # if no candidates were found in neighbourhood of current segment of t1
       else:
            non_corr_planes_found[non_corr_plane_idx] = np.array(seg_id_t1)
            non_corr_plane_idx+=1
            no_planes_in_neigh +=1


    # removes empty entries from array
    corr_planes_list_cut = corr_planes_list[:corr_plane_idx,:]

    # prints some info on how many correspondences were found and why some segments in t1 were not assigned a corresponding segment in t2
    print("only_1_plane_but_proba_too_low: ", only_1_plane_but_proba_too_low)
    print("no_planes_in_neigh: ", no_planes_in_neigh)
    print("more_than_1_plane_but_proba_too_low: ", more_than_1_plane_but_proba_too_low)
    print(corr_plane_idx)
    print(non_corr_plane_idx)

    return corr_planes_list_cut


def calc_CD_PB_M3C2(outdir, t1, t2, corr_planes_list, t1_plane_pts,t1_segment_metrics,t2_segment_metrics,
                    reg_error):

    # defines name of output files
    filename_plane_based_M3C2_corepts_only = str(Path(outdir)) + "\\" + t1 + "_" + t2 + "plane_based_M3C2_cog_only.xyz"
    filename_plane_based_M3C2 = str(Path(outdir)) + "\\" + t1 + "_" + t2 + "plane_based_M3C2.xyz"

    # opens output files
    fileobj_plane_based_M3C2_corepts = open(filename_plane_based_M3C2_corepts_only, "w")
    fileobj_plane_based_M3C2 = open(filename_plane_based_M3C2, "w")

    # for each pair of corresponding segments (plane pair)
    for plane_pair in tqdm(corr_planes_list):

        # gets segment id of segments
        seg_id_t1 = plane_pair[0]
        seg_id_t2 = plane_pair[1]

        # gets index to access segment metrics
        index_t1 = int(seg_id_t1 - 1)
        index_t2 = int(seg_id_t2 - 1)

        # in 2D array with points of all segments: finds points where 3rd column (segid) = current segment id and grab xyz
        subset = t1_plane_pts[t1_plane_pts[:, 3] == seg_id_t1, 0:3]

        # gets CoGs, and normal vectors of segments of current plane pair
        t1_CoG = t1_segment_metrics[index_t1][6:9]
        t2_CoG = t2_segment_metrics[index_t2][6:9]
        normal_vector_t1 = t1_segment_metrics[index_t1][3:6]
        normal_vector_t2 = t2_segment_metrics[index_t2][3:6]
        angle_normal_vectors = angle_between_mirr(normal_vector_t1, normal_vector_t2)
        nx = normal_vector_t1[0]
        ny = normal_vector_t1[1]
        nz = normal_vector_t1[2]
        no_pts_plane_t1 = t1_segment_metrics[index_t1][2]
        no_pts_plane_t2 = t2_segment_metrics[index_t2][2]

        # calculates M3C2 distance & level of detection (LoDetection)
        # according to Lague et al. (2013): https://doi.org/10.1016/j.isprsjprs.2013.04.009
        # M3C2 distance = distance between centroids of segments of current plane pair along normal vector of plane of t1
        M3C2_dist_m = normal_vector_t1.dot(t1_CoG - t2_CoG)

        # sigma already in cm
        sigma_plane_t1 = (t1_segment_metrics[index_t1][12]) * (
            t1_segment_metrics[index_t1][12])

        sigma_plane_t2 = (t2_segment_metrics[index_t2][12]) * (
            t2_segment_metrics[index_t2][12])

        LoDetection_cm = (1.96 * (sqrt(((sigma_plane_t1) / (no_pts_plane_t1)) + (
                (sigma_plane_t2) / (no_pts_plane_t2))))) + reg_error  # regerror: 1.1 cm
        LoDetection_m = LoDetection_cm / 100

        # checks if significant change was derived (1 = yes; 0 = no)
        if LoDetection_m < M3C2_dist_m:
            sign_change = 1
        else:
            sign_change = 0

        # writes output
        fileobj_plane_based_M3C2_corepts.write("%1.3f,%1.3f,%1.3f\n" % (t2_CoG[0], t2_CoG[1], t2_CoG[2]))
        fileobj_plane_based_M3C2.write("%1.3f,%1.3f,%1.3f,%1.3f,%1.3f,%i,%1.3f,%1.3f,%1.3f,%1.3f,%1.3f\n" %
                               (t1_CoG[0], t1_CoG[1], t1_CoG[2], M3C2_dist_m, LoDetection_m, seg_id_t1, sign_change,
                                   nx, ny, nz,angle_normal_vectors))

    # closes files
    fileobj_plane_based_M3C2_corepts.close()
    fileobj_plane_based_M3C2.close()



if __name__ == '__main__':
    main(r"J:\01_Projekte\AHK-4D\Paper\00_heidata\ASCII\output\20190719_segm_merged.xyz", \
            r"J:\01_Projekte\AHK-4D\Paper\00_heidata\ASCII\output\20190803_segm_merged.xyz",
         r"J:\01_Projekte\AHK-4D\Paper\00_heidata\ASCII\input\clf\clf.p", r"J:\01_Projekte\AHK-4D\Paper\00_heidata\ASCII\output",
            "20190719","20190803",
         0.012)

