#
# This file is part of the M3C2-EP distribution.
# Copyright (c) 2019-2021 Lukas Winiwarter, 3DGeo, Universitaet Heidelberg.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

from sklearn.neighbors import KDTree
import numpy as np
import math
import multiprocessing as mp
import pickle
import scipy.io as sio
import scipy.stats as sstats
import os
from collections import namedtuple
from xml.dom import minidom
import re
from tqdm import tqdm
import time
import sys

np.seterr(divide='ignore', invalid='ignore')

strip_name_finder = re.compile(r"^IdGridMov\[(.*?)\]$", flags=re.M)
strip_name_finder_Fix = re.compile(r"^IdGridFix\[(.*?)\]$", flags=re.M)
rPM_finder = re.compile(r"^RefPointMov\[(.*?)\]$", flags=re.M)
rPF_finder = re.compile(r"^RefPointFix\[(.*?)\]$", flags=re.M)
tfM_finder = re.compile(r"TrafPars\[(.*?)\]", flags=re.S)

M3C2MetaInfo = namedtuple('M3C2MetaInfo', ('spInfos', 'tfM', 'Cxx', 'redPoint', 'searchrad', 'maxdist'))
SPMetaInfo = namedtuple('SPMetaInfo', ('origin', 'sigma_range', 'sigma_yaw', 'sigma_scan', 'ppm'))

eijk = np.zeros((3, 3, 3))
eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1

dij = np.zeros((3, 3))
dij[0, 0] = dij[1, 1] = dij[2, 2] = 1


def picklebig(obj, file):
    max_bytes = 2 ** 31 - 1
    # write
    bytes_out = pickle.dumps(obj)
    with open(file, 'wb') as f_out:
        for idx in range(0, len(bytes_out), max_bytes):
            f_out.write(bytes_out[idx:idx + max_bytes])


def unpicklebig(file):
    max_bytes = 2 ** 31 - 1
    bytes_in = bytearray(0)
    input_size = os.path.getsize(file)
    with open(file, 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    return pickle.loads(bytes_in)


n = np.zeros((3,))
poa_pts = np.zeros((3, 100))
path_opt = np.einsum_path('mi, ijk, j, kn -> mn', dij, eijk, n, poa_pts, optimize='optimal')


def getAlongAcrossSqBatch(pts, poa, n):
    pts_poa = pts - poa[:, np.newaxis]
    alongs = n.dot(pts_poa)
    poa_pts = poa[:, np.newaxis] - pts
    crosses = np.einsum('mi, ijk, j, kn -> mn', dij, eijk, n, poa_pts, optimize=path_opt[0])
    across2 = np.einsum('ij, ij -> j', crosses, crosses)
    return (alongs, across2)


def read_from_las(path):
    from laspy.file import File as LasFile
    inFile = LasFile(path, mode='r')
    coords = np.vstack((inFile.x, inFile.y, inFile.z)).transpose()
    try:
        normals = getattr(inFile, 'normals', None)
    except:
        normals = None
    if normals is None:
        try:
            n0 = getattr(inFile, 'normals0', None)
            n1 = getattr(inFile, 'normals1', None)
            n2 = getattr(inFile, 'normals2', None)
            normals = np.stack((n0, n1, n2)).T
        except:
            normals = None

    if "FileSourceID" in inFile.points.dtype.fields["point"][0].fields:
        scanpos = inFile.points["point"]["FileSourceID"]
    else:
        try:
            scanpos = getattr(inFile, 'pt_src_id', None)
        except:
            scanpos = None

    if "amplitude" in inFile.points.dtype.fields["point"][0].fields:
        amp = inFile.points["point"]["amplitude"]
    elif "Amplitude" in inFile.points.dtype.fields["point"][0].fields:
        amp = inFile.points["point"]["Amplitude"]
    else:
        amp = None
    if "deviation" in inFile.points.dtype.fields["point"][0].fields:
        dev = inFile.points["point"]["deviation"]
    elif "Deviation" in inFile.points.dtype.fields["point"][0].fields:
        dev = inFile.points["point"]["Deviation"]
    else:
        dev = None
    return coords, normals, scanpos, amp, dev


las_data_types = {
    int: 4,
    float: 10,  # or 10 for double prec
    np.dtype(np.float64): 10,
    np.dtype(np.int32): 4
}


def write_to_las(path, points, attrdict):
    LasFile = None  # full reset
    LasHeader = None
    from laspy.file import File as LasFile
    from laspy.header import Header as LasHeader
    hdr = LasHeader(version_major=1, version_minor=4)
    outFile = LasFile(path, mode="w", header=hdr)
    for attrname in attrdict:
        if attrname == "normals":
            outFile.define_new_dimension(name=attrname, data_type=30, description=attrname)
            continue
        try:
            dt = 9  # default data type
            if attrdict[attrname].dtype in las_data_types:
                dt = las_data_types[attrdict[attrname].dtype]
            else:
                print("Unknown data type: '%s', attemping saving as float." % attrdict[attrname].dtype)
            outFile.define_new_dimension(name=attrname.lower(), data_type=dt, description=attrname.lower())
        except Exception as e:
            print("Failed adding dimension %s: %s" % (attrname.lower(), e))
    xmin, ymin, zmin = np.min(points, axis=0)
    outFile.header.offset = [xmin, ymin, zmin]
    outFile.header.scale = [0.001, 0.001, 0.001]
    outFile.x = points[:, 0]
    outFile.y = points[:, 1]
    outFile.z = points[:, 2]
    for attrname in attrdict:
        setattr(outFile, attrname.lower(), attrdict[attrname])
    outFile.close()
    outFile = None


def process_corepoint_list(corepoints, corepoint_normals,
                           p1_idx, p1_shm_name, p1_size, p1_positions,
                           p2_idx, p2_shm_name, p2_size, p2_positions,
                           M3C2Meta, idx, return_dict, pbarQueue):
    pbarQueue.put((0, 1))
    p1_shm = mp.shared_memory.SharedMemory(name=p1_shm_name)
    p2_shm = mp.shared_memory.SharedMemory(name=p2_shm_name)
    p1_coords = np.ndarray(p1_size, dtype=np.float, buffer=p1_shm.buf)
    p2_coords = np.ndarray(p2_size, dtype=np.float, buffer=p2_shm.buf)

    max_dist = M3C2Meta['maxdist']
    search_radius = M3C2Meta['searchrad']

    M3C2_vals = np.full((corepoints.shape[0]), np.nan, dtype=np.float64)
    M3C2_vals_old = np.full((corepoints.shape[0]), np.nan, dtype=np.float64)
    M3C2_uncs = np.full((corepoints.shape[0]), np.nan, dtype=np.float64)
    M3C2_LoD = np.full((corepoints.shape[0]), np.nan, dtype=np.float64)
    M3C2_LoD_sig = np.full((corepoints.shape[0]), np.nan, dtype=np.float64)
    M3C2_leg_LoD = np.full((corepoints.shape[0]), np.nan, dtype=np.float64)
    M3C2_leg_LoD_sig = np.full((corepoints.shape[0]), np.nan, dtype=np.float64)
    M3C2_flags = np.full((corepoints.shape[0]), np.nan, dtype=np.int32)
    M3C2_N1 = np.full((corepoints.shape[0]), np.nan, dtype=np.int32)
    M3C2_N2 = np.full((corepoints.shape[0]), np.nan, dtype=np.int32)
    M3C2_C1 = np.full((corepoints.shape[0]), np.nan, dtype=np.int32)
    M3C2_C2 = np.full((corepoints.shape[0]), np.nan, dtype=np.int32)
    M3C2_Cxx1max = np.full((corepoints.shape[0]), np.nan, dtype=np.float64)
    M3C2_Cxx2max = np.full((corepoints.shape[0]), np.nan, dtype=np.float64)

    for cp_idx, p1_neighbours in enumerate(p1_idx):
        flag = 0
        n = corepoint_normals[cp_idx]
        p1_curr_pts = p1_coords[p1_neighbours, :]
        along1, acrossSq1 = getAlongAcrossSqBatch(p1_curr_pts.T, corepoints[cp_idx], n)
        p1_curr_pts = p1_curr_pts[np.logical_and(np.abs(along1) <= max_dist, acrossSq1 <= search_radius ** 2), :]
        p1_scanPos = p1_positions[p1_neighbours]
        p1_scanPos = p1_scanPos[np.logical_and(np.abs(along1) <= max_dist, acrossSq1 <= search_radius ** 2)]
        if p1_curr_pts.shape[0] < M3C2Meta["minneigh"]:
            M3C2_flags[cp_idx] = 1
            pbarQueue.put((1, 0))  # point processed
            M3C2_N1[cp_idx] = p1_curr_pts.shape[0]
            continue
        elif p1_curr_pts.shape[0] > M3C2Meta["maxneigh"]:
            p1_curr_pts = p1_curr_pts[np.argsort(acrossSq1[:M3C2Meta['maxneigh']])]
            p1_scanPos = p1_scanPos[np.argsort(acrossSq1[:M3C2Meta['maxneigh']])]
            flag = 2

        Cxx = M3C2Meta['Cxx']
        tfM = M3C2Meta['tfM']
        origins = np.array([SP['origin'] for SP in M3C2Meta['spInfos'][0]])
        redPoint = M3C2Meta['redPoint']
        sigmas = np.array([[SP['sigma_range'],
                            SP['sigma_range'],
                            SP['sigma_scan'],
                            SP['sigma_yaw']]
                           for SP in M3C2Meta['spInfos'][0]])

        p1_weighted_CoG, p1_local_Cxx = get_local_mean_and_Cxx_nocorr(Cxx, tfM, origins, redPoint, sigmas, p1_curr_pts,
                                                                      p1_scanPos, epoch=0,
                                                                      tf=False)  # only one dataset has been transformed
        along1_var = np.var(along1[np.logical_and(np.abs(along1) <= max_dist, acrossSq1 <= search_radius ** 2)])

        p2_neighbours = p2_idx[cp_idx]
        p2_curr_pts = p2_coords[p2_neighbours, :]
        along2, acrossSq2 = getAlongAcrossSqBatch(p2_curr_pts.T, corepoints[cp_idx], n)
        p2_curr_pts = p2_curr_pts[np.logical_and(np.abs(along2) <= max_dist, acrossSq2 <= search_radius ** 2), :]
        p2_scanPos = p2_positions[p2_neighbours]
        p2_scanPos = p2_scanPos[np.logical_and(np.abs(along2) <= max_dist, acrossSq2 <= search_radius ** 2)]
        if p2_curr_pts.shape[0] < M3C2Meta["minneigh"]:
            M3C2_flags[cp_idx] = 1
            pbarQueue.put((1, 0))  # point processed
            M3C2_N2[cp_idx] = p2_curr_pts.shape[0]
            continue
        elif p2_curr_pts.shape[0] > M3C2Meta["maxneigh"]:
            p2_curr_pts = p2_curr_pts[np.argsort(acrossSq2[:M3C2Meta['maxneigh']])]
            p2_scanPos = p2_scanPos[np.argsort(acrossSq2[:M3C2Meta['maxneigh']])]
            flag = 2

        origins = np.array([SP['origin'] for SP in M3C2Meta['spInfos'][1]])
        sigmas = np.array([[SP['sigma_range'],
                            SP['sigma_range'],
                            SP['sigma_scan'],
                            SP['sigma_yaw']]
                           for SP in M3C2Meta['spInfos'][1]])
        p2_weighted_CoG, p2_local_Cxx = get_local_mean_and_Cxx_nocorr(Cxx, tfM, origins, redPoint, sigmas, p2_curr_pts,
                                                                      p2_scanPos, epoch=1, tf=True)
        along2_var = np.var(along2[np.logical_and(np.abs(along2) <= max_dist, acrossSq2 <= search_radius ** 2)])

        p1_CoG = p1_weighted_CoG
        p2_CoG = p2_weighted_CoG

        p1_CoG_Cxx = p1_local_Cxx
        p2_CoG_Cxx = p2_local_Cxx

        p1_p2_CoG_Cxx = np.zeros((6, 6))
        p1_p2_CoG_Cxx[0:3, 0:3] = p1_CoG_Cxx
        p1_p2_CoG_Cxx[3:6, 3:6] = p2_CoG_Cxx

        M3C2_dist = n.dot(p1_CoG - p2_CoG)
        M3C2_dist_old = n.dot(np.mean(p1_curr_pts, axis=0) - np.mean(p2_curr_pts, axis=0))
        F = np.hstack([-n, n])
        M3C2_unc = np.sqrt(np.dot(F, np.dot(p1_p2_CoG_Cxx, F)))

        M3C2_vals[cp_idx] = M3C2_dist
        M3C2_vals_old[cp_idx] = M3C2_dist_old
        M3C2_uncs[cp_idx] = M3C2_unc

        N1 = p1_curr_pts.shape[0]
        N2 = p2_curr_pts.shape[0]

        sigmaD = p1_CoG_Cxx + p2_CoG_Cxx

        p = 3  # three dimensional
        Tsqalt = n.T.dot(np.linalg.inv(sigmaD)).dot(n)

        M3C2_LoD[cp_idx] = np.sqrt(sstats.chi2.ppf(.95, p) / Tsqalt)

        M3C2_LoD_sig[cp_idx] = np.abs(M3C2_dist) >= M3C2_LoD[cp_idx]
        M3C2_leg_LoD[cp_idx] = 1.96 * (
                    np.sqrt(along1_var / p1_curr_pts.shape[0] + along2_var / p2_curr_pts.shape[0]) + M3C2Meta[
                'leg_ref_err'])
        M3C2_leg_LoD_sig[cp_idx] = np.abs(M3C2_dist_old) > M3C2_leg_LoD[cp_idx]
        M3C2_flags[cp_idx] = flag
        M3C2_N1[cp_idx] = N1
        M3C2_N2[cp_idx] = N2
        M3C2_C1[cp_idx] = np.count_nonzero(np.unique(p1_scanPos))
        M3C2_C2[cp_idx] = np.count_nonzero(np.unique(p2_scanPos))
        cxx1val, cxx1vec = np.linalg.eig(p1_CoG_Cxx)
        cxx2val, cxx2vec = np.linalg.eig(p2_CoG_Cxx)
        M3C2_Cxx1max[cp_idx] = np.sqrt(np.max(cxx1val))
        M3C2_Cxx2max[cp_idx] = np.sqrt(np.max(cxx2val))

        # if cp_idx % 10 == 0: pbarQueue.put((10, 0))  # point processed
        pbarQueue.put((1, 0))  # point processed

    return_dict[idx] = {'unc': M3C2_uncs,
                        'lod_leg': M3C2_leg_LoD,
                        'lod_leg_sig': M3C2_leg_LoD_sig,
                        'lod_new': M3C2_LoD,
                        'lod_new_sig': M3C2_LoD_sig,
                        'lod_diff': M3C2_leg_LoD - M3C2_leg_LoD,
                        'val': M3C2_vals,
                        'val_old': M3C2_vals_old,
                        'flag': M3C2_flags,
                        'm3c2_n1': M3C2_N1,
                        'm3c2_n2': M3C2_N2,
                        'm3c2_c1': M3C2_C1,
                        'm3c2_c2': M3C2_C2,
                        'm3c2_cxx1max': M3C2_Cxx1max,
                        'm3c2_cxx2max': M3C2_Cxx2max,
                        }
    pbarQueue.put((0, -1))
    p1_shm.close()
    p2_shm.close()


def get_local_mean_and_Cxx_nocorr(Cxx, tfM, origins, redPoint, sigmas, curr_pts, curr_pos, epoch, tf=True):
    nPts = curr_pts.shape[0]
    # Cxx = M3C2Meta['Cxx']
    A = np.tile(np.eye(3), (nPts, 1))
    ATP = np.zeros((3, 3 * nPts))
    # tfM = M3C2Meta['tfM'] if tf else np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    tfM = tfM if tf else np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    dx = np.zeros((nPts,), dtype=np.float)
    dy = np.zeros((nPts,), dtype=np.float)
    dz = np.zeros((nPts,), dtype=np.float)
    rrange = np.zeros((nPts,), dtype=np.float)
    sinscan = np.zeros((nPts,), dtype=np.float)
    cosscan = np.zeros((nPts,), dtype=np.float)
    cosyaw = np.zeros((nPts,), dtype=np.float)
    sinyaw = np.zeros((nPts,), dtype=np.float)
    sigmaRange = np.zeros((nPts,), dtype=np.float)
    sigmaYaw = np.zeros((nPts,), dtype=np.float)
    sigmaScan = np.zeros((nPts,), dtype=np.float)

    for scanPosId in np.unique(curr_pos):
        scanPos = np.array(origins[scanPosId - 1, :])
        scanPosPtsIdx = curr_pos == scanPosId

        dd = curr_pts[scanPosPtsIdx, :] - scanPos[np.newaxis, :]
        dlx, dly, dlz = dd[:, 0], dd[:, 1], dd[:, 2]
        yaw = np.arctan2(dly, dlx)
        planar_dist = np.hypot(dlx, dly)
        scan = np.pi / 2 - np.arctan(dlz / planar_dist)
        rrange[scanPosPtsIdx] = np.hypot(planar_dist, dlz)
        sinscan[scanPosPtsIdx] = np.sin(scan)
        cosscan[scanPosPtsIdx] = np.cos(scan)
        sinyaw[scanPosPtsIdx] = np.sin(yaw)
        cosyaw[scanPosPtsIdx] = np.cos(yaw)

        dr = curr_pts[scanPosPtsIdx, :] - redPoint
        dx[scanPosPtsIdx] = dr[:, 0]
        dy[scanPosPtsIdx] = dr[:, 1]
        dz[scanPosPtsIdx] = dr[:, 2]

        sigmaRange[scanPosPtsIdx] = np.array(
            np.sqrt(sigmas[scanPosId - 1][0] ** 2 +
                    sigmas[scanPosId - 1][1] * 1e-6 * rrange[scanPosPtsIdx] ** 2))  # a + b*d
        sigmaYaw[scanPosPtsIdx] = np.array(sigmas[scanPosId - 1][2])
        sigmaScan[scanPosPtsIdx] = np.array(sigmas[scanPosId - 1][3])

    if tf:
        SigmaXiXj = (dx ** 2 * Cxx[0, 0] +  # a11a11
                     2 * dx * dy * Cxx[0, 1] +  # a11a12
                     dy ** 2 * Cxx[1, 1] +  # a12a12
                     2 * dy * dz * Cxx[1, 2] +  # a12a13
                     dz ** 2 * Cxx[2, 2] +  # a13a13
                     2 * dz * dx * Cxx[0, 2] +  # a11a13
                     2 * (dx * Cxx[0, 9] +  # a11tx
                          dy * Cxx[1, 9] +  # a12tx
                          dz * Cxx[2, 9]) +  # a13tx
                     Cxx[9, 9])  # txtx

        SigmaYiYj = (dx ** 2 * Cxx[3, 3] +  # a21a21
                     2 * dx * dy * Cxx[3, 4] +  # a21a22
                     dy ** 2 * Cxx[4, 4] +  # a22a22
                     2 * dy * dz * Cxx[4, 5] +  # a22a23
                     dz ** 2 * Cxx[5, 5] +  # a23a23
                     2 * dz * dx * Cxx[3, 5] +  # a21a23
                     2 * (dx * Cxx[3, 10] +  # a21ty
                          dy * Cxx[4, 10] +  # a22ty
                          dz * Cxx[5, 10]) +  # a23ty
                     Cxx[10, 10])  # tyty

        SigmaZiZj = (dx ** 2 * Cxx[6, 6] +  # a31a31
                     2 * dx * dy * Cxx[6, 7] +  # a31a32
                     dy ** 2 * Cxx[7, 7] +  # a32a32
                     2 * dy * dz * Cxx[7, 8] +  # a32a33
                     dz ** 2 * Cxx[8, 8] +  # a33a33
                     2 * dz * dx * Cxx[6, 8] +  # a31a33
                     2 * (dx * Cxx[6, 11] +  # a31tz
                          dy * Cxx[7, 11] +  # a32tz
                          dz * Cxx[8, 11]) +  # a33tz
                     Cxx[11, 11])  # tztz

        SigmaXiYj = (Cxx[9, 10] +  # txty
                     dx * Cxx[0, 10] +  # a11ty
                     dy * Cxx[1, 10] +  # a12ty
                     dz * Cxx[2, 10] +  # a13ty
                     dx * (Cxx[3, 9] +
                           Cxx[0, 3] * dx +
                           Cxx[1, 3] * dy +
                           Cxx[2, 3] * dz) +
                     dy * (Cxx[4, 9] +
                           Cxx[0, 4] * dx +
                           Cxx[1, 4] * dy +
                           Cxx[2, 4] * dz) +
                     dz * (Cxx[5, 9] +
                           Cxx[0, 5] * dx +
                           Cxx[1, 5] * dy +
                           Cxx[2, 5] * dz)
                     )

        SigmaXiZj = (Cxx[9, 11] +  # txtz
                     dx * Cxx[0, 11] +  # a11tz
                     dy * Cxx[1, 11] +  # a12tz
                     dz * Cxx[2, 11] +  # a13tz
                     dx * (Cxx[6, 9] +
                           Cxx[0, 6] * dx +
                           Cxx[1, 6] * dy +
                           Cxx[2, 6] * dz) +
                     dy * (Cxx[7, 9] +
                           Cxx[0, 7] * dx +
                           Cxx[1, 7] * dy +
                           Cxx[2, 7] * dz) +
                     dz * (Cxx[8, 9] +
                           Cxx[0, 8] * dx +
                           Cxx[1, 8] * dy +
                           Cxx[2, 8] * dz)
                     )

        SigmaYiZj = (Cxx[10, 11] +  # tytz
                     dx * Cxx[6, 10] +  # a21tx
                     dy * Cxx[7, 10] +  # a22tx
                     dz * Cxx[8, 10] +  # a23tx
                     dx * (Cxx[3, 11] +
                           Cxx[3, 6] * dx +
                           Cxx[3, 7] * dy +
                           Cxx[3, 8] * dz) +
                     dy * (Cxx[4, 11] +
                           Cxx[4, 6] * dx +
                           Cxx[4, 7] * dy +
                           Cxx[4, 8] * dz) +
                     dz * (Cxx[5, 11] +
                           Cxx[5, 6] * dx +
                           Cxx[5, 7] * dy +
                           Cxx[5, 8] * dz)
                     )
        C11 = np.sum(SigmaXiXj)  # sum over all j
        C12 = np.sum(SigmaXiYj)  # sum over all j
        C13 = np.sum(SigmaXiZj)  # sum over all j
        C22 = np.sum(SigmaYiYj)  # sum over all j
        C23 = np.sum(SigmaYiZj)  # sum over all j
        C33 = np.sum(SigmaZiZj)  # sum over all j
        local_Cxx = np.array([[C11, C12, C13], [C12, C22, C23], [C13, C23, C33]])
    else:
        local_Cxx = np.zeros((3, 3))

    C11p = ((tfM[0, 0] * cosyaw * sinscan +  # dX/dRange - measurements
             tfM[0, 1] * sinyaw * sinscan +
             tfM[0, 2] * cosscan) ** 2 * sigmaRange ** 2 +
            (- 1 * tfM[0, 0] * rrange * sinyaw * sinscan +  # dX/dYaw
             tfM[0, 1] * rrange * cosyaw * sinscan) ** 2 * sigmaYaw ** 2 +
            (tfM[0, 0] * rrange * cosyaw * cosscan +  # dX/dScan
             tfM[0, 1] * rrange * sinyaw * cosscan +
             -1 * tfM[0, 2] * rrange * sinscan) ** 2 * sigmaScan ** 2)

    C12p = ((tfM[1, 0] * cosyaw * sinscan +  # dY/dRange - measurements
             tfM[1, 1] * sinyaw * sinscan +
             tfM[1, 2] * cosscan) *
            (tfM[0, 0] * cosyaw * sinscan +  # dX/dRange - measurements
             tfM[0, 1] * sinyaw * sinscan +
             tfM[0, 2] * cosscan) * sigmaRange ** 2 +
            (- 1 * tfM[1, 0] * rrange * sinyaw * sinscan +  # dY/dYaw
             tfM[1, 1] * rrange * cosyaw * sinscan) *
            (- 1 * tfM[0, 0] * rrange * sinyaw * sinscan +  # dX/dYaw
             tfM[0, 1] * rrange * cosyaw * sinscan) * sigmaYaw ** 2 +
            (tfM[0, 0] * rrange * cosyaw * cosscan +  # dX/dScan
             tfM[0, 1] * rrange * sinyaw * cosscan +
             -1 * tfM[0, 2] * rrange * sinscan) *
            (tfM[1, 0] * rrange * cosyaw * cosscan +  # dY/dScan
             tfM[1, 1] * rrange * sinyaw * cosscan +
             -1 * tfM[1, 2] * rrange * sinscan) * sigmaScan ** 2)

    C22p = ((tfM[1, 0] * cosyaw * sinscan +  # dY/dRange - measurements
             tfM[1, 1] * sinyaw * sinscan +
             tfM[1, 2] * cosscan) ** 2 * sigmaRange ** 2 +
            (- 1 * tfM[1, 0] * rrange * sinyaw * sinscan +  # dY/dYaw
             tfM[1, 1] * rrange * cosyaw * sinscan) ** 2 * sigmaYaw ** 2 +
            (tfM[1, 0] * rrange * cosyaw * cosscan +  # dY/dScan
             tfM[1, 1] * rrange * sinyaw * cosscan +
             -1 * tfM[1, 2] * rrange * sinscan) ** 2 * sigmaScan ** 2)

    C23p = ((tfM[1, 0] * cosyaw * sinscan +  # dY/dRange - measurements
             tfM[1, 1] * sinyaw * sinscan +
             tfM[1, 2] * cosscan) *
            (tfM[2, 0] * cosyaw * sinscan +  # dZ/dRange - measurements
             tfM[2, 1] * sinyaw * sinscan +
             tfM[2, 2] * cosscan) * sigmaRange ** 2 +
            (- 1 * tfM[1, 0] * rrange * sinyaw * sinscan +  # dY/dYaw
             tfM[1, 1] * rrange * cosyaw * sinscan) *
            (- 1 * tfM[2, 0] * rrange * sinyaw * sinscan +  # dZ/dYaw
             tfM[2, 1] * rrange * cosyaw * sinscan) * sigmaYaw ** 2 +
            (tfM[2, 0] * rrange * cosyaw * cosscan +  # dZ/dScan
             tfM[2, 1] * rrange * sinyaw * cosscan +
             -1 * tfM[2, 2] * rrange * sinscan) *
            (tfM[1, 0] * rrange * cosyaw * cosscan +  # dY/dScan
             tfM[1, 1] * rrange * sinyaw * cosscan +
             -1 * tfM[1, 2] * rrange * sinscan) * sigmaScan ** 2)

    C33p = ((tfM[2, 0] * cosyaw * sinscan +  # dZ/dRange - measurements
             tfM[2, 1] * sinyaw * sinscan +
             tfM[2, 2] * cosscan) ** 2 * sigmaRange ** 2 +
            (- 1 * tfM[2, 0] * rrange * sinyaw * sinscan +  # dZ/dYaw
             tfM[2, 1] * rrange * cosyaw * sinscan) ** 2 * sigmaYaw ** 2 +
            (tfM[2, 0] * rrange * cosyaw * cosscan +  # dZ/dScan
             tfM[2, 1] * rrange * sinyaw * cosscan +
             -1 * tfM[2, 2] * rrange * sinscan) ** 2 * sigmaScan ** 2)

    C13p = ((tfM[2, 0] * cosyaw * sinscan +  # dZ/dRange - measurements
             tfM[2, 1] * sinyaw * sinscan +
             tfM[2, 2] * cosscan) *
            (tfM[0, 0] * cosyaw * sinscan +  # dX/dRange - measurements
             tfM[0, 1] * sinyaw * sinscan +
             tfM[0, 2] * cosscan) * sigmaRange ** 2 +
            (- 1 * tfM[2, 0] * rrange * sinyaw * sinscan +  # dZ/dYaw
             tfM[2, 1] * rrange * cosyaw * sinscan) *
            (- 1 * tfM[0, 0] * rrange * sinyaw * sinscan +  # dX/dYaw
             tfM[0, 1] * rrange * cosyaw * sinscan) * sigmaYaw ** 2 +
            (tfM[2, 0] * rrange * cosyaw * cosscan +  # dZ/dScan
             tfM[2, 1] * rrange * sinyaw * cosscan +
             -1 * tfM[2, 2] * rrange * sinscan) *
            (tfM[0, 0] * rrange * cosyaw * cosscan +  # dX/dScan
             tfM[0, 1] * rrange * sinyaw * cosscan +
             -1 * tfM[0, 2] * rrange * sinscan) * sigmaScan ** 2)
    local_Cxx[0, 0] += np.sum(C11p)
    local_Cxx[0, 1] += np.sum(C12p)
    local_Cxx[0, 2] += np.sum(C13p)
    local_Cxx[1, 0] += np.sum(C12p)
    local_Cxx[1, 1] += np.sum(C22p)
    local_Cxx[1, 2] += np.sum(C23p)
    local_Cxx[2, 1] += np.sum(C23p)
    local_Cxx[2, 0] += np.sum(C13p)
    local_Cxx[2, 2] += np.sum(C33p)

    # Get mean without correlation (averages out anyway, or something...)
    for pii in range(nPts):
        Cxx = np.array([[C11p[pii], C12p[pii], C13p[pii]],
                        [C12p[pii], C22p[pii], C23p[pii]],
                        [C13p[pii], C23p[pii], C33p[pii]]])
        if np.linalg.det(Cxx) == 0:
            Cxx = np.eye(3)
        Cix = np.linalg.inv(Cxx)
        ATP[:, pii * 3:(pii + 1) * 3] = Cix
    N = np.dot(ATP, A)
    Qxx = np.linalg.inv(N)  # can only have > 0 in main diagonal!
    pts_m = curr_pts.mean(axis=0)
    l = (curr_pts - pts_m).flatten()
    mean = np.dot(Qxx, np.dot(ATP, l)) + pts_m

    return mean, local_Cxx / nPts


def updatePbar(total, queue, maxProc):
    desc = "Processing core points"
    pCount = 0
    pbar = tqdm(total=total, ncols=100, desc=desc + " (%02d/%02d Process(es))" % (pCount, maxProc))
    while True:
        inc, process = queue.get()
        pbar.update(inc)
        if process != 0:
            pCount += process
            pbar.set_description(desc + " (%02d/%02d Process(es))" % (pCount, maxProc))


def main(p1_file, p2_file, core_point_file, CxxFile, trafFile, outFile):
    p1_pickle_file = p1_file.replace(".laz", "_kd.pickle")
    p2_pickle_file = p2_file.replace(".laz", "_kd.pickle")

    VZ_2000_sigmaRange = 0.005
    VZ_2000_ppm = 0
    VZ_2000_sigmaScan = 0.00027 / 4
    VZ_2000_sigmaYaw = 0.00027 / 4

    SP3 = {'origin': [652992.6490, 5189116.7022, 2526.2710],
           'sigma_range': VZ_2000_sigmaRange,
           'sigma_scan': VZ_2000_sigmaScan,
           'sigma_yaw': VZ_2000_sigmaYaw,
           'ppm': VZ_2000_ppm}
    SP7 = {'origin': [652949.7085, 5189182.2139, 2473.9986],
           'sigma_range': VZ_2000_sigmaRange,
           'sigma_scan': VZ_2000_sigmaScan,
           'sigma_yaw': VZ_2000_sigmaYaw,
           'ppm': VZ_2000_ppm}
    SP4 = {'origin': [652917.8332, 5189284.5585, 2423.7433],
           'sigma_range': VZ_2000_sigmaRange,
           'sigma_scan': VZ_2000_sigmaScan,
           'sigma_yaw': VZ_2000_sigmaYaw,
           'ppm': VZ_2000_ppm}
    SP5 = {'origin': [652804.5869, 5189190.5423, 2456.0348],
           'sigma_range': VZ_2000_sigmaRange,
           'sigma_scan': VZ_2000_sigmaScan,
           'sigma_yaw': VZ_2000_sigmaYaw,
           'ppm': VZ_2000_ppm}
    SP2 = {'origin': [652812.0904, 5189246.1069, 2433.7296],
           'sigma_range': VZ_2000_sigmaRange,
           'sigma_scan': VZ_2000_sigmaScan,
           'sigma_yaw': VZ_2000_sigmaYaw,
           'ppm': VZ_2000_ppm}
    SP1 = {'origin': [652831.6805, 5189073.5765, 2523.7454],
           'sigma_range': VZ_2000_sigmaRange,
           'sigma_scan': VZ_2000_sigmaScan,
           'sigma_yaw': VZ_2000_sigmaYaw,
           'ppm': VZ_2000_ppm}
    SP6 = {'origin': [652862.9167, 5189292.7994, 2403.6955],
           'sigma_range': VZ_2000_sigmaRange,
           'sigma_scan': VZ_2000_sigmaScan,
           'sigma_yaw': VZ_2000_sigmaYaw,
           'ppm': VZ_2000_ppm}

    dom1 = minidom.parse(trafFile)
    params = dom1.getElementsByTagName("Parameter")
    for param in params:
        if param.getAttribute('Name') == 'outTrafPars':
            filecnt = 0
            for val in param.childNodes:
                sval = val.firstChild.nodeValue
                strip_id = strip_name_finder.search(sval).group(1)
                if os.path.split(p1_file)[1] in strip_id or val == param.childNodes[-1]:
                    # found strip - use this parameter set
                    print("found transformation parameters for %s (R|t)" % strip_id)
                    refPointMov = np.array([float(val) for val in rPM_finder.search(sval).group(1).split()])
                    tfM = np.array([float(val) for val in tfM_finder.search(sval).group(1).split()])
                    break
                filecnt += 1
            break

    trafoParams = 12
    Cxx = sio.loadmat(CxxFile)['Cxx'][(filecnt - 1) * trafoParams:filecnt * trafoParams,
          (filecnt - 1) * trafoParams:filecnt * trafoParams]
    Cxx = np.pad(Cxx, ((0, 6), (0, 6)), mode='constant', constant_values=0)
    # Cxx = np.zeros(Cxx.shape) uncomment to ignore coregistration errors
    tfM = tfM.reshape((3, 4))
    SPs2017 = [SP1, SP2, SP3, SP4, SP5, SP6, SP7]
    SPs2018 = [SP1, SP2, SP3, SP4, SP5, SP6, SP7]
    for SP in SPs2017:
        SP['sigma_yaw'] = 0.0003 / 4
        SP['sigma_scan'] = 0.0003 / 4

    M3C2Meta = {'spInfos': [SPs2017, SPs2018],
                'tfM': tfM,
                'Cxx': Cxx,
                'redPoint': refPointMov,
                'searchrad': 0.5,
                'maxdist': 3,
                'minneigh': 5,
                'maxneigh': 100000,
                'leg_ref_err': 0.02}

    NUM_THREADS = 4
    NUM_BLOCKS = 16

    gettrace = getattr(sys, 'gettrace', None)
    if gettrace is None:
        pass
    elif gettrace():
        print('Debugging mode detected, running only single thread mode')
        NUM_THREADS = 1

    LEAF_SIZE = 256

    # load file
    print("Loading point clouds")
    p1_coords, _, p1_positions, _, _ = read_from_las(p1_file)
    p2_coords, _, p2_positions, _, _ = read_from_las(p2_file)

    # build kd tree
    if not os.path.exists(p1_pickle_file):
        print("Building kd-Tree for first PC")
        p1_kdtree = KDTree(p1_coords, leaf_size=LEAF_SIZE)
        picklebig(p1_kdtree, p1_pickle_file)
    else:
        print("Loading pre-built kd-Tree for first PC")
        p1_kdtree = unpicklebig(p1_pickle_file)

    # build kd tree
    # transform p2
    p2_coords = p2_coords - refPointMov
    p2_coords = np.dot(tfM[:3, :3], p2_coords.T).T + tfM[:, 3] + refPointMov
    if not os.path.exists(p2_pickle_file):
        print("Building kd-Tree for second PC")
        p2_kdtree = KDTree(p2_coords, leaf_size=LEAF_SIZE)
        picklebig(p2_kdtree, p2_pickle_file)
    else:
        print("Loading pre-built kd-Tree for second PC")
        p2_kdtree = unpicklebig(p2_pickle_file)

    # load query points
    query_coords, query_norms, _, _, _ = read_from_las(core_point_file)
    idx_randomized = np.arange(query_coords.shape[0])
    np.random.shuffle(idx_randomized)
    query_coords = query_coords[idx_randomized, :]
    query_norms = query_norms[idx_randomized, :]

    if query_norms is None:
        print("Core point point cloud needs normals set. Exiting.")
        exit(-1)
    subsample = False
    if subsample:
        sub_idx = np.random.choice(np.arange(0, query_coords.shape[0]), 2000)
        query_coords = query_coords[sub_idx]
        query_norms = query_norms[sub_idx]
    query_coords_subs = np.array_split(query_coords, NUM_BLOCKS)
    query_norms_subs = np.array_split(query_norms, NUM_BLOCKS)
    print("Total: %d core points" % (query_coords.shape[0]))

    # start mp
    manager = mp.Manager()
    return_dict = manager.dict()

    # prepare shared memory
    p1_coords_shm = mp.shared_memory.SharedMemory(create=True, size=p1_coords.nbytes)
    p1_coords_sha = np.ndarray(p1_coords.shape, dtype=p1_coords.dtype, buffer=p1_coords_shm.buf)
    p1_coords_sha[:] = p1_coords[:]
    p2_coords_shm = mp.shared_memory.SharedMemory(create=True, size=p2_coords.nbytes)
    p2_coords_sha = np.ndarray(p2_coords.shape, dtype=p2_coords.dtype, buffer=p2_coords_shm.buf)
    p2_coords_sha[:] = p2_coords[:]

    max_dist = M3C2Meta['maxdist']
    search_radius = M3C2Meta['searchrad']
    effective_search_radius = math.hypot(max_dist, search_radius)

    print("Querying neighbours")
    pbarQueue = mp.Queue()
    pbarProc = mp.Process(target=updatePbar, args=(query_coords.shape[0], pbarQueue, NUM_THREADS))
    pbarProc.start()
    procs = []

    last_started_idx = -1
    running_ps = []
    while True:
        if len(running_ps) < NUM_THREADS:
            last_started_idx += 1
            if last_started_idx < len(query_coords_subs):
                curr_subs = query_coords_subs[last_started_idx]
                p1_idx = p1_kdtree.query_radius(curr_subs, r=effective_search_radius)
                p2_idx = p2_kdtree.query_radius(curr_subs, r=effective_search_radius)

                p = mp.Process(target=process_corepoint_list, args=(
                    curr_subs, query_norms_subs[last_started_idx],
                    p1_idx, p1_coords_shm.name, p1_coords.shape, p1_positions,
                    p2_idx, p2_coords_shm.name, p2_coords.shape, p2_positions, M3C2Meta, last_started_idx, return_dict,
                    pbarQueue))
                procs.append(p)

                procs[last_started_idx].start()
                running_ps.append(last_started_idx)
            else:
                break
        for running_p in running_ps:
            if not procs[running_p].is_alive():
                running_ps.remove(running_p)
        time.sleep(1)

    for p in procs:
        p.join()
    print("\nAll threads terminated.")
    pbarQueue.put(1)
    pbarProc.terminate()
    p1_coords_shm.close()
    p1_coords_shm.unlink()
    p2_coords_shm.close()
    p2_coords_shm.unlink()

    out_attrs = {key: np.empty(query_coords.shape[0], dtype=val.dtype) for key, val in return_dict[0].items()}
    for key in out_attrs:
        curr_start = 0
        for i in range(NUM_BLOCKS):
            curr_len = return_dict[i][key].shape[0]
            out_attrs[key][curr_start:curr_start + curr_len] = return_dict[i][key]
            curr_start += curr_len
    out_attrs['normals'] = query_norms
    write_to_las(outFile, query_coords, out_attrs)


if __name__ == '__main__':
    import glob

    tbegin = time.time()
    outTemp = r"results\2017_2018a\%s.laz"
    if not os.path.exists(os.path.split(outTemp)[0]):
        os.makedirs(os.path.split(outTemp)[0])

    for tile in tqdm(glob.glob(r'2018a\2018a_65*00_gnd.laz')):
        tile = tile.lower().replace('.laz', '').replace(r'2018a\2018a_', '').replace('_gnd', '')
        print("==========================================")
        print("==   Start tile %20s    ==" % tile)
        print("==========================================")
        tstart = time.time()
        p2_file = r"2018a\2018a_%s_gnd.laz" % tile  # moving
        p1_file = r"2017\2017_%s_gnd.laz" % tile  # fixed
        core_point_file = r"corepoints\2018_cp_%s.las" % tile
        if not all([os.path.exists(p) for p in [p1_file, p2_file, core_point_file]]):
            print("Skipping tile (file missing)")
            print([os.path.exists(p) for p in [p1_file, p2_file, core_point_file]])
            print([p for p in [p1_file, p2_file, core_point_file]])
            continue
        CxxFile = r"ICPOut\ICPout_2018Ato2017\cxx.mat"
        trafFile = r"ICPOut\ICPout_2018Ato2017\icp_output_2018Ato2017.xml"
        outFile = outTemp % tile
        if os.path.exists(outFile):
            print("== > Skipping tile '%s' due to existing output file." % tile)
            continue
        main(p1_file, p2_file, core_point_file, CxxFile, trafFile, outFile)
        tend = time.time()
        print("== > Tile %s finished (Took %.3f s)." % (tile, (tend - tstart)))

    print("==========================================")
    print("==            Merging files             ==")
    print("==========================================")
    outMerge = outTemp % "all"
    outFull = outTemp % "*"
    os.system(r"lasmerge.exe -i %s -o %s" % (outFull, outMerge))
    tstop = time.time()
    print("== > Merge complete. Total processing took %.3f s" % (tstop - tbegin))
