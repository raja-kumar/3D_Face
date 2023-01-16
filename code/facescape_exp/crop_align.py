
import numpy as np
import chumpy as ch
from os.path import join
import os
import ast

from psbody.mesh import Mesh
from smpl_webuser.serialization import load_model
from sbody.mesh_distance import ScanToMesh
from sbody.robustifiers import GMOf
from sbody.alignment.objectives import sample_from_mesh
from fitting.landmarks import load_embedding, landmark_error_3d, mesh_points_by_barycentric_coordinates
from fitting.util import load_binary_pickle, write_simple_obj, safe_mkdir, get_unit_factor

from smpl_webuser.posemapper import Rodrigues
from scipy.sparse.linalg import cg

from flame_fit import *

import argparse


def rigid_scan_2_mesh_alignment(scan, mesh, visualize=False):
    options = {'sparse_solver': lambda A, x: cg(A, x, maxiter=2000)[0]}
    options['disp'] = 1.0
    options['delta_0'] = 0.1
    options['e_3'] = 1e-4

    s = ch.ones(1)
    r = ch.zeros(3)
    R = Rodrigues(r)
    t = ch.zeros(3)
    trafo_mesh = s*(R.dot(mesh.v.T)).T + t

    sampler = sample_from_mesh(scan, sample_type='vertices')
    s2m = ScanToMesh(scan, trafo_mesh, mesh.f, scan_sampler=sampler, signed=False, normalize=False)

    if visualize:       
        #Visualization code
        mv = MeshViewer()
        mv.set_static_meshes([scan])
        tmp_mesh = Mesh(trafo_mesh.r, mesh.f)
        tmp_mesh.set_vertex_colors('light sky blue')
        mv.set_dynamic_meshes([tmp_mesh])
        def on_show(_):
            tmp_mesh = Mesh(trafo_mesh.r, mesh.f)
            tmp_mesh.set_vertex_colors('light sky blue')
            mv.set_dynamic_meshes([tmp_mesh])
    else:
        def on_show(_):
            pass

    ch.minimize(fun={'dist': s2m, 's_reg': 100*(ch.abs(s)-s)}, x0=[s, r, t], callback=on_show, options=options)
    return s,Rodrigues(r),t


def compute_mask_51(grundtruth_landmark_points):
    #  Take the nose-bottom and go upwards a bit:
    nose_bottom = np.array(grundtruth_landmark_points[16])
    nose_bridge = (np.array(grundtruth_landmark_points[22]) + np.array(grundtruth_landmark_points[25])) / 2  # between the inner eye corners
    face_centre = nose_bottom + 0.3 * (nose_bridge - nose_bottom)
    # Compute the radius for the face mask:
    outer_eye_dist = np.linalg.norm(np.array(grundtruth_landmark_points[19]) - np.array(grundtruth_landmark_points[28]))
    nose_dist = np.linalg.norm(nose_bridge - nose_bottom)
    # mask_radius = 1.2 * (outer_eye_dist + nose_dist) / 2
    mask_radius = 1.4 * (outer_eye_dist + nose_dist) / 2
    return (face_centre, mask_radius)


def compute_mask(grundtruth_landmark_points):
    #  Take the nose-bottom and go upwards a bit:
    nose_bottom = np.array(grundtruth_landmark_points[4])
    nose_bridge = (np.array(grundtruth_landmark_points[1]) + np.array(grundtruth_landmark_points[2])) / 2  # between the inner eye corners
    face_centre = nose_bottom + 0.3 * (nose_bridge - nose_bottom)
    # Compute the radius for the face mask:
    outer_eye_dist = np.linalg.norm(np.array(grundtruth_landmark_points[0]) - np.array(grundtruth_landmark_points[3]))
    nose_dist = np.linalg.norm(nose_bridge - nose_bottom)
    # mask_radius = 1.2 * (outer_eye_dist + nose_dist) / 2
    mask_radius = 1.4 * (outer_eye_dist + nose_dist) / 2
    return (face_centre, mask_radius)


def crop_face_scan(groundtruth_vertices, groundtruth_faces, grundtruth_landmark_points, lm_kind=51):
    # Compute mask
    if lm_kind == 51:
        face_centre, mask_radius = compute_mask_51(grundtruth_landmark_points)
    elif lm_kind == 7:
        face_centre, mask_radius = compute_mask(grundtruth_landmark_points)

    # Compute mask vertex indiced
    dist = np.linalg.norm(groundtruth_vertices-face_centre, axis=1)
    ids = np.where(dist <= mask_radius)[0]

    # Mask scan
    masked_gt_scan = Mesh(v=groundtruth_vertices, f=groundtruth_faces)
    masked_gt_scan.keep_vertices(ids)
    return masked_gt_scan



def procrustes(X, Y, scaling=True, reflection='best'):

    n, m = X.shape
    ny, my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0 ** 2.).sum()
    ssY = (Y0 ** 2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m - my)), 0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection is not 'best':

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:, -1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA ** 2

        # transformed coords
        Z = normX * traceTA * np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY / ssX - 2 * traceTA * normY / normX
        Z = normY * np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my, :]
    c = muX - b * np.dot(muY, T)

    # transformation values
    tform = {'rotation': T, 'scale': b, 'translation': c}

    return d, Z, tform


def compute_rigid_alignment(masked_gt_scan, grundtruth_landmark_points, 
                            predicted_mesh_vertices, predicted_mesh_faces, predicted_mesh_landmark_points):

    grundtruth_landmark_points = np.array(grundtruth_landmark_points)
    predicted_mesh_landmark_points = np.array(predicted_mesh_landmark_points)

    d, Z, tform = procrustes(grundtruth_landmark_points, predicted_mesh_landmark_points, scaling=True, reflection='best')

    # Use tform to transform all vertices in predicted_mesh_vertices to the ground truth reference space:
    predicted_mesh_vertices_aligned = tform['scale']*(tform['rotation'].T.dot(predicted_mesh_vertices.T).T) + tform['translation']

    # Refine rigid alignment
    s , R, t = rigid_scan_2_mesh_alignment(masked_gt_scan, Mesh(predicted_mesh_vertices_aligned, predicted_mesh_faces))
    predicted_mesh_vertices_aligned = s*(R.dot(predicted_mesh_vertices_aligned.T)).T + t
    return (predicted_mesh_vertices_aligned, masked_gt_scan)


def flame_lm(model):
    lmk_emb_path = './models/flame_static_embedding.pkl' 
    lmk_face_idx, lmk_b_coords = load_embedding(lmk_emb_path)
    model_lmks = mesh_points_by_barycentric_coordinates( model.v, model.f, lmk_face_idx, lmk_b_coords )
    return model_lmks


def read_lm(path):
    if path.split('.')[-1] == 'txt':
        file01 = open(path, 'r')
        lms = file01.readlines()
        file01.close()
        lm_pred = np.zeros((7, 3))
        for i in range(7):
            a = float(lms[i].split(' ')[0])
            b = float(lms[i].split(' ')[1])
            c = float(lms[i].split(' ')[2])
            lm_pred[i, :] = [a, b, c]
    elif path.split('.')[-1] == 'pp':
        file01 = open(path, 'r')
        lms = file01.readlines()
        file01.close()
        lm_pred = np.zeros((7, 3))
        for i in range(7):
            line = lms[i+8]
            a = float(line[line.find('x="')+3 : line.find('x="')+10])
            b = float(line[line.find('y="')+3 : line.find('y="')+10])
            c = float(line[line.find('z="')+3 : line.find('z="')+10])
            lm_pred[i, :] = [a, b, c]
    elif path.split('.')[-1] == 'npy':
        lm = np.load(path)
        if lm.shape[0] == 51:
            lm_pred = np.zeros((7, 3))
            lm_pred[0, :] = lm[19, :]
            lm_pred[1, :] = lm[22, :]
            lm_pred[2, :] = lm[25, :]
            lm_pred[3, :] = lm[28, :]
            lm_pred[4, :] = lm[16, :]
            lm_pred[5, :] = lm[31, :]
            lm_pred[6, :] = lm[37, :]
        elif lm.shape[0] == 7:
            lm_pred = lm
        else:
            raise 'only 51 and 7 landmark point is supported in npy file'
    else:
        raise 'unsupported landmark file format.'
    return lm_pred

def cal_error(scan, pred):
    sampler = sample_from_mesh(scan, sample_type='vertices')
    s2m_test = ScanToMesh(scan, pred.v, pred.f, scan_sampler=sampler, signed=False, normalize=False)
    return np.mean(np.array(s2m_test)), np.mean(np.sqrt(np.array(s2m_test))), s2m_test


def pipeline(scan_path, lm_scan_path, pred_path, lm_pred_path, output_folder, crop_para, align_para):

    scan = Mesh(filename=scan_path)
    pred = Mesh(filename=pred_path)
    mean = Mesh(filename='data/mean.obj')

    lm_scan = read_lm(lm_scan_path)
    lm_mean = read_lm('data/mean_landmark3d.npy')

    if align_para:
        lm_pred = read_lm(lm_pred_path)
    else:
        lm_pred = lm_mean

    # # crop
    print('-------------------------------------------')
    if crop_para:
        print('crop GT and pred...')
        scan = crop_face_scan(scan.v, scan.f, lm_scan, lm_kind=lm_scan.shape[0])
        pred = crop_face_scan(pred.v, pred.f, lm_pred, lm_kind=lm_pred.shape[0])
        mean = crop_face_scan(mean.v, mean.f, lm_mean, lm_kind=lm_mean.shape[0])

    # align scan with mean face
    if align_para:
        print('align the GT and pred to mean face...')
        scan.v, mean = compute_rigid_alignment(mean, lm_mean, scan.v, scan.f, lm_scan)
        if not lm_pred_path == 'data/mean_landmark3d.npy':
            pred.v, mean = compute_rigid_alignment(mean, lm_mean, pred.v, pred.f, lm_pred)

    # mse_sm, mae_sm, _ = cal_error(scan, mean)
    # print('GT-mean dist: ', mse_sm, mae_sm)
    # np.save( join( out_folder, 'GT2mean.npy' ), es_sp)

    mse_sp, mae_sp, es_sp = cal_error(scan, pred)
    print('GT-pred dist: ', mse_sp)
    # np.save( join( out_folder, 'GT2pred.npy' ), es_sp)

    # write_simple_obj( mesh_v=scan.v, mesh_f=scan.f, filepath=join(output_folder, 'GT.obj') , verbose=False )
    # write_simple_obj( mesh_v=pred.v, mesh_f=pred.f, filepath=join(output_folder, 'pred.obj') , verbose=False )
    # write_simple_obj( mesh_v=mean.v, mesh_f=mean.f, filepath=join(output_folder, 'mean.obj') , verbose=False )


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='FLAME fitting for a 3D scan')
    parser.add_argument('-i', '--pred_path', default='output/Elias/flame.obj', type=str)
    parser.add_argument('-lm_i', '--lm_i', default='data/mean_landmark3d.npy', type=str)
    parser.add_argument('-gt', '--ground_truth_path', default='outputs/Elias/scan.obj', type=str)
    parser.add_argument('-lm_gt', '--lm_gt', default='data/mean_landmark3d.npy', type=str)
    parser.add_argument('-o', '--output_folder', default='outputs/test/', type=str)
    parser.add_argument('-crop', '--crop_para', default='True')
    parser.add_argument('-align', '--align_para', default='False')
    args = parser.parse_args()

    scan_path      = args.ground_truth_path
    lm_scan_path   = args.lm_gt
    pred_path      = args.pred_path
    lm_pred_path   = args.lm_i
    output_folder  = args.output_folder
    crop_para      = ast.literal_eval(args.crop_para)
    align_para     = ast.literal_eval(args.align_para)

    # if not os.path.exists(output_folder):
    #     os.makedirs(output_folder)

    errors = pipeline(scan_path, lm_scan_path, pred_path, lm_pred_path, output_folder, crop_para, align_para)


