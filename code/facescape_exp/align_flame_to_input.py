from fitting.landmarks import load_embedding, landmark_error_3d, mesh_points_by_barycentric_coordinates
import numpy as np
from psbody.mesh import Mesh

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
from fitting.landmarks import load_embedding, landmark_error_3d, mesh_points_by_barycentric_coordinates, load_picked_points
from fitting.util import load_binary_pickle, write_simple_obj, safe_mkdir, get_unit_factor

from smpl_webuser.posemapper import Rodrigues
from scipy.sparse.linalg import cg

from flame_fit import *

import argparse

lmk_emb_path = './models/flame_static_embedding.pkl'
lmk_face_idx, lmk_b_coords = load_embedding(lmk_emb_path)

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

def get_lm_flame7(pred_flame, lmk_face_idx, lmk_b_coords):
    lm_flame = mesh_points_by_barycentric_coordinates( pred_flame.v, pred_flame.f, lmk_face_idx, lmk_b_coords )
    return lm_flame
    
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
         

if __name__ == '__main__':

  pred_flame_path = '/content/drive/MyDrive/3D_Face/flame-fitting (1)/output/facescape/40_44_fit_scan_result.obj'
  pred_flame = Mesh(filename=pred_flame_path)

  lm_flame = get_lm_flame7(pred_flame, lmk_face_idx, lmk_b_coords)
  #np.save('/content/drive/MyDrive/3D_Face/flame-fitting (1)/output/facescape/flame_lmk.npy', lm_flame)

  original_scan_path = '/content/drive/MyDrive/3D_Face/dataset/facescape/4044_scan.obj'
  original_scan = Mesh(filename=original_scan_path)
  lm_original_scan_path = '/content/drive/MyDrive/3D_Face/dataset/facescape/40_44_scan_picked_points.pp'
  lm_original_scan = load_picked_points(lm_original_scan_path)

  new_flame_vertices, masked_gt_scan = compute_rigid_alignment(original_scan, lm_original_scan, pred_flame.v, pred_flame.f, lm_flame)
  write_simple_obj( mesh_v=new_flame_vertices, mesh_f=pred_flame.f, filepath=join('/content/drive/MyDrive/3D_Face/flame-fitting (1)/output/facescape/projected_flame', 'proj_flame.obj') , verbose=False )