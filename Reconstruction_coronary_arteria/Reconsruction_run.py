import numpy as np
import json
from Transformation import TransformationMatrix
from Reconstruction import Reconstruction_methods

def Reconstruction_run(curves_pr1,curves_pr2,radius_pr1,radius_pr2,angles1,angles2):
    curve_pr1 = TransformationMatrix.tr_points(curves_pr1,0.22)
    curve_pr2 = TransformationMatrix.tr_points(curves_pr2,0.22)
    recon = Reconstruction_methods(curve_pr1,curve_pr2,radius_pr1,radius_pr2, angles1, angles2)
    recon.calibrate1()
    recon.partial_matching()
    recon.reconstruct_3d_curve()
    recon.calc_norm_vectors_2d()
    recon.calc_tangent_vectors_3d()
    recon.reconstruct_3d_rad()
    radius = [[value] for value in recon.radius_3d]
    return np.hstack((recon.curve_3d,radius))

