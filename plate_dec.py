"""
Code to compute the eigenstates of the plate equation using PyDEC.
Created October 2018
Aaron Gregory
"""

from __future__ import division, print_function

from pydec import simplicial_complex
from pydec.fem import whitney_innerproduct

from dolfin import *
from petsc4py import PETSc
from mesh_gen import load_mesh

from matplotlib.pylab import figure, gca, show
import matplotlib.tri as tri
from matplotlib import pylab

import scipy
import scipy.sparse.linalg
import numpy as np
from os import mkdir
import time

time_step = 5

def mkdirs(s):
    try:
        mkdir('eigen')
    except:
        pass
    
    try:
        mkdir('eigen/%s' % (s))
    except:
        pass
    
    try:
        mkdir('eigen/%s/vectors' % (s))
    except:
        pass

def CSR_to_PETSc(mat, quiet = True):
    """
    Converts a sparse scipy matrix into a DOLFIN compatible format, using
    PETSc as a midpoint to avoid the PETScMatrix().set(...) bug.
    
    CSR format: https://www.scipy-lectures.org/advanced/scipy_sparse/csr_matrix.html
    
    Code originally from https://fenicsproject.org/qa/13125/how-to-turn-a-numpy-array-to-petscmatrix/
    """
    M = PETSc.Mat().create()
    M.setSizes(mat.shape)
    M.setType("aij") # another name for csr
    M.setUp()
    
    next_update_time = time.time() + time_step
    for i in range(len(mat.indptr) - 1):
        M.setValues([i], mat.indices[mat.indptr[i]:mat.indptr[i+1]],
                    mat.data[mat.indptr[i]:mat.indptr[i+1]])
        if not quiet and time.time() > next_update_time:
            next_update_time = time.time() + time_step
            print("%5d / %5d (%3d%% )" % (i + 1, len(mat.indptr) - 1, ((i + 1) * 100) // (len(mat.indptr) - 1)))
    
    M.assemble()
    if not quiet:
        print("Done.")
    
    return PETScMatrix(M)

def get_eigs_PyDEC(meshname, use_FEM, quiet = False, draw = False, load = False):
    # Loading the mesh and constructing the simplicial complex
    if not quiet:
        print("")
        print("Loading '%s'..." % meshname)
    
    vers, tris = load_mesh(meshname)
    triangulation = tri.Triangulation(vers[:,0], vers[:,1], triangles=tris)
    sc = simplicial_complex((vers,tris)) 
    N0 = sc[0].num_simplices
    
    if use_FEM:
        meshname += "_FEM"
    
    interpolate = True
    if draw: # Draw the mesh
        figure()
        gca().triplot(vers[:,0], vers[:,1], tris)
        show()
    
    if load:
        if not quiet:
            print("Loading matrix...")
        A = scipy.sparse.load_npz('eigen/%s/A.npz' % meshname)
    else:
        mkdirs(meshname)
        
        # Constructing the matrix equation Ax = b
        # A is of the form
        # [ 0  L ] N0
        # [ L  0 ] N0
        #  N0 N0
        #
        # The Laplacian L is dependent on whether we are
        #   using finite elements (Whitney 1-forms)
        
        if not quiet:
            print("Constructing matrix...")
        if use_FEM:
            L = sc[0].d.T * whitney_innerproduct(sc,1) * sc[0].d
        else:
            L = sc[0].d.T * sc[1].star * sc[0].d
        
        # This matrix takes a long time to construct for large meshes
        A = scipy.sparse.bmat([[None, L], [L, None]], format='csr')
        scipy.sparse.save_npz('eigen/%s/A.npz' % meshname, A)
    
    b = np.zeros(N0 + N0)
    all_Us = np.zeros(N0)
    all_Vs = np.zeros(N0)
    
    # Finding points and indices for the boundary and interior
    # We don't have to convert the simplices to indices with
    # sc[0].simplex_to_index[...], because the nth 0-simplex is named n.
    b_is = np.unique(sc.boundary())
    i_is = np.delete(np.arange(N0), b_is, 0)
    
    # Number of internal and external points
    N0b = len(b_is)
    N0i = N0 - N0b
    
    # Making sure that all the indices are accounted for
    assert(N0i == len(i_is))
    
    if N0i == 0:
        if not quiet:
            print("No internal points in mesh! Finishing...")
        
        np.savetxt("eigen/" + meshname + "/l_final.txt", np.array([]))
        np.savetxt("eigen/" + meshname + "/l_final.txt", np.array([]))
        
        if not quiet:
            print("Done.")
        
        return np.array([]), np.array([])
    
    # Adjusting the system with U and V equal to 0 on the boundary
    # Same as the method used by Hirani in the Darcy flow example from
    # 'PyDEC: Software and Algorithms for Discretization of Exterior Calculus'
    b = b - A * np.concatenate((all_Us, all_Vs))
    keep = np.concatenate((i_is, i_is + N0))
    b = b[keep]
    A = A[keep][:,keep]
    
    if not quiet:
        print("Converting to PETSc...")
    
    A_petsc = CSR_to_PETSc(A, quiet)
    
    if not quiet:
        print("Preparing eigensolver...")
    
    # Create eigensolver
    # If the eigenvectors are taking too long to converge, either
    # increase the tolerance or decrease the max iterations
    eigensolver = SLEPcEigenSolver(A_petsc)
    eigensolver.parameters['spectrum'] = 'target magnitude'
    eigensolver.parameters['spectral_transform'] = 'shift-and-invert' # find the smallest eigenvalues first
    eigensolver.parameters['spectral_shift'] = 1e-6
    eigensolver.parameters['tolerance'] = 1e-15
    eigensolver.parameters['maximum_iterations'] = int(1e4)
#    eigensolver.parameters["problem_type"] = "hermitian"
    
    # Compute all eigenvalues of A x = \lambda x
    if not quiet:
        print("Computing eigenvalues...", end=' ')
    eigensolver.solve(1000) # number of eigenvectors to compute
    num_eigs = eigensolver.get_number_converged()

    if not quiet:
        print("(Found %d)" % num_eigs)
    
    lambdas = np.zeros(num_eigs)
    vs = np.zeros((num_eigs, N0i * 2))
    
    if not quiet:
        print("Computing eigenvectors...")
    
    next_update_time = time.time() + time_step 
    for i in range(num_eigs):
        r, c, rx, cx = eigensolver.get_eigenpair(i)
        lambdas[i] = r
        vs[i] = rx.vec().getArray()
        
        np.savetxt("eigen/" + meshname + "/vectors/l%04d.txt" % i, np.array([lambdas[i]]))
        np.savetxt("eigen/" + meshname + "/vectors/v%04d.txt" % i, vs[i])
        
        if not quiet and time.time() > next_update_time:
            if draw:
                full_eigenvector = np.zeros(N0)
                full_eigenvector[i_is] = vs[i][:N0i]
                figure()
                if interpolate:
                    gca().tricontourf(triangulation, full_eigenvector, cmap='seismic')
                else:
                    gca().scatter(vers[:,0], vers[:,1], c=full_eigenvector, cmap='seismic', linewidth=0.5)
                show()
            next_update_time = time.time() + time_step
            print("%4d / %4d (%3d%% )" % (i + 1, num_eigs, ((i + 1) * 100) // num_eigs))
    
    if not quiet:
        print("Done.")
    
    # We only want the real part
    lambdas = np.array(lambdas)
    vs = np.array(vs)
    
    # Retain only the positive eigenvalues, sorted in ascending order
    keep = lambdas > 0
    vs = vs[keep]
    lambdas = lambdas[keep]
    
    order = lambdas.argsort()
    lambdas = lambdas[order]
    vs = vs[order]
    
    # Reinserting the initial conditions and changing the ordering of axes.
    # Now final_vs[i, :] gives the ith eigenvector, instead of vs[:N0i,i] giving
    # the ith eigenvalue on the internal mesh points.
    final_vs = np.zeros((vs.shape[0], N0))
    final_vs[:,i_is] = vs[:,:N0i]

    np.savetxt("eigen/" + meshname + "/l_final.txt", lambdas)
    np.savetxt("eigen/" + meshname + "/v_final.txt", final_vs)
        
    if draw and not quiet: # Draw some eigenstates
        for i in range(min(5, len(final_vs))):
            figure()
            gca().set_title("Eigenvector for lambda[%d] = %f" % (i, lambdas[i]))
            if interpolate:
                gca().tricontourf(triangulation, final_vs[i], cmap='seismic')
            else:
                gca().scatter(vers[:,0], vers[:,1], c=final_vs[i], cmap='seismic', linewidth=0.5)
            show()
            
    return lambdas, final_vs

def gen_eigen_range_PyDEC(name, sufs = [""]):
    for s in sufs:
        get_eigs_PyDEC("%s%s" % (name, s), use_FEM=True , quiet=False, draw=False, load=False)
        get_eigs_PyDEC("%s%s" % (name, s), use_FEM=False, quiet=False, draw=False, load=False)


if __name__ == "__main__":
    gen_eigen_range_PyDEC("square", ["_%d" % i for i in range(10)])


