'''
Reads an xml representation of 
code inspired from https://fenicsproject.org/qa/9414/fenics-mesh-generation-mark-inner-region/ 

Authors: Alexander Jansing, Aaron Gregory
Course: MAT 560 - Numerical Differential Equations
Date: April 29th, 2018
Project: Finding Eigenvalues and Eigenvectors of Plates
'''

from __future__ import division, print_function

from dolfin import *
import pylab
from os import mkdir
import numpy as np
import time
from matplotlib.pylab import figure, gca, show
import matplotlib.tri as tri
from mesh_gen import load_mesh

time_step = 5

# Telling FEniCS not to rearrange the mesh ordering
parameters['reorder_dofs_serial'] = False

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

def get_eigs_FEniCS(meshname, quiet = True, draw = False):
    if not quiet:
        print("")
        print("Loading '%s'..." % meshname)
    
    vers, tris = load_mesh(meshname)
    triangulation = tri.Triangulation(vers[:,0], vers[:,1], triangles=tris)
    interpolate = True
    if draw: # Draw the mesh
        figure()
        gca().triplot(vers[:,0], vers[:,1], tris)
        show()
    
    mesh = Mesh('meshes/%s/mesh.xml' % meshname)
    meshname += "_FEniCS"
    mkdirs(meshname)
    
    if not quiet:
        print("Generating function space...")
    
    U = FiniteElement('CG', triangle, 1)
    V = FiniteElement('CG', triangle, 1)
    W = FunctionSpace(mesh, V * U)

    class DirichletBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary

    # Define boundary condition
    u0 = Constant(0.0)
    boundaryCondition1 = DirichletBC(W.sub(0), u0, DirichletBoundary())
    boundaryCondition2 = DirichletBC(W.sub(1), u0, DirichletBoundary())

    # Define the bilinear form
    (u, v) = TrialFunction(W)
    (f1, f2) = TestFunction(W)
    a = -(dot(grad(u), grad(f2)) + dot(grad(v), grad(f1)))*dx
    L = (u*f1 + v*f2)*dx

    if not quiet:
        print("Assembling matrix...")
    
    # Create the matrices
    A = PETScMatrix()
    b = PETScMatrix()
    assemble(a, tensor = A)
    assemble(L, tensor = b)
    boundaryCondition1.apply(A)
    boundaryCondition2.apply(A)

    if not quiet:
        print("Preparing eigensolver...")
    
    # Create eigensolver
    eigensolver = SLEPcEigenSolver(A, b)
    eigensolver.parameters['spectrum'] = 'target magnitude'
    eigensolver.parameters['spectral_transform'] = 'shift-and-invert'
    eigensolver.parameters['spectral_shift'] = 1e-6
    eigensolver.parameters['tolerance'] = 1e-15
    eigensolver.parameters['maximum_iterations'] = int(1e4)

    # Compute all eigenvalues of A x = \lambda x
    if not quiet:
        print("Computing eigenvalues...", end=' ')
    
    eigensolver.solve(1000)
    num_eigs = eigensolver.get_number_converged()

    if not quiet:
        print("(Found %d)" % num_eigs)
    
    lambdas = np.zeros(num_eigs)
    vs = np.zeros((num_eigs, len(vers) * 2))
    
    if not quiet:
        print("Computing eigenvectors...")

    next_update_time = time.time() + time_step 
#    u = Function(W)
    for i in range(num_eigs):
        r, c, rx, cx = eigensolver.get_eigenpair(i)
        lambdas[i] = r
        vs[i] = rx.vec().getArray()
        np.savetxt("eigen/" + meshname + "/vectors/l%04d.txt" % i, np.array([lambdas[i]]))
        np.savetxt("eigen/" + meshname + "/vectors/v%04d.txt" % i, vs[i])
        
        if not quiet and time.time() > next_update_time:
            if draw:
#                u.vector()[:] = rx
#                plot(u.sub(0), cmap='coolwarm')
                figure()
                ax = gca()
                if interpolate:
                    ax.tricontourf(triangulation, vs[i], cmap='coolwarm')
                else:
                    ax.scatter(vers[:,0], vers[:,1], c=vs[i], cmap='coolwarm', linewidth=0.5)
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
    
    np.savetxt("eigen/" + meshname + "/l_final.txt", lambdas)
    np.savetxt("eigen/" + meshname + "/v_final.txt", vs)
        
    if draw and not quiet: # Draw some eigenstates
        for i in range(min(5, len(vs))):
#            # code for saving images. Not used currently.
#            u = Function(W)
#            u.vector()[:] = vs[i]
#            plot(u.sub(0), cmap='coolwarm')
#            pylab.savefig('eigen/%s/images/%04d.png' % (baseName, i), bbox_inches='tight')
            
            figure()
            ax = gca()
            gca().set_title("Eigenvector for lambda[%d] = %f" % (i, lambdas[i]))
            if interpolate:
                gca().tricontourf(triangulation, vs[i][:len(vers)], cmap='coolwarm')
            else:
                gca().scatter(vers[:,0], vers[:,1], c=vs[i][:len(vers)], cmap='coolwarm', linewidth=0.5)
            show()
    
    return lambdas, vs

if __name__ == "__main__":
    # Test for PETSc and SLEPc
    if not has_linear_algebra_backend("PETSc"):
        print("Error: DOLFIN has not been configured with PETSc. Exiting.")
        exit()

    if not has_slepc():
        print("Error: DOLFIN has not been configured with SLEPc. Exiting.")
        exit()

def gen_eigen_range_FEniCS(name, sufs = [""]):
    for s in sufs:
        get_eigs_FEniCS("%s%s" % (name, s), quiet=False, draw=False)



if __name__ == "__main__":
    gen_eigen_range_FEniCS("square", ["_%d" % i for i in range(10)])



