"""
Processing data from the eigenstates of the plate equation.
Created December 2018
Aaron Gregory
"""

from __future__ import division, print_function

from matplotlib import pylab, pyplot
from matplotlib.pylab import figure, gca, show
import matplotlib.tri as tri

from mesh_gen import load_mesh

import numpy as np
import time
from os import mkdir

from scipy.spatial import KDTree

# The number of seconds between printing progress updates
time_step = 5

#############################################
#                                           #
#    Functions for generating truth data    #
#                                           #
#############################################

def analytic_square(data, num):
    '''
    Loads data with the num first eigenvector/value pairs for the square mesh,
    sampled across data["vers"].
    '''
    N = np.ceil(np.sqrt(num))
    ls = np.zeros((N, N, 3))
    for i in range(N):
        for j in range(N):
            ls[i,j] = [np.pi**2 * ((i+1)**2 + (j+1)**2), i+1, j+1]
    
    ls = ls.reshape((N**2, 3))
    ls = ls[ls[:,0].argsort()][:num]
    
    def f(e):
        return lambda x: np.sin(e[0] * np.pi * x[0]) * np.sin(e[1] * np.pi * x[1])
    
    vs = np.zeros((num, len(data["vers"])))
    for i in range(num):
        vs[i] = np.apply_along_axis(f(ls[i][1:]), 1, data["vers"])
    
    data["eigs"] = vs
    data["ls"] = ls[:, 0]

######################################################
#                                                    #
#    Functions for loading data and preprocessing    #
#                                                    #
######################################################

def load_data(meshname, cl, opts = None, suf = ""):
    '''
    Loads vertices, triangles, eigenvector/value pairs, and prepares a Triangulation.
    
    opts can have the following key-value pairs set:
        ls: whether eigenvalues are loaded
        vs: whether eigenvectors are loaded
        tri: whether a Triangulation object is created (useful for drawing)
        map: whether a vertex mapping onto a higher density mesh is loaded
        quiet: suppress visible output
        l0: scale all eigenvalues so that the first is l0. This is here because
            PyDEC's eigenvalues are dependent on cl, but (mostly) only up to scalar factor.
    '''
    if opts is None:
        opts = dict()
    
    if "ls" not in opts.keys():
        opts["ls"] = True
    
    if "vs" not in opts.keys():
        opts["vs"] = True
    
    if "tri" not in opts.keys():
        opts["tri"] = True
    
    if "map" not in opts.keys():
        opts["map"] = True
    
    if "quiet" not in opts.keys():
        opts["quiet"] = True
    
    if not opts["quiet"]:
        print("Loading mesh...")
    
    vers, tris = load_mesh(meshname)
    triangulation = None
    if opts["tri"]:
        if not opts["quiet"]:
            print("Building triangulation...")
        triangulation = tri.Triangulation(vers[:,0], vers[:,1], triangles=tris)
    
    meshname += suf
    
    ls = None
    if opts["ls"]:
        if not opts["quiet"]:
            print("Loading eigenvalues...")
        ls = np.loadtxt("eigen/" + meshname + "/l_final.txt")
    
    vs = None
    if opts["vs"]:
        if not opts["quiet"]:
            print("Loading eigenvectors...")
        vs = np.loadtxt("eigen/" + meshname + "/v_final.txt")
        if not opts["quiet"]:
            print("Normalizing eigenvectors...")
        for v in vs:
            v /= np.linalg.norm(v)
    
    mapping = None
    if opts["map"]:
        mapping = np.loadtxt("mapping_data/" + meshname + ".txt", dtype = int)
    
    if "l0" not in opts.keys() or opts["l0"] is None:
        l0 = ls[0]
    else:
        l0 = opts["l0"]
    
    if not opts["quiet"]:
        print("Done.")
    
    return {"vers":vers, "tris":tris, "trg":triangulation, "ls":ls * l0 / ls[0], "vs":vs, "cl":cl, "map":mapping}

def load_block(name, sufs_cls = [("_truth", 0.01), ("", 1)], l0 = None, quiet = True, suf = ""):
    '''
    Expects the first (suffix, cl) pair to be for the truth data
    '''
    if not quiet:
        print("Loading '%s'..." % name + sufs_cls[0][0])
    
    data = [load_data(name + sufs_cls[0][0], sufs_cls[0][1], {"l0":l0, "quiet":quiet, "map":False})]
    if l0 is None:
        l0 = np.sqrt(data[0]["ls"][0])
    
    for s, cl in sufs_cls[1:]:
        if not quiet:
            print("Loading '%s'..." % (name + s))
        data += [load_data(name + s, cl, {"l0":l0, "quiet":quiet}, suf)]
    
    return np.array(data)

def index_map(data_from, data_to_tree, quiet = True):
    '''
    Finds a closest-point mapping from data_from["vers"] to data_to["vers"].
    
    If M_high is a high density mesh with a pointwise function F defined on it,
    then F can be sampled on a low density mesh M_low with F[index_map(M_low, M_high)].
    '''
    next_update_time = time.time() + time_step
    N = len(data_from["vers"])
    result = np.zeros(N, dtype=int)
    
    for i in range(N):
        _, result[i] = data_to_tree.query(data_from["vers"][i]) #find_near_index(data_from["vers"][i], data_to["vers"], result[i - 1], data_to["cl"] * 0.66)
        if not quiet and time.time() > next_update_time:
            next_update_time = time.time() + time_step
            print("%5d / %5d (%3d%% )" % (i + 1, N, ((i + 1) * 100) // N))
    
    return result

def gen_maps(name, sufs_cls = [("_truth", 0.01), ("", 1)], quiet = True):
    try:
        mkdir('mapping_data')
    except:
        pass
    
    next_update_time = time.time() + time_step
    if not quiet:
        print("Loading '%s'..." % (name + sufs_cls[0][0]))
    
    vers, tris = load_mesh(name + sufs_cls[0][0])
    del tris
    truth = {"vers":vers, "cl":sufs_cls[0][1]}
    truth_tree = KDTree(truth["vers"])
    
    for s, cl in sufs_cls[1:]:
        if not quiet and time.time() > next_update_time:
            print("Starting '%s'..." % (name + s))
            next_update_time = time.time() + time_step
        
        vers, tris = load_mesh(name + s)
        del tris
        data = {"vers":vers, "cl":cl}
        mapping = index_map(data, truth_tree, quiet)
        np.savetxt("mapping_data/" + name + s + ".txt", mapping, fmt="%d")
    
    if not quiet:
        print("Done")

def calc_spectrum_err(truth, data):
    data["ls_err"] = data["ls"] - truth["ls"][:data["ls"].shape[0]]

def calc_vector_err(truth, data):
    tol_block = 50
    tol_mult = 5
    tol = []
    for i in range(int(np.ceil(truth["ls"].shape[0] / tol_block))):
        R = range(tol_block * i, min(tol_block * (i+1), truth["ls"].shape[0] - 1))
        diffs = np.array([truth["ls"][j+1] - truth["ls"][j] for j in R])
        diffs.sort()
        tol += [diffs[diffs.shape[0] // 3] * tol_mult]
    
    blocks = []
    
    n = 0
    while n < truth["ls"].shape[0]:
        block_mask = np.abs(truth["ls"] - truth["ls"][n]) < tol[n // tol_block]
        blocks += [truth["vs"][block_mask][:, data["map"]]]
        n = np.arange(truth["ls"].shape[0])[block_mask][-1] + 1
    
    data["vs_err"] = np.ones(data["vs"].shape[0]) * -1
    for i in range(data["vs"].shape[0]):
        for b in blocks:
#            if data["vers"].shape[0] > 2000 and i not in b:
#                continue
            
            x = np.linalg.lstsq(b.T, data["vs"][i], rcond=None)[0]
            if data["vs_err"][i] < 0:
                data["vs_err"][i] = np.linalg.norm(b.T.dot(x) - data["vs"][i])
            else:
                data["vs_err"][i] = min(data["vs_err"][i], np.linalg.norm(b.T.dot(x) - data["vs"][i]))

def calc_errs(truth, data):
    for d in data:
        # remove any extra eigenvectors
        if d["ls"].shape[0] > truth["ls"].shape[0]:
            d["ls"] = d["ls"][:truth["ls"].shape[0]]
            d["vs"] = d["vs"][:truth["vs"].shape[0]]
        
        calc_spectrum_err(truth, d)
        calc_vector_err(truth, d)

#######################################
#                                     #
#    Functions for displaying data    #
#                                     #
#######################################

def draw_errs(data):
    if type(data) is dict:
        data = [data]
    
    figure(figsize=(4, 4))
    gca().set_title("eigenvector error: Log (1 - e)")
    for d in data:
        gca().scatter(np.arange(d["vs_err"].shape[0]), np.log(1 - d["vs_err"]))
    show()
    
    figure(figsize=(4, 4))
    gca().set_title("eigenvalue error")
    for d in data:
        gca().scatter(np.arange(d["ls_err"].shape[0]), d["ls_err"])
    show()

def draw_mesh(data):
    figure()
    gca().triplot(data["vers"][:,0], data["vers"][:,1], data["tris"])
    show()

def draw_eig(data, i):
    '''
    If i is iterable, then all eigenvalues j in i will be drawn.
    '''
    if type(i) is not int:
        for j in i:
            draw_eig(data, j)
        return
    
    figure()
    gca().tricontourf(data["trg"], data["vs"][i], 20, cmap='seismic')
    show()

def draw_spectrum(data, rng = None):
    if type(data) is dict:
        data = [data]
    
    figure()
    for d in data:
        if rng is None or len(rng) > len(d["ls"]):
            x = np.arange(len(d["ls"]))
            y = d["ls"]
        else:
            x = rng
            y = d["ls"][rng]
        gca().scatter(x, np.sqrt(y), s=25, edgecolors='none')
    show()

# When in doubt, the horizontal axis is the eigenvalue/vector number
# i.e. the points shown have coordinates (x, f(lambda[x]))
if __name__ == "__main__":
    print("Computing mappings (this can take a while)...")
    gen_maps("square", [("_%d" % i, 0.05 + 0.01 * i) for i in range(10)], quiet = False)
    
    print("Loading data...")
    data = load_block("square", [("_%d" % i, 0.05 + 0.01 * i) for i in range(10)], l0 = np.pi**2 * 2)
    
    print("Some computed eigenvalues:")
    draw_spectrum(data[0:4])
    
    print("Computing errors (this can take a while)...")
    calc_errs(data[0], data[1:])
    
    print("Some errors:")
    draw_errs(data[1:4])
    
    print("Some eigenvectors:")
    draw_eig(data[0], 150)
    draw_eig(data[1], 150)
    draw_eig(data[2], 150)
    draw_eig(data[3], 150)


