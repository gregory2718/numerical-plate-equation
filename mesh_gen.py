"""
Mesh generation code.
Created November 2018
Aaron Gregory
"""

from __future__ import division, print_function

try:
    import pygmsh
except:
    # can happen with a poor internet connection
    print("Warning: pygmsh failed to import")

from numpy import savetxt, loadtxt
import os, shutil
from matplotlib.pylab import figure, gca, show
import dolfin_convert

def create_mesh(name, shape, cl, folder = "./meshes", delete_prev = False, draw = False):
    """
    Generates a 2D mesh with characteristic length cl.
    
    Supported shapes:
        'square'  A square from (0,0) to (1,1).
        'circle'  A cirlce located at (0,0) with radius 1.
        'square_annulus' A square annulus from (0,0) to (1,1) with width 0.25.
        'annulus' An annulus located at (0,0) with inner radius 0.5 and outer radius 1.
    
    The vertex and triangulation data are written to 'vers.txt' and 'tris.txt', under folder/name/
    If the necessary directories do not exist they will be created.
    
    If delete_prev is True, any preexisting files under folder/name/ will be deleted.
    Otherwise nothing will be removed.
    """
    
    # create the geometry
    geom = pygmsh.opencascade.Geometry(cl, cl)
    if shape is "square":
        geom.add_rectangle([ 0, 0, 0 ], 1, 1)
    elif shape is "circle":
        geom.add_disk([ 0, 0, 0 ], 1)
    elif shape is "square_annulus":
        outer = geom.add_rectangle([ 0, 0, 0 ], 1, 1)
        inner = geom.add_rectangle([ 0.25, 0.25, 0 ], 0.5, 0.5)
        geom.boolean_difference([outer], [inner])
    elif shape is "annulus":
        outer = geom.add_disk([ 0, 0, 0 ], 1.0)
        inner = geom.add_disk([ 0, 0, 0 ], 0.5)
        geom.boolean_difference([outer], [inner])
    else:
        raise NameError("Shape '%s' not recognized" % shape)
    
    # make any required directories
    if not os.path.isdir(folder):
        os.mkdir(folder)
    
    if os.path.isdir(folder + "/" + name):
        if delete_prev:
            shutil.rmtree(folder + "/" + name)
        else:
            raise OSError("Directory '%s' already exists" % name)
    
    os.mkdir(folder + "/" + name)
    
    # generate and save the mesh
    points, cells, point_data, cell_data, field_data = pygmsh.generate_mesh(geom, geo_filename = folder + "/" + name + "/mesh.geo")
    points = points[:,:2] # remove the z coordinates
    
    savetxt(folder + "/" + name + "/verts.txt", points)
    savetxt(folder + "/" + name + "/tris.txt", cells['triangle'], fmt="%d")
    
    os.system("gmsh -2 " + folder + "/" + name + "/mesh.geo")
    dolfin_convert.main([folder + "/" + name + "/mesh.msh", folder + "/" + name + "/mesh.xml"])
    
    # plot the mesh if requested
    if draw:
        figure()
        gca().triplot(points[:,0], points[:,1], cells['triangle'])
        show()
    
    return points, cells['triangle']


def load_mesh(name, folder = "./meshes"):
    return loadtxt(folder + "/" + name + "/verts.txt"), loadtxt(folder + "/" + name + "/tris.txt", dtype=int)

def gen_mesh_range(name, sufs_cls = [("", 1)], delete_prev = False, draw = False):
    for s, cl in sufs_cls:
        create_mesh('%s%s' % (name, s), name, cl, delete_prev=delete_prev, draw=draw)



if __name__ == "__main__":
    gen_mesh_range("square", [("_%d" % i, 0.05 + 0.01 * i) for i in range(10)], delete_prev=True, draw=True)


