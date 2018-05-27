import os
import pickle
import numpy as np
from time import time
import multiprocessing as mp
from scipy.spatial import Delaunay
from imageio import imwrite, imread
from scipy.misc import face, imresize
from skimage.transform import radon, iradon

ADD_NOISE = True
noise_level = 1e-1


def get_mesh_pts(N, img_size):
    """
    Algorithm is as follows:
    First I will place approx 4*sqrt(N) equispaced points around the boundary
    of the square and then distributed the remaining N-4*sqrt(N) points uniformly
    at random inside the square. Then, I call the Delaunay algorithm to make the
    triangulations. 

    Delaunay triangulations were chosen as they are shown to have
    piecewise linear functions on these triangulations achieve get the lowest sup-norm
    amongst all triangulations [1].

    Ref.

    [1] S M Omohundro. The Delaunay triangulation and function learning, 1989.
    """
    # specifying points on boundary

    n_edge_pts = int(np.sqrt(N))-1
    bdry_points = np.linspace(0, img_size, n_edge_pts+1)
    points = np.zeros((N, 2))
    t = np.zeros(n_edge_pts)
    points[:n_edge_pts, :] = np.stack((bdry_points[:-1], t), axis=1)
    points[n_edge_pts:2*n_edge_pts,
           :] = np.stack((t+img_size, bdry_points[:-1]), axis=1)
    points[2*n_edge_pts:3*n_edge_pts,
           :] = np.stack((bdry_points[1:], t+img_size), axis=1)
    points[3*n_edge_pts:4*n_edge_pts,
           :] = np.stack((t, bdry_points[1:]), axis=1)
    # addition of slack makes points that are not very close to boundary
    slack = 5
    points[4*n_edge_pts:] = slack + \
        np.random.rand(N-4*n_edge_pts, 2)*(img_size-slack)
    return points


def make_triangulations(pts):
    """Wrapper for Delaunay triangulations"""
    tri = Delaunay(pts)
    return tri


def make_mesh(N, img_size):
    """Wrapper to generate random subspace, given number of
    vertices (N) and the square domain size (img_size)
    """
    pts = get_mesh_pts(N, img_size)
    tri = make_triangulations(pts)
    return tri, pts


def area(p1, p2, p3):
    """Gets area of triangle. Refer:
    http://www.crazyforcode.com/check-point-lies-triangle/
    """
    return np.abs(p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1]))/2.0


def is_pt_in_triangle(pt, tri):
    """Gets the coordinates of the triangle and pt and checks if
    the point is inside or outside the triangle

    The point is expected to be a pixel center
    """
    p0, p1, p2 = tri
    At = area(p0, p1, p2)
    for simplex in pt:
        A1 = area(pt, p0, p1)
        A2 = area(pt, p1, p2)
        A3 = area(pt, p2, p0)

    if np.abs(At-(A1+A2+A3)) < 1e-6:
        return True
    else:
        return False


def build_hashmap(tri, img_size):
    """Creates mappings from each pixel to a triangle (pix2tri)
    and from each triangle to a list of pixels (tri2pix)

    Takes in the Delaunay trianguled object (tri) and 
    the canvas size (img_size)
    """
    # sanity check here
    assert tri.points.max() == img_size, 'Triangulation and img_size don\'t match'
    from collections import defaultdict

    # using defaultdict to avoid try, except blocks
    pix2tri = defaultdict(tuple)
    tri2pix = defaultdict(list)
    points = tri.points

    for r in range(img_size):
        for c in range(img_size):
            found = False
            pt = np.array((r+0.5, c+0.5))

            # search for pixel in all triangles
            for k in range(len(tri.simplices)):
                tri_ind = tri.simplices[k]

                tri_pts = np.array([points[k] for k in tri_ind])
                if is_pt_in_triangle(pt, tri_pts):
                    pix2tri[(r, c)] = k
                    tri2pix[k].append((r, c))
                    found = True
                    break

            if not found:
                raise ValueError(
                    'Could not find match for pixel %s !' % str((r, c)))

    return pix2tri, tri2pix


def pickle_dump(fname, arr, protocol=2):
    """Saves arr as a pkl file, 
    protocol specified for backwards compatibility
    """
    if not fname.endswith(".pkl"):
        fname = fname + '.pkl'

    with open(fname, 'wb') as f:
        pickle.dump(arr, f, protocol=2)

    return None


def pickle_load(fname):
    """Loads fname pkl file, 
    and returns the array
    """
    with open(fname, 'rb') as f:
        return pickle.load(f)


def make_projector(tri2pix, n, img_size):
    """Takes in the hashmap of triangle id to pixel coordinates
    to return the low-dim subspace projection matrix
    :param tri2pix: map from triangle_id to pixels
    :param n: the number of triangles
    :param img_size: img_size
    """
    projector = np.zeros((n, img_size**2))

    for triangle, pixlist in tri2pix.copy().items():
        for pix in pixlist:
            projector[triangle][pix[0]*img_size + pix[1]] = 1.0
    sum_along_axis = projector.sum(axis=1, keepdims=True)
    # if a triangle did not get any pixel
    sum_along_axis[sum_along_axis == 0] = 1
    projector /= sum_along_axis
    return projector


########################################################################
# Generate meshes using multiprocessing

def main_producer(i, Npoints, img_size, dirgroup_prefix):
    # seed
    np.random.seed()

    # make a mesh
    tri, pts = make_mesh(Npoints, img_size)
    Ntriang = len(tri.simplices)

    # dirname = dirgroup_prefix + '_%dtri_%d' % (Ntriang, i)
    dirname = dirgroup_prefix + '_%d'%(Ntriang)
    if not os.path.exists(dirname):
        try:
            os.makedirs(dirname)
            os.makedirs(dirname+'/extras')
        except FileExistsError:
            # takes care of rare race conditions
            pass

    # build its hashmap
    pix2tri, tri2pix = build_hashmap(tri, img_size)

    # store the hashmap and mesh in the extras/ maybe used later for graphics
    pickle_dump(dirname + '/extras/pix2tri%d'%i, pix2tri)
    pickle_dump(dirname + '/extras/tri2pix%d'%i, tri2pix)
    pickle_dump(dirname + '/extras/tri%d'%i, tri)

    # make the projector
    projector = make_projector(tri2pix, Ntriang, img_size)

    # store the projector
    pickle_dump(dirname + '/P%d'%i, projector)
    pickle_dump(dirname + '/Pinv%d'%i, np.linalg.pinv(projector))

    return None


def main(Nmeshes, Npoints, img_size, dirgroup_prefix='mesh'):

    # init multipocessing
    nprocs = min(4, Nmeshes)
    pool = mp.Pool(processes=nprocs)

    # generate Nmeshes each with same number of triangles
    results = [pool.apply_async(main_producer,
                                args=(i, Npoints,
                                      img_size, dirgroup_prefix)) for i in range(Nmeshes)]

    # wait for results
    for p in results:
        p.get()

    return None

if __name__ == "__main__":
    # 36 points generates a 50 triangle mesh
    main(10, 36, 128, 'mesh_trial')