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
    points[4*n_edge_pts:] = slack+np.random.rand(N-4*n_edge_pts, 2)*(img_size-slack)
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
            pt = np.array((r+0.5, c+0.5))  # get center of pixel

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


def generate_random_function(tri, tri2pix, img_size):
    img = np.zeros((img_size, img_size))
    for key in tri2pix:
        g = np.random.rand()
        for idx in tri2pix[key]:
            img[idx] = g
    return img


# In[19]:


def pickle_dump(fname, arr, protocol=2):
    if not fname.endswith(".pkl"):
        fname = fname + '.pkl'

    with open(fname, 'wb') as f:
        pickle.dump(arr, f, protocol=2)

    return None


def pickle_load(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


# ## make projector matrix

# In[24]:


def make_projector(tri2pix, n, img_size):
    """Takes in the hashmap of triangle id to pixel coordinates
    to return the low-dim subspace projection matrix
    :param tri2pix: triangle_id to pixel map
    :param n: the number of triangles
    :param img_size: img_size
    """
    projector = np.zeros((n, img_size**2))

    for triangle, pixlist in tri2pix.copy().items():
        for pix in pixlist:
            projector[triangle][pix[0]*img_size + pix[1]] = 1.0
    sum_along_axis = projector.sum(axis=1, keepdims=True)
    sum_along_axis[sum_along_axis == 0] = 1
    projector /= sum_along_axis
    return projector


# ## Generate images


# In[30]:


def img_gen(n, img_size, group_prefix):

    out = np.zeros((n, img_size, img_size))
    fbp = np.zeros((n, img_size, img_size))

    tri, pts = make_mesh(34, img_size) 
    pix2tri, tri2pix = build_hashmap(tri, img_size)

    theta = np.linspace(0, 180, 5, endpoint=False)

    for i in range(n):
        out[i] = generate_random_function(tri, tri2pix, img_size)
        # fbp[i] = iradon(radon(out[i], theta=theta, circle=False),
        #                 theta=theta, circle=False)
        # fbp[i] -= fbp[i].min()
        # fbp[i] /= fbp[i].max()

        # imwrite(group_prefix+'/image_%d.png'%(i), img)
    np.save(group_prefix+'/mesh_imgs', out)
    # np.save(group_prefix+'/fbp', fbp)


# In[31]:

def main_producer(i, Npoints, Nimages_per_mesh, img_size, dirgroup_prefix):

    # make a mesh
    tri, pts = make_mesh(Npoints, img_size)
    Ntriang = len(tri.simplices)

    # dirname
    dirname = dirgroup_prefix + '_%dtri_%d' % (Ntriang, i)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    # build its hashmap
    pix2tri, tri2pix = build_hashmap(tri, img_size)

    # store the hashmap and mesh in the directory
    pickle_dump(dirname + '/pix2tri', pix2tri)
    pickle_dump(dirname + '/tri2pix', tri2pix)
    pickle_dump(dirname + '/mesh', tri)

    # make the projector
    projector = make_projector(tri2pix, Ntriang, img_size)

    # store the projector
    pickle_dump(dirname + '/P', projector)
    pickle_dump(dirname + '/Pinv', np.linalg.pinv(projector))

    # generate Nimages for this mesh
    img_gen(Nimages_per_mesh, tri, tri2pix, img_size, dirname)

    return None


def main(Nmeshes, Npoints, Nimages_per_mesh, img_size, dirgroup_prefix='mesh'):
    nprocs = min(12, Nmeshes)
    pool = mp.Pool(processes=nprocs)

    results = [pool.apply_async(main_producer,
                                args=(i, Npoints,
                                      Nimages_per_mesh,
                                      img_size, dirgroup_prefix)) for i in range(Nmeshes)]

    for p in results:
        p.get()

    return None

# ## Multiprocessing module


def gen_mesh_and_apply_proj(img, Npoints):
    # np.random.seed(int(time()))
    np.random.seed()
    img_size = img.shape[0]
    # print((img_size, Npoints))

    tri, pts = make_mesh(Npoints, img_size)

    # number of triangles
    Ntriang = len(tri.simplices)

    # build its hashmap
    pix2tri, tri2pix = build_hashmap(tri, img_size)

    # make the projector
    projector = make_projector(tri2pix, Ntriang, img_size)
    P_img = projector.dot(img.flatten())

    if ADD_NOISE:
        noise = np.random.rand(len(P_img))
        noise[noise > noise_level] = 0
        P_img += noise

    projected_img = np.linalg.pinv(projector).dot(P_img)
    projected_img = projected_img.reshape(img_size, img_size)

    return projected_img, projector, P_img


def mp_class(nprocs, nmeshes, img, npoints, dirname):
    # output = mp.Queue()

    pool = mp.Pool(processes=nprocs)

    results = [pool.apply_async(gen_mesh_and_apply_proj, args=(
        img.copy(), npoints)) for _ in range(nmeshes)]

    # output = [p.get() for p in results]
    output_img = []
    projectors = []
    coefficients = []

    for i, p in enumerate(results):
        img, proj, coeff = p.get()

        projectors.append(proj)
        coefficients.append(coeff)

        imwrite(dirname+'/%d.png' % i, img)
        output_img.append(img)

    return output_img, projectors, coefficients


def make_dir(dirname):
    """if dirname is not present, make a directory dirname"""
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    return None


if __name__ == "__main__":
    main(200, 20, 100, 128, 'mesh_gan_meshes')

    # img = face().mean(axis=-1)
    # img_size = 128
    # img = imresize(img, (img_size,img_size))/255.0

    # dirname = "./face_100tri_128x128_50meshes_1e-1noise"
    # make_dir(dirname)

    # imwrite('face.png', img)
    # output, projectors, coefficients = mp_class(12, 50, img, 65, dirname)

    # output = np.array(output)
    # projectors = np.array(projectors)
    # coefficients = np.array(coefficients)
    # imwrite(dirname+'/result_avg.png', output.mean(axis=0))

    # coefficients = coefficients.reshape(-1)
    # projectors = projectors.reshape(-1,img_size*img_size)

    # proj_Tproj = projectors.T.dot(projectors) + noise_level*np.eye(img_size*img_size)
    # final_img = np.linalg.pinv(proj_Tproj).dot(projectors.T.dot(coefficients))

    # imwrite(dirname+'/result_eqn.png', final_img.reshape(img_size,img_size))
