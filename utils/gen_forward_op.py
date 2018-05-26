import numpy as np
import matplotlib.pyplot as plt
from gen_forward_op_parser import gen_forward_op_parser

def check_bounds(pt, pt0, pt1):
    """Checks if the pt is within range of segment (pt0,pt1)"""
    return np.logical_and(
        np.logical_and(pt[:,0]>=min(pt0[0], pt1[0]), pt[:,0]<=max(pt0[0], pt1[0])),
        np.logical_and(pt[:,1]>=min(pt0[1], pt1[1]), pt[:,1]<=max(pt0[1], pt1[1])))

def get_line_params(end_pts):
    """Given a 2(npts) x 2(dim) of array of end_pts return line params
    I will use the cross product trick here
    """
    homogenized_pts = np.append(end_pts, np.ones((2,1)), axis=1)
    line_params = np.cross(homogenized_pts[0], homogenized_pts[1])
    line_params /= line_params[-1]
    
    # cross gives ax+by+c = 0, further code assumes ax+by=c
    # hence, the next line
    line_params[-1] *= -1
    
    return line_params

def get_li(im, end_pts, grid_size):
    """Gets the intersection of the line defined by 
    line parameters with the cartesian grid defined
    using grid size. origin is assumed to be the bottom-
    left of the grid
    
    params:
        im (2d array): takes in gray scale image
        line_params (ndarray): a 2(npts) x 2(dim) of array of end_pts
        
        grid_size (int): a cartesian grid of the given grid_size
        is created with x=i and y=i lines with $i\in[grid_size]$
        
    returns:
        all intersection points with the grid
    """

    line_params = get_line_params(end_pts)
    grid_size = int(grid_size)
    a,b,c = line_params

    # first make the grid
    x = np.arange(grid_size)
    y = np.arange(grid_size)

    # calc interesections
    x_ = np.stack((x, (c - a*x)/b), axis=1)
    y_ = np.stack(((c - b*y)/a, y), axis=1)
    int_pts = np.concatenate((x_,y_), axis=0)

    # clean the pts
    idx_to_keep = check_bounds(int_pts, end_pts[0], end_pts[1])
    new_int_points = int_pts[idx_to_keep]
    new_int_points = np.unique(np.append(new_int_points, end_pts, axis=0), axis=0)
    
    
    # python's pixel coordinate frame
    # python's pixel centers have integer coordinates. (i.e. pixel 10,10) will occupy
    # a Cartesian grid from [9.5,10.5]x[9.5,10.5]. So the grid that we calculated
    # our intersections with needs to be shifted by (0.5, 0.5) to get it in the required
    # frame for calculation of pixels which intersect

    # sort the pts acc to x-coordinate
    ind = np.argsort(new_int_points[:,0])
    sorted_int_pts = new_int_points[ind] + np.array([[0.5,0.5]])

    # calculate line_integral
    rs = []
    cs = []
    n = len(sorted_int_pts) - 1
    line_integral = np.zeros(n)
    # Now, for calculating the pixel location that straddles any two consecutive points
    # in the sorted points array, I use the midpoint. The midpoint of the two points, 
    # will always be inside required pixel. So if I cast it as int, I should have the pixel 
    # coordinate. However, since, the pixel center is at integer coordinates, I add an additional
    # 0.5 to before the cast.
    
    for i in range(n):
        dist = np.linalg.norm(sorted_int_pts[i+1]-sorted_int_pts[i])
        mp = (sorted_int_pts[i+1]+sorted_int_pts[i])/2.0
        r = int(mp[1]+0.5) # python transposes images, hence 1 here and 0 for column
        c = int(mp[0]+0.5)
        rs.append(r)
        cs.append(c)
        line_integral[i] = im[r,c]*dist
    
    return line_integral, sorted_int_pts, (rs,cs)

def test_get_li():
    # ## Testing for `get_li` module
    
    # here, I test a two sensor setup on an image with just ones
    # this test only checks if I have picked up the correct pixels
    # and have calculated the correct intersection points.
    
    # to check, look at the plot and see if the 'x' are on the pixel 
    # edges and all the pixels where the dashed blue line crosses the 
    # image should have some random color in them.
    
    end_pts = np.array([[23.45, 34.56],[100.97, 85.56]])
    im = np.ones((128,128))
    li, pts, x_ids = get_li(im, end_pts, 128)
    
    # %matplotlib qt
    for i in range(len(x_ids[0])):
        im[x_ids[0][i],x_ids[1][i]] = np.random.rand()+1.1
    plt.figure(figsize=(10,10))
    plt.imshow(im)
    plt.plot(end_pts[:,0]+0.5,end_pts[:,1]+0.5,'--')
    plt.scatter(pts[:,0], pts[:,1], marker='x', c='r')


# ## Scale to random sensor grid
def setup_grid(nsensors, grid_size):
    """setup a random grid of sensors on the image"""
    np.random.seed(0)
    c = np.array([grid_size/2.0, grid_size/2.0])
    r = grid_size/2.0
    sensor_locs = np.zeros((nsensors, 2))
    
#     pt = np.zeros(2)
    for i in range(nsensors):
        pt = np.zeros(2)
        while np.linalg.norm(pt-c)>r:
            pt = np.random.uniform(low=0.0, high=1.0, size=(2,))*grid_size
    
        sensor_locs[i]=pt
    return sensor_locs

def plot_sg(sensor_locs):
#    norms = np.linalg.norm(sensor_locs-np.array([[64.0,64.0]]), axis=-1)
#    print(norms)
#    if np.all(norms<=64):
#        print('Grid ok!')

    plt.figure(figsize=(5,5))
    plt.scatter(sensor_locs[:,0], sensor_locs[:,1])
    plt.xlim((0,128))
    plt.ylim((0,128))
    
    plt.show()

from tqdm import tqdm 
def get_forward_op(sensor_locs, grid_size):
    """sets up forward op"""
    nsensors = len(sensor_locs)
    n_measurements = int(nsensors*(nsensors-1)/2)
    grid_size = int(grid_size)
    print("Getting %d measurements from %d sensors!"%(n_measurements, nsensors))
    
    F = np.zeros((n_measurements, grid_size**2))
    
    end_pts = np.zeros((2,2))
    ct = 0
    for i in tqdm(range(nsensors)):
        for j in range(i+1, nsensors):
            end_pts[0] = sensor_locs[i]
            end_pts[1] = sensor_locs[j]
            im = np.ones((grid_size,grid_size))
            li, _, x_ids = get_li(im, end_pts, grid_size) 
            
            for ii in range(len(x_ids[0])):
                r,c = x_ids[0][ii],x_ids[1][ii]
                F[ct,r*grid_size+c] = li[ii]

            ct+=1
    return F

def apply_F(F, Finv, im):
    """Projects `im` in range of F"""
    return (Finv@(F@im.reshape(-1))).reshape(128,128)

def store_mats(F, nsensors):
    """takes F, calculates its pseudoinverse and saves both
    as npy arrays"""
    Finv = np.linalg.pinv(F)
    np.save('../' + str(nsensors) + '_forward.npy', F)
    np.save('../' + str(nsensors) + '_pinverse.npy', Finv)
    print('Operators stored successfully!')
    return

def gen_mask(points, grid_size):
    from matplotlib.path import Path
    from scipy.spatial import ConvexHull
    hull = ConvexHull(points)
    hull_path = Path( points[hull.vertices] )
    
    grid = np.zeros((grid_size, grid_size))
    for x in range(grid_size):
        for y in range(grid_size):
            grid[x,y] = hull_path.contains_point((x,y))
    
    grid = np.rot90(grid)
    grid = grid[::-1,:]
    grid = grid.flatten()
    np.save('../' + str(points.shape[0]) + '_mask.npy', grid)
    return grid
    
def main():
    args = gen_forward_op_parser()
    
    nsensors = args.n
    grid_size = args.g
    sensor_locs = setup_grid(nsensors, grid_size)
#    plot_sg(sensor_locs)
    
    gen_mask(sensor_locs, grid_size)
    
    F = get_forward_op(sensor_locs, grid_size)
    
    store_mats(F, nsensors)
    
#    Finv = np.linalg.pinv(F)
#    im = np.ones((128,128))
#    g = apply_F(F, Finv, im)
#    plt.figure()
#    plt.imshow(g)
#    plt.show()
    return None

###############################################################################
if __name__ == "__main__":
    main()

