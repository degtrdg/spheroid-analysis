import imageio
from scipy.spatial import ConvexHull
from cellpose import plot, utils
import os
from scipy import interpolate
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy
from scipy.stats import chi2
from sklearn.covariance import MinCovDet

# Plotting functions
# Cell level
def get_cell_masks(img, outlines):
    masked_imgs= []
    for o in outlines:
        x = o[:,0]
        y = o[:,1]
        x = np.r_[x, x[0]]
        y = np.r_[y, y[0]]
        tck, u = interpolate.splprep([x, y], s=0, per=True)
        xi, yi = interpolate.splev(np.linspace(0, 1, 1000), tck)
        contour = np.array([[xii, yii] for xii, yii in zip(xi.astype(int), yi.astype(int))])
        mask    = np.zeros_like(img)
        cv2.fillPoly(mask, pts=[contour], color=(255, 255, 255))
        masked_img = cv2.bitwise_and(img, mask)
        masked_imgs.append(masked_img)
    return masked_imgs

def get_cell_means(file):
    # red limit
    dat_red = np.load(f'red/{file}_seg.npy', allow_pickle=True).item()
    img_red = dat_red['img']
    # blue limit
    dat_blue = np.load(f'blue/{file}_seg.npy', allow_pickle=True).item()
    img_blue = dat_blue['img']

    # Use cell_count to pick which channel has the most cells
    cell_counts = cell_count(file)
    if cell_counts[0] > cell_counts[1]:
        outlines = utils.outlines_list(dat_blue['masks'])
    else:
        outlines = utils.outlines_list(dat_red['masks'])

    masked_imgs_blue = get_cell_masks(img_blue,outlines)
    means_blue = np.array([np.mean(get_pixel_intensities(img_blue, np.where(masked_img > 0))) for masked_img in masked_imgs_blue])
    masked_imgs_red = get_cell_masks(img_red,outlines)
    means_red = np.array([np.mean(get_pixel_intensities(img_red, np.where(masked_img > 0))) for masked_img in masked_imgs_red])

    return means_blue, means_red, outlines, dat_blue

def blue_red_cell_scatter(file,data):
    font = {'weight':'bold','size': 22}
    plt.rc('font', **font)
    
    if 'augment' in data and data['augment']:
        means_blue1, means_red1, outlines, dat_blue = get_cell_means(file[0])
        means_blue2, means_red2, outlines, dat_blue = get_cell_means(file[0])
        means_blue = np.concatenate((means_blue1, means_blue2))
        means_red = np.concatenate((means_red1, means_red2))
    else:
        means_blue, means_red, outlines, dat_blue = get_cell_means(file)

    # plot image with outlines overlaid in red (this is for blue segmentation)
    plt.figure(figsize=(12,10))
    plt.imshow(dat_blue['img'])
    for o in outlines:
        plt.plot(o[:,0], o[:,1], color='r')

    if 'remove_outliers' in data and data['remove_outliers']:
        points = np.array([means_red, means_blue]).T
        outliers, _ = mahalanobis_method(points)
        # Remove outliers from points
        points = np.delete(points, outliers, axis=0)
        means_red = points[:,0]
        means_blue = points[:,1]
        # print(f'Number of cells after removing outliers: {len(points)}')

    # scatter plot
    plt.plot(means_red, means_blue, 'o')
    plt.xlabel('Mean mKate2')
    plt.ylabel('Mean CFP')
    plt.gca().set_ylim(-0.5, data['lim'])
    plt.gca().set_xlim(-0.5, data['lim'])
    plt.title(f'Mean CFP vs Mean mKate2 Intensity at frame {file}')

    # Convex Hull around the data points
    points = np.array([means_red, means_blue]).T
    hull = ConvexHull(points)
    plt.plot(np.append(points[hull.vertices,0],points[hull.vertices[0],0]), np.append(points[hull.vertices,1],points[hull.vertices[0],1]), 'r--', lw=2)

    # intrinsic and extrinsic noise
    intrinsic, extrinsic, total = get_noise_not_centered(means_red, means_blue)
    data["intrinsics"].append(intrinsic)
    data["extrinsics"].append(extrinsic)
    data["totals"].append(total)
    data["cv_red"].append(np.std(means_red)/np.mean(means_red))
    data["cv_blue"].append(np.std(means_blue)/np.mean(means_blue))

def plot_cell_mean_hist(file, data):
    font = {'weight' : 'normal','size'   : 15}
    plt.rc('font', **font)
    # red limit
    dat_red = np.load(f'red/{file}_seg.npy', allow_pickle=True).item()
    img_red = dat_red['img']
    # blue limit
    dat_blue = np.load(f'blue/{file}_seg.npy', allow_pickle=True).item()
    img_blue = dat_blue['img']

    outlines = utils.outlines_list(dat_blue['masks'])
    # Use cell_count to pick which channel has the most cells
    cell_counts = cell_count(file)
    if cell_counts[0] > cell_counts[1]:
        outlines = utils.outlines_list(dat_blue['masks'])
    else:
        outlines = utils.outlines_list(dat_red['masks'])

    masked_imgs_blue = get_cell_masks(img_blue,outlines)
    means_blue = np.array([np.mean(get_pixel_intensities(img_blue, np.where(masked_img > 0))) for masked_img in masked_imgs_blue])
    masked_imgs_red = get_cell_masks(img_red,outlines)
    means_red = np.array([np.mean(get_pixel_intensities(img_red, np.where(masked_img > 0))) for masked_img in masked_imgs_red])

    # Exclude outliers in the data
    def reject_outliers(data, m=1):
        return data[abs(data - np.mean(data)) < m * np.std(data)]
    means_blue = reject_outliers(means_blue)
    means_red = reject_outliers(means_red)

    fig, axs = plt.subplots(2, 2,)
    fig.set_size_inches(20, 10)
    axs[0, 0].imshow(img_blue)
    axs[0, 0].set_title(f'CFP channel at frame {file}')
    axs[0, 0].set_yticks([])
    axs[0, 0].set_xticks([])
    for o in outlines:
        axs[0, 0].plot(o[:,0], o[:,1], color='r')

    axs[0, 1].hist(means_blue, bins = np.arange(500, data['max_limit']*.5 + 100, 100))
    axs[0, 1].set(xlabel='Mean intensity', ylabel='Count')
    axs[0, 1].set_ylim(0,6)
    axs[0, 1].set_title('Mean CFP expression')

    axs[1, 0].imshow(img_red)
    axs[1, 0].set_title(f'mKate2 channel at frame {file}')
    axs[1, 0].set_yticks([])
    axs[1, 0].set_xticks([])
    for o in outlines:
        axs[1, 0].plot(o[:,0], o[:,1], color='r')

    axs[1, 1].hist(means_red, np.arange(500, data['max_limit']*.5 + 100, 100))
    axs[1, 1].set_ylim(0,6)
    axs[1, 1].set(xlabel='Mean intensity', ylabel='Count')
    axs[1, 1].set_title('Mean mKate2 expression')

    fig.tight_layout()
    fig.show()

# Cell utils
def get_pixel_intensities(img, pts):
    pixel_intensities = img[pts[0], pts[1]] 
    return pixel_intensities

def _xy_lim(file):
    # red limit
    dat = np.load(f'red/{file}_seg.npy', allow_pickle=True).item()
    img = dat['img']
    outlines = utils.outlines_list(dat['masks'])
    masked_imgs_red = get_cell_masks(img,outlines)
    means_red = np.array([np.mean(get_pixel_intensities(img, np.where(masked_img > 0))) for masked_img in masked_imgs_red])

    # blue limit
    dat = np.load(f'blue/{file}_seg.npy', allow_pickle=True).item()
    img = dat['img']
    masked_imgs_blue = get_cell_masks(img,outlines)
    means_blue = np.array([np.mean(get_pixel_intensities(img, np.where(masked_img > 0))) for masked_img in masked_imgs_blue])

    return max((np.max(means_red),np.max(means_blue)))

def max_limit(data,percent_of_side=0.1):
    max_lim = 0
    for idx in range(len(data)):
        if idx >= len(data)*percent_of_side and idx <= len(data)*(1-percent_of_side):
            max_lim = max(max_lim, _xy_lim(f'{idx}'))
    return max_lim

def cell_count(file):
    dat_blue = np.load(f'blue/{file}_seg.npy', allow_pickle=True).item()
    dat_red = np.load(f'red/{file}_seg.npy', allow_pickle=True).item()
    img_blue = dat_blue['img']
    img_red = dat_red['img']
    outlines = utils.outlines_list(dat_blue['masks'])
    masked_imgs_blue = get_cell_masks(img_blue,outlines)
    outlines = utils.outlines_list(dat_red['masks'])
    masked_imgs_red = get_cell_masks(img_red,outlines)
    return np.array([len(masked_imgs_blue),len(masked_imgs_red)])

def get_noise_elowitz(points_blue, points_red):
    """
    From Elowitz and assumes that noise is on y = x line
    """
    points_blue_mean = np.mean(points_blue)
    points_red_mean = np.mean(points_red)
    intrinsic = (1/len(points_blue))*sum([0.5*(blue - red)**2 for blue, red in zip(points_blue, points_red)])/(points_blue_mean * points_red_mean)
    extrinsic = (1/len(points_blue))*(sum([blue*red for blue, red in zip(points_blue, points_red)]) - points_red_mean*points_blue_mean) /(points_blue_mean * points_red_mean)
    total = (1/len(points_blue))*(sum([0.5*(blue**2 + red**2) for blue, red in zip(points_blue, points_red)])-  points_red_mean*points_blue_mean)/(points_red_mean*points_blue_mean)
    return (intrinsic, extrinsic, total)

def get_noise_not_centered(points_blue, points_red):
    """
    From Bleris and does not assume that noise is on y = x line
    """
    cov = np.cov(points_blue, points_red)
    points_blue_mean = np.mean(points_blue)
    points_red_mean = np.mean(points_red)
    # Extrinsic
    extrinsic = np.sqrt(cov[0,1])/(points_blue_mean * points_red_mean)
    # Intrinsic
    delta = 1
    red_std = np.std(points_red)
    blue_std = np.std(points_blue)
    alpha = blue_std**2 - (red_std**2)*delta + np.sqrt((blue_std**2 - (red_std**2)*delta)**2 + 4*delta*cov)
    beta = 2*np.sqrt(cov)
    rms = np.sqrt((alpha**2*red_std**2-2*alpha*beta*np.sqrt(cov)+beta**2*blue_std**2)/(alpha**2+beta**2))
    total = (rms**2)/(points_blue_mean * points_red_mean)
    return (rms, extrinsic, total)

# Pixel level
def blue_v_red_dist(i,data):
    plt.hist(np.log(data['points_red'][i][0]/data['points_blue'][i][0]), bins=100, color='b', alpha=0.5, label='Outer',density=True)
    plt.axvline((np.log(data['points_red'][i][0]/data['points_blue'][i][0])).mean(), color='b', linestyle='dashed', linewidth=1)
    data['outer_mean'].append((np.log(data['points_red'][i][0]/data['points_blue'][i][0])).mean())
    plt.hist(np.log(data['points_red'][i][1]/data['points_blue'][i][1]), bins=100, color='r', alpha=0.5, label='Inner',density=True)
    plt.axvline((np.log(data['points_red'][i][1]/data['points_blue'][i][1])).mean(), color='r', linestyle='dashed', linewidth=1)
    data['inner_mean'].append((np.log(data['points_red'][i][1]/data['points_blue'][i][1])).mean())
    plt.legend(loc='upper right')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.title(f'Frequency of Pixel Intensities at {i}')

def blue_v_red_dist_median(i,data):
        #Blue vs red dist with images and mean over stack
        plt.hist(np.log(data['points_red'][i][0]/data['points_blue'][i][0]), bins=100, color='b', alpha=0.5, label='Outer',density=True)
        plt.axvline(np.median(np.log(data['points_red'][i][0]/data['points_blue'][i][0])), color='b', linestyle='dashed', linewidth=1)
        data['outer_mean'].append(np.median(np.log(data['points_red'][i][0]/data['points_blue'][i][0])))
        plt.hist(np.log(data['points_red'][i][1]/data['points_blue'][i][1]), bins=100, color='r', alpha=0.5, label='Inner',density=True)
        plt.axvline(np.median(np.log(data['points_red'][i][1]/data['points_blue'][i][1])), color='r', linestyle='dashed', linewidth=1)
        data['inner_mean'].append(np.median(np.log(data['points_red'][i][1]/data['points_blue'][i][1])))
        plt.legend(loc='upper right')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.title(f'Frequency of Pixel Intensities at {i}')

def density_scatter(i,data):
    plt.scatter(data['points_blue'][i][0], data['points_red'][i][0] , c='b', marker='o', label='Outer',alpha=0.01)
    plt.scatter(data['points_blue'][i][1], data['points_red'][i][1], c='r', marker='o', label='Inner',alpha=0.01)
    plt.legend(loc='upper left')
    plt.xlabel('Blue')
    plt.ylabel('Red')
    ax = plt.gca()
    #set log scale
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.title(f'Blue vs Red Intensity at {i}')

# Stats
def mahalanobis_method(data):
    #Minimum covariance determinant
    rng = np.random.RandomState(0)
    real_cov = np.cov(data.T)
    X = rng.multivariate_normal(mean=np.mean(data, axis=0), cov=real_cov, size=506)
    cov = MinCovDet(random_state=0).fit(X)
    mcd = cov.covariance_ #robust covariance metric
    robust_mean = cov.location_  #robust mean
    inv_covmat = scipy.linalg.inv(mcd) #inverse covariance metric
    
    #Robust M-Distance
    x_minus_mu = data - robust_mean
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    md = np.sqrt(mahal.diagonal())
    
    #Flag as outlier
    outlier = []
    C = np.sqrt(chi2.ppf((1-0.05), df=data.shape[1]))#degrees of freedom = number of variables
    for index, value in enumerate(md):
        if value > C:
            outlier.append(index)
        else:
            continue
    return outlier, md

# General
def create_gif(name_of_file, plotting_function, percent_of_side, images, data, no_images=False):
    left = len(images)*percent_of_side 
    right = len(images)*(1-percent_of_side)
    create_gif_left_right(name_of_file, plotting_function, left, right, images, data, no_images=no_images)

def create_gif_left_right(name_of_file, plotting_function, left, right, images, data, no_images=False):
    filenames = []
    for i in range(left, right):
        plotting_function(i, data)
        # create file name and append it to a list
        filename = f'{i}.png'
        filenames.append(filename)
        
        # save frame
        plt.savefig(filename)
        plt.close()
    
    # build gif
    with imageio.get_writer(name_of_file, mode='I') as writer:
        for i in range(left, right):
            imgs = [Image.open(filenames[i])] if no_images else [Image.fromarray(x) for x in images[i]] + [Image.open(filenames[i])]
            _pil_grid(imgs).save('temp.png')
            image = imageio.imread('temp.png')
            os.remove('temp.png')
            writer.append_data(image)
            
    # Remove files
    for filename in set(filenames):
        os.remove(filename)

def create_gif_filenames(name_of_file, plotting_function, filenames, data):
    # For use in aug data with 2 files for each image 
    for i in filenames:
        plotting_function(i, data)
        # create file name and append it to a list
        filename = f'{i[0]}.png'
        # save frame
        plt.savefig(filename)
        plt.close()
    
    # build gif
    with imageio.get_writer(name_of_file, mode='I') as writer:
        for i in filenames:
            imgs = [Image.open(f'{i[0]}.png')]
            _pil_grid(imgs).save('temp.png')
            image = imageio.imread('temp.png')
            os.remove('temp.png')
            writer.append_data(image)
            
    # Remove files
    for filename in set(filenames):
        os.remove(f'{filename[0]}.png')

def _pil_grid(images, max_horiz=np.iinfo(int).max):
    n_images = len(images)
    n_horiz = min(n_images, max_horiz)
    h_sizes, v_sizes = [0] * n_horiz, [0] * (n_images // n_horiz)
    for i, im in enumerate(images):
        h, v = i % n_horiz, i // n_horiz
        h_sizes[h] = max(h_sizes[h], im.size[0])
        v_sizes[v] = max(v_sizes[v], im.size[1])
    h_sizes, v_sizes = np.cumsum([0] + h_sizes), np.cumsum([0] + v_sizes)
    im_grid = Image.new('RGB', (h_sizes[-1], v_sizes[-1]), color='white')
    for i, im in enumerate(images):
        im_grid.paste(im, (h_sizes[i % n_horiz], v_sizes[i // n_horiz]))
    return im_grid
