import imageio
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Plotting functions
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

def create_gif(name_of_file, plotting_function, percent_of_side, images, data, no_images=False):
    filenames = []
    for i in range(len(images)):
        if i >= len(images)*percent_of_side and i <= len(images)*(1-percent_of_side):
            plotting_function(i, data)
        # create file name and append it to a list
        filename = f'{i}.png'
        filenames.append(filename)
        
        # save frame
        plt.savefig(filename)
        plt.close()
    
    # build gif
    with imageio.get_writer(name_of_file, mode='I') as writer:
        for i in range(len(filenames)):
            if i >= len(images)*percent_of_side and i <= len(images)*(1-percent_of_side):
                imgs = [Image.open(filenames[i])] if no_images else [Image.fromarray(x) for x in images[i]] + [Image.open(filenames[i])]
                _pil_grid(imgs).save('temp.png')
                image = imageio.imread('temp.png')
                os.remove('temp.png')
                writer.append_data(image)
            
    # Remove files
    for filename in set(filenames):
        os.remove(filename)

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
