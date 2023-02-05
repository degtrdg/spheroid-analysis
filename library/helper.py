import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

def get_intensity_points(img, composite, outer_bool=False, ellipse_percentage=0.5): 
    clean_composite = _get_clean_composite(composite)
    inner_mask = _get_ellipse_mask(clean_composite, ellipse_percentage=ellipse_percentage)

    #Remove inner from composite to get outer
    not_inner_mask= cv2.bitwise_not(inner_mask)
    outer_only = cv2.bitwise_and(clean_composite.copy(), not_inner_mask)

    # Inner
    rows_ellipse, cols_ellipse  = np.where(inner_mask > 0)
    pts_ellipse = (rows_ellipse, cols_ellipse)

    rows_outer, cols_outer = np.where(outer_only > 0)
    pts_outer = (rows_outer, cols_outer)

    #Get list of pixel intensities that are in the inner rectangle
    def get_pixel_intensities(img, pts):
        pixel_intensities = img[pts[0], pts[1]] 
        return pixel_intensities

    outer = get_pixel_intensities(img, pts_outer)
    inner = get_pixel_intensities(img, pts_ellipse)
    
    if outer_bool:
        pts_outer = np.where(clean_composite > 0)
        outer = get_pixel_intensities(img, pts_outer)
        inner = np.array([0 for _ in range(len(outer))], dtype=np.uint8)

    return (outer, inner)

def get_intensity_pictures(composite,outer_bool=False): 
    clean_composite = _get_clean_composite(composite)
    black = np.full(clean_composite.shape, 0, dtype=np.uint8)

    if outer_bool:
        return (black , clean_composite, clean_composite)

    inner_mask = _get_ellipse_mask(clean_composite)

    #Apply the inner_mask to the image
    inner_masked_image = cv2.bitwise_and(clean_composite.copy(), inner_mask) # Inner

    #Remove inner from composite to get outer
    not_inner_mask= cv2.bitwise_not(inner_mask)
    outer_only = cv2.bitwise_and(clean_composite.copy(), not_inner_mask)

    return (inner_masked_image, outer_only, clean_composite)

def _get_rectangle(clean_composite):
    left = (math.inf,0)
    right = (0,0)
    top = (0, math.inf)
    bottom = (0,0)
    threshold_intensity = 10
    #Find left most edge
    for y, row in enumerate(clean_composite):
        for x, pixel in enumerate(row):
            if pixel >= threshold_intensity and x < left[0]:
                left = (x,y)
            #Find right most edge
            if pixel >= threshold_intensity and x > right[0]:
                right = (x,y)
            #Find top most edge
            if pixel >= threshold_intensity and y < top[1]:
                top = (x,y)
            #Find bottom most edge
            if pixel >= threshold_intensity and y > bottom[1]:
                bottom = (x,y)

    #Given four points find top left and bottom right of the rectangle
    def find_top_left_bottom_right(p1,p2,p3,p4):
        top_left = (min(p1[0],p2[0],p3[0],p4[0]),min(p1[1],p2[1],p3[1],p4[1]))
        bottom_right = (max(p1[0],p2[0],p3[0],p4[0]),max(p1[1],p2[1],p3[1],p4[1]))
        return top_left, bottom_right

    top_left, bottom_right = find_top_left_bottom_right(left,right,top,bottom)
    center = (int((top_left[0] + bottom_right[0])/2), int((top_left[1] + bottom_right[1])/2))
    return (top_left, bottom_right, center)

def _get_clean_composite(composite):
    # Binary threshold
    _ , thresh = cv2.threshold(composite, 5, 10, cv2.THRESH_BINARY)
    contours,_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    #Find biggest contour
    contours = [max(contours, key = cv2.contourArea)]

    black = np.full(thresh.shape, 0, dtype=np.uint8)

    remove_background_noise_mask = cv2.drawContours(black, contours, -1, 255, -1)
    clean_composite = cv2.bitwise_and(composite, remove_background_noise_mask)

    return clean_composite

def _get_ellipse_mask(clean_composite, ellipse_percentage = 0.5):
    #Find ellipse
    (major, minor), center = _get_ellipse(clean_composite, ellipse_percentage)

    #Make a mask of the ellipse that sets everything outside the ellipse to 0
    inner_mask = np.zeros(clean_composite.shape, dtype=np.uint8)
    inner_mask = cv2.ellipse(inner_mask,
                         center,
                         (major, minor)
                         , 0, 0, 360, (255,255 ,255), -1)
    return inner_mask

def _get_ellipse(clean_composite, ellipse_percentage):
    top_left, bottom_right, center = _get_rectangle(clean_composite)

    def x_radius(p1,p2):
        return (p2[0]-p1[0])/2
    def y_radius(p1,p2):
        return (p2[1]-p1[1])/2

    return (int(x_radius(top_left,bottom_right)*ellipse_percentage), int(y_radius(top_left,bottom_right)*ellipse_percentage)), center