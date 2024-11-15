import numpy as np
import cv2
from skimage import measure
from skimage.measure import regionprops
from scipy.ndimage import binary_dilation
from scipy.ndimage import label
from sklearn.cluster import KMeans
from tqdm import tqdm



def find_dominant_colors(image, num_colors=3, dark_percent=0.2):
    # Convert the image to grayscale to identify dark areas
    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Flatten the grayscale image and find the intensity threshold for the darkest 10%
    sorted_intensities = np.sort(grayscale, axis=None)
    lower_threshold = sorted_intensities[int(len(sorted_intensities) * dark_percent)]

    upper_threshold = sorted_intensities[int(len(sorted_intensities) * (1-dark_percent))]
    # Create a mask for the darkest 10% of the image
    dark_mask = (grayscale >= lower_threshold) & (grayscale <= upper_threshold)
    
    # Extract pixels from the original image where the dark_mask is True
    dark_pixels = image[dark_mask]

    # Apply k-means clustering to find dominant colors in the dark areas
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(dark_pixels)

    # Get the cluster centers, which represent the dominant colors
    dominant_colors = kmeans.cluster_centers_.astype(int)
    
    return dominant_colors

def find_top_bottom(fov, top, fov_coords):
    if top:
        range = -1600
    else:
        range = 1600
    differences = np.abs(np.array(fov_coords)[:,0] - (np.array(fov_coords)[fov,0]+range))
    closest_indices = np.argsort(differences)[:1000]
    idx = np.argsort(np.abs(np.array(fov_coords)[closest_indices,1] - (np.array(fov_coords)[fov,1])))[1:10]
    return closest_indices[idx][0]


def filter_white_matter(data, pos):
    res = [None]*np.shape(data)[1] 
    fovs = list(range(0, np.shape(data)[1]))
    num_colors = 3

    for fov in tqdm(fovs):
        colors = find_dominant_colors(data[0, fov,:,:,:], num_colors=num_colors)

        classifications = []
        for color in colors[0:num_colors]:
            if color[0]-color[2]>=25:  # Red channel much greater
                classifications.append(False)#more red
            else:  
                classifications.append(True)#more blue
        
        res[fov] = np.all(classifications)

    #Add back independent "false" FOVs (horizontally) and remove independent "true" FOVs (horizontally)
    for i in range(1, len(res) - 1):  # Avoid first and last elements
        if not res[i]:  # If the current FOV is white matter (False)
            # Check the neighbors (left and right)
            if res[i - 1] and res[i + 1]:  # If both neighbors are True (grey matter)
                res[i] = True
        else: # Current FOV is grey matter (True)
            if not(res[i - 1]) and not(res[i + 1]):  # If both neighbors are False (white matter)
                res[i] = False
    
    #remove incorrect 'falses' by checking FOV above and below
    for i in range(1, len(res) - 1):  # Avoid first and last elements
        if not res[i]:  # If the current FOV is white matter (False)
            # Check the neighbors (top and bottom)
            top = find_top_bottom(i, True, pos)
            bottom = find_top_bottom(i, False, pos)
            if res[top] and res[bottom]:  # If both neighbors are True (grey matter)
                res[i] = True
            if not(res[top]) and not(res[bottom]):  # If both neighbors are False (white matter)
                res[i] = False

    return res



def process_SST(sst, filter_type):
    white_val = 0.9 #0.9 good val
    med_multiplier = 0.9 #0.7 for high filter, 0.9 for low filter
    min_size = 20 #20 good size
    #filter_type==True --> high filter, False --> low filter
    sst = np.dot(sst[:, :, :3], [0.2989, 0.5870, 0.1140]) #greyscale of SST image

    
    if filter_type:
        med_multiplier = 0.7
    
    non_white_pixels = sst[sst < white_val]
    med = np.median(non_white_pixels) 


    if med<0.87:
        #print("Median pixel value too low, no SST in this FOV")
        med = 0.87

    median_threshold = np.zeros_like(sst)#store SST pixels above a set threshold value
    median_threshold[:, :] = np.where(sst > med*med_multiplier, 0, 255)#Black where above mean (lighter), white otherwise (SST)


    labeled_mask, num_labels = measure.label(median_threshold, connectivity=2, return_num=True)#get masks and label them
    regions = measure.regionprops(labeled_mask)#measure how big the masks are

    #Store SST masks above set size threshold
    filtered_sst_mask = np.zeros_like(median_threshold, dtype=bool)

    #Check if SST is big enough to keep:
    for region in regions:
        if region.area >= min_size:
            # Add the region to the filtered mask
            filtered_sst_mask[labeled_mask == region.label] = True

    return filtered_sst_mask


def process_cellpose(masks):
    props = regionprops(masks)

    #Calculate size of each cell
    cell_sizes = {prop.label: prop.area for prop in props}

    #Flatted cellpose masks (for image viewing later)
    seg_flat = np.zeros_like(masks)
    seg_flat[:,:] = np.where(masks > 0, 1, 0)
    

    return cell_sizes, seg_flat


def coloc(masks, filtered_sst_mask, cell_sizes):
    radius_percent = 0.5 #0.5 good val
    # Initialize an array to store the result
    cell_somatostatin_positive = []
    somatostatin_mask = filtered_sst_mask.astype(bool)

    positive_cells_mask = np.zeros_like(masks, dtype=bool)
    dilated_masks = np.zeros_like(masks, dtype=bool)

    for cell_id in np.unique(masks):
        if cell_id == 0:
            continue  # Skip background (0 represents background)

        # Create a binary mask for the current cell
        cell_binary_mask = (masks == cell_id)

        # Expand the cell mask by x pixels
        radius = int(np.sqrt(cell_sizes[int(cell_id)])*radius_percent)
        dilated_cell_mask = binary_dilation(cell_binary_mask, iterations=radius)
        dilated_masks[dilated_cell_mask] = cell_id

        # Check for overlap with somatostatin
        if np.any(dilated_cell_mask & somatostatin_mask):
            positive_cells_mask |= cell_binary_mask  # Add this cell to the positive cells mask
            cell_somatostatin_positive.append(cell_id)
            x,y=np.where(dilated_cell_mask& somatostatin_mask)
            somatostatin_mask[x,y]=0 #remove that SST mask so it can't be used again
    
    return cell_somatostatin_positive, positive_cells_mask
