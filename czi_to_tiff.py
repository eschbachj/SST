import os, os.path
import imageio as im
from aicsimageio.readers import CziReader
from tqdm import tqdm
import random
import numpy as np
import pandas as pd
import gc

from color_deconv import check_blur, deconv
from run_cellpose import cellpose_func
from image_processing import filter_white_matter, process_SST, process_cellpose, coloc
from plotting import plot_results




def load_czi(file, result_file_path):
    # Get reader
    print("reading CZI file " + file)
    reader = CziReader(file)
    data = reader.data  # returns all image data from the file
    dims = reader.dims  # reads the dimensions found in metadata
    pos = reader.get_mosaic_tile_positions() # top left location of each FOV
    coordinates_dict = {index: coord for index, coord in enumerate(pos)}


    pd.DataFrame(coordinates_dict).to_csv(result_file_path+'coordinates_dict.csv', index=False)
    del coordinates_dict
    del pos
    gc.collect()

    return data, dims

def save_tiles_as_tiff(czi, dims, file):
    print("saving CZI image tiles as tiffs")
    path = str(file.removesuffix('.czi'))
    # Create the directory or perform other operations
    os.makedirs(path,exist_ok=True)
    print(f"Directory '{path}' created.")   
    for m in tqdm(range(dims['M'][0])):
        im_path = path +'/'+ f'{m:04}' + '.tif'
        if not(os.path.exists(im_path)):
            im.imwrite(im_path, czi[0,m,:,:,:])


#******************************************** Main processing Function: *****************************************************#

def load_save_czi(image, path_to_files, result_file_path, cellpose_model, save, count):
    data, dims = load_czi(image, result_file_path) #data has the array for the entire image
    fov = dims['M'][0]

    print(image.replace(path_to_files,'')+" has "+str(fov)+" FOVs")
    if save:
        save_tiles_as_tiff(data, dims, image)

    print("Finding white matter...")
    res = filter_white_matter(data)
    print(str(np.sum(res)) + " white matter FOVs will be skipped")
    
    blur = []
    SST_count = [0,0]
    cell_count = 0

    rand = []
    for x in range(count):
        rand.append(random.randint(0,fov))

    fov_dict = {k: [] for k in range(fov)}

    #keeps track of how many SST positive cells there are for each FOV

    print("Checking for blurred FOVs and running color deconvolution:")
    #loop through every FOV
    for m in tqdm(range(0, fov)):
        if(res[m]):#Then it is grey matter
            if(check_blur(data[0,m,:,:,:])): #each FOV
                sst, cells = deconv(data[0,m,:,:,:]) #returns sst image and cells image
                masks = cellpose_func(cells, cellpose_model, path_to_files.replace('/czi_files/','')) #run cellpose on cells image, returning the masks, flows, and styles
                
                #Process cellpose masks
                cell_sizes, seg_flat = process_cellpose(masks) 
                #cell_sizes = dictionary with cell_id --> size
                #seg_flat = flattened cellpose masks (no id anymore)
                cell_count += len(cell_sizes)
                #cell_count = number of total cells
                
                #Process SST with high and low filtering
                high_filtered_sst = process_SST(sst, True)
                low_filtered_sst = process_SST(sst, False)
                filtered_sst = [high_filtered_sst, low_filtered_sst]

                #Find SST+ cells
                SST_cells = []
                #whole image of sst+ cells
                for cnt, x in enumerate(filtered_sst): #go through high and low filtered (only 2 loops)
                    cell_somatostatin_positive, positive_cells_mask = coloc(masks, x, cell_sizes)
                    #cell_somatostatin_positive --> ids of sst+ cells
                    SST_cells.append(positive_cells_mask)
                    SST_count[cnt] += len(cell_somatostatin_positive)

                    fov_dict[m].append(len(cell_somatostatin_positive))
                
                    if m in rand:
                        print("Saving results for FOV "+str(m)+"...")
                        plot_results(data[0,m,:,:,:], sst, cells, seg_flat, filtered_sst[cnt], SST_cells[cnt], cnt, image.replace(path_to_files,''), m)
                fov_dict[m].append(len(cell_sizes))
            else:
                blur.append(m)
                fov_dict[m] = [0,0,0]
        else:
            fov_dict[m] = [0,0,0]

    #Check how many FOVs were blurry
    prc_blur = len(blur)/fov*100
    print(str(len(blur))+" FOVs were blurry ("+str(prc_blur)+"%% of FOVs are blurry) and will be skipped")

    del data
    gc.collect()

    return blur, cell_count, SST_count, fov_dict
