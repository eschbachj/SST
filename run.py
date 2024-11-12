from glob import glob
from tqdm import tqdm
from czi_to_tiff import load_save_czi
from plotting import *
import pandas as pd
import argparse
from pathlib import Path



#*******************************************************************************************************************#
def main(cellpose_path, save_count, save_bool):
    path_to_curr = Path('./').resolve() #absolute path to current location
    path_to_files = str(path_to_curr.parent)

    #Find czi file in folder, then split it into its FOVs
    folders = glob(path_to_files+"/czi_files/*", recursive = True) #gives all things found in path_to_files (folders and files)
    images = [path for path in folders if path.endswith(".czi")] #gives only czi files
    print(str(len(images))+" image(s) found:")
    print(images)

    #For each czi image in the folder (images)
    for im in tqdm(images):
        tmp_path = im.replace(path_to_files+"/czi_files/", '')
        result_file_path = './Results/'+tmp_path.replace('.czi','')+"/"

        if not os.path.exists('./Results/'):
            os.makedirs('./Results/')

        #The call to start processing of im
        blur, cells, SST_cells, fov_dict = load_save_czi(im, path_to_files+"/czi_files/", result_file_path, cellpose_path, save_bool, save_count) #returns results of processing
        print(str(SST_cells[0])+" SST-positive cells found with high-filtering")
        print(str(SST_cells[1])+" SST-positive cells found with low-filtering")
        print(str(cells)+" total cells found by cellpose")

        print("Saving results...")
        if not os.path.exists(result_file_path):
            os.makedirs(result_file_path)

        df_fov_dict = pd.DataFrame(fov_dict)
        df_fov_dict.to_csv(result_file_path+'fov_dict.csv', index=False)
        
        df_blur = pd.DataFrame(blur)
        df_blur.to_csv(result_file_path+'blur.csv', index=False)


#*******************************************************************************************************************#

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline to find SST.")
    parser.add_argument("-p", "--path", required=True, help="Path to the pretrained cellpose model (/TRAINING-FOLDER/models/MODEL-NAME)")
    parser.add_argument("-c", "--count", type=int, default=10, help="Number of random FOV image results to save (default is 0)")
    parser.add_argument("-s", "--save", type=bool, default=False, help="Save all czi images or not (default is False -- recommended)")
    
    args = parser.parse_args()
    
    main(args.path, args.count, args.save)
