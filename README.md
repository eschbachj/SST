Files should be organized as follows:

/
├── pipeline/                   --> Run program from this location
│   ├── color_deconv.py
│   ├── czi_to_tiff.py
│   ├── image_processing.py
│   ├── plotting.py
│   ├── run_cellpose.py
│   ├── run_pipeline.py
│   ├── run.py
│   ├── requirements.txt        --> Location of all package versions needed in conda env
│   ├── README.md
│   └── Results/                --> Created by running run.py 
├── czi_files/
│   ├── foo1.czi
│   ├── ...
│   └── fooN.czi
├── training_gpu/               --> Any cellpose training folder
│   └── models/
│       └── cellpose_model      --> Pretrained cellpose model
└── 

Note: If you are copying a pretrained model from another location/device, make sure to check the .cellpose/models/ folder to make sure that 1) The model is copied there as well, and 2) The model is included in the 'gui_models.txt' file. The .cellpose folder is usually located in /Users/{username}/ (press cmd+shift+. to see hidden files)




To run: 

cd pipeline
python run.py -p <path to pretrained cellpose model> -c <number of random FOV results to save (OPTIONAL)> -s <True or False, whether or not to save or not save all FOVs as tiffs (False recommended for time) (OPTIONAL)>


run.py
    Main file

