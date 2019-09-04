TrichoTracking
==============

**TrichoTracking** is a python package that processes image sequences of gliding filaments. 
It finds and quantifies the movement of filament pairs. 
Implementational details are found [here].


Installation
------------

### a) Install OpenCV
Follow the steps from the 
[OpenCV installation guide](https://docs.opencv.org/4.1.0/da/df6/tutorial_py_table_of_contents_setup.html).

### b) Install Trichotracking
Download or clone *Trichotracking*:

    $ git clone https://github.com/giuliaschneider/Trichotracking.git
    cd Trichotracking
    python3 setup.py install
    



Usage
-----

### a) Preprocess data
The bash or python scripts in the folder [bin/preprocess](bin/preprocess) preprocess the experimental image sequence.

  1) **Rename and move images**
  
     If source images are saved in different folders (where they are e.g. numbered from 0 to 999) the images can be renamed and 
     moved to the *data* folder with [rename_files.sh](bin/preprocess/rename_files.sh):
  
         $ cd bin/preprocess
         $ ./rename_files.sh <path-to-parent-folder> [data-folders]
      
  2) **Sort images from different camera positions**
  
     If the image data folder contains images from different camera positions, they can be moved to separate directories
     with [move_files.sh](bin/preprocess/move_files.sh):
         
         $ ./move_files.sh <path-to-parent-folder> [folders-to-be-created]
         
     Due to a bug in our camera trigger, our camera sometimes takes two pictures of the same position. 
     The script discards therefore the first image if a second image is taken within a time span of 1 second.
      
  3) **Generate movie of image sequence**: [movie.py](bin/preprocess/movie.py)
  
         $ python3 movie.py <path-to-parent-folder> [data-folders] 
         
  4) **Background**
  
     [background.py](bin/preprocess/background.py) calculates the background and chamber image 
     and segments the first image by background subtraction with the given threshold:
     
         $ python3 background.py <path-to-source-folder> <threshold> [data-folders]


### b) Process image sequence
The image sequence is processed in three steps:
  1) Segmentation by background subtraction and thresholding
  2) Linking of particles by nearest neighbor matching
  3) Merging and splitting of particles

Process the images by calling [run.py](bin/run.py):

    $ run.py --src <path-to-image-folder> --px <px> --expId <expId> [OPTIONS]

    Options:
      --src         Source directory of image sequence
      --px          Px length in [µm/px]
      --expId       Experiment identifier to uniquely label tracks
      --dest        Result directory {src/result}
      --plot        Flag indicating if intermediate results are shown {False}
      --dark        Flag indicating if images are darkfield {False}
      --blur        Flag indicating if images should be blurred {True}
      --dLink       Maximal linking distance in px {10}
      --dMerge      Maximal merging distance in px {10}
      --dMergeBox   Maximal merging distance of minimal boxes in px {10}
      --kChamber    Kernel size to erode chamber {400}
      --dt          Image sequence capture time
      
The script outputs different files (per default in a *result* folder in the image directory):
- **tracks.csv**: data of each particle at each time step
- **pixellist.pkl**: list of contours of each particle at each time step
- **tracks_meta.csv**: meta data of each track
- **aggregates_meta.csv**: meta data of each aggregate track
- **pairs_meta.csv**: meta data of each aligned filament pair track
- **pair_tracks.csv**: segmentation data of each aligned filament pair track
- **times.csv**: list of image capture time in seconds since epoch
- **overlap**: cropped images of segmented filament pairs

The output files are further described in [Output.md](Output.md).

    
  
      
Example
-------
The image sequence in the [example folder](example/Control) was preprocessed with the following commands:

    $ cd bin/preprocess
    
    # Move images from data1 and data2 to data folder
    $ ./rename_files.sh ../../example data1 data2
    
    # Sort files into specific experimental folders 
    $ ./move_files.sh ../../example Control Menadione
    
    # Generate movie
    $ python3 movie.py ../../example Control Menadione
    
    # Calculate background and test threshold (darkfield)
    $ python3 background.py ../../example 28 Control Menadione
    
The image sequence for the *Control* experiment is processed with:
    
    $ cd bin
    $ python3 run.py \
        --src ../example/Control \
        --px 5 \
        --expId example \
        --dark True \
        --dLink 17 \
        --dMerge 20 \
        --dMergeBox 7 \
        --kChamber 900 \
        --thresh 28 \
        --dt 20  

The script calculates the time between images from the modification time of the image file. Since the capture time is not known 
after downloading or cloning the images, the imaging frequency *dt* in seconds was added as argument.  
    



  
