TrichoTracking
==============

**TrichoTracking** is a python package that processes image sequences of gliding filaments. 
It finds and quantifies the movement of filament pairs. 
Implementational details are found [here].


Installation
------------

### a) Install OpenCV
Follow the steps from the [OpenCV installation guide](https://docs.opencv.org/4.1.0/da/df6/tutorial_py_table_of_contents_setup.html).

### b) Install Trichotracking
Download or clone *Trichotracking*:

    $ git clone https://github.com/giuliaschneider/Trichotracking.git
    cd Trichotracking
    python3 setup.py install
    



Usage
-----

### a) Preprocess data
The bash or python scripts in the folder [bin/preprocess](bin/preprocess) preprocess the experimental image sequence.

  1) If source images are in different folders in which they are number from 0 to 999. Rename and move the images to data folder:
      
  2) If images were captured at different camera positions, they can be moved to separate directories:
    
  3) Generate movie of image sequence:
  
  4) Calculate background image and test threshold for segmentation by background subtraction.
         

### b) Process image sequence
The image sequence is processed in three steps:
  1) Segmentation by background substraction and thresholding
  2) Linking of particles by nearest neighbor matching
  3) Merging and splitting of particles

Process the images by calling [bin/run.py]([bin/run.py]):

    $ run.py --src <dirpath> --px <px> [OPTIONS]

    Options:
      --src         Source directory of image sequence
      --px          Px length in [µm/px]
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
    
  - **tracks.csv**: contains data of each particle at each time step
  
      - angle: angle between horizontal axis and object in degrees
      - area: area
      - cx: centroid x-coordinate
      - cy: centroid y-coordinate
      - bx: x-coordinate of lower left corner of bounding box
      - by: y-coordinate of lower left corner of bounding box
      - bw: width of bounding box
      - bh: height of bounding box
      - eccentricity: eccentricity of fitted ellipse
      - ew1: largest eigenvalue
      - ew2: smallest eigenvalue
      - ews: quotient of eigenvalues (ew2/ew1)
      - frame: frame number
      - index: index number of each particle at each time step
      - length: length of filament in pixel
      - min_box: corners of minimal bounding box
      - min_box_h: height of minimal bounding rectangle
      - min_box_w: width of minimal bounding rectangle
      - trackNr: number of track
  - *pixellist.pkl*: pickle file contains list of contours of each particle at each time step, merge with tracks.csv over `index`
  - *tracks_meta.csv*: contains meta data of each track
    
      - trackNr: number of track
      - startTime: frame number of track start
      - endTime: frame number of track end
      - nFrames: number of track frames
      - length_mean: 
      - type: track type (1: single filament, 2: aligned filament pair, 3: ghosting filament pair, 4: aggregate of multiple filaments)
  - *aggregates_meta.csv*: contains meta data of each aggregate track
  - *pairs_meta.csv*: contains meta data of each aligned filament pair track
    
  - *pair_tracks.csv*: contains segmentation data of each aligned filament pair track 
      
      - length1 / length2: length of longer and shorter filaments in pixel
      - cx1 / cy1 / cx2 / cy2: x / y-coordinate of centroid of longer and shorter filaments
      - dirx1 / diry1: x
      - length_overlap,
      - xlov
      - ylov
      - pos_short
      - frame
      - trackNr
      - block
      
Example
-------
    $ cd bin/preprocess
    $ ./rename_files.sh ../../example/ data1 data2
    $ ./move_files.sh ../../example/data Control Menadione
    $ python3 movie.py ../../example/ Control Menadione
    $ python3 background.py ../../example/ 28 Control Menadione
    
    $ cd bin
    $ python3 run.py \
        --src ../example \
        --px 5 \
        --dark True \
        --dLink 17 \
        --dMerge 20 \
        --dMergeBox 7 \
        --kChamber 900 \
        --thresh 28


  
