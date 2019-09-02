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
  
         $ cd bin/preprocess
         $ ./rename_files.sh <path-to-parent-folder> data1 data2
      
  2) If images were captured at different camera positions, they can be moved to separate directories:
    
         $ ./move_files.sh <path-to-source-folder> Control Menadione
  3) Generate movie of image sequence:
  
         $ python3 movie.py <path-to-source-folder> Control Menadione  
  4) Calculate background image and test threshold for segmentation by background subtraction.
     
         $ python3 background.py <path-to-source-folder> <threshold> Control Menadione


### b) Process image sequence
The image sequence is processed in three steps:
  1) Segmentation by background substraction and thresholding
  2) Linking of particles by nearest neighbor matching
  3) Merging and splitting of particles

Process the images by calling [bin/run.py]([bin/run.py]):

    $ run.py --src <path-to-image-folder> --px <px> [OPTIONS]

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
      
### c) Output
The script outputs different files (per default in a *result* folder in the image directory):
    
  - **tracks.csv**: contains data of each particle at each time step
      - `trackNr`: number of track
      - `label`: unique track identifier to compare tracks from different experiments (expId + trackNr)
      - `index`: index number of each particle at each time step
      - `frame`: frame number
      - `time`: image time in seconds since epoch (Linux)
      - `timestamp`: image time as timestamp
      - `angle`: angle between horizontal axis and object in degrees
      - `area`: area
      - `cx`: centroid x-coordinate in pixel
      - `cx_um`: centroid x-coordinate in microns
      - `cx_ma`: smoothed centroid x-coordinate in microns
      - `cy`: centroid y-coordinate in pixel
      - `cy_um`: centroid y-coordinate in microns
      - `cy_ma`: smoothed centroid y-coordinate in microns
      - `bx`: x-coordinate of lower left corner of bounding box
      - `by`: y-coordinate of lower left corner of bounding box
      - `bw`: width of bounding box
      - `bh`: height of bounding box
      - `eccentricity`: eccentricity of fitted ellipse
      - `ew1`: largest eigenvalue
      - `ew2`: smallest eigenvalue
      - `ews`: quotient of eigenvalues (ew2/ew1)
      - `length`: filament length in pixel
      - `length_um`: filament length in micro meter
      - `min_box`: corners of minimal bounding box
      - `min_box_h`: height of minimal bounding rectangle
      - `min_box_w`: width of minimal bounding rectangle
      - `reversal`: boolean, 1 if particle reverses at current frame
      - `v`: signed particle velocity, positive if particle is moving in positive y-direction
      - `v_abs`: absolute particle velocity
      
  - **pixellist.pkl**: pickle file contains list of contours of each particle at each time step, merge with tracks.csv over `index`
  
  - **tracks_meta.csv**: contains meta data of each track
    
      - `trackNr`: number of track
      - `startTime`: frame number of track start
      - `endTime`: frame number of track end
      - `nFrames`: number of track frames
      - `length_mean`: track-averaged filament length in pixel
      - `type`: track type (1: single filament, 2: aligned filament pair, 3: ghosting filament pair, 4: aggregate of multiple filaments)
      - `vabs_mean`: track averaged absolute particle velocity
      - `vabs_mean_without_stalling`: track averaged absolute particle velocity (v > 0.1 µm/s)
      - `fstalling`: fraction of time the particle is stalling (v > 0.1 µm/s)
      - `n_reversals`: number of reversals
      - `f_reversal`: theoretical reversal frequency = n_reversals / track duration
      
  - **aggregates_meta.csv**: contains meta data of each aggregate track
      - `trackNr`: number of track
      - `t0`: frame number of track start
      - `t1`: frame number of track end
      - `tracks0`: tracks merged
      - `stracks0`: single filament tracks merged
      - `tracks1`: tracks split
      - `stracks1`: single filament tracks split
      - `mTrack1`: first merged or split track
      - `mTrack2`: second merged or split track
      - `n`: number of single filaments
      - `breakup`: breakup reason (1: aggregate with other particle, 2: splits up, 3: movie finished, 4: unknown)
      
  - **pairs_meta.csv**: contains meta data of each aligned filament pair track
      - `trackNr`: number of track
      - `length1` / `length2`: length of longer / shorter filaments in pixel
      - `breakup`: breakup reason (1: aggregate with other particle, 2: splits up, 3: movie finished, 4: unknown)
      - `couldSegment`: boolean indicating if segmenation was successful
      - `failFraction`: fraction of frames which were not segmented
      - `time`:  image time in seconds since epoch (Linux)
      - `v1_mean` /`v2_mean`: track averaged velocity of longer / shorter filament 
      - `relative_v_mean`: track averaged relative velocity
      - `type`: filament pair type (1: reversing, non-separating; 2: reversing, separating; 3: non-reversing, non-separating; 4: non-reversing, separating)
      - `n_reversals`: number of reversals
      - `f_reversal`: theoretical reversal frequency = n_reversals / track duration
      - `lol_reversals_mean`: track averaged lol at reversals in microns
      - `lol_reversals_normed_mean`: track averaged normed lol at reversals
      - `total_length`: summed filament length
      - `rel_v_mean_without_stalling`: track averaged relative velocity (v<sub>rel</sub> > 0.1 µm/s)
      - `fstalling`: fraction of time the particle is stalling (v<sub>rel</sub> > 0.1 µm/s)
    
  - *pair_tracks.csv*: contains segmentation data of each aligned filament pair track 
      - `frame`: frame number
      - `trackNr`: track number
      - `time`: image time in seconds since epoch (Linux)
      - `timestamp`: image time as timestamp
      - `label`: ?? 
      - `length1` / `length2`: length of longer / shorter filaments in pixel
      - `l1_um`/ `l2_um`: length of longer / shorter filaments in microns
      - `cx1` / `cy1` / `cx2` / `cy2`: x / y-coordinate of centroid of longer / shorter filaments in pixel
      - `cx1_um`/ `cy1_um`/ `cx2_um`/ `cy2_um`: x / y-coordinate of centroid of longer / shorter filaments in microns
      - `cx1_ma`/ `cx2_ma`/ `cy1_ma`/ `cy2_ma`: smoothed x / y-coordinate of centroid of longer / shorter filaments in microns
      - `dirx1` / `diry1`: x / y-direction of longer filament
      - `length_overlap`: length of overlap region in pixel
      - `lo_um`: length of overlap region in microns
      - `xlol` / `xlol_um`:  lack of overlap (LOL) in longitudinal direction in pixel / microns
      - `ylol` / `ylol_um`:  LOL in lateral direction in pixel / microns
      - `lol_norm`: LOL normalized by shorter filament
      - `lol_norm_ma`: smoothed normalized LOL
      - `lol_norm_ma_abs`: absolute smoothed normalize LOL
      - `lol_ma`: smoothed longitudinal LOL in microns
      - `pos_rel`: relative position of the short along longer filament
      - `pos`: position of the short fil along long fil in microns
      - `pos_ma`: smoothed position of the short fil along long fil in microns
      - `v1` / `v2`: longer / shorter filament velocity
      - `v_rel`: relative filament pair velocity in microns per second
      - `v_rel_ma`: smoothed relative filament pair velocity in microns per second
      - `v_rel_abs`: absolute, smoothed relative filament pair velocity in microns per second
      - `reversal`: boolean, 1 if filament pair reverses at current frame
      - `lol_reversals_normed`: normed LOL at reversals
      - `lol_reversals`: LOL in microns at reveresals
      - `v_lol`: signed relative velocity, positive for increasing LOL
      
Example
-------

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


  
