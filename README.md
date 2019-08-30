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
    

Use scripts in bin to process experiments. 



Usage
-----

### a) Preprocess data
The bash or python scripts in the folder [bin/preprocess](bin/preprocess) preprocess the experimental image sequence.

  1) If source images are in different folders in which they are number from 0 to 999. Rename and move the images to data folder:
      
         $ ./rename_files.sh ../../example/ data1 data2
      
  2) If images were captured at different camera positions, they can be moved to separate directories:
    
         $ ./move_files.sh ../../example/data Control Menadione
 
  3) Generate movie of image sequence:
  
         $ python3 movie.py ../../example/ Control Menadione
   
  4) Calculate background image and test threshold for segmentation by background subtraction.
         
         $ python3 background.py ../../example/ 28 Control Menadione

### b) Process image sequence
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
      
Example:

    $ python3 run.py \
        --src ../example/Control \
        --px 5 \
        --dark True \
        --dLink 17 \
        --dMerge 20 \
        --dMergeBox 7 \
        --kChamber 900 \
        --thresh 28


  
