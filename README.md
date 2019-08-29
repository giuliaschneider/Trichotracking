TrichoTracking
==============

**TrichoTracking** is a python package that processes image sequences of gliding filaments. 
It finds and quantifies the movement of filament pairs. 
Implementational details are found [here].

Usage
-----

    $ trichotracking.py --src <dirpath> --px <px> [OPTIONS]

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


Preprocess image sequence
-------------------------
Adjust directories and parameters in [preprocess.sh](bin/preprocess/preprocess.sh) to move images from different camera positions
to separate directoris and create a movie from each image sequence.  


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