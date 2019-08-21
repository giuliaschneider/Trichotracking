TrichoTracking
==============

**TrichoTracking** is python package that processes image sequences of gliding filaments. 
It finds and quantifies the movement of filament pairs. 
Implementational details are found [here](https://github.com/giuliaschneider/Trichotracking/blob/master/Quantification-of-filament-interactions.pdf
)


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


Installation
------------