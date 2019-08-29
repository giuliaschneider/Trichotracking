#!/usr/bin/env bash

# Preprocess files
# 1. If images in different source folders, rename files and move them to one folder: dataDir
# 2. If different positions were imaged, move them to separate folders
# 3. Generate a movie of the image sequence
# 4. Calculate the background and roi image

baseDir="/home/giu/Documents/Trichos_Uli_Setup/data/SpeedTest/7mm"
dataDir="$baseDir/data"
dataDirs1="data1"
dataDirs2="data2"

posDir1="Control_undiluted"
posDir2="Control_diluted"
posDir3="Menadione_diluted"
posDir4="Menadione_undiluted"



#./rename_files.sh "$baseDir" "$dataDirs1" "$dataDirs2"
./move_files.sh "$dataDir" "$posDir1" "$posDir2" "$posDir3" "$posDir4"
./movie.sh "$baseDir" "$posDir1" "$posDir2" "$posDir3" "$posDir4"
#./background_threshold.sh "$baseDir" "$posDir2" "$posDir3"


#python3 /home/giu/Documents/projects/Trichotracking/bin/run.py \
#--src "$baseDir/$posDir2" \
#--px 5 \
#--dark True \
#--dLink 20 \
#--dMerge 20 \
#--dMergeBox 10 \
#--kChamber 900 \
#--thresh 28
#
#python3 /home/giu/Documents/projects/Trichotracking/bin/run.py \
#--src "$baseDir/$posDir3" \
#--px 5 \
#--dark True \
#--dLink 20 \
#--dMerge 20 \
#--dMergeBox 10 \
#--kChamber 900 \
#--thresh 28


