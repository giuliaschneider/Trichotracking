#!/usr/bin/env bash

# Data directory is passed as argument
datadir=$1; echo $datadir
dirs=( $2 $3 $4 $5); echo "${dirs[@]}"

# Save directories
cd "$datadir"; cd ..
parentdir=$(pwd)


mkdir ${dirs[@]}
ndirs="${#dirs[@]}"

i=0
d=0

# Iterate pictures
cd "$datadir"
files=(*.JPG)
imax="${#files[@]}"

for file in "${files[@]}"; do
  # Epoch time of file
  t0=$(date -r $file +"%s");

  i=$((i+1))
  if [ $i -lt $imax ]; then
   t1=$(date -r "${files[$i]}" +"%s");
   diff=$(($t1 - $t0))
   if [ $diff -gt 1 ]; then
     subdir="${dirs[$d]}"
     echo $d
     echo $subdir
     dir="$parentdir/$subdir"
     cd "$datadir"
     mv *$file "$dir"
     echo "$file" "$dir"
     d=$(( (d+1) % $ndirs ))
   fi
  fi
done


