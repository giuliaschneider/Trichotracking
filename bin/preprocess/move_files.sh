#!/usr/bin/env bash

# Data directory is passed as argument
args=( $@ );
datadir=$1; echo $datadir
dirs=${args[@]:1}; echo "${dirs[@]}"



# Save directories
cd "$datadir"; cd ..
parentdir=$(pwd)
datadir="$parentdir/data"


mkdir ${dirs[@]}
ndirs="${#args[@]}"
ndirs=$(( ndirs-1 ))


i=0
d=1

# Iterate pictures
cd "$datadir"
echo "$(pwd)"
files=(*.JPG)
imax="${#files[@]}"

for file in "${files[@]}"; do
  # Epoch time of file
  t0=$(date -r $file +"%s");

  i=$((i+1))
  if [ $i -lt $imax ]; then
   t1=$(date -r "${files[$i]}" +"%s");
   diff=$(($t1 - $t0))
   echo $diff
   if [ $diff -gt 1 ]; then
     subdir="${args[$d]}"
     echo $d
     echo $subdir
     dir="$parentdir/$subdir"
     cd "$datadir"
     mv *$file "$dir"
     echo "$file" "$dir"
     d=$(( (d+1) % $ndirs + 1 ))
   fi
  fi
done


