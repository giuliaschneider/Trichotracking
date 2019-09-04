#!/usr/bin/env bash

# Data directory is passed as argument
args=( $@ );
parentdir=$1;
dirs=${args[@]:1};

# Save directories
cd "$parentdir"
datadir="$(pwd)/data"

mkdir ${dirs[@]}
ndirs="${#args[@]}"
ndirs=$(( ndirs-1 ))

i=0
d=1

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
     subdir="${args[$d]}"
     dir="$parentdir/$subdir"
     cd "$datadir"
     mv *$file "$dir"
     echo "$file" "$dir"
     d=$(( d % $ndirs + 1 ))
   fi
  fi
done


