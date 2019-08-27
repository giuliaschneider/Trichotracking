#!/usr/bin/env bash

datadir=$1; echo $datadir
dirs=( $2 $3 $4 $5); echo "${dirs[@]}"

scriptPath=".."

for subdir in "${dirs[@]}"; do
  dir="$datadir/$subdir"
  echo $dir
  python3 "$scriptPath/movie.py" $dir
done;
