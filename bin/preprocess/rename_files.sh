#!/usr/bin/env bash

srcDir=$1; echo $srcDir
dirs=( $2 $3 ); echo "${dirs[@]}"

cd "$srcDir"

i=0
for dir in "${dirs[@]}"; do
    cd "$dir"
    i=$((i+1))
    mmv \* "$i"_\#1
    mv *.JPG "$srcDir/data"
    cd "$srcDir"
done


