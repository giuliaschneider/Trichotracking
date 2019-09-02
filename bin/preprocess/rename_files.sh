#!/usr/bin/env bash

args=("$@");
baseDir=$1;

cd "$baseDir"

mkdir "data"

i=0
for dir in "${args[@]:1}"; do
    cd "$dir"
    i=$((i+1))
    mmv \* "$i"_\#1
    mv *.JPG "$baseDir/data"
    cd ..
done


