#!/usr/bin/env bash

args=( $@ );
baseDir=$1;
dirs=${args[@]:1};

cd "$srcDir"

mkdir "$srcDir/data"

i=0
for dir in "${dirs[@]}"; do
    cd "$dir"
    i=$((i+1))
    mmv \* "$i"_\#1
    mv *.JPG "$srcDir/data"
    cd "$srcDir"
done


