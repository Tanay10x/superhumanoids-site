#!/bin/bash
# encode.sh <robot>  -> site assets/frames/<robot>/{d,m}/NNNN.webp
set -e
R=$1; OUT=/home/tanay/sh-build/out
declare -A NAME=([humanoid]=humanoid [quadruped]=quadruped [arm]=arm)
DST=/home/tanay/superhumanoids-site/assets/frames/${NAME[$R]}
rm -rf $DST/d $DST/m; mkdir -p $DST/d $DST/m
for f in $OUT/$R/*.png; do
  ffmpeg -loglevel error -y -i $f -c:v libwebp -quality 70 -compression_level 6 $DST/d/$(basename $f .png).webp
done
for f in $OUT/${R}_m/*.png; do
  ffmpeg -loglevel error -y -i $f -c:v libwebp -quality 70 -compression_level 6 $DST/m/$(basename $f .png).webp
done
du -sh $DST/d $DST/m
