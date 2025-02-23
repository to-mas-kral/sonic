for f in ../*.exr*; do magick "$f" -crop 256x256+768+768 crop/"$f" ; done
