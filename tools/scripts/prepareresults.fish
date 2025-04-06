#!/usr/bin/env fish

./compare_images.fish ../../cmake-build-release/out/kitchenpc/normal.exr-131072.exr  ../../cmake-build-release/out/kitchenpc/lg.exr ./csvs/kitchenpc-lg.csv
./compare_images.fish ../../cmake-build-release/out/kitchenpc/normal.exr-131072.exr  ../../cmake-build-release/out/kitchenpc/normal.exr ./csvs/kitchenpc-normal.csv

./compare_images.fish ../../cmake-build-release/out/cboxf/normal.exr-131072.exr  ../../cmake-build-release/out/cboxf/lg.exr ./csvs/cboxf-lg.csv
./compare_images.fish ../../cmake-build-release/out/cboxf/normal.exr-131072.exr  ../../cmake-build-release/out/cboxf/normal.exr ./csvs/cboxf-normal.csv

./compare_images.fish ../../cmake-build-release/out/cbox/normal.exr-131072.exr  ../../cmake-build-release/out/cbox/lg.exr ./csvs/cbox-lg.csv
./compare_images.fish ../../cmake-build-release/out/cbox/normal.exr-131072.exr  ../../cmake-build-release/out/cbox/normal.exr ./csvs/cbox-normal.csv

./compare_images.fish ../../cmake-build-release/out/staircaseph/normal.exr-131072.exr  ../../cmake-build-release/out/staircaseph/lg.exr ./csvs/staircaseph-lg.csv
./compare_images.fish ../../cmake-build-release/out/staircaseph/normal.exr-131072.exr  ../../cmake-build-release/out/staircaseph/normal.exr ./csvs/staircaseph-normal.csv

./compare_images.fish ../../cmake-build-release/out/machines/normal.exr-131072.exr  ../../cmake-build-release/out/machines/lg.exr ./csvs/machines-lg.csv
./compare_images.fish ../../cmake-build-release/out/machines/normal.exr-131072.exr  ../../cmake-build-release/out/machines/normal.exr ./csvs/machines-normal.csv

python ./convergence_graphs.py

./crop_images.fish ../../cmake-build-release/out/kitchenpc/normal.exr 430 360 128 128 ./cropimages/kitchenpc/
./crop_images.fish ../../cmake-build-release/out/kitchenpc/lg.exr 430 360 128 128 ./cropimages/kitchenpc/

./crop_images.fish ../../cmake-build-release/out/cboxf/normal.exr 861 896 128 128 ./cropimages/cboxf/
./crop_images.fish ../../cmake-build-release/out/cboxf/lg.exr 861 896 128 128 ./cropimages/cboxf/

./crop_images.fish ../../cmake-build-release/out/cbox/normal.exr 861 896 128 128 ./cropimages/cbox/
./crop_images.fish ../../cmake-build-release/out/cbox/lg.exr 861 896 128 128 ./cropimages/cbox/

./crop_images.fish ../../cmake-build-release/out/staircaseph/normal.exr 190 680 128 128 ./cropimages/staircaseph/
./crop_images.fish ../../cmake-build-release/out/staircaseph/lg.exr 190 680 128 128 ./cropimages/staircaseph/

./crop_images.fish ../../cmake-build-release/out/staircaseph/normal.exr 480 400 128 128 ./cropimages/staircaseph/
./crop_images.fish ../../cmake-build-release/out/staircaseph/lg.exr 480 400 128 128 ./cropimages/staircaseph/

./crop_images.fish ../../cmake-build-release/out/machines/lg.exr 1290 200 128 128 ./cropimages/machines/
./crop_images.fish ../../cmake-build-release/out/machines/normal.exr 1290 200 128 128 ./cropimages/machines/
