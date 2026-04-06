#!/usr/bin/env bash

set -euo pipefail

DATASET_DIR=/mnt/newdisk/datasets/carla_datasets/CarlaOccV1
SAVE_DIR=/mnt/disk/datasets/carla_datasets/CarlaOccV1_zip

cd ${DATASET_DIR}
if [ ! -d ${SAVE_DIR} ]; then
    mkdir -p ${SAVE_DIR}
    mkdir -p ${SAVE_DIR}/all_rgb
    mkdir -p ${SAVE_DIR}/all_depth
    mkdir -p ${SAVE_DIR}/all_semantics
    mkdir -p ${SAVE_DIR}/all_normal
    mkdir -p ${SAVE_DIR}/all_lidar
    mkdir -p ${SAVE_DIR}/all_semantic_lidar
    mkdir -p ${SAVE_DIR}/all_occupancy
fi

TOWN_NAMES=("Town01_Opt" "Town02_Opt" "Town03_Opt" "Town04_Opt" "Town05_Opt" "Town06_Opt" "Town07_Opt" "Town10HD_Opt")
SEQUENCES=(00 01 02 03 04 05 06 07 08 09 10 11 12)

for TOWN_NAME in ${TOWN_NAMES[@]}; do
    for SEQUENCE in ${SEQUENCES[@]}; do
       tar -cvf - ${TOWN_NAME}_Seq${SEQUENCE}/rgb/ | pigz > ${SAVE_DIR}/all_rgb/${TOWN_NAME}_Seq${SEQUENCE}_rgb.tar.gz
       tar -cvf - ${TOWN_NAME}_Seq${SEQUENCE}/depth/ | pigz > ${SAVE_DIR}/all_depth/${TOWN_NAME}_Seq${SEQUENCE}_depth.tar.gz
       tar -cvf - ${TOWN_NAME}_Seq${SEQUENCE}/semantics/ | pigz > ${SAVE_DIR}/all_semantics/${TOWN_NAME}_Seq${SEQUENCE}_semantics.tar.gz
       tar -cvf - ${TOWN_NAME}_Seq${SEQUENCE}/normal/ | pigz > ${SAVE_DIR}/all_normal/${TOWN_NAME}_Seq${SEQUENCE}_normal.tar.gz
       tar -cvf - ${TOWN_NAME}_Seq${SEQUENCE}/lidar/ | pigz > ${SAVE_DIR}/all_lidar/${TOWN_NAME}_Seq${SEQUENCE}_lidar.tar.gz
       tar -cvf - ${TOWN_NAME}_Seq${SEQUENCE}/semantic_lidar/ | pigz > ${SAVE_DIR}/all_semantic_lidar/${TOWN_NAME}_Seq${SEQUENCE}_semantic_lidar.tar.gz
       tar -cvf - ${TOWN_NAME}_Seq${SEQUENCE}/occupancy/vs_0_1/ | pigz > ${SAVE_DIR}/all_occupancy/${TOWN_NAME}_Seq${SEQUENCE}_occupancy_vs_0_1.tar.gz
       tar -cvf - ${TOWN_NAME}_Seq${SEQUENCE}/occupancy/vs_0_2_forward_view/ | pigz > ${SAVE_DIR}/all_occupancy/${TOWN_NAME}_Seq${SEQUENCE}_occupancy_vs_0_2_forward_view.tar.gz
       tar -cvf - ${TOWN_NAME}_Seq${SEQUENCE}/occupancy/vs_0_4_surround_view/ | pigz > ${SAVE_DIR}/all_occupancy/${TOWN_NAME}_Seq${SEQUENCE}_occupancy_vs_0_4_surround_view.tar.gz
    done
done

tar -cvf - calib | pigz > ${SAVE_DIR}/calib.tar.gz
tar -cvf - */poses | pigz > ${SAVE_DIR}/all_poses.tar.gz
tar -cvf - */traffic_info | pigz > ${SAVE_DIR}/all_traffic_info.tar.gz