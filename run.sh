#! /bin/bash

view=20
offset=10

CUDA_VISIBLE_DEVICES=1 python scripts/dps.py \
        --mission rlsd \
        --view $view \
        --offset $offset \
        --img_size 512 \
        --image_path /data/yuxuan/code/RadianceFieldStudio/outputs/hotdog_${view}_${offset}/gt_warped.png \
        --mask_path /data/yuxuan/code/RadianceFieldStudio/outputs/hotdog_${view}_${offset}/to_inpaint.png \
        --output_path /data/yuxuan/code/stablediffusion/result/rlsd_${view}_${offset}.png \
        --y_path /data/yuxuan/code/stablediffusion/result/rlsd_y.png \
        --gt_path /data/yuxuan/code/stablediffusion/result/rlsd_gt.png \