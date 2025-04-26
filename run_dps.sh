#! /bin/bash

view=20
offset=30

CUDA_VISIBLE_DEVICES=1 python scripts/dps.py \
        --mission dps \
        --view $view \
        --offset $offset \
        --image_path /data/yuxuan/code/RadianceFieldStudio/exports/hotdog_${view}_${offset}/gt_warped.png \
        --mask_path /data/yuxuan/code/RadianceFieldStudio/exports/hotdog_${view}_${offset}/to_inpaint.png \
        --output_path /data/yuxuan/code/stablediffusion/result/dps_${view}_${offset}.png \
        # --y_path /data/yuxuan/code/stablediffusion/result/rlsd_y.png \
        # --gt_path /data/yuxuan/code/stablediffusion/result/rlsd_gt.png \