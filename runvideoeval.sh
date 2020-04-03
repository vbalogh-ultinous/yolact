#!/bin/bash

FILENAMES=${1-/data/vbalogh/datasets/personDetData/gt/Ulf/video_files.txt}
INDIR=${2-/data/vbalogh/datasets/personDetData/gt/Ulf}
OUTDIR=${3-/data/vbalogh/datasets/personDetData/output_detections/yolact_detections/Ulf/DL/}
MODEL=${4-weights/yolact_plus_base_54_800000.pth}
CONFIG=${5-yolact_plus_base_config}
SCORE=${6-0.22}

#for filename in $INDIR/*$EXTENSION; do
while read f; do
    echo $f
    mkdir -p "$OUTDIR"
    python3 custom_eval.py --trained_model $MODEL --config $CONFIG --score_threshold $SCORE --top_k 15 --video_multiframe 4 --video $INDIR/$f:$OUTDIR/out_${SCORE}_$f --coco_class person --display_masks 0 --csv $OUTDIR/csv/$f.csv
    #python3 custom_eval.py --trained_model weights/yolact_plus_person_SO.pth --config yolact_plus_person_config --score_threshold 0.22 --top_k 15 --video_multiframe 4 --video $INDIR/$f:$OUTDIR/$PREFIX$f --coco_class person --display_masks 0 --csv $OUTDIR/csv/$f.csv

    
    
done < $FILENAMES
