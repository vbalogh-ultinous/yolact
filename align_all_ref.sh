#!/bin/bash

METRICS=${1-yolact_metrics.json}
HEAD=${2-/data/vbalogh/head_det_corpus_v3/bounding_boxes/train_v3.csv}


python ../Mask_RCNN/align.py --head $HEAD --person /data/vbalogh/yolact/detections/film1 --images /data/vbalogh/head_det_corpus_v3/film1 --outdir /data/vbalogh/yolact/align/film1 --name film1 --metrics $METRICS --reference /data/vbalogh/MaskRCNN/detections/film1
python ../Mask_RCNN/align.py --head $HEAD --person /data/vbalogh/yolact/detections/film2 --images /data/vbalogh/head_det_corpus_v3/film2 --outdir /data/vbalogh/yolact/align/film2 --name film2 --metrics $METRICS --reference /data/vbalogh/MaskRCNN/detections/film2
python ../Mask_RCNN/align.py --head $HEAD --person /data/vbalogh/yolact/detections/film3 --images /data/vbalogh/head_det_corpus_v3/film3 --outdir /data/vbalogh/yolact/align/film3 --name film3 --metrics $METRICS --reference /data/vbalogh/MaskRCNN/detections/film3
python ../Mask_RCNN/align.py --head $HEAD --person /data/vbalogh/yolact/detections/film4 --images /data/vbalogh/head_det_corpus_v3/film4 --outdir /data/vbalogh/yolact/align/film4 --name film4 --metrics $METRICS --reference /data/vbalogh/MaskRCNN/detections/film4
python ../Mask_RCNN/align.py --head $HEAD --person /data/vbalogh/yolact/detections/film5 --images /data/vbalogh/head_det_corpus_v3/film5 --outdir /data/vbalogh/yolact/align/film5 --name film5 --metrics $METRICS --reference /data/vbalogh/MaskRCNN/detections/film5
python ../Mask_RCNN/align.py --head $HEAD --person /data/vbalogh/yolact/detections/film6 --images /data/vbalogh/head_det_corpus_v3/film6 --outdir /data/vbalogh/yolact/align/film6 --name film6 --metrics $METRICS --reference /data/vbalogh/MaskRCNN/detections/film6
python ../Mask_RCNN/align.py --head $HEAD --person /data/vbalogh/yolact/detections/film7 --images /data/vbalogh/head_det_corpus_v3/film7 --outdir /data/vbalogh/yolact/align/film7 --name film7 --metrics $METRICS --reference /data/vbalogh/MaskRCNN/detections/film7
python ../Mask_RCNN/align.py --head $HEAD --person /data/vbalogh/yolact/detections/film8 --images /data/vbalogh/head_det_corpus_v3/film8 --outdir /data/vbalogh/yolact/align/film8 --name film8 --metrics $METRICS --reference /data/vbalogh/MaskRCNN/detections/film8
python ../Mask_RCNN/align.py --head $HEAD --person /data/vbalogh/yolact/detections/film9 --images /data/vbalogh/head_det_corpus_v3/film9 --outdir /data/vbalogh/yolact/align/film9 --name film9 --metrics $METRICS --reference /data/vbalogh/MaskRCNN/detections/film9

python ../Mask_RCNN/align.py --head $HEAD --person /data/vbalogh/yolact/detections/rossmann_cash1 --images /data/vbalogh/head_det_corpus_v3/rossmann_cash1 --outdir /data/vbalogh/yolact/align/rossmann_cash1 --name rossmann_cash1 --metrics $METRICS --reference /data/vbalogh/MaskRCNN/detections/rossmann_cash1
python ../Mask_RCNN/align.py --head $HEAD --person /data/vbalogh/yolact/detections/rossmann_cash2 --images /data/vbalogh/head_det_corpus_v3/rossmann_cash2 --outdir /data/vbalogh/yolact/align/rossmann_cash2 --name rossmann_cash2 --metrics $METRICS --reference /data/vbalogh/MaskRCNN/detections/rossmann_cash2
python ../Mask_RCNN/align.py --head $HEAD --person /data/vbalogh/yolact/detections/rossmann_entrance --images /data/vbalogh/head_det_corpus_v3/rossmann_entrance --outdir /data/vbalogh/yolact/align/rossmann_entrance --name rossmann_entrance --metrics $METRICS --reference /data/vbalogh/MaskRCNN/detections/rossmann_entrance
python ../Mask_RCNN/align.py --head $HEAD --person /data/vbalogh/yolact/detections/rossmann_line --images /data/vbalogh/head_det_corpus_v3/rossmann_line --outdir /data/vbalogh/yolact/align/rossmann_line --name rossmann_line --metrics $METRICS --reference /data/vbalogh/MaskRCNN/detections/rossmann_line

python ../Mask_RCNN/align.py --head $HEAD --person /data/vbalogh/yolact/detections/HollywoodHeads/JPEGImages --images /data/vbalogh/head_det_corpus_v3/HollywoodHeads/JPEGImages --outdir /data/vbalogh/yolact/align/HollywoodHeads/JPEGImages --name HollywoodHeads --metrics $METRICS --reference /data/vbalogh/MaskRCNN/detections/HollywoodHeads/JPEGImages
python ../Mask_RCNN/align.py --head $HEAD --person /data/vbalogh/yolact/detections/MPII/images --images /data/vbalogh/head_det_corpus_v3/MPII/images --outdir /data/vbalogh/yolact/align/MPII/images --name MPII --metrics $METRICS --reference /data/vbalogh/MaskRCNN/detections/MPII/images


