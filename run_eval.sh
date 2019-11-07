#!/bin/bash

MODEL=${1-weights/yolact_resnet101_base_54_800000.pth}
INPUT=${2-/data/vbalogh/head_det_corpus_v3/film9/}
OUTPUT=${3-/data/vbalogh/yolact/detections/head_det_corpus_v3/film9/}

python custom_eval.py --trained_model=$MODEL --score_threshold=0.3 --top_k=100 --benchmark --images=$INPUT:$OUTPUT
