#!/bin/bash

python custom_eval.py --trained_model weights/yolact_im700_54_800000.pth --score_threshold 0.3 --top_k 100 --benchmark --images /data/vbalogh/head_det_corpus_v3/film8:/data/vbalogh/yolact/detections/film8
python custom_eval.py --trained_model weights/yolact_im700_54_800000.pth --score_threshold 0.3 --top_k 100 --benchmark --images /data/vbalogh/head_det_corpus_v3/film9:/data/vbalogh/yolact/detections/film9
python custom_eval.py --trained_model weights/yolact_im700_54_800000.pth --score_threshold 0.3 --top_k 100 --benchmark --images /data/vbalogh/head_det_corpus_v3/film7:/data/vbalogh/yolact/detections/film7
python custom_eval.py --trained_model weights/yolact_im700_54_800000.pth --score_threshold 0.3 --top_k 100 --benchmark --images /data/vbalogh/head_det_corpus_v3/film6:/data/vbalogh/yolact/detections/film6
python custom_eval.py --trained_model weights/yolact_im700_54_800000.pth --score_threshold 0.3 --top_k 100 --benchmark --images /data/vbalogh/head_det_corpus_v3/film5:/data/vbalogh/yolact/detections/film5
python custom_eval.py --trained_model weights/yolact_im700_54_800000.pth --score_threshold 0.3 --top_k 100 --benchmark --images /data/vbalogh/head_det_corpus_v3/film4:/data/vbalogh/yolact/detections/film4
python custom_eval.py --trained_model weights/yolact_im700_54_800000.pth --score_threshold 0.3 --top_k 100 --benchmark --images /data/vbalogh/head_det_corpus_v3/film3:/data/vbalogh/yolact/detections/film3
python custom_eval.py --trained_model weights/yolact_im700_54_800000.pth --score_threshold 0.3 --top_k 100 --benchmark --images /data/vbalogh/head_det_corpus_v3/film2:/data/vbalogh/yolact/detections/film2
python custom_eval.py --trained_model weights/yolact_im700_54_800000.pth --score_threshold 0.3 --top_k 100 --benchmark --images /data/vbalogh/head_det_corpus_v3/film1:/data/vbalogh/yolact/detections/film1

python custom_eval.py --trained_model weights/yolact_im700_54_800000.pth --score_threshold 0.3 --top_k 100 --benchmark --images /data/vbalogh/head_det_corpus_v3/HollywoodHeads/JPEGImages:/data/vbalogh/yolact/detections/HolylwoodHeads/JPEGImages
python custom_eval.py --trained_model weights/yolact_im700_54_800000.pth --score_threshold 0.3 --top_k 100 --benchmark --images /data/vbalogh/head_det_corpus_v3/MPII/images:/data/vbalogh/yolact/detections/MPII/images
