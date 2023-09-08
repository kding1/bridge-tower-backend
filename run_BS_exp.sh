#!/bin/bash
echo "fp32 runs"
python -u eval_flickr.py  --precision 'fp32' --batch_size 1 --calib_samples 32 |& tee -a ./BS_Exp3_log.txt
python -u eval_flickr.py  --precision 'fp32' --batch_size 16 --calib_samples 32 |& tee -a ./BS_Exp3_log.txt
python -u eval_flickr.py  --precision 'fp32' --batch_size 32 --calib_samples 32 |& tee -a ./BS_Exp3_log.txt
python -u eval_flickr.py  --precision 'fp32' --batch_size 64 --calib_samples 64 |& tee -a ./BS_Exp3_log.txt

# echo "bf16 runs"
# python -u eval_flickr.py  --precision 'bf16' --batch_size 1 --calib_samples 32|& tee -a ./BS_Exp2_log.txt
# python -u eval_flickr.py  --precision 'bf16' --batch_size 16 --calib_samples 32 |& tee -a ./BS_Exp2_log.txt
# python -u eval_flickr.py  --precision 'bf16' --batch_size 32 --calib_samples 32 |& tee -a ./BS_Exp2_log.txt
# python -u eval_flickr.py  --precision 'bf16' --batch_size 64 --calib_samples 64 |& tee -a ./BS_Exp2_log.txt

echo "int8 runs"
python -u eval_flickr.py  --precision 'int8' --batch_size 1 --calib_samples 32 |& tee -a ./BS_Exp3_log.txt
python -u eval_flickr.py  --precision 'int8' --batch_size 16 --calib_samples 32 |& tee -a ./BS_Exp3_log.txt
python -u eval_flickr.py  --precision 'int8' --batch_size 32 --calib_samples 32 |& tee -a ./BS_Exp3_log.txt
python -u eval_flickr.py  --precision 'int8' --batch_size 64 --calib_samples 64 |& tee -a ./BS_Exp3_log.txt

