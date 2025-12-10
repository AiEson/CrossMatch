
python train_cross_match.py \
--exp "25_75_AdamW_14000_lab4" \
--conf_thresh 0.75 \
--label_num 4 \
--max_iterations 14000 \
--optimizer AdamW \
--base_lr 0.001 

python train_cross_match.py \
--exp "25_75_AdamW_14000_lab8" \
--conf_thresh 0.75 \
--label_num 8 \
--max_iterations 14000 \
--optimizer AdamW \
--base_lr 0.001 

python train_cross_match.py \
--exp "25_75_AdamW_14000_lab16" \
--conf_thresh 0.75 \
--label_num 16 \
--max_iterations 14000 \
--optimizer AdamW \
--base_lr 0.001

python test.py --model "25_75_AdamW_14000_lab4" --epoch_num 14001
python test.py --model "25_75_AdamW_14000_lab8" --epoch_num 14001
python test.py --model "25_75_AdamW_14000_lab16" --epoch_num 14001
