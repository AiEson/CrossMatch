
python train_cross_match.py \
--exp "25_75_AdamW_9000_lab4" \
--conf_thresh 0.85 \
--label_num 4 \
--max_iterations 9000 \
--optimizer AdamW \
--base_lr 0.001 

python train_cross_match.py \
--exp "25_75_AdamW_9000_lab8" \
--conf_thresh 0.85 \
--label_num 8 \
--max_iterations 9000 \
--optimizer AdamW \
--base_lr 0.001 

python train_cross_match.py \
--exp "25_75_AdamW_9000_lab16" \
--conf_thresh 0.85 \
--label_num 16 \
--max_iterations 9000 \
--optimizer AdamW \
--base_lr 0.001

python test.py --model "25_75_AdamW_9000_lab4" --epoch_num 9001
python test.py --model "25_75_AdamW_9000_lab8" --epoch_num 9001
python test.py --model "25_75_AdamW_9000_lab16" --epoch_num 9001