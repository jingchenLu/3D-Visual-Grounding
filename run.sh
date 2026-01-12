# CUDA_VISIBLE_DEVICES=0 python3 scripts/joint_scripts/train_3dvlp.py --use_multiview --use_normal --batch_size 4 --epoch 200 --lang_num_max 8 --coslr --lr 0.002 --no_caption --lang_num_aug 0 --unfreeze 6  --debug --use_con --use_diou_loss
# python3 scripts/joint_scripts/train_3dvlp.py --gpu 1 --use_multiview --use_normal --batch_size 8 --epoch 230 --lang_num_max 8 --coslr --lr 0.002 --no_caption --lang_num_aug 0 --unfreeze 6  --debug --use_con --use_diou_loss

# python3 scripts/joint_scripts/train_3dvlp.py --gpu 0 --use_multiview --use_normal --batch_size 8 --epoch 200 --lang_num_max 8 --coslr --lr 0.002 --no_caption --lang_num_aug 0 --unfreeze 6  --debug --use_con --use_diou_loss

CUDA_VISIBLE_DEVICES=1 python3 scripts/joint_scripts/train_3dvlp.py --use_checkpoint /home/ljc/work/3DVLP/outputs/exp_joint/2026-01-08_00-26-44  --gpu 0 --use_multiview --use_normal --batch_size 8 --epoch 220 --lang_num_max 8 --coslr --lr 0.002 --no_caption --lang_num_aug 0 --unfreeze 6  --debug --use_con --use_diou_loss
