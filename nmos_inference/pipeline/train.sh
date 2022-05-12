#! /bin/bash
PYTHON=/home/charon/anaconda3/envs/ml37/bin/python
JOBS=1
DELAY=5
# model
MODEL_NAME=(nmos_fcn)
INPUT_SIZE=(2)
ENCODER_HIDDEN_SIZE=(32)
ARG_COMP_HIDDEN_SIZE=(32)
# optimization
LOSS=(diy)
LR=(0.001)
MAX_EPOCHS=(50)
OPTIMIZER=(Adam)
LR_SCHEDULER=(cosine)
TRAIN_BATCH_SIZE=(256)
WEIGHT_DECAY=(0.05)
DROPOUT=(0.3)
# AUGMENTATION
AUG=False
AUG_PROB=(0.0)
# hardware
STRATEGY=None
PRECISION=(32)
GPUS=(2)
# randomness
SEED=(42)
# logging
EXP_NAME=(FCN)
RUN=(1)


parallel --delay=$DELAY --linebuffer --jobs=$JOBS $PYTHON train_by_cmd.py --model_name={1} --encoder_hidden_size={2} \
--arg_comp_hidden_size={3} --lr={4} --max_epochs={5} --optimizer={6} --lr_scheduler={7} --train_batch_size={8} --weight_decay={9} \
--precision={10} --seed={11} --strategy={12} --gpus={13} --exp_name={14} --run={15} --aug={16} --aug_prob={17} --dropout={18} \
--input_size={19} --loss={20} \
  ::: ${MODEL_NAME[@]} ::: ${ENCODER_HIDDEN_SIZE[@]} ::: ${ARG_COMP_HIDDEN_SIZE[@]} ::: ${LR[@]} ::: ${MAX_EPOCHS[@]} \
  ::: ${OPTIMIZER[@]} ::: ${LR_SCHEDULER[@]} ::: ${TRAIN_BATCH_SIZE[@]} ::: ${WEIGHT_DECAY[@]} \
  ::: ${PRECISION[@]} ::: ${SEED[@]} ::: ${STRATEGY} ::: ${GPUS[@]} ::: ${EXP_NAME[@]} ::: ${RUN[@]} ::: ${AUG} ::: ${AUG_PROB[@]} \
  ::: ${DROPOUT[@]} ::: ${INPUT_SIZE[@]} ::: ${LOSS[@]}

