set -e

SaveInterval=2
SavePath="./model/lora"
PromptFile="config/prompt/template1_train.json"
RandomPrompt=1
ExpName="exp_B2DiffuRL_b5_p3"
Seed=300
Beta1=1
Beta2=1
BatchCnt=32
StageCnt=100
SplitStepLeft=14
SplitStepRight=20
TrainEpoch=2
AccStep=64
LR=0.0001
ModelVersion="sdv1.4"
NumStep=20
History_Cnt=8
PosThreshold=0.5
NegThreshold=-0.5
SplitTime=5
Dev_Id=0

CUDA_FALGS="--config.dev_id ${Dev_Id}"
SAMPLE_FLAGS="--config.sample.num_batches_per_epoch ${BatchCnt} --config.sample.num_steps ${NumStep} --config.prompt_file ${PromptFile} --config.prompt_random_choose ${RandomPrompt} --config.split_time ${SplitTime}" # 
EXP_FLAGS="--config.exp_name ${ExpName} --config.save_path ${SavePath} --config.pretrained.model ${ModelVersion}"


for i in $(seq 0 $((StageCnt-1)))
do
    interval=$((SplitStepRight-SplitStepLeft+1))
    level=$((i*interval/StageCnt))
    cur_split_step=$((level+SplitStepLeft))

    RUN_FLAGS="--config.run_name stage${i} --config.split_step ${cur_split_step} --config.eval.history_cnt ${History_Cnt} --config.eval.pos_threshold ${PosThreshold} --config.eval.neg_threshold ${NegThreshold}"
    temp_seed=$((Seed+i))
    RANDOM_FLAGS="--config.seed ${temp_seed}"
    TRAIN_FLAGS="--config.train.save_interval ${SaveInterval} --config.train.num_epochs ${TrainEpoch} --config.train.beta1 ${Beta1} --config.train.beta2 ${Beta2} --config.train.gradient_accumulation_steps ${AccStep} --config.train.learning_rate ${LR}"
    LORA_FLAGS=""
    if [ $i != 0 ]; then
        minus_i=$((i-1))
        cur_epoch=${TrainEpoch}
        checkpoint=$((cur_epoch/SaveInterval))
        LORA_FLAGS="--config.resume_from ${SavePath}/${ExpName}/stage${minus_i}/checkpoints/checkpoint_${checkpoint}"
    fi

    echo "||=========== round: ${i} ===========||"
    echo $CUDA_FALGS
    echo $TRAIN_FLAGS
    echo $SAMPLE_FLAGS
    echo $RANDOM_FLAGS
    echo $EXP_FLAGS
    echo $RUN_FLAGS
    echo $LORA_FLAGS

    python3 run_sample.py $CUDA_FALGS $TRAIN_FLAGS $SAMPLE_FLAGS $RANDOM_FLAGS $EXP_FLAGS $RUN_FLAGS $LORA_FLAGS $SELECT_FLAGS
    python3 run_select.py $CUDA_FALGS $TRAIN_FLAGS $SAMPLE_FLAGS $RANDOM_FLAGS $EXP_FLAGS $RUN_FLAGS $LORA_FLAGS $SELECT_FLAGS
    python3 run_train.py $CUDA_FALGS $TRAIN_FLAGS $SAMPLE_FLAGS $RANDOM_FLAGS $EXP_FLAGS $RUN_FLAGS $LORA_FLAGS $SELECT_FLAGS

    sleep 2
done

