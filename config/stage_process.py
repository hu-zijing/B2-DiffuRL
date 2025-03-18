import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    ###### General ######
    # exp_name
    config.exp_name = "test1"
    # run name for wandb logging and checkpoint saving -- if not provided, will be auto-generated based on the datetime.
    config.run_name = "stage0"
    # random seed for reproducibility.
    config.seed = 42
    # top-level logging directory for checkpoint saving.
    config.save_path = "./model/lora"
    # mixed precision training. options are "fp16", "bf16", and "no". half-precision speeds up training significantly.
    config.mixed_precision = "fp16" # "fp16"
    # allow tf32 on Ampere GPUs, which can speed up training.
    config.allow_tf32 = True # False # True
    # whether or not to use LoRA. LoRA reduces memory usage significantly by injecting small weight matrices into the
    # attention layers of the UNet. with LoRA, fp16, and a batch size of 1, finetuning Stable Diffusion should take
    # about 10GB of GPU memory. beware that if LoRA is disabled, training will take a lot of memory and saved checkpoint
    # files will also be large.
    config.use_lora = True
    # resume training from a checkpoint. either an exact checkpoint directory (e.g. checkpoint_50), or a directory
    # containing checkpoints, in which case the latest one will be used. `config.use_lora` must be set to the same value
    # as the run that generated the saved checkpoint.
    config.resume_from = ""
    # dev id
    config.dev_id = 0
    # split_step
    config.split_step = 14
    # split_time
    config.split_time = 3
    # prompt directly used
    config.prompt = "" # "a dolphin riding a bike"
    # prompt file
    config.prompt_file = "config/prompt/template1_train.json"
    # randomly choose prompt or equally
    config.prompt_random_choose = 1

    ###### Pretrained Model ######
    config.pretrained = pretrained = ml_collections.ConfigDict()
    # base model to load. either a path to a local directory, or a model name from the HuggingFace model hub.
    # pretrained.model = "stablediffusionapi/anything-v5"
    pretrained.model = "./model/stablediffusion/sdv1.4"
    # revision of the model to load.
    pretrained.revision = "main"

    ###### Sampling ######
    config.sample = sample = ml_collections.ConfigDict()
    # # number of sampler inference steps.
    sample.num_steps = 20
    # whether use classifier-free guidance
    sample.cfg = True
    # classifier-free guidance weight. 1.0 is no guidance.
    sample.guidance_scale = 5.0
    # weight of x_0
    sample.beta = 1.0
    # eta parameter for the DDIM sampler. this controls the amount of noise injected into the sampling process, with 0.0
    # being fully deterministic and 1.0 being equivalent to the DDPM sampler.
    sample.eta = 1.0
    # batch size
    sample.batch_size = 8
    # number of batches to sample per epoch. the total number of samples per epoch is `num_batches_per_epoch *
    # batch_size * num_gpus`.
    sample.num_batches_per_epoch = 10
    # save_interval
    sample.save_interval = 1

    ###### Training ######
    config.train = train = ml_collections.ConfigDict()
    # batch size (per GPU!) to use for training.
    train.batch_size = 2
    # whether to use the 8bit Adam optimizer from bitsandbytes.
    train.use_8bit_adam = False
    # learning rate.
    train.learning_rate = 1e-4
    # Adam beta1.
    train.adam_beta1 = 0.9
    # Adam beta2.
    train.adam_beta2 = 0.999
    # Adam weight decay.
    train.adam_weight_decay = 1e-4
    # Adam epsilon.
    train.adam_epsilon = 1e-8
    # number of gradient accumulation steps. the effective batch size is `batch_size * num_gpus *
    # gradient_accumulation_steps`.
    train.gradient_accumulation_steps = 1
    # maximum gradient norm for gradient clipping.
    train.max_grad_norm = 1.0
    # number of inner epochs per outer epoch. each inner epoch is one iteration through the data collected during one
    # outer epoch's round of sampling.
    train.num_inner_epochs = 1
    # whether or not to use classifier-free guidance during training. if enabled, the same guidance scale used during
    # sampling will be used during training.
    train.cfg = True
    # the fraction of timesteps to train on. if set to less than 1.0, the model will be trained on a subset of the
    # timesteps for each sample. this will speed up training but reduce the accuracy of policy gradient estimates.
    train.timestep_fraction = 1.0
    # coefficient of p() in loss
    train.alpha = 1.0
    # coefficient of l2 regulizer in loss
    train.beta = 0
    # coefficient of pos sample
    train.beta1 = 1.0
    # coefficient of neg sample
    train.beta2 = 1.0
    # The coefficient constraining the probability ratio. Equivalent to restricting the Q-values within a certain range.
    train.eps = 0.1
    # training epoch
    train.num_epochs = 10
    # clip advantages to the range [-adv_clip_max, adv_clip_max].
    train.adv_clip_max = 5
    # checkpoints save interval
    train.save_interval = 5
    # number of checkpoints to keep before overwriting old ones.
    config.train.num_checkpoint_limit = 21

    ###### Evaluation & Selection ######
    config.eval = eval = ml_collections.ConfigDict()
    
    # select threshold
    eval.pos_threshold = 0.7
    eval.neg_threshold = -0.7

    # remain history cnt
    eval.history_cnt = 8

    return config
