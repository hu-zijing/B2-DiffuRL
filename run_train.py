import contextlib
import os
import copy
import datetime
import sys
script_path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(script_path)))
from absl import app, flags
from ml_collections import config_flags
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from diffusion.ddim_with_logprob import ddim_step_with_logprob
import torch
from functools import partial
import tqdm
import tree
import json
from utils.utils import load_sample_stage, seed_everything


tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/stage_process.py", "Training configuration.")

logger = get_logger(__name__)

def main(_):
    # basic Accelerate and logging setup
    config = FLAGS.config

    torch.cuda.set_device(config.dev_id)

    if config.exp_name:
        unique_id = config.exp_name
    else: 
        unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    os.makedirs(os.path.join(config.save_path, unique_id), exist_ok=True)

    if config.run_name:
        stage_id = config.run_name
    else: 
        stage_id = "stage"+str(os.listdir(os.path.join(config.save_path, unique_id)))
        
    save_dir = os.path.join(config.save_path, unique_id, stage_id)

    if config.resume_from:
        print("loading model. Please Wait.")
        config.resume_from = os.path.normpath(os.path.expanduser(config.resume_from))
        if "checkpoint_" not in os.path.basename(config.resume_from):
            # get the most recent checkpoint in this directory
            checkpoints = list(filter(lambda x: "checkpoint_" in x, os.listdir(config.resume_from)))
            if len(checkpoints) == 0:
                raise ValueError(f"No checkpoints found in {config.resume_from}")
            config.resume_from = os.path.join(
                config.resume_from,
                sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))[-1],
            )
        print("load successfully!")

    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)
    num_train_timesteps_2 = int(config.split_step * config.train.timestep_fraction)

    accelerator_config = ProjectConfiguration(
        project_dir=save_dir, # os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.train.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps * num_train_timesteps_2,
    )
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="rl-training", config=config.to_dict(), init_kwargs={"wandb": {"name": unique_id+"_"+stage_id}}
        )
    logger.info(f"\n{config}")

    seed_everything(config.seed)

    # load scheduler, tokenizer and models.
    pipeline = StableDiffusionPipeline.from_pretrained(config.pretrained.model, torch_dtype=torch.float16) ## float16
    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.unet.requires_grad_(not config.use_lora)
    # disable safety checker
    pipeline.safety_checker = None
    # make the progress bar nicer
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )
    # switch to DDIM scheduler
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to inference_dtype
    pipeline.vae.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    
    if config.use_lora:
        pipeline.unet.to(accelerator.device, dtype=inference_dtype)
    
        
    if config.use_lora:
        # Set correct lora layers
        lora_attn_procs = {}
        for name in pipeline.unet.attn_processors.keys():
            cross_attention_dim = (
                None if name.endswith("attn1.processor") else pipeline.unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = pipeline.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(pipeline.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = pipeline.unet.config.block_out_channels[block_id]

            lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
        pipeline.unet.set_attn_processor(lora_attn_procs)
        trainable_layers = AttnProcsLayers(pipeline.unet.attn_processors)
    else:
        trainable_layers = pipeline.unet

    # set up diffusers-friendly checkpoint saving with Accelerate

    def save_model_hook(models, weights, output_dir):
        assert len(models) == 1
        if config.use_lora and isinstance(models[0], AttnProcsLayers):
            pipeline.unet.save_attn_procs(output_dir)
        elif not config.use_lora and isinstance(models[0], UNet2DConditionModel):
            models[0].save_pretrained(os.path.join(output_dir, "unet"))
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        weights.pop()  # ensures that accelerate doesn't try to handle saving of the model

    def load_model_hook(models, input_dir):
        assert len(models) == 1
        if config.use_lora and isinstance(models[0], AttnProcsLayers):
            tmp_unet = UNet2DConditionModel.from_pretrained(
                config.pretrained.model, revision=config.pretrained.revision, subfolder="unet"
            )
            tmp_unet.load_attn_procs(input_dir)
            models[0].load_state_dict(AttnProcsLayers(tmp_unet.attn_processors).state_dict())
            del tmp_unet
        elif not config.use_lora and isinstance(models[0], UNet2DConditionModel):
            load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
            models[0].register_to_config(**load_model.config)
            models[0].load_state_dict(load_model.state_dict())
            del load_model
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        models.pop()  # ensures that accelerate doesn't try to handle loading of the model
    

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize the optimizer
    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        trainable_layers.parameters(),
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    # generate negative prompt embeddings
    neg_prompt_embed = pipeline.text_encoder(
        pipeline.tokenizer(
            [""],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)
    )[0]
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)

    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast

    # Prepare everything with our `accelerator`.
    trainable_layers, optimizer = accelerator.prepare(trainable_layers, optimizer)

    # Train!
    samples_per_epoch = config.sample.batch_size * accelerator.num_processes * config.sample.num_batches_per_epoch
    total_train_batch_size = (
        config.train.batch_size * accelerator.num_processes * config.train.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {config.train.num_epochs}")
    logger.info(f"  Sample batch size per device = {config.sample.batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
    logger.info(f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}")
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")

    assert config.sample.batch_size >= config.train.batch_size
    assert config.sample.batch_size % config.train.batch_size == 0
    # assert samples_per_epoch % total_train_batch_size == 0

    if config.resume_from:
        logger.info(f"Resuming from {config.resume_from}")
        accelerator.load_state(config.resume_from)
        
    # get sample dict
    samples = load_sample_stage(save_dir)

    accelerator.save_state()

    pipeline.scheduler.set_timesteps(config.sample.num_steps, device=accelerator.device)
    init_samples = copy.deepcopy(samples)
    LossRecord = []
    GradRecord = []
    for epoch in range(config.train.num_epochs):
        # shuffle samples along batch dimension
        samples = {}
        LossRecord.append([])
        GradRecord.append([])

        total_batch_size = init_samples["eval_scores"].shape[0]
        perm = torch.randperm(total_batch_size)
        samples = {k: v[perm] for k, v in init_samples.items()}

        perms = torch.stack( # v2["timesteps"].shape[1]
            [torch.randperm(init_samples["timesteps"].shape[1]) for _ in range(total_batch_size)]
        )
        for key in ["latents", "next_latents", "log_probs", "timesteps"]:
            samples[key] = samples[key][torch.arange(total_batch_size)[:, None], perms]
                    
        # training
        pipeline.unet.train()
        for idx in tqdm(range(0,total_batch_size//2*2,config.train.batch_size),
                    desc="Update",
                    position=2,
                    leave=False, 
                        ):

            LossRecord[epoch].append([])
            GradRecord[epoch].append([])

            sample = tree.map_structure(lambda value: value[idx:idx+config.train.batch_size].to(accelerator.device), samples)

            # cfg, classifier-free-guidance
            if config.train.cfg:
                embeds = torch.cat([train_neg_prompt_embeds, sample["prompt_embeds"]])
            else:
                embeds = sample["prompt_embeds"]
            
            for t in tqdm(
                range(sample["timesteps"].shape[1]),
                desc="Timestep",
                position=3,
                leave=False,
                disable=not accelerator.is_local_main_process,
            ):  
                
                evaluation_score = sample["eval_scores"][:]

                with accelerator.accumulate(pipeline.unet):
                    with autocast():
                        if config.train.cfg:
                            noise_pred = pipeline.unet(
                                torch.cat([sample["latents"][:, t]] * 2),
                                torch.cat([sample["timesteps"][:, t]] * 2),
                                embeds,
                            ).sample
                            
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + config.sample.guidance_scale * (noise_pred_text - noise_pred_uncond)

                        else:
                            noise_pred = pipeline.unet(
                                sample["latents"][:, t], sample["timesteps"][:, t], embeds
                            ).sample

                        _, total_prob, _ = ddim_step_with_logprob(
                            pipeline.scheduler,
                            noise_pred,
                            sample["timesteps"][:, t],
                            sample["latents"][:, t],
                            eta=config.sample.eta,
                            prev_sample=sample["next_latents"][:, t],
                        )
                        total_ref_prob = sample["log_probs"][:, t]

                ratio = torch.exp(total_prob - total_ref_prob)
                temp_beta1 = torch.ones_like(evaluation_score)*config.train.beta1
                temp_beta2 = torch.ones_like(evaluation_score)*config.train.beta2
                sample_weight = torch.where(evaluation_score>0, temp_beta1, temp_beta2)
                advantages = torch.clamp(
                            evaluation_score,
                            -config.train.adv_clip_max,
                            config.train.adv_clip_max,
                        )*sample_weight
                unclipped_loss = -advantages * ratio
                clipped_loss = -advantages * torch.clamp(
                    ratio,
                    1.0 - config.train.eps,
                    1.0 + config.train.eps,
                )
                loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))

                accelerator.backward(loss)
                total_norm = None
                if accelerator.sync_gradients:
                    total_norm = accelerator.clip_grad_norm_(trainable_layers.parameters(), config.train.max_grad_norm)
                LossRecord[epoch][idx//config.train.batch_size].append(loss.cpu().item())
                GradRecord[epoch][idx//config.train.batch_size].append(total_norm.cpu().item() if total_norm is not None else None)
                optimizer.step()
                optimizer.zero_grad()
                
        if (epoch+1) % config.train.save_interval == 0 :
            accelerator.save_state()

    os.makedirs(os.path.join(save_dir, 'eval'), exist_ok=True)
    with open(os.path.join(save_dir, 'eval', 'loss.json'),'w') as f:
        json.dump(LossRecord, f)
    with open(os.path.join(save_dir, 'eval', 'grad.json'),'w') as f:
        json.dump(GradRecord, f)


if __name__ == "__main__":
    app.run(main)
