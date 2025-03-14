import os
import sys
script_path = os.path.abspath(__file__)
sys.path.append(os.path.dirname(os.path.dirname(script_path)))
from absl import app, flags
from ml_collections import config_flags
import torch
from functools import partial
import tqdm
import pickle
import tqdm
import json
from bert_score import score, BERTScorer
from PIL import Image
import open_clip
import numpy as np
from utils.utils import seed_everything

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/stage_process.py", "Sampling configuration.")

def score_fn1(ground, img_dir, save_dir, config): 
    unique_id = config.exp_name

    device = f"cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2B-s32B-b79K')
    tokenizer = open_clip.get_tokenizer('ViT-H-14')
    model = model.to(device)

    eval_list = sorted(os.listdir(img_dir))

    similarity = []
    maximum_onetime = 8
    for i in range(0, len(eval_list), maximum_onetime): 
        image_input = torch.tensor(np.stack([preprocess(Image.open(os.path.join(img_dir, image))).numpy() for image in eval_list[i:i+maximum_onetime]])).to(device)
        text_inputs = tokenizer(ground[i:i+maximum_onetime]).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity.append( (image_features @ text_features.T)[torch.arange(maximum_onetime), torch.arange(maximum_onetime)] )
    similarity = torch.cat(similarity)

    R = similarity.cpu().detach()
    print(R[:10])

    os.makedirs(os.path.join(save_dir, 'eval'), exist_ok=True)
    with open(os.path.join(save_dir, 'eval', 'scores.pkl'),'wb') as f:
        pickle.dump(R, f)

    each_score = {}
    for idx, prompt in enumerate(ground): 
        if prompt in each_score:
            each_score[prompt].append(R[idx:idx+1])
        else: 
            each_score[prompt] = [R[idx:idx+1]]

    history_data = []
    if config.eval.history_cnt > 0:
        if os.path.exists(os.path.join(config.save_path, unique_id, 'history_scores.pkl')):
            with open(os.path.join(config.save_path, unique_id, 'history_scores.pkl'), 'rb') as f:
                history_data = pickle.load(f)
        if len(history_data) > config.eval.history_cnt:
            history_data = history_data[-config.eval.history_cnt:]
    data_mean = {}
    data_std = {}
    cur_data = {}
    combine_data = {}
    for k,v in each_score.items():
        cur_data[k] = torch.cat(v, axis=0)
        combine_data[k] = torch.cat([d[k] for d in history_data if k in d]+[cur_data[k]], axis=0)
        data_mean[k] = combine_data[k].mean().item()
        data_std[k] = combine_data[k].std().item()
    history_data.append(cur_data)
    if len(history_data) > config.eval.history_cnt:
        history_data = history_data[-config.eval.history_cnt:]
    print("==== history_scores ====")
    for k,v in combine_data.items():
        print(k, v.shape)
    print('history data length', len(history_data))
    with open(os.path.join(config.save_path, unique_id, 'history_scores.pkl'), 'wb') as f:
        pickle.dump(history_data, f)

    print(data_mean)

    sum_scores = [(s-data_mean[ground[idx]])/(data_std[ground[idx]]+1e-8) for idx, s in enumerate(R)]

    # print(sum_scores)
    sum_scores = torch.tensor(sum_scores) # , dtype=torch.float16)

    return sum_scores

def main(_):
    # basic Accelerate and logging setup
    config = FLAGS.config

    torch.cuda.set_device(config.dev_id)
    seed_everything(config.seed)

    unique_id = config.exp_name

    if config.run_name:
        stage_id = config.run_name
    else: 
        stage_id = "stage"+str(os.listdir(os.path.join(config.save_path, unique_id))-1)
        
    save_dir = os.path.join(config.save_path, unique_id, stage_id)

    with open(os.path.join(save_dir, 'sample.pkl'), 'rb') as f:
        samples = pickle.load(f)
    with open(os.path.join(save_dir, 'prompt.json'), 'r') as f: 
        ground = json.load(f)

    ## evaluation
    img_dir = os.path.join(save_dir, 'images')
    eval_scores = score_fn1(ground, img_dir, save_dir, config)
    samples['eval_scores'] = eval_scores

    def get_new_unit():
        return {
            'prompt_embeds': [], 
            'timesteps': [], 
            'log_probs': [], 
            'latents': [], 
            'next_latents': [], 
            'eval_scores': []
        }
    data = get_new_unit()

    total_batch_size = samples['eval_scores'].shape[0]
    data_size = total_batch_size // config.sample.batch_size
    for b in range(config.sample.batch_size): 
        cur_sample_num = 1 
        batch_samples = {k:v[torch.arange(b, total_batch_size, config.sample.batch_size)] for k,v in samples.items()}
        
        t_left = config.sample.num_steps - config.split_step
        t_right = config.sample.num_steps

        prompt_embeds = batch_samples['prompt_embeds'][torch.arange(0, data_size, cur_sample_num)]
        timesteps = batch_samples['timesteps'][torch.arange(0, data_size, cur_sample_num), t_left:t_right]
        log_probs = batch_samples['log_probs'][torch.arange(0, data_size, cur_sample_num), t_left:t_right]
        latents = batch_samples['latents'][torch.arange(0, data_size, cur_sample_num), t_left:t_right]
        next_latents = batch_samples['next_latents'][torch.arange(0, data_size, cur_sample_num), t_left:t_right]

        score = batch_samples['eval_scores'][torch.arange(0, data_size, cur_sample_num)]
        score = score.reshape(-1, config.split_time)
        max_idx = score.argmax(dim=1)
        min_idx = score.argmin(dim=1)
        for j, s in enumerate(score): 
            for p_n in range(2):
                if p_n==0 and s[max_idx[j]] >= config.eval.pos_threshold: 
                    used_idx = max_idx[j]
                    used_idx_2 = j*config.split_time+max_idx[j]
                elif p_n==1 and s[min_idx[j]] < config.eval.neg_threshold: 
                    used_idx = min_idx[j]
                    used_idx_2 = j*config.split_time+min_idx[j]
                else: 
                    continue

                data['prompt_embeds'].append(prompt_embeds[used_idx_2]) # j*split_times[i-1]
                data['timesteps'].append(timesteps[used_idx_2]) # j*split_times[i-1]

                data['log_probs'].append(log_probs[used_idx_2])
                data['latents'].append(latents[used_idx_2])
                data['next_latents'].append(next_latents[used_idx_2])
                data['eval_scores'].append(s[used_idx])

        cur_sample_num *= config.split_time

    if len(data.keys()) != 0:
        data = {k:torch.stack(v, dim=0) for k,v in data.items()}

    print("train_data.shape:")
    for k,v in data.items():
        print(f"{k}:", v.shape)

    with open(os.path.join(save_dir, 'sample_stage.pkl'), 'wb') as f:
        pickle.dump(data, f)

if __name__ == "__main__":
    app.run(main)
