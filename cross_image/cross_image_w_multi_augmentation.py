#%%
import sys
import os

# Ensure the repo root is in sys.path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import argparse
import importlib
import json
import os
import random
import more_itertools
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict, deque
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Tuple
from torchvision import transforms
from utils.attack_tool import (
    add_extra_args, find_next_run_dir, get_available_gpus, get_img_id_train_prompt_map,
    get_intended_token_ids, get_subset, load_datasets, load_model, seed_everything
)
from utils.eval_model import BaseEvalModel
from utils.eval_tools import (
    get_eval_icl, load_icl_example, get_vqa_type,
    cap_instruction, cls_instruction, load_img_specific_questions, vqa_agnostic_instruction, 
    plot_loss, postprocess_generation,record_format_summary, record_format_summary_affect
)
from PIL import Image

def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

def multi_augmentation(x: torch.Tensor, num_scale: int) -> torch.Tensor:
    """
    Apply multiple augmentation variants on an image tensor.

    Args:
        x: Input image tensor shaped like [B, C, H, W].
        num_scale: Number of augmented copies to produce.

    Returns:
        Concatenated augmented image tensor shaped like [B * num_scale, C, H, W].
    """
    if num_scale <= 0:
        return x

    if x.dim() < 4:
        raise ValueError(
            f"multi_augmentation expects tensor with at least 4 dims (B, C, H, W), got {x.shape}"
        )

    original_shape = tuple(x.shape)
    spatial_shape = tuple(x.shape[-3:])
    flattened_base = x.reshape(-1, *spatial_shape)
    augmented_images = []

    def reshape_like_original(tensor_4d: torch.Tensor) -> torch.Tensor:
        return tensor_4d.reshape(*original_shape)

    def augment_diverse_input(img: torch.Tensor) -> torch.Tensor:
        if "DI" not in globals():
            return img
        return DI(img, prob=0.7)

    def augment_rdi(img: torch.Tensor) -> torch.Tensor:
        if "RDI_" not in globals():
            return img
        return RDI_(img)

    def augment_rotation(img: torch.Tensor) -> torch.Tensor:
        angle = torch.randint(-15, 16, (1,), device=img.device).item()
        return transforms.functional.rotate(
            img,
            angle,
            interpolation=transforms.InterpolationMode.BILINEAR,
        )

    def augment_horizontal_flip(img: torch.Tensor) -> torch.Tensor:
        return torch.flip(img, dims=[-1])

    def augment_vertical_flip(img: torch.Tensor) -> torch.Tensor:
        return torch.flip(img, dims=[-2])

    def augment_translation(img: torch.Tensor) -> torch.Tensor:
        shift_x = torch.randint(-10, 11, (1,), device=img.device).item()
        shift_y = torch.randint(-10, 11, (1,), device=img.device).item()
        return torch.roll(img, shifts=(shift_y, shift_x), dims=(-2, -1))

    def _scale_crop(img: torch.Tensor, low: float, high: float) -> torch.Tensor:
        scale = low + (high - low) * torch.rand(1, device=img.device).item()
        img_size = img.shape[-1]
        new_size = max(1, int(img_size * scale))
        scaled = F.interpolate(img, size=(new_size, new_size), mode="bilinear", align_corners=False)
        if new_size > img_size:
            start = (new_size - img_size) // 2
            return scaled[:, :, start:start + img_size, start:start + img_size]

        pad = (img_size - new_size) // 2
        pad_max = img_size - new_size
        pad_right = pad_max - pad
        pad_bottom = pad_max - pad
        return F.pad(scaled, (pad, pad_right, pad, pad_bottom), mode="constant", value=0)

    def augment_scale_crop_1(img: torch.Tensor) -> torch.Tensor:
        return _scale_crop(img, 0.8, 1.0)

    def augment_scale_crop_2(img: torch.Tensor) -> torch.Tensor:
        return _scale_crop(img, 1.0, 1.2)

    augmentation_methods = [
        augment_rotation,
        augment_horizontal_flip,
        augment_vertical_flip,
        augment_translation,
        augment_scale_crop_1,
        augment_scale_crop_2,
    ]
    if "DI" in globals():
        augmentation_methods.append(augment_diverse_input)
    if "RDI_" in globals():
        augmentation_methods.append(augment_rdi)

    for _ in range(num_scale):
        x_aug = flattened_base.clone()
        for _ in range(2):
            aug_method = augmentation_methods[torch.randint(0, len(augmentation_methods), (1,)).item()]
            try:
                x_aug = aug_method(x_aug)
            except Exception as exc:
                print(aug_method)
                print(f"Augmentation failed: {exc}")
                x_aug = flattened_base.clone()
        augmented_images.append(reshape_like_original(x_aug))

    return torch.cat(augmented_images, dim=0)

def attack(
    args: argparse.Namespace,
    eval_model: BaseEvalModel,
    max_generation_length: int = 5,
    num_beams: int = 3,
    length_penalty: float = -2.0,
    num_shots: int = 2,
    alpha1: float = 1/255,
    epsilon: float = 32/255,
    iters: int = 200,
    alpha2: float = 0.01,
    fraction: float = 0.01,
    target: str = "unknown<|endofchunk|>",
    base_dir: str = "./",
    prompt_num: int = 1,
    datasets: Tuple[Dataset, Dataset] = None,
):
    model_name = args.model_name
    method = args.method
    
    save_perturb_iterations = sorted(set(list(range(600, iters, 100)) + [iters - 1])) if iters > 0 else []
    cropa_end = 300
    step = max((cropa_end//prompt_num),1)
    cropa_iter = [i for i in range(step,cropa_end+1, step)] # text perturb update iterations 
        
    tokenizer = eval_model.tokenizer
    target = target.lower().strip().replace("_", " ")
    target_token_len = len(tokenizer.encode(target)) - 1
    print("target_token_len is:",target_token_len)

    train_dataset, test_dataset = datasets if datasets is not None else load_datasets(args = args)
    train_batch_demo_samples,test_batch_demo_samples = load_icl_example(train_dataset)
    test_dataset = get_subset(dataset=test_dataset,frac=fraction)
            
    # Create a unique directory based on current running id to avoid overwriting
    output_dir = f"frac_{fraction}"
    output_dir = os.path.join(base_dir,output_dir)
    output_dir = find_next_run_dir(output_dir)
    os.makedirs(output_dir,exist_ok=True)     
    
    with open(args.vqav2_eval_annotations_json_path, "r") as f:
        eval_file =  json.load(f)
    annos = eval_file["annotations"]
    ques_id_to_img_id = {i["question_id"]:i["image_id"] for i in annos}    
    
    assert prompt_num >= 0, "require at least one question"
    img_id_to_train_prompt = get_img_id_train_prompt_map(prompt_num)
    
    total_vqa_success_rate = []
    total_cls_success_rate = []
    total_cap_success_rate = []
    result_json = defaultdict(lambda: defaultdict(list))
    task_list = ["vqa","vqa_specific","cls","cap"]
    
    vqa_specific_instruction = load_img_specific_questions() 
    with open("data/clean_train_vqa_map.json") as f:
        clean_vqa_model_output = json.load(f)
    
    access_order_dict = dict()
    perturb_list_dict = dict()
    noise_dict = dict()

    if model_name in ["blip2","instructblip"]:
        noise = torch.randn([1,3,224,224],requires_grad=True,device = device)
        lm_emb = eval_model.model.language_model.get_input_embeddings()
    else:
        noise = torch.randn([1,1,3,224,224],requires_grad=True,device = device)
        lm_emb = eval_model.model.lang_encoder.get_input_embeddings()
    #aug_test_set = AugmentedCroPADataset(test_dataset)
    
    for ep in tqdm(range(iters), desc = "Epoch"):
        for id, item in enumerate(test_dataset):
            img_id = str(ques_id_to_img_id[item["question_id"]])
            
            item_images = []
            item_text = []    
            total_prompt_list  = img_id_to_train_prompt[img_id]
            
            if num_shots > 0:
                print("batch_demo_samples is:",train_batch_demo_samples)
                context_images = [x["image"] for x in train_batch_demo_samples]
            else:
                context_images = []
            item_images.append(context_images + [item["image"]])
            
            if num_shots > 0:
                test_item_images = [[x["image"] for x in test_batch_demo_samples]+[item_images[0][-1]]]
            else:
                test_item_images = [[item_images[0][-1]]]
            
            train_context_text = "".join([
                    eval_model.get_vqa_prompt(question=x["question"],answer=x["answers"][0])
                    for x in train_batch_demo_samples
            ])
            if num_shots == 0:
                train_context_text = train_context_text.replace("<image>", "")
            if model_name in ["blip2","instructblip"]:
                train_context_text=""

            for ques in total_prompt_list:
                item_text.append(train_context_text + eval_model.get_vqa_prompt(question=ques)+" "+target)
                
            labels_list = []
            input_ids_list = []
            context_token_len_list = []
            attention_mask_list = []
            target_token_len_list = []
            qformer_input_ids_list = []
            qformer_attention_mask_list = []
            target_encodings = tokenizer.encode(target,return_tensors="pt")
            
            for ques_text in item_text:
                input_encodings = tokenizer(
                        ques_text,padding="longest",
                        truncation=True,return_tensors="pt",max_length=2000)
                context_token_len = len(tokenizer.encode(train_context_text))
                context_token_len_list.append(context_token_len)
                input_ids = input_encodings["input_ids"].to(device)
                attention_mask = input_encodings["attention_mask"].to(device)
                
                if not target.startswith("no target"):
                    target_id = tokenizer.encode(target)[1:]           
                    labels= get_intended_token_ids(input_ids,target_id)
                else:
                    original_ques_text = ques_text.split("<image>Question:")[-1].split(" Short")[0]
                    target_text = clean_vqa_model_output[img_id][original_ques_text][1]
                    #print("target_text is:",target_text)
                    target_id = tokenizer.encode(target_text)[1:]          
                    labels= get_intended_token_ids(input_ids,target_id)
                    
                labels_list.append(labels)
                input_ids_list.append(input_ids)
                attention_mask_list.append(attention_mask)
                target_token_len_list.append(len(target_id))
                if model_name=="instructblip":
                    qformer_text_encoding = eval_model.qformer_tokenizer(ques_text,padding="longest",
                                            truncation=True,return_tensors="pt",max_length=2000).to(device)
                    qformer_input_ids_list.append(qformer_text_encoding["input_ids"])
                    qformer_attention_mask_list.append(qformer_text_encoding["attention_mask"])
                
            input_x_original = eval_model._prepare_images_no_normalize(item_images).to(device)

            if img_id in perturb_list_dict:
                perturb_list = perturb_list_dict[img_id]
            else:
                perturb_list = []
                for i in input_ids_list:
                    perturb = torch.zeros_like(lm_emb(i),device="cpu",requires_grad=True)
                    perturb_list.append(perturb)
            
            if img_id in access_order_dict:
                access_order = access_order_dict[img_id]
            else:
                if ep%prompt_num == 0:
                    access_order = list(range(prompt_num))
                    random.shuffle(access_order)
                access_order_dict[img_id] = access_order

            text_idx = access_order[ep%prompt_num]
            context_token_len = context_token_len_list[text_idx]
            input_x = input_x_original.clone().detach()

            if model_name=="open_flamingo":
                input_x[0,-1] = input_x[0,-1] + noise
            elif model_name in ["instructblip","blip2"]:
                input_x = input_x + noise
            labels = labels_list[text_idx]
            input_ids = input_ids_list[text_idx]
            attention_mask = attention_mask_list[text_idx]
            qformer_input_ids = None
            qformer_attention_mask = None
            if model_name == "instructblip":
                qformer_input_ids = qformer_input_ids_list[text_idx]
                qformer_attention_mask = qformer_attention_mask_list[text_idx]

            if args.num_scale > 0:
                input_x = multi_augmentation(input_x, args.num_scale)
                repeat_factor = input_x.shape[0]
                input_ids = input_ids.repeat(repeat_factor, 1)
                attention_mask = attention_mask.repeat(repeat_factor, 1)
                labels = labels.repeat(repeat_factor, 1)
                if model_name == "instructblip":
                    qformer_input_ids = qformer_input_ids.repeat(repeat_factor, 1)
                    qformer_attention_mask = qformer_attention_mask.repeat(repeat_factor, 1)
            
            inputs_embeds_original = lm_emb(input_ids).clone().detach()
            supports_text_perturb = model_name not in ["blip2"]
            text_perturb = (
                torch.tensor(perturb_list[text_idx], requires_grad=True, device=device)
                if supports_text_perturb
                else None
            )

            inputs_embeds = inputs_embeds_original + text_perturb if text_perturb is not None else None
            if method == "baseline" or not supports_text_perturb:
                inputs_embeds = None
            if model_name=="open_flamingo":
                loss = eval_model.model(  
                    inputs_embeds=inputs_embeds,
                    lang_x=input_ids,
                    vision_x=input_x,                
                    attention_mask=attention_mask,
                    labels=labels
                )[0]
            elif model_name=="blip2":
                loss = eval_model.model(  
                    input_ids=input_ids,
                    pixel_values=input_x,                
                    attention_mask=attention_mask,
                    labels=labels
                )[0]
            elif model_name=="instructblip":
                loss = eval_model.model(  
                    inputs_embeds=inputs_embeds,
                    input_ids=input_ids,
                    pixel_values=input_x,                
                    attention_mask=attention_mask,
                    labels=labels,
                    normalize_vision_input = True,
                    qformer_input_ids = qformer_input_ids,
                    qformer_attention_mask= qformer_attention_mask
                )[0]
            loss.backward()
            
            grad = noise.grad.detach()
            if method!="baseline" and text_perturb is not None and text_perturb.grad is not None:
                text_grad = text_perturb.grad.detach()
                mask = torch.ones_like(inputs_embeds)
                mask[:,:context_token_len] = 0
                mask[:,-target_token_len_list[text_idx]:] = 0
            
            if not target.startswith("no target"):
                d = torch.clamp(noise - alpha1 * torch.sign(grad), min=-epsilon, max=epsilon)                
                if method=="cropa" and ep in cropa_iter and text_perturb is not None:
                    text_perturb.data = torch.clamp(text_perturb+ mask*torch.sign(text_grad)*alpha2,min = -0.23,max = 0.27)                    
            else: 
                d = torch.clamp(noise + alpha1 * torch.sign(grad), min=-epsilon, max=epsilon)
                if method=="cropa" and ep in cropa_iter and text_perturb is not None:
                    text_perturb.data = torch.clamp(text_perturb - mask*torch.sign(text_grad)*alpha2,min = -0.23,max = 0.27)                    
            noise.data = d
            noise.grad.zero_()

            if method!="baseline" and text_perturb is not None and text_perturb.grad is not None:
                text_perturb.grad.zero_()
                perturb_list[text_idx] = text_perturb.clone().detach().cpu()
            perturb_list_dict[img_id] = perturb_list

        noise_dict[ep] = noise.clone().detach()
        os.makedirs(f"{output_dir}/{ep}",exist_ok=True)
        np.save(f"{output_dir}/{ep}/noise.npy", noise_dict[ep].cpu().numpy())
        if ep not in save_perturb_iterations:
            continue
        # REPLACE WITH TRAIN_DATASET IF OOD DATA IS TO BE USED
        # You may add id < 50 if train_dataset is more than 50 in size
        for id, item in enumerate(test_dataset):
            img_id = str(ques_id_to_img_id[item["question_id"]])
            attack = torch.tensor(noise_dict[ep],requires_grad=True,device=device)
            vqa_sample = vqa_agnostic_instruction()
            vqa_specific_sample = vqa_specific_instruction[img_id]
            prompt_list = [vqa_sample,vqa_specific_sample[:10],cls_instruction(),cap_instruction()]
            vqa_stats  = {"number":{"success":0,"total":0},
                        "yes_no":{"success":0,"total":0},
                        "what":{"success":0,"total":0},
                        "where":{"success":0,"total":0},
                        "other":{"success":0,"total":0}}
            
            template_list = [eval_model.get_vqa_prompt,eval_model.get_vqa_prompt,eval_model.get_classification_prompt,eval_model.get_caption_prompt]
            result_list = [total_vqa_success_rate,total_cls_success_rate,total_cap_success_rate]
            item_images = []
            item_images.append([item["image"]])
            test_item_images = [[item_images[0][-1]]]
            for i in range(len(prompt_list)):
                task_name = task_list[i]
                instruction_list = prompt_list[i]
                template_func = template_list[i]
                success_count = 0
                target_success_count = 0
                test_context_text = get_eval_icl(task_name,num_shots, test_batch_demo_samples,eval_model)
                for batch_ques in more_itertools.chunked(instruction_list,args.eval_batch_size):
                    if task_name == "vqa" or task_name=="vqa_specific":
                        eval_text = [test_context_text+template_func(ques) for ques in batch_ques]
                    else:           
                        eval_text = [test_context_text+"<image>"+instruction+" Output:" for instruction in batch_ques]
                    if model_name in ["blip2","instructblip"]:
                        eval_text = ["Context:"+instruction+" Answer:" for instruction in batch_ques]
                    if model_name=="instructblip":
                        test_item_images=item_images[0]
                    outputs = eval_model.get_outputs_attack(
                                            attack = attack,batch_images=test_item_images*len(batch_ques),
                                            batch_text=eval_text,max_generation_length=max_generation_length,
                                            num_beams=num_beams,length_penalty=length_penalty)                        
                    if not args.quick_eval:
                        clean_outputs = eval_model.get_outputs(
                                            batch_images=test_item_images*len(batch_ques),
                                            batch_text=eval_text,max_generation_length=max_generation_length,
                                            num_beams=num_beams,length_penalty=length_penalty)
                    process_function = postprocess_generation
                    new_predictions = list(map(process_function, outputs))
                    clean_newpredictions = list(map(process_function, clean_outputs)) if not args.quick_eval else None
                    for i in range(len(new_predictions)):
                        target_attack_is_success = False
                        if clean_newpredictions is not None and new_predictions[i]!=clean_newpredictions[i]:
                            success_count+=1    
                        if new_predictions[i].strip().lower() ==target.lower().split("<")[0].strip():
                            target_success_count+=1 
                            target_attack_is_success = True
                        if task_name == "vqa" or task_name=="vqa_specific":
                            prompt_type = get_vqa_type(batch_ques[i])
                            if target_attack_is_success:vqa_stats[prompt_type]["success"]+=1 
                            vqa_stats[prompt_type]["total"]+=1
                with open(f"{output_dir}/results_{ep}.txt","a") as f:
                    f.write(f"{task_name}, target num: {target_success_count}, total attack num: {success_count}\n")
                result_json[ep][task_name].append({"count":success_count,"target_count":target_success_count})            
            for i in vqa_stats.keys():
                result_json[ep][i].append(vqa_stats[i])
    
    vqa_stats  = {"number":{"success":0,"total":0},
                    "yes_no":{"success":0,"total":0},
                    "what":{"success":0,"total":0},
                    "where":{"success":0,"total":0},
                    "other":{"success":0,"total":0}}
    result_summary = {}
    for ep in save_perturb_iterations:
        mean_res = {}    
        for t_name in task_list:
            mean_success_count = np.mean([i["count"] for i in result_json[ep][t_name]]) if not args.quick_eval else -1
            mean_target_success_count = np.mean([i["target_count"] for i in  result_json[ep][t_name]])
            if t_name == "vqa" or t_name == "vqa_specific"  :
                rate = "{:.4f}".format(mean_target_success_count/10)
                affect_rate = "{:.4f}".format(mean_success_count/10) 
            else:
                rate = "{:.4f}".format(mean_target_success_count/20)
                affect_rate = "{:.4f}".format(mean_success_count/20)             
            mean_res[t_name]={
                "target_rate":rate,
                "mean_count":mean_success_count,
                "mean_target_count":mean_target_success_count,
                "mean_affect_rate":affect_rate
            }
        mean_vqa_stats = {}
        for i in vqa_stats.keys():
            splited_success = np.mean([i["success"] for i in result_json[ep][i]])
            mean_total_num = np.mean([i["total"] for i in result_json[ep][i]])
            mean_vqa_stats[i] = {"mean_success":splited_success,
                                "mean_total_num":mean_total_num,
                                "mean_success_rate":"{:.2f}".format(splited_success/mean_total_num)}
        result_json["avg"] = mean_res
        result_json["vqa_stats"] = mean_vqa_stats
        result_summary[ep] ={"avg":mean_res, "vqa_stats":mean_vqa_stats}
        json.dump(result_json,open(f"{output_dir}/total_success_rate_{ep}.json","w"))
    json.dump(result_summary,open(f"{output_dir}/summary.json","w"))
    record_format_summary(result_summary,output_dir) 
    if not args.quick_eval:
        record_format_summary_affect(result_summary,output_dir) 
 
if __name__=="__main__":
    seed_everything(42)
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--prompt_num", type=int, default=10,
                        help="The number of prompts utilized during the optimization phase")
    parser.add_argument("--device", type=int, default=-1,
                        help="The device id of the GPU to use")
    parser.add_argument("--iter_num", "--iters", dest="iter_num", type=int, default=300,
                        help="The num of attack iterations")
    parser.add_argument("--model_name", type=str, default="instructblip", #before: instructblip
                        help="The num of attack iter")
    parser.add_argument("--quick_eval", type=str2bool, default=False,
                        help="set to false to generate the result given clean images")
    parser.add_argument("--fraction", type=float, default=0.05,
                        help="The fraction of the test dataset to use")
    parser.add_argument("--shot", type=int, default=0,
                        help="The num of in context learning examples to use, specific for Flamingo")
    parser.add_argument("--method", type=str, default="cropa",
                        help="The mehod of attack, either cropa or baseline")
    parser.add_argument("--target", type=str, default="unknown",
                        help="Target text to induce during the attack")
    parser.add_argument("--num_scale", type=int, default=4,
                        help="Number of augmented views to generate per optimization step")
    config_args = parser.parse_known_args()[0]
    assert config_args.method in ["cropa","baseline"], "method not supported"
    add_extra_args(config_args, config_args.model_name)
    
    module = importlib.import_module(f"models.{config_args.model_name}")
    if config_args.device >= 0:
        print("use specified gpu",config_args.device)
    else:
        config_args.device= get_available_gpus(5000)[0]
    device= f"cuda:{ config_args.device}"
    print("Device is:", device)
    eval_model = load_model(config_args.device,module,config_args.model_name)
    print("model loaded")

    train_dataset, test_dataset = load_datasets(config_args)
    num_shots = config_args.shot
    prompt_num = config_args.prompt_num

    if config_args.method == "baseline":    
        alpha2 = 0
    else:    
        prompt_num_to_alpha2 = config_args.prompt_num_to_alpha2    
        alpha2 = prompt_num_to_alpha2[prompt_num]

    target_text = config_args.target
    iter_num = config_args.iter_num
    method_dir = f"{config_args.method}_w_multi_aug_ns_{config_args.num_scale}"
    
    attack(
        config_args,
        eval_model = eval_model,
        max_generation_length = 5,
        num_beams= 3,
        length_penalty = -2.0,
        num_shots = num_shots,
        alpha1 = 1/255,
        epsilon = 16/255,
        fraction=config_args.fraction,
        iters = iter_num,
        target = target_text+config_args.eoc,
        base_dir = f"output/{config_args.model_name}_shots_{num_shots}/{method_dir}/num_{prompt_num}_{target_text}",
        alpha2 = alpha2 ,
        prompt_num=config_args.prompt_num,
        datasets=(train_dataset,  test_dataset),
    )
# %%
