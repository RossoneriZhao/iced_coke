import json
import clip
import torch
import numpy as np
import random
import ImageReward as RM
from diffusers import DiffusionPipeline, StableDiffusionAttendAndExcitePipeline

from .score import *
from .generate import *


def get_prompt_list(prompts_path):
	with open(prompts_path, "r") as f:
		prompts = json.load(f)

	prompt_list = []
	for key in prompts.keys():
		# "key" should fit template: prompt1, prompt2, prompt3 ...
		# "single_prompt" should be a dict, which should have 5 keys:
		# prompt, first_prompt, first_describe, second_prompt, second_describe

		single_prompt = prompts[key]
		prompt_list.append(single_prompt)

	return prompt_list


def get_model(ddpm_mode, model_name, refiner_name, score_mechanism, device):
	# Load Diffusion Models
	if ddpm_mode == "SDXL":
		model = DiffusionPipeline.from_pretrained(
			model_name, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
		).to(device)

		refiner = DiffusionPipeline.from_pretrained(
			refiner_name,
			text_encoder_2=model.text_encoder_2,
			vae=model.vae,
			torch_dtype=torch.float16,
			use_safetensors=True,
			variant="fp16",
		).to(device)

	elif ddpm_mode == "AAE":
		model = StableDiffusionAttendAndExcitePipeline.from_pretrained(
			"CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16
		).to(device)

		refiner = None

	else:
		raise Exception("Unknown DDPM mode!")

	# Load Scoring Model
	if score_mechanism == "clipscore":
		score_model, _ = clip.load("ViT-B/32", device=device, jit=False)
		score_model = score_model.eval()

	elif score_mechanism == "image_reward":
		score_model = RM.load("ImageReward-v1.0")

	else:
		raise Exception("Unknown Scoring Mechanism!")

	return model, refiner, score_model


def get_evaluator(
	eval_mechanism, threshold, score_func, num_imgs, shift_step_size, device, args,
	score_mechanism="clipscore", dynamic_start=0.025, dynamic_end=0.475, momentum=2, # Added Newly for v-2
):
	if eval_mechanism == "simple":
		evaluator = SimpleEvaluator(threshold, score_func, num_imgs, shift_step_size, device)
	elif eval_mechanism == "binary":
		evaluator = BinarySearchEvaluator(
			threshold, score_func, num_imgs, shift_step_size, device,
			score_mechanism, dynamic_start, dynamic_end, args.diffusion_steps, momentum, args.mom_count_thresh,
		)
	else:
		raise Exception("Unknown Evaluator Mechanism!")

	return evaluator


def get_good_prompt(prompt_dict, prompt_type, indices_model=None):
	if prompt_type == "first_and_second":
		p = "first " + prompt_dict["first_prompt"] + ", and second " + prompt_dict["prompt"]
		return p

	elif prompt_type == "first":
		return prompt_dict["first_prompt"]

	elif prompt_type == "AAE":
		indices = []
		assert indices_model != None
		indices_dict = indices_model.get_indices(prompt_dict["first_prompt"])

		first_prompt_word_list = prompt_dict["first_prompt"].split(" ")
		second_prompt_word_list = prompt_dict["second_prompt"].split(" ")
		prompt_word_list = first_prompt_word_list + second_prompt_word_list
		for i in range(len(prompt_word_list)):
			prompt_word_list[i] = prompt_word_list[i] + "</w>"

		for k, v in indices_dict.items():
			if v in prompt_word_list:
				indices.append(k)
		assert len(indices) != 0

		return prompt_dict["prompt"], indices

	else:
		return prompt_dict["prompt"]


def seed_all(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)


def get_runner(model, refiner, evaluator, device, args):
	# The type of Runner, i.e. RunLoop, will be assigned
	# in RunLoop.run_loop(), based on RunLoop.evaluator.eval_type

	# Prepare
	base_first_stage_ratio = float((args.dynamic_start + args.dynamic_end) / 2.)
	one_step_pr, dynamic_range = float(1 / args.diffusion_steps), (args.dynamic_end - args.dynamic_start)

	runner = RunLoop(
		model, refiner,
		args.batch_size, args.num_imgs,
		args.diffusion_steps, round(base_first_stage_ratio, 5), one_step_pr, dynamic_range,
		evaluator,
		args.num_iter,
		args.output_path,
		True if args.if_downsample=="True" else False,
		args.dynamic_end, args.dynamic_start,
		device,
		args.threshold, # score threshold in v-2
		args.distance_threshold, # v-2
	)

	return runner
