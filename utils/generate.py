import torch
import math
import numpy as np
from PIL import Image

from .save import *


class RunLoop:
	def __init__(
		self,
		model,
		refiner,
		batch_size, num_imgs,
		diffusion_steps, base_first_stage_ratio, one_step_pr, dynamic_range,
		evaluator,
		num_iter,
		save_path,
		if_downsample,
		dynamic_end, dynamic_start,
		device,
		score_threshold = 0.72, # v-2
		distance_threshold = 0.1, # v-2
	) -> None:
		self.model, self.refiner = model, refiner
		self.batch_size, self.num_imgs = batch_size, num_imgs
		self.diffusion_steps, self.base_first_stage_ratio, self.one_step_pr, self.dynamic_range = diffusion_steps, base_first_stage_ratio, one_step_pr, dynamic_range
		self.evaluator = evaluator
		self.num_iter = num_iter
		self.save_path = save_path
		self.if_downsample = if_downsample
		self.dynamic_end, self.dynamic_start = dynamic_end, dynamic_start
		self.device = device

		# Newly Added in v-2
		self.distance_threshold = distance_threshold
		self.score_threshold = score_threshold

	def run_loop(self, prompt_dict):
		if self.evaluator.eval_type == "binary":
			best_dynamic_ratio, best_score1, best_score2 = self.binary_search_loop(prompt_dict)

		else:
			raise Exception("Unknown Evaluation Type!")

		return best_dynamic_ratio, best_score1, best_score2

	def binary_search_loop(self, prompt_dict):
		# TODO: fill in the binary search
		# Prepare
		first_stage_ratio = self.base_first_stage_ratio
		iters, first_result, if_acknowledge = 0, None, False
		score_dict = {}
		prompt_path_name = prompt_to_pathname(prompt_dict["prompt"])

		# Parameters for Binary Search
		low, high, current_step = math.ceil(self.dynamic_start * self.diffusion_steps), math.ceil(self.dynamic_end * self.diffusion_steps), math.ceil(self.base_first_stage_ratio * self.diffusion_steps)
		last_direction = 0

		# Run Loop
		while True:
			# Generate Images
			image_list, first_result_dict_ = single_generate(
				self.model,
				self.refiner,
				first_result,
				self.batch_size,
				self.num_imgs,
				self.diffusion_steps,
				first_stage_ratio,
				prompt_dict["prompt"],
				prompt_dict["first_prompt"],
			)

			# Get Scores
			log("Calculating Scores..")
			score1, score2 = self.evaluator.get_score(
				image_list,
				prompt_dict["first_prompt"],
				prompt_dict["first_describe"],
				prompt_dict["second_prompt"],
				prompt_dict["second_describe"],
				iters,
			)
			log(f"The score of first prompt is {score1} and of the second prompt is {score2}")

			# Save Single Result
			save_imgs(
				image_list,
				first_stage_ratio,
				prompt_path_name,
				self.save_path,
				self.if_downsample,
			)
			score_dict[str(first_stage_ratio)] = [score1, score2]

			# Check If Stop Early
			if (score1 > self.score_threshold and score2 > self.score_threshold) or (abs(score1 - score2) < self.distance_threshold):
				if_acknowledge = True
				best_dynamic_ratio = first_stage_ratio
				best_score1, best_score2 = score1, score2
				log("Break because of an acceptable result.")
				break

			# Choose the Next Step ID
			if score1 > score2:
				low = low
				high = current_step
				current_step = self.evaluator.evaluate_score(low, high, last_direction)
				last_direction = -1
			else:
				low = current_step
				high = high
				current_step = self.evaluator.evaluate_score(low, high, last_direction)
				last_direction = 1

			if current_step == None:
				log("Break because of binary search end.")
				break
			else:
				first_stage_ratio = round(current_step / self.diffusion_steps, 5)
				log(f"The next round will generate images from step {current_step}, the corresponding dynamic ratio is {first_stage_ratio}")

			# Pick the First Stage Result
			if iters == 0:
				first_result_dict = first_result_dict_
			first_result = first_result_dict[str(first_stage_ratio)].clone().to(self.device)

			iters += 1

		if not if_acknowledge:
			best_dynamic_ratio, best_score1, best_score2 = self.evaluator.find_best_dynamic_ratio(score_dict)

		return best_dynamic_ratio, best_score1, best_score2
			

# In v-2, generation loop in the form of function has been deprecated.
def gen_and_eval_loop(
	model,
	refiner,
	batch_size, num_imgs,
	diffusion_steps, base_first_stage_ratio, one_step_pr, dynamic_range,
	prompt_dict,
	evaluator,
	num_iter,
	save_path,
	if_downsample,
	dynamic_end, dynamic_start,
	device,
):
	# Generate and Evaluate
	iters, score_dict, first_result, used_ratio = 0, {}, None, []
	first_stage_ratio = base_first_stage_ratio
	prompt_path_name = prompt_to_pathname(prompt_dict["prompt"])
	while True:
		used_ratio.append(first_stage_ratio)
		image_list, first_result_dict_ = single_generate(
			model,
			refiner,
			first_result,
			batch_size,
			num_imgs,
			diffusion_steps,
			first_stage_ratio,
			prompt_dict["prompt"],
			prompt_dict["first_prompt"],
		)

		log("Calculating Scores..")
		score1, score2 = evaluator.get_score(
			image_list,
			prompt_dict["first_prompt"],
			prompt_dict["first_describe"],
			prompt_dict["second_prompt"],
			prompt_dict["second_describe"],
			iters,
		)
		log(f"The score of first prompt is {score1} and of the second prompt is {score2}")

		save_imgs(
			image_list,
			first_stage_ratio,
			prompt_path_name,
			save_path,
			if_downsample,
		)
		score_dict[str(first_stage_ratio)] = [score1, score2]

		dynamic_shift, if_acknowledge = evaluator.evaluate_score(score1, score2, one_step_pr, dynamic_range)

		if if_acknowledge:
			best_dynamic_ratio = first_stage_ratio
			best_score1 = score1
			best_score2 = score2

			log("Break because of an acceptable result.")
			break
		else:
			first_stage_ratio += dynamic_shift
			first_stage_ratio = round(first_stage_ratio, 5)
			# Check if first_stage_ratio
			if first_stage_ratio < dynamic_start or first_stage_ratio > dynamic_end:
				iters += num_iter
				break
			if first_stage_ratio in used_ratio:
				if evaluator.shift_step_size == 1:
					iters += num_iter
				else:
					evaluator.adjust_step_size()
					used_score1, used_score2 = score_dict[str(first_stage_ratio)][0], score_dict[str(first_stage_ratio)][1]
					dynamic_shift, _ = evaluator.evaluate_score(used_score1, used_score2, one_step_pr, dynamic_range)
					first_stage_ratio += dynamic_shift
					first_stage_ratio = round(first_stage_ratio, 5)

			if iters == 0:
				first_result_dict = first_result_dict_
			first_result = first_result_dict[str(first_stage_ratio)].clone().to(device)

		iters += 1
		if iters >= num_iter:
			log("Break because of limited num of iterations.")
			break

	if not if_acknowledge:
		best_dynamic_ratio, best_score1, best_score2 = evaluator.find_best_dynamic_ratio(score_dict)

	return best_dynamic_ratio, best_score1, best_score2


def single_generate(
	model,
	refiner,
	first_result,
	batch_size,
	num_imgs,
	diffusion_steps,
	first_stage_ratio,
	prompt,
	first_prompt,
):
	num_iter = num_imgs // batch_size
	image_list = []

	if first_result == None:
		log(f"The first generation using 2 dynamic models with base dynamic ratio: {first_stage_ratio}")
		log(f"{num_iter} batches need to be generated.")
		first_result_dict = None
		for _ in range(num_iter):
			first_result, first_result_single = model(
				prompt=first_prompt,
				num_inference_steps=diffusion_steps,
				denoising_end=0.5,
				output_type="latent",
				num_images_per_prompt=batch_size,
			)

			device = first_result.images.device
			first_result.images.to("cpu")
			del first_result
			first_result = first_result_single[str(first_stage_ratio)]

			image_list_ = refiner(
				prompt=prompt,
				num_inference_steps=diffusion_steps,
				denoising_start=first_stage_ratio,
				image=first_result.to(device),
				num_images_per_prompt=batch_size,
			).images

			first_result_dict = merge_first_result_dict(
				first_result_dict,
				first_result_single,
			)

			image_list = image_list + image_list_

		return image_list, first_result_dict

	else:
		log("#"*20)
		log(f"The generation with new dynamic ratio: {first_stage_ratio}")
		log(f"{num_iter} batches need to be generated.")
		for i in range(num_iter):
			s_idx = int(i * batch_size)
			e_idx = int(i * batch_size + batch_size)
			image_list_ = refiner(
				prompt=prompt,
				num_inference_steps=diffusion_steps,
				denoising_start=first_stage_ratio,
				image=first_result[s_idx: e_idx],
				num_images_per_prompt=batch_size,
			).images

			image_list = image_list + image_list_

		return image_list, None


def merge_first_result_dict(first_result_dict, first_result_single):
	if first_result_dict == None:
		return first_result_single

	else:
		for key, value in first_result_dict.items():
			first_result_dict[key] = torch.cat([value, first_result_single[key]])

		return first_result_dict


def bz_img_to_PIL_list(image_tensor):
	image_list = []
	bz = image_tensor.shape[0]
	image_tensor = ((image_tensor + 1) * 127.5).clamp(0, 255).to(torch.uint8)
	image_tensor = image_tensor.permute(0, 2, 3, 1)
	image_tensor = image_tensor.contiguous()
	image_np = image_tensor.cpu().numpy().astype(np.uint8)
	for i in range(bz):
		image_list.append(Image.fromarray(image_np[i]))

	return image_list


def single_generate_simple_model(
	model,
	prompt,
	num_imgs,
	batch_size,
	return_type,
):
	num_iter = num_imgs // batch_size
	image_list = []
	log(f"{num_iter} batches need to be generated.")
	for _ in range(num_iter):
		image_i = model(prompt)
		# assert isinstance(image_i, torch.Tensor)

		if return_type == "pil_list":
			image_list = image_list + image_i
		else:
			image_list = image_list + bz_img_to_PIL_list(image_i)

	return image_list


def prompt_to_pathname(prompt):
	words = prompt.split(" ")
	path_name = "_".join(words)

	return path_name
