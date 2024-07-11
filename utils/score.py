import numpy as np
import math

from .clipscore import *
from .save import *


def process_image(image_list, clipmodel, device):
	images = extract_all_images(image_list, clipmodel, device, batch_size=64, num_workers=8)

	return images


def simple_merge_score12(score1, score2, threshold):
	score12 = [threshold-score1, threshold-score2]
	merged_score = max(score12)

	return merged_score


class SimpleEvaluator:
	def __init__(self, threshold, score_model, num_imgs, shift_step_size, device):
		self.threshold = threshold
		self.score_model = score_model
		self.num_imgs = num_imgs
		self.merge_score_method = simple_merge_score12
		self.shift_step_size = shift_step_size
		self.device = device

		self.eval_type = "simple"

	def adjust_step_size(self):
		assert self.shift_step_size > 1
		self.shift_step_size -= 1

	def get_score(
		self,
		image_list,
		first_prompt,
		first_describe,
		second_prompt,
		second_describe,
		iters,
	):
		# Prepare Images
		image_array = process_image(image_list, self.score_model, self.device)

		# Prepare Prompts
		if iters == 0:
			self.first_prompt_list, self.first_describe_list = [], []
			self.second_prompt_list, self.second_describe_list = [], []
			for _ in range(self.num_imgs):
				self.first_prompt_list.append(first_prompt)
				self.first_describe_list.append(first_describe)
				self.second_prompt_list.append(second_prompt)
				self.second_describe_list.append(second_describe)

		# Compute the First Score
		_, first_prompt_scores, _ = get_clip_score(
			self.score_model,
			image_array,
			self.first_prompt_list,
			self.device
		)
		_, first_describe_scores, _ = get_clip_score(
			self.score_model,
			image_array,
			self.first_describe_list,
			self.device
		)
		first_score = max([first_prompt_scores.mean(), first_describe_scores.mean()])

		# Compute the Second Score
		_, second_prompt_scores, _ = get_clip_score(
			self.score_model,
			image_array,
			self.second_prompt_list,
			self.device
		)
		_, second_describe_scores, _ = get_clip_score(
			self.score_model,
			image_array,
			self.second_describe_list,
			self.device
		)
		second_score = max([second_prompt_scores.mean(), second_describe_scores.mean()])

		return first_score, second_score

	def evaluate_score(self, score1, score2, one_step_pr, dynamic_range):
		if (score1 > self.threshold) and (score2 > self.threshold):
			return None, True

		# The distance between two points(score1, score2) and the line(y=x+0).
		# Deduction:
		## d = np.abs(score1 - score1) / np.sqrt(2)
		## scale = d / np.sqrt(0.5)
		scale = np.abs(score1 - score1)
		dynamic_shift = float(((scale * dynamic_range // one_step_pr) + 1) * self.shift_step_size * one_step_pr)
		assert dynamic_shift % one_step_pr == 0

		if score1 > score2:
			dynamic_shift *= -1

		return dynamic_shift, False

	def find_best_dynamic_ratio(self, score_dict):
		merged_score_dict = {}
		for key in score_dict.keys():
			score1, score2 = score_dict[key][0], score_dict[key][1]
			merged_score_dict[key] = self.merge_score_method(score1, score2, self.threshold)

		zips = zip(merged_score_dict.values(), merged_score_dict.keys())
		sorted_list = sorted(zips)

		best_dynamic_ratio = sorted_list[0][1]
		best_score1, best_score2 = score_dict[best_dynamic_ratio][0], score_dict[best_dynamic_ratio][1]

		return best_dynamic_ratio, best_score1, best_score2


def get_average_score(score_dict):
	low_score, high_score = 0., 0.
	length = len(score_dict.keys())
	for key in score_dict.keys():
		if score_dict[key][1] > score_dict[key][2]:
			high_score += score_dict[key][1]
			low_score += score_dict[key][2]
		else:
			high_score += score_dict[key][2]
			low_score += score_dict[key][1]

	avg_low_score = float(low_score / length)
	avg_high_score = float(high_score / length)

	return avg_low_score, avg_high_score, length


class BinarySearchEvaluator(SimpleEvaluator):
	def __init__(
		self, threshold, score_model, num_imgs, shift_step_size, device,
		score_type, dynamic_start, dynamic_end, diffusion_steps, momentum = 0, mom_count_thresh = 2,
	):
		super().__init__(threshold, score_model, num_imgs, shift_step_size, device)
		self.score_type = score_type
		self.momentum = momentum
		self.eval_type = "binary"
		self.dynamic_start, self.dynamic_end = dynamic_start, dynamic_end
		self.diffusion_steps = diffusion_steps
		self.mom_count_thresh, self.mom_count_used = mom_count_thresh, 0

	def get_score(
		self,
		image_list,
		first_prompt,
		first_describe,
		second_prompt,
		second_describe,
		iters,
	):
		if self.score_type == "clipscore":
			return super().get_score(
				image_list,
				first_prompt,
				first_describe,
				second_prompt,
				second_describe,
				iters,
			)

		else:
			raise Exception("Unknown Scoring Mechanism!")

	def evaluate_score(
		self, low, high, direction,
	):
		# Direction must in [0, -1, 1]
		if high - low <= 1:
			return None

		mid = math.ceil((low + high) / 2)
		assert direction in [0, -1, 1]
		mid = mid + int(direction * self.momentum)
		if direction != 0:
			self.check_mom()

		if mid > math.ceil(self.dynamic_end * self.diffusion_steps):
			mid = math.ceil(self.dynamic_end * self.diffusion_steps)
		elif mid < math.ceil(self.dynamic_start * self.diffusion_steps):
			mid = math.ceil(self.dynamic_start * self.diffusion_steps)
		else:
			mid = mid
		
		return mid

	def check_mom(self):
		if self.momentum == 0:
			return

		assert self.momentum > 0
		self.mom_count_used += 1
		if self.mom_count_used == self.mom_count_thresh:
			log("Reduce the momentum.")
			self.mom_count_used = 0
			self.momentum -= 1
