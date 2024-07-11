import argparse
from .save import *


def safety_check(args):
	dynamic_start_step = float(args.diffusion_steps * args.dynamic_start)
	dynamic_end_step = float(args.diffusion_steps * args.dynamic_end)
	if dynamic_start_step % 1 != 0:
		raise Exception("diffusion_steps x dynamic_start should be int!")

	if dynamic_end_step % 1 != 0:
		raise Exception("diffusion_steps x dynamic_end should be int!")


def get_args():
	parser = argparse.ArgumentParser()

	parser.add_argument(
		"--prompt_path",
		type=str,
		default="xxx.json",
		help="Where to load prompts."
	)

	parser.add_argument(
		"--output_path",
		type=str,
		help="Where to save results and logs."
	)

	parser.add_argument(
		"--ddpm_mode",
		type=str,
		default="SDXL",
		help="Which Diffusion Model to generate images."
	)

	parser.add_argument(
		"--model_name",
		type=str,
		default="stabilityai/stable-diffusion-xl-base-1.0",
		help="Which model to generate images."
	)

	parser.add_argument(
		"--refiner_name",
		type=str,
		default="stabilityai/stable-diffusion-xl-refiner-1.0",
		help="Which refiner to generate images."
	)

	parser.add_argument(
		"--batch_size",
		type=int,
		default=4,
	)

	parser.add_argument(
		"--num_imgs",
		type=int,
		default=8,
		help="Num of images per prompt."
	)

	parser.add_argument(
		"--diffusion_steps",
		type=int,
		default=40,
		help="The num of steps in generation."
	)

	parser.add_argument(
		"--score_mechanism",
		type=str,
		default="clipscore",
		help="Which scoring mechanism to evaluate."
	)

	parser.add_argument(
		"--eval_mechanism",
		type=str,
		default="simple",
		help="Which evaluator to evaluate."
	)

	parser.add_argument(
		"--threshold",
		type=float,
		default=0.74,
		help="Stop generation and acknowledge the results."
	)

	parser.add_argument(
		"--dynamic_start",
		type=float,
		default=0.2,
		help="The lowest step of dynamic."
	)

	parser.add_argument(
		"--dynamic_end",
		type=float,
		default=0.4,
		help="The highest step of dynamic."
	)

	parser.add_argument(
		"--num_iter",
		type=int,
		default=5,
		help="Max num of generation iterations."
	)

	parser.add_argument(
		"--if_downsample",
		type=str,
		default="True",
		help="If downsample, it will save 192x192 images additionally."
	)

	parser.add_argument(
		"--shift_step_size",
		type=int,
		default=1,
		help="The step size when shifting th dynamic ratio."
	)

	parser.add_argument(
		"--prompt_type",
		type=str,
		default="prompt",
		help="The type of prompt you want use."
	)

	parser.add_argument(
		"--distance_threshold",
		type=float,
		default=0.1,
		help="The threshold of distance score."
	)

	parser.add_argument(
		"--momentum",
		type=int,
		default=1,
		help="The momentum of binary search."
	)

	parser.add_argument(
		"--mom_count_thresh",
		type=int,
		default=2,
		help="The max count of using the current momentum."
	)

	args = parser.parse_args()

	return args


def log_args(args):
	log("#" * 100)
	log("View All Input Parameters:")
	for k, v in vars(args).items():
		log(f"{k} = {v}")
	log("#" * 100)
