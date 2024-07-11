import logging
import torch
import warnings
warnings.filterwarnings("ignore")

from utils.load import *
from utils.build import *
from utils.generate import *
from utils.save import *


def main():
	seed_all(0)

	args = get_args()
	safety_check(args)
	if not os.path.exists(args.output_path):
		os.makedirs(args.output_path)
	logging.basicConfig(filename = args.output_path + "/logs.log", level=logging.INFO)
	device = "cuda" if torch.cuda.is_available() else "cpu"
	log("Results and logs will be saved at " + args.output_path)
	log_args(args)

	# Get Prompt
	prompt_list = get_prompt_list(args.prompt_path)

	# Get Painter: Diffusion Models
	log("Loading Models...")
	model, refiner, score_model = get_model(
		args.ddpm_mode,
		args.model_name,
		args.refiner_name,
		args.score_mechanism,
		device,
	)

	# Get Eval Mechanism
	evaluator = get_evaluator(
		args.eval_mechanism,
		args.threshold,
		score_model,
		args.num_imgs,
		args.shift_step_size,
		device,
	)

	# Run
	log("Start Running...")
	best_ratio_and_score_dict = {}
	base_first_stage_ratio = float((args.dynamic_start + args.dynamic_end) / 2.)
	one_step_pr, dynamic_range = float(1 / args.diffusion_steps), (args.dynamic_end - args.dynamic_start)
	log("#"*100)
	for i in range(len(prompt_list)):
		prompt_dict = prompt_list[i]
		log(f"Generating {i}/{len(prompt_list)}...")
		log('Generating Images Using Prompt:"' + prompt_dict["prompt"] + '" ...')
		log("The first prompt is " + prompt_dict["first_prompt"] + " and the second prompt is " + prompt_dict["second_prompt"])

		# Generate and Evaluate Loop
		best_dynamic_ratio, best_score1, best_score2 = gen_and_eval_loop(
			model,
			refiner,
			args.batch_size, args.num_imgs,
			args.diffusion_steps, round(base_first_stage_ratio, 5), one_step_pr, dynamic_range,
			prompt_dict,
			evaluator,
			args.num_iter,
			args.output_path,
			True if args.if_downsample=="True" else False,
			args.dynamic_end, args.dynamic_start,
			device,
		)

		# Collect Results
		best_ratio_and_score_dict[prompt_dict["prompt"]] = [best_dynamic_ratio, best_score1, best_score2]
		torch.save(
			best_ratio_and_score_dict,
			args.output_path + "/prompt_ratio_score1_score2.pt"
		)
		log(f"The best first stage steps ratio is {best_dynamic_ratio}")
		log(f"The score of its first prompt is {best_score1}")
		log(f"The score of its second prompt is {best_score2}")
		log("#"*100)

	log("Generating Has Finished.")

	log("Calculating the average score...")
	avg_low_score, avg_high_score, length = get_average_score(best_ratio_and_score_dict)
	assert length == len(prompt_list)
	log(f"A total of {length} are used to generate images.")
	log(f"Their low scores average is {avg_low_score}")
	log(f"Their high scores average is {avg_high_score}")

	log("Running Has Finished.")


if __name__ == "__main__":
	main()
