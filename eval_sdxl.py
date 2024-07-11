import logging
import torch
import warnings
warnings.filterwarnings("ignore")

from utils.load import *
from utils.build import *
from utils.generate import *
from utils.save import *


# 此文件帮助你使用自己的模型进行采样
# 需要你实现的 code 会进行标注和提示
def main():
	seed_all(0)

	args = get_args()
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
	########################################################################
	# TODO 1
	# 你需要前往 utils/build.py 文件中完善 get_model 函数
	# 你需要添加一个"elif ddpm_mode == "xxx":"
	# 然后完成：模型的初始化、参数读取、device设置
	#
	# 在返回时，你需要返回三个东西，model，refiner，score_model，你只需要实现model的创建就可，并写死refiner=None，score_model部分不需要管
	#
	# 如果你使用 api 当然也是可以的，注意上面的返回就可以
	# 你需要保证你的模型可以被当作是一个函数：输入为一个字符串（单个prompt），输出为一个torch.Tensor (batch_size, 3, h, w)
	model, refiner, score_model = get_model(
		args.ddpm_mode,
		args.model_name,
		args.refiner_name,
		args.score_mechanism,
		device,
	)

	def sdxl_forward(prompt):
		first_result = model(
			prompt=prompt,
			num_inference_steps=args.diffusion_steps,
			denoising_end=0.8,
			output_type="latent",
			num_images_per_prompt=args.batch_size,
		).images.float()

		image_list = refiner(
			prompt=prompt,
			num_inference_steps=args.diffusion_steps,
			denoising_start=0.8,
			image=first_result,
			num_images_per_prompt=args.batch_size,
		).images

		return image_list
	########################################################################

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
	score_dict, i = {}, 0
	log("#"*100)
	for prompt_dict in prompt_list:
		i += 1
		used_prompt = get_good_prompt(prompt_dict, args.prompt_type)
		# Generating
		log(f"Generating {i}/{len(prompt_list)}...")
		log('Generating Images Using Prompt:"' + used_prompt + '" ...')
		image_list = single_generate_simple_model(
			sdxl_forward,
			used_prompt,
			args.num_imgs,
			args.batch_size,
			return_type="pil_list",
		)

		# Calculating Scores
		log("Calculating Scores..")
		score1, score2 = evaluator.get_score(
			image_list,
			prompt_dict["first_prompt"],
			prompt_dict["first_describe"],
			prompt_dict["second_prompt"],
			prompt_dict["second_describe"],
			iters=0,
		)
		log(f"The score of first prompt is {score1} and of the second prompt is {score2}")

		# Collect Results
		simple_save(
			image_list,
			prompt_to_pathname(used_prompt),
			args.output_path,
			args.if_downsample,
		)
		score_dict[used_prompt] = [0., score1, score2]

		log("#"*100)

	log("Generating Has Finished.")

	log("Calculating the average score...")
	avg_low_score, avg_high_score, length = get_average_score(score_dict)
	assert length == len(prompt_list)
	log(f"A total of {length} are used to generate images.")
	log(f"Their low scores average is {avg_low_score}")
	log(f"Their high scores average is {avg_high_score}")

	log("Running Has Finished.")


if __name__ == "__main__":
	main()
