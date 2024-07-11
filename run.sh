DDPM_FLAGS="--ddpm_mode SDXL --model_name stabilityai/stable-diffusion-xl-base-1.0 --refiner_name stabilityai/stable-diffusion-xl-refiner-1.0 --diffusion_steps 40"
EVAL_FLAGS="--score_mechanism clipscore --eval_mechanism binary --threshold 0.72 --distance_threshold 0.01"
DYNAMIC_FLAGS="--dynamic_start 0.025 --dynamic_end 0.475 --num_iter 5 --shift_step_size 1 --momentum 1 --mom_count_thresh 2"

CUDA_VISIBLE_DEVICES=0 python run.py \
		--prompt_path prompt.json \
		--output_path results/xxx \
		--batch_size 4 \
		--num_imgs 64 \
		--if_downsample True \
		$DDPM_FLAGS \
		$EVAL_FLAGS \
		$DYNAMIC_FLAGS \
