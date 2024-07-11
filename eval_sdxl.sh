DDPM_FLAGS="--ddpm_mode SDXL --model_name stabilityai/stable-diffusion-xl-base-1.0 --refiner_name stabilityai/stable-diffusion-xl-refiner-1.0 --diffusion_steps 40"
EVAL_FLAGS="--score_mechanism clipscore --eval_mechanism simple"

CUDA_VISIBLE_DEVICES=0 python eval_sdxl.py \
		--prompt_path xxx.json \
		--output_path results/xxx \
		--batch_size 4 \
		--num_imgs 40 \
		--if_downsample True \
		--prompt_type first_and_second \
		$DDPM_FLAGS \
		$EVAL_FLAGS \
