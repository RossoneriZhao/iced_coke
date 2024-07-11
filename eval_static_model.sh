DDPM_FLAGS="--ddpm_mode xxx --model_name xxx --diffusion_steps 40"
EVAL_FLAGS="--score_mechanism clipscore --eval_mechanism simple"

CUDA_VISIBLE_DEVICES=0 python eval_static_model.py \
		--prompt_path xxx.json \
		--output_path results/xxx \
		--batch_size 4 \
		--num_imgs 40 \
		--if_downsample True \
		--prompt_type prompt \
		$DDPM_FLAGS \
		$EVAL_FLAGS \
