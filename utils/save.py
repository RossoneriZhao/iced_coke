import os
import logging
from PIL import Image


def save_imgs(img_list, first_stage_ratio, prompt, save_path, if_downsample):
	base_path = save_path + "/" + prompt + "/" + str(first_stage_ratio)
	if not os.path.exists(base_path):
		os.makedirs(base_path)
	else:
		i = 1
		while True:
			i += 1
			if not os.path.exists(base_path + f"_touch{i}"):
				base_path_new = base_path + f"_touch{i}"
				break
		base_path = base_path_new
		os.mkdir(base_path)

	if if_downsample:
		down_path = base_path + "/down_sample"
		os.mkdir(down_path)

	for i in range(len(img_list)):
		name = f"/image_{i}.png"
		img_list[i].save(base_path + name)

		if if_downsample:
			down_img_i = img_list[i].resize((192, 192), Image.ANTIALIAS)
			down_img_i.save(down_path + name)


def simple_save(image_list, prompt, save_path, if_downsample):
	base_path = save_path + "/" + prompt
	if not os.path.exists(base_path):
		os.makedirs(base_path)

	if if_downsample:
		down_path = base_path + "/down_sample"
		os.mkdir(down_path)

	for i in range(len(image_list)):
		name = f"/image_{i}.png"
		image_list[i].save(base_path + name)

		if if_downsample:
			down_img_i = image_list[i].resize((192, 192), Image.ANTIALIAS)
			down_img_i.save(down_path + name)


def ending(save_path, best_dynamic_ratio, prompt):
	base_path = save_path + "/" + prompt + "/" + str(best_dynamic_ratio)
	assert os.path.exists(base_path)

	os.rename(
		base_path,
		base_path + "_best"
	)


def log(info):
	print(info)
	logging.info(info)
