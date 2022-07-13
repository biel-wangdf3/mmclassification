

# 1: choose easy base dataset from \\10.211.64.54\dataset\SBU_AOI to generate base dataset \\10.211.64.54\dataset\wdf\base_dataset 
# that only contain easy level by manual


# 2: use easy base dataset to generate rough model 
# model path: /home/wdf/workspace/open-mmlab/mmclassification/work_dirs/vgg16bn_8xb32_sbu/latest.pth


# 3: use the rough model to generate handle, difficult, error by score 


# 4: split base dataset to train and val


# 5: repeat step 3-4


# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
import re
from tkinter import image_names

import numpy as np
import shutil
import os
import mmcv
import cv2

from mmcls.apis import inference_model, init_model, show_result_pyplot



configs = "./configs/vgg/vgg16bn_8xb32_sbu.py"
checkpoint = "./work_dirs/vgg16bn_8xb32_sbu/latest.pth"
device = 'cuda:3'
show = False

img_folder_path = "/home_ext/dataset/SBU_AOI/"
img_save_path = "/home/wdf/workspace/open-mmlab/mmclassification/save_base_dataset"

class_name = os.listdir(img_folder_path)
class_name = ['6.09', '6.16', '6.24']


model = init_model(configs, checkpoint, device=device)

'''
for class_n in class_name:      # class: ok, ng ;  label : 1, 0
    for defa in os.listdir(os.path.join(img_folder_path,class_n)):                  # defect class : stain, dot
        for rank in os.listdir(os.path.join(img_folder_path,class_n,defa)):         # defect level : easy, handle, hard, error
            for img in os.listdir(os.path.join(img_folder_path,class_n,defa,rank)): # img list 
                print(os.path.join(img_folder_path,class_n,defa,rank,img))

                image = os.path.join(img_folder_path,class_n,defa,rank,img)
                result = inference_model(model, image)

                pred_score = result["pred_score"]           # float
                pred_label = result["pred_label"]           # numpy.int64
                pred_class = result["pred_class"]           # str
                labels = defa

                img = img.split('.bmp')[0] + '_' + pred_class + '_' + f'{pred_score:.2f}'+ '.bmp'

                result_show = {
                    'pred_score': pred_score,
                    'pred_label': pred_label,
                    'pred_class': pred_class,
                    "true_label": labels
                }

                if hasattr(model, 'module'):
                    model = model.module

                difficult_path = os.path.join(img_save_path,class_n,defa,"hard")
                handle_path = os.path.join(img_save_path,class_n,defa,"handle")
                easy_path = os.path.join(img_save_path,class_n,defa,"easy")
                error_path = os.path.join(img_save_path,class_n,defa,"error")

                if not os.path.exists(difficult_path):
                    os.makedirs(difficult_path)
                if not os.path.exists(handle_path):
                    os.makedirs(handle_path)
                if not os.path.exists(easy_path):
                    os.makedirs(easy_path)
                if not os.path.exists(error_path):
                    os.makedirs(error_path)

                if result["pred_class"] != class_n:
                    # model.show_result(
                    # image,
                    # result,
                    # show=show,
                    # out_file=os.path.join(error_path,img),
                    # )
                    # shutil.copy(os.path.join(image), error_path)
                    os.system(f'cp {image} {os.path.join(error_path, img)}')

                if result["pred_score"] > 0.9:
                    os.system(f'cp {image} {os.path.join(easy_path, img)}')

                if result["pred_score"] > 0.7 and  result["pred_score"] <= 0.9:
                    os.system(f'cp {image} {os.path.join(handle_path, img)}')
                    # shutil.copy(os.path.join(image), handle_path)

                if  result["pred_score"] <= 0.7:
                    # shutil.copy(os.path.join(image), difficult_path)  
                    os.system(f'cp {image} {os.path.join(difficult_path, img)}')
'''


label = {'滚轮印':'ok', '毛刷印':'ok', '不规则脏污':'ng', '划伤':'ng', 
         '难区分':'ng', '干水印':'ng', '油污':'ng', '毛丝':'ng', }

for class_n in class_name:      # class: ok, ng ;  label : 1, 0
    for defa in os.listdir(os.path.join(img_folder_path,class_n)):                  # defect class : stain, dot
        # for rank in os.listdir(os.path.join(img_folder_path,class_n,defa)):         # defect level : easy, handle, hard, error
            for img in os.listdir(os.path.join(img_folder_path,class_n,defa)): # img list 
                print(os.path.join(img_folder_path,class_n,defa,img))

                image = os.path.join(img_folder_path,class_n,defa,img)
                result = inference_model(model, image)

                pred_score = result["pred_score"]           # float
                pred_label = result["pred_label"]           # numpy.int64
                pred_class = result["pred_class"]           # str
                labels = defa

                img = img.split('.bmp')[0] + '_' + pred_class + '_' + f'{pred_score:.2f}'+ '.bmp'

                result_show = {
                    'pred_score': pred_score,
                    'pred_label': pred_label,
                    'pred_class': pred_class,
                    "true_label": labels
                }

                if hasattr(model, 'module'):
                    model = model.module

                difficult_path = os.path.join(img_save_path,class_n,defa,"hard")
                handle_path = os.path.join(img_save_path,class_n,defa,"handle")
                easy_path = os.path.join(img_save_path,class_n,defa,"easy")
                error_path = os.path.join(img_save_path,class_n,defa,"error")

                if not os.path.exists(difficult_path):
                    os.makedirs(difficult_path)
                if not os.path.exists(handle_path):
                    os.makedirs(handle_path)
                if not os.path.exists(easy_path):
                    os.makedirs(easy_path)
                if not os.path.exists(error_path):
                    os.makedirs(error_path)

                if result["pred_class"] != label[defa]:
                    # model.show_result(
                    # image,
                    # result,
                    # show=show,
                    # out_file=os.path.join(error_path,img),
                    # )
                    # shutil.copy(os.path.join(image), error_path)
                    os.system(f'cp {image} {os.path.join(error_path, img)}')

                elif result["pred_score"] > 0.9:
                    os.system(f'cp {image} {os.path.join(easy_path, img)}')

                elif result["pred_score"] > 0.7 and  result["pred_score"] <= 0.9:
                    os.system(f'cp {image} {os.path.join(handle_path, img)}')
                    # shutil.copy(os.path.join(image), handle_path)

                elif  result["pred_score"] <= 0.7:
                    # shutil.copy(os.path.join(image), difficult_path)  
                    os.system(f'cp {image} {os.path.join(difficult_path, img)}')




