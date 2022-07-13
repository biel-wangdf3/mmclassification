


import os
import shutil


ori_data_path = "/home_ext/whm_ext/wdf_basedataset"

tgt_train_path = "/home_ext/whm_ext/wdf_basedataset_traing/train"
tgt_val_path = "/home_ext/whm_ext/wdf_basedataset_traing/val"

path_list = os.listdir(ori_data_path)
split_rate = 0.7

for mtpath in path_list:
    for path_2 in os.listdir(os.path.join(ori_data_path, mtpath)):
        for path_3 in os.listdir(os.path.join(ori_data_path, mtpath,path_2)):

        # target train and val path
            train_path_full_path = os.path.join(tgt_train_path, mtpath,path_2,path_3)
            val_path_full_path = os.path.join(tgt_val_path, mtpath,path_2,path_3)
            # print(train_path_full_path)
            # if train and val target path is not exist, make it
            if not os.path.exists(train_path_full_path):
                os.makedirs(train_path_full_path)
            if not os.path.exists(val_path_full_path):
                os.makedirs(val_path_full_path)

            # list images under now folder
            mfile_list = os.listdir(os.path.join(ori_data_path, mtpath,path_2,path_3))
            len_mfile_list = len(mfile_list)
            train_mfile_list = mfile_list[: round(len_mfile_list * split_rate)]
            val_mfile_list = mfile_list[round(len_mfile_list * split_rate) :]
            # print(train_mfile_list)

            # copy train files
            for tfile in train_mfile_list:
                tfile_full_path = os.path.join(train_path_full_path, tfile)
                # print(os.path.join(ori_data_path, mtpath, path_2,path_3,tfile))dd
                shutil.copy(os.path.join(ori_data_path, mtpath, path_2,path_3,tfile), tfile_full_path)
            # copy val files
            for vfile in val_mfile_list:
                vfile_full_path = os.path.join(val_path_full_path, vfile)
                shutil.copy(os.path.join(ori_data_path, mtpath, path_2,path_3,vfile), vfile_full_path)
print("split train/val dataset done")


