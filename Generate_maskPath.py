import os
'''
生成伪语义信息的路径文件
'''

read_file = '../1sw/data/train_augvoc.txt'
save_file = '../1sw/data/train_gen_v1.txt'
mask_path = '../1sw/output/pascal_voc/0927/bs/val_voc/no_crf/'
result = []
img_gt_name_list = open(read_file).read().splitlines()
with open(save_file, 'w') as f:
    for path in img_gt_name_list:
        path = path.split(" ")[0]
        f.write(path)
        filename = path.split('/')[-1].split('.')[0]
        f.write(os.path.join(mask_path, filename+'.png'))
        f.write('\n')