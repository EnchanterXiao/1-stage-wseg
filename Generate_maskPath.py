import os
'''
生成伪语义信息的路径文件
'''

read_file = '../1sw/data/train_augvoc.txt'
save_file = '../1sw/data/train_gen_bsl.txt'
mask_path = '/home/lwq/sdb1/xiaoxin/WSSS/1sw/output/pascal_voc/v0920/bsl/train_augvoc_1/no_crf'
result = []
img_gt_name_list = open(read_file).read().splitlines()
with open(save_file, 'w') as f:
    for path in img_gt_name_list:
        path = path.split(" ")[0]
        f.write(path)
        filename = path.split('/')[-1].split('.')[0]
        f.write(' ')
        f.write(os.path.join(mask_path, filename+'.png'))
        f.write('\n')