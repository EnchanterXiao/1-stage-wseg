from torch.utils import data
from .pascal_voc import VOCSegmentation

datasets = {
    'pascal_voc': VOCSegmentation
}

def get_num_classes(args):
    return datasets[args.dataset.lower()].NUM_CLASS

def get_class_names(args):
    return datasets[args.dataset.lower()].CLASSES

def get_dataloader(args, cfg, split, batch_size=None, test_mode=None, scoremap_path=''):
    # assert split in ('train', 'train_voc', 'val'), "Unknown split '{}'".format(split)

    dataset_name = args.dataset.lower()
    dataset_cls = datasets[dataset_name]
    dataset = dataset_cls(cfg, split, test_mode, root=cfg.DATASET.ROOT, scoremap_path=scoremap_path)

    kwargs = {'num_workers': args.workers, 'pin_memory': True}
    shuffle, drop_last = [True, True] if split != 'val_voc' else [False, False]
    # print(split, shuffle)

    if batch_size is None:
        batch_size = cfg.TRAIN.BATCH_SIZE

    return data.DataLoader(dataset, batch_size=batch_size,
                           drop_last=drop_last, shuffle=shuffle, 
                           **kwargs)
