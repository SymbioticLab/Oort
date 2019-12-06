import torch, os, shutil


def prepare_val_folder(folder, wnids):
    img_files = sorted([os.path.join(folder, file) for file in os.listdir(folder)])

    for wnid in set(wnids):
        os.mkdir(os.path.join(folder, wnid))

    for wnid, img_file in zip(wnids, img_files):
        shutil.move(img_file, os.path.join(folder, wnid, os.path.basename(img_file)))

val_wnids = torch.load("/gpfs/gpfs0/groups/chowdhury/fanlai/imagenet/meta.bin")[1]
prepare_val_folder("/gpfs/gpfs0/groups/chowdhury/fanlai/imagenet/val", val_wnids)
