# from modules import *
# from data import *
# from collections import defaultdict
# from multiprocessing import Pool
# import hydra
# import torch.multiprocessing as mp
# from crf import dense_crf
# from omegaconf import DictConfig
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# import torch.nn.functional as F
# from train_segmentation import LitUnsupervisedSegmenter
# import numpy as np
# from PIL import Image
# import os
# from os.path import join

# torch.multiprocessing.set_sharing_strategy('file_system')

# def _apply_crf(tup):
#     return dense_crf(tup[0], tup[1])

# def batched_crf(pool, img_tensor, prob_tensor):
#     outputs = pool.map(_apply_crf, zip(img_tensor.detach().cpu(), prob_tensor.detach().cpu()))
#     return torch.cat([torch.from_numpy(arr).unsqueeze(0) for arr in outputs], dim=0)

# def save_segmentation_images(img, label, pred, save_dir, filename, colormap):
#     """Save original, ground truth and predicted segmentation images"""
#     try:
#         # Create directories if they don't exist
#         os.makedirs(join(save_dir, "original"), exist_ok=True)
#         os.makedirs(join(save_dir, "ground_truth"), exist_ok=True)
#         os.makedirs(join(save_dir, "prediction"), exist_ok=True)
        
#         # Convert and save original image
#         img_np = img.cpu().numpy().transpose(1, 2, 0)
#         img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
#         if img_np.shape[2] > 3:
#             img_np = img_np[:, :, :3]
#         Image.fromarray(img_np).save(join(save_dir, "original", f"{filename}.png"))

        
#         # Convert and save ground truth and prediction
#         for data, folder in [(label, "ground_truth"), (pred, "prediction")]:
#             seg_np = data.cpu().numpy()
#             colored_seg = np.zeros((seg_np.shape[0], seg_np.shape[1], 3), dtype=np.uint8)
#             for i in range(len(colormap)):
#                 mask = (seg_np == i)
#                 colored_seg[mask] = colormap[i]
#             Image.fromarray(colored_seg).save(join(save_dir, folder, f"{filename}.png"))
#         print(f"Successfully saved images for {filename}")
#     except Exception as e:
#         print(f"Error saving images for {filename}: {str(e)}")

# @hydra.main(config_path="configs", config_name="eval_config.yaml", version_base='1.1')
# def my_app(cfg: DictConfig) -> None:
#     # Create save directories
#     save_dir = "/kaggle/working/potsdam_results"
#     os.makedirs(join(save_dir, "original"), exist_ok=True)
#     os.makedirs(join(save_dir, "ground_truth"), exist_ok=True)
#     os.makedirs(join(save_dir, "prediction"), exist_ok=True)

#     for model_path in cfg.model_paths:
#         print(f"Loading model from checkpoint: {model_path}")
#         model = LitUnsupervisedSegmenter.load_from_checkpoint(model_path)
#         model.eval().cuda()

#         # Handle DataParallel if enabled
#         if cfg.use_ddp:
#             par_model = torch.nn.DataParallel(model.net)
#             par_projection = torch.nn.DataParallel(model.projection)
#             par_prediction = torch.nn.DataParallel(model.prediction)
#         else:
#             par_model = model.net
#             par_projection = model.projection
#             par_prediction = model.prediction

#         # Load test dataset
#         test_dataset = ContrastiveSegDataset(
#             data_dir=cfg.data_dir,
#             dataset_name="potsdam",
#             crop_type=None,
#             image_set="val",
#             transform=get_transform(cfg.res, False, "center"),
#             target_transform=get_transform(cfg.res, True, "center"),
#             cfg=model.cfg,
#             mask=True
#         )

#         test_loader = DataLoader(
#             test_dataset,
#             cfg.batch_size,
#             shuffle=False,
#             num_workers=cfg.num_workers,
#             pin_memory=True
#         )

#         # Process images
#         with Pool(cfg.num_workers + 5) as pool:
#             for i, batch in enumerate(tqdm(test_loader)):
#                 with torch.no_grad():
#                     img = batch["img"].cuda()
#                     label = batch["label"].cuda()
#                     image_index = batch["mask"]

#                     # Original prediction
#                     feats1 = par_model(img)
#                     _, code1 = par_projection(feats1)
#                     code1 = F.interpolate(code1, label.shape[-2:], mode='bilinear', align_corners=False)

#                     # Augmented prediction (horizontal flip)
#                     feats2 = par_model(img.flip(dims=[3]))
#                     _, code2 = par_projection(feats2)
#                     code2 = F.interpolate(code2.flip(dims=[3]), label.shape[-2:], mode='bilinear', align_corners=False)

#                     # Average predictions
#                     code_avg = (code1 + code2) / 2
#                     _, products = par_prediction(code_avg)
#                     cluster_probs = torch.log_softmax(products * 2, dim=1)

#                     if cfg.run_crf:
#                         cluster_preds = batched_crf(pool, img, cluster_probs).argmax(1).cuda()
#                     else:
#                         cluster_preds = cluster_probs.argmax(1)

#                     # Save images for each item in batch

#                     for b in range(img.shape[0]):
#                         # img_name = image_index[b]
#                         img_name = f"{i * cfg.batch_size + b:04d}" 
#                         print(f"Processing image {img_name}, type: {type(img_name)}")
#                         print(f"Saving image {img_name}")
#                         save_segmentation_images(
#                             img[b],
#                             label[b],
#                             cluster_preds[b],
#                             save_dir,
#                             img_name,
#                             model.label_cmap
#                         )

#                     # Update metrics
#                     model.test_cluster_metrics.update(cluster_preds, label)

#         # Print final metrics
#         tb_metrics = {**model.test_cluster_metrics.compute()}
#         print(f"Metrics for {model_path}: {tb_metrics}")

# if __name__ == "__main__":
#     mp.set_start_method('spawn')
#     my_app()

from modules import *
from data import *
from collections import defaultdict
from multiprocessing import Pool
import hydra
import torch.multiprocessing as mp
from crf import dense_crf
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from train_segmentation import LitUnsupervisedSegmenter
import numpy as np
from PIL import Image
import os
from os.path import join

torch.multiprocessing.set_sharing_strategy('file_system')

def _apply_crf(tup):
    return dense_crf(tup[0], tup[1])

def batched_crf(pool, img_tensor, prob_tensor):
    outputs = pool.map(_apply_crf, zip(img_tensor.detach().cpu(), prob_tensor.detach().cpu()))
    return torch.cat([torch.from_numpy(arr).unsqueeze(0) for arr in outputs], dim=0)

def save_segmentation_images(img, label, pred, save_dir, filename, colormap):
    """Save original, ground truth and predicted segmentation images"""
    try:
        os.makedirs(join(save_dir, "original"), exist_ok=True)
        os.makedirs(join(save_dir, "ground_truth"), exist_ok=True)
        os.makedirs(join(save_dir, "prediction"), exist_ok=True)
        
        # Save original image with proper RGB values
        img_np = img.cpu().numpy().transpose(1, 2, 0)
        # Scale to 0-255 without clipping to preserve brightness
        img_np = (img_np * 255).astype(np.uint8)
        if img_np.shape[2] > 3:
            img_np = img_np[:, :, :3]
        Image.fromarray(img_np).save(join(save_dir, "original", f"{filename}.png"))
        
        # Fixed colormap for Potsdam dataset
        fixed_colormap = np.array([
            [68, 1, 84],      # Purple
            [33, 145, 140],   # Turquoise
            [253, 231, 37]    # Yellow
        ])
        
        # Save ground truth and prediction with correct colors
        for data, folder in [(label, "ground_truth"), (pred, "prediction")]:
            seg_np = data.cpu().numpy()
            colored_seg = np.zeros((seg_np.shape[0], seg_np.shape[1], 3), dtype=np.uint8)
            
            for class_idx in range(len(fixed_colormap)):
                mask = (seg_np == class_idx)
                colored_seg[mask] = fixed_colormap[class_idx]
            
            Image.fromarray(colored_seg).save(join(save_dir, folder, f"{filename}.png"))
            
    except Exception as e:
        print(f"Error saving images for {filename}: {str(e)}")

@hydra.main(config_path="configs", config_name="eval_config.yaml", version_base='1.1')
def my_app(cfg: DictConfig) -> None:
    save_dir = "/kaggle/working/potsdam_results"
    os.makedirs(save_dir, exist_ok=True)
     

    for model_path in cfg.model_paths:
        print(f"Loading model from checkpoint: {model_path}")
        model = LitUnsupervisedSegmenter.load_from_checkpoint(model_path)
        model.eval().cuda()

        if cfg.use_ddp:
            par_model = torch.nn.DataParallel(model.net)
            par_projection = torch.nn.DataParallel(model.projection)
            par_prediction = torch.nn.DataParallel(model.prediction)
        else:
            par_model = model.net
            par_projection = model.projection
            par_prediction = model.prediction

        test_dataset = ContrastiveSegDataset(
            data_dir=cfg.data_dir,
            dataset_name="potsdam",
            crop_type=None,
            image_set="val",
            transform=get_transform(cfg.res, False, "center"),
            target_transform=get_transform(cfg.res, True, "center"),
            cfg=model.cfg,
            mask=True
        )

        test_loader = DataLoader(
            test_dataset,
            cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True
        )
        # Fixed colormap for Potsdam dataset
        fixed_colormap = np.array([
            [68, 1, 84],      # Purple
            [33, 145, 140],   # Turquoise
            [253, 231, 37]    # Yellow
        ])

        with Pool(cfg.num_workers + 5) as pool:
            for i, batch in enumerate(tqdm(test_loader)):
                with torch.no_grad():
                    img = batch["img"].cuda()
                    label = batch["label"].cuda()
                    image_index = batch["mask"]

                    feats1 = par_model(img)
                    _, code1 = par_projection(feats1)
                    code1 = F.interpolate(code1, label.shape[-2:], mode='bilinear', align_corners=False)

                    feats2 = par_model(img.flip(dims=[3]))
                    _, code2 = par_projection(feats2)
                    code2 = F.interpolate(code2.flip(dims=[3]), label.shape[-2:], mode='bilinear', align_corners=False)

                    code_avg = (code1 + code2) / 2
                    _, products = par_prediction(code_avg)
                    cluster_probs = torch.log_softmax(products * 2, dim=1)

                    if cfg.run_crf:
                        cluster_preds = batched_crf(pool, img, cluster_probs).argmax(1).cuda()
                    else:
                        cluster_preds = cluster_probs.argmax(1)

                    for b in range(img.shape[0]):
                        img_name = f"{i * cfg.batch_size + b:04d}"
                        save_segmentation_images(
                            img[b],
                            label[b],
                            cluster_preds[b],
                            save_dir,
                            img_name,
                            fixed_colormap
                        )

                    model.test_cluster_metrics.update(cluster_preds, label)

        metrics = model.test_cluster_metrics.compute()
        print(f"Test Metrics: {metrics}")

if __name__ == "__main__":
    mp.set_start_method('spawn')
    my_app()

