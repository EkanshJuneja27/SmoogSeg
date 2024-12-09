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

# torch.multiprocessing.set_sharing_strategy('file_system')

# def _apply_crf(tup):
#     return dense_crf(tup[0], tup[1])

# def batched_crf(pool, img_tensor, prob_tensor):
#     outputs = pool.map(_apply_crf, zip(img_tensor.detach().cpu(), prob_tensor.detach().cpu()))
#     return torch.cat([torch.from_numpy(arr).unsqueeze(0) for arr in outputs], dim=0)

# @hydra.main(config_path="configs", config_name="eval_config.yaml", version_base='1.1')
# def my_app(cfg: DictConfig) -> None:
#     # Define result directory for predictions
#     data_dir = cfg.data_dir
#     result_dir = "../results/predictions/{}".format(cfg.experiment_name)
#     os.makedirs(join(result_dir, "img"), exist_ok=True)
#     os.makedirs(join(result_dir, "label"), exist_ok=True)
#     os.makedirs(join(result_dir, "cluster"), exist_ok=True)

#     # Loop through all model paths provided in the configuration
#     for model_path in cfg.model_paths:
#         print(f"Loading model from checkpoint: {model_path}")
#         model = LitUnsupervisedSegmenter.load_from_checkpoint(model_path)

#         # Load the test dataset
#         loader_crop = "center"
#         test_dataset = ContrastiveSegDataset(
#             data_dir=data_dir,
#             dataset_name=model.cfg.dataset_name,
#             crop_type=None,
#             image_set="val",
#             transform=get_transform(cfg.res, False, loader_crop),  # Ensure resolution consistency (e.g., 224)
#             target_transform=get_transform(cfg.res, True, loader_crop),
#             cfg=model.cfg,
#             mask=True,
#         )

#         test_loader = DataLoader(
#             test_dataset,
#             cfg.batch_size,
#             shuffle=False,
#             num_workers=4,  # Increased workers to 4 for faster data loading
#             pin_memory=True,
#         )

#         # Set the model to evaluation mode and move to GPU
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

#         # Determine dataset-specific parameters
#         if model.cfg.dataset_name == "cocostuff27":
#             all_good_images = range(2500)
#         elif model.cfg.dataset_name == "cityscapes":
#             all_good_images = range(600)
#         elif model.cfg.dataset_name == "potsdam":
#             all_good_images = range(900)
#         else:
#             raise ValueError("Unknown Dataset {}".format(model.cfg.dataset_name))

#         batch_nums = torch.tensor([n // (cfg.batch_size) for n in all_good_images])
#         batch_offsets = torch.tensor([n % (cfg.batch_size) for n in all_good_images])

#         saved_data = defaultdict(list)

#         # Use multiprocessing pool for CRF if enabled
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

#                     # Average predictions from original and augmented inputs
#                     code_avg = (code1 + code2) / 2

#                     # Final prediction probabilities and CRF post-processing (if enabled)
#                     _, products_avg = par_prediction(code_avg)
#                     cluster_probs_avg = torch.log_softmax(products_avg * 2, dim=1)

#                     if cfg.run_crf:
#                         cluster_preds_avg = batched_crf(pool, img, cluster_probs_avg).argmax(1).cuda()
#                     else:
#                         cluster_preds_avg = cluster_probs_avg.argmax(1)

#                     # Update metrics with predictions and labels
#                     model.test_cluster_metrics.update(cluster_preds_avg, label)

#         # Compute and print metrics after evaluation
#         tb_metrics = {**model.test_cluster_metrics.compute()}
#         print(f"Metrics for {model_path}: {tb_metrics}")

#         # Save metrics to a JSON file
#         metrics_path = join(result_dir, "metrics.json")
#         with open(metrics_path, "w") as f:
#             json.dump(tb_metrics, f)
#         print(f"Metrics saved at {metrics_path}")

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
    # Convert and save original image
    img_np = img.cpu().numpy().transpose(1, 2, 0)
    img_np = (img_np * 255).astype(np.uint8)
    Image.fromarray(img_np).save(join(save_dir, "original", f"{filename}.png"))
    
    # Convert and save ground truth and prediction
    for data, folder in [(label, "ground_truth"), (pred, "prediction")]:
        seg_np = data.cpu().numpy()
        colored_seg = np.zeros((seg_np.shape[0], seg_np.shape[1], 3), dtype=np.uint8)
        for i in range(len(colormap)):
            colored_seg[seg_np == i] = colormap[i]
        Image.fromarray(colored_seg).save(join(save_dir, folder, f"{filename}.png"))

@hydra.main(config_path="configs", config_name="eval_config.yaml", version_base='1.1')
def my_app(cfg: DictConfig) -> None:
    # Create save directories
    save_dir = "/kaggle/working/potsdam_results"
    os.makedirs(join(save_dir, "original"), exist_ok=True)
    os.makedirs(join(save_dir, "ground_truth"), exist_ok=True)
    os.makedirs(join(save_dir, "prediction"), exist_ok=True)

    for model_path in cfg.model_paths:
        print(f"Loading model from checkpoint: {model_path}")
        model = LitUnsupervisedSegmenter.load_from_checkpoint(model_path)
        model.eval().cuda()

        # Handle DataParallel if enabled
        if cfg.use_ddp:
            par_model = torch.nn.DataParallel(model.net)
            par_projection = torch.nn.DataParallel(model.projection)
            par_prediction = torch.nn.DataParallel(model.prediction)
        else:
            par_model = model.net
            par_projection = model.projection
            par_prediction = model.prediction

        # Load test dataset
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

        # Determine dataset-specific parameters
        if model.cfg.dataset_name == "cocostuff27":
            all_good_images = range(2500)
        elif model.cfg.dataset_name == "cityscapes":
            all_good_images = range(600)
        elif model.cfg.dataset_name == "potsdam":
            all_good_images = range(900)
        else:
            raise ValueError("Unknown Dataset {}".format(model.cfg.dataset_name))

        batch_nums = torch.tensor([n // (cfg.batch_size) for n in all_good_images])
        batch_offsets = torch.tensor([n % (cfg.batch_size) for n in all_good_images])

        saved_data = defaultdict(list)

        # Process images
        with Pool(cfg.num_workers + 5) as pool:
            for i, batch in enumerate(tqdm(test_loader)):
                with torch.no_grad():
                    img = batch["img"].cuda()
                    label = batch["label"].cuda()
                    image_index = batch["mask"]

                    # Original prediction
                    feats1 = par_model(img)
                    _, code1 = par_projection(feats1)
                    code1 = F.interpolate(code1, label.shape[-2:], mode='bilinear', align_corners=False)

                    # Augmented prediction (horizontal flip)
                    feats2 = par_model(img.flip(dims=[3]))
                    _, code2 = par_projection(feats2)
                    code2 = F.interpolate(code2.flip(dims=[3]), label.shape[-2:], mode='bilinear', align_corners=False)

                    # Average predictions
                    code_avg = (code1 + code2) / 2
                    _, products = par_prediction(code_avg)
                    cluster_probs = torch.log_softmax(products * 2, dim=1)

                    if cfg.run_crf:
                        cluster_preds = batched_crf(pool, img, cluster_probs).argmax(1).cuda()
                    else:
                        cluster_preds = cluster_probs.argmax(1)

                    # Save images for each item in batch
                    for b in range(img.shape[0]):
                        # Handle image index as string directly
                        img_name = image_index[b]
                        if img_name in test_dataset.dataset.files:
                            save_segmentation_images(
                                img[b],
                                label[b],
                                cluster_preds[b],
                                save_dir,
                                img_name,
                                model.label_cmap
                            )

                    # Update metrics
                    model.test_cluster_metrics.update(cluster_preds, label)

        # Print final metrics
        tb_metrics = {**model.test_cluster_metrics.compute()}
        print(f"Metrics for {model_path}: {tb_metrics}")

if __name__ == "__main__":
    mp.set_start_method('spawn')
    my_app()