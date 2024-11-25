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
#             transform=get_transform(cfg.res, False, loader_crop),
#             target_transform=get_transform(cfg.res, True, loader_crop),
#             cfg=model.cfg,
#             mask=True,
#         )

#         test_loader = DataLoader(test_dataset, cfg.batch_size,
#                                  shuffle=False, num_workers=cfg.num_workers,
#                                  pin_memory=True)

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
#                     image_index = batch['mask']

#                     # Forward pass through the model and CRF post-processing
#                     feats1 = par_model(img)
#                     feats2 = par_model(img.flip(dims=[3]))
#                     _, code1 = par_projection(feats1)
#                     _, code2 = par_projection(feats2)

#                     code = (code1 + code2.flip(dims=[3])) / 2
#                     code = F.interpolate(code, label.shape[-2:], mode='bilinear', align_corners=False)
#                     _, products = par_prediction(code)
#                     cluster_probs = torch.log_softmax(products * 2, dim=1)

#                     if cfg.run_crf:
#                         cluster_preds = batched_crf(pool, img, cluster_probs).argmax(1).cuda()
#                     else:
#                         cluster_preds = cluster_probs.argmax(1)

#                     # Update metrics with predictions and labels
#                     model.test_cluster_metrics.update(cluster_preds, label)

#         # Compute and print metrics after evaluation
#         tb_metrics = {**model.test_cluster_metrics.compute()}
#         print(f"Metrics for {model_path}: {tb_metrics}")

#         import json

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
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

torch.multiprocessing.set_sharing_strategy('file_system')

def _apply_crf(tup):
    return dense_crf(tup[0], tup[1])

def batched_crf(pool, img_tensor, prob_tensor):
    outputs = pool.map(_apply_crf, zip(img_tensor.detach().cpu(), prob_tensor.detach().cpu()))
    return torch.cat([torch.from_numpy(arr).unsqueeze(0) for arr in outputs], dim=0)

def calculate_iou(pred, label):
    pred = pred.cpu().numpy()
    label = label.cpu().numpy()
    intersection = np.logical_and(pred, label)
    union = np.logical_or(pred, label)
    iou = np.sum(intersection) / (np.sum(union) + 1e-10)
    return iou

# def save_result(img, label, pred, save_dir, idx, label_cmap):
#     # Create directories
#     original_dir = join(save_dir, "original")
#     gt_dir = join(save_dir, "ground_truth")
#     pred_dir = join(save_dir, "smooseg")
    
#     os.makedirs(original_dir, exist_ok=True)
#     os.makedirs(gt_dir, exist_ok=True)
#     os.makedirs(pred_dir, exist_ok=True)
    
#     print(f"Saving image {idx} to directories:")
    
#     # Convert tensors to numpy arrays
#     img = img.cpu().numpy().transpose(1, 2, 0)
#     img = (img * 255).astype(np.uint8)
#     label = label.cpu().numpy()
#     pred = pred.cpu().numpy()
    
#     # Save original image
#     Image.fromarray(img).save(join(save_dir, "original", f'image_{idx}.png'))
    
#     # Save ground truth with proper coloring
#     plt.figure(figsize=(8, 8))
#     plt.imshow(label_cmap[label])
#     plt.axis('off')
#     plt.savefig(join(save_dir, "ground_truth", f'label_{idx}.png'), 
#                 bbox_inches='tight', pad_inches=0, dpi=150)
#     plt.close()
    
#     # Save prediction with proper coloring
#     plt.figure(figsize=(8, 8))
#     plt.imshow(label_cmap[pred])
#     plt.axis('off')
#     plt.savefig(join(save_dir, "smooseg", f'pred_{idx}.png'), 
#                 bbox_inches='tight', pad_inches=0, dpi=150)
#     plt.close()

def save_result(img, label, pred, save_dir, idx, label_cmap):
    # Create directories
    os.makedirs(join(save_dir, "original"), exist_ok=True)
    os.makedirs(join(save_dir, "ground_truth"), exist_ok=True)
    os.makedirs(join(save_dir, "predictions"), exist_ok=True)
    
    # Convert tensors to numpy arrays and process original image
    img = img.cpu().numpy().transpose(1, 2, 0)
    img = np.clip(img * 255, 0, 255).astype(np.uint8)  # Proper normalization
    
    # Process label and prediction
    label = label.cpu().numpy().astype(np.uint8)
    pred = pred.cpu().numpy().astype(np.uint8)
    
    # Save original image using PIL for better quality
    Image.fromarray(img).save(join(save_dir, "original", f'image_{idx}.png'))
    
    # Create label visualization with proper coloring
    label_colored = np.zeros_like(img)
    for class_idx, color in enumerate(label_cmap):
        mask = label == class_idx
        label_colored[mask] = color
    
    # Create prediction visualization with same coloring scheme
    pred_colored = np.zeros_like(img)
    for class_idx, color in enumerate(label_cmap):
        mask = pred == class_idx
        pred_colored[mask] = color
    
    # Save ground truth and prediction with high quality
    plt.imsave(join(save_dir, "ground_truth", f'label_{idx}.png'), 
               label_colored.astype(np.uint8))
    plt.imsave(join(save_dir, "predictions", f'pred_{idx}.png'), 
               pred_colored.astype(np.uint8))

@hydra.main(config_path="configs", config_name="eval_config.yaml", version_base='1.1')
def my_app(cfg: DictConfig) -> None:
    results_dir = "/kaggle/working/results"  # Change this line
    os.makedirs(results_dir, exist_ok=True)
    print(f"Created results directory at: {results_dir}")   
    
    for model_path in cfg.model_paths:
        print(f"Loading model from checkpoint: {model_path}")
        model = LitUnsupervisedSegmenter.load_from_checkpoint(model_path)
        model.eval().cuda()

        test_dataset = ContrastiveSegDataset(
            data_dir=cfg.data_dir,
            dataset_name=model.cfg.dataset_name,
            crop_type=None,
            image_set="val",
            transform=get_transform(cfg.res, False, "center"),
            target_transform=get_transform(cfg.res, True, "center"),
            cfg=model.cfg,
            mask=True,
        )

        test_loader = DataLoader(
            test_dataset,
            cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )

        if cfg.use_ddp:
            par_model = torch.nn.DataParallel(model.net)
            par_projection = torch.nn.DataParallel(model.projection)
            par_prediction = torch.nn.DataParallel(model.prediction)
        else:
            par_model = model.net
            par_projection = model.projection
            par_prediction = model.prediction

        results = []
        with Pool(cfg.num_workers + 5) as pool:
            for batch in tqdm(test_loader):
                with torch.no_grad():
                    img = batch["img"].cuda()
                    label = batch["label"].cuda()

                    # Original prediction
                    feats1 = par_model(img)
                    _, code1 = par_projection(feats1)
                    code1 = F.interpolate(code1, label.shape[-2:], mode='bilinear', align_corners=False)

                    # Augmented prediction (horizontal flip)
                    feats2 = par_model(img.flip(dims=[3]))
                    _, code2 = par_projection(feats2)
                    code2 = F.interpolate(code2.flip(dims=[3]), label.shape[-2:], mode='bilinear', align_corners=False)

                    # Average predictions from original and augmented inputs
                    code_avg = (code1 + code2) / 2

                    # Final prediction probabilities and CRF post-processing (if enabled)
                    _, products_avg = par_prediction(code_avg)
                    cluster_probs_avg = torch.log_softmax(products_avg * 2, dim=1)
                    
                    if cfg.run_crf:
                        cluster_preds_avg = batched_crf(pool, img, cluster_probs_avg).argmax(1).cuda()
                    else:
                        cluster_preds_avg = cluster_probs_avg.argmax(1)

                    # Update metrics with predictions and labels
                    model.test_cluster_metrics.update(cluster_preds_avg, label)
                    
                    # Calculate IoU and store results for each image
                    for idx in range(len(img)):
                        iou = calculate_iou(cluster_preds_avg[idx], label[idx])
                        results.append({
                            'iou': iou,
                            'img': img[idx],
                            'label': label[idx],
                            'pred': cluster_preds_avg[idx],
                            'idx': len(results)
                        })

        # Sort by IoU and save top 10
        results.sort(key=lambda x: x['iou'], reverse=True)
        print(f"Found {len(results)} total results, saving top 10...")
        for i, result in enumerate(results[:10]):
            save_result(
                result['img'],
                result['label'],
                result['pred'],
                results_dir,
                f'top_{i+1}_iou_{result["iou"]:.3f}',
                model.label_cmap
            )
        # Sort by IoU and save top 10


        # Compute and print metrics after evaluation
        tb_metrics = {**model.test_cluster_metrics.compute()}
        print(f"Metrics for {model_path}: {tb_metrics}")
        
        import json
        # Save metrics
        metrics = model.test_cluster_metrics.compute()
        with open(join(results_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f)
        print(f"Results saved in {results_dir}")

if __name__ == "__main__":
    mp.set_start_method('spawn')
    my_app()
