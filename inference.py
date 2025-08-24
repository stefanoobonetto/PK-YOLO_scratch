#!/usr/bin/env python3
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from multimodal_pk_yolo import create_model
from brats_dataset import BraTSDataset, collate_fn
from utils.utils_inference import (
    parse_args,
    decode_yolo_predictions,
    save_visualization,
    time_synchronized
)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("inference")

def inference():
    args = parse_args()

    save_dir = Path(args.save_dir)
    (save_dir / "vis").mkdir(parents=True, exist_ok=True)
    pred_path = save_dir / args.json_name

    ds = BraTSDataset(args.data_dir, split=args.split, img_size=args.img_size, augment=False)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True,
                    collate_fn=collate_fn)

    device = torch.device(args.device)
    model = create_model(num_classes=1, input_channels=4,
                         pretrained_path=args.weights, device=device.type)
    model.eval()

    anchors = model.anchors.clone() if hasattr(model, "anchors") else torch.tensor([
        [[10, 13], [16, 30], [33, 23]],
        [[30, 61], [62, 45], [59, 119]],
        [[116, 90], [156, 198], [373, 326]],
    ], dtype=torch.float32)

    if args.half and device.type == 'cuda':
        model = model.half()

    n_images = 0
    n_saved_vis = 0
    t0 = time_synchronized()

    fjson = open(pred_path, 'w', encoding='utf-8')

    with torch.no_grad():
        for bi, batch in enumerate(dl):
            imgs = batch['images'].to(device, non_blocking=True)
            if args.half and device.type == 'cuda':
                imgs = imgs.half()

            slice_ids = batch['slice_ids']
            B, _, H, W = imgs.shape

            # Forward
            preds = model(imgs)

            # Decode
            dets_batch = decode_yolo_predictions(
                preds, anchors=anchors, img_size=args.img_size,
                conf_thresh=args.conf_thresh,
                iou_thresh=args.iou_thresh,
                max_dets=args.max_dets,
                num_classes=1
            )

            # Write JSONL and visualizations
            for i in range(B):
                sid = slice_ids[i]
                dets = dets_batch[i]
                record = {
                    "slice_id": sid,
                    "detections": dets
                }
                fjson.write(json.dumps(record) + "\n")

                # Save visualization periodically (uses GT if available)
                if args.save_vis and (bi % args.save_interval == 0):
                    gt_boxes = batch['bboxes'][i]
                    gt_labels = batch['labels'][i]
                    save_path = save_dir / "vis" / f"{sid}_pred.png"
                    title = f"{args.split} | {sid} | conf>{args.conf_thresh} iouNMS={args.iou_thresh}"
                    save_visualization(save_path, imgs[i].float().cpu(), gt_boxes, gt_labels, dets, title)
                    n_saved_vis += 1

            n_images += B

            if (bi + 1) % 10 == 0:
                logger.info(f"[{bi+1}/{len(dl)}] processed {n_images} images")

    fjson.close()
    dt = time_synchronized() - t0
    logger.info(f"Done. Processed {n_images} images in {dt:.2f}s "
                f"({n_images / max(dt,1e-6):.1f} img/s).")
    logger.info(f"Predictions written to: {pred_path}")
    if args.save_vis:
        logger.info(f"Visualizations saved in: {save_dir / 'vis'}")

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    inference()
