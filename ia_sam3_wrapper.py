from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torchvision.ops.boxes import batched_nms, box_area

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import plot_results
from ia_logging import ia_logging
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam3.model.sam3_image import Sam3Image
from sam3.model.sam3_image_processor import Sam3Processor
from sam2.utils.amg import (
    area_from_rle,
    batch_iterator,
    batched_mask_to_box,
    box_xyxy_to_xywh,
    build_all_layer_point_grids,
    calculate_stability_score,
    generate_crop_boxes,
    is_box_near_crop_edge,
    mask_to_rle_pytorch,
    MaskData,
    remove_small_regions,
    rle_to_mask,
    uncrop_boxes_xyxy,
    uncrop_masks,
    uncrop_points,
)

class Sam3Wrapper:
    def __init__(self, model: Sam3Image, **kwargs):
        """
        Wrapper for SAM 3 model to provide an interface compatible with
        SamAutomaticMaskGenerator and SamPredictor.
        """
        self.model = model

        # Store AMG parameters
        self.points_per_side = kwargs.get("points_per_side", 32)
        self.points_per_batch = kwargs.get("points_per_batch", 64)
        self.pred_iou_thresh = kwargs.get("pred_iou_thresh", 0.8)
        self.stability_score_thresh = kwargs.get("stability_score_thresh", 0.95)
        self.stability_score_offset = kwargs.get("stability_score_offset", 1.0)
        self.box_nms_thresh = kwargs.get("box_nms_thresh", 0.7)
        self.crop_n_layers = kwargs.get("crop_n_layers", 0)
        self.crop_nms_thresh = kwargs.get("crop_nms_thresh", 0.7)
        self.crop_overlap_ratio = kwargs.get("crop_overlap_ratio", 512 / 1500)
        self.crop_n_points_downscale_factor = kwargs.get("crop_n_points_downscale_factor", 1)
        self.min_mask_region_area = kwargs.get("min_mask_region_area", 0)

        # Determine confidence threshold for SAM 3 based on UI param or default
        # SAM 3 usually uses 0.5 default. If pred_iou_thresh is significantly different, we might use it?
        # But pred_iou_thresh is for filtering AFTER prediction.
        # Sam3Processor has confidence_threshold for internal filtering.
        # We'll set it low enough to get results, then filter manually.
        self.processor = Sam3Processor(model, confidence_threshold=0.4, device=model.device)

        # AMG State
        self.point_grids = build_all_layer_point_grids(
            self.points_per_side,
            self.crop_n_layers,
            self.crop_n_points_downscale_factor,
        )

        self.kwargs = kwargs

        # Internal state for predictor mode
        self._inference_state = None
        self._is_image_set = False

    @property
    def device(self):
        return self.model.device

    def to(self, device):
        self.model.to(device)
        self.processor.device = device
        return self

    def generate(self, image: Union[np.ndarray, Image.Image]) -> List[Dict[str, Any]]:
        """
        Generate masks for the entire image.
        Compatible with SamAutomaticMaskGenerator.generate
        """
        # Check for text prompt
        text_prompt = self.kwargs.get("text_prompt", None)

        if text_prompt:
             return self._generate_with_text(image, text_prompt)

        # Fallback to Grid Search (AMG)
        ia_logging.info("Generating SAM 3 masks using automatic grid search")
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image

        return self._generate_with_grid(image_np)

    def _generate_with_text(self, image: Union[np.ndarray, Image.Image], text_prompt: str) -> List[Dict[str, Any]]:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        inference_state = self.processor.set_image(image)
        # Use the provided text prompt
        inference_state = self.processor.set_text_prompt(text_prompt, inference_state)

        # Extract results
        masks = inference_state["masks"] # [N, 1, H, W] boolean
        scores = inference_state["scores"] # [N]
        boxes = inference_state["boxes"] # [N, 4] xyxy

        # Convert to list of dicts format expected by Inpaint Anything
        results = []
        for i in range(len(masks)):
            mask = masks[i].cpu().numpy().squeeze() # Squeeze (1, H, W) -> (H, W)
            score = scores[i].item()
            box = boxes[i].cpu().numpy().tolist() # xyxy

            # Convert xyxy to xywh
            x0, y0, x1, y1 = box
            w = x1 - x0
            h = y1 - y0
            bbox = [x0, y0, w, h]

            # area
            area = np.sum(mask)

            results.append({
                "segmentation": mask,
                "area": area,
                "bbox": bbox,
                "predicted_iou": score, # Using score as predicted_iou proxy
                "point_coords": [], # No point coords for text prompt
                "stability_score": score, # Using score as stability proxy
                "crop_box": [0, 0, image.width, image.height]
            })

        return results

    def _generate_with_grid(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Generates masks for the given image using grid points.
        Adapted from SAM2AutomaticMaskGenerator.
        """
        # Generate masks
        mask_data = self._generate_masks_grid(image)

        # Encode masks (binary mask mode usually for this extension)
        if mask_data is None:
            return []

        mask_data["segmentations"] = [rle_to_mask(rle) for rle in mask_data["rles"]]

        # Write mask records
        curr_anns = []
        for idx in range(len(mask_data["segmentations"])):
            ann = {
                "segmentation": mask_data["segmentations"][idx],
                "area": area_from_rle(mask_data["rles"][idx]),
                "bbox": box_xyxy_to_xywh(mask_data["boxes"][idx]).tolist(),
                "predicted_iou": mask_data["iou_preds"][idx].item(),
                "point_coords": [mask_data["points"][idx].tolist()],
                "stability_score": mask_data["stability_score"][idx].item(),
                "crop_box": box_xyxy_to_xywh(mask_data["crop_boxes"][idx]).tolist(),
            }
            curr_anns.append(ann)

        return curr_anns

    def _generate_masks_grid(self, image: np.ndarray) -> MaskData:
        orig_size = image.shape[:2]
        crop_boxes, layer_idxs = generate_crop_boxes(
            orig_size, self.crop_n_layers, self.crop_overlap_ratio
        )

        # Iterate over image crops
        data = MaskData()
        for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
            crop_data = self._process_crop(image, crop_box, layer_idx, orig_size)
            data.cat(crop_data)

        # Remove duplicate masks between crops
        if len(crop_boxes) > 1:
            # Prefer masks from smaller crops
            scores = 1 / box_area(data["crop_boxes"])
            scores = scores.to(data["boxes"].device)
            keep_by_nms = batched_nms(
                data["boxes"].float(),
                scores,
                torch.zeros_like(data["boxes"][:, 0]),  # categories
                iou_threshold=self.crop_nms_thresh,
            )
            data.filter(keep_by_nms)
        data.to_numpy()
        return data

    def _process_crop(
        self,
        image: np.ndarray,
        crop_box: List[int],
        crop_layer_idx: int,
        orig_size: Tuple[int, ...],
    ) -> MaskData:
        # Crop the image and calculate embeddings
        x0, y0, x1, y1 = crop_box
        cropped_im = image[y0:y1, x0:x1, :]
        cropped_im_size = cropped_im.shape[:2]

        # Set image in processor
        self.set_image(cropped_im)

        # Get points for this crop
        points_scale = np.array(cropped_im_size)[None, ::-1]
        points_for_image = self.point_grids[crop_layer_idx] * points_scale

        # Generate masks for this crop in batches
        data = MaskData()
        for (points,) in batch_iterator(self.points_per_batch, points_for_image):
            batch_data = self._process_batch(
                points, cropped_im_size, crop_box, orig_size, normalize=True
            )
            data.cat(batch_data)
            del batch_data

        # No reset needed for processor usually, just set_image next time

        # Remove duplicates within this crop.
        keep_by_nms = batched_nms(
            data["boxes"].float(),
            data["iou_preds"],
            torch.zeros_like(data["boxes"][:, 0]),  # categories
            iou_threshold=self.box_nms_thresh,
        )
        data.filter(keep_by_nms)

        # Return to the original image frame
        data["boxes"] = uncrop_boxes_xyxy(data["boxes"], crop_box)
        data["points"] = uncrop_points(data["points"], crop_box)
        data["crop_boxes"] = torch.tensor([crop_box for _ in range(len(data["rles"]))])

        return data

    def _process_batch(
        self,
        points: np.ndarray,
        im_size: Tuple[int, ...],
        crop_box: List[int],
        orig_size: Tuple[int, ...],
        normalize=False,
    ) -> MaskData:
        orig_h, orig_w = orig_size

        # Run model on this batch
        # Points: [B, 2]
        # SAM 3 predict takes points.
        # We reuse self.predict which handles normalization and prompt creation.

        # Wait, self.predict currently takes ONE set of points/boxes and returns top masks.
        # For AMG, we want 1 mask per point? Or multimask?
        # SAM 2 AMG gets 3 masks per point and filters them.

        # We need to call predict per point or batched?
        # SAM 3 `Sam3Processor` operates on single image.
        # Does `forward_grounding` support batch of prompts?
        # `Prompt` class supports batch of boxes/points.
        # `Prompt` dim: N_points x B x 2
        # If we have 1 image (B=1), we can have N points.
        # But `Sam3Image` usually treats N_points as defining ONE object (part of same prompt).
        # We want to treat each point as a separate prompt for a separate object.

        # SAM 2 `_predict` handles batching of prompts?
        # SAM 2 `SAM2ImagePredictor._predict` takes `point_coords` [B, N, 2].

        # For SAM 3, `Sam3Image` doesn't natively support "batch of prompts for single image" easily
        # unless we duplicate image features or rely on `forward_grounding` handling multiple queries?
        # But `Prompt` structure [N_points, B, 2] implies B is batch size (images).

        # If we want 64 points to generate 64 independent masks:
        # We might need to loop? Or duplicate image features?

        # `Sam3Image._get_img_feats` handles `img_batch`.
        # If we pass `img_ids` as [0, 0, 0...], it duplicates features?

        # `Sam3Processor` `set_image` sets `backbone_out`.
        # `Sam3Image` methods often take `find_input`.
        # `find_input.img_ids`.

        # If we want to batch prompts:
        # We can construct `Prompt` with B=batch_size (e.g. 64).
        # And `find_input` with `img_ids` = [0, 0, ... 0] (length 64).
        # And `text_ids` = [0 ...].

        # But `Sam3Processor` hides this.
        # We might need to access `self.model` directly like in `predict`.

        # Let's try to construct batched input.
        points_batch_size = len(points)

        # Points: [N_points_total, 2]
        # We want [1 point, B=N_points_total, 2]

        points_t = torch.as_tensor(points, dtype=torch.float32, device=self.device)
        # Normalize? self.predict does it.
        # `points` passed here are pixels? Yes.

        points_t[:, 0] /= im_size[1] # W
        points_t[:, 1] /= im_size[0] # H

        # [B, 2] -> [1, B, 2]
        point_embeddings = points_t.unsqueeze(0)

        # Labels: [1, B] (positive)
        point_labels = torch.ones(1, points_batch_size, dtype=torch.long, device=self.device)
        point_mask = torch.zeros(points_batch_size, 1, dtype=torch.bool, device=self.device)

        from sam3.model.geometry_encoders import Prompt
        from sam3.model.data_misc import FindStage

        geometric_prompt = Prompt(
            point_embeddings=point_embeddings,
            point_mask=point_mask,
            point_labels=point_labels
        )

        # We need to "expand" the backbone output to match batch size?
        # `Sam3Image._get_img_feats` logic:
        # `img_ids` selects features.
        # If `backbone_out` has features for image 0.
        # We pass `img_ids` = [0] * batch_size.

        img_ids = torch.zeros(points_batch_size, dtype=torch.long, device=self.device)
        text_ids = torch.zeros(points_batch_size, dtype=torch.long, device=self.device) # Dummy text ids

        # Construct find_input
        # We need to ensure `backbone_out` is compatible.
        # `Sam3Processor.set_image` sets `backbone_out`.

        # `Sam3Image` expects `backbone_out` to contain `language_features`.
        # If we didn't call `set_text_prompt`, it might not have them?
        # `add_geometric_prompt` logic handles this:
        # "If 'language_features' not in state... set text prompt to 'visual'".

        if "language_features" not in self._inference_state["backbone_out"]:
             dummy_text = self.model.backbone.forward_text(["visual"], device=self.device)
             self._inference_state["backbone_out"].update(dummy_text)

        # `language_features` will be [1, Seq, C].
        # `text_ids` will index into batch dim 0. So it works.

        # `find_input`
        find_input = FindStage(
            img_ids=img_ids,
            text_ids=text_ids,
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=point_embeddings,
            input_points_mask=point_mask,
        )

        # Forward
        out = self.model.forward_grounding(
            backbone_out=self._inference_state["backbone_out"],
            find_input=find_input,
            find_target=None,
            geometric_prompt=geometric_prompt
        )

        # Output processing
        # `pred_logits`: [B, Q, 1]
        # `pred_masks`: [B, Q, H, W]
        # Q is number of queries (200?).

        # For each point in batch B, we have Q predictions.
        # We want to select the best masks for each point.
        # SAM 2 AMG does `multimask_output=True` (3 masks).
        # We have 200 masks.

        scores = out["pred_logits"].squeeze(-1).sigmoid() # [B, Q]
        masks = out["pred_masks"] # [B, Q, H, W]

        # Interpolate masks
        masks_up = torch.nn.functional.interpolate(
            masks,
            size=im_size,
            mode="bilinear",
            align_corners=False
        ) # [B, Q, H, W]

        # For each batch item, we want top K masks?
        # Or filter by score?
        # SAM 2 keeps 3 masks.

        # Let's take top 3 masks per point.
        topk = 3
        scores_topk, indices_topk = torch.topk(scores, k=topk, dim=1) # [B, 3]

        # Gather masks
        # masks_up: [B, Q, H, W]
        # indices_topk: [B, 3]
        # We need to gather.

        # Expand indices: [B, 3, H, W]
        # This is expensive.

        # Loop over batch is easier?
        # batch_size usually 64.

        res_masks = []
        res_iou_preds = []

        for b in range(points_batch_size):
            inds = indices_topk[b] # [3]
            m = masks_up[b, inds, :, :] # [3, H, W]
            s = scores_topk[b] # [3]

            res_masks.append(m)
            res_iou_preds.append(s)

        masks_tensor = torch.stack(res_masks) # [B, 3, H, W]
        iou_preds_tensor = torch.stack(res_iou_preds) # [B, 3]

        # Serialize predictions and store in MaskData
        # Flatten B and 3 -> [B*3, H, W]
        data = MaskData(
            masks=masks_tensor.flatten(0, 1),
            iou_preds=iou_preds_tensor.flatten(0, 1),
            points=torch.as_tensor(points, device=self.device).repeat_interleave(topk, dim=0),
        )

        # Filter by IOU
        if self.pred_iou_thresh > 0.0:
            keep_mask = data["iou_preds"] > self.pred_iou_thresh
            data.filter(keep_mask)

        # Calculate stability score
        data["stability_score"] = calculate_stability_score(
            data["masks"], 0.0, self.stability_score_offset # Mask threshold 0.0 (logit)
        )
        if self.stability_score_thresh > 0.0:
            keep_mask = data["stability_score"] >= self.stability_score_thresh
            data.filter(keep_mask)

        # Threshold masks and calculate boxes
        data["masks"] = data["masks"] > 0.0
        data["boxes"] = batched_mask_to_box(data["masks"])

        # Filter boxes that touch crop boundaries
        keep_mask = ~is_box_near_crop_edge(
            data["boxes"], crop_box, [0, 0, orig_w, orig_h]
        )
        if not torch.all(keep_mask):
            data.filter(keep_mask)

        # Compress to RLE
        data["masks"] = uncrop_masks(data["masks"], crop_box, orig_h, orig_w)
        data["rles"] = mask_to_rle_pytorch(data["masks"])
        del data["masks"]

        return data

    # --- Predictor Interface (for interactive mode) ---

    def set_image(self, image: np.ndarray, image_format: str = "RGB") -> None:
        """
        Calculates the image embeddings for the provided image.
        """
        if image_format != "RGB":
             raise ValueError(f"SAM 3 wrapper only supports RGB, got {image_format}")

        pil_image = Image.fromarray(image)
        self._inference_state = self.processor.set_image(pil_image)
        self._is_image_set = True

    def predict(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        multimask_output: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict masks for the given input prompts, using the currently set image.
        """
        if not self._is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        img_w = self._inference_state["original_width"]
        img_h = self._inference_state["original_height"]

        from sam3.model.geometry_encoders import Prompt

        box_embeddings = None
        box_labels = None
        box_mask = None

        point_embeddings = None
        point_labels_tensor = None
        point_mask = None

        device = self.device

        if box is not None:
            if box.ndim == 1:
                box = box[None, :]

            # Convert XYXY to CXCYWH
            x1, y1, x2, y2 = box[:, 0], box[:, 1], box[:, 2], box[:, 3]
            w = x2 - x1
            h = y2 - y1
            cx = x1 + w / 2
            cy = y1 + h / 2

            # Normalize
            cx /= img_w
            cy /= img_h
            w /= img_w
            h /= img_h

            box_tensor = torch.tensor(np.stack([cx, cy, w, h], axis=1), dtype=torch.float32, device=device)
            box_embeddings = box_tensor.unsqueeze(1)

            box_labels = torch.ones(box.shape[0], 1, dtype=torch.long, device=device) # Positive
            box_mask = torch.zeros(1, box.shape[0], dtype=torch.bool, device=device) # Not masked

        if point_coords is not None:
            coords = point_coords.copy().astype(float)
            coords[:, 0] /= img_w
            coords[:, 1] /= img_h

            point_tensor = torch.tensor(coords, dtype=torch.float32, device=device)
            # Add batch dim: N_points x B x 2
            point_embeddings = point_tensor.unsqueeze(1)

            if point_labels is None:
                point_labels = np.ones(len(point_coords))

            point_labels_t = torch.tensor(point_labels, dtype=torch.long, device=device)
            point_labels_tensor = point_labels_t.unsqueeze(1)

            point_mask = torch.zeros(1, len(point_coords), dtype=torch.bool, device=device)

        geometric_prompt = Prompt(
            box_embeddings=box_embeddings,
            box_mask=box_mask,
            box_labels=box_labels,
            point_embeddings=point_embeddings,
            point_mask=point_mask,
            point_labels=point_labels_tensor
        )

        # Ensure text features exist (visual prompt fallback)
        if "language_features" not in self._inference_state["backbone_out"]:
             dummy_text = self.model.backbone.forward_text(["visual"], device=self.device)
             self._inference_state["backbone_out"].update(dummy_text)

        out = self.model.forward_grounding(
            backbone_out=self._inference_state["backbone_out"],
            find_input=self.processor.find_stage, # Reusing the one from processor
            find_target=None,
            geometric_prompt=geometric_prompt
        )

        pred_logits_scores = out["pred_logits"] # [B=1, Q, 1] (class scores)
        pred_masks_logits = out["pred_masks"]   # [B=1, Q, H, W] (mask logits)

        scores = pred_logits_scores[0, :, 0].sigmoid() # [Q]
        masks = pred_masks_logits[0] # [Q, H, W]

        # Interpolate masks to original size
        masks_up = torch.nn.functional.interpolate(
            masks.unsqueeze(1),
            size=(img_h, img_w),
            mode="bilinear",
            align_corners=False
        ).squeeze(1)

        # Select top K
        scores_val, topk_indices = torch.topk(scores, k=3 if multimask_output else 1)

        best_masks = masks_up[topk_indices] # [3, H, W]
        best_scores = scores_val # [3]

        masks_np = (best_masks > 0.0).cpu().numpy().astype(bool) # Logits > 0 == Prob > 0.5
        scores_np = best_scores.cpu().numpy()
        logits_np = best_masks.cpu().numpy()

        return masks_np, scores_np, logits_np

    def get_image_embedding(self):
        pass
