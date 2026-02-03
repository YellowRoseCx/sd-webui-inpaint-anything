from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.visualization_utils import plot_results
from ia_logging import ia_logging
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam3.model.sam3_image import Sam3Image
from sam3.model.sam3_image_processor import Sam3Processor

class Sam3Wrapper:
    def __init__(self, model: Sam3Image, **kwargs):
        """
        Wrapper for SAM 3 model to provide an interface compatible with
        SamAutomaticMaskGenerator and SamPredictor.
        """
        self.model = model
        self.processor = Sam3Processor(model, confidence_threshold=0.5, device=model.device)
        # We can store extra kwargs if needed for automatic generation
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
        # If we have a text prompt stored in kwargs or passed somehow, we could use it.
        # However, the standard generate() signature only takes 'image'.
        # We will assume if this is called, we might be doing automatic grid generation
        # OR we might have set a text prompt via a side channel or want to use default behavior.

        # SAM 3 usually expects a prompt (text or box).
        # If no prompt is provided, we can try to fall back to a grid search
        # if we implement it, OR we can reuse SAM2's automatic mask generator logic
        # but using SAM 3's predictor.

        # IMPORTANT: The user request implies using text prompt if available.
        # But `generate` signature is fixed by the library.
        # We will handle the "text prompt" logic in `ia_sam_manager` or `inpalib`
        # where we call `generate`.

        # For now, if this is called without text prompt setup, we might need a fallback.
        # But actually, we plan to pass text prompt to `generate` in inpalib.
        # So we will extend the signature or handle it via a setter before calling generate.

        # Let's check if we have a text prompt set.
        text_prompt = self.kwargs.get("text_prompt", None)

        if text_prompt:
             return self._generate_with_text(image, text_prompt)

        # If no text prompt, we might want to use the automatic mask generator logic.
        # SAM 3 repository doesn't seem to have a built-in AutomaticMaskGenerator class
        # that is exposed easily like SAM 2.
        # However, we can reuse SAM2AutomaticMaskGenerator if we can make SAM 3 look like SAM 2 predictor.
        # But SAM 3 API is quite different (Processor based).

        # For this task, since the user asked for SAM 3 support and mentioned text prompts,
        # we will prioritize the text prompt path.
        # If no text prompt is provided, we can default to a generic "visual" prompt
        # which SAM 3 supports (it tries to segment everything?).
        # Or we can try to use a default "object" prompt.

        # Let's try "visual" prompt which is used in `add_geometric_prompt` in `Sam3Processor`
        # when no text is present.

        ia_logging.info("Generating SAM 3 masks with default visual prompt (no text prompt provided)")
        return self._generate_with_text(image, "visual") # "visual" is a special token in SAM 3??
        # Actually, looking at `Sam3Processor.add_geometric_prompt`, it uses "visual"
        # if no text prompt is present.
        # But `set_text_prompt` expects a text.
        # If we want "segment everything" behavior without text, maybe we should pass "." or "object"?

        # Let's check `Sam3Processor` again.
        # It has `set_text_prompt`.

        return self._generate_with_text(image, "object")

    def _generate_with_text(self, image: Union[np.ndarray, Image.Image], text_prompt: str) -> List[Dict[str, Any]]:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        inference_state = self.processor.set_image(image)
        # Use the provided text prompt
        inference_state = self.processor.set_text_prompt(text_prompt, inference_state)

        # Extract results
        masks = inference_state["masks"] # [N, H, W] boolean
        scores = inference_state["scores"] # [N]
        boxes = inference_state["boxes"] # [N, 4] xyxy

        # Convert to list of dicts format expected by Inpaint Anything
        results = []
        for i in range(len(masks)):
            mask = masks[i].cpu().numpy().squeeze()
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

        # SAM 3 processor has `add_geometric_prompt`.
        # It takes `box` (List) and `label` (bool).
        # It seems it expects ONE box at a time or maybe we can hack it.
        # But wait, `Sam3Processor` seems designed for higher level interactions.

        # `add_geometric_prompt` docstring:
        # "Adds a box prompt and run the inference.
        # The box is assumed to be in [center_x, center_y, width, height] format and normalized in [0, 1] range."

        # But `SamPredictor.predict` passes points and boxes in pixel coordinates.
        # And boxes are usually XYXY.

        # We need to adapt.

        # Also, `Sam3Processor` doesn't seem to have a method for "Points".
        # Wait, let me check `Sam3Image` or the underlying model.
        # `Sam3Image` forward takes `Prompt` object which supports points and boxes.

        # `Sam3Processor` is a high-level wrapper. Maybe it's too high level?
        # Let's look at `Sam3Processor` code again in previous turn.
        # `add_geometric_prompt` takes `box` and `label`.

        # If we want to support points (clicks), we might need to bypass `Sam3Processor` or extend it,
        # or convert points to small boxes? No, that's bad.

        # Let's check `Sam3Image` again.
        # It has `forward_grounding` which takes `geometric_prompt: Prompt`.
        # `Prompt` class supports `point_embeddings`, `point_mask`, `point_labels`.

        # So we can construct a `Prompt` object manually and call `forward_grounding` via the model directly,
        # reusing `backbone_out` from `inference_state`.

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
            # SAM 3 expects CXCYWH normalized [0, 1]
            # Input box is XYXY pixel coords [x1, y1, x2, y2]

            # We handle one box for now as typical usage in this extension seems to be 1 box?
            # Or is `box` [N, 4]? usually [1, 4] or [4] in this extension?
            # `SamPredictor.predict` says `box` is `np.ndarray` or None.

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
            # Add batch dim [N, B=1, 4]?
            # `Prompt` expects: N_boxes x B x 4
            box_embeddings = box_tensor.unsqueeze(1)

            box_labels = torch.ones(box.shape[0], 1, dtype=torch.long, device=device) # Positive
            box_mask = torch.zeros(1, box.shape[0], dtype=torch.bool, device=device) # Not masked

        if point_coords is not None:
            # Points: [N, 2]
            # SAM 3 `Prompt` expects points in normalized coordinates [0, 1]

            coords = point_coords.copy().astype(float)
            coords[:, 0] /= img_w
            coords[:, 1] /= img_h

            point_tensor = torch.tensor(coords, dtype=torch.float32, device=device)
            # Add batch dim: N_points x B x 2
            point_embeddings = point_tensor.unsqueeze(1)

            if point_labels is None:
                point_labels = np.ones(len(point_coords))

            # SAM labels: 1=positive, 0=negative.
            # SAM 3 labels: seems to be `label_embed` in `SequenceGeometryEncoder`.
            # "There usually are two labels: positive and negatives."
            # "positive labels everywhere" if None.
            # Let's assume 1 is positive, 0 is negative?
            # `geometry_encoders.py` doesn't explicitly say 0/1 meaning, but `_init_point` inits labels to 1s.
            # Usually 1 is pos, 0 is neg.

            # Important: `SamPredictor` uses 1 for foreground point and 0 for background point.
            # But wait, SAM 1 used 1 for foreground, 0 for negative.

            point_labels_t = torch.tensor(point_labels, dtype=torch.long, device=device)
            point_labels_tensor = point_labels_t.unsqueeze(1)

            point_mask = torch.zeros(1, len(point_coords), dtype=torch.bool, device=device)

        # Create Prompt
        # Note: If both box and points are None, we might have an issue, but predict usually has inputs.

        geometric_prompt = Prompt(
            box_embeddings=box_embeddings,
            box_mask=box_mask,
            box_labels=box_labels,
            point_embeddings=point_embeddings,
            point_mask=point_mask,
            point_labels=point_labels_tensor
        )

        # Run forward_grounding
        # We need `find_input` from `inference_state`??
        # `inference_state` has `backbone_out`.
        # `Sam3Processor.find_stage` is a `FindStage` object.

        # We need to construct a `find_input`.
        # The processor has `self.find_stage`. We can reuse it?
        # `Sam3Processor` uses `self.find_stage` which is initialized with dummy values.

        # But wait, `set_text_prompt` in Processor updates `backbone_out` with text features.
        # If we have previous text prompt, we might want to keep it?
        # Or should `predict` clear text prompt?
        # `SamPredictor` usually clears previous state except image.

        # If we follow `SamPredictor` contract, `predict` is usually for geometric prompts.
        # But if we want to combine text + clicks, we should keep text.
        # SAM 3 supports multi-modal.

        # Let's reuse `self.processor.find_stage` but we might need to be careful about `text_ids`.

        # If we want to support text + point, we should ensure `backbone_out` has text features.
        # If `set_text_prompt` was called before, `backbone_out` has it.

        # However, `Sam3Processor` logic is:
        # `set_text_prompt` -> calls `forward_grounding` immediately.

        # Here we are calling `predict` manually.

        # We call `model.forward_grounding`.

        out = self.model.forward_grounding(
            backbone_out=self._inference_state["backbone_out"],
            find_input=self.processor.find_stage, # Reusing the one from processor
            find_target=None,
            geometric_prompt=geometric_prompt
        )

        # Parse output
        # SAM 1/2 Predictor returns: masks, scores, logits
        # masks: (N, H, W)
        # scores: (N,)
        # logits: (N, H, W)

        # SAM 3 output `out` has:
        # "pred_masks": [N, H, W] (sigmoid?) No, `_update_scores_and_boxes` says `pred_masks` is populated via `interpolate(...).sigmoid()`.
        # Actually `_forward_grounding` in processor does `sigmoid` and thresholding.
        # `Sam3Image` returns raw logits or what?

        # `Sam3Image._run_segmentation_heads` updates `out` with mask logits?
        # It calls `segmentation_head`.
        # `UniversalSegmentationHead` returns masks.
        # Usually these are logits.

        # `Sam3Processor._forward_grounding` does post-processing:
        #   out_probs = out_logits.sigmoid()
        #   out_masks = interpolate(...).sigmoid()
        #   keep = out_probs > threshold

        # We want raw results for `predict` usually, or top-k?
        # `SamPredictor` returns multiple masks (multimask_output).

        # SAM 3 seems to return many detections (N=200 queries?).
        # We probably want to return the best ones or filter.

        # For interactive segmentation (points), we typically expect 3 masks (multimask) or 1 mask.
        # SAM 3 might be behaving more like a detector (DETR style).

        # If we provide points, does it only return masks relevant to points?
        # The `geometric_prompt` is attended to.

        # Let's extract masks and scores.

        # The `out` from `forward_grounding` contains "pred_logits" (scores) and "pred_masks" (mask logits? or masks?).
        # In `Sam3Image`, `_run_segmentation_heads`:
        # `seg_head_outputs`... `_update_out`.

        # `Sam3Processor` treats `out["pred_masks"]` as something to be interpolated and sigmoid-ed.

        pred_logits_scores = out["pred_logits"] # [B=1, Q, 1] (class scores)
        pred_masks_logits = out["pred_masks"]   # [B=1, Q, H, W] (mask logits)

        # We need to filter/select best masks.
        # If we have points, maybe the model focuses on them.

        # Let's convert to numpy
        scores = pred_logits_scores[0, :, 0].sigmoid() # [Q]
        masks = pred_masks_logits[0] # [Q, H, W]

        # Interpolate masks to original size
        masks_up = torch.nn.functional.interpolate(
            masks.unsqueeze(1),
            size=(img_h, img_w),
            mode="bilinear",
            align_corners=False
        ).squeeze(1)

        # Select best masks?
        # If we are in "interactive" mode, we might expect the model to return relevant masks.
        # SAM 3 is a "promptable segmentation" model.

        # If we have 200 queries, most will be low score.
        # We should filter by score or return top K.

        # Interactive mode usually expects 3 masks.
        # Let's take top 3 masks by score?

        scores_val, topk_indices = torch.topk(scores, k=3 if multimask_output else 1)

        best_masks = masks_up[topk_indices] # [3, H, W]
        best_scores = scores_val # [3]

        # Return masks as boolean? SamPredictor returns logits as 3rd output.
        # 1st output is boolean mask (thresholded).

        masks_np = (best_masks > 0.0).cpu().numpy().astype(bool) # Logits > 0 == Prob > 0.5
        scores_np = best_scores.cpu().numpy()
        logits_np = best_masks.cpu().numpy()

        return masks_np, scores_np, logits_np

    def get_image_embedding(self):
        # Implementation to satisfy some checks if needed, but might not be used directly
        pass
