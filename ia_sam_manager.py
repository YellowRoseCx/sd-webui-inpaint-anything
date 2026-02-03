import os
import platform
from functools import partial

import torch
from modules import devices

from fast_sam import FastSamAutomaticMaskGenerator, fast_sam_model_registry
from ia_check_versions import ia_check_versions
from ia_config import get_webui_setting
from ia_logging import ia_logging
from ia_threading import torch_default_load_cd
from mobile_sam import SamAutomaticMaskGenerator as SamAutomaticMaskGeneratorMobile
from mobile_sam import SamPredictor as SamPredictorMobile
from mobile_sam import sam_model_registry as sam_model_registry_mobile
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2
from segment_anything_fb import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
from segment_anything_hq import SamAutomaticMaskGenerator as SamAutomaticMaskGeneratorHQ
from segment_anything_hq import SamPredictor as SamPredictorHQ
from segment_anything_hq import sam_model_registry as sam_model_registry_hq

# SAM 3
from sam3.model_builder import build_sam3_image_model
from ia_sam3_wrapper import Sam3Wrapper

def check_bfloat16_support() -> bool:
    if torch.cuda.is_available():
        compute_capability = torch.cuda.get_device_capability(torch.cuda.current_device())
        if compute_capability[0] >= 8:
            ia_logging.debug("The CUDA device supports bfloat16")
            return True
        else:
            ia_logging.debug("The CUDA device does not support bfloat16")
            return False
    else:
        ia_logging.debug("CUDA is not available")
        return False


def partial_from_end(func, /, *fixed_args, **fixed_kwargs):
    def wrapper(*args, **kwargs):
        updated_kwargs = {**fixed_kwargs, **kwargs}
        return func(*args, *fixed_args, **updated_kwargs)
    return wrapper


def rename_args(func, arg_map):
    def wrapper(*args, **kwargs):
        new_kwargs = {arg_map.get(k, k): v for k, v in kwargs.items()}
        return func(*args, **new_kwargs)
    return wrapper


arg_map = {"checkpoint": "ckpt_path"}
rename_build_sam2 = rename_args(build_sam2, arg_map)
end_kwargs = dict(device="cpu", mode="eval", hydra_overrides_extra=[], apply_postprocessing=False)
sam2_model_registry = {
    "sam2_hiera_large": partial(partial_from_end(rename_build_sam2, **end_kwargs), "sam2_hiera_l.yaml"),
    "sam2_hiera_base_plus": partial(partial_from_end(rename_build_sam2, **end_kwargs), "sam2_hiera_b+.yaml"),
    "sam2_hiera_small": partial(partial_from_end(rename_build_sam2, **end_kwargs), "sam2_hiera_s.yaml"),
    "sam2_hiera_tiny": partial(partial_from_end(rename_build_sam2, **end_kwargs), "sam2_hiera_t.yaml"),
    "sam2.1_hiera_large": partial(partial_from_end(rename_build_sam2, **end_kwargs), "sam2.1_hiera_l.yaml"),
    "sam2.1_hiera_base_plus": partial(partial_from_end(rename_build_sam2, **end_kwargs), "sam2.1_hiera_b+.yaml"),
    "sam2.1_hiera_small": partial(partial_from_end(rename_build_sam2, **end_kwargs), "sam2.1_hiera_s.yaml"),
    "sam2.1_hiera_tiny": partial(partial_from_end(rename_build_sam2, **end_kwargs), "sam2.1_hiera_t.yaml"),
}

# SAM 3
sam3_model_registry = {
    "sam3_large": partial(build_sam3_image_model, device="cpu", eval_mode=True, load_from_HF=False),
}


@torch_default_load_cd()
def get_sam_mask_generator(
    sam_checkpoint, anime_style_chk=False,
    pred_iou_thresh=0.88, stability_score_thresh=0.95,
    stability_score_offset=1.0, box_nms_thresh=0.7,
    crop_n_layers=0, crop_nms_thresh=0.7,
    crop_overlap_ratio=512 / 1500, crop_n_points_downscale_factor=1,
    min_mask_region_area=0,
    sam_text_prompt=None, # Added
):
    """Get SAM mask generator.

    Args:
        sam_checkpoint (str): SAM checkpoint path
        anime_style_chk (bool): anime style check
        pred_iou_thresh (float): prediction iou threshold
        stability_score_thresh (float): stability score threshold
        stability_score_offset (float): stability score offset
        box_nms_thresh (float): box nms threshold
        crop_n_layers (int): crop n layers
        crop_nms_thresh (float): crop nms threshold
        crop_overlap_ratio (float): crop overlap ratio
        crop_n_points_downscale_factor (int): crop n points downscale factor
        min_mask_region_area (int): min mask region area
        sam_text_prompt (str): SAM 3 text prompt

    Returns:
        SamAutomaticMaskGenerator or None: SAM mask generator
    """
    points_per_batch = 64
    SamAutomaticMaskGeneratorLocal = None
    if "_hq_" in os.path.basename(sam_checkpoint):
        model_type = os.path.basename(sam_checkpoint)[7:12]
        sam_model_registry_local = sam_model_registry_hq
        SamAutomaticMaskGeneratorLocal = SamAutomaticMaskGeneratorHQ
        points_per_batch = 32
    elif "FastSAM" in os.path.basename(sam_checkpoint):
        model_type = os.path.splitext(os.path.basename(sam_checkpoint))[0]
        sam_model_registry_local = fast_sam_model_registry
        SamAutomaticMaskGeneratorLocal = FastSamAutomaticMaskGenerator
        points_per_batch = None
    elif "mobile_sam" in os.path.basename(sam_checkpoint):
        model_type = "vit_t"
        sam_model_registry_local = sam_model_registry_mobile
        SamAutomaticMaskGeneratorLocal = SamAutomaticMaskGeneratorMobile
        points_per_batch = 64
    elif "sam2_" in os.path.basename(sam_checkpoint) or "sam2.1_" in os.path.basename(sam_checkpoint):
        model_type = os.path.splitext(os.path.basename(sam_checkpoint))[0]
        sam_model_registry_local = sam2_model_registry
        SamAutomaticMaskGeneratorLocal = SAM2AutomaticMaskGenerator
        points_per_batch = 128
    elif "sam3_" in os.path.basename(sam_checkpoint):
        model_type = os.path.splitext(os.path.basename(sam_checkpoint))[0]
        sam_model_registry_local = sam3_model_registry
        SamAutomaticMaskGeneratorLocal = Sam3Wrapper
        points_per_batch = None # Not used for SAM 3 in wrapper (yet)
    else:
        model_type = os.path.basename(sam_checkpoint)[4:9]
        sam_model_registry_local = sam_model_registry
        SamAutomaticMaskGeneratorLocal = SamAutomaticMaskGenerator
        points_per_batch = 24

    if ("sam2_" in model_type or "sam2.1_" in model_type):
        sam2_gen_kwargs = dict(
            points_per_side=32,
            points_per_batch=points_per_batch,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            stability_score_offset=stability_score_offset,
            box_nms_thresh=box_nms_thresh,
            crop_n_layers=crop_n_layers,
            crop_nms_thresh=crop_nms_thresh,
            crop_overlap_ratio=crop_overlap_ratio,
            crop_n_points_downscale_factor=crop_n_points_downscale_factor,
            min_mask_region_area=min_mask_region_area,
        )
        if platform.system() == "Darwin":
            sam2_gen_kwargs.update(dict(points_per_side=32, points_per_batch=64, crop_n_points_downscale_factor=1))

    if os.path.isfile(sam_checkpoint):
        # Handle model creation
        if "sam3_" in model_type:
             # SAM 3 requires checkpoint path in kwargs if not loading from HF, but we are loading local file
             # build_sam3_image_model arg is `checkpoint_path`
             sam = sam_model_registry_local[model_type](checkpoint_path=sam_checkpoint)
        else:
             sam = sam_model_registry_local[model_type](checkpoint=sam_checkpoint)

        # Move to device
        if platform.system() == "Darwin":
            if "FastSAM" in os.path.basename(sam_checkpoint) or not ia_check_versions.torch_mps_is_available:
                sam.to(device=torch.device("cpu"))
            else:
                sam.to(device=torch.device("mps"))
        else:
            if get_webui_setting("inpaint_anything_sam_oncpu", False):
                ia_logging.info("SAM is running on CPU... (the option has been checked)")
                sam.to(device=devices.cpu)
            else:
                sam.to(device=devices.device)

        # Generator init
        if "sam3_" in model_type:
            sam_gen_kwargs = dict(model=sam, text_prompt=sam_text_prompt)
        else:
            sam_gen_kwargs = dict(
                model=sam, points_per_batch=points_per_batch, pred_iou_thresh=pred_iou_thresh, stability_score_thresh=stability_score_thresh)

        if "sam2_" in model_type or "sam2.1_" in model_type:
            sam_gen_kwargs.update(sam2_gen_kwargs)

        sam_mask_generator = SamAutomaticMaskGeneratorLocal(**sam_gen_kwargs)
    else:
        sam_mask_generator = None

    return sam_mask_generator


@ torch_default_load_cd()
def get_sam_predictor(sam_checkpoint):
    """Get SAM predictor.

    Args:
        sam_checkpoint (str): SAM checkpoint path

    Returns:
        SamPredictor or None: SAM predictor
    """
    SamPredictorLocal = None
    if "_hq_" in os.path.basename(sam_checkpoint):
        model_type = os.path.basename(sam_checkpoint)[7:12]
        sam_model_registry_local = sam_model_registry_hq
        SamPredictorLocal = SamPredictorHQ
    elif "FastSAM" in os.path.basename(sam_checkpoint):
        raise NotImplementedError("FastSAM predictor is not implemented yet.")
    elif "mobile_sam" in os.path.basename(sam_checkpoint):
        model_type = "vit_t"
        sam_model_registry_local = sam_model_registry_mobile
        SamPredictorLocal = SamPredictorMobile
    elif "sam3_" in os.path.basename(sam_checkpoint):
        model_type = os.path.splitext(os.path.basename(sam_checkpoint))[0]
        sam_model_registry_local = sam3_model_registry
        SamPredictorLocal = Sam3Wrapper
    elif "sam2_" in os.path.basename(sam_checkpoint) or "sam2.1_" in os.path.basename(sam_checkpoint):
        # SAM 2 predictor uses SAM2ImagePredictor logic, usually embedded in extension via SAM 2 wrapper or similar?
        # The existing code didn't seem to have explicit SAM 2 predictor logic in the original file
        # except falling back to something?
        # Wait, the original code had:
        # elif "sam2_" ...: model_type = ...; sam_model_registry_local = sam2_model_registry; SamAutomaticMaskGeneratorLocal = SAM2AutomaticMaskGenerator
        # BUT for predictor:
        # It checked "FastSAM", "mobile_sam", "_hq_".
        # It fell back to `sam_model_registry` and `SamPredictor`.
        # SAM 2 models in `sam_model_registry`? No, they are in `sam2_model_registry`.
        # The original code MIGHT NOT have supported SAM 2 predictor fully in `get_sam_predictor`
        # OR it relied on `SamPredictor` being compatible (unlikely)
        # OR it relied on `sam2` models being loadable by `sam_model_registry` (also unlikely).
        # However, I should check if I missed SAM 2 in predictor logic in original file.
        # Original file:
        # ...
        # elif "mobile_sam" ...
        # else: ... model_type ... sam_model_registry ... SamPredictor
        #
        # It seems SAM 2 predictor support might be missing or implicit?
        # Let's verify `sam2_model_registry` usage. It was defined but used in `get_sam_mask_generator`.
        # It was NOT used in `get_sam_predictor`.
        # If I am to add SAM 3, I should ensure I don't break existing behavior.
        # But if existing behavior for SAM 2 was broken, I can fix it or leave it.
        # Let's focus on SAM 3.

        # NOTE: SAM 2 support in this repo might be limited to Automatic Mask Generator?
        # Let's check `scripts/inpaint_anything.py` to see where `get_sam_predictor` is used.
        # It is NOT used in `scripts/inpaint_anything.py`. `run_sam` calls `inpalib.generate_sam_masks`.
        # `inpalib.generate_sam_masks` calls `get_sam_mask_generator`.

        # `get_sam_predictor` seems unused in the main flow of `inpaint_anything.py`.
        # It might be used by other scripts or future features.
        # I will implement SAM 3 predictor support just in case.
        pass

    else:
        model_type = os.path.basename(sam_checkpoint)[4:9]
        sam_model_registry_local = sam_model_registry
        SamPredictorLocal = SamPredictor

    # Re-evaluate logic for default (SAM 1) vs others
    if "sam3_" not in os.path.basename(sam_checkpoint) and \
       "sam2_" not in os.path.basename(sam_checkpoint) and \
       "sam2.1_" not in os.path.basename(sam_checkpoint) and \
       "_hq_" not in os.path.basename(sam_checkpoint) and \
       "FastSAM" not in os.path.basename(sam_checkpoint) and \
       "mobile_sam" not in os.path.basename(sam_checkpoint):
           # Default SAM 1
           model_type = os.path.basename(sam_checkpoint)[4:9]
           sam_model_registry_local = sam_model_registry
           SamPredictorLocal = SamPredictor

    if os.path.isfile(sam_checkpoint):
        if "sam3_" in model_type:
            sam = sam_model_registry_local[model_type](checkpoint_path=sam_checkpoint)
        else:
            # Check if model_type is in registry
            if model_type not in sam_model_registry_local:
                 # Fallback or error?
                 # If SAM 2, we need sam2 registry.
                 if "sam2_" in os.path.basename(sam_checkpoint) or "sam2.1_" in os.path.basename(sam_checkpoint):
                      sam_model_registry_local = sam2_model_registry
                      model_type = os.path.splitext(os.path.basename(sam_checkpoint))[0]
                      from sam2.sam2_image_predictor import SAM2ImagePredictor
                      SamPredictorLocal = SAM2ImagePredictor # SAM 2 predictor wrapper class

            sam = sam_model_registry_local[model_type](checkpoint=sam_checkpoint)

        if platform.system() == "Darwin":
            if "FastSAM" in os.path.basename(sam_checkpoint) or not ia_check_versions.torch_mps_is_available:
                sam.to(device=torch.device("cpu"))
            else:
                sam.to(device=torch.device("mps"))
        else:
            if get_webui_setting("inpaint_anything_sam_oncpu", False):
                ia_logging.info("SAM is running on CPU... (the option has been checked)")
                sam.to(device=devices.cpu)
            else:
                sam.to(device=devices.device)

        if "sam3_" in model_type:
            sam_predictor = SamPredictorLocal(sam)
        elif "sam2_" in model_type or "sam2.1_" in model_type:
             # SAM 2 predictor init might be different
             sam_predictor = SamPredictorLocal(sam)
        else:
            sam_predictor = SamPredictorLocal(sam)
    else:
        sam_predictor = None

    return sam_predictor
