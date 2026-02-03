# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

if not GlobalHydra.instance().is_initialized():
    # This points to the `sam2_configs` directory in the root of the repo
    # Assuming this file is in `sam2/__init__.py`, then `..` is the repo root.
    config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sam2_configs")
    initialize_config_dir(config_dir=config_dir, version_base="1.2")
