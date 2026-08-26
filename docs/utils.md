# Utilities

Small helpers shared across the model pages: loading an image or video into the array shape
the processors expect, drawing results, and the class-name lists that turn label ids back
into words. One page per module in `zeromodels/utils/`.

| Page | Module | Covers |
|---|---|---|
| [**Image utils**](utils_image.md) | `image_util.py` | `load_image`, `get_data_format`, `standardize_input_shape` |
| [**Video utils**](utils_video.md) | `video_util.py` | `load_video`, `VideoMetadata`, `sample_frames`, the three decoders |
| [**Visualization utils**](utils_visualization.md) | `visualization_util.py` | `plot_detections`, `plot_segmentation`, `plot_depth`, `plot_sam_masks` |
| [**Class labels**](utils_labels.md) | `labels_util.py` | COCO, ADE20K, Cityscapes, and VOC class-name lists |

```python
from zeromodels.utils import (
    load_image,
    get_data_format,
    standardize_input_shape,
    load_video,
    sample_frames,
    VideoMetadata,
    VIDEO_DECODERS,
    plot_detections,
    plot_segmentation,
    plot_depth,
    plot_sam_masks,
)
from zeromodels.utils.labels_util import COCO_80_CLASSES, ADE20K_150_CLASSES
```

Everything here is optional. The processors accept plain PIL images and NumPy arrays, so
you can skip these entirely and use PIL, OpenCV, and Matplotlib directly.

See also [Main Classes](main_classes.md) for the model, processor, and generation bases.
