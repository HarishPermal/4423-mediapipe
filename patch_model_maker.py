"""Patch MediaPipe Model Maker to retain background-only detector images."""

from pathlib import Path
import sysconfig


path = (
    Path(sysconfig.get_paths()["purelib"])
    / "mediapipe_model_maker"
    / "python"
    / "vision"
    / "object_detector"
    / "dataset_util.py"
)
if not path.is_file():
    raise RuntimeError(f"Model Maker dataset utility was not found at {path}")
source = path.read_text()

old_helper = """  bbox_feature_dict = {
      'image/object/bbox/xmin': tfrecord_lib.convert_to_feature(data['xmin']),
      'image/object/bbox/xmax': tfrecord_lib.convert_to_feature(data['xmax']),
      'image/object/bbox/ymin': tfrecord_lib.convert_to_feature(data['ymin']),
      'image/object/bbox/ymax': tfrecord_lib.convert_to_feature(data['ymax']),
      'image/object/class/label': tfrecord_lib.convert_to_feature(
          data['category_id']
      ),
  }
"""
new_helper = """  if not data['xmin']:
    # convert_to_feature cannot infer a feature type from an empty list.
    bbox_feature_dict = {
        'image/object/bbox/xmin': tf.train.Feature(
            float_list=tf.train.FloatList(value=[])
        ),
        'image/object/bbox/xmax': tf.train.Feature(
            float_list=tf.train.FloatList(value=[])
        ),
        'image/object/bbox/ymin': tf.train.Feature(
            float_list=tf.train.FloatList(value=[])
        ),
        'image/object/bbox/ymax': tf.train.Feature(
            float_list=tf.train.FloatList(value=[])
        ),
        'image/object/class/label': tf.train.Feature(
            int64_list=tf.train.Int64List(value=[])
        ),
    }
  else:
    bbox_feature_dict = {
        'image/object/bbox/xmin': tfrecord_lib.convert_to_feature(data['xmin']),
        'image/object/bbox/xmax': tfrecord_lib.convert_to_feature(data['xmax']),
        'image/object/bbox/ymin': tfrecord_lib.convert_to_feature(data['ymin']),
        'image/object/bbox/ymax': tfrecord_lib.convert_to_feature(data['ymax']),
        'image/object/class/label': tfrecord_lib.convert_to_feature(
            data['category_id']
        ),
    }
"""
old_skip = """      if not data['xmin']:
        # Skip examples which have no valid annotations
        continue
"""

if new_helper in source and old_skip not in source:
    print(f"Patch already applied: {path}")
else:
    if old_helper not in source:
        raise RuntimeError(f"Expected bbox helper was not found in {path}")
    if source.count(old_skip) != 1:
        raise RuntimeError(f"Expected one Pascal VOC skip block in {path}")
    source = source.replace(old_helper, new_helper, 1)
    source = source.replace(old_skip, "", 1)
    path.write_text(source)
    print(f"Patched: {path}")
