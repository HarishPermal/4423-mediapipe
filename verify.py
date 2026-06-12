import tempfile
from pathlib import Path

from PIL import Image
from mediapipe_model_maker import object_detector


POSITIVE_XML = """\
<annotation>
  <filename>positive.jpg</filename>
  <size><width>64</width><height>64</height><depth>3</depth></size>
  <object>
    <name>square</name>
    <bndbox><xmin>16</xmin><ymin>16</ymin><xmax>48</xmax><ymax>48</ymax></bndbox>
  </object>
</annotation>
"""

NEGATIVE_XML = """\
<annotation>
  <filename>negative.jpg</filename>
  <size><width>64</width><height>64</height><depth>3</depth></size>
</annotation>
"""


def create_pascal_voc_dataset(root: Path) -> None:
    images = root / "images"
    annotations = root / "Annotations"
    images.mkdir(parents=True)
    annotations.mkdir(parents=True)

    Image.new("RGB", (64, 64), "white").save(images / "positive.jpg")
    Image.new("RGB", (64, 64), "gray").save(images / "negative.jpg")
    (annotations / "positive.xml").write_text(POSITIVE_XML)
    (annotations / "negative.xml").write_text(NEGATIVE_XML)


def main() -> int:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir) / "voc"
        create_pascal_voc_dataset(root)
        dataset = object_detector.Dataset.from_pascal_voc_folder(
            str(root), cache_dir=str(Path(temp_dir) / "cache")
        )
        box_counts = sorted(
            int(example["groundtruth_boxes"].shape[-2])
            for example in dataset.gen_tf_dataset()
        )

    print("Input images: 2")
    print(f"Loaded images: {dataset.size}")
    print(f"Ground-truth box counts: {box_counts}")

    if dataset.size != 2 or box_counts != [0, 1]:
        raise AssertionError("Background-only image was not retained")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
