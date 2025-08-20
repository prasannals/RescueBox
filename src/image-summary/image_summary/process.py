from pathlib import Path
from typing import Iterable, Set

from .model import ensure_model_exists, describe_image

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tiff"}


def iter_image_files(directory: Path) -> Iterable[Path]:
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
            yield path


def process_images(model: str, input_dir: str, output_dir: str) -> Set[str]:
    ensure_model_exists(model)
    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError(f"Input directory '{input_dir}' does not exist.")
    if not input_path.is_dir():
        raise ValueError(f"Input directory '{input_dir}' is not a directory.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    processed_files: Set[str] = set()
    for image_path in iter_image_files(input_path):
        summary_text = describe_image(model, str(image_path))
        # Include the original filename with extension to avoid collisions
        out_file = output_path / (image_path.name + ".txt")
        out_file.write_text(summary_text, encoding="utf-8")
        processed_files.add(str(out_file))

    return processed_files


