"""Face alignment integration test — real model, no mocks."""
import os
from pathlib import Path

import cv2
import pytest

from pipeline.aligner import FaceAligner

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FACES_DIR = PROJECT_ROOT / "faces"
OUTPUT_DIR = PROJECT_ROOT / "tests" / "output" / "aligned"
OUTPUT_DIR_API = PROJECT_ROOT / "tests" / "output" / "aligned_api"


def _collect_face_images() -> list[Path]:
    """Collect all supported image files from faces/ directory."""
    if not FACES_DIR.exists():
        return []
    return [
        p for p in FACES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


@pytest.mark.integration
def test_face_alignment_outputs_aligned_images():
    """End-to-end: align all faces/ images, save aligned output for visual inspection."""
    face_images = _collect_face_images()

    if not face_images:
        pytest.skip("No face images found in faces/ directory")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    aligner = FaceAligner(backend="insightface")
    try:
        for img_path in face_images:
            img = cv2.imread(str(img_path))
            assert img is not None, f"Failed to read image: {img_path}"

            result = aligner.align(img)
            assert result is not None, (
                f"Alignment returned None for {img_path.name} — "
                "check that the image contains a detectable face"
            )

            output_path = OUTPUT_DIR / f"{img_path.stem}_aligned.jpg"
            cv2.imwrite(str(output_path), result.aligned_face)
    finally:
        aligner.close()


@pytest.mark.integration
def test_face_alignment_api_backend_outputs_aligned_images():
    """End-to-end with API backend: align faces/ images via FaceApiClient, save aligned output for comparison."""
    face_images = _collect_face_images()

    if not face_images:
        pytest.skip("No face images found in faces/ directory")

    OUTPUT_DIR_API.mkdir(parents=True, exist_ok=True)

    aligner = FaceAligner(backend="api")
    try:
        for img_path in face_images:
            img = cv2.imread(str(img_path))
            assert img is not None, f"Failed to read image: {img_path}"

            result = aligner.align(img)
            assert result is not None, (
                f"Alignment returned None for {img_path.name} — "
                "check that the API response contains a detectable face"
            )

            output_path = OUTPUT_DIR_API / f"{img_path.stem}_aligned_api.jpg"
            cv2.imwrite(str(output_path), result.aligned_face)
    finally:
        aligner.close()
