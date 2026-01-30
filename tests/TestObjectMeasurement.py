import cv2
import pytest
from pathlib import Path

from ObjectMeasurement import ObjectMeasurer

A4_MM = (210.0, 297.0)
LETTER_MM = (215.9, 279.4)  # 8.5in x 11in

# https://www.nerdwallet.com/ca/p/article/credit-cards/credit-card-size
CREDIT_CARD_W_CM = 8.56
CREDIT_CARD_H_CM = 5.398

STANDARD_TOLERANCE_CM = 0.33

def cleanup_debug_images(keep_image_path: str) -> None:
    """
    Remove all image files in the directory of `keep_image_path`
    except the file explicitly named by `keep_image_path`.

    This is useful for test debug folders where each test run should
    leave only the most relevant image artifact.
    """
    keep_path = Path(keep_image_path).resolve()
    directory = keep_path.parent

    if not directory.exists():
        return

    for p in directory.iterdir():
        if not p.is_file():
            continue
        if p == keep_path:
            continue
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}:
            try:
                p.unlink()
            except OSError:
                pass

@pytest.mark.parametrize(
    "slug, image_path, reference_size_mm, expected_count, expected_w_cm, expected_h_cm, tol_cm",
    [
        ("one", "../test-images/one/one.jpg", A4_MM, 2, 9.15, 5.0, STANDARD_TOLERANCE_CM),  # adjust tol if your lighting/edges vary
        ("ucard-one","../test-images/ucard-one/ucard-one.jpg", LETTER_MM, 1, CREDIT_CARD_W_CM, CREDIT_CARD_H_CM, STANDARD_TOLERANCE_CM),
        ("ucard-two","../test-images/ucard-two/ucard-two.jpg", LETTER_MM, 1, CREDIT_CARD_W_CM, CREDIT_CARD_H_CM, STANDARD_TOLERANCE_CM),
        ("ucard-one-off-axis","../test-images/ucard-one-off-axis/ucard-one-off-axis.jpg", LETTER_MM, 1, CREDIT_CARD_W_CM, CREDIT_CARD_H_CM, STANDARD_TOLERANCE_CM),
        ("ucard-two-off-axis","../test-images/ucard-two-off-axis/ucard-two-off-axis.jpg", LETTER_MM, 1, CREDIT_CARD_W_CM, CREDIT_CARD_H_CM, STANDARD_TOLERANCE_CM),
    ],
)
def test_1jpg_two_objects_about_9x5(slug, image_path, reference_size_mm, expected_count, expected_w_cm, expected_h_cm, tol_cm):
    img = cv2.imread(image_path)
    assert img is not None, f"Could not load image at path: {image_path}"

    measurer = ObjectMeasurer(scale=3, reference_size_mm=reference_size_mm)
    measurer.slug = slug
    measurer.debug_path = f"../test-images/{slug}"
    cleanup_debug_images(image_path)
    measurements = measurer.measure(img)

    print(f"Expected {expected_count} contours, got {len(measurements)}: ")
    print(f"{[(m.width_cm, m.height_cm) for m in measurements]}")

    assert len(measurements) == expected_count

    # Order-independent: every measurement should be ~9x5 (allow swapped orientation too).
    for m in measurements:
        ok_normal = (
            m.width_cm == pytest.approx(expected_w_cm, abs=tol_cm)
            and m.height_cm == pytest.approx(expected_h_cm, abs=tol_cm)
        )
        ok_swapped = (
            m.width_cm == pytest.approx(expected_h_cm, abs=tol_cm)
            and m.height_cm == pytest.approx(expected_w_cm, abs=tol_cm)
        )
        assert ok_normal or ok_swapped, (
            f"Unexpected measurement (w,h)=({m.width_cm},{m.height_cm}); "
            f"expected about ({expected_w_cm},{expected_h_cm}) +/- {tol_cm} cm"
        )