import cv2
import pytest
from pathlib import Path

from ObjectMeasurer import ObjectMeasurer

# SHORT SIDE / X-AXIS FIRST
A4_MM = (210.0, 297.0)
LETTER_MM = (215.9, 279.4)  # 8.5in x 11in
PORTRAIT_POSTER_BOARD_MM = (561.975, 711.2)
ODDBALL_BLACK_POSTER_BOARD_MM = (508, 752.475)
DOUBLE_STACKED_LANDSCAPE_POSTER_BOARD_MM = (711.2, 1123.95)
LIGHTBOX_MAT_MM = (746.125, 1098.55)


# https://www.nerdwallet.com/ca/p/article/credit-cards/credit-card-size
CREDIT_CARD_W_CM = 8.56
CREDIT_CARD_H_CM = 5.398

STANDARD_TOLERANCE_CM = 0.33
STANDARD_TOLERANCE_SHIRT_CM = 3

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
    "slug, scale, image_path, reference_size_mm, tol_cm",
    [
        # Works in Production
        ("ucard-one", 1,"../test-images/ucard-one/ucard-one.jpg", LETTER_MM, STANDARD_TOLERANCE_CM),
        ("iswic-folded", 1, "../test-images/iswic-folded/img_6a1db6e03ad2a7.16505140.jpg", PORTRAIT_POSTER_BOARD_MM, STANDARD_TOLERANCE_SHIRT_CM),
        ("iswic-folded2", 1, "../test-images/iswic-folded2/img_6a1db77d7f1973.02751931.jpg", PORTRAIT_POSTER_BOARD_MM, STANDARD_TOLERANCE_SHIRT_CM),

        # Fails in production
        ("goldy-lightblue", 1, "../test-images/goldy-lightblue/img_6a1db5fd6d4312.38941970.jpg", PORTRAIT_POSTER_BOARD_MM, STANDARD_TOLERANCE_SHIRT_CM),
        ("nike-letter", 1, "../test-images/nike-letter/img_6a7b67eeb4dcd8.69842033.jpg", LETTER_MM, STANDARD_TOLERANCE_SHIRT_CM),
    ],
)
def test_images_processes(slug, scale, image_path, reference_size_mm, tol_cm):
    img = cv2.imread(image_path)
    assert img is not None, f"Could not load image at path: {image_path}"

    measurer = ObjectMeasurer(scale = scale, reference_size_mm=reference_size_mm)
    measurer.slug = slug
    measurer.debug_path = f"../test-images/{slug}"
    cleanup_debug_images(image_path)
    measurements = measurer.measure(img)

    assert measurer.debug['object_contour_svg'] is not None

@pytest.mark.parametrize(
    "slug, scale, image_path, reference_size_mm, expected_count, expected_w_cm, expected_h_cm, tol_cm",
    [
        ("one", 1, "../test-images/one/one.jpg", A4_MM, 2, 9.15, 5.0, STANDARD_TOLERANCE_CM),  # adjust tol if your lighting/edges vary
        ("ucard-one", 1,"../test-images/ucard-one/ucard-one.jpg", LETTER_MM, 1, CREDIT_CARD_W_CM, CREDIT_CARD_H_CM, STANDARD_TOLERANCE_CM),
        ("ucard-one", 2,"../test-images/ucard-one/ucard-one.jpg", LETTER_MM, 1, CREDIT_CARD_W_CM, CREDIT_CARD_H_CM, STANDARD_TOLERANCE_CM),
        ("ucard-two", 1, "../test-images/ucard-two/ucard-two.jpg", LETTER_MM, 1, CREDIT_CARD_W_CM, CREDIT_CARD_H_CM, STANDARD_TOLERANCE_CM),
        ("ucard-one-off-axis", 1,"../test-images/ucard-one-off-axis/ucard-one-off-axis.jpg", LETTER_MM, 1, CREDIT_CARD_W_CM, CREDIT_CARD_H_CM, STANDARD_TOLERANCE_CM),
        ("ucard-two-off-axis", 1,"../test-images/ucard-two-off-axis/ucard-two-off-axis.jpg", LETTER_MM, 1, CREDIT_CARD_W_CM, CREDIT_CARD_H_CM, STANDARD_TOLERANCE_CM),

        ("iswic", 1, "../test-images/iswic/iswic.jpg", LIGHTBOX_MAT_MM, 1, 70.485, 69.5325, STANDARD_TOLERANCE_SHIRT_CM),
        ("goldy", 1, "../test-images/goldy/goldy.jpg", ODDBALL_BLACK_POSTER_BOARD_MM, 1, 45.72, 40.5, STANDARD_TOLERANCE_SHIRT_CM),
        ("cherokee", 1, "../test-images/cherokee/cherokee.jpg", ODDBALL_BLACK_POSTER_BOARD_MM, 1, 40.9575, 51.435, STANDARD_TOLERANCE_SHIRT_CM),

        ("iswic", 2, "../test-images/iswic/iswic.jpg", LIGHTBOX_MAT_MM, 1, 70.485, 69.5325, STANDARD_TOLERANCE_SHIRT_CM),
        ("iswic-folded", 2, "../test-images/iswic-folded/img_6a1db6e03ad2a7.16505140.jpg", PORTRAIT_POSTER_BOARD_MM, 1, 70.485, 69.5325, STANDARD_TOLERANCE_SHIRT_CM),
    ],
)
def test_1jpg_two_objects_about_9x5(slug, scale, image_path, reference_size_mm, expected_count, expected_w_cm, expected_h_cm, tol_cm):
    img = cv2.imread(image_path)
    assert img is not None, f"Could not load image at path: {image_path}"

    measurer = ObjectMeasurer(scale = scale, reference_size_mm=reference_size_mm)
    measurer.slug = slug
    measurer.debug_path = f"../test-images/{slug}"
    cleanup_debug_images(image_path)
    measurements = measurer.measure(img)

    print(f"Expected {expected_count} contours, got {len(measurements)}: ")
    print(f"{[(m.width_cm, m.height_cm) for m in measurements]}")

    width_cm, height_cm = measurements[0].width_cm, measurements[0].height_cm
    print(f"Width: {width_cm}, Height: {height_cm}")

    # assert len(measurements) == expected_count

    # print(measurer.debug)

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
        break

    print(measurer.debug['object_contour_svg'])
    assert measurer.debug['object_contour_svg'] is not None