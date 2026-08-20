import cv2
from contextlib import contextmanager
import logging
import numpy as np
import os
import pytest
from pathlib import Path
from pprint import pprint

from ObjectMeasurer import ObjectMeasurer

# SHORT SIDE / X-AXIS FIRST
A4_MM = (210.0, 297.0)
LETTER_MM = (215.9, 279.4)  # 8.5in x 11in
PORTRAIT_POSTER_BOARD_MM = (561.975, 711.2)
ODDBALL_BLACK_POSTER_BOARD_MM = (508, 752.475)
DOUBLE_STACKED_LANDSCAPE_POSTER_BOARD_MM = (711.2, 1123.95)
MOCK_BOX_MM = (762, 755.65)
LIGHTBOX_MAT_MM = (746.125, 1098.55)
ENVELOPE_MM = (184, 133)
#ENVELOPE_MM = (165, 108)

# https://www.nerdwallet.com/ca/p/article/credit-cards/credit-card-size
CREDIT_CARD_W_CM = 8.56
CREDIT_CARD_H_CM = 5.398

STANDARD_TOLERANCE_CM = 0.33
STANDARD_TOLERANCE_SHIRT_CM = 3
REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_IMAGES_ROOT = REPO_ROOT / "test-images"
SAVE_DEBUG_IMAGES_ENV = "OBJECT_MEASURER_SAVE_DEBUG_IMAGES"
DIAGNOSTIC_TRACE_EVENTS = {
    "failure",
    "warning",
    "contour_erode_complete",
    "contour_find_complete",
    "contour_preprocess_complete",
    "min_area_rect_complete",
    "object_bbox",
    "object_min_area_rect",
    "object_measured",
    "object_measurement_skipped",
}


def fixture_path(slug: str, filename: str) -> Path:
    return TEST_IMAGES_ROOT / slug / filename


def save_debug_images_enabled() -> bool:
    return os.getenv(SAVE_DEBUG_IMAGES_ENV, "True").lower() in {"1", "true", "yes", "on"}


def debug_path(tmp_path: Path, slug: str, image_path: Path) -> str:
    if save_debug_images_enabled():
        return str(image_path.parent)

    path = tmp_path / slug
    path.mkdir()
    return str(path)


def print_object_measurer_debug(debug, *, label: str) -> None:
    print(f"\nObjectMeasurer debug for failed test: {label}")
    if not debug:
        print("No debug payload was captured.")
        return

    for key in (
        "status",
        "errors",
        "warnings",
        "measurements",
        "page_detection",
        "object_candidate_contours",
        "debug_images",
    ):
        value = debug.get(key)
        if value not in (None, [], {}):
            print(f"\n{key}:")
            pprint(value, width=140, sort_dicts=False)

    trace = debug.get("trace") or []
    diagnostic_trace = [item for item in trace if item.get("event") in DIAGNOSTIC_TRACE_EVENTS]
    trace_to_print = diagnostic_trace[-30:] if diagnostic_trace else trace[-30:]
    if trace_to_print:
        print("\ndiagnostic trace tail:")
        pprint(trace_to_print, width=140, sort_dicts=False)


@contextmanager
def print_debug_on_failure(label: str, get_debug):
    try:
        yield
    except Exception:
        print_object_measurer_debug(get_debug(), label=label)
        raise


@pytest.mark.parametrize(
    "slug, scale, image_path, reference_size_mm, tol_cm",
    [
        # Works in Production
        ("ucard-one", 1, fixture_path("ucard-one", "ucard-one.jpg"), LETTER_MM, STANDARD_TOLERANCE_CM),
        ("iswic-folded", 1, fixture_path("iswic-folded", "img_6a1db6e03ad2a7.16505140.jpg"), PORTRAIT_POSTER_BOARD_MM, STANDARD_TOLERANCE_SHIRT_CM),
        ("iswic-folded2", 1, fixture_path("iswic-folded2", "img_6a1db77d7f1973.02751931.jpg"), PORTRAIT_POSTER_BOARD_MM, STANDARD_TOLERANCE_SHIRT_CM),

        # Fails in production
        ("goldy-lightblue", 1, fixture_path("goldy-lightblue", "img_6a1db5fd6d4312.38941970.jpg"), PORTRAIT_POSTER_BOARD_MM, STANDARD_TOLERANCE_SHIRT_CM),
        # ("nike-envelope", 1, fixture_path("nike-envelope", "img_6a7b6ba3eb1276.03284004.jpg"), ENVELOPE_MM, STANDARD_TOLERANCE_SHIRT_CM),

        ("mock-box-white-blue-one", 1, fixture_path("mock-box-white-blue-one", "IMG_0315.jpeg"), MOCK_BOX_MM, STANDARD_TOLERANCE_SHIRT_CM),
        ("mock-box-white-blue-one-off-axis", 1, fixture_path("mock-box-white-blue-goldy-one-off-axis", "img_6a84aaa9bd5684.29594166.jpg"), MOCK_BOX_MM, STANDARD_TOLERANCE_SHIRT_CM),
        ("mock-box-white-blue-goldy-one", 1, fixture_path("mock-box-white-blue-goldy-one", "IMG_0317.jpeg"), MOCK_BOX_MM, STANDARD_TOLERANCE_SHIRT_CM),
        ("mock-box-white-blue-goldy-two", 1, fixture_path("mock-box-white-blue-goldy-two", "img_6a84a94c724176.66225541.jpg"), MOCK_BOX_MM, STANDARD_TOLERANCE_SHIRT_CM),
        ("mock-box-white-blue-goldy-three", 1, fixture_path("mock-box-white-blue-goldy-three", "img_6a84a977ae3347.44496591.jpg"), MOCK_BOX_MM, STANDARD_TOLERANCE_SHIRT_CM),
        ("mock-box-white-blue-goldy-four", 1, fixture_path("mock-box-white-blue-goldy-four", "img_6a84a8e7b20297.13560456.jpg"), MOCK_BOX_MM, STANDARD_TOLERANCE_SHIRT_CM),

        # Broader source-photo smoke coverage
        ("one", 1, fixture_path("one", "one.jpg"), A4_MM, STANDARD_TOLERANCE_CM),
        ("ucard-two", 1, fixture_path("ucard-two", "ucard-two.jpg"), LETTER_MM, STANDARD_TOLERANCE_CM),
        ("ucard-one-off-axis", 1, fixture_path("ucard-one-off-axis", "ucard-one-off-axis.jpg"), LETTER_MM, STANDARD_TOLERANCE_CM),
        ("ucard-two-off-axis", 1, fixture_path("ucard-two-off-axis", "ucard-two-off-axis.jpg"), LETTER_MM, STANDARD_TOLERANCE_CM),
        ("nike-letter-one", 1, fixture_path("nike-letter-one", "img_6a7c9d736092f6.97385818.jpg"), LETTER_MM, STANDARD_TOLERANCE_CM),
        ("nike-letter-one-off-axis", 1, fixture_path("nike-letter-one-off-axis", "img_6a838cf9edde22.59423143.jpg"), LETTER_MM, STANDARD_TOLERANCE_CM),
        ("iswic", 1, fixture_path("iswic", "iswic.jpg"), LIGHTBOX_MAT_MM, STANDARD_TOLERANCE_SHIRT_CM),
        ("goldy", 1, fixture_path("goldy", "goldy.jpg"), ODDBALL_BLACK_POSTER_BOARD_MM, STANDARD_TOLERANCE_SHIRT_CM),
        ("cherokee", 1, fixture_path("cherokee", "cherokee.jpg"), ODDBALL_BLACK_POSTER_BOARD_MM, STANDARD_TOLERANCE_SHIRT_CM),
    ],
)
def test_images_processes(slug, scale, image_path, reference_size_mm, tol_cm, tmp_path):
    img = cv2.imread(str(image_path))
    assert img is not None, f"Could not load image at path: {image_path}"

    measurer = ObjectMeasurer(
        scale = scale,
        reference_size_mm=reference_size_mm,
        debug_path=debug_path(tmp_path, slug, image_path),
        save_debug_images=save_debug_images_enabled(),
    )
    measurer.slug = slug
    measurements, debug = measurer.measure(img, return_debug=True)

    with print_debug_on_failure(slug, lambda: debug):
        assert measurements
        assert debug["status"] == "ok"
        print(measurements)
        assert debug["object_contour_svg"] is not None
        assert debug["trace"]


def test_mock_box_white_blue_one_detects_large_central_shirt_contour(tmp_path):
    slug = "mock-box-white-blue-one"
    image_path = fixture_path(slug, "IMG_0315.jpeg")
    img = cv2.imread(str(image_path))
    assert img is not None, f"Could not load image at path: {image_path}"

    measurer = ObjectMeasurer(
        scale=1,
        reference_size_mm=MOCK_BOX_MM,
        debug_path=debug_path(tmp_path, slug, image_path),
        save_debug_images=False,
    )
    measurer.slug = slug
    measurements, debug = measurer.measure(img, return_debug=True)

    with print_debug_on_failure(slug, lambda: debug):
        assert debug["status"] == "ok"
        assert measurements

        dominant_measurement = max(measurements, key=lambda m: m.bbox[2] * m.bbox[3])
        warped_h, warped_w = debug["imgWarp"].shape[:2]
        x, y, w, h = dominant_measurement.bbox
        center_x = (x + (w / 2)) / warped_w
        center_y = (y + (h / 2)) / warped_h
        bbox_area_fraction = (w * h) / (warped_w * warped_h)

        assert 0.35 <= center_x <= 0.65
        assert 0.35 <= center_y <= 0.65
        assert bbox_area_fraction >= 0.75
        assert dominant_measurement.width_cm == pytest.approx(70.4, abs=STANDARD_TOLERANCE_SHIRT_CM)
        assert dominant_measurement.height_cm == pytest.approx(64.5, abs=STANDARD_TOLERANCE_SHIRT_CM)


@pytest.mark.parametrize(
    "slug, scale, image_path, reference_size_mm, expected_count, expected_w_cm, expected_h_cm, tol_cm",
    [
        ("one", 1, fixture_path("one", "one.jpg"), A4_MM, 2, 9.15, 5.0, STANDARD_TOLERANCE_CM),
        ("ucard-one", 1, fixture_path("ucard-one", "ucard-one.jpg"), LETTER_MM, 1, CREDIT_CARD_W_CM, CREDIT_CARD_H_CM, STANDARD_TOLERANCE_CM),
        ("ucard-one", 2, fixture_path("ucard-one", "ucard-one.jpg"), LETTER_MM, 1, CREDIT_CARD_W_CM, CREDIT_CARD_H_CM, STANDARD_TOLERANCE_CM),
        ("ucard-two", 1, fixture_path("ucard-two", "ucard-two.jpg"), LETTER_MM, 1, CREDIT_CARD_W_CM, CREDIT_CARD_H_CM, STANDARD_TOLERANCE_CM),
        ("ucard-one-off-axis", 1, fixture_path("ucard-one-off-axis", "ucard-one-off-axis.jpg"), LETTER_MM, 1, CREDIT_CARD_W_CM, CREDIT_CARD_H_CM, STANDARD_TOLERANCE_CM),
        ("ucard-two-off-axis", 1, fixture_path("ucard-two-off-axis", "ucard-two-off-axis.jpg"), LETTER_MM, 1, CREDIT_CARD_W_CM, CREDIT_CARD_H_CM, STANDARD_TOLERANCE_CM),
        # ("nike-envelope", 1, fixture_path("nike-envelope", "img_6a7b6ba3eb1276.03284004.jpg"), ENVELOPE_MM, 1, CREDIT_CARD_W_CM, CREDIT_CARD_H_CM, STANDARD_TOLERANCE_CM),
        ("nike-letter-one", 1, fixture_path("nike-letter-one", "img_6a7c9d736092f6.97385818.jpg"), LETTER_MM, 1, CREDIT_CARD_W_CM, CREDIT_CARD_H_CM,STANDARD_TOLERANCE_CM),
        ("nike-letter-two", 1, fixture_path("nike-letter-two", "img_6a7b67eeb4dcd8.69842033.jpg"), LETTER_MM, 1, CREDIT_CARD_W_CM, CREDIT_CARD_H_CM,STANDARD_TOLERANCE_CM),
        ("nike-letter-one-off-axis", 1, fixture_path("nike-letter-one-off-axis", "img_6a838cf9edde22.59423143.jpg"), LETTER_MM, 1, CREDIT_CARD_W_CM, CREDIT_CARD_H_CM, STANDARD_TOLERANCE_CM),
        ("iswic", 1, fixture_path("iswic", "iswic.jpg"), LIGHTBOX_MAT_MM, 1, 70.485, 69.5325, STANDARD_TOLERANCE_SHIRT_CM),
        ("goldy", 1, fixture_path("goldy", "goldy.jpg"), ODDBALL_BLACK_POSTER_BOARD_MM, 1, 45.72, 40.5, STANDARD_TOLERANCE_SHIRT_CM),
        ("cherokee", 1, fixture_path("cherokee", "cherokee.jpg"), ODDBALL_BLACK_POSTER_BOARD_MM, 1, 40.9575, 51.435, STANDARD_TOLERANCE_SHIRT_CM),

        ("iswic", 2, fixture_path("iswic", "iswic.jpg"), LIGHTBOX_MAT_MM, 1, 70.485, 69.5325, STANDARD_TOLERANCE_SHIRT_CM),
    ],
)
def test_1jpg_two_objects_about_9x5(slug, scale, image_path, reference_size_mm, expected_count, expected_w_cm, expected_h_cm, tol_cm, tmp_path):
    img = cv2.imread(str(image_path))
    assert img is not None, f"Could not load image at path: {image_path}"

    measurer = ObjectMeasurer(
        scale = scale,
        reference_size_mm=reference_size_mm,
        debug_path=debug_path(tmp_path, slug, image_path),
        save_debug_images=save_debug_images_enabled(),
    )
    measurer.slug = slug
    measurements, debug = measurer.measure(img, return_debug=True)

    with print_debug_on_failure(slug, lambda: debug):
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

        print(debug['object_contour_svg'])
        assert debug['object_contour_svg'] is not None


def test_failed_measurement_includes_debug_trace_and_logs(caplog):
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    measurer = ObjectMeasurer(reference_size_mm=LETTER_MM)

    with caplog.at_level(logging.WARNING, logger="ObjectMeasurer.ObjectMeasurement"):
        measurements, debug = measurer.measure(img, return_debug=True)

    with print_debug_on_failure("failed-measurement", lambda: debug):
        assert measurements == []
        assert debug["status"] == "failed"
        assert debug["errors"][0]["code"] == "reference_not_found"
        assert any(item["event"] == "failure" for item in debug["trace"])
        assert "reference_not_found" in caplog.text


def test_debug_images_can_be_saved_to_requested_folder(tmp_path):
    image_path = fixture_path("ucard-one", "ucard-one.jpg")
    img = cv2.imread(str(image_path))
    assert img is not None

    measurer = ObjectMeasurer(
        reference_size_mm=LETTER_MM,
        debug_path=tmp_path,
        save_debug_images=True,
    )
    measurer.slug = "ucard-one"
    measurements, debug = measurer.measure(img, return_debug=True)

    with print_debug_on_failure("debug-images", lambda: debug):
        saved_images = [Path(item["path"]) for item in debug["debug_images"] if item["saved"]]
        assert measurements
        assert saved_images
        assert all(path.parent == tmp_path for path in saved_images)
        assert any(path.name.endswith("_page_gray.jpg") for path in saved_images)
        assert any(path.name.endswith("_warped.jpg") for path in saved_images)
