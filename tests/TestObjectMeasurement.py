import cv2
import pytest

from ObjectMeasurement import ObjectMeasurer

A4_MM = (210.0, 297.0)
LETTER_MM = (215.9, 279.4)  # 8.5in x 11in

@pytest.mark.parametrize(
    "image_path, reference_size_mm, expected_count, expected_w_cm, expected_h_cm, tol_cm",
    [
        ("../test-images/one/one.jpg", A4_MM, 2, 9.0, 5.0, 1.0),  # adjust tol if your lighting/edges vary
    ],
)
def test_1jpg_two_objects_about_9x5(image_path, reference_size_mm, expected_count, expected_w_cm, expected_h_cm, tol_cm):
    img = cv2.imread(image_path)
    assert img is not None, f"Could not load image at path: {image_path}"

    measurer = ObjectMeasurer(scale=3, reference_size_mm=reference_size_mm)

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