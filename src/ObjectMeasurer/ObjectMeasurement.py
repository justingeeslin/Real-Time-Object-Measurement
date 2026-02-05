from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple, Union

import cv2
import numpy as np


@dataclass(frozen=True)
class ContourMeasurement:
    """A single measured object within the warped reference plane."""

    contour: np.ndarray  # polygon points (approx) for the object
    bbox: Tuple[int, int, int, int]  # (x, y, w, h) in pixels of the warped image
    width_cm: float
    height_cm: float


class ObjectMeasurer:
    """Measure object contours in an image using a planar reference (e.g., an A4 sheet).

    This class includes the contour + warp utilities (ported from your utlis.py) as static methods.

    Flow:
      1) Find the largest 4-corner contour (the reference page).
      2) Warp the image into a known-size plane.
      3) Find 4-corner object contours inside the warped plane.
      4) Compute width/height using findDis on reordered corner points.

    Measurements are returned in centimeters, matching the legacy script behavior.
    """

    def __init__(
        self,
        *,
        scale: int = 3,
        reference_size_mm: Tuple[float, float] = (210.0, 297.0),
        page_min_area: int = 50_000,
        object_min_area: int = 2_000,
        page_filter_corners: int = 4,
        object_filter_corners: int = 4,
        object_canny_thresholds: Tuple[int, int] = (50, 50),
        pixels_to_mm_divisor: float = 10.0,
        warp_pad: int = 20,
    ) -> None:
        self.scale = int(scale)
        self.ref_w_mm = float(reference_size_mm[0])
        self.ref_h_mm = float(reference_size_mm[1])

        self.page_min_area = int(page_min_area)
        self.object_min_area = int(object_min_area)

        self.page_filter_corners = int(page_filter_corners)
        self.object_filter_corners = int(object_filter_corners)

        self.object_canny_thresholds = (int(object_canny_thresholds[0]), int(object_canny_thresholds[1]))
        self.pixels_to_mm_divisor = float(pixels_to_mm_divisor)

        # Warped plane size in pixels (same idea as the original script)
        self.wP = int(self.ref_w_mm * self.scale)
        self.hP = int(self.ref_h_mm * self.scale)

        self.warp_pad = int(warp_pad)
        self.debugImageCounter = 0
        self.debug = {}

    def _saveDebugImage(self, image: np.ndarray, name) -> None:
        try:
            cv2.imwrite(f"{self.debug_path}/{self.debugImageCounter}_{self.slug}_{name}.jpg", image)
        except Exception:
            pass
        self.debugImageCounter += 1

    # ---- utilities (ported from utlis.py) ----

    @staticmethod
    def reorder(myPoints: np.ndarray) -> np.ndarray:
        myPointsNew = np.zeros_like(myPoints)
        myPoints = myPoints.reshape((4, 2))
        add = myPoints.sum(1)
        myPointsNew[0] = myPoints[np.argmin(add)]
        myPointsNew[3] = myPoints[np.argmax(add)]
        diff = np.diff(myPoints, axis=1)
        myPointsNew[1] = myPoints[np.argmin(diff)]
        myPointsNew[2] = myPoints[np.argmax(diff)]
        return myPointsNew

    @staticmethod
    def warpImg(img: np.ndarray, points: np.ndarray, w: int, h: int, pad: int = 20) -> np.ndarray:
        points = ObjectMeasurer.reorder(points)
        pts1 = np.float32(points)
        pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarp = cv2.warpPerspective(img, matrix, (w, h))
        imgWarp = imgWarp[pad : imgWarp.shape[0] - pad, pad : imgWarp.shape[1] - pad]
        return imgWarp

    @staticmethod
    def findDis(pts1: np.ndarray, pts2: np.ndarray) -> float:
        return float(((pts2[0] - pts1[0]) ** 2 + (pts2[1] - pts1[1]) ** 2) ** 0.5)

    # ---- measurement API ----

    def measure(
        self,
        img: np.ndarray,
        *,
        return_debug: bool = False,
    ) -> Union[List[ContourMeasurement], Tuple[List[ContourMeasurement], Dict[str, Any]]]:
        """Measure objects in the provided BGR image."""

        self.debug: Dict[str, Any] = {}

        # Get the page contour
        originalSlug = self.slug
        self.slug = self.slug + "_page"
        imgContours, conts = self._getContours(img, minArea=self.page_min_area, filter=self.page_filter_corners)
        self.debug["imgContours_page"] = imgContours
        self.debug["page_contours"] = conts

        if len(conts) == 0:
            measurements: List[ContourMeasurement] = []
            return (measurements, self.debug) if return_debug else measurements

        biggest = conts[0][2]

        # --- debug: draw minAreaRect in blue ---
        rect = cv2.minAreaRect(biggest)
        box = cv2.boxPoints(rect)  # 4x2 float array
        box = box.astype(np.int32)

        img_with_rect = img.copy()
        cv2.drawContours(img_with_rect, [box], 0, (0, 255, 0), 3)
        self.debug["img_minAreaRect"] = img_with_rect
        self._saveDebugImage(img_with_rect, "minAreaRect")

        img_with_paper_contour = img.copy()
        cv2.drawContours(img_with_paper_contour, [biggest], 0, (0, 0, 255), 3)
        self.debug["img_with_paper_contour"] = img_with_paper_contour
        self._saveDebugImage(img_with_paper_contour, "paper_contour")

        (_, _), (paper_w, paper_h), paper_angle = cv2.minAreaRect(biggest)

        # iron out OpenCV's tricky width and height intepretation
        if paper_angle < 45:
            pass
        elif paper_angle < 90+45:
            print("Swapping width and height")
            temp = paper_w
            paper_w = paper_h
            paper_h = temp
        else:
            print("No swap needed its just upside down which is fine for a symetrial object")

        is_landscape = paper_w > paper_h
        print(f"{self.slug} angle: {paper_angle}")

        if is_landscape:
            print(f"{self.slug} is landscape, Rotating to portrait..")

            def rotate_contour_90_cw(cnt, img_shape):
                h, w = img_shape[:2]
                cnt_rot = cnt.copy()
                cnt_rot[:, 0, 0] = h - 1 - cnt[:, 0, 1]  # new x
                cnt_rot[:, 0, 1] = cnt[:, 0, 0]  # new y
                return cnt_rot

            biggest = rotate_contour_90_cw(biggest, img.shape)

            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        else:
            print(f"{self.slug} is not landscape")

        self.slug = originalSlug

        imgWarp = ObjectMeasurer.warpImg(img, biggest, self.wP, self.hP, pad=self.warp_pad)
        self.debug["imgWarp"] = imgWarp
        self._saveDebugImage(imgWarp, "warped")

        imgContours2, conts2 = self._getContours(
            imgWarp,
            minArea=100,
            filter=0,
            cThr=[self.object_canny_thresholds[0], self.object_canny_thresholds[1]],
            draw=False,
        )
        self.debug["imgContours_objects"] = imgContours2
        self.debug["object_contours"] = conts2

        # --- debug: draw all detected object contours on the warped image ---
        img_with_object_contours = imgWarp.copy()
        for idx, obj in enumerate(conts2):
            cnt = obj[4]  # raw contour

            cv2.drawContours(img_with_object_contours, [cnt], -1, (0, 0, 255), 2)
            x, y, bw, bh = obj[3]
            cv2.rectangle(img_with_object_contours, (x, y), (x + bw, y + bh), (255, 0, 0), 2)
            cv2.putText(
                img_with_object_contours,
                str(idx),
                (x + 5, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            # Print the axis-aligned bbox dimensions (blue rectangle) in cm
            bbox_w_cm = float(bw) / self.pixels_to_mm_divisor
            bbox_h_cm = float(bh) / self.pixels_to_mm_divisor
            print(f"[draw bbox] obj#{idx}: w={bbox_w_cm:.4f}cm h={bbox_h_cm:.4f}cm (axis-aligned)")
        self.debug["img_object_contours_drawn"] = img_with_object_contours
        self._saveDebugImage(img_with_object_contours, "object_contours_drawn")

        # --- debug: draw minAreaRect boxes for each object (useful for polygons) ---
        img_with_object_boxes = imgWarp.copy()
        for idx, obj in enumerate(conts2):
            cnt = obj[4]
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = box.astype(np.int32)
            cv2.drawContours(img_with_object_boxes, [box], 0, (0, 255, 0), 2)

            # if return_debug:
            (_, _), (w_box, h_box), _ = rect
            width_px = max(float(w_box), float(h_box))
            height_px = min(float(w_box), float(h_box))
            width_cm = width_px / self.pixels_to_mm_divisor
            height_cm = height_px / self.pixels_to_mm_divisor
            print(f"[draw minAreaRect] obj#{idx}: w={width_cm:.4f}cm h={height_cm:.4f}cm (rotated)")
        self.debug["img_object_minAreaRect"] = img_with_object_boxes
        self._saveDebugImage(img_with_object_boxes, "object_minAreaRect")

        measurements = self._measure_objects(conts2)

        return (measurements, self.debug) if return_debug else measurements

    def measure_from_path(
        self,
        path: str,
        *,
        return_debug: bool = False,
    ) -> Union[List[ContourMeasurement], Tuple[List[ContourMeasurement], Dict[str, Any]]]:
        """Read an image from disk and measure it."""

        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        return self.measure(img, return_debug=return_debug)

    def _measure_objects(self, conts2: Sequence[Any]) -> List[ContourMeasurement]:
        out: List[ContourMeasurement] = []

        for obj in conts2:
            pts = obj[2]  # approx polygon

            # If it's a clean 4-corner polygon, keep the legacy corner-to-corner measurement.
            # Otherwise (polygons, noisy contours), use the rotated minimum-area bounding box.
            if len(pts) == 4:
                print("Using rectangle distance method...")
                nPoints = ObjectMeasurer.reorder(pts)

                # Legacy behavior: divide points by scale before findDis, then divide by 10 and label as cm.
                w_px = ObjectMeasurer.findDis(nPoints[0][0] // self.scale, nPoints[1][0] // self.scale)
                h_px = ObjectMeasurer.findDis(nPoints[0][0] // self.scale, nPoints[2][0] // self.scale)
            else:
                print("Using polygon distance method...")
                # Use raw contour for a tighter fit than the simplified approx polygon.
                cnt = obj[4]
                (_, _), (w_box, h_box), _ = cv2.minAreaRect(cnt)
                # minAreaRect returns widths/heights in pixels of the warped image.
                w_px, h_px = float(w_box // self.scale), float(h_box // self.scale)

            # Normalize so width is the longer side and height is the shorter side.
            width_px = max(w_px, h_px)
            height_px = min(w_px, h_px)

            width_cm = width_px / self.pixels_to_mm_divisor
            height_cm = height_px / self.pixels_to_mm_divisor

            if getattr(self, "debug", False):
                method = "corners" if len(pts) == 4 else "minAreaRect"
                print(f"[measure] {method}: w={width_cm:.2f}cm h={height_cm:.2f}cm")

            x, y, bw, bh = obj[3]
            out.append(
                ContourMeasurement(
                    contour=pts,
                    bbox=(int(x), int(y), int(bw), int(bh)),
                    width_cm=float(width_cm),
                    height_cm=float(height_cm),
                )
            )

        return out


    def _getContours(
        self,
        img: np.ndarray,
        cThr: List[int] = [100, 100],
        minArea: int = 1000,
        filter: int = 0,
        draw: bool = False,
    ) -> Tuple[np.ndarray, List[Any]]:
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self._saveDebugImage(imgGray, "gray")
        imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
        self._saveDebugImage(imgBlur, "blur")
        imgCanny = cv2.Canny(imgBlur, cThr[0], cThr[1])
        self._saveDebugImage(imgCanny, "edgeDetect")
        kernel = np.ones((5, 5))
        imgDial = cv2.dilate(imgCanny, kernel, iterations=3)
        self._saveDebugImage(imgDial, "dilate")
        imgThre = cv2.erode(imgDial, kernel, iterations=2)
        self._saveDebugImage(imgThre, "_erode")

        contours, hiearchy = cv2.findContours(imgThre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        finalCountours: List[Any] = []
        for i in contours:
            area = cv2.contourArea(i)
            if area > minArea:
                peri = cv2.arcLength(i, True)
                approx = cv2.approxPolyDP(i, 0.02 * peri, True)
                bbox = cv2.boundingRect(approx)
                if filter > 0:
                    if len(approx) == filter:
                        finalCountours.append([len(approx), area, approx, bbox, i])
                else:
                    finalCountours.append([len(approx), area, approx, bbox, i])

        finalCountours = sorted(finalCountours, key=lambda x: x[1], reverse=True)
        if draw:
            for con in finalCountours:
                cv2.drawContours(img, con[4], -1, (0, 0, 255), 3)
        return img, finalCountours


def _draw_measurements(imgWarp: np.ndarray, measurements: Sequence[ContourMeasurement]) -> np.ndarray:
    """Draw polylines/arrows/labels on a warped image (demo helper)."""

    imgOut = imgWarp.copy()

    for m in measurements:
        cv2.polylines(imgOut, [m.contour], True, (0, 255, 0), 2)

        nPoints = ObjectMeasurer.reorder(m.contour)

        cv2.arrowedLine(
            imgOut,
            (nPoints[0][0][0], nPoints[0][0][1]),
            (nPoints[1][0][0], nPoints[1][0][1]),
            (255, 0, 255),
            3,
            8,
            0,
            0.05,
        )
        cv2.arrowedLine(
            imgOut,
            (nPoints[0][0][0], nPoints[0][0][1]),
            (nPoints[2][0][0], nPoints[2][0][1]),
            (255, 0, 255),
            3,
            8,
            0,
            0.05,
        )

        x, y, w, h = m.bbox
        cv2.putText(
            imgOut,
            f"{m.width_cm}cm",
            (x + 30, y - 10),
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
            1.5,
            (255, 0, 255),
            2,
        )
        cv2.putText(
            imgOut,
            f"{m.height_cm}cm",
            (x - 70, y + h // 2),
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
            1.5,
            (255, 0, 255),
            2,
        )

    return imgOut

PORTRAIT_POSTER_BOARD_MM = (561.975, 711.2)
def main() -> None:
    """Interactive demo (webcam or single image path), similar to the original script."""

    ###################################
    cap = cv2.VideoCapture(0)
    cap.set(10, 160)
    cap.set(3, 1920)
    cap.set(4, 1080)

    measurer = ObjectMeasurer(scale=1, reference_size_mm=PORTRAIT_POSTER_BOARD_MM)
    measurer.slug = "main"
    ###################################

    while True:
        success, img = cap.read()
        if not success or img is None:
            continue

        measurements, debug = measurer.measure(img, return_debug=True)

        # Show the warped plane with the contour overlays (best view for measurement correctness)
        imgWarp = debug.get("imgWarp")
        if imgWarp is not None:
            imgAnnotated = _draw_measurements(imgWarp, measurements)
            cv2.imshow("Warped (Contours)", imgAnnotated)
        else:
            print("No warped image")

        # Also show the debug frame that draws *all* detected object contours + axis-aligned bboxes
        # obj_debug = debug.get("img_object_contours_drawn")
        # if obj_debug is not None:
        #     cv2.imshow("Warped (Detected Objects)", obj_debug)
        # else:
        #     print("No object detected")

        # Original feed (smaller) so you can see what the camera sees
        imgSmall = cv2.resize(img, (0, 0), None, 0.5, 0.5)
        cv2.imshow("Original", imgSmall)

        obj_debug = debug.get("imgWarp")
        if obj_debug is not None:
            cv2.imshow("Warped (Detected Objects)", obj_debug)
        else:
            print("No object detected")


        # Allow quitting with ESC or 'q'
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()