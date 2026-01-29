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

    # ---- utilities (ported from utlis.py) ----

    @staticmethod
    def getContours(
        img: np.ndarray,
        cThr: List[int] = [100, 100],
        showCanny: bool = False,
        minArea: int = 1000,
        filter: int = 0,
        draw: bool = False,
    ) -> Tuple[np.ndarray, List[Any]]:
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
        imgCanny = cv2.Canny(imgBlur, cThr[0], cThr[1])
        kernel = np.ones((5, 5))
        imgDial = cv2.dilate(imgCanny, kernel, iterations=3)
        imgThre = cv2.erode(imgDial, kernel, iterations=2)
        if showCanny:
            cv2.imshow("Canny", imgThre)

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

        debug: Dict[str, Any] = {}

        imgContours, conts = ObjectMeasurer.getContours(img, minArea=self.page_min_area, filter=self.page_filter_corners)
        debug["imgContours_page"] = imgContours
        debug["page_contours"] = conts

        if len(conts) == 0:
            measurements: List[ContourMeasurement] = []
            return (measurements, debug) if return_debug else measurements

        biggest = conts[0][2]

        imgWarp = ObjectMeasurer.warpImg(img, biggest, self.wP, self.hP, pad=self.warp_pad)
        debug["imgWarp"] = imgWarp

        imgContours2, conts2 = ObjectMeasurer.getContours(
            imgWarp,
            minArea=self.object_min_area,
            filter=self.object_filter_corners,
            cThr=[self.object_canny_thresholds[0], self.object_canny_thresholds[1]],
            draw=False,
        )
        debug["imgContours_objects"] = imgContours2
        debug["object_contours"] = conts2

        measurements = self._measure_objects(conts2)

        return (measurements, debug) if return_debug else measurements

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
            nPoints = ObjectMeasurer.reorder(pts)

            # Keep legacy behavior: divide points by scale before findDis, then divide by 10 and label as cm.
            w = ObjectMeasurer.findDis(nPoints[0][0] // self.scale, nPoints[1][0] // self.scale)
            h = ObjectMeasurer.findDis(nPoints[0][0] // self.scale, nPoints[2][0] // self.scale)

            width_cm = round((w / self.pixels_to_mm_divisor), 1)
            height_cm = round((h / self.pixels_to_mm_divisor), 1)

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


def main() -> None:
    """Interactive demo (webcam or single image path), similar to the original script."""

    ###################################
    webcam = True
    path = "1.jpg"

    cap = cv2.VideoCapture(0)
    cap.set(10, 160)
    cap.set(3, 1920)
    cap.set(4, 1080)

    measurer = ObjectMeasurer(scale=3)
    ###################################

    while True:
        if webcam:
            success, img = cap.read()
            if not success or img is None:
                continue
        else:
            img = cv2.imread(path)
            if img is None:
                raise FileNotFoundError(f"Could not read image: {path}")

        measurements, debug = measurer.measure(img, return_debug=True)

        imgWarp = debug.get("imgWarp")
        if imgWarp is not None:
            imgAnnotated = _draw_measurements(imgWarp, measurements)
            cv2.imshow("A4", imgAnnotated)

        imgSmall = cv2.resize(img, (0, 0), None, 0.5, 0.5)
        cv2.imshow("Original", imgSmall)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()