from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from OpenCVContourSVGConverter import OpenCVContourSVGConverter

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ContourMeasurement:
    """A single measured object within the warped reference plane."""

    contour: np.ndarray  # polygon points (approx) for the object
    bbox: Tuple[int, int, int, int]  # (x, y, w, h) in pixels of the warped image
    width_cm: float
    height_cm: float


@dataclass(frozen=True)
class ReferenceContourCandidate:
    """A candidate planar reference surface found in the source image."""

    points: np.ndarray
    raw_contour: np.ndarray
    area: float
    bbox: Tuple[int, int, int, int]
    method: str
    score: float
    metadata: Dict[str, Any]


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
        scale: int = 1,
        reference_size_mm: Tuple[float, float] = (210.0, 297.0),
        page_min_area: int = 50_000,
        object_min_area: int = 2_000,
        page_filter_corners: int = 4,
        object_filter_corners: int = 4,
        object_canny_thresholds: Tuple[int, int] = (50, 50),
        object_min_dimension_cm: float = 1.0,
        pixels_to_mm_divisor: float = 10.0,
        warp_pad: int = 20,
        page_kernel_size: int = 5,
        debug_path: Optional[Union[str, Path]] = None,
        save_debug_images: Optional[bool] = None,
    ) -> None:
        self.scale = int(scale)
        self.ref_w_mm = float(reference_size_mm[0])
        self.ref_h_mm = float(reference_size_mm[1])

        self.page_min_area = int(page_min_area)
        self.object_min_area = int(object_min_area)

        self.page_filter_corners = int(page_filter_corners)
        self.object_filter_corners = int(object_filter_corners)

        self.object_canny_thresholds = (int(object_canny_thresholds[0]), int(object_canny_thresholds[1]))
        self.object_min_dimension_cm = float(object_min_dimension_cm)
        self.pixels_to_mm_divisor = float(pixels_to_mm_divisor)
        self.page_kernel_size = self._odd_kernel_size(page_kernel_size)

        # Warped plane size in pixels (same idea as the original script)
        self.wP = int(self.ref_w_mm * self.scale)
        self.hP = int(self.ref_h_mm * self.scale)

        self.warp_pad = int(warp_pad)
        self.debugImageCounter = 0
        self.debug = {}
        self.debug_path = None if debug_path is None else str(debug_path)
        self.save_debug_images = save_debug_images

        self.slug = "untitled"

    def _saveDebugImage(self, image: np.ndarray, name) -> None:
        path = getattr(self, "debug_path", None)
        save_debug_images = getattr(self, "save_debug_images", None)
        should_save = bool(path) if save_debug_images is None else bool(save_debug_images)
        if not should_save:
            return
        if not path:
            self._trace("debug_image_save_skipped", name=str(name), reason="debug_path_not_set")
            return

        out_path = Path(path) / f"{self.debugImageCounter}_{self.slug}_{name}.jpg"
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            saved = cv2.imwrite(str(out_path), image)
            self.debug.setdefault("debug_images", []).append(
                {"name": str(name), "path": str(out_path), "saved": bool(saved)}
            )
            if not saved:
                self._trace("debug_image_save_failed", name=str(name), path=str(out_path))
        except Exception as exc:
            self._trace("debug_image_save_error", name=str(name), path=str(out_path), error=str(exc))
        self.debugImageCounter += 1

    def _trace(self, event: str, **fields: Any) -> None:
        if not hasattr(self, "debug"):
            return
        record = {"event": event}
        record.update({key: self._debug_value(value) for key, value in fields.items()})
        self.debug.setdefault("trace", []).append(record)
        logger.debug("ObjectMeasurer trace: %s", record)

    def _record_failure(self, code: str, message: str, **fields: Any) -> None:
        error = {"code": code, "message": message}
        error.update({key: self._debug_value(value) for key, value in fields.items()})
        self.debug["status"] = "failed"
        self.debug.setdefault("errors", []).append(error)
        self._trace("failure", code=code, message=message, **fields)
        logger.warning("ObjectMeasurer failed: %s (%s)", message, code)

    @staticmethod
    def _debug_value(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return {"shape": tuple(int(v) for v in value.shape), "dtype": str(value.dtype)}
        if isinstance(value, tuple):
            return tuple(ObjectMeasurer._debug_value(v) for v in value)
        if isinstance(value, list):
            return [ObjectMeasurer._debug_value(v) for v in value]
        if isinstance(value, dict):
            return {str(k): ObjectMeasurer._debug_value(v) for k, v in value.items()}
        return value

    @staticmethod
    def _odd_kernel_size(value: int) -> int:
        kernel_size = max(3, int(round(value)))
        if kernel_size % 2 == 0:
            kernel_size += 1
        return kernel_size

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
        self.debugImageCounter = 0
        self.debug.update(
            {
                "status": "started",
                "errors": [],
                "object_contour_svg": None,
                "measurements": [],
            }
        )
        self._trace(
            "measure_start",
            image_shape=None if img is None else tuple(int(v) for v in img.shape),
            scale=self.scale,
            reference_size_mm=(self.ref_w_mm, self.ref_h_mm),
        )

        if img is None or not isinstance(img, np.ndarray) or img.size == 0:
            self._record_failure("invalid_image", "Input image is empty or unreadable")
            measurements: List[ContourMeasurement] = []
            return (measurements, self.debug) if return_debug else measurements

        # Get the page contour
        originalSlug = self.slug
        self.slug = self.slug + "_page"
        page_candidate = self._find_reference_contour(img)

        if page_candidate is None:
            self.slug = originalSlug
            self._record_failure(
                "reference_not_found",
                "No reference surface contour could be detected",
                page_min_area=self.page_min_area,
                page_filter_corners=self.page_filter_corners,
            )
            measurements: List[ContourMeasurement] = []
            return (measurements, self.debug) if return_debug else measurements

        biggest = page_candidate.points
        self.debug["page_detection"] = {
            "method": page_candidate.method,
            "area": page_candidate.area,
            "bbox": page_candidate.bbox,
            "score": page_candidate.score,
            "metadata": page_candidate.metadata,
        }
        self._trace(
            "reference_selected",
            method=page_candidate.method,
            area=page_candidate.area,
            bbox=page_candidate.bbox,
            score=page_candidate.score,
        )

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
            self._trace("reference_dimensions_swapped", angle=paper_angle)
            temp = paper_w
            paper_w = paper_h
            paper_h = temp
        else:
            self._trace("reference_upside_down", angle=paper_angle)

        is_landscape = paper_w > paper_h
        self._trace(
            "reference_orientation",
            angle=paper_angle,
            paper_w=paper_w,
            paper_h=paper_h,
            is_landscape=is_landscape,
        )

        if is_landscape:
            self._trace("reference_rotate_to_portrait")

            def rotate_contour_90_cw(cnt, img_shape):
                h, w = img_shape[:2]
                cnt_rot = cnt.copy()
                cnt_rot[:, 0, 0] = h - 1 - cnt[:, 0, 1]  # new x
                cnt_rot[:, 0, 1] = cnt[:, 0, 0]  # new y
                return cnt_rot

            biggest = rotate_contour_90_cw(biggest, img.shape)

            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        else:
            self._trace("reference_already_portrait")

        self.slug = originalSlug

        imgWarp = ObjectMeasurer.warpImg(img, biggest, self.wP, self.hP, pad=self.warp_pad)
        self.debug["imgWarp"] = imgWarp
        self._saveDebugImage(imgWarp, "warped")

        imgContours2, conts2 = self._getContours(
            imgWarp,
            minArea=self.object_min_area,
            filter=0,
            cThr=[self.object_canny_thresholds[0], self.object_canny_thresholds[1]],
            draw=False,
            stage="objects",
            kernel_size=self._odd_kernel_size(5 * self.scale),
        )
        self.debug["imgContours_objects"] = imgContours2
        self.debug["object_contours"] = conts2
        # --- debug: draw all detected object contours on the warped image ---
        img_with_object_contours = imgWarp.copy()

        converter = OpenCVContourSVGConverter()
        for idx, obj in enumerate(conts2):
            cnt = obj[4]  # raw contour

            try:
                if self.debug["object_contour_svg"] is None:
                    self.debug["object_contour_svg"] = converter.convert([cnt])
            except Exception as exc:
                self._trace("object_svg_conversion_failed", index=idx, error=str(exc))

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
            self._trace(
                "object_bbox",
                index=idx,
                width_cm=bbox_w_cm,
                height_cm=bbox_h_cm,
                bbox=(int(x), int(y), int(bw), int(bh)),
            )
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
            self._trace("object_min_area_rect", index=idx, width_cm=width_cm, height_cm=height_cm)
        self.debug["img_object_minAreaRect"] = img_with_object_boxes
        self._saveDebugImage(img_with_object_boxes, "object_minAreaRect")

        measurements = self._measure_objects(conts2)
        self.debug["measurements"] = [
            {"width_cm": m.width_cm, "height_cm": m.height_cm, "bbox": m.bbox} for m in measurements
        ]
        if measurements:
            self.debug["status"] = "ok"
            self._trace("measure_complete", object_count=len(measurements))
        else:
            self._record_failure(
                "object_not_found",
                "Reference surface was found, but no object contour passed filtering",
                object_min_area=self.object_min_area,
            )

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
                method = "corners"
                nPoints = ObjectMeasurer.reorder(pts)

                # Legacy behavior: divide points by scale before findDis, then divide by 10 and label as cm.
                w_px = ObjectMeasurer.findDis(nPoints[0][0] // self.scale, nPoints[1][0] // self.scale)
                h_px = ObjectMeasurer.findDis(nPoints[0][0] // self.scale, nPoints[2][0] // self.scale)
            else:
                method = "minAreaRect"
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

            if min(width_cm, height_cm) < self.object_min_dimension_cm:
                self._trace(
                    "object_measurement_skipped",
                    reason="below_min_dimension",
                    width_cm=width_cm,
                    height_cm=height_cm,
                    min_dimension_cm=self.object_min_dimension_cm,
                )
                continue

            self._trace("object_measured", method=method, width_cm=width_cm, height_cm=height_cm)

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
        cThr: Sequence[int] = (100, 100),
        minArea: int = 1000,
        filter: int = 0,
        draw: bool = False,
        stage: str = "contours",
        kernel_size: Optional[int] = None,
        retrieval_mode: int = cv2.RETR_EXTERNAL,
        approx_epsilon: float = 0.02,
    ) -> Tuple[np.ndarray, List[Any]]:
        # Blur and Edge Detect based on scale
        kernel_size = self._odd_kernel_size(kernel_size if kernel_size is not None else 5 * self.scale)
        self._trace(
            "contour_preprocess_start",
            stage=stage,
            kernel_size=kernel_size,
            canny_thresholds=(int(cThr[0]), int(cThr[1])),
            min_area=minArea,
            filter=filter,
            retrieval_mode=int(retrieval_mode),
            approx_epsilon=approx_epsilon,
        )

        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self._saveDebugImage(imgGray, "gray")
        imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
        self._saveDebugImage(imgBlur, "blur")
        imgCanny = cv2.Canny(imgBlur, cThr[0], cThr[1])
        self._saveDebugImage(imgCanny, "edgeDetect")
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        imgDial = cv2.dilate(imgCanny, kernel, iterations=3)
        self._saveDebugImage(imgDial, "dilate")
        imgThre = cv2.erode(imgDial, kernel, iterations=2)
        self._saveDebugImage(imgThre, "_erode")

        contours, hiearchy = cv2.findContours(imgThre, retrieval_mode, cv2.CHAIN_APPROX_SIMPLE)
        finalCountours: List[Any] = []
        for i in contours:
            area = cv2.contourArea(i)
            if area > minArea:
                peri = cv2.arcLength(i, True)
                approx = cv2.approxPolyDP(i, approx_epsilon * peri, True)
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
        self._trace(
            "contour_preprocess_complete",
            stage=stage,
            contour_count=len(contours),
            accepted_count=len(finalCountours),
            top_candidates=self._contour_summaries(finalCountours[:5]),
        )
        return img, finalCountours

    def _find_reference_contour(self, img: np.ndarray) -> Optional[ReferenceContourCandidate]:
        imgContours, conts = self._getContours(
            img,
            minArea=self.page_min_area,
            filter=self.page_filter_corners,
            stage="page_primary",
            kernel_size=self.page_kernel_size,
        )
        self.debug["imgContours_page"] = imgContours
        self.debug["page_contours"] = conts

        if conts:
            contour = conts[0]
            return self._make_reference_candidate(
                points=contour[2],
                raw_contour=contour[4],
                method="canny_external_4_point",
                metadata={"corner_count": contour[0]},
                img_shape=img.shape,
            )

        self._trace("reference_primary_empty")

        saturated = self._reference_candidates_from_color(img, color_family="saturated")
        if saturated:
            best = saturated[0]
            self._trace(
                "reference_color_saturated_best",
                method=best.method,
                score=best.score,
                area=best.area,
                bbox=best.bbox,
            )
            if best.score <= 0.28 and best.metadata.get("area_fraction", 0.0) >= 0.08:
                return best

        bright = self._reference_candidates_from_color(img, color_family="bright")
        if bright:
            best = bright[0]
            self._trace(
                "reference_color_bright_best",
                method=best.method,
                score=best.score,
                area=best.area,
                bbox=best.bbox,
            )
            if best.score <= 0.32 and best.metadata.get("area_fraction", 0.0) >= 0.08:
                return best

        relaxed = self._reference_candidates_from_relaxed_edges(img)
        if relaxed:
            best = relaxed[0]
            self._trace(
                "reference_relaxed_edges_best",
                method=best.method,
                score=best.score,
                area=best.area,
                bbox=best.bbox,
            )
            if best.score <= 0.35:
                return best

        return None

    def _reference_candidates_from_color(
        self, img: np.ndarray, *, color_family: str
    ) -> List[ReferenceContourCandidate]:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        masks: List[Tuple[str, np.ndarray]] = []

        if color_family == "saturated":
            masks.extend(self._dominant_hue_masks(hsv))
        elif color_family == "bright":
            masks.append(
                (
                    "bright_low_saturation_reference",
                    cv2.inRange(hsv, np.array([0, 0, 135]), np.array([179, 80, 255])),
                )
            )
        else:
            raise ValueError(f"Unknown color family: {color_family}")

        candidates: List[ReferenceContourCandidate] = []
        kernel_width = self._odd_kernel_size(max(9, min(img.shape[:2]) // 120))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, kernel_width))

        for name, mask in masks:
            cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            strategy_candidates = self._candidates_from_contours(
                contours,
                img_shape=img.shape,
                method=name,
                min_area=max(self.page_min_area, int(img.shape[0] * img.shape[1] * 0.015)),
            )
            candidates.extend(strategy_candidates)
            self._trace(
                "reference_color_candidates",
                family=color_family,
                method=name,
                contour_count=len(contours),
                accepted_count=len(strategy_candidates),
                top_candidates=self._candidate_summaries(strategy_candidates[:5]),
            )

        if color_family == "saturated":
            large_surface_candidates = [
                item for item in candidates if item.metadata.get("area_fraction", 0.0) >= 0.25
            ]
            if large_surface_candidates:
                candidates = large_surface_candidates

        return sorted(candidates, key=lambda item: item.score)

    def _dominant_hue_masks(self, hsv: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        hue, saturation, value = cv2.split(hsv)
        saturated_pixels = (saturation > 25) & (value > 50)
        if not np.any(saturated_pixels):
            return []

        hist = cv2.calcHist([hue], [0], saturated_pixels.astype(np.uint8), [18], [0, 180]).reshape(-1)
        top_bins = np.argsort(hist)[::-1][:4]

        masks: List[Tuple[str, np.ndarray]] = []
        for bin_index in top_bins:
            if hist[bin_index] < 0.02 * float(saturated_pixels.sum()):
                continue
            center = int((bin_index + 0.5) * 10)
            lower = center - 12
            upper = center + 12
            base_mask = cv2.inRange(hsv, np.array([0, 25, 50]), np.array([179, 220, 255]))
            if lower < 0:
                hue_mask = cv2.bitwise_or(
                    cv2.inRange(hue, 0, upper),
                    cv2.inRange(hue, 180 + lower, 179),
                )
            elif upper > 179:
                hue_mask = cv2.bitwise_or(
                    cv2.inRange(hue, lower, 179),
                    cv2.inRange(hue, 0, upper - 180),
                )
            else:
                hue_mask = cv2.inRange(hue, lower, upper)
            masks.append((f"saturated_hue_{center}", cv2.bitwise_and(base_mask, hue_mask)))

        return masks

    def _reference_candidates_from_relaxed_edges(self, img: np.ndarray) -> List[ReferenceContourCandidate]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 1)
        canny = cv2.Canny(blur, 100, 100)
        kernel = np.ones((self.page_kernel_size, self.page_kernel_size), np.uint8)
        threshold = cv2.erode(cv2.dilate(canny, kernel, iterations=3), kernel, iterations=2)
        contours, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        candidates = self._candidates_from_contours(
            contours,
            img_shape=img.shape,
            method="canny_retr_list_relaxed",
            min_area=self.page_min_area,
            epsilon_values=(0.01, 0.015, 0.02, 0.03, 0.05, 0.08),
        )
        self._trace(
            "reference_relaxed_edge_candidates",
            contour_count=len(contours),
            accepted_count=len(candidates),
            top_candidates=self._candidate_summaries(candidates[:5]),
        )
        return sorted(candidates, key=lambda item: item.score)

    def _candidates_from_contours(
        self,
        contours: Sequence[np.ndarray],
        *,
        img_shape: Tuple[int, ...],
        method: str,
        min_area: int,
        epsilon_values: Sequence[float] = (0.01, 0.02, 0.03, 0.05, 0.08),
    ) -> List[ReferenceContourCandidate]:
        candidates: List[ReferenceContourCandidate] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_area:
                continue
            peri = cv2.arcLength(contour, True)
            if peri <= 0:
                continue
            for epsilon in epsilon_values:
                approx = cv2.approxPolyDP(contour, epsilon * peri, True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    candidate = self._make_reference_candidate(
                        points=approx,
                        raw_contour=contour,
                        method=method,
                        metadata={"approx_epsilon": float(epsilon), "corner_count": int(len(approx))},
                        img_shape=img_shape,
                    )
                    candidates.append(candidate)
                    break
        return sorted(candidates, key=lambda item: item.score)

    def _make_reference_candidate(
        self,
        *,
        points: np.ndarray,
        raw_contour: np.ndarray,
        method: str,
        metadata: Dict[str, Any],
        img_shape: Tuple[int, ...],
    ) -> ReferenceContourCandidate:
        points = points.astype(np.int32)
        area = float(cv2.contourArea(raw_contour))
        x, y, w, h = cv2.boundingRect(points)
        rect = cv2.minAreaRect(points)
        (_, _), (rect_w, rect_h), angle = rect
        image_h, image_w = img_shape[:2]
        ref_ratio = max(self.ref_w_mm, self.ref_h_mm) / max(1.0, min(self.ref_w_mm, self.ref_h_mm))
        rect_ratio = max(rect_w, rect_h) / max(1.0, min(rect_w, rect_h))
        ratio_error = abs(float(np.log(max(rect_ratio, 1e-6) / ref_ratio)))
        touches_edge = x <= 2 or y <= 2 or x + w >= image_w - 2 or y + h >= image_h - 2
        area_fraction = area / float(image_h * image_w)
        fill_ratio = area / float(max(1, w * h))
        score = ratio_error
        score += 0.45 if touches_edge else 0.0
        score += max(0.0, 0.05 - area_fraction) * 3.0
        score += max(0.0, 0.55 - fill_ratio)

        candidate_metadata = dict(metadata)
        candidate_metadata.update(
            {
                "rect_ratio": float(rect_ratio),
                "reference_ratio": float(ref_ratio),
                "ratio_error": float(ratio_error),
                "touches_edge": bool(touches_edge),
                "area_fraction": float(area_fraction),
                "fill_ratio": float(fill_ratio),
                "angle": float(angle),
            }
        )
        return ReferenceContourCandidate(
            points=points,
            raw_contour=raw_contour,
            area=area,
            bbox=(int(x), int(y), int(w), int(h)),
            method=method,
            score=float(score),
            metadata=candidate_metadata,
        )

    def _candidate_summaries(self, candidates: Sequence[ReferenceContourCandidate]) -> List[Dict[str, Any]]:
        return [
            {
                "method": c.method,
                "area": round(c.area, 2),
                "bbox": c.bbox,
                "score": round(c.score, 4),
                "rect_ratio": round(float(c.metadata.get("rect_ratio", 0.0)), 4),
                "touches_edge": bool(c.metadata.get("touches_edge", False)),
            }
            for c in candidates
        ]

    def _contour_summaries(self, contours: Sequence[Any]) -> List[Dict[str, Any]]:
        return [
            {
                "corners": int(c[0]),
                "area": round(float(c[1]), 2),
                "bbox": tuple(int(v) for v in c[3]),
            }
            for c in contours
        ]


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
