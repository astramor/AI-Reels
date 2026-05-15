#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import time
from pathlib import Path
from typing import List, Optional, Tuple

class FaceTracker:
    """
    Klasse für präzises Face- und Pose-Tracking in Videos.
    Optimiert für Kameraschnitte und cinematic Follow-Me Effekte.
    """

    def __init__(self):
        self.face_model_path = "models/face_landmarker.task"
        self.pose_model_path = "models/pose_landmarker.task"
        self._check_models()

    def _check_models(self):
        if not Path(self.face_model_path).exists() or not Path(self.pose_model_path).exists():
            print("Warnung: Mediapipe Task-Modelle fehlen in models/. Tracking wird fehlschlagen.")

    def _median(self, lst: List[float]) -> Optional[float]:
        if not lst:
            return None
        s = sorted(lst)
        n = len(s)
        mid = n // 2
        return (s[mid - 1] + s[mid]) / 2 if n % 2 == 0 else s[mid]

    def _detect_camera_cut(self, prev_frame, curr_frame, threshold: float = 30.0) -> bool:
        """
        Einfache Kameraschnitt-Erkennung über Histogramm-Differenz.
        """
        if prev_frame is None or curr_frame is None:
            return False
        import cv2
        hist1 = cv2.calcHist([prev_frame], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([curr_frame], [0], None, [256], [0, 256])
        diff = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return diff < 0.7  # Wenn Korrelation < 70%, ist es wahrscheinlich ein Schnitt

    def compute_face_centers(
        self,
        video_path: Path,
        start_s: float,
        end_s: float,
        sample_fps: float = 20.0,  # Erhöht für flüssigeres Tracking
        min_conf: float = 0.4,
    ) -> List[Tuple[float, float, float]]:
        """
        Berechnet Gesichtszentren mit der robusten FaceDetection-Lösung.
        Optimiert auf 20fps für weichere Bewegungen.
        """
        try:
            import cv2
            import mediapipe as mp
            import mediapipe.python.solutions.face_detection as mp_face
        except ImportError:
            return []

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        start_f = max(0, int(start_s * fps))
        end_f = int(end_s * fps)
        step = max(1, int(fps / max(0.1, sample_fps)))
        
        out = []
        prev_gray = None
        last_cx, last_cy = W / 2, H / 2

        # Wir nutzen die robuste FaceDetection (Bounding Box) statt Landmarker (Mesh)
        with mp_face.FaceDetection(
            model_selection=1, min_detection_confidence=min_conf
        ) as face_detection:
            curr_f = start_f
            cap.set(cv2.CAP_PROP_POS_FRAMES, curr_f)
            
            while curr_f <= end_f:
                ok, frame = cap.read()
                if not ok: break
                
                # Schnitt-Erkennung
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                is_cut = self._detect_camera_cut(prev_gray, gray)
                prev_gray = gray
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = face_detection.process(rgb_frame)
                t_rel = curr_f / fps - start_s
                
                found = False
                if res.detections:
                    # Bestes Gesicht wählen (das größte/zentralste)
                    best_score = -1
                    best_center = None
                    
                    for detection in res.detections:
                        bbox = detection.location_data.relative_bounding_box
                        cx = (bbox.xmin + bbox.width / 2) * W
                        cy = (bbox.ymin + bbox.height / 2) * H
                        
                        area = bbox.width * bbox.height
                        dist_to_last = ((cx - last_cx)**2 + (cy - last_cy)**2)**0.5
                        
                        score = area * (1.0 / (1.0 + dist_to_last / W))
                        if score > best_score:
                            best_score = score
                            best_center = (cx, cy)
                    
                    if best_center:
                        last_cx, last_cy = best_center
                        out.append((t_rel, last_cx, last_cy))
                        found = True

                if not found:
                    if is_cut:
                        last_cx, last_cy = W / 2, H / 2
                    out.append((t_rel, last_cx, last_cy))
                
                next_f = curr_f + step
                cap.set(cv2.CAP_PROP_POS_FRAMES, next_f)
                curr_f = next_f
                
        cap.release()
        return out

    def compute_pose_head_centers(
        self, video_path: Path, start_s: float, end_s: float, sample_fps: float = 12.0
    ) -> List[Tuple[float, float, float]]:
        try:
            import cv2
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
        except ImportError:
            return []

        base_options = python.BaseOptions(model_asset_path=self.pose_model_path)
        options = vision.PoseLandmarkerOptions(base_options=base_options, num_poses=1)

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        out = []
        with vision.PoseLandmarker.create_from_options(options) as landmarker:
            curr_f = max(0, int(start_s * fps))
            end_f = int(end_s * fps)
            step = max(1, int(fps / sample_fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, curr_f)
            
            while curr_f <= end_f:
                ok, frame = cap.read()
                if not ok: break
                
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                res = landmarker.detect(mp_image)
                if res.pose_landmarks:
                    lm = res.pose_landmarks[0]
                    idxs_head = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                    pts = [(lm[i].x * W, lm[i].y * H) for i in idxs_head if lm[i].presence > 0.5]
                    if pts:
                        out.append((curr_f / fps - start_s, sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)))
                
                curr_f += step
                cap.set(cv2.CAP_PROP_POS_FRAMES, curr_f)
        cap.release()
        return out

    def robust_start_center(
        self,
        video_path: Path,
        clip_abs_start: float,
        rng: float = 1.0,
        steps: int = 3,
        grid: float = 0.4,
        face_sample: float = 8.0,
        face_min_conf: float = 0.5,
        use_pose: bool = True,
        debug_dir: Optional[Path] = None,
    ) -> Optional[Tuple[float, float]]:
        half = max(0.5, rng / 2.0)
        win_start = max(0.0, clip_abs_start - half)
        win_end = clip_abs_start + half
        
        # Nutzt jetzt intern die korrigierte compute_face_centers (FaceDetection)
        faces = self.compute_face_centers(
            video_path, win_start, win_end, sample_fps=face_sample, min_conf=face_min_conf
        )
        
        if not faces and use_pose:
            faces = self.compute_pose_head_centers(
                video_path, win_start, win_end, sample_fps=face_sample
            )
            
        if not faces:
            # Fallback auf 1080p Mitte (angepasst an Zielformat)
            return (540, 960)
            
        cx = self._median([p[1] for p in faces])
        cy = self._median([p[2] for p in faces])
        return (cx, cy)

    def apply_dead_zone(self, track: List[Tuple[float, float, float]], dead_zone_px: float) -> List[Tuple[float, float, float]]:
        if not track: return []
        cx, cy = track[0][1], track[0][2]
        new_track = []
        for t, x, y in track:
            dx, dy = x - cx, y - cy
            dist = (dx**2 + dy**2)**0.5
            if dist > dead_zone_px:
                ratio = (dist - dead_zone_px) / dist
                cx += dx * ratio
                cy += dy * ratio
            new_track.append((t, cx, cy))
        return new_track

    def smooth_track(
        self, 
        track: List[Tuple[float, float, float]], 
        window_length: int = 51, 
        polyorder: int = 2, 
        deadzone_px: float = 80.0  # Erhöhte Deadzone: Ignoriert starkes Kopfwackeln
    ) -> List[Tuple[float, float, float]]:
        """
        Beruhigt Tracking-Daten durch einen Savitzky-Golay-Filter und einen 
        'Cinematic Lazy Follow' mit Soft-Deadzone.
        """
        if not track:
            return []
        
        # 1. Savitzky-Golay Filter (Grundrauschen filtern)
        try:
            import numpy as np
            from scipy.signal import savgol_filter
            
            times = np.array([p[0] for p in track])
            xs = np.array([p[1] for p in track])
            ys = np.array([p[2] for p in track])
            
            if len(track) >= window_length:
                xs = savgol_filter(xs, window_length, polyorder)
                ys = savgol_filter(ys, window_length, polyorder)
            elif len(track) > polyorder + 1:
                w = len(track) if len(track) % 2 != 0 else len(track) - 1
                if w > polyorder:
                    xs = savgol_filter(xs, w, polyorder)
                    ys = savgol_filter(ys, w, polyorder)
        except ImportError:
            times = [p[0] for p in track]
            xs = [p[1] for p in track]
            ys = [p[2] for p in track]
        
        # 2. Cinematic Lazy Follow mit Soft Deadzone
        smoothed_track = []
        if len(xs) > 0:
            cam_x, cam_y = xs[0], ys[0]
            
            # Extrem träger Faktor (zuvor 0.15). 0.03 macht Schwenks sehr langsam und majestätisch.
            smoothing_factor = 0.03 
            
            for i in range(len(xs)):
                t = times[i]
                target_x, target_y = xs[i], ys[i]
                
                dx = target_x - cam_x
                dy = target_y - cam_y
                dist = (dx**2 + dy**2)**0.5
                
                # Die Kamera bewegt sich NUR, wenn die Person die Deadzone verlässt
                if dist > deadzone_px:
                    # Wir ziehen die Kamera nicht direkt ins Gesicht, 
                    # sondern nur sanft an den Rand der erlaubten Zone
                    pull_x = target_x - (dx / dist) * deadzone_px
                    pull_y = target_y - (dy / dist) * deadzone_px
                    
                    cam_x += (pull_x - cam_x) * smoothing_factor
                    cam_y += (pull_y - cam_y) * smoothing_factor
                
                smoothed_track.append((float(t), float(cam_x), float(cam_y)))
                
        return smoothed_track

    def interpolate_track_to_fps(self, track: List[Tuple[float, float, float]], target_fps: float = 30.0) -> List[Tuple[float, float, float]]:
        """
        Interpoliert die Tracking-Punkte linear auf eine feste Framerate (z.B. 30 FPS).
        Das zwingt FFmpeg dazu, bei jedem Frame weich zu aktualisieren, statt zwischen 
        weit auseinanderliegenden Keyframes zu springen.
        """
        if not track or len(track) < 2:
            return track
            
        import numpy as np
        
        times = [p[0] for p in track]
        xs = [p[1] for p in track]
        ys = [p[2] for p in track]
        
        # Berechne die neue, dichte Zeitachse
        start_t, end_t = times[0], times[-1]
        num_frames = max(2, int((end_t - start_t) * target_fps) + 1)
        new_times = np.linspace(start_t, end_t, num_frames)
        
        # Lineare Interpolation der X- und Y-Koordinaten
        new_xs = np.interp(new_times, times, xs)
        new_ys = np.interp(new_times, times, ys)
        
        return [(float(t), float(x), float(y)) for t, x, y in zip(new_times, new_xs, new_ys)]

    def reduce_keyframes(self, track: List[Tuple[float, float, float]], max_keys: int = 40) -> List[Tuple[float, float, float]]:
        """Erhöht die Keyframe-Dichte für weichere FFmpeg-Interpolation."""
        if not track or len(track) <= max_keys: return track
        return [track[round(i * (len(track)-1) / (max_keys-1))] for i in range(max_keys)]
