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
        sample_fps: float = 10.0,
        min_conf: float = 0.4,
    ) -> List[Tuple[float, float, float]]:
        """
        Berechnet Gesichtszentren. Erkennt Kameraschnitte und ist robuster gegen "Verlieren".
        """
        try:
            import cv2
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
        except ImportError:
            return []

        if not Path(self.face_model_path).exists():
            return []

        base_options = python.BaseOptions(model_asset_path=self.face_model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=2,  # Wir schauen nach bis zu 2 Gesichtern, um das beste zu wählen
            min_face_detection_confidence=min_conf,
            min_tracking_confidence=min_conf
        )

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

        with vision.FaceLandmarker.create_from_options(options) as landmarker:
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
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                res = landmarker.detect(mp_image)
                t_rel = curr_f / fps - start_s
                
                found = False
                if res.face_landmarks:
                    # Bestes Gesicht wählen (das größte/zentralste)
                    best_score = -1
                    best_center = None
                    
                    for landmarks in res.face_landmarks:
                        cx = sum([lm.x for lm in landmarks]) / len(landmarks) * W
                        cy = sum([lm.y for lm in landmarks]) / len(landmarks) * H
                        
                        # Bewertung: Größe der Bounding Box + Nähe zur Mitte
                        xs = [lm.x for lm in landmarks]
                        ys = [lm.y for lm in landmarks]
                        area = (max(xs) - min(xs)) * (max(ys) - min(ys))
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
                    # Wenn verloren, halten wir die Position kurz oder gehen langsam zur Mitte
                    # Bei einem Schnitt setzen wir auf Mitte zurück
                    if is_cut:
                        last_cx, last_cy = W / 2, H / 2
                    out.append((t_rel, last_cx, last_cy))
                
                next_f = curr_f + step
                cap.set(cv2.CAP_PROP_POS_FRAMES, next_f)
                curr_f = next_f
                
        cap.release()
        return out

    def compute_pose_head_centers(
        self, video_path: Path, start_s: float, end_s: float, sample_fps: float = 8.0
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

    def smooth_track(self, track: List[Tuple[float, float, float]], win_sec: float) -> List[Tuple[float, float, float]]:
        if not track: return []
        alpha = 0.08 / max(0.05, win_sec) # Noch weicher
        sx, sy = track[0][1], track[0][2]
        new_track = [(track[0][0], sx, sy)]
        for i in range(1, len(track)):
            t, x, y = track[i]
            dt = max(0.001, t - track[i-1][0])
            c_alpha = min(1.0, alpha * (dt / 0.033))
            sx = c_alpha * x + (1 - c_alpha) * sx
            sy = c_alpha * y + (1 - c_alpha) * sy
            new_track.append((t, sx, sy))
        return new_track

    def reduce_keyframes(self, track: List[Tuple[float, float, float]], max_keys: int) -> List[Tuple[float, float, float]]:
        if not track or len(track) <= max_keys: return track
        return [track[round(i * (len(track)-1) / (max_keys-1))] for i in range(max_keys)]
