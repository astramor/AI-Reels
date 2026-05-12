# domains/video/renderer.py

import shlex
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple
from loguru import logger

# Import der zentralen Settings
from core.config import settings
from core.time_utils import seconds_to_hms_ms
from core.commands import run_command


class FFmpegRenderer:
    @staticmethod
    def seconds_to_hms(t: float) -> str:
        return seconds_to_hms_ms(t)

    @staticmethod
    def escape_drawtext(txt: str) -> str:
        return (
            txt.strip()
            .replace("\\", "\\\\")
            .replace("'", "'\\''")
            .replace(":", "\\:")
            .replace("\n", "\\\\n")
        )

    def _apply_safe_margin(
        self,
        sw: int,
        sh: int,
        tw: int,
        th: int,
        cx: float,
        cy: float,
        mx: int = 180,
        my: int = 140,
    ) -> Tuple[int, int]:
        min_cx = mx + tw // 2
        max_cx = sw - (mx + tw // 2)
        min_cy = my + th // 2
        max_cy = sh - (my + th // 2)
        cx = int(max(min_cx, min(cx, max_cx)))
        cy = int(max(min_cy, min(cy, max_cy)))
        x = max(0, min(cx - tw // 2, max(0, sw - tw)))
        y = max(0, min(cy - th // 2, max(0, sh - th)))
        return x, y

    def blur_bg_filter(self, target_w: int, target_h: int) -> str:
        return (
            f"split[bg][fg];"
            f"[bg]scale={target_w}:{target_h}:force_original_aspect_ratio=increase,crop={target_w}:{target_h},gblur=sigma=20:steps=2[bg_blur];"
            f"[fg]scale={target_w}:-1:flags=lanczos[fg_scaled];"
            f"[bg_blur][fg_scaled]overlay=(W-w)/2:(H-h)/2:format=auto,format=yuv420p"
        )

    def cover_scale_crop_filter(self, target_w: int, target_h: int) -> str:
        scale = f"scale='if(gt(a,{target_w}/{target_h}),-1,{target_w})':'if(gt(a,{target_w}/{target_h}),{target_h},-1)':flags=lanczos"
        crop = f"crop={target_w}:{target_h}"
        return f"{scale},{crop},format=yuv420p"

    def cover_scale_crop_filter_with_center(
        self,
        src_w: int,
        src_h: int,
        target_w: int,
        target_h: int,
        center_x: float,
        center_y: float,
    ) -> str:
        s = max(target_w / src_w, target_h / src_h)
        sw = int(round(src_w * s))
        sh = int(round(src_h * s))
        scx = center_x * s
        scy = center_y * s
        x = int(round(scx - target_w / 2))
        y = int(round(scy - target_h / 2))
        x = max(0, min(x, max(0, sw - target_w)))
        y = max(0, min(y, max(0, sh - target_h)))
        return f"scale={sw}:{sh}:flags=lanczos,crop={target_w}:{target_h}:{x}:{y},format=yuv420p"

    def cover_scale_crop_filter_with_track(
        self,
        src_w: int,
        src_h: int,
        target_w: int,
        target_h: int,
        track: List[Tuple[float, float, float]],
    ) -> str:
        if not track:
            return self.cover_scale_crop_filter(target_w, target_h)
        s = max(target_w / src_w, target_h / src_h)
        sw = int(round(src_w * s))
        sh = int(round(src_h * s))

        def lerp(keys):
            if not keys:
                return "0"
            if len(keys) == 1:
                return str(int(round(keys[0][1])))
            expr = ""
            for i in range(len(keys) - 1):
                t0, v0 = keys[i]
                t1, v1 = keys[i + 1]
                dt = max(1e-3, t1 - t0)
                cond = f"lt(t,{t1})"
                body = f"(({v0}) + ({v1}-{v0})*((t-({t0}))/{dt}))"
                expr = f"if({cond},{body}," if i == 0 else expr + f"if({cond},{body},"
            expr += f"{keys[-1][1]}" + (")" * (len(keys) - 1))
            return expr

        key_x, key_y = [], []
        for t, cx, cy in track:
            scx = cx * s
            scy = cy * s
            x, y = self._apply_safe_margin(
                sw, sh, target_w, target_h, scx, scy, mx=180, my=140
            )
            key_x.append((round(t, 3), x))
            key_y.append((round(t, 3), y))
        x_clamped = f"max(min({lerp(key_x)},{max(0, sw - target_w)}),0)"
        y_clamped = f"max(min({lerp(key_y)},{max(0, sh - target_h)}),0)"
        return f"scale={sw}:{sh}:flags=lanczos,crop={target_w}:{target_h}:x='{x_clamped}':y='{y_clamped}',format=yuv420p"

    def build_ffmpeg_cmd(
        self,
        input_video: Path,
        start_ts: float,
        duration: float,
        out_file: Path,
        *,
        ass_file: Optional[Path] = None,
        subtitles: Optional[Path] = None,
        overlay_title: Optional[str] = None,
        reencode: bool = True,
        video_codec: str = "libx264",
        audio_codec: str = "aac",
        target_w: Optional[int] = None,
        target_h: Optional[int] = None,
        cover_filter_override: Optional[str] = None,
        loudnorm: bool = False,
        fade_in: bool = False,
        fade_in_sec: float = 0.5,
        fade_out: bool = False,
        fade_sec: float = 1.5,
        overlay_margin_top: int = 140,
        overlay_fontsize: int = 40,
        color_grading: bool = False,
        music_file: Optional[Path] = None,
        music_volume: float = 0.15,
        blur_bg: bool = False,
    ) -> List[str]:

        # --- Settings aus der config.py beziehen ---
        crf = settings.ffmpeg_crf
        preset = settings.ffmpeg_preset
        audio_bitrate = settings.ffmpeg_audio_bitrate
        ducking_threshold = settings.ffmpeg_ducking_threshold
        ducking_ratio = settings.ffmpeg_ducking_ratio
        ducking_attack = settings.ffmpeg_ducking_attack
        ducking_release = settings.ffmpeg_ducking_release

        # HIER: -y hinzugefügt, um Überschreiben zu erzwingen
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "info"]

        # 1. INPUTS
        cmd += ["-ss", self.seconds_to_hms(start_ts)]
        cmd += ["-i", str(input_video)]

        if music_file:
            cmd += ["-stream_loop", "-1", "-i", str(music_file)]

        cmd += ["-t", f"{duration:.3f}"]

        # --- VIDEO FILTER CHAIN ---
        v_filters = []

        if target_w and target_h:
            if blur_bg:
                v_filters.append(self.blur_bg_filter(target_w, target_h))
            elif cover_filter_override:
                v_filters.append(cover_filter_override)
            else:
                v_filters.append(self.cover_scale_crop_filter(target_w, target_h))

        if color_grading:
            v_filters.append("eq=saturation=1.3:contrast=1.1:brightness=-0.02")

        if ass_file and ass_file.exists():
            ass_path = (
                str(ass_file)
                .replace("\\", "/")
                .replace(":", "\\:")
                .replace("'", "'\\''")
            )
            v_filters.append(f"ass='{ass_path}'")
        elif subtitles:
            fs = "Fontname=Arial,Fontsize=6,PrimaryColour=&H0000FFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=1.2,Shadow=0,MarginV=70,Alignment=2,Bold=1".replace(
                "'", r"\'"
            )
            v_filters.append(
                f"subtitles='{subtitles.as_posix()}':charenc=UTF-8:force_style='{fs}'"
            )

        if overlay_title:
            txt = self.escape_drawtext(overlay_title)
            enable_expr = (
                f"enable='gte(t,{fade_in_sec})'"
                if (fade_in and fade_in_sec > 0)
                else ""
            )
            draw_cmd = f"drawtext=text='{txt}':fontcolor=white:fontsize={overlay_fontsize}:line_spacing=12:borderw=5:bordercolor=black:shadowx=3:shadowy=3:shadowcolor=black@0.6:x=(w-text_w)/2:y={overlay_margin_top}"
            if enable_expr:
                draw_cmd += f":{enable_expr}"
            v_filters.append(draw_cmd)

        if fade_in and fade_in_sec > 0:
            v_filters.append(f"fade=t=in:st=0:d={fade_in_sec}")
        if fade_out and duration is not None:
            st = max(0, duration - fade_sec)
            v_filters.append(f"fade=t=out:st={st:.3f}:d={fade_sec:.3f}")

        # --- AUDIO FILTER CHAIN ---
        af_filters = []
        if loudnorm:
            af_filters.append(
                "volume=6.0,dynaudnorm=f=100:g=31:m=10,loudnorm=I=-9:TP=-0.1:LRA=5:print_format=summary"
            )

        if music_file:
            # Voice + Music mixing logic
            voice_chain = "[0:a]"
            if af_filters:
                voice_chain += ",".join(af_filters)
            voice_chain += "[voice_norm];"

            music_chain = f"[1:a]volume={music_volume}[music_vol];"
            music_chain += f"[music_vol][voice_norm]sidechaincompress=threshold={ducking_threshold}:ratio={ducking_ratio}:attack={ducking_attack}:release={ducking_release}[music_ducked];"

            mix_chain = "[voice_norm][music_ducked]amix=inputs=2:duration=first:dropout_transition=0:weights=1 1:normalize=0[summed];"
            mix_chain += "[summed]alimiter=limit=-0.1dB:level_in=1.0:level_out=1.0:asc=1[a_pre_fade]"
            
            a_fades = []
            if fade_in and fade_in_sec > 0:
                a_fades.append(f"afade=t=in:st=0:d={fade_in_sec}")
            if fade_out:
                st = max(0, duration - fade_sec)
                a_fades.append(f"afade=t=out:st={st:.3f}:d={fade_sec:.3f}")

            fade_chain = ""
            if a_fades:
                fade_chain = ";" + "[a_pre_fade]" + ",".join(a_fades) + "[a_out]"
            else:
                fade_chain = ";[a_pre_fade]aformat=sample_fmts=fltp[a_out]"
            
            # Combine video and audio in filter_complex
            v_chain = "[0:v]"
            if v_filters:
                v_chain += ",".join(v_filters)
            v_chain += "[v_out];"

            full_complex = v_chain + voice_chain + music_chain + mix_chain + fade_chain
            cmd += ["-filter_complex", full_complex, "-map", "[v_out]", "-map", "[a_out]"]
        else:
            # No music, simplified chains
            if fade_in and fade_in_sec > 0:
                af_filters.append(f"afade=t=in:st=0:d={fade_in_sec}")
            if fade_out and duration is not None:
                st = max(0, duration - fade_sec)
                af_filters.append(f"afade=t=out:st={st:.3f}:d={fade_sec:.3f}")

            if v_filters:
                cmd += ["-vf", ",".join(v_filters)]
            if af_filters:
                cmd += ["-af", ",".join(af_filters)]

        cmd += ["-c:v", video_codec]
        if "nvenc" in video_codec:
            cmd += [
                "-rc",
                "vbr",
                "-cq",
                str(crf),
                "-preset",
                "p7",
                "-pix_fmt",
                "yuv420p",
            ]
        else:
            cmd += ["-crf", str(crf), "-preset", preset]

        cmd += [
            "-c:a",
            audio_codec,
            "-b:a",
            audio_bitrate,
            "-movflags",
            "+faststart",
            str(out_file),
        ]
        return cmd

    def run_ffmpeg(self, cmd: List[str]) -> None:
        logger.info(f"Starte FFmpeg...")
        try:
            run_command(cmd)
            logger.success("Rendering erfolgreich beendet.")
        except Exception as e:
            logger.error(f"FFmpeg failed: {e}")
            raise RuntimeError(f"ffmpeg failed: {e}") from e
