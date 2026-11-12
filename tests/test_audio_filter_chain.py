from pathlib import Path

from domains.video.renderer import EBU_R128_AUDIO_FILTER_CHAIN, FFmpegRenderer


def test_ebu_r128_chain_contains_expected_filters():
    assert "highpass=f=80" in EBU_R128_AUDIO_FILTER_CHAIN
    assert "afftdn=nf=-25" in EBU_R128_AUDIO_FILTER_CHAIN
    assert (
        "acompressor=threshold=-18dB:ratio=3:attack=15:release=150:makeup=2"
        in EBU_R128_AUDIO_FILTER_CHAIN
    )
    assert (
        "loudnorm=I=-14:TP=-1.0:LRA=9:print_format=summary"
        in EBU_R128_AUDIO_FILTER_CHAIN
    )


def test_no_music_branch_uses_ebu_r128_chain():
    renderer = FFmpegRenderer()
    cmd = renderer.build_ffmpeg_cmd(
        input_video=Path("in.mp4"),
        start_ts=0.0,
        duration=10.0,
        out_file=Path("out.mp4"),
        target_w=1080,
        target_h=1920,
    )
    assert "-af" in cmd
    af_chain = cmd[cmd.index("-af") + 1]
    assert EBU_R128_AUDIO_FILTER_CHAIN in af_chain


def test_music_branch_uses_ebu_r128_chain_and_true_peak_limiter():
    renderer = FFmpegRenderer()
    cmd = renderer.build_ffmpeg_cmd(
        input_video=Path("in.mp4"),
        start_ts=0.0,
        duration=10.0,
        out_file=Path("out.mp4"),
        target_w=1080,
        target_h=1920,
        music_file=Path("music.mp3"),
        music_volume=0.08,
    )
    assert "-filter_complex" in cmd
    filter_complex = cmd[cmd.index("-filter_complex") + 1]
    assert EBU_R128_AUDIO_FILTER_CHAIN in filter_complex
    assert "alimiter=limit=-1.0dB" in filter_complex
