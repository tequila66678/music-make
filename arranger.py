"""
跑操音乐编排核心引擎  v2 — 纯 numpy/soundfile/librosa，零外部依赖

功能：
  - BPM 自动检测
  - 变速匹配目标 BPM
  - 多段编排（入场→跑操→放松→退场）
  - 交叉淡入淡出混音
  - 导出单文件 WAV（如需 MP3 装 ffmpeg 后改 format="mp3"）
"""

import os
import json
import asyncio
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import soundfile as sf


# ══════════════════════════════════════════════
# 数据模型
# ══════════════════════════════════════════════

@dataclass
class Track:
    """单个音频轨道"""
    filepath: str
    name: str = ""
    bpm: float = 0.0
    duration: float = 0.0         # 秒
    samples: Optional[np.ndarray] = None   # 原始音频数据
    sample_rate: int = 0

    def __repr__(self):
        return f"Track({self.name}, {self.bpm:.1f}BPM, {self.duration:.1f}s)"


@dataclass
class Segment:
    """编排段落"""
    label: str
    target_bpm: int
    duration: float       # 目标时长（秒）
    audio: Optional[np.ndarray] = None
    sample_rate: int = 44100


@dataclass
class Transition:
    """段间过渡语"""
    label: str = ""              # 如 "入场→跑操"
    text: str = ""               # 过渡语文本（空则跳过）
    audio: Optional[np.ndarray] = None  # 生成的语音数据
    audio_path: str = ""         # 手动上传的音频路径（优先于 TTS）
    tts_voice: str = "zh-CN-YunjianNeural"  # TTS 发音人（默认激情男声）
    sentence_gap_sec: float = 0.5    # 句间停顿（秒），0=不停顿
    lead_silence_sec: float = 0.0    # 前导静音（秒）
    tail_silence_sec: float = 1.0    # 尾随静音（秒）
    volume: float = 1.0              # 音量倍率，1.0=原始，最高3.0
    sample_rate: int = 44100


# 可用 TTS 音色
TTS_VOICES = {
    "zh-CN-YunjianNeural": "云健 (男·激情·适合口令)",
    "zh-CN-YunyangNeural": "云扬 (男·专业·新闻播报)",
    "zh-CN-YunxiNeural": "云希 (男·活泼·年轻有力)",
    "zh-CN-YunxiaNeural": "云夏 (男·可爱)",
    "zh-CN-XiaoxiaoNeural": "晓晓 (女·温柔)",
}


# 默认过渡语（使用激情男声 Yunjian）
DEFAULT_TRANSITIONS = [
    Transition(
        label="入场→跑操",
        text="全体立正，向前看齐，向前看。下面我们进行跑操，请注意班级间距，全体注意，跑步走！",
        tts_voice="zh-CN-YunjianNeural",
        sentence_gap_sec=0.5,
        lead_silence_sec=0.0,
        tail_silence_sec=1.0,
    ),
    Transition(
        label="跑操→放松",
        text="跑步结束，各班原地踏步，调整呼吸。下面进行放松活动。",
        tts_voice="zh-CN-YunjianNeural",
        sentence_gap_sec=0.5,
        lead_silence_sec=0.0,
        tail_silence_sec=1.0,
    ),
    Transition(
        label="放松→退场",
        text="放松活动结束，请各班有序退场，注意脚下安全。",
        tts_voice="zh-CN-YunjianNeural",
        sentence_gap_sec=0.5,
        lead_silence_sec=0.0,
        tail_silence_sec=1.0,
    ),
]


# ══════════════════════════════════════════════
# BPM 检测
# ══════════════════════════════════════════════

def detect_bpm(filepath: str) -> float:
    """检测音频 BPM"""
    import librosa
    y, sr = librosa.load(filepath, sr=None, duration=60)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(tempo[0]) if hasattr(tempo, '__iter__') else float(tempo)
    return round(bpm, 1)


def load_track(filepath: str, target_sr: int = 44100) -> Track:
    """加载音频文件并检测 BPM"""
    name = os.path.splitext(os.path.basename(filepath))[0]
    samples, sr = sf.read(filepath)

    # 统一采样率
    if sr != target_sr and len(samples) > 0:
        from scipy.signal import resample
        num_new = int(len(samples) * target_sr / sr)
        if samples.ndim == 1:
            samples = np.interp(np.linspace(0, len(samples)-1, num_new),
                               np.arange(len(samples)), samples)
        else:
            # 立体声 → 下混单声道
            samples = samples.mean(axis=1)
            samples = np.interp(np.linspace(0, len(samples)-1, num_new),
                               np.arange(len(samples)), samples)
        sr = target_sr
    elif samples.ndim > 1:
        samples = samples.mean(axis=1)  # 立体声 → 单声道

    duration = len(samples) / sr

    try:
        bpm = detect_bpm(filepath)
    except Exception:
        bpm = 120.0
    # BPM 检测失败时（纯音调等）使用默认值
    if bpm <= 0:
        bpm = 120.0

    return Track(
        filepath=filepath,
        name=name,
        bpm=bpm,
        duration=duration,
        samples=samples.astype(np.float32),
        sample_rate=sr,
    )


# ══════════════════════════════════════════════
# 变速 & 混音（纯 numpy）
# ══════════════════════════════════════════════

def time_stretch(samples: np.ndarray, orig_bpm: float, target_bpm: float, sr: int) -> np.ndarray:
    """
    BPM 变速：重新采样实现变速不变调
    原理：调整采样率 → 改变播放速度 → 重采样回原始采样率
    """
    if orig_bpm <= 0 or target_bpm <= 0:
        return samples

    if abs(target_bpm - orig_bpm) < 2:
        return samples

    ratio = target_bpm / orig_bpm
    new_len = int(len(samples) / ratio)
    old_idx = np.linspace(0, len(samples) - 1, new_len)
    return np.interp(old_idx, np.arange(len(samples)), samples).astype(np.float32)


def crossfade(a: np.ndarray, b: np.ndarray, fade_samples: int) -> np.ndarray:
    """两段音频交叉淡入淡出拼接"""
    if fade_samples <= 0:
        return np.concatenate([a, b])

    fade = min(fade_samples, min(len(a), len(b)) // 2)
    if fade < 1:
        return np.concatenate([a, b])

    # 重叠区域：a 淡出 + b 淡入
    ramp_out = np.linspace(1.0, 0.0, fade, dtype=np.float32)
    ramp_in  = np.linspace(0.0, 1.0, fade, dtype=np.float32)

    a_end = a[-fade:] * ramp_out
    b_start = b[:fade] * ramp_in

    return np.concatenate([a[:-fade], a_end + b_start, b[fade:]]).astype(np.float32)


# ══════════════════════════════════════════════
# TTS 语音合成（edge-tts）
# ══════════════════════════════════════════════

async def _edge_tts_async(text: str, output_path: str, voice: str = "zh-CN-XiaoxiaoNeural"):
    """异步调用 edge-tts 生成语音文件"""
    import edge_tts
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)


def generate_tts(text: str, output_path: str, voice: str = "zh-CN-XiaoxiaoNeural") -> str:
    """生成 TTS 语音文件，返回文件路径"""
    try:
        asyncio.run(_edge_tts_async(text, output_path, voice))
    except RuntimeError:
        # 已有 event loop（如 Flask 环境）
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_edge_tts_async(text, output_path, voice))
    return output_path


def text_to_speech_array(text: str, sample_rate: int = 44100,
                         voice: str = "zh-CN-YunjianNeural",
                         sentence_gap_sec: float = 0.5,
                         lead_silence_sec: float = 0.0,
                         tail_silence_sec: float = 1.0,
                         volume: float = 1.0) -> np.ndarray:
    """
    将文字转语音，返回 numpy 音频数组

    支持:
      - 按句号/感叹号/问号/换行拆句，句间插入停顿
      - 前导静音 / 尾随静音
      - 多音色选择
    """
    import tempfile
    import re

    # 1. 拆句（保留标点跟在句末）
    sentences = re.split(r'(?<=[。！？\n])', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        sentences = [text] if text.strip() else []

    all_parts = []

    # 2. 前导静音
    if lead_silence_sec > 0:
        all_parts.append(np.zeros(int(lead_silence_sec * sample_rate), dtype=np.float32))

    # 3. 逐句生成 TTS + 句间停顿
    for i, sent in enumerate(sentences):
        fd, tmp_path = tempfile.mkstemp(suffix='.mp3')
        os.close(fd)
        try:
            generate_tts(sent, tmp_path, voice)
            samples, sr = sf.read(tmp_path)
            # 统一采样率 + 单声道
            if sr != sample_rate or samples.ndim > 1:
                if samples.ndim > 1:
                    samples = samples.mean(axis=1)
                if sr != sample_rate:
                    num_new = int(len(samples) * sample_rate / sr)
                    samples = np.interp(
                        np.linspace(0, len(samples)-1, num_new),
                        np.arange(len(samples)), samples)
            all_parts.append(samples.astype(np.float32))

            # 句间停顿（最后一句不加）
            if i < len(sentences) - 1 and sentence_gap_sec > 0:
                all_parts.append(np.zeros(int(sentence_gap_sec * sample_rate), dtype=np.float32))
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    # 4. 尾随静音
    if tail_silence_sec > 0:
        all_parts.append(np.zeros(int(tail_silence_sec * sample_rate), dtype=np.float32))

    if not all_parts:
        return np.zeros(int(sample_rate * 0.5), dtype=np.float32)

    result = np.concatenate(all_parts).astype(np.float32)

    # 音量调节（避免削波）
    if volume != 1.0:
        result = (result * volume).clip(-1.0, 1.0)

    return result


def load_transition_audio(path: str, sample_rate: int = 44100) -> np.ndarray:
    """加载自定义过渡语音文件"""
    samples, sr = sf.read(path)
    if sr != sample_rate:
        num_new = int(len(samples) * sample_rate / sr)
        if samples.ndim == 1:
            samples = np.interp(
                np.linspace(0, len(samples)-1, num_new),
                np.arange(len(samples)), samples)
        else:
            samples = samples.mean(axis=1)
            samples = np.interp(
                np.linspace(0, len(samples)-1, num_new),
                np.arange(len(samples)), samples)
    elif samples.ndim > 1:
        samples = samples.mean(axis=1)
    return samples.astype(np.float32)


def arrange_audio(
    tracks: list[Track],
    target_bpm: int,
    total_duration: float,
    sample_rate: int,
    fade_sec: float = 2.0,
) -> np.ndarray:
    """
    将多首曲目编排为连续音频

    - target_bpm > 0: 自动变速匹配目标 BPM
    - target_bpm = 0: 使用原速不拉伸（快速模式）
    - 循环使用曲目直到达到目标时长
    - 交叉淡入淡出过渡
    """
    if not tracks:
        return np.zeros(int(total_duration * sample_rate), dtype=np.float32)

    target_samples = int(total_duration * sample_rate)
    fade_samples = int(fade_sec * sample_rate)

    # 变速所有曲目（target_bpm=0 时跳过变速，保持原速）
    no_stretch = (target_bpm <= 0)
    if no_stretch:
        print(f"  原速模式: 不拉伸")
        stretched = [t.samples for t in tracks]
    else:
        stretched = [
            time_stretch(t.samples, t.bpm, target_bpm, sample_rate)
            for t in tracks
        ]

    # 循环拼接
    result = np.array([], dtype=np.float32)
    idx = 0
    while len(result) < target_samples:
        seg = stretched[idx % len(stretched)]
        remaining = target_samples - len(result)
        if len(seg) > remaining:
            seg = seg[:remaining]

        if len(result) == 0:
            result = seg
        else:
            result = crossfade(result, seg, fade_samples)

        idx += 1

    if len(result) > target_samples:
        result = result[:target_samples]

    return result


# ══════════════════════════════════════════════
# 预设模板
# ══════════════════════════════════════════════

PRESET_STANDARD = [
    Segment(label="入场", target_bpm=120, duration=3 * 60, sample_rate=44100),
    Segment(label="跑操", target_bpm=140, duration=22 * 60, sample_rate=44100),
    Segment(label="放松", target_bpm=100, duration=3 * 60, sample_rate=44100),
    Segment(label="退场", target_bpm=120, duration=2 * 60, sample_rate=44100),
]

PRESET_QUICK = [
    Segment(label="入场", target_bpm=130, duration=2 * 60, sample_rate=44100),
    Segment(label="跑操", target_bpm=150, duration=15 * 60, sample_rate=44100),
    Segment(label="放松", target_bpm=100, duration=2 * 60, sample_rate=44100),
    Segment(label="退场", target_bpm=120, duration=1 * 60, sample_rate=44100),
]


# ══════════════════════════════════════════════
# 编排主流程
# ══════════════════════════════════════════════

def build_arrangement(
    segment_tracks: dict[str, list[Track]],
    preset: list[Segment],
    fade_sec: float = 2.0,
    sample_rate: int = 44100,
) -> list[Segment]:
    """构建完整编排"""
    root_tracks = segment_tracks.get("_root", [])
    for seg_template in preset:
        seg_template.sample_rate = sample_rate
        tracks = segment_tracks.get(seg_template.label)
        # 如果该标签没有专属子文件夹，使用通用回退池
        if not tracks and root_tracks:
            tracks = root_tracks
        if not tracks:
            print(f"  ! [{seg_template.label}] 无曲目，静音填充")
            seg_template.audio = np.zeros(
                int(seg_template.duration * sample_rate), dtype=np.float32
            )
        else:
            seg_template.audio = arrange_audio(
                tracks,
                target_bpm=seg_template.target_bpm,
                total_duration=seg_template.duration,
                sample_rate=sample_rate,
                fade_sec=fade_sec,
            )
        dur = len(seg_template.audio) / sample_rate if seg_template.audio is not None else 0
        print(f"  ok [{seg_template.label}] {seg_template.target_bpm}BPM {dur:.0f}s")
    return preset


def export_arrangement(
    segments: list[Segment],
    output_path: str,
    sample_rate: int = 44100,
    transitions: list = None,
    fade_sec: float = 2.0,
) -> str:
    """导出编排为单个 WAV，可选插入段间过渡语"""
    if not segments:
        raise ValueError("无有效段落")

    # 构建交织序列：seg0, trans0, seg1, trans1, seg2, trans2, seg3
    # 过渡语独立生效：不依赖前后段落是否有音频，有内容就插入
    sequence = []
    for i, seg in enumerate(segments):
        if seg.audio is not None and len(seg.audio) > 0:
            sequence.append(seg.audio)
        # 插入过渡语（不限 i < len(segments)-1，支持末尾过渡语）
        if transitions and i < len(transitions):
            trans = transitions[i]
            trans_audio = None
            # 优先使用手动上传的音频
            if trans.audio_path and os.path.exists(trans.audio_path):
                try:
                    trans_audio = load_transition_audio(trans.audio_path, sample_rate)
                    print(f"  过渡 [{trans.label}] 加载自定义音频 {os.path.basename(trans.audio_path)}")
                except Exception as e:
                    print(f"  过渡 [{trans.label}] 加载失败: {e}")
            # 否则用 TTS 生成
            elif trans.text and trans.text.strip():
                try:
                    print(f"  过渡 [{trans.label}] TTS 生成: {trans.text[:30]}...")
                    trans_audio = text_to_speech_array(
                        trans.text, sample_rate,
                        voice=trans.tts_voice,
                        sentence_gap_sec=trans.sentence_gap_sec,
                        lead_silence_sec=trans.lead_silence_sec,
                        tail_silence_sec=trans.tail_silence_sec,
                        volume=trans.volume,
                    )
                except Exception as e:
                    print(f"  过渡 [{trans.label}] TTS 失败: {e}")
            if trans_audio is not None and len(trans_audio) > 0:
                sequence.append(trans_audio)

    if not sequence:
        raise ValueError("无有效音频数据")

    # 拼接所有段落+过渡语（带淡入淡出）
    fade_samples = int(fade_sec * sample_rate)
    combined = sequence[0]
    for item in sequence[1:]:
        combined = crossfade(combined, item, fade_samples)

    sf.write(output_path, combined, sample_rate)
    dur = len(combined) / sample_rate
    print(f"\n  导出: {output_path}")
    print(f"  总时长: {int(dur // 60)}分{int(dur % 60)}秒")
    return output_path


def load_music_directory(directory: str) -> dict[str, list[Track]]:
    """
    从目录加载音乐

    支持两种布局，可混合使用:
      A) 分阶段子文件夹（优先）: music/入场/  music/跑操/  music/xxx/
         每个子文件夹名 = 段落标签名，支持任意动态段落
      B) 根目录直放（回退池）: music/cool.wav  music/run.wav
         根目录下的音频作为"通用池"，匹配不到子文件夹的段落自动使用

    优先匹配同名子文件夹，找不到则用通用池
    """
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"目录不存在: {directory}")

    EXT = ('.mp3', '.wav', '.flac', '.m4a', '.ogg')
    segment_tracks = {}

    # 1. 加载根目录音频 → 通用回退池
    root_tracks = []
    try:
        entries = sorted(os.listdir(directory))
    except PermissionError:
        entries = []

    for entry in entries:
        full_path = os.path.join(directory, entry)
        if os.path.isfile(full_path) and entry.lower().endswith(EXT):
            try:
                t = load_track(full_path)
                root_tracks.append(t)
                print(f"  加载 [通用] {t}")
            except Exception as e:
                print(f"  失败 [通用] {entry}: {e}")

    # 2. 加载各子目录音频 → 绑定到对应段落标签
    for entry in entries:
        phase_dir = os.path.join(directory, entry)
        if not os.path.isdir(phase_dir):
            continue
        tracks = []
        for f in sorted(os.listdir(phase_dir)):
            if f.lower().endswith(EXT):
                try:
                    t = load_track(os.path.join(phase_dir, f))
                    tracks.append(t)
                    print(f"  加载 [{entry}] {t}")
                except Exception as e:
                    print(f"  失败 [{entry}] {f}: {e}")
        if tracks:
            segment_tracks[entry] = tracks

    # 3. 对没有专属子文件夹的段落，使用通用回退池
    if root_tracks:
        segment_tracks["_root"] = root_tracks

    return segment_tracks


def save_project(
    segments: list[Segment],
    segment_tracks: dict,
    output_dir: str,
    basename: str = "running_arrangement",
    transitions: list = None,
    fade_sec: float = 2.0,
):
    """保存项目配置 + 导出 WAV"""
    config = {
        "segments": [
            {
                "label": s.label,
                "target_bpm": s.target_bpm,
                "duration_sec": s.duration,
                "tracks_used": [t.name for t in segment_tracks.get(s.label, [])],
            }
            for s in segments
        ]
    }
    if transitions:
        config["transitions"] = [
            {"label": t.label, "text": t.text, "audio_path": t.audio_path}
            for t in transitions
        ]
    cfg_path = os.path.join(output_dir, f"{basename}.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"  配置: {cfg_path}")

    wav_path = export_arrangement(
        segments,
        os.path.join(output_dir, f"{basename}.wav"),
        transitions=transitions,
        fade_sec=fade_sec,
    )
    return cfg_path, wav_path
