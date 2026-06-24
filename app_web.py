"""
跑操音乐编排器 — Web 前端
启动: python app_web.py
自动打开浏览器访问 http://localhost:5000
"""
import os
import sys
import json
import io
import webbrowser
import threading
from pathlib import Path

# 修复 Windows GBK 终端下的 emoji/中文输出问题
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import soundfile as sf
from flask import Flask, request, jsonify, render_template, send_file, make_response
from werkzeug.utils import secure_filename

from arranger import (
    Track,
    Segment,
    Transition,
    DEFAULT_TRANSITIONS,
    TTS_VOICES,
    load_track,
    build_arrangement,
    export_arrangement,
    load_music_directory,
    time_stretch,
    arrange_audio,
    crossfade,
    save_project,
    generate_tts,
    text_to_speech_array,
)

# ── 配置 ──────────────────────────────
BASE_DIR = Path(__file__).parent
MUSIC_DIR = BASE_DIR / "music"
OUTPUT_DIR = BASE_DIR / "output"
MUSIC_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

ALLOWED_EXT = {'.mp3', '.wav', '.flac', '.m4a', '.ogg', '.aiff', '.aac'}

import re
def safe_dirname(name: str) -> str:
    """清理文件夹名：去掉路径分隔符和危险字符，保留中文"""
    name = name.strip()
    if not name:
        return "未分类"
    return re.sub(r'[<>:"/\\|?*\x00]', '_', name)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB

# ── 路由: 主页 ─────────────────────────
@app.route('/')
def index():
    resp = make_response(render_template('index.html'))
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp

# ── 路由: 上传 ─────────────────────────
@app.route('/upload', methods=['POST'])
def upload():
    phase = request.form.get('phase', '未分类')
    # 直接用标签做文件夹名（支持任意动态段落）
    dir_name = safe_dirname(phase)
    phase_dir = MUSIC_DIR / dir_name
    phase_dir.mkdir(exist_ok=True)

    files = request.files.getlist('files')
    if not files or not files[0].filename:
        return jsonify({"error": "未选择文件"}), 400

    results = []
    for f in files:
        ext = Path(f.filename).suffix.lower()
        if ext not in ALLOWED_EXT:
            continue

        filename = secure_filename(f.filename)
        # 避免重名
        save_path = phase_dir / filename
        counter = 1
        while save_path.exists():
            stem = Path(filename).stem
            save_path = phase_dir / f"{stem}_{counter}{ext}"
            counter += 1

        f.save(str(save_path))

        try:
            track = load_track(str(save_path))
            results.append({
                "name": track.name,
                "bpm": track.bpm,
                "duration": round(track.duration, 1),
                "filename": save_path.name,
            })
        except Exception as e:
            results.append({
                "name": Path(filename).stem,
                "bpm": 0,
                "duration": 0,
                "filename": save_path.name,
                "error": str(e),
            })

    return jsonify({"tracks": results, "phase": phase})


# ── 路由: 列出音乐 ─────────────────────
@app.route('/api/music', methods=['GET'])
def list_music():
    """返回已上传的所有音乐文件及 BPM，动态扫描所有子目录"""
    segment_tracks = load_music_directory(str(MUSIC_DIR))
    result = {}
    for phase_label, track_list in segment_tracks.items():
        result[phase_label] = [
            {
                "name": t.name,
                "bpm": t.bpm,
                "duration": round(t.duration, 1),
                "filename": Path(t.filepath).name,
            }
            for t in track_list
        ]

    # 去重: 如果根目录文件与子目录文件同名，子目录优先
    if "_root" in result:
        root_names = set()
        for phase_label in list(result.keys()):
            if phase_label != "_root":
                for t in result[phase_label]:
                    root_names.add(t["filename"])
        result["_root"] = [t for t in result["_root"] if t["filename"] not in root_names]
        if not result["_root"]:
            del result["_root"]

    return jsonify({"tracks": result})


# ── 路由: 删除音乐 ─────────────────────
@app.route('/api/music/<phase>/<filename>', methods=['DELETE'])
def delete_music(phase, filename):
    phase_dir = MUSIC_DIR / safe_dirname(phase)
    file_path = phase_dir / secure_filename(filename)
    if file_path.exists():
        file_path.unlink()
    return jsonify({"ok": True})


# ── 路由: TTS 音色列表 ─────────────────
@app.route('/api/tts/voices', methods=['GET'])
def tts_voices():
    """返回可用 TTS 音色"""
    return jsonify({"voices": TTS_VOICES})

# ── 路由: TTS 预览 ─────────────────────
@app.route('/api/tts/preview', methods=['POST'])
def tts_preview():
    """生成过渡语 TTS 预览音频"""
    data = request.get_json()
    text = (data or {}).get('text', '').strip()
    voice = (data or {}).get('voice', 'zh-CN-YunjianNeural')
    sentence_gap = float((data or {}).get('sentence_gap', 0.5))
    lead_silence = float((data or {}).get('lead_silence', 0.0))
    tail_silence = float((data or {}).get('tail_silence', 1.0))
    volume = float((data or {}).get('volume', 1.0))
    if not text:
        return jsonify({"error": "文本为空"}), 400

    import tempfile
    fd, tmp_path = tempfile.mkstemp(suffix='.wav')
    os.close(fd)
    try:
        audio = text_to_speech_array(
            text, sample_rate=44100, voice=voice,
            sentence_gap_sec=sentence_gap,
            lead_silence_sec=lead_silence,
            tail_silence_sec=tail_silence,
            volume=volume,
        )
        sf.write(tmp_path, audio, 44100)
        return send_file(
            tmp_path,
            mimetype='audio/wav',
            as_attachment=True,
            download_name='tts_preview.wav'
        )
    except Exception as e:
        return jsonify({"error": f"TTS 生成失败: {str(e)}"}), 500


# ── 路由: 上传自定义过渡语音 ─────────────
@app.route('/api/tts/upload', methods=['POST'])
def upload_transition_audio():
    """上传自定义过渡语音文件"""
    f = request.files.get('file')
    if not f or not f.filename:
        return jsonify({"error": "未选择文件"}), 400

    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        return jsonify({"error": f"不支持的格式: {ext}"}), 400

    filename = secure_filename(f.filename)
    save_path = OUTPUT_DIR / f"transition_{filename}"
    f.save(str(save_path))
    return jsonify({
        "filename": save_path.name,
        "path": str(save_path),
    })


# ── 路由: 生成编排 ─────────────────────
@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    if not data or 'segments' not in data:
        return jsonify({"error": "缺少 segments 参数"}), 400

    fade_sec = float(data.get('fade_sec', 2.0))
    sample_rate = 44100

    # 1. 构建预设
    preset = []
    for s in data['segments']:
        preset.append(Segment(
            label=s['label'],
            target_bpm=int(s['target_bpm']),
            duration=float(s['duration']),
            sample_rate=sample_rate,
        ))

    # 2. 构建 Transition 对象
    transitions = []
    for t_data in data.get('transitions', []):
        t = Transition(
            label=t_data.get('label', ''),
            text=t_data.get('text', ''),
            audio_path=t_data.get('audio_path', ''),
            tts_voice=t_data.get('tts_voice', 'zh-CN-YunjianNeural'),
            sentence_gap_sec=float(t_data.get('sentence_gap', 0.5)),
            lead_silence_sec=float(t_data.get('lead_silence', 0.0)),
            tail_silence_sec=float(t_data.get('tail_silence', 1.0)),
            volume=float(t_data.get('volume', 1.0)),
            sample_rate=sample_rate,
        )
        transitions.append(t)

    # 3. 加载音乐库
    print("\n[Web] 加载音乐库...")
    segment_tracks = load_music_directory(str(MUSIC_DIR))
    if not segment_tracks:
        return jsonify({"error": "没有找到任何音频文件！请先上传音乐。"}), 400

    # 4. 构建编排
    print("[Web] 构建编排...")
    try:
        segments = build_arrangement(
            segment_tracks,
            preset=preset,
            fade_sec=fade_sec,
            sample_rate=sample_rate,
        )
    except Exception as e:
        return jsonify({"error": f"编排构建失败: {str(e)}"}), 500

    # 5. 导出（含过渡语）
    print("[Web] 导出...")
    import time
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    basename = f"arrangement_{timestamp}"

    try:
        # 使用带过渡语的导出（过滤掉空过渡语）
        active_transitions = [t for t in transitions if t.text.strip() or t.audio_path]
        wav_path = export_arrangement(
            segments,
            str(OUTPUT_DIR / f"{basename}.wav"),
            sample_rate=sample_rate,
            transitions=active_transitions if active_transitions else None,
            fade_sec=fade_sec,
        )
        total_dur = sf.info(str(wav_path)).duration

        # 保存配置
        save_project(segments, segment_tracks, str(OUTPUT_DIR),
                     basename=basename, transitions=active_transitions, fade_sec=fade_sec)

        download_filename = f"{basename}.wav"

    except Exception as e:
        return jsonify({"error": f"导出失败: {str(e)}"}), 500

    return jsonify({
        "filename": download_filename,
        "duration": round(total_dur, 1),
        "segments": [
            {"label": s.label, "target_bpm": s.target_bpm, "duration": s.duration}
            for s in segments
        ],
        "transitions": [
            {"label": t.label, "text": t.text}
            for t in active_transitions
        ] if active_transitions else [],
    })


# ── 路由: 下载 ─────────────────────────
@app.route('/download/<filename>')
def download(filename):
    safe = secure_filename(filename)
    file_path = OUTPUT_DIR / safe
    if not file_path.exists():
        return jsonify({"error": "文件不存在"}), 404
    return send_file(str(file_path), as_attachment=True, download_name=safe)


# ── 启动 ──────────────────────────────
def open_browser():
    webbrowser.open('http://localhost:5000')

if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("  Running Music Arranger v2 (Web)")
    print("=" * 50)
    print(f"  Music dir: {MUSIC_DIR}")
    print(f"  Output dir: {OUTPUT_DIR}")
    print(f"  Open: http://localhost:5000")
    print("=" * 50 + "\n")

    threading.Timer(1.0, open_browser).start()
    app.run(host='0.0.0.0', port=5000, debug=False)
