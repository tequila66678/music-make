"""
跑操音乐编排器 — 命令行工具

用法:
  # 加载音乐目录，使用标准模板
  python app.py music/ --output output.mp3

  # 自定义分段参数
  python app.py music/ --entrance 120 3:00 --run 140 22:00 --cooldown 100 3:00 --exit 120 2:00

  # 列出音乐文件及 BPM
  python app.py music/ --list

  # 预览（仅输出信息，不生成文件）
  python app.py music/ --dry-run
"""

import argparse
import os
import sys
from arranger import (
    load_music_directory,
    load_track,
    build_arrangement,
    export_arrangement,
    save_project,
    PRESET_STANDARD,
    PRESET_QUICK,
    Segment,
    Track,
)


def parse_time(t: str) -> float:
    """解析时间字符串 '3:30' | '210' → 秒数"""
    t = t.strip()
    if ":" in t:
        parts = t.split(":")
        return int(parts[0]) * 60 + int(parts[1])
    return float(t)


def build_custom_preset(args) -> list[Segment]:
    """根据命令行参数构建自定义预设"""
    segments = []
    if args.entrance:
        bpm, dur = args.entrance
        segments.append(Segment(label="入场", target_bpm=int(bpm), duration=parse_time(dur)))
    if args.run:
        bpm, dur = args.run
        segments.append(Segment(label="跑操", target_bpm=int(bpm), duration=parse_time(dur)))
    if args.cooldown:
        bpm, dur = args.cooldown
        segments.append(Segment(label="放松", target_bpm=int(bpm), duration=parse_time(dur)))
    if args.exit_seg:
        bpm, dur = args.exit_seg
        segments.append(Segment(label="退场", target_bpm=int(bpm), duration=parse_time(dur)))
    return segments or None


def cmd_list(args):
    """列出音乐文件及 BPM"""
    music_dir = args.music_dir
    if not os.path.isdir(music_dir):
        print(f"错误: 目录不存在 '{music_dir}'")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  音乐库: {music_dir}")
    print(f"{'='*60}")

    phases = ["入场", "跑操", "放松", "退场", None]
    for phase in phases:
        phase_dir = os.path.join(music_dir, phase) if phase else music_dir
        if not os.path.isdir(phase_dir):
            continue
        label = phase or "根目录"
        print(f"\n  [{label}]")
        files = [f for f in sorted(os.listdir(phase_dir))
                 if f.lower().endswith(('.mp3', '.wav', '.flac', '.m4a', '.ogg'))]
        if not files:
            print(f"    (空)")
            continue
        for f in files:
            fp = os.path.join(phase_dir, f)
            try:
                t = load_track(fp)
                print(f"    {f:45s}  {t.bpm:6.1f} BPM  {t.duration:.0f}s")
            except Exception as e:
                print(f"    {f:45s}  加载失败: {e}")


def cmd_generate(args):
    """生成跑操音乐编排文件"""
    music_dir = args.music_dir
    output = args.output or "running_arrangement.mp3"

    # 确保输出目录存在
    out_dir = os.path.dirname(os.path.abspath(output))

    print(f"\n{'='*60}")
    print(f"  跑操音乐编排器")
    print(f"{'='*60}")

    # 1. 加载
    print(f"\n[1/4] 加载音乐库...")
    segment_tracks = load_music_directory(music_dir)
    if not segment_tracks:
        print("  没有找到任何音频文件！")
        sys.exit(1)

    # 2. 确定预设
    preset = build_custom_preset(args)
    if preset is None:
        preset = PRESET_QUICK if args.quick else PRESET_STANDARD

    print(f"\n[2/4] 编排段落...")
    for s in preset:
        track_names = [t.name for t in segment_tracks.get(s.label, [])]
        print(f"  {s.label}: {s.target_bpm}BPM × {s.duration/60:.0f}min  ← {track_names or '(无曲目)'}")

    if args.dry_run:
        print(f"\n  (dry-run 模式，不生成文件)")
        return

    # 3. 构建
    print(f"\n[3/4] 构建混音...")
    segments = build_arrangement(
        segment_tracks,
        preset=preset,
        fade_sec=args.fade / 1000.0,
    )

    # 4. 导出
    print(f"\n[4/4] 导出...")
    basename = os.path.splitext(os.path.basename(output))[0]
    wav_path = os.path.join(out_dir, basename + ".wav")
    save_project(segments, segment_tracks, out_dir, basename=basename)

    print(f"\n{'='*60}")
    print(f"  完成！")
    print(f"{'='*60}")
    print(f"  输出: {os.path.abspath(wav_path)}")
    total_sec = sum(s.duration for s in segments)
    print(f"  总时长: {int(total_sec // 60)}分{int(total_sec % 60)}秒")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="跑操音乐编排器 — 自动 BPM 检测、变速、混音、多段编排",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python app.py music/                         使用标准模板 (入场3'+跑操22'+放松3'+退场2')
  python app.py music/ --quick                 快速模板 (入场2'+跑操15'+放松2'+退场1')
  python app.py music/ --output 周一跑操.mp3    指定输出文件名
  python app.py music/ --entrance 130 2:30 --run 145 20:00    自定义参数
  python app.py music/ --list                  列出所有音乐及 BPM
  python app.py music/ --dry-run               预览，不生成文件
  python app.py music/ --fade 3000             交叉淡入淡出 3 秒
        """,
    )
    parser.add_argument("music_dir", nargs="?", default="music",
                        help="音乐文件目录")
    parser.add_argument("--output", "-o", default=None,
                        help="输出文件路径 (默认: running_arrangement.mp3)")
    parser.add_argument("--quick", action="store_true",
                        help="使用快速模板 (适合测试)")
    parser.add_argument("--list", action="store_true",
                        help="列出音乐文件及 BPM，不生成")
    parser.add_argument("--dry-run", action="store_true",
                        help="预览模式，不生成文件")
    parser.add_argument("--fade", type=int, default=2000,
                        help="交叉淡入淡出毫秒数 (默认: 2000)")
    parser.add_argument("--entrance", nargs=2, metavar=("BPM", "DURATION"),
                        help="入场段: BPM 时长 (如 120 3:00)")
    parser.add_argument("--run", nargs=2, metavar=("BPM", "DURATION"),
                        help="跑操段: BPM 时长 (如 140 22:00)")
    parser.add_argument("--cooldown", nargs=2, metavar=("BPM", "DURATION"),
                        help="放松段: BPM 时长 (如 100 3:00)")
    parser.add_argument("--exit-seg", nargs=2, metavar=("BPM", "DURATION"),
                        dest="exit_seg", help="退场段: BPM 时长 (如 120 2:00)")

    args = parser.parse_args()

    if args.list:
        cmd_list(args)
    else:
        cmd_generate(args)


if __name__ == "__main__":
    main()
