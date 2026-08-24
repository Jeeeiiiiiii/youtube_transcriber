"""
Video Captioner
Transcribes a video (YouTube captions or Whisper AI) and burns the
transcription onto the video itself as captions, producing a new video file.

Supports YouTube URLs and Google Drive share links (and most direct video URLs).

Reuses the transcription logic from transcriber.py.
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile

# Reuse the proven helpers from the existing transcriber.
from transcriber import get_video_id, fetch_captions


def is_youtube_url(url: str) -> bool:
    """True if the URL/ID looks like YouTube (vs Drive or a direct link)."""
    if re.search(r'(youtube\.com|youtu\.be)', url, re.IGNORECASE):
        return True
    # A bare 11-char video ID with no scheme/host is treated as YouTube.
    return bool(re.match(r'^[A-Za-z0-9_-]{11}$', url))


def is_drive_url(url: str) -> bool:
    """True if the URL is a Google Drive link."""
    return bool(re.search(r'drive\.google\.com', url, re.IGNORECASE))


def is_drive_folder_url(url: str) -> bool:
    """True if the URL points to a Google Drive *folder* (not a single file)."""
    return bool(re.search(r'drive\.google\.com/drive/(?:u/\d+/)?folders/',
                          url, re.IGNORECASE))


# Container extensions we treat as video files when scanning a folder.
VIDEO_EXTS = ('.mp4', '.mov', '.mkv', '.avi', '.webm', '.m4v', '.flv',
              '.wmv', '.mpg', '.mpeg', '.ts', '.3gp')


def drive_file_id(url: str) -> str:
    """Extract the file ID from common Google Drive URL formats."""
    patterns = [
        r'/file/d/([A-Za-z0-9_-]+)',      # .../file/d/<id>/view
        r'[?&]id=([A-Za-z0-9_-]+)',        # ...?id=<id>  (open?id=, uc?id=)
    ]
    for pattern in patterns:
        m = re.search(pattern, url)
        if m:
            return m.group(1)
    raise ValueError(f"Could not extract a Drive file ID from: {url}")


def srt_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format: HH:MM:SS,mmm."""
    if seconds < 0:
        seconds = 0
    millis = int(round(seconds * 1000))
    hours, millis = divmod(millis, 3600_000)
    mins, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02d}:{mins:02d}:{secs:02d},{millis:03d}"


def segments_to_srt(segments: list) -> str:
    """Build an SRT subtitle document from transcript segments.

    Each segment is a dict with 'text', 'start', and 'duration'. To avoid
    captions overlapping, an entry's end is clamped to the next entry's start.
    """
    lines = []
    index = 0
    for i, seg in enumerate(segments):
        start = float(seg.get('start', 0.0))
        end = start + float(seg.get('duration', 0.0))

        # Clamp end so it does not run into the next subtitle.
        if i + 1 < len(segments):
            next_start = float(segments[i + 1].get('start', end))
            if end > next_start:
                end = next_start
        # Guarantee a minimum visible duration.
        if end <= start:
            end = start + 0.8

        text = str(seg.get('text', '')).strip()
        if not text:
            continue

        index += 1
        lines.append(str(index))
        lines.append(f"{srt_timestamp(start)} --> {srt_timestamp(end)}")
        lines.append(text)
        lines.append("")  # blank line between entries

    return "\n".join(lines) + "\n"


def _find_downloaded(dest_dir: str) -> str:
    """Return the path to the downloaded 'video.*' file in dest_dir."""
    candidates = [f for f in os.listdir(dest_dir)
                  if f.startswith("video.") and not f.endswith(".part")]
    if not candidates:
        raise RuntimeError("Failed to download video")
    # Prefer mp4/mkv container if multiple files lingered.
    candidates.sort(key=lambda f: (not f.endswith(('.mp4', '.mkv')), f))
    return os.path.join(dest_dir, candidates[0])


def download_with_ytdlp(url: str, dest_dir: str, youtube: bool) -> str:
    """Download a video with yt-dlp to dest_dir, return its path.

    For YouTube we ask for a merged <=1080p mp4. For Drive/other sources the
    file is usually a single muxed stream, so we just take the best available.
    """
    try:
        import yt_dlp
    except ImportError:
        print("Error: yt-dlp is not installed. Install with: pip install yt-dlp")
        sys.exit(1)

    print("  Downloading video...")
    out_template = os.path.join(dest_dir, "video.%(ext)s")
    if youtube:
        fmt = ('bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/'
               'best[height<=1080][ext=mp4]/best')
    else:
        # Prefer a stream that genuinely HAS audio (needed for transcription).
        # TikTok serves its top resolution as h265 video-only AND mislabels it
        # as having audio, so 'acodec' metadata can't be trusted. We instead
        # prefer TikTok's muxed 'download' format, then any h264/avc stream
        # (which really do carry audio), before falling back.
        fmt = ('download/best[vcodec*=h264]/best[vcodec*=avc]/'
               'best[acodec!=none]/best')
    ydl_opts = {
        'format': fmt,
        'merge_output_format': 'mp4',
        'outtmpl': out_template,
        'quiet': True,
        'no_warnings': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return _find_downloaded(dest_dir)


def drive_download_file(file_id: str, out_path: str) -> None:
    """Download a single public Drive file by ID using a streaming HTTP request.

    Uses the drive.usercontent endpoint with confirm=t, which serves the raw
    bytes directly (handling the large-file 'virus scan' page). This is more
    reliable than gdown's per-file logic. The file must be shared as
    'Anyone with the link'.
    """
    import requests

    session = requests.Session()
    base = 'https://drive.usercontent.google.com/download'
    params = {'id': file_id, 'export': 'download', 'confirm': 't'}
    r = session.get(base, params=params, stream=True)

    # If Drive still returns the HTML confirmation page, resubmit its form.
    if 'text/html' in r.headers.get('Content-Type', ''):
        body = r.text
        action = re.search(r'<form[^>]+action="([^"]+)"', body)
        inputs = dict(re.findall(
            r'<input[^>]*name="([^"]+)"[^>]*value="([^"]*)"', body))
        if action:
            r = session.get(action.group(1), params=inputs, stream=True)

    r.raise_for_status()
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024 * 256):
            if chunk:
                f.write(chunk)


def download_from_drive(url: str, dest_dir: str) -> str:
    """Download a single Google Drive file. Robust streaming first, then
    falls back to yt-dlp's Google Drive extractor.

    The file must be shared as 'Anyone with the link'.
    """
    file_id = drive_file_id(url)
    out_path = os.path.join(dest_dir, "video.mp4")

    try:
        print("  Downloading from Google Drive...")
        drive_download_file(file_id, out_path)
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            return out_path
    except Exception as e:
        print(f"  Direct download failed ({e}); trying yt-dlp...")

    return download_with_ytdlp(url, dest_dir, youtube=False)


def fetch_source_video(url: str, dest_dir: str) -> str:
    """Route a URL to the right downloader based on its source."""
    if is_drive_url(url):
        return download_from_drive(url, dest_dir)
    return download_with_ytdlp(url, dest_dir, youtube=is_youtube_url(url))


def list_drive_folder(url: str) -> list:
    """List a shared Drive folder's files as (relative_path, file_id) tuples.

    Recurses into subfolders. Requires gdown for the folder enumeration.
    """
    try:
        import gdown
    except ImportError:
        print("Error: listing Drive folders needs gdown. Install with: pip install gdown")
        sys.exit(1)

    items = gdown.download_folder(url=url, skip_download=True, quiet=True,
                                  use_cookies=False)
    result = []
    for it in (items or []):
        rel = getattr(it, 'path', None) or getattr(it, 'local_path', None)
        fid = getattr(it, 'id', None)
        if rel and fid:
            result.append((rel, fid))
    return result


def list_drive_videos(url: str) -> list:
    """List just the video files in a shared Drive folder as (rel_path, id)."""
    print("  Listing Google Drive folder...")
    entries = list_drive_folder(url)
    video_entries = sorted((rel, fid) for rel, fid in entries
                           if rel.lower().endswith(VIDEO_EXTS))
    skipped = len(entries) - len(video_entries)
    print(f"  Folder has {len(entries)} file(s); {len(video_entries)} video(s)"
          + (f", {skipped} non-video skipped." if skipped else "."))
    return video_entries


def transcribe_file_with_whisper(media_path: str, model_size: str = 'base') -> list:
    """Transcribe a local media file with Whisper (no extra download)."""
    try:
        import whisper
    except ImportError:
        print("Error: Whisper is not installed. Install with: pip install openai-whisper")
        sys.exit(1)

    print(f"  Loading Whisper model ({model_size})...")
    model = whisper.load_model(model_size)
    print("  Transcribing (this may take a while)...")
    result = model.transcribe(media_path)

    segments = []
    for seg in result['segments']:
        segments.append({
            'text': seg['text'].strip(),
            'start': seg['start'],
            'duration': seg['end'] - seg['start'],
        })
    return segments


def transcribe_file_with_whisper(media_path: str, model_size: str = 'base') -> list:
    """Transcribe a local media file with Whisper (no extra download)."""
    try:
        import whisper
    except ImportError:
        print("Error: Whisper is not installed. Install with: pip install openai-whisper")
        sys.exit(1)

    print(f"  Loading Whisper model ({model_size})...")
    model = whisper.load_model(model_size)
    print("  Transcribing (this may take a while)...")
    result = model.transcribe(media_path)

    segments = []
    for seg in result['segments']:
        segments.append({
            'text': seg['text'].strip(),
            'start': seg['start'],
            'duration': seg['end'] - seg['start'],
        })
    return segments


def burn_subtitles(video_path: str, srt_path: str, output_path: str,
                   font_size: int = 24, soft: bool = False) -> None:
    """Render the SRT onto the video with ffmpeg.

    If soft=True, embed the subtitles as a toggleable track instead of
    burning them in (much faster, no re-encode).
    """
    if soft:
        print("  Embedding soft subtitle track (no re-encode)...")
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", srt_path,
            "-c", "copy",
            "-c:s", "mov_text",
            "-metadata:s:s:0", "language=eng",
            output_path,
        ]
        subprocess.run(cmd, check=True)
        return

    # Burn-in. The subtitles filter is fussy about Windows paths (the drive
    # colon and backslashes break filter parsing), so run ffmpeg from the
    # folder containing the SRT and reference it by bare filename.
    work_dir = os.path.dirname(os.path.abspath(srt_path))
    srt_name = os.path.basename(srt_path)
    style = (f"FontName=Arial,FontSize={font_size},PrimaryColour=&H00FFFFFF,"
             f"OutlineColour=&H00000000,BorderStyle=1,Outline=2,Shadow=1,"
             f"MarginV=30")
    vf = f"subtitles={srt_name}:force_style='{style}'"

    print("  Burning captions into video (re-encoding)...")
    cmd = [
        "ffmpeg", "-y",
        "-i", os.path.abspath(video_path),
        "-vf", vf,
        "-c:a", "copy",
        os.path.abspath(output_path),
    ]
    subprocess.run(cmd, check=True, cwd=work_dir)


def upload_with_rclone(local_path: str, dest: str, rclone: str = "rclone") -> None:
    """Mirror local_path to an rclone remote destination (e.g.
    'gdrive:Captioned Course'), preserving the folder structure.

    Requires rclone to be installed and a configured remote (run
    'rclone config' once to sign in to Google Drive).
    """
    if os.path.isdir(local_path):
        # Copy the directory's contents into dest, preserving subfolders.
        cmd = [rclone, "copy", local_path, dest, "--progress", "--transfers", "2"]
    else:
        # Single file -> copy into the dest folder.
        cmd = [rclone, "copyto", local_path,
               dest.rstrip("/") + "/" + os.path.basename(local_path),
               "--progress"]
    print(f"  Uploading to {dest} via rclone...")
    subprocess.run(cmd, check=True)


def render_captions(video_path: str, output_path: str, segments: list,
                    args) -> None:
    """Build an SRT from segments and burn/embed it onto video_path."""
    srt_text = segments_to_srt(segments)
    srt_path = os.path.splitext(video_path)[0] + ".captions.srt"
    with open(srt_path, 'w', encoding='utf-8') as f:
        f.write(srt_text)

    if args.keep_srt:
        kept = os.path.splitext(os.path.abspath(output_path))[0] + ".srt"
        with open(kept, 'w', encoding='utf-8') as f:
            f.write(srt_text)
        print(f"  Saved subtitles to: {kept}")

    burn_subtitles(video_path, srt_path, output_path,
                   font_size=args.font_size, soft=args.soft)


def process_folder(args, tmpdir: str) -> None:
    """Caption every video in a Drive folder, one at a time.

    Each video is downloaded, captioned, and its temp source deleted before
    moving on -- so disk usage stays low and results appear incrementally.
    Re-running skips videos that are already captioned (resume support).
    """
    video_entries = list_drive_videos(args.url)
    if not video_entries:
        print("No video files found in that Drive folder.")
        sys.exit(1)

    # Folder mode writes into a directory (args.output if the user gave a
    # non-default, extension-less name; otherwise 'captioned_videos').
    out_dir = args.output
    if out_dir == 'captioned.mp4' or os.path.splitext(out_dir)[1]:
        out_dir = 'captioned_videos'
    os.makedirs(out_dir, exist_ok=True)

    total = len(video_entries)
    print(f"\nCaptioning {total} video(s) into: {os.path.abspath(out_dir)}\n")
    failures = []
    skipped_done = 0
    for i, (rel, fid) in enumerate(video_entries, 1):
        # Preserve the folder's subdirectory layout in the output so that
        # same-named videos in different subfolders don't collide.
        rel_noext = os.path.splitext(rel)[0]
        output_path = os.path.join(out_dir, rel_noext + "_captioned.mp4")
        os.makedirs(os.path.dirname(output_path) or out_dir, exist_ok=True)
        print(f"[{i}/{total}] {rel}", flush=True)

        # Resume: skip videos already captioned in a previous run.
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print("  already captioned; skipping.\n", flush=True)
            skipped_done += 1
            continue

        local_src = os.path.join(tmpdir, "src" + os.path.splitext(rel)[1])
        try:
            print("  Downloading from Drive...", flush=True)
            drive_download_file(fid, local_src)
            print(f"  Transcribing with Whisper ({args.model} model)...", flush=True)
            segments = transcribe_file_with_whisper(local_src, args.model)
            print(f"  Transcribed {len(segments)} segments", flush=True)
            if not segments:
                raise RuntimeError("no transcript segments produced")
            render_captions(local_src, output_path, segments, args)
            print(f"  -> {output_path}\n", flush=True)
        except subprocess.CalledProcessError as e:
            print(f"  ffmpeg failed ({e.returncode}); skipping this file.\n", flush=True)
            failures.append(rel)
        except Exception as e:
            print(f"  Error: {e}; skipping this file.\n", flush=True)
            failures.append(rel)
        finally:
            # Free the temp source video before the next download.
            if os.path.exists(local_src):
                try:
                    os.remove(local_src)
                except OSError:
                    pass

    done = total - len(failures) - skipped_done
    print(f"Done. Captioned {done} new + {skipped_done} already-done = "
          f"{done + skipped_done}/{total} into {os.path.abspath(out_dir)}")
    if failures:
        print("Failed:")
        for f in failures:
            print(f"  - {f}")

    # Optional: mirror the captioned folder up to a Drive remote.
    if getattr(args, 'upload', None) and (done + skipped_done):
        try:
            upload_with_rclone(out_dir, args.upload, args.rclone_path)
            print(f"Uploaded to {args.upload}")
        except subprocess.CalledProcessError as e:
            print(f"Upload failed (rclone exit {e.returncode}). "
                  f"Have you run 'rclone config' to set up the remote?")


def main():
    parser = argparse.ArgumentParser(
        description='Transcribe a video and burn the captions onto it. '
                    'Supports YouTube, Google Drive, and direct video URLs.'
    )
    parser.add_argument('url', help='YouTube URL/ID, Google Drive share link, '
                                    'or direct video URL')
    parser.add_argument(
        '-m', '--method',
        choices=['auto', 'captions', 'whisper'],
        default='whisper',
        help='auto: try captions first, fall back to whisper. '
             'captions: YouTube captions only. '
             'whisper: AI transcription (default: whisper)'
    )
    parser.add_argument(
        '-l', '--language', default='en',
        help='Language code for captions lookup (default: en)'
    )
    parser.add_argument(
        '--model', default='base',
        choices=['tiny', 'base', 'small', 'medium', 'large'],
        help='Whisper model size (default: base)'
    )
    parser.add_argument(
        '-o', '--output', default='captioned.mp4',
        help='Output video file path (default: captioned.mp4)'
    )
    parser.add_argument(
        '--font-size', type=int, default=24,
        help='Caption font size (default: 24)'
    )
    parser.add_argument(
        '--soft', action='store_true',
        help='Embed a toggleable subtitle track instead of burning captions in '
             '(fast, no re-encode)'
    )
    parser.add_argument(
        '--keep-srt', action='store_true',
        help='Also save the generated .srt next to the output video'
    )
    parser.add_argument(
        '--upload', default=None,
        help='After captioning, upload the result to this rclone remote, '
             'preserving folder structure (e.g. "gdrive:Captioned Course"). '
             'Requires rclone configured via "rclone config".'
    )
    parser.add_argument(
        '--rclone-path', default='rclone',
        help='Path to the rclone executable (default: rclone on PATH)'
    )

    args = parser.parse_args()

    # Folder mode: download the whole Drive folder and caption every video.
    if is_drive_folder_url(args.url):
        print("Source: Google Drive folder")
        with tempfile.TemporaryDirectory() as tmpdir:
            process_folder(args, tmpdir)
        return

    youtube = is_youtube_url(args.url)

    # YouTube captions only make sense for YouTube. For Drive/direct links,
    # force Whisper regardless of the requested method.
    if not youtube and args.method != 'whisper':
        print("Non-YouTube source detected -> using Whisper (no YouTube captions).")
        args.method = 'whisper'

    video_id = None
    if youtube:
        try:
            video_id = get_video_id(args.url)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
        print(f"Video ID: {video_id}")
    elif is_drive_url(args.url):
        print("Source: Google Drive")
    else:
        print("Source: direct/other URL")

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Download the actual video.
        try:
            video_path = fetch_source_video(args.url, tmpdir)
        except Exception as e:
            print(f"Error downloading video: {e}")
            sys.exit(1)

        # 2. Get timed transcript segments.
        segments = None
        if youtube and args.method in ('auto', 'captions'):
            try:
                print("Fetching YouTube captions...")
                segments = fetch_captions(video_id, args.language)
                print(f"  Got {len(segments)} segments")
            except Exception as e:
                if args.method == 'captions':
                    print(f"Error: Could not fetch captions - {e}")
                    sys.exit(1)
                print("  No captions available, falling back to Whisper...")

        if segments is None or args.method == 'whisper':
            print(f"Transcribing with Whisper ({args.model} model)...")
            try:
                segments = transcribe_file_with_whisper(video_path, args.model)
                print(f"  Transcribed {len(segments)} segments")
            except Exception as e:
                print(f"Error during transcription: {e}")
                sys.exit(1)

        if not segments:
            print("Error: no transcript segments produced.")
            sys.exit(1)

        # 3. Build the SRT and burn (or embed) onto the video.
        try:
            render_captions(video_path, args.output, segments, args)
        except subprocess.CalledProcessError as e:
            print(f"Error: ffmpeg failed ({e.returncode}). Is ffmpeg installed and on PATH?")
            sys.exit(1)

    print(f"\nDone. Captioned video saved to: {os.path.abspath(args.output)}")

    if args.upload:
        try:
            upload_with_rclone(args.output, args.upload, args.rclone_path)
            print(f"Uploaded to {args.upload}")
        except subprocess.CalledProcessError as e:
            print(f"Upload failed (rclone exit {e.returncode}). "
                  f"Have you run 'rclone config' to set up the remote?")


if __name__ == '__main__':
    main()
