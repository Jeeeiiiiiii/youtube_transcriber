"""
YouTube Transcript Extractor
Extracts transcripts from YouTube videos using captions or Whisper AI.
"""

import argparse
import re
import sys
import os


def get_video_id(url: str) -> str:
    """Extract video ID from various YouTube URL formats."""
    patterns = [
        r'(?:v=)([a-zA-Z0-9_-]{11})',
        r'(?:youtu\.be/)([a-zA-Z0-9_-]{11})',
        r'(?:embed/)([a-zA-Z0-9_-]{11})',
        r'(?:shorts/)([a-zA-Z0-9_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    # Maybe it's just a raw video ID
    if re.match(r'^[a-zA-Z0-9_-]{11}$', url):
        return url
    raise ValueError(f"Could not extract video ID from: {url}")


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS or MM:SS format."""
    hours = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


def fetch_captions(video_id: str, language: str = 'en') -> list:
    """Fetch existing captions from YouTube.

    Prefers manually created captions (more accurate) over auto-generated ones.
    Uses youtube-transcript-api v1.x instance-based API.
    """
    from youtube_transcript_api import YouTubeTranscriptApi

    api = YouTubeTranscriptApi()
    transcript_list = api.list(video_id)

    # Prefer manual captions (more accurate)
    manual = None
    generated = None
    for t in transcript_list:
        if t.language_code == language or t.language_code.startswith(language):
            if not t.is_generated:
                manual = t
                break
            else:
                generated = t

    if manual:
        print(f"  Source: manual captions ({manual.language_code})")
        raw = manual.fetch()
    elif generated:
        print(f"  Source: auto-generated captions ({generated.language_code})")
        raw = generated.fetch()
    else:
        # Let the API pick the best match
        raw = api.fetch(video_id, languages=[language])
        print(f"  Source: captions ({language})")

    # Convert FetchedTranscriptSnippet objects to dicts
    segments = []
    for entry in raw:
        segments.append({
            'text': entry.text,
            'start': entry.start,
            'duration': entry.duration,
        })
    return segments


def is_youtube_url(url: str) -> bool:
    """True if the URL/ID looks like YouTube (vs any other video page)."""
    if re.search(r'(youtube\.com|youtu\.be)', url, re.IGNORECASE):
        return True
    # A bare 11-char video ID with no scheme/host is treated as YouTube.
    return bool(re.match(r'^[A-Za-z0-9_-]{11}$', url))


def ffmpeg_location():
    """Path to an ffmpeg binary: system ffmpeg if on PATH, else imageio-ffmpeg's."""
    import shutil
    if shutil.which('ffmpeg'):
        return None  # yt-dlp finds it on its own
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


def download_audio(url: str, dest_dir: str, referer: str = None) -> str:
    """Download a URL's audio with yt-dlp into dest_dir and return its path.

    Works for YouTube and any site yt-dlp supports (e.g. Bunny Stream embeds,
    which need the embedding site as --referer). Audio-only formats are
    preferred; for HLS sources that only offer muxed streams, the smallest
    stream is taken since only the audio matters for transcription.
    """
    try:
        import yt_dlp
    except ImportError:
        print("Error: yt-dlp is not installed.")
        print("Install with: pip install yt-dlp")
        sys.exit(1)

    print("  Downloading audio...")
    ydl_opts = {
        'format': 'bestaudio/worst[acodec!=none]/worst',
        'outtmpl': os.path.join(dest_dir, 'audio.%(ext)s'),
        'quiet': True,
        'no_warnings': True,
        'noprogress': True,
    }
    if referer:
        ydl_opts['http_headers'] = {'Referer': referer}
    ffmpeg = ffmpeg_location()
    if ffmpeg:
        ydl_opts['ffmpeg_location'] = ffmpeg

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    files = [f for f in os.listdir(dest_dir)
             if f.startswith('audio.') and not f.endswith('.part')]
    if not files:
        raise RuntimeError("Failed to download audio")
    return os.path.join(dest_dir, files[0])


# Loaded Whisper models, keyed by (backend, size), so batch runs load once.
_MODELS = {}


def transcribe_audio_file(media_path: str, model_size: str = 'base',
                          language: str = None) -> list:
    """Transcribe a local audio/video file.

    Uses faster-whisper when installed (several times faster on CPU), and
    falls back to OpenAI Whisper otherwise.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        WhisperModel = None

    if WhisperModel is not None:
        key = ('faster', model_size)
        if key not in _MODELS:
            print(f"  Loading faster-whisper model ({model_size})...")
            name = 'large-v3' if model_size == 'large' else model_size
            _MODELS[key] = WhisperModel(name, device='auto', compute_type='int8')
        print("  Transcribing (this may take a while)...")
        # vad_filter is left off: its onnxruntime VAD segfaults on some
        # Windows machines.
        raw, _info = _MODELS[key].transcribe(media_path, language=language)
        return [{'text': s.text.strip(), 'start': s.start,
                 'duration': s.end - s.start} for s in raw]

    try:
        import whisper
    except ImportError:
        print("Error: no Whisper backend is installed.")
        print("Install with: pip install faster-whisper  (or openai-whisper + ffmpeg)")
        sys.exit(1)

    key = ('openai', model_size)
    if key not in _MODELS:
        print(f"  Loading Whisper model ({model_size})...")
        _MODELS[key] = whisper.load_model(model_size)
    print("  Transcribing (this may take a while)...")
    result = _MODELS[key].transcribe(media_path, language=language)
    return [{'text': s['text'].strip(), 'start': s['start'],
             'duration': s['end'] - s['start']} for s in result['segments']]


def transcribe_with_whisper(url: str, model_size: str = 'base',
                            referer: str = None, language: str = None) -> list:
    """Download a URL's audio and transcribe it with Whisper."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = download_audio(url, tmpdir, referer=referer)
        return transcribe_audio_file(audio_path, model_size, language)


def format_transcript(segments: list, timestamps: bool = True) -> str:
    """Format transcript segments into readable text."""
    lines = []
    for seg in segments:
        text = seg.get('text', str(seg)) if isinstance(seg, dict) else str(seg)
        if timestamps and isinstance(seg, dict) and 'start' in seg:
            ts = format_timestamp(seg['start'])
            lines.append(f"[{ts}] {text}")
        else:
            lines.append(text)
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Extract transcripts from YouTube videos'
    )
    parser.add_argument('url', help='YouTube video URL/ID, or any video URL '
                                    'yt-dlp supports (transcribed with Whisper)')
    parser.add_argument(
        '-m', '--method',
        choices=['auto', 'captions', 'whisper'],
        default='auto',
        help='auto: try captions first, fall back to whisper. '
             'captions: YouTube captions only. '
             'whisper: AI transcription (default: auto)'
    )
    parser.add_argument(
        '-l', '--language', default='en',
        help='Language code (default: en)'
    )
    parser.add_argument(
        '--model',
        default='base',
        choices=['tiny', 'base', 'small', 'medium', 'large'],
        help='Whisper model size - larger = more accurate but slower (default: base)'
    )
    parser.add_argument(
        '--no-timestamps', action='store_true',
        help='Output transcript without timestamps'
    )
    parser.add_argument(
        '-o', '--output',
        help='Save transcript to a file instead of printing'
    )
    parser.add_argument(
        '--referer',
        help='Referer header for the download, e.g. the site a video is '
             'embedded on (needed for Bunny Stream embeds)'
    )

    args = parser.parse_args()

    youtube = is_youtube_url(args.url)
    segments = None

    if youtube:
        try:
            video_id = get_video_id(args.url)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
        print(f"Video ID: {video_id}")
    elif args.method == 'captions':
        print("Error: captions are only available for YouTube videos")
        sys.exit(1)

    # Try captions first (unless whisper is explicitly requested)
    if youtube and args.method in ('auto', 'captions'):
        try:
            print("Fetching YouTube captions...")
            segments = fetch_captions(video_id, args.language)
            print(f"  Got {len(segments)} segments")
        except Exception as e:
            if args.method == 'captions':
                print(f"Error: Could not fetch captions - {e}")
                sys.exit(1)
            print(f"  No captions available, falling back to Whisper...")

    # Use Whisper if captions failed or whisper was requested
    if segments is None or args.method == 'whisper':
        print(f"Transcribing with Whisper ({args.model} model)...")
        try:
            segments = transcribe_with_whisper(args.url, args.model,
                                               referer=args.referer,
                                               language=args.language)
            print(f"  Transcribed {len(segments)} segments")
        except Exception as e:
            print(f"Error during transcription: {e}")
            sys.exit(1)

    # Format and output
    transcript = format_transcript(segments, timestamps=not args.no_timestamps)

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(transcript)
        print(f"\nTranscript saved to: {args.output}")
    else:
        print(f"\n{'='*60}")
        print(transcript)
        print(f"{'='*60}")


if __name__ == '__main__':
    main()
