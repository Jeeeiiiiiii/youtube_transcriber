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


def transcribe_with_whisper(url: str, model_size: str = 'base') -> list:
    """Download audio and transcribe with OpenAI Whisper."""
    try:
        import whisper
    except ImportError:
        print("Error: Whisper is not installed.")
        print("Install with: pip install openai-whisper")
        print("You also need ffmpeg installed on your system.")
        sys.exit(1)

    try:
        import yt_dlp
    except ImportError:
        print("Error: yt-dlp is not installed.")
        print("Install with: pip install yt-dlp")
        sys.exit(1)

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        print("  Downloading audio...")
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(tmpdir, 'audio.%(ext)s'),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'quiet': True,
            'no_warnings': True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        audio_files = [f for f in os.listdir(tmpdir)
                       if f.endswith(('.mp3', '.m4a', '.wav', '.webm', '.opus'))]
        if not audio_files:
            raise RuntimeError("Failed to download audio")
        audio_path = os.path.join(tmpdir, audio_files[0])

        print(f"  Loading Whisper model ({model_size})...")
        model = whisper.load_model(model_size)

        print("  Transcribing (this may take a while)...")
        result = model.transcribe(audio_path)

    segments = []
    for seg in result['segments']:
        segments.append({
            'text': seg['text'].strip(),
            'start': seg['start'],
            'duration': seg['end'] - seg['start'],
        })

    return segments


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
    parser.add_argument('url', help='YouTube video URL or video ID')
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

    args = parser.parse_args()

    try:
        video_id = get_video_id(args.url)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"Video ID: {video_id}")

    segments = None

    # Try captions first (unless whisper is explicitly requested)
    if args.method in ('auto', 'captions'):
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
            segments = transcribe_with_whisper(args.url, args.model)
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
