"""
Batch Transcriber
Transcribes every video listed in a manifest TSV, saving one transcript per
episode. Re-running skips episodes that already have a transcript, so an
interrupted run can simply be started again.

Manifest format (tab-separated, with a header row):
    series  season  episode  title  url

The url can be a Bunny Stream embed (https://iframe.mediadelivery.net/embed/
<library>/<guid>), a Google Drive file link, YouTube, or anything yt-dlp
supports.

Building a manifest from a site whose watch pages define a `VIDEOS` array
(e.g. nmagneticr.com): open any watch page while logged in, and run this in
the browser console, then paste the copied text into manifests/<name>.tsv:

    copy(['series\\tseason\\tepisode\\ttitle\\turl'].concat(VIDEOS.flatMap(v =>
      (v.seasons || [{season: '', episodes: v.episodes || []}]).flatMap(s =>
        (s.episodes || []).map((e, i) => [v.title, s.season || '', i + 1,
          e.title, e.url].join('\\t'))))).join('\\n'))

Reuses the download/transcription logic from transcriber.py and captioner.py.
"""

import argparse
import csv
import os
import re
import sys
import tempfile

from transcriber import (download_audio, format_transcript, is_youtube_url,
                         fetch_captions, get_video_id, transcribe_audio_file)


def safe_name(text: str) -> str:
    """Make text safe to use as a Windows/Unix file or folder name."""
    text = text.replace('|', '-')
    text = re.sub(r'[<>:"/\\?*\x00-\x1f]', '', text)
    text = re.sub(r'\s+', ' ', text).strip(' .')
    return text[:120] or 'untitled'


def read_manifest(path: str) -> list:
    """Read manifest rows as dicts, skipping blank lines."""
    with open(path, encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        return [row for row in reader if row.get('url')]


def output_path(out_dir: str, row: dict) -> str:
    """Where an episode's transcript goes: <out>/<series>/<season>/NN - title.txt"""
    parts = [out_dir, safe_name(row['series'])]
    if row.get('season'):
        parts.append(safe_name(row['season']))
    episode = int(row.get('episode') or 0)
    parts.append(f"{episode:02d} - {safe_name(row['title'])}.txt")
    return os.path.join(*parts)


def transcribe_row(row: dict, args) -> list:
    """Download one manifest entry's media and transcribe it."""
    url = row['url']

    if is_youtube_url(url):
        try:
            return fetch_captions(get_video_id(url), args.language)
        except Exception:
            print("  No captions available, falling back to Whisper...")

    with tempfile.TemporaryDirectory() as tmpdir:
        if re.search(r'drive\.google\.com', url, re.IGNORECASE):
            from captioner import download_from_drive
            media_path = download_from_drive(url, tmpdir)
        else:
            media_path = download_audio(url, tmpdir, referer=args.referer)
        return transcribe_audio_file(media_path, args.model, args.language)


def main():
    parser = argparse.ArgumentParser(
        description='Transcribe every video listed in a manifest TSV'
    )
    parser.add_argument('manifest', help='Path to the manifest .tsv file')
    parser.add_argument(
        '-o', '--out-dir',
        help='Folder for transcripts (default: transcripts/<manifest name>)'
    )
    parser.add_argument(
        '--referer',
        help='Referer header for downloads, i.e. the site the videos are '
             'embedded on (needed for Bunny Stream, e.g. https://nmagneticr.com/)'
    )
    parser.add_argument(
        '--model',
        default='base',
        choices=['tiny', 'base', 'small', 'medium', 'large'],
        help='Whisper model size - larger = more accurate but slower (default: base)'
    )
    parser.add_argument(
        '-l', '--language', default='en',
        help='Language code (default: en)'
    )
    parser.add_argument(
        '--only',
        help='Only process rows whose series/season/title contains this text '
             '(case-insensitive)'
    )
    parser.add_argument(
        '--limit', type=int,
        help='Stop after transcribing this many episodes'
    )
    parser.add_argument(
        '--no-timestamps', action='store_true',
        help='Write transcripts without timestamps'
    )

    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    if args.only:
        needle = args.only.lower()
        rows = [r for r in rows
                if needle in f"{r['series']} {r['season']} {r['title']}".lower()]

    out_dir = args.out_dir or os.path.join(
        'transcripts', os.path.splitext(os.path.basename(args.manifest))[0])

    print(f"{len(rows)} episode(s) in manifest; writing to {out_dir}")

    done = skipped = 0
    failed = []
    for i, row in enumerate(rows, 1):
        path = output_path(out_dir, row)
        label = f"[{i}/{len(rows)}] {row['series']} / {row['season']} / {row['title']}"
        if os.path.exists(path):
            skipped += 1
            continue
        if args.limit is not None and done >= args.limit:
            break

        print(f"\n{label}")
        try:
            segments = transcribe_row(row, args)
        except Exception as e:
            print(f"  FAILED: {e}")
            failed.append((label, str(e)))
            continue

        transcript = format_transcript(segments,
                                       timestamps=not args.no_timestamps)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Write to a temp name first so a crash never leaves a partial file
        # that a re-run would mistake for a finished transcript.
        with open(path + '.part', 'w', encoding='utf-8') as f:
            f.write(f"{row['title']}\n{row['url']}\n\n{transcript}\n")
        os.replace(path + '.part', path)
        done += 1
        print(f"  Saved {len(segments)} segments -> {path}")

    print(f"\nDone: {done} transcribed, {skipped} already existed, "
          f"{len(failed)} failed.")
    for label, err in failed:
        print(f"  {label}: {err}")
    if failed:
        sys.exit(1)


if __name__ == '__main__':
    main()
