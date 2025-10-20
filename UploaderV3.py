#!/usr/bin/env python3

import asyncio
import sys
import os
import time
import logging
import re
import json
import mimetypes
import psutil
import shlex
import argparse
from os import path as ospath
from time import time
from hashlib import md5
from contextlib import suppress
from functools import partial
from html import escape

# --- Pyrogram/Third-party Imports ---
try:
    from pyrogram import Client
    from pyrogram.errors import (
        BadRequest,
        FloodWait,
        RPCError,
    )
    from pyrogram.errors.exceptions.bad_request_400 import PhotoSaveFileInvalid

    try:
        from pyrogram.errors import FloodPremiumWait
    except ImportError:
        FloodPremiumWait = FloodWait
    from pyrogram.types import (
        InputMediaDocument,
        InputMediaPhoto,
        InputMediaVideo,
    )
except ImportError:
    print(
        "Error: Missing required libraries. Please install them: "
        "pip install pyrofork tgcrypto tenacity pillow aiofiles psutil langcodes"
    )
    sys.exit(1)

try:
    # --- *** Pillow is now CRITICAL for the workaround ***---
    from PIL import Image, ImageOps
except ImportError:
    print("Error: Missing PIL (Pillow). Please install: pip install Pillow")
    sys.exit(1)

try:
    from tenacity import (
        RetryError,
        retry,
        retry_if_exception_type,
        stop_after_attempt,
        wait_exponential,
    )
except ImportError:
    print("Error: Missing tenacity. Please install: pip install tenacity")
    sys.exit(1)

try:
    import aiofiles
    from aiofiles.os import (
        path as aiopath,
        remove,
        rename,
        makedirs,
    )
    from aioshutil import rmtree
except ImportError:
    print("Error: Missing aiofiles/aioshutil. Please install: pip install aiofiles aioshutil")
    sys.exit(1)

try:
    from langcodes import Language
except ImportError:
    print("Error: Missing langcodes. Please install: pip install langcodes")
    sys.exit(1)


# --- Configuration ---
# PLEASE FILL THESE VALUES
API_ID = 1234567  # Your API ID from my.telegram.org
API_HASH = "your_api_hash"  # Your API Hash from my.telegram.org
BOT_TOKEN = "your_bot_token"  # Your Bot Token from BotFather

# --- Script-level Globals ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
LOGGER = logging.getLogger(__name__)

# Hardcoded config values from the original repo
DOWNLOAD_DIR = "downloads/"  # Used for storing thumbnails
THUMBNAIL_DIR = ospath.join(DOWNLOAD_DIR, "thumbnails") # Specific dir for thumbs
BinConfig = {"FFMPEG_NAME": "ffmpeg"}  # Path/name of ffmpeg executable
try:
    cpu_no = psutil.cpu_count(logical=False) or 1
except:
    cpu_no = 1
SIZE_UNITS = ["B", "KB", "MB", "GB", "TB", "PB"]
TG_THUMB_SIZE = (320, 320) # Max thumb dimension for Telegram


# =================================================================================
# --- Inlined Helper Functions ---
# (From bot/helper/ext_utils/status_utils.py)
# =================================================================================

def get_readable_file_size(size_in_bytes):
    if not size_in_bytes:
        return "0B"
    index = 0
    while size_in_bytes >= 1024 and index < len(SIZE_UNITS) - 1:
        size_in_bytes /= 1024
        index += 1
    return f"{size_in_bytes:.2f}{SIZE_UNITS[index]}"


def get_readable_time(seconds: int):
    periods = [("d", 86400), ("h", 3600), ("m", 60), ("s", 1)]
    result = ""
    for period_name, period_seconds in periods:
        if seconds >= period_seconds:
            period_value, seconds = divmod(seconds, period_seconds)
            result += f"{int(period_value)}{period_name}"
    return result


def time_to_seconds(time_duration):
    try:
        parts = time_duration.split(":")
        if len(parts) == 3:
            hours, minutes, seconds = map(float, parts)
        elif len(parts) == 2:
            hours = 0
            minutes, seconds = map(float, parts)
        elif len(parts) == 1:
            hours = 0
            minutes = 0
            seconds = float(parts[0])
        else:
            return 0
        return hours * 3600 + minutes * 60 + seconds
    except Exception:
        return 0


# =================================================================================
# --- Inlined Helper Functions ---
# (From bot/helper/ext_utils/bot_utils.py)
# =================================================================================

async def cmd_exec(cmd, shell=False):
    """
    Executes a shell command asynchronously.
    """
    if shell:
        if isinstance(cmd, list):
             cmd = " ".join(cmd)

    if shell:
        process = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
    else:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

    stdout, stderr = await process.communicate()
    stdout = stdout.decode().strip()
    stderr = stderr.decode().strip()
    return stdout, stderr, process.returncode


async def sync_to_async(func, *args, **kwargs):
    """
    Runs a synchronous function in an asynchronous executor.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, partial(func, *args, **kwargs)
    )


# =================================================================================
# --- Inlined Helper Functions ---
# (From bot/helper/ext_utils/files_utils.py)
# =================================================================================

def get_mime_type(file_path: str):
    """
    Gets the MIME type of a file.
    """
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        return "application/octet-stream"
    return mime_type


def is_archive(path: str):
    """
    Checks if a file is an archive based on MIME type.
    """
    mime_type = get_mime_type(path)
    return mime_type in [
        "application/zip",
        "application/x-zip-compressed",
        "application/x-7z-compressed",
        "application/x-rar-compressed",
        "application/x-tar",
        "application/x-bzip2",
        "application/x-gzip",
        "application/x-lzip",
        "application/x-lzma",
        "application/x-xz",
    ]


def is_archive_split(path: str):
    """
    Checks if a file is a split archive (basic check).
    """
    return re.search(r"\.part\d+\.rar$|\.r\d+$|\.z\d+$|\.00\d+$", path)


def get_base_name(file_path: str):
    """
    Gets the base name of a file, handling common archive extensions.
    """
    if file_path.endswith(".tar.bz2"):
        return file_path.rsplit(".tar.bz2", 1)[0]
    if file_path.endswith(".tar.gz"):
        return file_path.rsplit(".tar.gz", 1)[0]
    if file_path.endswith(".tar.xz"):
        return file_path.rsplit(".tar.xz", 1)[0]
    
    base_name, _ = ospath.splitext(file_path)
    if re.search(r"\.part\d+\.rar$|\.r\d+$", file_path):
        base_name, _ = ospath.splitext(base_name)
    elif re.search(r"\.z\d+$|\.00\d+$", file_path):
        base_name, _ = ospath.splitext(base_name)
    
    return base_name


# =================================================================================
# --- Inlined Helper Functions ---
# (From bot/helper/ext_utils/media_utils.py AND NEW)
# =================================================================================

async def generate_image_thumbnail(image_path):
    """
    Generates a JPEG thumbnail from an image file using Pillow.
    Returns path to the thumbnail or None on failure.
    """
    try:
        await makedirs(THUMBNAIL_DIR, exist_ok=True)
        thumb_path = ospath.join(THUMBNAIL_DIR, f"{time()}.jpg")

        def _create_thumb():
            with Image.open(image_path) as img:
                # Use ImageOps.fit to resize and crop to the target size
                # centering the crop and preserving aspect ratio as much as possible.
                # Convert to RGB to ensure saving as JPEG works.
                thumb_img = ImageOps.fit(img.convert("RGB"), TG_THUMB_SIZE, Image.Resampling.LANCZOS)
                thumb_img.save(thumb_path, "JPEG", quality=85) # Save as JPEG

        await sync_to_async(_create_thumb)

        if await aiopath.exists(thumb_path):
            return thumb_path
        else:
            LOGGER.error(f"Failed to save generated thumbnail for: {image_path}")
            return None
    except Exception as e:
        LOGGER.error(f"Error generating thumbnail for {image_path}: {e}", exc_info=True)
        return None

def get_md5_hash(up_path):
    """
    Calculates the MD5 hash of a file.
    """
    md5_hash = md5()
    with open(up_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            md5_hash.update(byte_block)
        return md5_hash.hexdigest()


async def get_media_info(path, extra_info=False):
    """
    Gets media information using ffprobe.
    Returns (duration, artist, title) or (duration, qual, lang, subs) if extra_info is True.
    """
    try:
        result = await cmd_exec(
            [
                "ffprobe",
                "-hide_banner",
                "-loglevel",
                "error",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                path,
            ]
        )
    except Exception as e:
        LOGGER.error(f"Get Media Info: {e}. Mostly File not found! - File: {path}")
        return (0, "", "", "") if extra_info else (0, None, None)

    if result[0] and result[2] == 0:
        try:
            ffresult = json.loads(result[0])
        except json.JSONDecodeError:
            LOGGER.error(f"Failed to parse ffprobe output: {result[0]}")
            return (0, "", "", "") if extra_info else (0, None, None)

        fields = ffresult.get("format")
        if fields is None:
            LOGGER.error(f"get_media_info: No 'format' field in ffprobe output: {result[0]}")
            return (0, "", "", "") if extra_info else (0, None, None)

        duration = round(float(fields.get("duration", 0)))
        
        if extra_info:
            lang, qual, stitles = "", "", ""
            if (streams := ffresult.get("streams")) and streams and streams[0].get(
                "codec_type"
            ) == "video":
                qual_height = int(streams[0].get("height", 0))
                qual_map = {
                    480: "480p", 540: "540p", 720: "720p", 
                    1080: "1080p", 2160: "2160p", 4320: "4320p", 8640: "8640p"
                }
                qual = next((v for k, v in qual_map.items() if qual_height <= k), f"{qual_height}p" if qual_height else "")

                for stream in streams:
                    if stream.get("codec_type") == "audio" and (
                        lc := stream.get("tags", {}).get("language")
                    ):
                        with suppress(Exception):
                            lc = Language.get(lc).display_name()
                        if lc not in lang:
                            lang += f"{lc}, "
                    if stream.get("codec_type") == "subtitle" and (
                        st := stream.get("tags", {}).get("language")
                    ):
                        with suppress(Exception):
                            st = Language.get(st).display_name()
                        if st not in stitles:
                            stitles += f"{st}, "
            return duration, qual, lang.rstrip(", "), stitles.rstrip(", ")
        
        tags = fields.get("tags", {})
        artist = tags.get("artist") or tags.get("ARTIST") or tags.get("Artist")
        title = tags.get("title") or tags.get("TITLE") or tags.get("Title")
        return duration, artist, title
        
    LOGGER.error(f"ffprobe command failed: {result[1]} (Code: {result[2]})")
    return (0, "", "", "") if extra_info else (0, None, None)


async def get_document_type(path):
    """
    Determines if a file is video, audio, or image using ffprobe and mime types.
    Returns (is_video, is_audio, is_image).
    """
    is_video, is_audio, is_image = False, False, False
    if (
        await sync_to_async(is_archive, path)
        or await sync_to_async(is_archive_split, path)
        or re.search(r".+(\.|_)(rar|7z|zip|bin)(\.0*\d+)?$", path, re.IGNORECASE)
    ):
        return is_video, is_audio, is_image
        
    mime_type = await sync_to_async(get_mime_type, path)
    if mime_type.startswith("image"):
        return False, False, True

    # Avoid running ffprobe on non-media types for performance
    if not (mime_type.startswith("video") or mime_type.startswith("audio")):
        return False, False, False

    try:
        result = await cmd_exec(
            [
                "ffprobe",
                "-hide_banner",
                "-loglevel",
                "error",
                "-print_format",
                "json",
                "-show_streams",
                path,
            ]
        )
    except Exception as e:
        LOGGER.error(f"Get Document Type: {e}. Mostly File not found! - File: {path}")
        # Rely on mime type if ffprobe fails
        if mime_type.startswith("audio"):
            return False, True, False
        if mime_type.startswith("video"):
            is_video = True
        return is_video, is_audio, is_image

    if result[0] and result[2] == 0:
        try:
            fields = json.loads(result[0]).get("streams")
        except json.JSONDecodeError:
            LOGGER.error(f"Failed to parse ffprobe output: {result[0]}")
            fields = None
            
        if fields is None:
            LOGGER.error(f"get_document_type: No 'streams' field in ffprobe output: {result[0]}")
            # Fallback to mime type if ffprobe output is unusable
            if mime_type.startswith("video"): is_video = True
            if mime_type.startswith("audio"): is_audio = True
            return is_video, is_audio, is_image
            
        for stream in fields:
            if stream.get("codec_type") == "video":
                codec_name = stream.get("codec_name", "").lower()
                if codec_name not in {"mjpeg", "png", "bmp", "gif"}: # Filter out image codecs
                    is_video = True
            elif stream.get("codec_type") == "audio":
                is_audio = True
    else:
        # ffprobe command failed, rely on mime type
        if mime_type.startswith("video"):
            is_video = True
        elif mime_type.startswith("audio"):
            is_audio = True
            
    return is_video, is_audio, is_image


async def take_ss(video_file, ss_nb) -> bool | str:
    """
    Takes a number of screenshots from a video file.
    Returns the directory path where screenshots are saved, or False on failure.
    """
    duration = (await get_media_info(video_file))[0]
    if duration == 0:
        LOGGER.error("take_ss: Can't get the duration of video")
        return False
        
    dirpath, name = ospath.split(video_file)
    name, _ = ospath.splitext(name)
    ss_dirpath = ospath.join(dirpath, f"{name}_mltbss") # Store SS in a subfolder
    await makedirs(ss_dirpath, exist_ok=True)
    
    interval = duration // (ss_nb + 1)
    cap_time = interval
    cmds = []
    for i in range(ss_nb):
        output = ospath.join(ss_dirpath, f"SS.{name}_{i:02}.png")
        cmd = [
            BinConfig["FFMPEG_NAME"],
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{cap_time}",
            "-i",
            video_file,
            "-q:v",
            "1",
            "-frames:v",
            "1",
            "-threads",
            f"{max(1, cpu_no // 2)}",
            output,
        ]
        cap_time += interval
        cmds.append(cmd_exec(cmd))
        
    try:
        results = await asyncio.wait_for(asyncio.gather(*cmds), timeout=60)
        if any(res[2] != 0 for res in results):
            LOGGER.error(
                f"Error while creating screenshots. stderr: {next((res[1] for res in results if res[2] != 0), 'Unknown error')}"
            )
            await rmtree(ss_dirpath, ignore_errors=True)
            return False
    except asyncio.TimeoutError:
        LOGGER.error(f"Error creating screenshots: Timeout.")
        await rmtree(ss_dirpath, ignore_errors=True)
        return False
        
    return ss_dirpath


async def get_audio_thumbnail(audio_file):
    """
    Extracts embedded cover art from an audio file.
    """
    await makedirs(THUMBNAIL_DIR, exist_ok=True)
    output = ospath.join(THUMBNAIL_DIR, f"{time()}.jpg")
    cmd = [
        BinConfig["FFMPEG_NAME"],
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        audio_file,
        "-an",
        "-vcodec",
        "copy",
        "-threads",
        f"{max(1, cpu_no // 2)}",
        output,
    ]
    try:
        _, err, code = await asyncio.wait_for(cmd_exec(cmd), timeout=60)
        if code != 0 or not await aiopath.exists(output):
            LOGGER.error(f"Error extracting audio thumbnail: {err}")
            return None
    except asyncio.TimeoutError:
        LOGGER.error(f"Error extracting audio thumbnail: Timeout.")
        return None
    return output


async def get_video_thumbnail(video_file, duration):
    """
    Generates a thumbnail for a video file.
    """
    await makedirs(THUMBNAIL_DIR, exist_ok=True)
    output = ospath.join(THUMBNAIL_DIR, f"{time()}.jpg")
    
    if duration is None:
        duration = (await get_media_info(video_file))[0]
    if duration == 0:
        duration = 3
    ss_time = duration // 2
    
    cmd = [
        BinConfig["FFMPEG_NAME"],
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{ss_time}",
        "-i",
        video_file,
        "-vf",
        f"thumbnail,scale={TG_THUMB_SIZE[0]}:-1", # Generate thumb and scale
        "-q:v",
        "5", # Quality
        "-vframes",
        "1", # Single frame
        "-threads",
        "1",
        output,
    ]
    try:
        _, err, code = await asyncio.wait_for(cmd_exec(cmd), timeout=60)
        if code != 0 or not await aiopath.exists(output):
            LOGGER.error(f"Error extracting video thumbnail: {err}")
            return None
    except asyncio.TimeoutError:
        LOGGER.error(f"Error extracting video thumbnail: Timeout.")
        return None
    return output


async def get_multiple_frames_thumbnail(video_file, layout, keep_screenshots=False):
    """
    Creates a tiled thumbnail from multiple video frames.
    """
    if not layout:
        return None
        
    layout = re.sub(r"(\d+)\D+(\d+)", r"\1x\2", layout)
    ss_nb_match = layout.split("x")
    if len(ss_nb_match) != 2 or not ss_nb_match[0].isdigit() or not ss_nb_match[1].isdigit():
        LOGGER.error(f"Invalid layout value: {layout}")
        return None
        
    ss_nb = int(ss_nb_match[0]) * int(ss_nb_match[1])
    if ss_nb == 0:
        LOGGER.error(f"Invalid layout value (0 screenshots): {layout}")
        return None
        
    ss_dirpath = await take_ss(video_file, ss_nb)
    if not ss_dirpath:
        return None
        
    await makedirs(THUMBNAIL_DIR, exist_ok=True)
    output = ospath.join(THUMBNAIL_DIR, f"{time()}.jpg")
    
    cmd = [
        BinConfig["FFMPEG_NAME"],
        "-hide_banner",
        "-loglevel",
        "error",
        "-pattern_type",
        "glob",
        "-i",
        f"{ospath.normpath(ss_dirpath)}/*", # Use normalized path
        "-vf",
        f"tile={layout}, scale={TG_THUMB_SIZE[0]}:-1", # Tile and scale
        "-q:v",
        "1",
        "-frames:v",
        "1",
        "-f",
        "mjpeg",
        "-threads",
        f"{max(1, cpu_no // 2)}",
        output,
    ]
    try:
        _, err, code = await asyncio.wait_for(cmd_exec(cmd), timeout=60)
        if code != 0 or not await aiopath.exists(output):
            LOGGER.error(f"Error combining thumbnails: {err}")
            return None
    except asyncio.TimeoutError:
        LOGGER.error(f"Error combining thumbnails: Timeout.")
        return None
    finally:
        if not keep_screenshots:
            await rmtree(ss_dirpath, ignore_errors=True)
            
    return output


# =================================================================================
# --- Core Uploader Class ---
# (Adapted from bot/helper/mirror_leech_utils/upload_utils/telegram_uploader.py)
# =================================================================================

class StandaloneTelegramUploader:
    def __init__(self, client: Client, file_path: str, chat_id, 
                 message_thread_id=None, reply_to_msg_id=None, thumb_path=None,
                 force_document=False):
        self._client = client
        self._up_path = file_path
        self._chat_id = chat_id
        self._message_thread_id = message_thread_id
        self._reply_to_msg_id = reply_to_msg_id
        self._thumb_param = thumb_path if thumb_path and thumb_path.lower() != "none" else None
        
        self._last_uploaded = 0
        self._processed_bytes = 0
        self._start_time = time()
        self._is_cancelled = False
        self._sent_message = None
        self._file_name = ospath.basename(file_path)
        
        self._is_corrupted = False
        self._gen_thumb_path = None # Stores path of ANY generated thumb for cleanup
        self._force_document = force_document

    async def _upload_progress(self, current, total):
        if self._is_cancelled:
            if hasattr(self._client, "stop_transmission"):
                self._client.stop_transmission()
            return
            
        chunk_size = current - self._last_uploaded
        self._last_uploaded = current
        self._processed_bytes += chunk_size
        
        # --- Simple Progress Logging ---
        try:
            percentage = (self._processed_bytes / total) * 100
            elapsed_time = time() - self._start_time
            if elapsed_time == 0: elapsed_time = 0.001
            speed = self._processed_bytes / elapsed_time
            if speed == 0: speed = 0.001
            eta = (total - self._processed_bytes) / speed
            
            display_name = (
                (self._file_name[:40] + "...")
                if len(self._file_name) > 43
                else self._file_name
            )
            
            progress_line = (
                f"\rUploading {display_name}: {percentage:.2f}% "
                f"({get_readable_file_size(self._processed_bytes)}/{get_readable_file_size(total)}) "
                f"@{get_readable_file_size(speed)}/s ETA: {get_readable_time(int(eta))}"
            )
            sys.stdout.write(progress_line.ljust(80)) 
            sys.stdout.flush()
        except Exception as e:
            LOGGER.warning(f"Upload progress error: {e}")
        # --- End Progress Logging ---

    async def _cleanup_generated_thumb(self):
        """Removes the generated thumbnail if it exists and wasn't the user's."""
        if self._gen_thumb_path and await aiopath.exists(self._gen_thumb_path):
             # Only delete if it's not the one the user provided
            if self._gen_thumb_path != self._thumb_param:
                LOGGER.info(f"Cleaning up generated thumbnail: {self._gen_thumb_path}")
                await remove(self._gen_thumb_path)
            self._gen_thumb_path = None # Reset path

    @retry(
        wait=wait_exponential(multiplier=2, min=4, max=8),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(Exception),
    )
    async def _upload_file(self, cap_mono, file, o_path, force_document=False):
        # Reset state for potential retries
        self._last_uploaded = 0
        await self._cleanup_generated_thumb() # Clean up thumb from previous failed attempt
        
        # Determine final force_document state
        is_forced = self._force_document or force_document
            
        # Validate custom thumbnail path
        current_thumb = self._thumb_param
        if current_thumb is not None and not await aiopath.exists(current_thumb):
            LOGGER.warning(f"Custom thumbnail not found: {current_thumb}. Resetting to None.")
            current_thumb = None
            
        self._is_corrupted = False

        try:
            is_video, is_audio, is_image = await get_document_type(self._up_path)

            # --- Thumbnail Generation Logic ---
            if current_thumb is None: # Only generate if no valid custom thumb exists
                if is_video:
                    current_thumb = await get_video_thumbnail(self._up_path, None)
                    self._gen_thumb_path = current_thumb # Store path for cleanup
                elif is_audio:
                    current_thumb = await get_audio_thumbnail(self._up_path)
                    self._gen_thumb_path = current_thumb
                # --- *** NEW: Generate thumb for images when forced as document *** ---
                elif is_image and is_forced:
                    LOGGER.info("Generating thumbnail for image being sent as document...")
                    current_thumb = await generate_image_thumbnail(self._up_path)
                    self._gen_thumb_path = current_thumb # Store path for cleanup
            # --- End Thumbnail Logic ---

            common_args = {
                "chat_id": self._chat_id,
                "message_thread_id": self._message_thread_id,
                "reply_to_message_id": self._reply_to_msg_id,
                "caption": cap_mono,
                "disable_notification": True,
                "progress": self._upload_progress,
                "thumb": current_thumb # Pass the determined thumb (custom or generated)
            }

            if self._is_cancelled:
                LOGGER.info("Upload cancelled before starting.")
                return

            # --- Upload Logic ---
            if is_forced:
                LOGGER.info(f"Uploading {self._file_name} as DOCUMENT (forced).")
                self._sent_message = await self._client.send_document(
                    document=self._up_path,
                    force_document=True, # Explicitly True here
                    **common_args,
                )
            elif is_video:
                LOGGER.info(f"Uploading {self._file_name} as VIDEO.")
                duration = (await get_media_info(self._up_path))[0]
                width, height = 0, 0
                if current_thumb: # Use determined thumb
                    try:
                        # Use await sync_to_async for Pillow operations
                        async def get_dims(p):
                            with Image.open(p) as img: return img.size
                        width, height = await sync_to_async(get_dims, current_thumb)
                    except Exception as e:
                        LOGGER.warning(f"Could not get thumb dimensions: {e}. Using defaults.")
                        width, height = 480, 320
                else:
                    width, height = 480, 320

                self._sent_message = await self._client.send_video(
                    video=self._up_path,
                    duration=duration,
                    width=width,
                    height=height,
                    supports_streaming=True,
                    **common_args, # common_args already includes 'thumb'
                )
            elif is_audio:
                LOGGER.info(f"Uploading {self._file_name} as AUDIO.")
                duration, artist, title = await get_media_info(self._up_path)
                self._sent_message = await self._client.send_audio(
                    audio=self._up_path,
                    duration=duration,
                    performer=artist,
                    title=title,
                    **common_args, # common_args already includes 'thumb'
                )
            elif is_image:
                LOGGER.info(f"Uploading {self._file_name} as PHOTO.")
                # send_photo does not use 'thumb', remove it from args
                photo_args = common_args.copy()
                del photo_args['thumb']
                self._sent_message = await self._client.send_photo(
                    photo=self._up_path,
                    **photo_args,
                )
            else: # Fallback to document (shouldn't happen often with get_document_type)
                LOGGER.info(f"Uploading {self._file_name} as DOCUMENT (auto-detected).")
                self._sent_message = await self._client.send_document(
                    document=self._up_path,
                    force_document=True,
                    **common_args, # common_args already includes 'thumb'
                )
            # --- End Upload Logic ---
                
            if self._sent_message:
                success_msg = f"\rSuccessfully uploaded: {self._file_name}"
                sys.stdout.write(success_msg.ljust(80) + "\n")
                sys.stdout.flush()

        except (FloodWait, FloodPremiumWait) as f:
            LOGGER.warning(f"\n{f}. Sleeping for {f.value * 1.3} seconds.")
            await asyncio.sleep(f.value * 1.3)
            # Don't clean thumb here, let retry handle it
            raise f
        except Exception as err:
            err_type = "RPCError: " if isinstance(err, RPCError) else ""
            LOGGER.error(f"\n{err_type}{err}. Path: {self._up_path}", exc_info=True)
            
            # This logic is for retrying when *not* forcing document initially
            if isinstance(err, (BadRequest, PhotoSaveFileInvalid)) and not is_forced:
                should_retry = False
                if isinstance(err, PhotoSaveFileInvalid):
                    LOGGER.error(f"Photo save failed. Retrying As Document: {self._up_path}")
                    should_retry = True
                elif "VIDEO_CONTENT_TYPE_INVALID" in str(err):
                     LOGGER.error(f"Bad video format. Retrying As Document: {self._up_path}")
                     should_retry = True
                
                if should_retry:
                    # Clean up thumb before retrying as doc might need a diff thumb
                    await self._cleanup_generated_thumb() 
                    return await self._upload_file(cap_mono, file, o_path, True)

            # If it's another error, or if we were already forcing document, raise it
            raise err
        finally:
            # Final cleanup after successful upload or exhausting retries
             await self._cleanup_generated_thumb()

    async def upload(self):
        """
        Public method to start the upload process.
        """
        try:
            if not await aiopath.exists(self._up_path):
                LOGGER.error(f"File not found: {self._up_path}")
                return
                
            f_size = await aiopath.getsize(self._up_path)
            if f_size == 0:
                LOGGER.error(f"File size is zero. Skipping: {self._file_name}")
                return

            LOGGER.info(f"Starting upload for: {self._file_name} (Size: {get_readable_file_size(f_size)})")
            
            simple_caption = f"<code>{escape(self._file_name)}</code>"
            
            self._start_time = time()
            # Initial call - respects _force_document but allows retry logic inside
            await self._upload_file(simple_caption, self._file_name, self._up_path, 
                                    force_document=self._force_document) 
            
        except RetryError as e:
            LOGGER.error(f"\nUpload FAILED after {e.last_attempt.attempt_number} attempts for {self._file_name}.")
            LOGGER.error(f"Last error: {e.last_attempt.exception()}")
            # Ensure cleanup happens even after all retries fail
            await self._cleanup_generated_thumb()
            raise e
        except Exception as e:
            LOGGER.error(f"\nAn unexpected error occurred during upload: {e}", exc_info=True)
             # Ensure cleanup happens on unexpected errors
            await self._cleanup_generated_thumb()
            raise e
        
        if self._is_cancelled:
            LOGGER.info(f"Upload was cancelled for {self._file_name}.")
            
    def cancel_upload(self):
        self._is_cancelled = True
        LOGGER.info(f"\nCancel request received for {self._file_name}. Will stop at next progress update.")


# =================================================================================
# --- Main Execution Logic ---
# =================================================================================

async def upload_file_task(app, file_path, chat_id, topic_id, reply_id, thumb, 
                           force_document=False):
    """
    Helper function to create and run an uploader instance for a single file.
    """
    try:
        if not await aiopath.exists(file_path):
            LOGGER.error(f"File does not exist: {file_path}")
            return False
        if await aiopath.getsize(file_path) == 0:
            LOGGER.warning(f"File size is 0. Skipping: {file_path}")
            return True
            
        uploader = StandaloneTelegramUploader(
            client=app,
            file_path=file_path,
            chat_id=chat_id,
            message_thread_id=topic_id,
            reply_to_msg_id=reply_id,
            thumb_path=thumb,
            force_document=force_document
        )
        await uploader.upload()
        return True
        
    except Exception as e:
        # Error is already logged inside uploader.upload() or _upload_file()
        # Just indicate failure for the main loop counter
        return False


async def main(input_path, chat_id, topic_id=None, reply_id=None, thumb=None, 
               force_document=False):
    if not os.path.exists(input_path):
        LOGGER.error(f"Path does not exist: {input_path}")
        return

    LOGGER.info("Initializing Pyrogram Client...")
    # Ensure thumbnail directory exists before starting client
    await makedirs(THUMBNAIL_DIR, exist_ok=True)
    
    async with Client(
        "standalone_uploader",
        api_id=API_ID,
        api_hash=API_HASH,
        bot_token=BOT_TOKEN
    ) as app:
        LOGGER.info("Client initialized.")
        
        if os.path.isfile(input_path):
            # --- Single File Mode ---
            await upload_file_task(app, input_path, chat_id, topic_id, reply_id, thumb, 
                                   force_document)
            
        elif os.path.isdir(input_path):
            # --- Folder Mode ---
            LOGGER.info(f"Processing folder: {input_path}")
            files_to_upload = []
            for dirpath, _, filenames in os.walk(input_path, topdown=True):
                # Skip thumbnail generation folders
                if ospath.basename(dirpath).endswith("_mltbss"):
                    continue
                filenames.sort()  
                for filename in filenames:
                    if not filename.startswith('.'):
                        files_to_upload.append(os.path.join(dirpath, filename))
            
            files_to_upload.sort() 
            
            total_files = len(files_to_upload)
            success_count = 0
            fail_count = 0
            LOGGER.info(f"Found {total_files} files to upload.")
            
            for i, file_path in enumerate(files_to_upload):
                LOGGER.info(f"\n--- Uploading file {i+1}/{total_files}: {file_path} ---")
                try:
                    success = await upload_file_task(app, file_path, chat_id, topic_id, 
                                                     reply_id, thumb, force_document)
                    if success:
                        LOGGER.info(f"--- Finished file {i+1}/{total_files} ---")
                        success_count += 1
                    else:
                        LOGGER.error(f"--- FAILED file {i+1}/{total_files}: {file_path} (Check logs above) ---")
                        fail_count += 1
                except Exception as e: # Catch errors not caught within upload_file_task (e.g., RetryError)
                    LOGGER.error(f"--- FAILED file {i+1}/{total_files}: {file_path}. Unhandled Error: {e} ---")
                    fail_count += 1
                    continue # Move to the next file
            
            LOGGER.info(f"\nFolder upload complete. {success_count} successful, {fail_count} failed.")
        
        else:
            LOGGER.error(f"Input path is not a file or directory: {input_path}")

    LOGGER.info("Upload process finished. Client session closed.")


if __name__ == "__main__":
    if API_ID == 1234567 or API_HASH == "your_api_hash" or BOT_TOKEN == "your_bot_token":
        LOGGER.error("!!! CONFIGURATION ERROR !!!")
        LOGGER.error("Please fill in your API_ID, API_HASH, and BOT_TOKEN at the top of the script.")
        sys.exit(1)
    
    # --- Argument Parser ---
    parser = argparse.ArgumentParser(
        description="Standalone Telegram Uploader for files and folders.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # === REQUIRED OPTIONAL ARGUMENTS ===
    parser.add_argument(
        "-p", "--path",
        required=True,
        help="(Required) Path to the file or folder to upload."
    )
    parser.add_argument(
        "-c", "--chat-id",
        type=int,
        required=True,
        help="(Required) Target Chat ID (e.g., -100123456 or 123456789)."
    )
    
    # === OTHER OPTIONAL ARGUMENTS ===
    parser.add_argument(
        "-t", "--topic-id",
        type=int,
        default=None,
        help="(Optional) Target Topic ID (message_thread_id) to upload files into."
    )
    parser.add_argument(
        "-r", "--reply-id",
        type=int,
        default=None,
        help="(Optional) Message ID to reply to."
    )
    parser.add_argument(
        "--thumb",
        default=None,
        help="(Optional) Path to a custom thumbnail file (e.g., /path/to/thumb.jpg)."
    )
    parser.add_argument(
        "-d", "--as-document",
        action="store_true",
        help="(Optional) Upload all files as documents, bypassing auto-detection."
    )
    
    args = parser.parse_args()

    # Validate thumbnail path
    _thumb_path = args.thumb
    if _thumb_path:
        if _thumb_path.lower() == "none":
            _thumb_path = None
        elif not os.path.exists(_thumb_path):
            LOGGER.warning(f"Thumbnail file not found: {_thumb_path}. Ignoring.")
            _thumb_path = None
            
    # --- Run Main Async Function ---
    try:
        asyncio.run(main(args.path, args.chat_id, args.topic_id, args.reply_id, _thumb_path,
                         force_document=args.as_document))
    except KeyboardInterrupt:
        LOGGER.info("\nProcess interrupted by user.")
    except Exception as e:
        LOGGER.error(f"An unexpected error occurred: {e}", exc_info=True)
    finally:
        # --- Final Cleanup of Thumbnail Directory ---
        # Attempt to remove the thumbnail directory if it exists and is empty
        try:
             if os.path.exists(THUMBNAIL_DIR) and not os.listdir(THUMBNAIL_DIR):
                  LOGGER.info(f"Cleaning up empty thumbnail directory: {THUMBNAIL_DIR}")
                  os.rmdir(THUMBNAIL_DIR)
        except Exception as e:
             LOGGER.warning(f"Could not remove thumbnail directory {THUMBNAIL_DIR}: {e}")