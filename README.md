# 🚀 Standalone Telegram Uploader

A simple, powerful, and standalone Python script to upload files or entire folders to Telegram chats, groups, or topics.  
All functionality is packed into a single file with no external dependencies beyond required Python libraries.

This script is an enhanced and streamlined version of the uploader feature from the [WZML‑X](https://github.com/SilentDemonSD/WZML-X) project.

---

## ✨ Features

- **Single File Upload** — Upload any individual file.
- **Folder Upload** — Recursively upload all files within a folder, maintaining order.
- **Topic Support** — Send files directly to a specific topic within a group.
- **Reply Support** — Reply to an existing message with your upload.
- **Force Document Mode** — Upload all files as documents to preserve original quality.
- **Auto Thumbnail Generation** — Automatically creates thumbnails for video and audio files.
- **Custom Thumbnails** — Use your own image as a thumbnail.
- **Standalone** — Everything you need in a single Python file.
- **Cross‑Platform** — Works seamlessly on Windows, Linux, and macOS.

---

## 🔧 Setup

### 1. Prerequisites

- **Python 3.8+** — Ensure Python is installed on your system.
- **FFmpeg** — Required for thumbnail generation and media info extraction.

#### Installation

**Windows:**
- Download FFmpeg from [ffmpeg.org](https://ffmpeg.org/download.html)
- Add the extracted `bin` folder to your system PATH.

**Linux (Debian/Ubuntu):**
```
sudo apt update && sudo apt install ffmpeg -y
```

---

### 2. Install Required Python Libraries

Use pip to install all dependencies:

```
pip install pyrofork tgcrypto tenacity pillow aiofiles psutil langcodes
```

---

### 3. Configure the Script 🔑

Open the Python script (`.py` file) in any text editor and insert your Telegram API credentials.  
You can obtain these from [my.telegram.org](https://my.telegram.org).

```
# PLEASE FILL THESE VALUES
API_ID = 1234567             # Your API ID from my.telegram.org
API_HASH = "your_api_hash"   # Your API Hash from my.telegram.org
BOT_TOKEN = "your_bot_token" # Your Bot Token from BotFather
```

---

## 🛠️ Usage

Run the script from your terminal or command prompt with appropriate arguments.

### Command Structure

```
python your_script_name.py -p "path/to/file_or_folder" -c CHAT_ID [options]
```

---

### Required Arguments

- `-p`, `--path` — Path to the file or folder to upload (use quotes).
- `-c`, `--chat-id` — ID of the target Telegram chat (e.g., `-100123456789`).

### Optional Arguments

- `-t`, `--topic-id` — ID of the topic/thread for sending files.
- `-r`, `--reply-id` — Message ID to reply to.
- `-d`, `--as-document` — Force all uploads as documents (no compression).
- `--thumb` — Path to a custom thumbnail image (e.g., `"path/to/thumb.jpg"`).

---

## 📋 Examples

### 1. Upload a Single File 📄
```
python script.py -p "C:\movies\my_video.mp4" -c -100123456789
```

### 2. Upload an Entire Folder 📂
Uploads all files inside the specified folder.
```
python script.py -p "/home/user/pictures/My Photos" -c -100123456789
```

### 3. Upload a Folder to a Specific Topic 🎯
```
python script.py -p "D:\My Project" -c -100123456789 -t 42
```

### 4. Upload a Video as a Document 💾
Prevents Telegram from compressing the file.
```
python script.py -p "vacation.mov" -c -100123456789 -d
```

### 5. Upload with a Custom Thumbnail 🖼️
```
python script.py -p "song.mp3" -c 123456789 --thumb "album_cover.png"
```

### 6. Combine Multiple Options ✨
Uploads an entire folder to topic #1045, replies to message #5021, and forces uploads as documents.
```
python script.py -p "C:\files\important_stuff" -c -100123456789 -t 1045 -r 5021 -d
```

---

## 🧩 Notes

- Large uploads may take time depending on Telegram’s rate limits.
- Make sure your bot has permission to send media in the target chat or topic.
- For safety, avoid public group uploads with sensitive data.
- Works flawlessly with both userbots and bot accounts.

