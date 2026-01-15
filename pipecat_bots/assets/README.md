# Avatar Assets for Audio2Face-2D

This directory contains avatar images for the NIM Assistant.

## Required Files

Place your avatar images here:

- **`avatar_female.png`** - Female anime assistant (cyberpunk style, pink hair, "NIM ASSISTANT" badge)
- **`avatar_male.png`** - Male anime assistant (same style, alternative option)

The bot defaults to the female avatar with the "aria" female voice.

## Image Requirements

For best results with Audio2Face-2D NIM:
- **Format**: PNG (recommended) or JPEG
- **Size**: 512x512 pixels or larger
- **Content**: Portrait showing face clearly, centered in frame
- **Style**: Anime/illustrated characters work well

## Configuration

### Switch Avatar Gender
```bash
# Use female avatar (default)
export AVATAR_VARIANT=female

# Use male avatar
export AVATAR_VARIANT=male
```

### Custom Avatar Path
```bash
# Use a custom avatar image
export NVIDIA_A2F_AVATAR=/path/to/your/custom_avatar.png
```

### Voice Options
The TTS voice automatically matches the avatar gender, but you can override:
```bash
# Female voices
export TTS_VOICE=aria    # Default female
export TTS_VOICE=sofia   # Alternative female

# Male voices
export TTS_VOICE=john    # Default male
export TTS_VOICE=jason   # Alternative male
export TTS_VOICE=leo     # Alternative male
```

## Running Without Avatar Images

If avatar images are missing, the bot will:
1. Log a warning about the missing image
2. Continue running without visual output (audio-only mode)

To explicitly disable avatar:
```bash
export ENABLE_AVATAR=false
```
