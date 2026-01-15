#!/usr/bin/env python3
"""Generate a simple anime-style avatar placeholder.

This creates a basic anime-style character portrait that works with
Audio2Face-2D NIM. Replace with your own image for better results.

Requirements:
    pip install pillow

Usage:
    python generate_avatar.py
"""

from pathlib import Path

try:
    from PIL import Image, ImageDraw, ImageFilter
except ImportError:
    print("Pillow not installed. Install with: pip install pillow")
    raise


def create_anime_avatar(
    size: tuple[int, int] = (512, 512),
    output_path: str = "avatar.png"
) -> Path:
    """Create a simple anime-style avatar.

    Args:
        size: Output image size (width, height)
        output_path: Path to save the avatar

    Returns:
        Path to the generated avatar
    """
    width, height = size

    # Create base image with gradient background
    img = Image.new("RGB", size, "#E8F0FE")
    draw = ImageDraw.Draw(img)

    # Background gradient (soft blue to white)
    for y in range(height):
        ratio = y / height
        r = int(232 + (255 - 232) * ratio)
        g = int(240 + (255 - 240) * ratio)
        b = int(254 + (255 - 254) * ratio)
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    # Face base (oval)
    face_center = (width // 2, height // 2 - 20)
    face_width = 180
    face_height = 220
    face_bbox = (
        face_center[0] - face_width // 2,
        face_center[1] - face_height // 2,
        face_center[0] + face_width // 2,
        face_center[1] + face_height // 2,
    )
    # Skin tone
    skin_color = "#FFE4C4"  # Bisque
    draw.ellipse(face_bbox, fill=skin_color, outline=skin_color)

    # Hair (anime style - covering top of head and sides)
    hair_color = "#4A3728"  # Dark brown
    # Top hair
    hair_top = [
        (face_center[0] - 100, face_center[1] - 60),
        (face_center[0] - 110, face_center[1] - 100),
        (face_center[0] - 80, face_center[1] - 130),
        (face_center[0] - 40, face_center[1] - 140),
        (face_center[0], face_center[1] - 145),
        (face_center[0] + 40, face_center[1] - 140),
        (face_center[0] + 80, face_center[1] - 130),
        (face_center[0] + 110, face_center[1] - 100),
        (face_center[0] + 100, face_center[1] - 60),
    ]
    draw.polygon(hair_top, fill=hair_color)

    # Side hair
    draw.polygon([
        (face_center[0] - 100, face_center[1] - 60),
        (face_center[0] - 105, face_center[1] + 60),
        (face_center[0] - 95, face_center[1] + 80),
        (face_center[0] - 80, face_center[1] + 10),
    ], fill=hair_color)

    draw.polygon([
        (face_center[0] + 100, face_center[1] - 60),
        (face_center[0] + 105, face_center[1] + 60),
        (face_center[0] + 95, face_center[1] + 80),
        (face_center[0] + 80, face_center[1] + 10),
    ], fill=hair_color)

    # Bangs
    bangs_color = "#5D4037"
    for i, offset in enumerate([-60, -30, 0, 30, 60]):
        x = face_center[0] + offset
        draw.polygon([
            (x - 15, face_center[1] - 100),
            (x + 15, face_center[1] - 100),
            (x + 10, face_center[1] - 40),
            (x - 10, face_center[1] - 45),
        ], fill=bangs_color)

    # Eyes (anime style - large and expressive)
    eye_y = face_center[1] - 10
    eye_spacing = 55

    for side in [-1, 1]:
        eye_x = face_center[0] + side * eye_spacing // 2

        # Eye white
        draw.ellipse([
            eye_x - 25, eye_y - 20,
            eye_x + 25, eye_y + 15
        ], fill="white", outline="#E0E0E0")

        # Iris (large, colorful)
        iris_color = "#6B5B95"  # Purple
        draw.ellipse([
            eye_x - 18, eye_y - 15,
            eye_x + 18, eye_y + 15
        ], fill=iris_color)

        # Pupil
        draw.ellipse([
            eye_x - 8, eye_y - 8,
            eye_x + 8, eye_y + 8
        ], fill="#1A1A2E")

        # Highlight (anime style)
        draw.ellipse([
            eye_x - 12, eye_y - 12,
            eye_x - 4, eye_y - 4
        ], fill="white")
        draw.ellipse([
            eye_x + 2, eye_y + 2,
            eye_x + 6, eye_y + 6
        ], fill="white")

        # Eyelashes (top)
        draw.arc([
            eye_x - 28, eye_y - 25,
            eye_x + 28, eye_y + 10
        ], start=180, end=360, fill="#2D2D2D", width=3)

    # Eyebrows
    eyebrow_y = eye_y - 35
    for side in [-1, 1]:
        brow_x = face_center[0] + side * eye_spacing // 2
        draw.arc([
            brow_x - 25, eyebrow_y - 10,
            brow_x + 25, eyebrow_y + 15
        ], start=200 if side < 0 else 320, end=340 if side < 0 else 220, fill="#4A3728", width=4)

    # Nose (subtle, anime style)
    nose_y = face_center[1] + 20
    draw.polygon([
        (face_center[0], nose_y - 5),
        (face_center[0] - 8, nose_y + 10),
        (face_center[0] + 8, nose_y + 10),
    ], fill="#E8C4A8")

    # Mouth (small, friendly smile)
    mouth_y = face_center[1] + 55
    draw.arc([
        face_center[0] - 20, mouth_y - 10,
        face_center[0] + 20, mouth_y + 15
    ], start=0, end=180, fill="#E57373", width=3)

    # Blush (anime style cheeks)
    blush_color = "#FFB6C1"
    for side in [-1, 1]:
        blush_x = face_center[0] + side * 65
        blush_y = face_center[1] + 25
        draw.ellipse([
            blush_x - 15, blush_y - 8,
            blush_x + 15, blush_y + 8
        ], fill=blush_color)

    # Neck
    neck_top = face_center[1] + face_height // 2 - 20
    draw.polygon([
        (face_center[0] - 35, neck_top),
        (face_center[0] + 35, neck_top),
        (face_center[0] + 45, height),
        (face_center[0] - 45, height),
    ], fill=skin_color)

    # Shoulders/clothing
    shirt_color = "#5C6BC0"  # Indigo
    draw.polygon([
        (0, height - 80),
        (face_center[0] - 45, neck_top + 40),
        (face_center[0] + 45, neck_top + 40),
        (width, height - 80),
        (width, height),
        (0, height),
    ], fill=shirt_color)

    # Collar
    collar_color = "#FFFFFF"
    draw.polygon([
        (face_center[0] - 40, neck_top + 40),
        (face_center[0], neck_top + 80),
        (face_center[0] + 40, neck_top + 40),
        (face_center[0], neck_top + 50),
    ], fill=collar_color, outline="#E0E0E0")

    # Soft blur for smoother appearance
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))

    # Save
    output = Path(output_path)
    img.save(output, "PNG")
    print(f"Avatar saved to: {output.absolute()}")
    return output


if __name__ == "__main__":
    avatar_path = Path(__file__).parent / "avatar.png"
    create_anime_avatar(output_path=str(avatar_path))
