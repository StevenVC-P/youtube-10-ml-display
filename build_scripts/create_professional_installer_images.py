"""
Create professional installer images for Inno Setup.

This script generates modern, professional-looking images for the Windows installer
with gradients, proper branding, and polished design.

Generated images:
  - wizard_large.bmp (164x314) - Large wizard sidebar image with gradient
  - wizard_small.bmp (55x55) - Small wizard header image
  - app_icon.ico (256x256) - Application icon with multiple sizes

Usage:
    python create_professional_installer_images.py
"""

from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import sys
import io

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except:
        pass

# Output directory
OUTPUT_DIR = Path(__file__).parent / "installer_images"
OUTPUT_DIR.mkdir(exist_ok=True)

# Modern Color Scheme - Deep Blue with Cyan Accents
PRIMARY_DARK = (15, 23, 42)       # Slate 900 - #0f172a
PRIMARY = (30, 41, 59)            # Slate 800 - #1e293b
PRIMARY_LIGHT = (51, 65, 85)      # Slate 700 - #334155
ACCENT = (6, 182, 212)            # Cyan 500 - #06b6d4
ACCENT_BRIGHT = (34, 211, 238)    # Cyan 400 - #22d3ee
TEXT_PRIMARY = (248, 250, 252)    # Slate 50 - #f8fafc
TEXT_SECONDARY = (203, 213, 225)  # Slate 300 - #cbd5e1
SUCCESS = (34, 197, 94)           # Green 500 - #22c55e
WARNING = (251, 146, 60)          # Orange 400 - #fb923c


def create_radial_gradient(width, height, center_color, edge_color):
    """Create a radial gradient from center to edges."""
    img = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(img)

    # Calculate center
    cx, cy = width // 2, height // 2
    max_radius = ((width/2)**2 + (height/2)**2)**0.5

    for y in range(height):
        for x in range(width):
            # Calculate distance from center
            distance = ((x - cx)**2 + (y - cy)**2)**0.5
            ratio = min(distance / max_radius, 1.0)

            # Interpolate colors
            r = int(center_color[0] + (edge_color[0] - center_color[0]) * ratio)
            g = int(center_color[1] + (edge_color[1] - center_color[1]) * ratio)
            b = int(center_color[2] + (edge_color[2] - center_color[2]) * ratio)

            img.putpixel((x, y), (r, g, b))

    return img


def create_vertical_gradient(width, height, top_color, bottom_color):
    """Create a smooth vertical gradient."""
    img = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(img)

    for y in range(height):
        ratio = y / height
        r = int(top_color[0] + (bottom_color[0] - top_color[0]) * ratio)
        g = int(top_color[1] + (bottom_color[1] - top_color[1]) * ratio)
        b = int(top_color[2] + (bottom_color[2] - top_color[2]) * ratio)

        draw.line([(0, y), (width, y)], fill=(r, g, b))

    return img


def create_wizard_large():
    """Create professional large wizard sidebar image (164x314)."""
    print("Creating professional wizard_large.bmp (164x314)...")

    # Create gradient background
    img = create_vertical_gradient(164, 314, PRIMARY_DARK, PRIMARY)
    draw = ImageDraw.Draw(img)

    # Try to load fonts
    try:
        title_font = ImageFont.truetype("segoeui.ttf", 22)
        subtitle_font = ImageFont.truetype("segoeui.ttf", 14)
        small_font = ImageFont.truetype("segoeui.ttf", 10)
    except:
        try:
            title_font = ImageFont.truetype("arial.ttf", 20)
            subtitle_font = ImageFont.truetype("arial.ttf", 12)
            small_font = ImageFont.truetype("arial.ttf", 9)
        except:
            title_font = ImageFont.load_default()
            subtitle_font = ImageFont.load_default()
            small_font = ImageFont.load_default()

    # Add subtle decorative circles in background
    for i, (x, y, r) in enumerate([(20, 40, 60), (140, 200, 80), (40, 270, 50)]):
        alpha = 10 + i * 5
        for radius in range(r, 0, -5):
            color_val = PRIMARY_LIGHT[0] + alpha
            draw.ellipse([x-radius, y-radius, x+radius, y+radius],
                        outline=(color_val, color_val, color_val), width=1)

    # Draw accent bar on left
    draw.rectangle([0, 0, 4, 314], fill=ACCENT)

    # Draw title "RETRO ML"
    title_text = "RETRO ML"
    title_bbox = draw.textbbox((0, 0), title_text, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    title_x = (164 - title_width) // 2

    # Title with shadow
    draw.text((title_x + 1, 41), title_text, fill=PRIMARY_DARK, font=title_font)
    draw.text((title_x, 40), title_text, fill=TEXT_PRIMARY, font=title_font)

    # Draw subtitle "TRAINER"
    subtitle_text = "TRAINER"
    subtitle_bbox = draw.textbbox((0, 0), subtitle_text, font=subtitle_font)
    subtitle_width = subtitle_bbox[2] - subtitle_bbox[0]
    subtitle_x = (164 - subtitle_width) // 2

    draw.text((subtitle_x, 70), subtitle_text, fill=ACCENT_BRIGHT, font=subtitle_font)

    # Draw decorative line
    draw.rectangle([40, 92, 124, 94], fill=ACCENT)

    # Draw icon elements - Modern game controller
    controller_y = 120

    # Controller body
    draw.rounded_rectangle([32, controller_y, 132, controller_y + 40],
                          radius=8, outline=ACCENT_BRIGHT, width=2)

    # D-pad (left side)
    dpad_x, dpad_y = 50, controller_y + 20
    draw.rectangle([dpad_x-2, dpad_y-8, dpad_x+2, dpad_y+8], fill=TEXT_SECONDARY)  # Vertical
    draw.rectangle([dpad_x-8, dpad_y-2, dpad_x+8, dpad_y+2], fill=TEXT_SECONDARY)  # Horizontal

    # Buttons (right side)
    button_x, button_y = 114, controller_y + 20
    draw.ellipse([button_x-6, button_y-6, button_x+6, button_y+6], fill=SUCCESS)  # A button
    draw.ellipse([button_x-18, button_y-12, button_x-6, button_y], fill=WARNING)  # Y button

    # AI/ML themed elements
    ml_y = 190

    # Draw neural network nodes
    for i, x in enumerate([50, 82, 114]):
        for j, y in enumerate([ml_y, ml_y + 25, ml_y + 50]):
            size = 3
            draw.ellipse([x-size, y-size, x+size, y+size], fill=ACCENT if (i+j) % 2 == 0 else TEXT_SECONDARY)

    # Draw connections between nodes
    for i in range(2):
        for j in range(3):
            x1, y1 = 50 + i*32, ml_y + j*25
            x2, y2 = 82 + i*32, ml_y + (j+1)*25 if j < 2 else ml_y + j*25
            draw.line([(x1, y1), (x2, y2)], fill=PRIMARY_LIGHT, width=1)

    # Bottom text
    tagline = "AI-Powered Game Training"
    tag_bbox = draw.textbbox((0, 0), tagline, font=small_font)
    tag_width = tag_bbox[2] - tag_bbox[0]
    draw.text(((164 - tag_width) // 2, 280), tagline, fill=TEXT_SECONDARY, font=small_font)

    # Save
    output_path = OUTPUT_DIR / "wizard_large.bmp"
    img.save(output_path, "BMP")
    print(f"✓ Created: {output_path}")
    return img


def create_wizard_small():
    """Create professional small wizard header image (55x55)."""
    print("Creating professional wizard_small.bmp (55x55)...")

    # Create gradient background
    img = create_radial_gradient(55, 55, PRIMARY_LIGHT, PRIMARY_DARK)
    draw = ImageDraw.Draw(img)

    # Draw accent circle
    draw.ellipse([2, 2, 53, 53], outline=ACCENT, width=2)

    # Draw game controller icon (simplified)
    # Controller body
    draw.rounded_rectangle([12, 20, 43, 35], radius=3, outline=ACCENT_BRIGHT, width=2)

    # D-pad
    dpad_x, dpad_y = 20, 27
    draw.rectangle([dpad_x-1, dpad_y-3, dpad_x+1, dpad_y+3], fill=TEXT_PRIMARY)
    draw.rectangle([dpad_x-3, dpad_y-1, dpad_x+3, dpad_y+1], fill=TEXT_PRIMARY)

    # Buttons
    draw.ellipse([32, 24, 36, 28], fill=SUCCESS)
    draw.ellipse([28, 21, 32, 25], fill=WARNING)

    # Save
    output_path = OUTPUT_DIR / "wizard_small.bmp"
    img.save(output_path, "BMP")
    print(f"✓ Created: {output_path}")
    return img


def create_app_icon():
    """Create professional application icon (256x256 with multiple sizes)."""
    print("Creating professional app_icon.ico (256x256 + multiple sizes)...")

    # Create main 256x256 icon
    img = create_radial_gradient(256, 256, PRIMARY, PRIMARY_DARK)
    draw = ImageDraw.Draw(img)

    # Draw outer circle with gradient effect
    for i in range(8):
        offset = i * 2
        alpha = 255 - i * 20
        draw.ellipse([10+offset, 10+offset, 246-offset, 246-offset],
                    outline=ACCENT if i % 2 == 0 else ACCENT_BRIGHT, width=2)

    # Draw game controller icon (large, centered)
    # Controller body
    draw.rounded_rectangle([60, 100, 196, 156], radius=12, outline=ACCENT_BRIGHT, width=4)
    draw.rounded_rectangle([64, 104, 192, 152], radius=10, fill=PRIMARY_LIGHT)

    # D-pad (left)
    dpad_x, dpad_y = 100, 128
    draw.rectangle([dpad_x-4, dpad_y-16, dpad_x+4, dpad_y+16], fill=TEXT_PRIMARY)
    draw.rectangle([dpad_x-16, dpad_y-4, dpad_x+16, dpad_y+4], fill=TEXT_PRIMARY)

    # Buttons (right)
    button_x, button_y = 156, 128
    draw.ellipse([button_x-12, button_y-12, button_x+12, button_y+12], fill=SUCCESS, outline=TEXT_PRIMARY, width=2)  # A
    draw.ellipse([button_x-32, button_y-20, button_x-8, button_y+4], fill=WARNING, outline=TEXT_PRIMARY, width=2)  # Y

    # Neural network decoration at bottom
    nn_y = 180
    for i, x in enumerate([90, 128, 166]):
        for j, y in enumerate([nn_y, nn_y + 20, nn_y + 40]):
            size = 4
            color = ACCENT if (i + j) % 2 == 0 else ACCENT_BRIGHT
            draw.ellipse([x-size, y-size, x+size, y+size], fill=color)

    # Connections
    for i in range(2):
        for j in range(2):
            x1, y1 = 90 + i*38, nn_y + j*20
            x2, y2 = 128 + i*38, nn_y + (j+1)*20
            draw.line([(x1, y1), (x2, y2)], fill=PRIMARY_LIGHT, width=2)

    # Try to load font for "ML" text
    try:
        font = ImageFont.truetype("segoeui.ttf", 36)
        bold_font = ImageFont.truetype("segoeuib.ttf", 36)
    except:
        try:
            font = ImageFont.truetype("arial.ttf", 32)
            bold_font = font
        except:
            font = ImageFont.load_default()
            bold_font = font

    # Draw "ML" text with shadow
    ml_text = "ML"
    ml_bbox = draw.textbbox((0, 0), ml_text, font=bold_font)
    ml_width = ml_bbox[2] - ml_bbox[0]
    ml_x = (256 - ml_width) // 2
    ml_y = 45

    draw.text((ml_x + 2, ml_y + 2), ml_text, fill=PRIMARY_DARK, font=bold_font)
    draw.text((ml_x, ml_y), ml_text, fill=ACCENT_BRIGHT, font=bold_font)

    # Create multiple icon sizes for better Windows compatibility
    icon_sizes = [(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)]
    icon_images = []

    for size in icon_sizes:
        if size == (256, 256):
            icon_images.append(img)
        else:
            resized = img.resize(size, Image.Resampling.LANCZOS)
            icon_images.append(resized)

    # Save as ICO with multiple sizes
    output_path = OUTPUT_DIR / "app_icon.ico"
    icon_images[0].save(
        output_path,
        format='ICO',
        sizes=[(img.width, img.height) for img in icon_images]
    )
    print(f"✓ Created: {output_path} (with sizes: {', '.join([f'{s[0]}x{s[1]}' for s in icon_sizes])})")
    return icon_images[0]


def main():
    """Generate all installer images."""
    print("=" * 60)
    print("PROFESSIONAL INSTALLER IMAGE GENERATOR")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    try:
        # Create images
        create_wizard_large()
        create_wizard_small()
        create_app_icon()

        print()
        print("=" * 60)
        print("✓ SUCCESS! All professional installer images created!")
        print("=" * 60)
        print()
        print("Generated files:")
        print(f"  • {OUTPUT_DIR / 'wizard_large.bmp'} (164x314)")
        print(f"  • {OUTPUT_DIR / 'wizard_small.bmp'} (55x55)")
        print(f"  • {OUTPUT_DIR / 'app_icon.ico'} (256x256 + multiple sizes)")
        print()
        print("Next steps:")
        print("  1. Review the generated images")
        print("  2. Build the installer: cd build_scripts && build_installer_modern.bat")
        print("  3. Test the installer on a clean system")

    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
