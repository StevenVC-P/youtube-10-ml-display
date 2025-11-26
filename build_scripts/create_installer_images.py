"""
Create placeholder installer images for Inno Setup.

This script generates simple placeholder images that can be used
for the installer until custom graphics are created.

Generated images:
  - wizard_large.bmp (164x314) - Large wizard sidebar image
  - wizard_small.bmp (55x55) - Small wizard header image
  - app_icon.ico (256x256) - Application icon

Usage:
    python create_installer_images.py
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

# Colors - Modern dark blue theme
DARK_BLUE = (25, 33, 48)      # #192130
ACCENT_BLUE = (66, 135, 245)  # #4287f5
LIGHT_GRAY = (200, 200, 200)  # #c8c8c8
WHITE = (255, 255, 255)
DARKER_BLUE = (15, 23, 38)    # #0f1726


def create_gradient(width, height, color1, color2):
    """Create a vertical gradient between two colors."""
    base = Image.new('RGB', (width, height), color1)
    top = Image.new('RGB', (width, height), color2)
    mask = Image.new('L', (width, height))
    mask_data = []
    for y in range(height):
        mask_data.extend([int(255 * (y / height))] * width)
    mask.putdata(mask_data)
    base.paste(top, (0, 0), mask)
    return base


def create_wizard_large():
    """Create the large wizard sidebar image (164x314)."""
    print("Creating wizard_large.bmp (164x314)...")

    # Create gradient background
    img = create_gradient(164, 314, DARKER_BLUE, DARK_BLUE)
    draw = ImageDraw.Draw(img)

    # Try to use a nice font, fall back to default if not available
    try:
        title_font = ImageFont.truetype("arial.ttf", 16)
        subtitle_font = ImageFont.truetype("arial.ttf", 11)
        small_font = ImageFont.truetype("arial.ttf", 9)
    except:
        title_font = ImageFont.load_default()
        subtitle_font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    # Draw title
    title = "Retro ML"
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    draw.text(
        ((164 - title_width) // 2, 30),
        title,
        fill=WHITE,
        font=title_font
    )

    # Draw subtitle
    subtitle = "Trainer"
    subtitle_bbox = draw.textbbox((0, 0), subtitle, font=title_font)
    subtitle_width = subtitle_bbox[2] - subtitle_bbox[0]
    draw.text(
        ((164 - subtitle_width) // 2, 50),
        subtitle,
        fill=ACCENT_BLUE,
        font=title_font
    )

    # Draw decorative elements - AI/Game themed
    # Draw a simple game controller icon
    draw.rectangle([52, 90, 112, 120], outline=ACCENT_BLUE, width=2)
    draw.ellipse([50, 100, 58, 108], fill=LIGHT_GRAY)  # Left button
    draw.ellipse([106, 100, 114, 108], fill=LIGHT_GRAY)  # Right button
    draw.rectangle([72, 98, 80, 110], outline=LIGHT_GRAY, width=1)  # D-pad left
    draw.rectangle([84, 98, 92, 110], outline=LIGHT_GRAY, width=1)  # D-pad right

    # Draw version info at bottom
    version = "v1.0.0"
    version_bbox = draw.textbbox((0, 0), version, font=small_font)
    version_width = version_bbox[2] - version_bbox[0]
    draw.text(
        ((164 - version_width) // 2, 280),
        version,
        fill=LIGHT_GRAY,
        font=small_font
    )

    # Draw tagline
    tagline1 = "Train AI to"
    tagline2 = "Master Atari"
    tagline_bbox1 = draw.textbbox((0, 0), tagline1, font=small_font)
    tagline_width1 = tagline_bbox1[2] - tagline_bbox1[0]
    tagline_bbox2 = draw.textbbox((0, 0), tagline2, font=small_font)
    tagline_width2 = tagline_bbox2[2] - tagline_bbox2[0]

    draw.text(
        ((164 - tagline_width1) // 2, 250),
        tagline1,
        fill=LIGHT_GRAY,
        font=small_font
    )
    draw.text(
        ((164 - tagline_width2) // 2, 265),
        tagline2,
        fill=ACCENT_BLUE,
        font=small_font
    )

    # Save as BMP
    output_path = OUTPUT_DIR / "wizard_large.bmp"
    img.save(output_path, "BMP")
    print(f"  ✓ Saved: {output_path}")


def create_wizard_small():
    """Create the small wizard header image (55x55)."""
    print("Creating wizard_small.bmp (55x55)...")

    # Create gradient background
    img = create_gradient(55, 55, DARKER_BLUE, DARK_BLUE)
    draw = ImageDraw.Draw(img)

    # Draw a simple icon - stylized "AI" or game controller
    # Simple geometric design
    draw.rectangle([12, 15, 43, 40], outline=ACCENT_BLUE, width=2)
    draw.ellipse([10, 22, 16, 28], fill=WHITE)  # Left button
    draw.ellipse([39, 22, 45, 28], fill=WHITE)  # Right button

    # Save as BMP
    output_path = OUTPUT_DIR / "wizard_small.bmp"
    img.save(output_path, "BMP")
    print(f"  ✓ Saved: {output_path}")


def create_app_icon():
    """Create application icon (256x256 ICO)."""
    print("Creating app_icon.ico (256x256)...")

    # Create gradient background
    img = create_gradient(256, 256, DARKER_BLUE, DARK_BLUE)
    draw = ImageDraw.Draw(img)

    # Try to use a nice font
    try:
        large_font = ImageFont.truetype("arial.ttf", 60)
        small_font = ImageFont.truetype("arial.ttf", 20)
    except:
        large_font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    # Draw main text
    text = "RM"
    text_bbox = draw.textbbox((0, 0), text, font=large_font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    draw.text(
        ((256 - text_width) // 2, (256 - text_height) // 2 - 20),
        text,
        fill=WHITE,
        font=large_font
    )

    # Draw subtitle
    subtitle = "ML"
    subtitle_bbox = draw.textbbox((0, 0), subtitle, font=small_font)
    subtitle_width = subtitle_bbox[2] - subtitle_bbox[0]

    draw.text(
        ((256 - subtitle_width) // 2, 160),
        subtitle,
        fill=ACCENT_BLUE,
        font=small_font
    )

    # Draw decorative border
    draw.rectangle([20, 20, 236, 236], outline=ACCENT_BLUE, width=3)

    # Save as ICO (multiple sizes for better Windows compatibility)
    output_path = OUTPUT_DIR / "app_icon.ico"

    # Create multiple sizes
    sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    icons = []

    for size in sizes:
        icon = img.resize(size, Image.Resampling.LANCZOS)
        icons.append(icon)

    # Save as ICO with multiple sizes
    icons[0].save(
        output_path,
        format='ICO',
        sizes=[(icon.width, icon.height) for icon in icons],
        append_images=icons[1:]
    )

    print(f"  ✓ Saved: {output_path}")


def create_readme():
    """Create README file with instructions for custom images."""
    readme_content = """# Installer Images

This directory contains images used by the Inno Setup installer.

## Current Images

- **wizard_large.bmp** (164x314) - Large sidebar image shown during installation
- **wizard_small.bmp** (55x55) - Small header icon
- **app_icon.ico** (256x256) - Application icon

## Customization

These are placeholder images generated by `create_installer_images.py`.

To create custom images:

1. **Create your own images** using any graphics editor (Photoshop, GIMP, Figma, etc.)

2. **Image specifications:**
   - wizard_large.bmp: 164x314 pixels, BMP format
   - wizard_small.bmp: 55x55 pixels, BMP format
   - app_icon.ico: 256x256 pixels (or multiple sizes), ICO format

3. **Design tips:**
   - Use colors that match your brand
   - Keep wizard_large simple with text/logo centered
   - wizard_small should be recognizable at small size
   - ICO should include multiple sizes (16x16, 32x32, 48x48, 64x64, 128x128, 256x256)

4. **Replace the files** in this directory with your custom images

5. **Rebuild installer** using `build_all.bat`

## Color Scheme (Current Placeholders)

- Dark Blue: #192130
- Accent Blue: #4287f5
- Light Gray: #c8c8c8
- White: #ffffff

## Regenerating Placeholders

If you want to regenerate the placeholder images:

```bash
python build_scripts/create_installer_images.py
```

This will overwrite existing images in this directory.
"""

    readme_path = OUTPUT_DIR / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)

    print(f"\n  ✓ Created: {readme_path}")


def main():
    """Main function to create all installer images."""
    print("=" * 60)
    print("CREATING INSTALLER IMAGES")
    print("=" * 60)
    print()

    try:
        # Check if PIL is available
        from PIL import Image
        print("✓ PIL (Pillow) is available\n")
    except ImportError:
        print("❌ ERROR: Pillow not installed")
        print("\nInstall with: pip install Pillow")
        sys.exit(1)

    # Create images
    create_wizard_large()
    create_wizard_small()
    create_app_icon()
    create_readme()

    print()
    print("=" * 60)
    print("✅ ALL IMAGES CREATED SUCCESSFULLY!")
    print("=" * 60)
    print()
    print(f"Images saved to: {OUTPUT_DIR}")
    print()
    print("These are placeholder images. For a professional look:")
    print("1. Create custom images using the specs in README.md")
    print("2. Replace the files in installer_images/")
    print("3. Rebuild the installer")
    print()


if __name__ == '__main__':
    main()
