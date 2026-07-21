"""
Generates a synthetic placeholder image so this demo doesn't depend on a
real patient photo. It's a rough stand-in for a skin lesion photo: an
irregular blob on a skin-tone background, with uneven lighting and no
scale reference - on purpose, so the "image quality" part of the demo
has something real to flag.

Run once: python generate_sample_image.py
Swap sample_image.png for a real photo whenever you have one to test with.
"""

from PIL import Image, ImageDraw

WIDTH, HEIGHT = 400, 400

img = Image.new("RGB", (WIDTH, HEIGHT), color=(224, 172, 135))  # skin tone
draw = ImageDraw.Draw(img)

# Uneven lighting: a soft darker gradient across one corner.
for y in range(HEIGHT):
    shade = int(40 * (y / HEIGHT))
    draw.line([(0, y), (WIDTH, y)], fill=(224 - shade, 172 - shade, 135 - shade))

# Irregular "lesion" blob, off-center, no scale reference next to it.
draw.polygon(
    [(180, 150), (230, 140), (260, 180), (250, 230), (200, 250), (160, 210)],
    fill=(120, 70, 50),
)

img.save("sample_image.png")
print("Saved sample_image.png")
