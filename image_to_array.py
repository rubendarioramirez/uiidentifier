from PIL import Image

img = Image.open("red_square.png")
width, height = img.size
pixels = list(img.getdata())

# Reshape into a 2D array of (R, G, B) tuples
pixel_array = [pixels[i * width:(i + 1) * width] for i in range(height)]

# Print as int array
print(f"Image size: {width}x{height}\n")
for row in pixel_array:
    int_row = [f"({r},{g},{b})" for r, g, b in row]
    print(" ".join(int_row))
