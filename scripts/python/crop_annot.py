from PIL import Image
BOX, ORIG = (0, 26, 586, 614), (586, 640)  # visible crop of 000000000285.jpg
im = Image.open("figs_v2/annot.png")
sx, sy = im.width / ORIG[0], im.height / ORIG[1]
box = (round(BOX[0] * sx), round(BOX[1] * sy), round(BOX[2] * sx), round(BOX[3] * sy))
im.crop(box).save("figs_v2/annot_crop.png")
print("annot.png", im.size, "mode", im.mode, "-> box", box, "crop", im.crop(box).size)
