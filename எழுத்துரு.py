import matplotlib

matplotlib.use("module://mplcairo.macosx")
from matplotlib import font_manager

எழுத்துருகள் = font_manager.findSystemFonts(fontpaths=None, fontext="ttf")
try:
    எழுத்துரு_பெயர் = next(எழு for எழு in எழுத்துருகள் if "NotoSansTamil-" in எழு)
except StopIteration:
    raise OSError(
        "NotoSansTamil என்று தமிழ் எழுத்துரு நிறுவல் செய்யவும்: "
        "https://fonts.google.com/noto/specimen/Noto+Sans+Tamil?noto.query=tamil"
    )
எழுத்துரு = font_manager.FontProperties(fname=எழுத்துரு_பெயர்)


def அளவுடன்_எழுத்துரு(அளவு):
    return font_manager.FontProperties(size=அளவு, fname=எழுத்துரு_பெயர்)
