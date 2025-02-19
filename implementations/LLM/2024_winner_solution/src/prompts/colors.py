from pydantic import BaseModel


class ColorData(BaseModel):
    name: str
    hex: str
    rgb: tuple[int, int, int]


color_map: dict[int, ColorData] = {
    0: ColorData(name="black", hex="#000000", rgb=(0, 0, 0)),
    1: ColorData(name="blue", hex="#0074D9", rgb=(0, 40, 230)),
    2: ColorData(name="red", hex="#FF4136", rgb=(230, 20, 20)),
    3: ColorData(name="green", hex="#2ECC40", rgb=(46, 204, 64)),
    4: ColorData(name="yellow", hex="#FFDC00", rgb=(255, 255, 0)),
    5: ColorData(name="grey", hex="#AAAAAA", rgb=(170, 170, 170)),
    6: ColorData(name="pink", hex="#F012BE", rgb=(255, 0, 195)),
    7: ColorData(name="orange", hex="#FF851B", rgb=(255, 133, 27)),
    8: ColorData(
        name="purple", hex="#9d00ff", rgb=(157, 0, 255)
    ),  # is sky blue on normal visuals
    9: ColorData(name="brown", hex="#870C25", rgb=(139, 69, 19)),
}

__all__ = ["color_map", "ColorData"]
