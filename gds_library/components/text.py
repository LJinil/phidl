from phidl import Device
import phidl.geometry as pg


def text(
    text: str = "abcd",
    size: int = 10,
    justify: str = "left",
    layer: int = 0,
    font: str = "DEPLOF",
) -> Device:
    """Generate a text object in GDS format.

    Args:
        text: The string to be displayed.
        size: Font size of the text.
        justify: Text alignment ("left", "center", "right").
        layer: GDS layer for the text.
        font: Font face to use. Default DEPLOF does not require additional libraries, otherwise freetype will be used to load fonts. Font can be given either by name (e.g. “Times New Roman”), or by file path. OTF or TTF fonts are supported.

    Returns:
        A PHIDL Device object containing the text.
    """
    
    D = pg.text(
        text=text,
        size=size,
        justify=justify,
        layer=layer,
        font=font,
    )

    return D
