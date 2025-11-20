from phidl import Device
import phidl.geometry as pg


def taper(
    length: float = 10,
    width1: float = 6,
    width2: float = 4,
    layer: int = 0,
) -> Device:
    """Generate a taper geometry.

    Args:
        length: Length of the taper in microns.
        width1: Width at the start (left/west port) in microns.
        width2: Width at the end (right/east port) in microns.
        layer: GDS layer for the geometry.

    Returns:
        A PHIDL Device object representing the taper geometry.
    """

    D = pg.taper(
        length = length,
        width1 = width1,
        width2 = width2,
        layer = layer,
    )

    return D

def ramp(
    length: float = 10,
    width1: float = 6,
    width2: float = 4,
    layer: int = 0,
) -> Device:
    """Generate a ramp geometry.

    Args:
        length: Length of the ramp in microns.
        width1: Width at the start (left/west port) in microns.
        width2: Width at the end (right/east port) in microns.
        layer: GDS layer for the geometry.

    Returns:
        A PHIDL Device object representing the ramp geometry.
    """

    D = pg.ramp(
        length = length,
        width1 = width1,
        width2 = width2,
        layer = layer,
    )

    return D
