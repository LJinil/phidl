from phidl import Device
import phidl.geometry as pg


def rectangle(
    width: float = 1.0,
    length: float = 10.0,
    port: bool = True,
    layer: int = 0,
) -> Device:
    """Generate a rectangular geometry.

    Args:
        width: Width of the rectangle in microns.
        length: Length of the rectangle in microns.
        layer: GDS layer for the geometry.

    Returns:
        A PHIDL Device object representing the rectangle.
    """

    D = pg.rectangle(size=(length, width), layer=layer)

    if port == True:
        D.add_port(
            name=1,
            midpoint=[0, width / 2],
            width=width,
            orientation=180,
        )

        D.add_port(
            name=2,
            midpoint=[length, width / 2],
            width=width,
            orientation=0,
        )

    return D


def cross(
    width: float = 0.5,
    length: float = 10.0,
    layer: int = 0,
) -> Device:
    """Generate a cross geometry.

    Args:
        width: Width of the cross in microns.
        length: Length of the cross in microns.
        layer: GDS layer for the geometry.

    Returns:
        A PHIDL Device object representing the cross.
    """

    D = pg.cross(
        length=length,
        width=width,
        layer=layer,
    )

    return D


def L(
    width: float = 10,
    size: tuple[float, float] = (10, 20),
    layer: int = 0,
) -> Device:
    """Generate a L-shape geometry.

    Args:
        width: Width of the L-shape in microns.
        size: Length and height of L-shape in mircons.
        layer: GDS layer for the geometry.

    Returns:
        A PHIDL Device object representing the L-shape geometry.
    """

    D = pg.L(
        width=width,
        size=size,
        layer=layer,
    )

    return D


def C(
    width: float = 10,
    size: tuple[float, float] = (10, 20),
    layer: int = 0,
) -> Device:
    """Generate a C-shape geometry.

    Args:
        width: Width of the C-shape.
        size: Length and height of C-shape.
        layer: GDS layer for the geometry.

    Returns:
        A PHIDL Device object representing the C-shape geometry.
    """

    D = pg.C(
        width=width,
        size=size,
        layer=layer,
    )

    return D
