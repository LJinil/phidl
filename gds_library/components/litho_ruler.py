from phidl import Device
import phidl.geometry as pg


def litho_ruler(
    height: float = 2,
    width: float = 0.5,
    spacing: float = 2.0,
    scale: tuple[float, ...] = (3, 1, 1, 1, 1, 2, 1, 1, 1, 1),
    num_marks: int = 21,
    layer: int = 0,
) -> Device:
    """Ruler structure for lithographic measurement.

    Includes marks of varying scales to allow for easy reading by eye.

    based on phidl.geometry

    Args:
        height: Height of the ruling marks in microns.
        width: Width of the ruling marks in microns.
        spacing: Center-to-center spacing of the ruling marks in microns.
        scale: Height scale pattern of marks.
        num_marks: Total number of marks to generate.
        layer: GDS layer for the geometry.

    Returns:
        A PHIDL Device object representing the lithographic ruler.
    """

    D = Device("litho ruler")
    for n in range(num_marks):
        h = height * scale[n % len(scale)]
        D << pg.rectangle(size=(width, h), layer=layer)

    D.distribute(direction="x", spacing=spacing, separation=False, edge="x")
    D.align(alignment="ymin")
    D.flatten()

    return D
