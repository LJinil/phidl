from phidl import Device

from gds_library.components.basics import rectangle
from gds_library.helpers import validate_positive


def poling_electrode(
    poling_length: float = 1000.0,
    teeth_length: float = 50.0,
    teeth_period: float = 5.0,
    teeth_gap: float = 20.0,
    teeth_duty_cycle: float = 0.35,
    pad_width: float = 100.0,
    direction_marker_width: float = 10.0,
    layer: int = 1,
) -> Device:
    """Generate a poling electrode structure.

    Args:
        poling_length: Total length of the poling region in microns.
        teeth_length: Length of each tooth in microns.
        teeth_period: Period of the teeth in microns.
        teeth_gap: Gap between the top and bottom rows of teeth in microns.
        teeth_duty_cycle: Duty cycle of the teeth (0 < duty cycle <= 1).
        pad_width: Width of the square electrode pads in microns.
        direction_marker_width: Width of the direction marker in microns.
        layer: GDS layer for the geometry.

    Returns:
        A PHIDL Device object representing the poling electrode.
    """

    for value, name in [
        (poling_length, "Poling length"),
        (teeth_length, "Teeth length"),
        (teeth_period, "Teeth period"),
        (teeth_gap, "Teeth gap"),
        (pad_width, "Pad width"),
        (direction_marker_width, "Direction marker width"),
    ]:
        validate_positive(value, name)

    D = Device("poling electrode")
    teeth_width = teeth_period * teeth_duty_cycle
    teeth = rectangle(
        width=teeth_length,
        length=teeth_width,
        layer=layer,
        port=False,
    )

    teeth_connection = rectangle(
        width=teeth_length / 10,
        length=poling_length + pad_width,
        layer=layer,
        port=False,
    )
    pad = rectangle(width=pad_width, length=pad_width, layer=layer, port=False)
    direction_marker = rectangle(
        width=direction_marker_width,
        length=direction_marker_width,
        layer=layer,
        port=False,
    )

    N = int(poling_length / teeth_period + 0.5)
    for i in range(N):
        D.add_ref(teeth).move([i * teeth_period, 0])
        D.add_ref(teeth).move([i * teeth_period, teeth_length + teeth_gap])

    D.add_ref(teeth_connection).move([-pad_width / 2, -teeth_length / 10])
    D.add_ref(teeth_connection).move([-pad_width / 2, 2 * teeth_length + teeth_gap])

    D.add_ref(pad).move([-pad_width, -pad_width - teeth_length / 10])
    D.add_ref(pad).move([poling_length, -pad_width - teeth_length / 10])
    D.add_ref(pad).move([-pad_width, 2.1 * teeth_length + teeth_gap])
    D.add_ref(pad).move([poling_length, 2.1 * teeth_length + teeth_gap])

    D.add_ref(direction_marker).move(
        [
            poling_length / 2 - direction_marker_width / 2,
            2.1 * teeth_length + teeth_gap + pad_width / 2 - direction_marker_width / 2,
        ]
    )

    return D
