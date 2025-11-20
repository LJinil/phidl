import numpy as np
from phidl import Device

from phidl import Device

from gds_library.components.basics import rectangle
from gds_library.components.tapers import taper
from gds_library.helpers import (
    parse_value,
    validate_positive,
    add_ports_to_device,
    rename_ports,
)

def waveguide_crossing(
    input_length: float = 3.0,
    taper_widths: float | tuple[float, float] = (0.7, 2.4),
    taper_length: float = 3.0,
    connect_length: float = 5.0,
    layer: int = 0,
) -> Device:
    """Full directional coupler.

    Args:
        coupler_length: Length of the straight coupler section in microns.
        coupler_widths: Widths of the bottom and top coupler waveguides (scalar or tuple for bottom/top)
        gap: Gap between the waveguides in microns.
        bends: Type of bends ("sine", "circular", or "euler").
        ends_length: Widths of the waveguides at the ends in microns. (scalar or tuple for left/right)
        ends_left_widths: Widths of the waveguides at the left ends in microns. (scalar or tuple for bottom/top)
        ends_right_widths: Widths of the waveguides at the right ends in microns. (scalar or tuple for bottom/top)
        ends_left_dists: Distances from the gap center to the left ends in microns. (scalar or tuple for bottom/top)
        ends_right_dists: Distances from the gap center to the right ends in microns. (scalar or tuple for bottom/top)
        num_pts: Number of points for smoothness in the bends.
        layer: GDS layer for the geometry.

    Returns:
        A PHIDL Device object representing the coupler ends.
    """

    taper_width_left, taper_width_right = parse_value(taper_widths, "Width", (1, 2))

    for value, name in [
        (taper_width_left, "Width"),
        (taper_width_right, "Width"),
        (input_length, "Length"),
        (taper_length, "Length"),
        (connect_length, "Length"),
    ]:
        validate_positive(value, name)

    D = Device("waveguide_crossing")

    input_wg = rectangle(
        width = taper_width_left,
        length = input_length,
        layer = layer,
    )
    input_taper = taper(
        width1 = taper_width_left,
        width2 = taper_width_right,
        length = taper_length,
        layer = layer,
    )
    connector = rectangle(
        width = taper_width_right,
        length = connect_length,
        layer = layer,
    )
    center = rectangle(
        width = taper_width_right,
        length = taper_width_right,
        layer = layer,
        port = False,
    )

    center_ref = D.add_ref(center)

    A = Device("arms")
    input_wg_ref = A.add_ref(input_wg)
    input_taper_ref = A.add_ref(input_taper)
    connector_ref = A.add_ref(connector)

    input_taper_ref.connect(port=1, destination=input_wg_ref.ports[2])
    connector_ref.connect(port=1, destination=input_taper_ref.ports[2])
    A.add_port(input_wg_ref.ports[1])

    A1 = D.add_ref(A)
    A2 = D.add_ref(A)
    A3 = D.add_ref(A)
    A4 = D.add_ref(A)

    A2.rotate(90)
    A2.ports[1].name = 2
    A3.rotate(180)
    A3.ports[1].name = 3
    A4.rotate(270)
    A4.ports[1].name = 4

    A1.xmax = center_ref.xmin
    A1.y = center_ref.y
    A2.x = center_ref.x
    A2.ymax = center_ref.ymin
    A3.xmin = center_ref.xmax
    A3.y = center_ref.y
    A4.x = center_ref.x
    A4.ymin = center_ref.ymax

    D.add_port(A1.ports[1])
    D.add_port(A2.ports[1])
    D.add_port(A3.ports[1])
    D.add_port(A4.ports[1])

    D.flatten()

    return D
