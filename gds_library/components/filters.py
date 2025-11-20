from phidl import Device

from gds_library.components.tapers import (
    taper,
    ramp,
)
from gds_library.components.bends import (
    bend_s_sine,
    bend_s_circular,
    bend_straight_bend_circular,
    bend_s_euler,
    bend_straight_bend_euler,
)
from gds_library.helpers import (
    parse_value,
    validate_positive,
    add_ports_to_device,
    rename_ports,
)

def polarization_splitter_rotator(
    rotator_widths: float | tuple[float, float] = (1.2, 1.8),
    rotator_length: float = 700.0,
    connect_length: float = 200.0,
    through_port_widths: float | tuple[float, float] = (2.4, 2.0),
    cross_port_widths: float | tuple[float, float] = (0.8, 1.0),
    coupler_length: float = 2000.0,
    gap: float = 0.4,
    cross_port_terminate: bool = True,
    cross_port_terminate_width: float = 0.4,
    cross_port_terminate_length: float = 25,
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


    rotator_width_left, rotator_width_right = parse_value(rotator_widths, "Width", (1, 2))
    through_port_width_left, through_port_width_right = parse_value(through_port_widths, "Width", (1, 2))
    cross_port_width_left, cross_port_width_right = parse_value(cross_port_widths, "Width", (1, 2))

    for value, name in [
        (rotator_width_left, "Width"),
        (rotator_width_right, "Width"),
        (through_port_width_left, "Width"),
        (through_port_width_right, "Width"),
        (cross_port_width_left, "Width"),
        (cross_port_width_right, "Width"),
        (rotator_length, "Length"),
        (connect_length, "Length"),
        (coupler_length, "Length"),
        (gap, "Gap"),
    ]:
        validate_positive(value, name)

    if cross_port_terminate:
        for value, name in [
            (cross_port_terminate_width, "Width"),
            (cross_port_terminate_length, "Length"),
        ]:
            validate_positive(value, name)  

    D = Device("polarization_splitter_rotator")

    rotator = taper(
        width1 = rotator_width_left,
        width2 = rotator_width_right,
        length = rotator_length,
        layer = layer,
    )
    connector = taper(
        width1 = rotator_width_right,
        width2 = through_port_width_left,
        length = connect_length,
        layer = layer,
    )

    splitter_th = ramp(
        width1 = through_port_width_left,
        width2 = through_port_width_right,
        length = coupler_length, 
        layer = layer,
    )
    splitter_cr = ramp(
        width1 = cross_port_width_right,
        width2 = cross_port_width_left,
        length = coupler_length,
        layer = layer,
    )   
    splitter_cr_end = taper(
        width1 = cross_port_width_left,
        width2 = cross_port_terminate_width,
        length = cross_port_terminate_length,
        layer = layer,
    )

    rotator_ref = D.add_ref(rotator)
    connector_ref = D.add_ref(connector)
    splitter_th_ref = D.add_ref(splitter_th)
    splitter_cr_ref = D.add_ref(splitter_cr)
    splitter_cr_end_ref = D.add_ref(splitter_cr_end)

    connector_ref.connect(port=1, destination=rotator_ref.ports[2])
    splitter_th_ref.connect(port=1, destination=connector_ref.ports[2])

    splitter_cr_ref.rotate(180)
    splitter_cr_ref.x = splitter_th_ref.x
    splitter_cr_ref.ymax = splitter_th_ref.ymin - gap
    splitter_cr_end_ref.connect(port=1, destination=splitter_cr_ref.ports[2])

    splitter_cr_ref.ports[1].name = 3
    D.add_port(rotator_ref.ports[1])
    D.add_port(splitter_th_ref.ports[2])
    D.add_port(splitter_cr_ref.ports[1])

    D.flatten()

    return D
