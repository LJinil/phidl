import numpy as np
from phidl import Device

from gds_library.components.basics import rectangle
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


def _create_bends(bend_type, length, ends_dist, width_start, width_end, num_pts, layer):
    if bend_type == "sine":
        return bend_s_sine(
            length=length,
            height=(-1 if ends_dist < 0 else 1) * np.abs(ends_dist),
            width=(width_start, width_end),
            num_pts=num_pts,
            layer=layer,
        )

    else:
        radius = (ends_dist**2 + length**2) / (4 * np.abs(ends_dist))
        angle = np.arcsin(
            2 * np.abs(ends_dist) * length / (ends_dist**2 + length**2)
        ) * (180 / np.pi)
        if np.abs(ends_dist) > length:
            bend_fn = (
                bend_straight_bend_circular
                if bend_type == "circular"
                else bend_straight_bend_euler
            )
            return bend_fn(
                radius=length / 2,
                width=(width_start, width_end),
                angle=(-90 if ends_dist < 0 else 90),
                length=np.abs(ends_dist) - length,
                num_pts=num_pts,
                layer=layer,
            )
        else:
            bend_fn = bend_s_circular if bend_type == "circular" else bend_s_euler
            return bend_fn(
                radius=radius,
                width=(width_start, width_end),
                angle=(-angle if ends_dist < 0 else angle),
                num_pts=num_pts,
                layer=layer,
            )


def coupler_straight(
    length: float = 30.0,
    widths: float | tuple[float, float] = 1.0,
    gap: float = 0.5,
    layer: int = 0,
) -> Device:
    """Generate a straight coupler with two parallel waveguides.

    Args:
        length: Length of the coupler in microns.
        widths: Tuple specifying the widths of the bottom and top waveguides (scalar or tuple for bottom/top).
        gap: Gap between the waveguides in microns.
        layer: GDS layer for the geometry.

    Returns:
        A PHIDL Device object representing the straight coupler.
    """

    width_bottom, width_top = parse_value(widths, "Width", (1, 2))

    for value, name in [
        (length, "Length"),
        (width_bottom, "Width"),
        (width_top, "Width"),
        (gap, "Gap"),
    ]:
        validate_positive(value, name)

    D = Device("coupler_straight")

    wg_b = rectangle(
        width=width_bottom,
        length=length,
        layer=layer,
    )
    wg_t = rectangle(
        width=width_top,
        length=length,
        layer=layer,
    )
    wg_b.x = wg_t.x
    wg_b.ymax = wg_t.ymin - gap

    rename_ports(wg_t, {2: 3, 1: 2})
    rename_ports(wg_b, {2: 4})

    D << wg_b
    D << wg_t

    add_ports_to_device(D, wg_t)
    add_ports_to_device(D, wg_b)

    D.flatten()

    return D


def coupler_ends(
    length: float = 20.0,
    coupler_widths: float | tuple[float, float] = 1.0,
    gap: float = 0.5,
    bends: str = "sine",
    ends_widths: float | tuple[float, float] = 1.0,
    ends_dists: float | tuple[float, float] = 5.0,
    num_pts: int = 120,
    layer: int = 0,
) -> Device:
    """Generate coupler ends with flexible bend options.

    Args:
        length: Length of the straight coupler section in microns.
        coupler_widths: Widths of the bottom and top coupler waveguides (scalar or tuple for bottom/top)
        gap: Gap between the waveguides in microns.
        bends: Type of bends ("sine", "circular", or "euler").
        ends_widths: Widths of the waveguides at the ends in microns. (scalar or tuple for bottom/top)
        ends_dists: Distances from the gap center to the ends in microns. (scalar or tuple for bottom/top)
        num_pts: Number of points for smoothness in the bends.
        layer: GDS layer for the geometry.

    Returns:
        A PHIDL Device object representing the coupler ends.
    """
    if bends not in ["sine", "circular", "euler"]:
        raise ValueError(
            f"Parameter 'bends' must be one of 'sine', 'circular', and 'euler'. Got {(bends)}."
        )

    coupler_width_bottom, coupler_width_top = parse_value(
        coupler_widths, "Coupler width", (1, 2)
    )
    ends_width_bottom, ends_width_top = parse_value(ends_widths, "Ends width", (1, 2))
    ends_dist_bottom, ends_dist_top = parse_value(ends_dists, "Ends distance", (1, 2))

    for value, name in [
        (length, "Length"),
        (gap, "Gap"),
        (coupler_width_bottom, "Width"),
        (coupler_width_top, "Width"),
        (ends_width_bottom, "Width"),
        (ends_width_top, "Width"),
        (ends_dist_bottom, "Distance"),
        (ends_dist_top, "Distance"),
        (num_pts, "Number of points"),
    ]:
        validate_positive(value, name)

    D = Device("coupler_ends")

    wge_b = _create_bends(
        bend_type=bends,
        length=length,
        ends_dist=-ends_dist_bottom,
        width_start=coupler_width_bottom,
        width_end=ends_width_bottom,
        num_pts=num_pts,
        layer=layer,
    )
    wge_t = _create_bends(
        bend_type=bends,
        length=length,
        ends_dist=ends_dist_top,
        width_start=coupler_width_top,
        width_end=ends_width_top,
        num_pts=num_pts,
        layer=layer,
    )

    wge_t.ymin = wge_b.ymax + gap

    rename_ports(wge_t, {2: 3, 1: 2})
    rename_ports(wge_b, {2: 4})

    D.add_ref(wge_b)
    D.add_ref(wge_t)

    add_ports_to_device(D, wge_t)
    add_ports_to_device(D, wge_b)

    D.flatten()

    return D


def coupler_full(
    coupler_length: float = 20.0,
    coupler_widths: float | tuple[float, float] = 1.0,
    gap: float = 0.5,
    bends: str = "sine",
    ends_length: float | tuple[float, float] = 10.0,
    ends_left_widths: float | tuple[float, float] = 1.0,
    ends_right_widths: float | tuple[float, float] = 1.0,
    ends_left_dists: float | tuple[float, float] = 5.0,
    ends_right_dists: float | tuple[float, float] = 5.0,
    num_pts: int = 120,
    layer: int = 0,
):
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
    if bends not in ["sine", "circular", "euler"]:
        raise ValueError(
            f"Parameter 'bends' must be one of 'sine', 'circular', and 'euler'. Got {(bends)}."
        )

    ends_left_length, ends_right_length = parse_value(
        ends_length, "Ends length", (1, 2)
    )

    D = Device("coupler_full")

    wgs = coupler_straight(
        length=coupler_length,
        widths=coupler_widths,
        gap=gap,
        layer=layer,
    )

    wge_l = coupler_ends(
        length=ends_left_length,
        coupler_widths=coupler_widths,
        gap=gap,
        bends=bends,
        ends_widths=ends_left_widths,
        ends_dists=ends_left_dists,
        num_pts=num_pts,
        layer=layer,
    )

    wge_r = coupler_ends(
        length=ends_right_length,
        coupler_widths=coupler_widths,
        gap=gap,
        bends=bends,
        ends_widths=ends_right_widths,
        ends_dists=ends_right_dists,
        num_pts=num_pts,
        layer=layer,
    )

    wgs_ref = D.add_ref(wgs)
    wge_l_ref = D.add_ref(wge_l)
    wge_r_ref = D.add_ref(wge_r)

    wge_l_ref.connect(port=2, destination=wgs_ref.ports[1])
    wge_r_ref.connect(port=1, destination=wgs_ref.ports[4])

    wge_l_ref.ports[1].name = 11
    wge_l_ref.ports[2].name = 12
    wge_l_ref.ports[3].name = 2
    wge_l_ref.ports[4].name = 1
    D.add_port(wge_l_ref.ports[3])
    D.add_port(wge_l_ref.ports[4])
    D.add_port(wge_r_ref.ports[3])
    D.add_port(wge_r_ref.ports[4])

    D.flatten()

    return D
