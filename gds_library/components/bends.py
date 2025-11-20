import numpy as np

import gdspy
from phidl import Device, Path
import phidl.path as pp

from gds_library.helpers import validate_positive, parse_value, create_cross_section


def bend_circular(
    radius: float = 30.0,
    width: float | tuple[float, float] = 1.0,
    width_type: str = "linear",
    angle: float = 90.0,
    num_pts: int = 720,
    layer: int = 0,
) -> Device:
    """Generate a circular bend waveguide.

    Args:
        radius: Radius of bend in microns.
        width: Widths of bend in microns. (scalar or tuple for start/end)
        width_type: Width type of bend waveguide.
        angle: Angle of bend waveguide in degrees.
        num_pts: Number of points for smoothness.
        layer: GDS layer for the geometry.

    Returns:
        A PHIDL Device object representing the circular bend.
    """

    width_start, width_end = parse_value(width, "Width", (1, 2))

    for value, name in [
        (radius, "Radius"),
        (width_start, "Width"),
        (width_end, "Width"),
        (num_pts, "Number of points"),
    ]:
        validate_positive(value, name)

    trans = pp.transition(
        cross_section1=create_cross_section(
            width=width_start, layer=layer, ports=(1, 2)
        ),
        cross_section2=create_cross_section(width=width_end, layer=layer, ports=(1, 2)),
        width_type=width_type,
    )

    D = pp.arc(
        radius=radius,
        angle=angle,
        num_pts=num_pts,
    ).extrude(
        width=trans,
        simplify=0.0003,
    )

    D.name = "bend_circular"

    return D


def bend_euler(
    radius: float = 30.0,
    width: float | tuple[float, float] = 1.0,
    width_type: str = "linear",
    angle: float = 90.0,
    num_pts: int = 720,
    use_eff: bool = True,
    layer: int = 0,
) -> Device:
    """Generate a Euler bend waveguide.

    Args:
        radius: Radius of bend in microns.
        width: Widths of bend in microns. (scalar or tuple for start/end)
        width_type: Width type of bend waveguide.
        angle: Angle of bend waveguide in degrees.
        num_pts: Number of points for smoothness.
        use_eff: use effective radius (default = True)
        layer: GDS layer for the geometry.

    Returns:
        A PHIDL Device object representing the Euler bend.
    """

    width_start, width_end = parse_value(width, "Width", (1, 2))

    for value, name in [
        (radius, "Radius"),
        (width_start, "Width"),
        (width_end, "Width"),
        (num_pts, "Number of points"),
    ]:
        validate_positive(value, name)

    trans = pp.transition(
        cross_section1=create_cross_section(
            width=width_start, layer=layer, ports=(1, 2)
        ),
        cross_section2=create_cross_section(width=width_end, layer=layer, ports=(1, 2)),
        width_type=width_type,
    )

    D = pp.euler(
        radius=radius,
        angle=angle,
        num_pts=num_pts,
        use_eff=use_eff,
    ).extrude(
        width=trans,
        simplify=0.0003,
    )

    D.name = "bend_euler"

    return D


def bend_s_circular(
    radius: float | tuple[float, float] = 30.0,
    width: float | tuple[float, float] = 1.0,
    width_type: str = "linear",
    angle: float | tuple[float, float] = 90.0,
    num_pts: int = 720,
    layer: int = 0,
) -> Device:
    """Generate a circular s-shape bended waveguide.

    Args:
        radius: Radius of the S-bend in microns (scalar or tuple for left/right).
        width: Width of the waveguide in microns (scalar or tuple for start/end).
        width_type: Width type of circular bend waveguide.
        angle: Angle of each bend in degrees (scalar or tuple for left/right).
        num_pts: Number of points for smoothness.
        layer: GDS layer for the geometry.

    Returns:
        A PHIDL Device object representing the circular S-bend.
    """

    radius_start, radius_end = parse_value(radius, "Radius", (1, 2))
    width_start, width_end = parse_value(width, "Width", (1, 2))
    angle_start, angle_end = parse_value(angle, "Angle", (1, 2))

    for value, name in [
        (radius_start, "Radius"),
        (radius_end, "Radius"),
        (width_start, "Width"),
        (width_end, "Width"),
        (num_pts, "Number of points"),
    ]:
        validate_positive(value, name)

    trans = pp.transition(
        cross_section1=create_cross_section(
            width=width_start, layer=layer, ports=(1, 2)
        ),
        cross_section2=create_cross_section(width=width_end, layer=layer, ports=(1, 2)),
        width_type=width_type,
    )

    P = Path()
    P.append(
        [
            pp.arc(
                radius=radius_start,
                angle=angle_start,
                num_pts=num_pts,
            ),
            pp.arc(
                radius=radius_end,
                angle=-angle_end,
                num_pts=num_pts,
            ),
        ]
    )
    D = P.extrude(
        width=trans,
        simplify=0.0003,
    )

    D.name = "bend_s_circular"

    return D


def bend_s_euler(
    radius: float | tuple[float, float] = 30.0,
    width: float | tuple[float, float] = 1.0,
    width_type: str = "linear",
    angle: float | tuple[float, float] = 90.0,
    num_pts: int = 720,
    use_eff: bool = True,
    layer: int = 0,
) -> Device:
    """Generate a Euler s-shape bended waveguide.

    Args:
        radius: Radius of the S-bend in microns (scalar or tuple for left/right).
        width: Width of the waveguide in microns (scalar or tuple for start/end).
        width_type: Width type of bend waveguide.
        angle: Angle of each bend in degrees (scalar or tuple for left/right).
        num_pts: Number of points for smoothness.
        use_eff: use effective radius (default = True)
        layer: GDS layer for the geometry.

    Returns:
        A PHIDL Device object representing the Euler S-bend.
    """

    radius_start, radius_end = parse_value(radius, "Radius", (1, 2))
    width_start, width_end = parse_value(width, "Width", (1, 2))
    angle_start, angle_end = parse_value(angle, "Angle", (1, 2))

    for value, name in [
        (radius_start, "Radius"),
        (radius_end, "Radius"),
        (width_start, "Width"),
        (width_end, "Width"),
        (num_pts, "Number of points"),
    ]:
        validate_positive(value, name)

    trans = pp.transition(
        cross_section1=create_cross_section(
            width=width_start, layer=layer, ports=(1, 2)
        ),
        cross_section2=create_cross_section(width=width_end, layer=layer, ports=(1, 2)),
        width_type=width_type,
    )

    P = Path()
    P.append(
        [
            pp.euler(
                radius=radius_start,
                angle=angle_start,
                num_pts=num_pts,
                use_eff=use_eff,
            ),
            pp.euler(
                radius=radius_end,
                angle=-angle_end,
                num_pts=num_pts,
                use_eff=use_eff,
            ),
        ]
    )
    D = P.extrude(
        width=trans,
        simplify=0.0003,
    )

    D.name = "bend_s_euler"

    return D


def bend_s_sine(
    length: float = 30.0,
    height: float = 10.0,
    width: float | tuple[float, float] = 1.0,
    num_pts: int = 720,
    layer: int = 0,
) -> Device:
    """Generate a sine-shape bended waveguide.

    Args:
        length: Length of the S-bend in microns.
        height: Height of the S-bend in microns.
        width: Width of the waveguide in microns (scalar or tuple for start/end).
        num_pts: Number of points for smoothness.
        layer: GDS layer for the geometry.

    Returns:
        A PHIDL Device object representing the sine-shaped S-bend.
    """

    width_start, width_end = parse_value(width, "Width", (1, 2))

    for value, name in [
        (width_start, "Width"),
        (width_end, "Width"),
        (num_pts, "Number of points"),
    ]:
        validate_positive(value, name)

    curve_fun = lambda t: [length * t, height * (1 - np.cos(np.pi * t)) / 2]
    curve_deriv_fun = lambda t: [
        length,
        height * (np.sin(np.pi * t) * np.pi) / 2,
    ]
    width_fun = (
        lambda t: (width_end - width_start) * (1 - np.cos(np.pi * t)) / 2 + width_end
    )

    route_path = gdspy.Path(width=width_start, initial_point=(0, 0))
    route_path.parametric(
        curve_fun,
        curve_deriv_fun,
        number_of_evaluations=num_pts,
        max_points=199,
        final_width=width_fun,
        final_distance=None,
    )
    route_path_polygons = route_path.polygons

    D = Device()
    D.add_polygon(route_path_polygons, layer=layer)
    D.add_port(name=1, midpoint=(0, 0), width=width_start, orientation=180)
    D.add_port(
        name=2,
        midpoint=[length, height],
        width=width_end,
        orientation=0,
    )
    D.info["length"] = route_path.length
    D.name = "bend_s_sine"

    return D


def bend_straight_bend_circular(
    radius: float | tuple[float, float] = 30.0,
    width: float | tuple[float, float] = 1.0,
    width_type: str = "linear",
    angle: float | tuple[float, float] = 90.0,
    length: float = 10.0,
    num_pts: int = 720,
    layer: int = 0,
) -> Device:
    """Generate a circular s-shape bended waveguide connected with straight waveguide.

    Args:
        radius: Radius of the S-bend in microns (scalar or tuple for left/right).
        width: Width of the waveguide in microns (scalar or tuple for start/end).
        width_type: Width type of bend waveguide.
        angle: Angle of each bend in degrees (scalar or tuple for left/right).
        num_pts: Number of points for smoothness.
        layer: GDS layer for the geometry.

    Returns:
        A PHIDL Device object.
    """

    radius_start, radius_end = parse_value(radius, "Radius", (1, 2))
    width_start, width_end = parse_value(width, "Width", (1, 2))
    angle_start, angle_end = parse_value(angle, "Angle", (1, 2))

    for value, name in [
        (radius_start, "Radius"),
        (radius_end, "Radius"),
        (width_start, "Width"),
        (width_end, "Width"),
        (num_pts, "Number of points"),
    ]:
        validate_positive(value, name)

    trans = pp.transition(
        cross_section1=create_cross_section(
            width=width_start, layer=layer, ports=(1, 2)
        ),
        cross_section2=create_cross_section(width=width_end, layer=layer, ports=(1, 2)),
        width_type=width_type,
    )

    P = Path()
    P.append(
        [
            pp.arc(
                radius=radius_start,
                angle=angle_start,
                num_pts=num_pts,
            ),
            pp.straight(
                length=length,
                num_pts=2,
            ),
            pp.arc(
                radius=radius_end,
                angle=-angle_end,
                num_pts=num_pts,
            ),
        ]
    )
    D = P.extrude(
        width=trans,
        simplify=0.0003,
    )

    D.name = "bend_straight_bend_circular"

    return D


def bend_straight_bend_euler(
    radius: float | tuple[float, float] = 30.0,
    width: float | tuple[float, float] = 1.0,
    width_type: str = "linear",
    angle: float | tuple[float, float] = 90.0,
    length: float = 10.0,
    num_pts: int = 720,
    use_eff: bool = True,
    layer: int = 0,
) -> Device:
    """Generate a Euler s-shape bended waveguide connected with straight waveguide.

    Args:
        radius: Radius of the S-bend in microns (scalar or tuple for left/right).
        width: Width of the waveguide in microns (scalar or tuple for start/end).
        width_type: Width type of bend waveguide.
        angle: Angle of each bend in degrees (scalar or tuple for left/right).
        num_pts: Number of points for smoothness.
        use_eff: use effective radius (default = True)
        layer: GDS layer for the geometry.

    Returns:
        A PHIDL Device object.
    """

    radius_start, radius_end = parse_value(radius, "Radius", (1, 2))
    width_start, width_end = parse_value(width, "Width", (1, 2))
    angle_start, angle_end = parse_value(angle, "Angle", (1, 2))

    for value, name in [
        (radius_start, "Radius"),
        (radius_end, "Radius"),
        (width_start, "Width"),
        (width_end, "Width"),
        (num_pts, "Number of points"),
    ]:
        validate_positive(value, name)

    trans = pp.transition(
        cross_section1=create_cross_section(
            width=width_start, layer=layer, ports=(1, 2)
        ),
        cross_section2=create_cross_section(width=width_end, layer=layer, ports=(1, 2)),
        width_type=width_type,
    )

    P = Path()
    P.append(
        [
            pp.euler(
                radius=radius_start,
                angle=angle_start,
                num_pts=num_pts,
                use_eff=use_eff,
            ),
            pp.straight(
                length=length,
                num_pts=2,
            ),
            pp.euler(
                radius=radius_end,
                angle=-angle_end,
                num_pts=num_pts,
                use_eff=use_eff,
            ),
        ]
    )
    D = P.extrude(
        width=trans,
        simplify=0.0003,
    )

    D.name = "bend_straight_bend_euler"

    return D
