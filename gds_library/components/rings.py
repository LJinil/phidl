from phidl import Device, Path, CrossSection
import phidl.path as pp

from gds_library.components.basics import rectangle
from gds_library.helpers import (
    parse_value,
    validate_positive,
    create_cross_section,
    add_ports_to_device,
    rename_ports,
)


def ring(
    radius: float = 10.0,
    width: float = 0.5,
    num_pts: int = 720,
    layer: int = 0,
) -> Device:
    """Generate a ring.

    Args:
        radius: Radius of the ring in microns.
        width: Width of the waveguide in microns.
        num_pts: Number of points for smoothness.
        layer: GDS layer for the geometry.

    Returns:
        A PHIDL Device object representing the ring .
    """

    for value, name in [
        (radius, "Radius"),
        (width, "Width"),
        (num_pts, "Number of points"),
    ]:
        validate_positive(value, name)

    ring_cs = create_cross_section(width, layer)

    P = Path()
    for _ in range(4):
        P.append(
            pp.arc(
                radius=radius,
                angle=90,
                num_pts=num_pts,
            )
        )

    D = P.extrude(
        width=ring_cs,
        simplify=0.0003,
    )

    return D


def ring_euler(
    radius: float = 10.0,
    width: float = 0.5,
    num_pts: int = 720,
    use_eff: bool = True,
    layer: int = 0,
) -> Device:
    """Generate a ring (with Euler curve).

    Args:
        radius: Radius of the ring in microns.
        width: Width of the waveguide in microns.
        num_pts: Number of points for smoothness.
        use_eff: use effective radius (default = True)
        layer: GDS layer for the geometry.

    Returns:
        A PHIDL Device object representing the ring .
    """

    for value, name in [
        (radius, "Radius"),
        (width, "Width"),
        (num_pts, "Number of points"),
    ]:
        validate_positive(value, name)

    ring_cs = create_cross_section(width, layer)

    P = Path()
    for _ in range(4):
        P.append(
            pp.euler(
                radius=radius,
                angle=90,
                num_pts=num_pts,
                use_eff=use_eff,
            )
        )

    D = P.extrude(
        width=ring_cs,
        simplify=0.0003,
    )

    D.name = "ring_euler"

    return D


def ring_single(
    radius: float = 10.0,
    width: float = 0.5,
    gap: float = 0.5,
    wg_length: float = 20.0,
    wg_width: float = 0.5,
    num_pts: int = 720,
    layer: int = 0,
) -> Device:
    """Generate a ring resonator with a single straight waveguides.

    Args:
        radius: Radius of the ring in microns.
        width: Width of the ring waveguide in microns.
        gap: Gap between the ring and the straight waveguides in microns.
        wg_length: Length of the straight waveguides in microns.
        wg_width: Width of the straight waveguides in microns.
        num_pts: Number of points for smoothness.
        layer: GDS layer for the geometry.

    Returns:
        A PHIDL Device object representing the ring resonator.
    """

    for value, name in [
        (radius, "Radius"),
        (width, "Width"),
        (gap, "Gap"),
        (wg_length, "Waveguide length"),
        (wg_width, "Waveguide width"),
        (num_pts, "Number of points"),
    ]:
        validate_positive(value, name)

    D = Device("ring_single")

    ring_cs = create_cross_section(width, layer)

    P = Path()
    for _ in range(4):
        P.append(
            pp.arc(
                radius=radius,
                angle=90,
                num_pts=num_pts,
            )
        )

    r = P.extrude(
        width=ring_cs,
        simplify=0.0003,
    )

    wg = rectangle(
        width=wg_width,
        length=wg_length,
        layer=layer,
    )

    wg.x = r.x
    wg.ymax = r.ymin - gap

    D << r
    D << wg

    add_ports_to_device(D, wg)

    D.flatten()

    return D


def ring_single_euler(
    radius: float = 10.0,
    width: float = 0.5,
    gap: float = 0.5,
    wg_length: float = 20.0,
    wg_width: float = 0.5,
    num_pts: int = 720,
    use_eff: bool = True,
    layer: int = 0,
) -> Device:
    """Generate a ring resonator (with Euler curve) with a single straight waveguides.

    Args:
        radius: Radius of the ring in microns.
        width: Width of the ring waveguide in microns.
        gap: Gap between the ring and the straight waveguides in microns.
        wg_length: Length of the straight waveguides in microns.
        wg_width: Width of the straight waveguides in microns.
        num_pts: Number of points for smoothness.
        use_eff: use effective radius (default = True)
        layer: GDS layer for the geometry.

    Returns:
        A PHIDL Device object representing the ring resonator.
    """

    for value, name in [
        (radius, "Radius"),
        (width, "Width"),
        (gap, "Gap"),
        (wg_length, "Waveguide length"),
        (wg_width, "Waveguide width"),
        (num_pts, "Number of points"),
    ]:
        validate_positive(value, name)

    D = Device("ring_single_euler")

    ring_cs = create_cross_section(width, layer)

    P = Path()
    for _ in range(4):
        P.append(
            pp.euler(
                radius=radius,
                angle=90,
                num_pts=num_pts,
                use_eff=use_eff,
            )
        )

    r = P.extrude(
        width=ring_cs,
        simplify=0.0003,
    )

    wg = rectangle(
        width=wg_width,
        length=wg_length,
        layer=layer,
    )

    wg.x = r.x
    wg.ymax = r.ymin - gap

    D << r
    D << wg

    add_ports_to_device(D, wg)

    D.flatten()

    return D


def ring_double(
    radius: float = 10.0,
    width: float = 0.5,
    gap: float | tuple[float, float] = 0.5,
    wg_length: float | tuple[float, float] = 20.0,
    wg_width: float | tuple[float, float] = 0.5,
    num_pts: int = 720,
    layer: int = 0,
) -> Device:
    """Generate a ring resonator with a double straight waveguides.

    Args:
        radius: Radius of the ring in microns.
        width: Width of the ring waveguide in microns.
        gap: Gap between the ring and the straight waveguides in microns. (scalar or tuple for bottom/top)
        wg_length: Length of the straight waveguides in microns. (scalar or tuple for bottom/top)
        wg_width: Width of the straight waveguides in microns. (scalar or tuple for bottom/top)
        num_pts: Number of points for smoothness.
        use_eff: use effective radius (default = True)
        layer: GDS layer for the geometry.

    Returns:
        A PHIDL Device object representing the ring resonator.
    """

    gap_bottom, gap_top = parse_value(gap, "Gap", (1, 2))
    wg_bottom_length, wg_top_length = parse_value(wg_length, "Waveguide length", (1, 2))
    wg_bottom_width, wg_top_width = parse_value(wg_width, "Waveguide width", (1, 2))

    for value, name in [
        (radius, "Radius"),
        (width, "Width"),
        (gap_bottom, "Gap"),
        (gap_top, "Gap"),
        (wg_bottom_length, "Waveguide length"),
        (wg_top_length, "Waveguide length"),
        (wg_bottom_width, "Waveguide width"),
        (wg_top_width, "Waveguide width"),
        (num_pts, "Number of points"),
    ]:
        validate_positive(value, name)

    D = Device("ring_double")

    ring_cs = create_cross_section(width, layer)

    P = Path()
    for _ in range(4):
        P.append(
            pp.arc(
                radius=radius,
                angle=90,
                num_pts=num_pts,
            )
        )

    r = P.extrude(
        width=ring_cs,
        simplify=0.0003,
    )

    wg_b = rectangle(
        width=wg_bottom_width,
        length=wg_bottom_length,
        layer=layer,
    )
    wg_t = rectangle(
        width=wg_top_width,
        length=wg_top_length,
        layer=layer,
    )

    wg_b.x = r.x
    wg_b.ymax = r.ymin - gap_bottom
    wg_t.x = r.x
    wg_t.ymin = r.ymax + gap_top

    rename_ports(wg_t, {1: 3, 2: 4})

    D << r
    D << wg_b
    D << wg_t

    add_ports_to_device(D, wg_b)
    add_ports_to_device(D, wg_t)

    D.flatten()

    return D


def ring_double_euler(
    radius: float = 10.0,
    width: float = 0.5,
    gap: float | tuple[float, float] = 0.5,
    wg_length: float | tuple[float, float] = 20.0,
    wg_width: float | tuple[float, float] = 0.5,
    num_pts: int = 720,
    use_eff: bool = True,
    layer: int = 0,
) -> Device:
    """Generate a ring resonator (with Euler curve) with a double straight waveguides.

    Args:
        radius: Radius of the ring in microns.
        width: Width of the ring waveguide in microns.
        gap: Gap between the ring and the straight waveguides in microns. (scalar or tuple for bottom/top)
        wg_length: Length of the straight waveguides in microns. (scalar or tuple for bottom/top)
        wg_width: Width of the straight waveguides in microns. (scalar or tuple for bottom/top)
        num_pts: Number of points for smoothness.
        use_eff: use effective radius (default = True)
        layer: GDS layer for the geometry.

    Returns:
        A PHIDL Device object representing the ring resonator.
    """

    gap_bottom, gap_top = parse_value(gap, "Gap", (1, 2))
    wg_bottom_length, wg_top_length = parse_value(wg_length, "Waveguide length", (1, 2))
    wg_bottom_width, wg_top_width = parse_value(wg_width, "Waveguide width", (1, 2))

    for value, name in [
        (radius, "Radius"),
        (width, "Width"),
        (gap_bottom, "Gap"),
        (gap_top, "Gap"),
        (wg_bottom_length, "Waveguide length"),
        (wg_top_length, "Waveguide length"),
        (wg_bottom_width, "Waveguide width"),
        (wg_top_width, "Waveguide width"),
        (num_pts, "Number of points"),
    ]:
        validate_positive(value, name)

    D = Device("ring_double_euler")

    ring_cs = create_cross_section(width, layer)

    P = Path()
    for _ in range(4):
        P.append(
            pp.euler(
                radius=radius,
                angle=90,
                num_pts=num_pts,
                use_eff=use_eff,
            )
        )

    r = P.extrude(
        width=ring_cs,
        simplify=0.0003,
    )

    wg_b = rectangle(
        width=wg_bottom_width,
        length=wg_bottom_length,
        layer=layer,
    )
    wg_t = rectangle(
        width=wg_top_width,
        length=wg_top_length,
        layer=layer,
    )

    wg_b.x = r.x
    wg_b.ymax = r.ymin - gap_bottom
    wg_t.x = r.x
    wg_t.ymin = r.ymax + gap_top

    rename_ports(wg_t, {1: 3, 2: 4})

    D << r
    D << wg_b
    D << wg_t

    add_ports_to_device(D, wg_b)
    add_ports_to_device(D, wg_t)

    D.flatten()

    return D


def racetrack(
    radius: float = 10.0,
    width: float = 0.5,
    length_x: float = 5.0,
    length_y: float = 2.0,
    num_pts: int = 720,
    layer: int = 0,
) -> Device:
    """Generate a racetrack resonator.

    Args:
        radius: Radius of the curved bends in microns.
        width: Width of the waveguide in microns.
        length_x: Length of the straight (in x direction) sections in microns.
        length_y: Length of the straight (in y direction) sections in microns.
        num_pts: Number of points for smoothness.
        layer: GDS layer for the geometry.

    Returns:
        A PHIDL Device object representing the racetrack resonator.
    """
    for value, name in [
        (radius, "Radius"),
        (width, "Width"),
        (length_x, "Straight length"),
        (length_y, "Straight length"),
        (num_pts, "Number of points"),
    ]:
        validate_positive(value, name)

    rt_cs = create_cross_section(width, layer)

    lengths = [length_y, length_x, length_y, length_x]
    P = Path()
    for l in lengths:
        P.append(
            pp.arc(
                radius=radius,
                angle=90,
                num_pts=num_pts,
            )
        )
        (
            P.append(
                pp.straight(
                    length=l,
                    num_pts=2,
                )
            )
            if l > 0
            else None
        )

    D = P.extrude(
        width=rt_cs,
        simplify=0.0003,
    )

    D.name = "racetrack"

    return D


def racetrack_euler(
    radius: float = 10.0,
    width: float = 0.5,
    length_x: float = 5.0,
    length_y: float = 2.0,
    num_pts: int = 720,
    use_eff: bool = True,
    layer: int = 0,
) -> Device:
    """Generate a racetrack resonator (with Euler curve).

    Args:
        radius: Radius of the curved bends in microns.
        width: Width of the waveguide in microns.
        length_x: Length of the straight (in x direction) sections in microns.
        length_y: Length of the straight (in y direction) sections in microns.
        num_pts: Number of points for smoothness.
        use_eff: use effective radius (default = True)
        layer: GDS layer for the geometry.

    Returns:
        A PHIDL Device object representing the racetrack resonator.
    """
    for value, name in [
        (radius, "Radius"),
        (width, "Width"),
        (length_x, "Straight length"),
        (length_y, "Straight length"),
        (num_pts, "Number of points"),
    ]:
        validate_positive(value, name)

    rt_cs = create_cross_section(width, layer)

    lengths = [length_y, length_x, length_y, length_x]
    P = Path()
    for l in lengths:
        P.append(
            pp.euler(
                radius=radius,
                angle=90,
                num_pts=num_pts,
                use_eff=use_eff,
            )
        )
        (
            P.append(
                pp.straight(
                    length=l,
                    num_pts=2,
                )
            )
            if l > 0
            else None
        )

    D = P.extrude(
        width=rt_cs,
        simplify=0.0003,
    )

    D.name = "racetrack_euler"

    return D


def racetrack_single(
    radius: float = 10.0,
    width: float = 0.5,
    length_x: float = 5.0,
    length_y: float = 2.0,
    gap: float = 0.5,
    wg_length: float = 20.0,
    wg_width: float = 0.5,
    num_pts: int = 720,
    layer: int = 0,
) -> Device:
    """Generate a racetrack resonator with a single coupling waveguide.

    Args:
        radius: Radius of the curved bends in microns.
        width: Width of the racetrack waveguide in microns.
        length_x: Length of the straight (in x direction) sections in microns.
        length_y: Length of the straight (in y direction) sections in microns.
        gap: Gap between the racetrack and the straight waveguide in microns.
        wg_length: Length of the coupling waveguide in microns.
        wg_width: Width of the coupling waveguide in microns.
        num_pts: Number of points for smoothness.
        layer: GDS layer for the geometry.

    Returns:
        A PHIDL Device object representing the racetrack resonator.
    """

    for value, name in [
        (radius, "Radius"),
        (width, "Width"),
        (length_x, "Straight length"),
        (length_y, "Straight length"),
        (gap, "Gap"),
        (wg_length, "Waveguide length"),
        (wg_width, "Waveguide width"),
        (num_pts, "Number of points"),
    ]:
        validate_positive(value, name)

    D = Device("racetrack_single")

    rt_cs = create_cross_section(width, layer)

    lengths = [length_y, length_x, length_y, length_x]
    P = Path()
    for l in lengths:
        P.append(
            pp.arc(
                radius=radius,
                angle=90,
                num_pts=num_pts,
            )
        )
        (
            P.append(
                pp.straight(
                    length=l,
                    num_pts=2,
                )
            )
            if l > 0
            else None
        )

    r = P.extrude(
        width=rt_cs,
        simplify=0.0003,
    )

    wg = rectangle(
        width=wg_width,
        length=wg_length,
        layer=layer,
    )

    wg.x = r.x
    wg.ymax = r.ymin - gap

    D << r
    D << wg

    add_ports_to_device(D, wg)

    D.flatten()

    return D


def racetrack_single_euler(
    radius: float = 10.0,
    width: float = 0.5,
    length_x: float = 5.0,
    length_y: float = 2.0,
    gap: float = 0.5,
    wg_length: float = 20.0,
    wg_width: float = 0.5,
    num_pts: int = 720,
    use_eff: bool = True,
    layer: int = 0,
) -> Device:
    """Generate a racetrack resonator (with Euler curve) with a single coupling waveguide.

    Args:
        radius: Radius of the curved bends in microns.
        width: Width of the racetrack waveguide in microns.
        length_x: Length of the straight (in x direction) sections in microns.
        length_y: Length of the straight (in y direction) sections in microns.
        gap: Gap between the racetrack and the straight waveguide in microns.
        wg_length: Length of the coupling waveguide in microns.
        wg_width: Width of the coupling waveguide in microns.
        num_pts: Number of points for smoothness.
        use_eff: use effective radius (default = True)
        layer: GDS layer for the geometry.

    Returns:
        A PHIDL Device object representing the racetrack resonator.
    """

    for value, name in [
        (radius, "Radius"),
        (width, "Width"),
        (length_x, "Straight length"),
        (length_y, "Straight length"),
        (gap, "Gap"),
        (wg_length, "Waveguide length"),
        (wg_width, "Waveguide width"),
        (num_pts, "Number of points"),
    ]:
        validate_positive(value, name)
    
    D = Device("racetrack_single_euler")

    rt_cs = CrossSection().add(
        width=width,
        offset=0,
        layer=layer,
    )

    lengths = [length_y, length_x, length_y, length_x]
    P = Path()
    for l in lengths:
        P.append(
            pp.euler(
                radius=radius,
                angle=90,
                num_pts=num_pts,
                use_eff=use_eff,
            )
        )
        (
            P.append(
                pp.straight(
                    length=l,
                    num_pts=2,
                )
            )
            if l > 0
            else None
        )

    r = P.extrude(
        width=rt_cs,
        simplify=0.0003,
    )

    wg = rectangle(
        width=wg_width,
        length=wg_length,
        layer=layer,
    )

    wg.x = r.x
    wg.ymax = r.ymin - gap

    D << r
    D << wg

    add_ports_to_device(D, wg)

    D.flatten()

    return D


def racetrack_double(
    radius: float = 10.0,
    width: float = 0.5,
    length_x: float = 5.0,
    length_y: float = 2.0,
    gap: float | tuple[float, float] = 0.5,
    wg_length: float | tuple[float, float] = 20.0,
    wg_width: float | tuple[float, float] = 0.5,
    num_pts: int = 720,
    layer: int = 0,
) -> Device:
    """Generate a racetrack resonator with a double coupling waveguide.

    Args:
        radius: Radius of the curved bends in microns.
        width: Width of the racetrack waveguide in microns.
        length_x: Length of the straight (in x direction) sections in microns.
        length_y: Length of the straight (in y direction) sections in microns.
        gap: Gap between the racetrack and the straight waveguide in microns. (scalar or tuple for bottom/top)
        wg_length: Length of the coupling waveguide in microns. (scalar or tuple for bottom/top)
        wg_width: Width of the coupling waveguide in microns. (scalar or tuple for bottom/top)
        num_pts: Number of points for smoothness.
        layer: GDS layer for the geometry.

    Returns:
        A PHIDL Device object representing the racetrack resonator.
    """

    gap_bottom, gap_top = parse_value(gap, "Gap", (1, 2))
    wg_bottom_length, wg_top_length = parse_value(wg_length, "Waveguide length", (1, 2))
    wg_bottom_width, wg_top_width = parse_value(wg_width, "Waveguide width", (1, 2))

    for value, name in [
        (radius, "Radius"),
        (width, "Width"),
        (length_x, "Straight length"),
        (length_y, "Straight length"),
        (gap_bottom, "Gap"),
        (gap_top, "Gap"),
        (wg_bottom_length, "Waveguide length"),
        (wg_top_length, "Waveguide length"),
        (wg_bottom_width, "Waveguide width"),
        (wg_top_width, "Waveguide width"),
        (num_pts, "Number of points"),
    ]:
        validate_positive(value, name)

    D = Device("racetrack_double")

    rt_cs = create_cross_section(width, layer)

    lengths = [length_y, length_x, length_y, length_x]
    P = Path()
    for l in lengths:
        P.append(
            pp.arc(
                radius=radius,
                angle=90,
                num_pts=num_pts,
            )
        )
        (
            P.append(
                pp.straight(
                    length=l,
                    num_pts=2,
                )
            )
            if l > 0
            else None
        )

    r = P.extrude(
        width=rt_cs,
        simplify=0.0003,
    )

    wg_b = rectangle(
        width=wg_bottom_width,
        length=wg_bottom_length,
        layer=layer,
    )
    wg_t = rectangle(
        width=wg_top_width,
        length=wg_top_length,
        layer=layer,
    )
    wg_b.x = r.x
    wg_b.ymax = r.ymin - gap_bottom
    wg_t.x = r.x
    wg_t.ymin = r.ymax + gap_top

    rename_ports(wg_t, {1: 3, 2: 4})

    D << r
    D << wg_b
    D << wg_t

    add_ports_to_device(D, wg_b)
    add_ports_to_device(D, wg_t)

    D.flatten()

    return D


def racetrack_double_euler(
    radius: float = 10.0,
    width: float = 0.5,
    length_x: float = 5.0,
    length_y: float = 2.0,
    gap: float | tuple[float, float] = 0.5,
    wg_length: float | tuple[float, float] = 20.0,
    wg_width: float | tuple[float, float] = 0.5,
    num_pts: int = 720,
    use_eff: bool = True,
    layer: int = 0,
) -> Device:
    """Generate a racetrack resonator with a double coupling waveguide.

    Args:
        radius: Radius of the curved bends in microns.
        width: Width of the racetrack waveguide in microns.
        length_x: Length of the straight (in x direction) sections in microns.
        length_y: Length of the straight (in y direction) sections in microns.
        gap: Gap between the racetrack and the straight waveguide in microns. (scalar or tuple for bottom/top)
        wg_length: Length of the coupling waveguide in microns. (scalar or tuple for bottom/top)
        wg_width: Width of the coupling waveguide in microns. (scalar or tuple for bottom/top)
        num_pts: Number of points for smoothness.
        use_eff: use effective radius (default = True)
        layer: GDS layer for the geometry.

    Returns:
        A PHIDL Device object representing the racetrack resonator.
    """

    gap_bottom, gap_top = parse_value(gap, "Gap", (1, 2))
    wg_bottom_length, wg_top_length = parse_value(wg_length, "Waveguide length", (1, 2))
    wg_bottom_width, wg_top_width = parse_value(wg_width, "Waveguide width", (1, 2))

    for value, name in [
        (radius, "Radius"),
        (width, "Width"),
        (length_x, "Straight length"),
        (length_y, "Straight length"),
        (gap_bottom, "Gap"),
        (gap_top, "Gap"),
        (wg_bottom_length, "Waveguide length"),
        (wg_top_length, "Waveguide length"),
        (wg_bottom_width, "Waveguide width"),
        (wg_top_width, "Waveguide width"),
        (num_pts, "Number of points"),
    ]:
        validate_positive(value, name)

    D = Device("racetrack_double_euler")

    rt_cs = create_cross_section(width, layer)

    lengths = [length_y, length_x, length_y, length_x]
    P = Path()
    for l in lengths:
        P.append(
            pp.euler(
                radius=radius,
                angle=90,
                num_pts=num_pts,
                use_eff=use_eff,
            )
        )
        (
            P.append(
                pp.straight(
                    length=l,
                    num_pts=2,
                )
            )
            if l > 0
            else None
        )

    r = P.extrude(
        width=rt_cs,
        simplify=0.0003,
    )

    wg_b = rectangle(
        width=wg_bottom_width,
        length=wg_bottom_length,
        layer=layer,
    )
    wg_t = rectangle(
        width=wg_top_width,
        length=wg_top_length,
        layer=layer,
    )
    wg_b.x = r.x
    wg_b.ymax = r.ymin - gap_bottom
    wg_t.x = r.x
    wg_t.ymin = r.ymax + gap_top

    rename_ports(wg_t, {1: 3, 2: 4})

    D << r
    D << wg_b
    D << wg_t

    add_ports_to_device(D, wg_b)
    add_ports_to_device(D, wg_t)

    D.flatten()

    return D