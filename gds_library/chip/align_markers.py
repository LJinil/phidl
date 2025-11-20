from phidl import Device

from gds_library.components.basics import cross


def align_marker(
    marker_length: float = 400.0,
    marker_width: float = 4.0,
    marker_dist_x: float = 24000.0,
    marker_dist_y: float = 16500.0,
    layer: int = 0,
) -> Device:
    """Generate align marker.

    Args:
        width: of the cross.
        chip_size: of the cross.
        layer: layer.
    """
    D = Device("Align marker")

    marker_pos_x = [
        -marker_dist_x / 2,
        -marker_dist_x / 2,
        marker_dist_x / 2,
        marker_dist_x / 2,
    ]
    marker_pos_y = [
        -marker_dist_y / 2,
        marker_dist_y / 2,
        marker_dist_y / 2,
        -marker_dist_y / 2,
    ]

    marker = cross(length=marker_length, width=marker_width, layer=layer)

    for q, mc in enumerate(zip(marker_pos_x, marker_pos_y)):
        D.add_ref(marker).move([mc[0], mc[1]])
        (
            D.add_ref(marker).move([mc[0] - 500, mc[1]])
            if q > 1
            else D.add_ref(marker).move([mc[0] + 500, mc[1]])
        )

    return D
