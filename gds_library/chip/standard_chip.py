from phidl import Device, Path, CrossSection
import phidl.path as pp

from gds_library.components.basics import rectangle
from gds_library.chip.align_markers import align_marker
from gds_library.chip.standard_size import standard_size


def chip(
    standard_chip_size: str = "full",
    add_chip_boundary: bool = True,
    add_align_marker: bool = True,
) -> Device:

    D = Device("Chip")

    chip_size, marker_info = standard_size(standard_chip_size)
    width, height = chip_size

    if add_chip_boundary:
        B = Device("Chip boundary")

        left = B.add_ref(
            rectangle(
                length=1.0,
                width=height + 1.0,
                port=False,
            )
        )
        left.x = -width / 2
        left.y = 0

        right = B.add_ref(
            rectangle(
                length=1.0,
                width=height + 1.0,
                port=False,
            )
        )
        right.x = width / 2
        right.y = 0

        top = B.add_ref(
            rectangle(
                length=width + 1.0,
                width=1.0,
                port=False,
            )
        )
        top.x = 0
        top.y = height / 2

        bottom = B.add_ref(
            rectangle(
                length=width + 1.0,
                width=1.0,
                port=False,
            )
        )
        bottom.x = 0
        bottom.y = -height / 2

        D << B

    if add_align_marker:
        M = align_marker(
            marker_length=marker_info[0],
            marker_width=marker_info[1],
            marker_dist_x=marker_info[2],
            marker_dist_y=marker_info[3],
        )
        D << M

    return D
