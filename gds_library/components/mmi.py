from phidl import Device

from gds_library.components.basics import rectangle
from gds_library.components.tapers import taper
from gds_library.helpers import validate_positive


def MMI(
    port_num: tuple[int, int] = (2, 2),
    mmi_length: float = 50.0,
    mmi_width: float = 7.0,
    port_length: float = 10.0,
    port_width: float = 1.0,
    port_width_at_mmi: float = 1.5,
    port_gap: float = 2.0,
    layer: int = 0,
) -> Device:
    """Generate a Multi-Mode Interferometer (MMI).

    Args:
        port_num: Tuple representing the number of input and output ports (M, N).
        mmi_length: Length of the MMI region in microns.
        mmi_width: Width of the MMI region in microns.
        port_length: Length of the tapers connecting to the MMI.
        port_width: Width of the ports in microns.
        port_gap: Gap between adjacent ports in microns.
        layer: GDS layer for the geometry.

    Returns:
        A PHIDL Device object representing the MMI.
    """

    for value, name in [
        (mmi_length, "MMI length"),
        (mmi_width, "MMI width"),
        (port_length, "Port length"),
        (port_width, "Port width"),
        (port_gap, "Port gap"),
    ]:
        validate_positive(value, name)

    if len(port_num) != 2 or any(n <= 0 for n in port_num):
        raise ValueError("port_num must be a tuple of two positive integers.")

    D = Device("MMI")

    m, n = port_num

    mmi = rectangle(
        width=mmi_width,
        length=mmi_length,
        port=False,
        layer=layer,
    )
    D << mmi

    port_left = taper(
        length=port_length,
        width1=port_width,
        width2=port_width_at_mmi,
        layer=layer,
    )
    port_left.xmax = mmi.xmin

    port_right = taper(
        length=port_length,
        width1=port_width_at_mmi,
        width2=port_width,
        layer=layer,
    )
    port_right.xmin = mmi.xmax

    for i in range(m):
        port_ref = D.add_ref(port_left).movey(
            mmi.y + (i - (m - 1) / 2) * (port_gap + port_width_at_mmi)
        )
        port_ref.ports[1].name = i + 1
        D.add_port(port_ref.ports[1])

    for j in range(n):
        port_ref = D.add_ref(port_right).movey(
            mmi.y + (j - (n - 1) / 2) * (port_gap + port_width_at_mmi)
        )
        port_ref.ports[2].name = m + j + 1
        D.add_port(port_ref.ports[2])

    D.flatten()

    return D
