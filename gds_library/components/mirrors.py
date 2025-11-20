from phidl import Device

from gds_library.components.mmi import MMI
from gds_library.components.bends import bend_s_sine, bend_circular
from gds_library.helpers import validate_positive

def Sagnac_mirror_MMI(
    mmi_length: float = 50.0,
    mmi_width: float = 7.0,
    port_length: float = 10.0,
    port_width: float = 1.0,
    port_width_at_mmi: float = 1.5,
    port_gap: float = 2.0,
    mirror_radius: float = 100.0,
    layer: int = 0,
) -> Device:
    """Generate a Sagnac Mirror with Multi Mode Interferometer (MMI).

    Args:
        mmi_length: length of MMI.
        mmi_width: width of MMI.
        port_length: length of ports.
        port_width: width of ports.
        port_width_at_mmi: width of ports at MMI.
        port_gap: gap between ports.
        mirror_radius: radius of loop mirror.
        layer: layer.

    Returns:
        A PHIDL Device object representing the Sagnac Mirror.
    """

    for value, name in [
        (mmi_length, "MMI length"),
        (mmi_width, "MMI width"),
        (port_length, "Port length"),
        (port_width, "Port width"),
        (port_width_at_mmi, "Port width"),
        (port_gap, "Port gap"),
    ]:
        validate_positive(value, name)

    D = Device("Sagnac_mirror_MMI")

    mmi = D.add_ref(MMI(
            port_num = (1,2),
            mmi_length = mmi_length,
            mmi_width = mmi_width,
            port_length = port_length,
            port_width = port_width,
            port_width_at_mmi = port_width_at_mmi,
            port_gap = port_gap,
            layer = layer,
        ))
    D.add_port(mmi.ports[1])

    bend_top = D.add_ref(bend_s_sine(
        length = 2*mirror_radius,
        height = mirror_radius-(port_gap+port_width_at_mmi)/2,
        width = port_width,
        layer = layer,
        ))
    bend_top.connect(bend_top.ports[1],mmi.ports[3])

    bend_bottom = D.add_ref(bend_s_sine(
        length = 2*mirror_radius,
        height = -mirror_radius+(port_gap+port_width_at_mmi)/2,
        width = port_width,
        layer = layer,
        ))
    bend_bottom.connect(bend_bottom.ports[1],mmi.ports[2])

    circ = D.add_ref(bend_circular(
        radius = mirror_radius,
        width = port_width,
        angle = 180,
        layer = layer,
        ))
    circ.connect(circ.ports[1],bend_bottom.ports[2])

    D.flatten()


    return D
