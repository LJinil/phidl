from phidl import CrossSection


# ==================================================
# 1. Input Validation and Parsing
# ==================================================


def validate_positive(value, name):
    """Ensure a value is positive."""
    if value <= 0:
        raise ValueError(f"{name} must be positive. Got {value}.")


def parse_value(value, name, expected_lengths=(1, 2)):
    """
    Parses input to ensure it's a scalar or list/tuple of valid lengths.

    Args:
        value: The value to parse (int, float, list, or tuple).
        name: The name of the parameter (for error messages).
        expected_lengths: Allowed lengths for lists/tuples.

    Returns:
        A tuple of parsed values, expanded as needed.

    Raises:
        ValueError: If the value is not scalar or a list/tuple of allowed lengths.
    """
    if isinstance(value, (int, float)):
        return (value,) * expected_lengths[
            -1
        ]  # Expand scalar to the longest expected length
    elif isinstance(value, (list, tuple)):
        if len(value) in expected_lengths:
            if len(value) == 1:
                return (value[0],) * expected_lengths[-1]
            elif len(value) == 2:
                return (
                    (value[0], value[0], value[1], value[1])
                    if len(expected_lengths) == 4
                    else tuple(value)
                )
            elif len(value) == 4:
                return tuple(value)
        raise ValueError(
            f"{name} must have length in {expected_lengths}. Got {len(value)}."
        )
    else:
        raise TypeError(f"{name} must be a scalar or a list/tuple. Got {type(value)}.")


# ==================================================
# 2. Geometry and Cross-Section Utilities
# ==================================================


def create_cross_section(width, layer, offset=0, name="wg", ports=(None, None)):
    """Helper to create a cross-section for waveguides."""
    cs = CrossSection()
    cs.add(width=width, offset=offset, layer=layer, name=name, ports=ports)
    return cs


# ==================================================
# 3. Port Handling Utilities
# ==================================================


def add_ports_to_device(device, component, ports_name = None):
    """
    Add ports from a component to a device and remove them from the original component.

    Args:
        device: The main device where ports are added.
        component: The component providing the ports.
        ports_name: Ports names
    """
    if ports_name is None:
        for key in component.ports.keys():
            port = component.ports[key]
            device.add_port(port)
            component.remove(port)
    else:
        for key in ports_name:
            port = component.ports[key]
            device.add_port(port)
            component.remove(port)


def rename_ports(component, port_map):
    """
    Rename ports of a component.

    Args:
        component: The component whose ports are to be renamed.
        port_map: A dictionary mapping old port names to new port names.
    """
    for old_name, new_name in port_map.items():
        if old_name in component.ports:
            component.ports[old_name].name = new_name
