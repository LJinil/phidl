def standard_size(
    standard_size: str = "full",
) -> tuple[tuple[float, float], tuple[float, float, float, float]]:

    if standard_size in ["full", "f", "Full"]:
        chip_size = 24500, 19000
        marker_info = 400, 4, 24000, 18000

    elif standard_size in ["half horizontal", "hh"]:
        chip_size = 24500, 18000
        marker_info = 400, 4, 24000, 16500

    elif standard_size in ["half vertical", "hv"]:
        chip_size = 24500, 18000
        marker_info = 400, 4, 24000, 16500

    elif standard_size in ["quarter", "q"]:
        chip_size = 24500, 18000
        marker_info = 400, 4, 24000, 16500

    else:
        chip_size = 24500, 18000
        marker_info = 400, 4, 24000, 16500

    return chip_size, marker_info
