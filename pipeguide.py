#!/usr/bin/env python3

from scipy.signal import bessel, filtfilt
import pyvista as pv
import numpy as np
from lxml import etree
import argparse

# Rotation matrix for 90 degrees around the x-axis
rotation_matrix_x_90 = np.array([
    [1, 0, 0],
    [0, 0, -1],
    [0, 1, 0]
])


def load_fuselage_model(file_path):
    mesh = pv.read(file_path)
    if mesh is None:
        raise ValueError(f"Failed to load mesh from {file_path}.")

    # Rotate the mesh if it's an OBJ file
    if file_path.lower().endswith('.obj'):
        print("Applying 90-degree rotation around the x-axis for OBJ file.")
        mesh.points = mesh.points @ rotation_matrix_x_90.T

    return mesh


def extract_fuselage_sections(mesh, num_sections):
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    x_range = np.linspace(xmin, xmax, num_sections + 1)
    sections = []
    for i in range(num_sections):
        start_x, end_x = x_range[i], x_range[i + 1]
        section = mesh.clip_box(
            bounds=(start_x, end_x, ymin, ymax, zmin, zmax), invert=False)
        if section.n_points == 0:
            print(f"Warning: Section between x={start_x} and x={end_x} is empty.")
        else:
            sections.append(section)
    return sections


def split_section_by_z(section):
    centroid = section.center
    points = section.points
    if points.size == 0:
        raise ValueError("Section has no points.")

    zmin, zmax = section.bounds[4], section.bounds[5]
    ymin, ymax = section.bounds[2], section.bounds[3]
    center_z = centroid[2]

    # Calculate distances to the edges
    distance_to_zmin = center_z - zmin
    distance_to_zmax = zmax - center_z

    # Define split values
    split_value_upper = center_z + distance_to_zmax
    split_value_lower = center_z - distance_to_zmin

    # Ensure split values are within bounds
    split_value_upper = min(split_value_upper, zmax)
    split_value_lower = max(split_value_lower, zmin)

    # Define clipping boxes
    mesh_upper = section.clip_box(
        bounds=(
            section.bounds[0],
            section.bounds[1],
            ymin,
            ymax,
            center_z,
            split_value_upper),
        invert=False)

    mesh_lower = section.clip_box(
        bounds=(
            section.bounds[0],
            section.bounds[1],
            ymin,
            ymax,
            split_value_lower,
            center_z),
        invert=False)

    return mesh_upper, mesh_lower


def split_section_by_y(section):
    centroid = section.center
    points = section.points
    if points.size == 0:
        raise ValueError("Section has no points.")

    ymin, ymax = section.bounds[2], section.bounds[3]
    zmin, zmax = section.bounds[4], section.bounds[5]
    center_y = centroid[1]

    # Calculate distances to the edges
    distance_to_ymin = center_y - ymin
    distance_to_ymax = ymax - center_y

    # Define split values
    split_value_left = center_y - distance_to_ymin
    split_value_right = center_y + distance_to_ymax

    # Ensure split values are within bounds
    split_value_left = max(split_value_left, ymin)
    split_value_right = min(split_value_right, ymax)

    # Define clipping boxes
    mesh_left = section.clip_box(
        bounds=(
            section.bounds[0],
            section.bounds[1],
            split_value_left,
            center_y,
            zmin,
            zmax),
        invert=False)

    mesh_right = section.clip_box(
        bounds=(
            section.bounds[0],
            section.bounds[1],
            center_y,
            split_value_right,
            zmin,
            zmax),
        invert=False)

    return mesh_left, mesh_right


def compute_centroid_and_width(section, width_axis):
    centroid = section.center
    points = section.points
    if points.size == 0:
        raise ValueError("Section has no points.")

    axis_index = {'x': 0, 'y': 1, 'z': 2}[width_axis]
    width_min = points[:, axis_index].min()
    width_max = points[:, axis_index].max()
    width = width_max - width_min

    centroid = (-centroid[0], -centroid[1], centroid[2])
    return centroid, width


def apply_bessel_filter(values, order, cutoff):
    b, a = bessel(order, cutoff, btype='low', analog=False)
    return filtfilt(b, a, values)


def create_xml_element(ax, ay, az, bx, by, bz, width, taper, midpoint):
    def format_number(value):
        formatted = f"{value:.3f}"
        # Strip minus sign if the value is -0.000
        if formatted == "-0.000":
            formatted = "0.000"
        return formatted

    def format_midpoint(value):
        return f"{int(value)}"

    element = etree.Element('fuselage')
    element.set('ax', format_number(ax))
    element.set('ay', format_number(ay))
    element.set('az', format_number(az))
    element.set('bx', format_number(bx))
    element.set('by', format_number(by))
    element.set('bz', format_number(bz))
    element.set('width', format_number(width))
    element.set('taper', format_number(taper))
    element.set('midpoint', format_midpoint(midpoint))
    return element


def get_max_lengths(sections_info):
    max_lengths = {
        'ax': 0, 'ay': 0, 'az': 0,
        'bx': 0, 'by': 0, 'bz': 0,
        'width': 0, 'taper': 0, 'midpoint': 0
    }

    for info in sections_info:
        ax, ay, az, bx, by, bz, width, taper = map(lambda x: f"{x:.3f}", info[:-1])
        midpoint = f"{int(info[-1])}"
        for key, value in zip(max_lengths.keys(), [ax, ay, az, bx, by, bz, width, taper, midpoint]):
            max_lengths[key] = max(max_lengths[key], len(value))

    return max_lengths


def format_element(element, max_lengths):
    formatted_attributes = []
    for key, value in element.attrib.items():
        # Calculate padding needed to match the longest value
        padding = max_lengths[key] - len(value)
        formatted_value = f"{' ' * padding}{value}"
        formatted_attributes.append(f'{key}="{formatted_value}"')

    tag = element.tag
    return f'<{tag} ' + ' '.join(formatted_attributes) + ' />'


def write_to_xml(sections_info, output_file, expected_num_sections):
    root = etree.Element('airplane')
    for info in sections_info:
        element = create_xml_element(*info)
        root.append(element)

    # Calculate the maximum lengths for each attribute
    max_lengths = get_max_lengths(sections_info)

    # Create formatted XML string
    formatted_xml_lines = ['<airplane>']
    for elem in root:
        formatted_xml_lines.append(format_element(elem, max_lengths))
    formatted_xml_lines.append('</airplane>')

    formatted_xml_string = '\n'.join(formatted_xml_lines)

    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(formatted_xml_string)

    num_elements = len(root)
    print(f"XML written with {num_elements} sections.")
    if num_elements < expected_num_sections:
        print(f"Warning: Fewer sections ({num_elements}) in XML than expected ({expected_num_sections}).")


def process_half(sections, axis, order, cutoff, part):
    centroids = []
    widths = []

    for section in sections:
        try:
            if part is None:
                current_section = section
            else:
                if axis == 'z':
                    mesh_upper, mesh_lower = split_section_by_z(section)
                    if part == 'upper' and mesh_upper.n_points > 0:
                        current_section = mesh_upper
                    elif part == 'lower' and mesh_lower.n_points > 0:
                        current_section = mesh_lower
                    else:
                        continue
                elif axis == 'y':
                    mesh_left, mesh_right = split_section_by_y(section)
                    if part == 'left' and mesh_left.n_points > 0:
                        current_section = mesh_left
                    elif part == 'right' and mesh_right.n_points > 0:
                        current_section = mesh_right
                    else:
                        continue

            centroid, width = compute_centroid_and_width(
                current_section, axis)
            centroids.append(centroid)
            widths.append(width)

        except ValueError as e:
            print(f"Error processing section: {e}")

    if len(centroids) != len(sections):
        print(f"Warning: Number of processed centroids ({len(centroids)}) does not match number of sections ({len(sections)})")

    smoothed_centroids_x = apply_bessel_filter(
        [c[0] for c in centroids], order, cutoff)
    smoothed_centroids_y = apply_bessel_filter(
        [c[1] for c in centroids], order, cutoff)
    smoothed_centroids_z = apply_bessel_filter(
        [c[2] for c in centroids], order, cutoff)
    smoothed_widths = apply_bessel_filter(widths, order, cutoff)

    return smoothed_centroids_x, smoothed_centroids_y, smoothed_centroids_z, smoothed_widths


def align_and_merge(upper_info, lower_info, left_info, right_info, expected_num_sections):
    sections_info = []
    all_infos = [upper_info, lower_info, left_info, right_info]

    def merge_zero_length_sections(info):
        merged_info = []
        i = 0
        while i < len(info[0]):
            ax = info[0][i]
            ay = info[1][i]
            az = info[2][i]
            bx = info[0][i + 1] if i + 1 < len(info[0]) else ax
            by = info[1][i + 1] if i + 1 < len(info[1]) else ay
            bz = info[2][i + 1] if i + 1 < len(info[2]) else az
            width_start = info[3][i]
            width_end = info[3][i + 1] if i + 1 < len(info[3]) else width_start

            # Calculate taper and midpoint for non-zero width sections
            taper = min(width_start, width_end) / max(width_start, width_end) if width_end != 0 else 1
            midpoint = 1.0 if width_start < width_end else 0.0 if width_start > width_end else 0.5
            width = max(width_start, width_end)

            # Check if the section is zero-length
            if (ax == bx) and (ay == by) and (az == bz):
                # Skip zero-length sections
                if i + 1 < len(info[0]):
                    # Merge zero-length section with the next one if possible
                    next_ax = info[0][i + 1]
                    next_ay = info[1][i + 1]
                    next_az = info[2][i + 1]
                    next_bx = info[0][i + 2] if i + 2 < len(info[0]) else next_ax
                    next_by = info[1][i + 2] if i + 2 < len(info[1]) else next_ay
                    next_bz = info[2][i + 2] if i + 2 < len(info[2]) else next_az
                    next_width_start = info[3][i + 1]
                    next_width_end = info[3][i + 2] if i + 2 < len(info[3]) else next_width_start

                    merged_section = (
                        ax, ay, az,
                        next_bx, next_by, next_bz,
                        max(width_start, next_width_end),
                        (taper + min(next_width_start, next_width_end) / max(next_width_start, next_width_end)) / 2,
                        (midpoint + (1.0 if next_width_start < next_width_end else 0.0 if next_width_start > next_width_end else 0.5)) / 2
                    )
                    merged_info.append(merged_section)
                    i += 2
                else:
                    i += 1
            else:
                merged_info.append((ax, ay, az, bx, by, bz, width, taper, midpoint))
                i += 1

        return merged_info

    # Merge zero-length sections for each part
    for info in all_infos:
        if info:
            sections_info.extend(merge_zero_length_sections(info))

    # Adjust to the exact number of sections
    if len(sections_info) < expected_num_sections:
        # If fewer sections, merge adjacent sections to meet the expected number
        while len(sections_info) < expected_num_sections:
            if len(sections_info) == 0:
                print("Error: No valid sections to merge.")
                break

            # Merge last two sections
            last_section = sections_info.pop()
            if len(sections_info) > 0:
                prev_section = sections_info.pop()

                ax, ay, az = prev_section[0:3]
                bx, by, bz = last_section[3:6]
                width = max(prev_section[6], last_section[6])
                taper = min(prev_section[6], last_section[6]) / width if width != 0 else 1
                midpoint = (prev_section[8] + last_section[8]) / 2

                new_section = (ax, ay, az, bx, by, bz, width, taper, midpoint)
                sections_info.append(new_section)

    return sections_info


def main(
        file_path,
        output_file,
        num_sections,
        order,
        cutoff,
        dual_axis_mode):
    global expected_num_sections
    expected_num_sections = num_sections

    try:
        mesh = load_fuselage_model(file_path)
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return
    except ValueError as ve:
        print(f"Error: {ve}")
        return

    try:
        sections = extract_fuselage_sections(mesh, num_sections)
    except ValueError as ve:
        print(f"Error: {ve}")
        return

    if not sections:
        print("Error: No valid sections were extracted.")
        return

    def merge_zero_length_sections(info):
        merged_info = []
        i = 0
        while i < len(info[0]):
            ax = info[0][i]
            ay = info[1][i]
            az = info[2][i]
            bx = info[0][i + 1] if i + 1 < len(info[0]) else ax
            by = info[1][i + 1] if i + 1 < len(info[1]) else ay
            bz = info[2][i + 1] if i + 1 < len(info[2]) else az
            width_start = info[3][i]
            width_end = info[3][i + 1] if i + 1 < len(info[3]) else width_start

            # Calculate taper and midpoint for non-zero length sections
            taper = min(width_start, width_end) / max(width_start, width_end) if width_end != 0 else 1
            midpoint = 1.0 if width_start < width_end else 0.0 if width_start > width_end else 0.5
            width = max(width_start, width_end)

            # Check if the section is zero-length
            if (ax == bx) and (ay == by) and (az == bz):
                # Skip zero-length sections
                if i + 1 < len(info[0]):
                    # Merge zero-length section with the next one if possible
                    next_ax = info[0][i + 1]
                    next_ay = info[1][i + 1]
                    next_az = info[2][i + 1]
                    next_bx = info[0][i + 2] if i + 2 < len(info[0]) else next_ax
                    next_by = info[1][i + 2] if i + 2 < len(info[1]) else next_ay
                    next_bz = info[2][i + 2] if i + 2 < len(info[2]) else next_az
                    next_width_start = info[3][i + 1]
                    next_width_end = info[3][i + 2] if i + 2 < len(info[3]) else next_width_start

                    merged_section = (
                        ax, ay, az,
                        next_bx, next_by, next_bz,
                        max(width_start, next_width_end),
                        (taper + min(next_width_start, next_width_end) / max(next_width_start, next_width_end)) / 2,
                        (midpoint + (1.0 if next_width_start < next_width_end else 0.0 if next_width_start > next_width_end else 0.5)) / 2
                    )
                    merged_info.append(merged_section)
                    i += 2
                else:
                    i += 1
            else:
                merged_info.append((ax, ay, az, bx, by, bz, width, taper, midpoint))
                i += 1

        return merged_info

    try:
        if dual_axis_mode:
            upper_info = process_half(
                sections, 'z', order, cutoff, part='upper')
            lower_info = process_half(
                sections, 'z', order, cutoff, part='lower')

            left_info = process_half(
                sections, 'y', order, cutoff, part='left')
            right_info = process_half(
                sections, 'y', order, cutoff, part='right')

            sections_info = align_and_merge(
                upper_info, lower_info, left_info, right_info, expected_num_sections)
        else:
            centroids = []
            widths = []
            for section in sections:
                try:
                    centroid, width = compute_centroid_and_width(
                        section, 'z')  # Assuming 'z' as default axis if not dual_axis_mode
                    centroids.append(centroid)
                    widths.append(width)
                except ValueError as e:
                    print(f"Error processing section: {e}")

            smoothed_centroids_x = apply_bessel_filter(
                [c[0] for c in centroids], order, cutoff)
            smoothed_centroids_y = apply_bessel_filter(
                [c[1] for c in centroids], order, cutoff)
            smoothed_centroids_z = apply_bessel_filter(
                [c[2] for c in centroids], order, cutoff)
            smoothed_widths = apply_bessel_filter(widths, order, cutoff)

            sections_info = []
            for i in range(len(sections)):
                ax, ay, az = smoothed_centroids_x[i], smoothed_centroids_y[i], smoothed_centroids_z[i]
                bx, by, bz = smoothed_centroids_x[i + 1] if i + 1 < len(smoothed_centroids_x) else ax, \
                    smoothed_centroids_y[i + 1] if i + 1 < len(smoothed_centroids_y) else ay, \
                    smoothed_centroids_z[i + 1] if i + 1 < len(smoothed_centroids_z) else az
                width_start = smoothed_widths[i]
                width_end = smoothed_widths[i + 1] if i + 1 < len(smoothed_widths) else width_start

                taper = min(width_start, width_end) / max(width_start, width_end) if width_end != 0 else 1
                midpoint = 1.0 if width_start < width_end else 0.0 if width_start > width_end else 0.5
                width = max(width_start, width_end)

                sections_info.append(
                    (ax, ay, az, bx, by, bz, width, taper, midpoint))

            # Merge zero-length sections
            sections_info = merge_zero_length_sections(
                (smoothed_centroids_x, smoothed_centroids_y, smoothed_centroids_z, smoothed_widths))

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return

    try:
        write_to_xml(sections_info, output_file, expected_num_sections)
    except IOError as e:
        print(f"Error writing XML file: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process fuselage model and generate XML")
    parser.add_argument(
        "input_file", help="Input file path for the fuselage model")
    parser.add_argument("-s", "--sections", type=int,
                        default=60, help="Number of sections (default: 60)")
    parser.add_argument("-f", "--filter-order", type=int,
                        default=2, help="Filter order (default: 2)")
    parser.add_argument("-c", "--filter-cutoff", type=float,
                        default=0.14, help="Filter cutoff (default: 0.14)")
    parser.add_argument("-o", "--output-file", default="yasim.xml",
                        help="Output XML file (default: yasim.xml)")
    parser.add_argument(
        "-d",
        "--dual-axis-mode",
        action="store_true",
        help="Process upper and lower halves separately and left and right sections")

    args = parser.parse_args()

    main(
        args.input_file,
        args.output_file,
        num_sections=args.sections,
        order=args.filter_order,
        cutoff=args.filter_cutoff,
        dual_axis_mode=args.dual_axis_mode)
