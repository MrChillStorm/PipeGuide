#!/usr/bin/env python3

import numpy as np
import pyvista as pv
import xml.etree.ElementTree as ET
from scipy.signal import bessel, filtfilt
import argparse


def load_fuselage_model(file_path):
    mesh = pv.read(file_path)
    if mesh is None:
        raise ValueError(f"Failed to load mesh from {file_path}.")
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
            print(f"Warning: Section between x={
                  start_x} and x={end_x} is empty.")
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
            zmax),
        invert=False)

    mesh_lower = section.clip_box(
        bounds=(
            section.bounds[0],
            section.bounds[1],
            ymin,
            ymax,
            zmin,
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
            ymin,
            center_y,
            zmin,
            zmax),
        invert=False)

    mesh_right = section.clip_box(
        bounds=(
            section.bounds[0],
            section.bounds[1],
            center_y,
            ymax,
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
    element = ET.Element('fuselage')
    element.set('ax', str(ax))
    element.set('ay', str(ay))
    element.set('az', str(az))
    element.set('bx', str(bx))
    element.set('by', str(by))
    element.set('bz', str(bz))
    element.set('width', str(width))
    element.set('taper', str(taper))
    element.set('midpoint', str(midpoint))
    return element


def write_to_xml(sections_info, output_file):
    root = ET.Element('airplane')
    for info in sections_info:
        element = create_xml_element(*info)
        root.append(element)
    tree = ET.ElementTree(root)
    tree.write(output_file, xml_declaration=True,
               encoding='utf-8', method="xml")

    num_elements = len(root)
    print(f"XML written with {num_elements} sections.")
    if num_elements < expected_num_sections:
        print(f"Warning: Fewer sections ({
              num_elements}) in XML than expected ({expected_num_sections}).")


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
        print(
            f"Warning: Number of processed centroids ({
                len(centroids)}) does not match number of sections ({
                len(sections)})")

    smoothed_centroids_x = apply_bessel_filter(
        [c[0] for c in centroids], order, cutoff)
    smoothed_centroids_y = apply_bessel_filter(
        [c[1] for c in centroids], order, cutoff)
    smoothed_centroids_z = apply_bessel_filter(
        [c[2] for c in centroids], order, cutoff)
    smoothed_widths = apply_bessel_filter(widths, order, cutoff)

    return smoothed_centroids_x, smoothed_centroids_y, smoothed_centroids_z, smoothed_widths


def align_and_merge(upper_info, lower_info, left_info, right_info):
    sections_info = []

    def process_section_info(info, position):
        for i in range(len(info[0])):
            ax = info[0][i]
            ay = info[1][i]
            az = info[2][i]
            bx = info[0][i + 1] if i + 1 < len(info[0]) else ax
            by = info[1][i + 1] if i + 1 < len(info[1]) else ay
            bz = info[2][i + 1] if i + 1 < len(info[2]) else az
            width_start = info[3][i]
            width_end = info[3][i + 1] if i + 1 < len(info[3]) else width_start

            taper = min(width_start, width_end) / max(width_start, width_end)
            midpoint = 1.0 if width_start < width_end else 0.0 if width_start > width_end else 0.5
            width = max(width_start, width_end)

            sections_info.append(
                (ax, ay, az, bx, by, bz, width, taper, midpoint))

    # Process each part
    process_section_info(upper_info, 'upper')
    process_section_info(lower_info, 'lower')
    process_section_info(left_info, 'left')
    process_section_info(right_info, 'right')

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
                upper_info, lower_info, left_info, right_info)
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
                    smoothed_centroids_z[i + 1] if i + \
                    1 < len(smoothed_centroids_z) else az
                width_start = smoothed_widths[i]
                width_end = smoothed_widths[i + 1] if i + \
                    1 < len(smoothed_widths) else width_start

                taper = min(width_start, width_end) / \
                    max(width_start, width_end)
                midpoint = 1.0 if width_start < width_end else 0.0 if width_start > width_end else 0.5
                width = max(width_start, width_end)

                sections_info.append(
                    (ax, ay, az, bx, by, bz, width, taper, midpoint))

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return

    try:
        write_to_xml(sections_info, output_file)
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
