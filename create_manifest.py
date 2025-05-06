import os
import json
from pathlib import Path
import re

def find_slideshow_pdf(file_path: Path, course_content_root: Path):
    """
    Attempts to find the corresponding Mx.y_*.pdf slideshow file
    by looking in the parent 'day' directory (e.g., '1', '2', '3').
    """
    # Traverse up until we find a directory named '1', '2', or '3'
    day_dir = None
    current = file_path.parent
    while current != course_content_root and current != current.parent:
        if re.match(r"^[1-3]$", current.name):
            day_dir = current
            break
        current = current.parent

    if day_dir:
        # Look for a PDF starting with M<module>.<day>_
        # First, find the module directory (e.g., 01-Introduction-to-AI)
        module_dir = None
        current = day_dir.parent
        while current != course_content_root and current != current.parent:
             # Match directories like XX-Some-Name or XX-SomeName
            if re.match(r"^\d{2}-[\w-]+", current.name):
                module_dir = current
                break
            current = current.parent

        if module_dir:
            module_num = module_dir.name.split('-')[0]
            day_num = day_dir.name
            pdf_prefix = f"M{int(module_num)}.{day_num}_" # Use int() to remove leading zero if present
            try:
                for item in day_dir.iterdir():
                    if item.is_file() and item.name.startswith(pdf_prefix) and item.suffix.lower() == '.pdf':
                        # Return the path relative to the course_content_root
                        return item.relative_to(course_content_root).as_posix()
            except FileNotFoundError:
                # Handle cases where the day_dir might not exist (shouldn't happen in normal flow)
                print(f"Warning: Directory not found: {day_dir}")
                return None
            except Exception as e:
                print(f"Error searching for PDF in {day_dir}: {e}")
                return None


    return None # No day directory or PDF found following the pattern

def create_content_manifest():
    """
    Scans specified directories for .py, .md, .ipynb, and .pdf files,
    extracts metadata, finds related slideshows, and saves to a JSON manifest.
    """
    script_location = Path(__file__).parent
    course_content_dir = script_location / "course-content"
    manifest_file = script_location / "content_manifest.json"

    content_manifest = []
    file_types_to_scan = {".py", ".md", ".ipynb", ".pdf", ".txt"}
    dirs_to_scan = {
        ".py": course_content_dir,
        ".md": course_content_dir,
        ".ipynb": course_content_dir,
        ".pdf": course_content_dir,
        ".txt": course_content_dir,
    }
    base_dirs = { # Base directory for calculating relative path
        course_content_dir: course_content_dir,
    }

    print(f"Scanning for {', '.join(sorted(list(file_types_to_scan)))} files...")
    print(f"Scanning {', '.join(sorted(list(file_types_to_scan)))} in: {course_content_dir}")

    processed_files = 0
    for file_ext, scan_dir in dirs_to_scan.items():
        base_dir = base_dirs[scan_dir]
        if not scan_dir.exists():
            print(f"Warning: Directory not found, skipping: {scan_dir}")
            continue

        for file_path in scan_dir.rglob(f"*{file_ext}"):
            if file_path.is_file():
                processed_files += 1
                relative_path = file_path.relative_to(base_dir)
                abs_path = file_path.resolve()

                # Extract Module and Day from path
                module = None
                day = None
                try:
                    # Find first part that looks like a module (e.g., "01-Intro...")
                    module_part = next((part for part in relative_path.parts if re.match(r"^\d{2}-", part)), None)
                    if module_part:
                         module = module_part.split('-')[0]
                         # Find first part *after* module that looks like a day ("1", "2", "3")
                         day_part_index = -1
                         for i, part in enumerate(relative_path.parts):
                             if part == module_part:
                                 day_part_index = i + 1
                                 break
                         if day_part_index != -1 and day_part_index < len(relative_path.parts):
                            day_part_candidate = relative_path.parts[day_part_index]
                            if re.match(r"^[1-3]$", day_part_candidate):
                                day = day_part_candidate

                except Exception as e:
                    print(f"Warning: Could not parse module/day for {relative_path}: {e}")


                # Determine the correct root for slideshow lookup (always course-content)
                slideshow_lookup_path = abs_path

                # Find corresponding slideshow PDF relative to course_content_dir
                slideshow_pdf_rel_path = None
                if file_ext != '.pdf':
                    slideshow_pdf_rel_path = find_slideshow_pdf(slideshow_lookup_path, course_content_dir)


                metadata = {
                    "absolute_path": abs_path.as_posix(),
                    "relative_path": relative_path.as_posix(), # Use posix for consistency
                    "file_type": file_ext,
                    "module": module,
                    "day": day,
                    "slideshow_pdf": slideshow_pdf_rel_path
                }
                content_manifest.append(metadata)

                if processed_files % 100 == 0:
                    print(f"Processed {processed_files} files...")


    print(f"\nFound and processed {len(content_manifest)} relevant files.")

    # Save the manifest
    try:
        with open(manifest_file, 'w', encoding='utf-8') as f:
            json.dump(content_manifest, f, indent=4)
        print(f"Successfully created manifest file: {manifest_file}")
    except Exception as e:
        print(f"Error writing manifest file: {e}")

if __name__ == "__main__":
    create_content_manifest()
