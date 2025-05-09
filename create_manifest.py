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

def is_transcript_filename(file_name):
    """
    Check if a filename matches transcript file pattern.
    """
    # Match patterns like XX-YY.txt or XX-Name_YY.txt
    return bool(re.match(r'^\d{2}-\d{2}\.txt$', file_name) or 
                re.match(r'^\d{2}-[\w-]+_\d{2}\.txt$', file_name))
    

def extract_module_day_from_transcript(file_name):
    """
    Extracts module and day information from a transcript file name.
    Expected format: XX-YY.txt or XX-Name_YY.txt where XX is module number and YY is day number.
    """
    try:
        # Handle regular format: XX-YY.txt
        if re.match(r'^\d{2}-\d{2}\.txt$', file_name):
            parts = file_name.split('-')
            module = parts[0]
            day = parts[1].split('.')[0]
            return module, day
        # Handle format: XX-Name_YY.txt
        elif re.match(r'^\d{2}-[\w-]+_\d{2}\.txt$', file_name):
            module = file_name.split('-')[0]
            day = file_name.split('_')[-1].split('.')[0]
            return module, day
        return None, None
    except Exception as e:
        print(f"Warning: Could not parse module/day for transcript {file_name}: {e}")
        return None, None

def create_content_manifest():
    """
    Scans specified directories for .py, .md, .ipynb, .pdf, .txt files,
    and transcript files, extracts metadata, finds related slideshows, 
    and saves to a JSON manifest.
    """
    script_location = Path(__file__).parent
    course_content_dir = script_location / "course-content"
    transcripts_dir = script_location / "transcripts"
    manifest_file = script_location / "content_manifest.json"

    content_manifest = []
    file_types_to_scan = {".py", ".md", ".ipynb", ".pdf", ".txt"}
    
    # Initialize counters for file types
    file_type_counts = {
        ".py": 0,
        ".md": 0,
        ".ipynb": 0,
        ".pdf": 0,
        ".txt": 0,
        "transcript": 0
    }

    print(f"Scanning for {', '.join(sorted(list(file_types_to_scan)))} files...")
    print(f"Scanning {', '.join(sorted(list(file_types_to_scan)))} in: {course_content_dir}")
    print(f"Scanning transcript files in: {transcripts_dir}")

    processed_files = 0

    # Process course content files
    if course_content_dir.exists():
        for file_ext in file_types_to_scan:
            for file_path in course_content_dir.rglob(f"*{file_ext}"):
                # Skip .txt files that match transcript naming pattern
                if file_ext == ".txt" and is_transcript_filename(file_path.name):
                    continue
                    
                if file_path.is_file():
                    processed_files += 1
                    file_type_counts[file_ext] += 1
                    relative_path = file_path.relative_to(course_content_dir)
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
                        "slideshow_pdf": slideshow_pdf_rel_path,
                        "is_transcript": False
                    }
                    content_manifest.append(metadata)

                    if processed_files % 100 == 0:
                        print(f"Processed {processed_files} files...")
    else:
        print(f"Warning: Course content directory not found: {course_content_dir}")

    # Process transcript files
    if transcripts_dir.exists():
        for file_path in transcripts_dir.glob("*.txt"):
            if file_path.is_file():
                processed_files += 1
                file_type_counts["transcript"] += 1
                relative_path = file_path.relative_to(transcripts_dir)
                abs_path = file_path.resolve()

                # Extract module and day from transcript filename
                module, day = extract_module_day_from_transcript(file_path.name)

                metadata = {
                    "absolute_path": abs_path.as_posix(),
                    "relative_path": relative_path.as_posix(),
                    "file_type": ".txt",
                    "module": module,
                    "day": day,
                    "slideshow_pdf": None,
                    "is_transcript": True
                }
                content_manifest.append(metadata)

                if processed_files % 100 == 0:
                    print(f"Processed {processed_files} files...")
    else:
        print(f"Warning: Transcripts directory not found: {transcripts_dir}")

    print(f"\nFound and processed {len(content_manifest)} relevant files.")
    
    # Print file type statistics
    print("\nFile type statistics:")
    for file_type, count in file_type_counts.items():
        if file_type == "transcript":
            print(f"  Transcript files: {count}")
        else:
            print(f"  {file_type} files: {count}")
    
    # Verify total count
    total_count = sum(file_type_counts.values())
    print(f"\nTotal files counted: {total_count}")
    if total_count != len(content_manifest):
        print(f"WARNING: Count mismatch! Manifest has {len(content_manifest)} entries.")
    else:
        print("Count matches manifest entries.")


    # Save the manifest
    try:
        with open(manifest_file, 'w', encoding='utf-8') as f:
            json.dump(content_manifest, f, indent=4)
        print(f"Successfully created manifest file: {manifest_file}")
    except Exception as e:
        print(f"Error writing manifest file: {e}")

if __name__ == "__main__":
    create_content_manifest()
