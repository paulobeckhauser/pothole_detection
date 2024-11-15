

def count_xml_files(folder_path):
    """
    Count the number of .xml files in a folder.

    Parameters:
    folder_path (str): Path to the folder.

    Returns:
    int: Number of .xml files in the folder.
    """
    xml_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.xml')]
    return len(xml_files)

def extract_number_from_string(s):
    match = re.search(r'\d+', s)
    if match:
        return int(match.group())
    return None
