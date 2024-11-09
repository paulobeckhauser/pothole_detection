import re

def extract_number_from_string(s):
    match = re.search(r'\d+', s)
    if match:
        return int(match.group())
    return None

def read_results_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    nb_boxes = []
    mabo_results = []
    
    for line in lines:
        if line.startswith("Number boxes:"):
            nb_boxes = [int(x) for x in re.findall(r'\d+', line)]
        elif line.startswith("MABO results:"):
            mabo_results = [float(x) for x in re.findall(r'\d+\.\d+', line)]
    
    return nb_boxes, mabo_results

def write_sorted_results_to_file(file_path, nb_boxes_sorted, mabo_results_sorted):
    with open(file_path, 'w') as file:
        file.write(f"Number boxes: {nb_boxes_sorted}\n")
        file.write(f"MABO results: {mabo_results_sorted}\n")

def main():
    input_file_path = 'mabo_results.txt'
    output_file_path = 'sorted_mabo_results.txt'

    # Read the results from the input file
    nb_boxes, mabo_results = read_results_from_file(input_file_path)

    # Ensure the values in nb_boxes and mabo_results are in order and connected
    paired_results = list(zip(nb_boxes, mabo_results))
    paired_results.sort()
    nb_boxes_sorted, mabo_results_sorted = zip(*paired_results)
    nb_boxes_sorted = list(nb_boxes_sorted)
    mabo_results_sorted = list(mabo_results_sorted)
    
    # Write the sorted results to the output file
    write_sorted_results_to_file(output_file_path, nb_boxes_sorted, mabo_results_sorted)
    print(f"Sorted results have been written to {output_file_path}")

if __name__ == "__main__":
    main()