import re
import matplotlib.pyplot as plt

def plot_results(nb_boxes, mabo_results):
    plt.plot(nb_boxes, mabo_results, marker='o')
    plt.xlabel('Number of Boxes')
    plt.ylabel('MABO')
    plt.title('MABO vs Number of Boxes')
    plt.grid(True)
    plt.savefig('plot.png')

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

def main():
    input_file_path = 'sorted_mabo_results.txt'
    
    # Read the results from the input file
    nb_boxes, mabo_results = read_results_from_file(input_file_path)
    
    # Plot the results
    plot_results(nb_boxes, mabo_results)
    
    print(f"Plot created from {input_file_path}")

if __name__ == "__main__":
    main()