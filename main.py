from dataloader import DataLoader

def main():
    data_instance = DataLoader(name="Data Loader")
    data_instance.load_dataset(data_folder='annotated_images', output_folder='output')

if __name__ == "__main__":
    main()
