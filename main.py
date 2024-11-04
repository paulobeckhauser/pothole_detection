from dataloader import DataLoader

def main():
    data_instance = DataLoader(name="Data Loader")
    data_instance.load_dataset(data_folder='images', output_folder = 'annotated_images')

if __name__ == "__main__":
    main()
