# import dataset from kaggle
def download_data():
    path = kagglehub.dataset_download("andrewmvd/lung-and-colon-cancer-histopathological-images")
    print("Path to dataset files:", path)


download_data()
