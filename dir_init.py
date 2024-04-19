import os


def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print("Directory created successfully.")
    else:
        print("Directory already exists.")


directory_path = "output"
directory2_path = "data"
create_directory(directory_path)
create_directory(directory2_path)
