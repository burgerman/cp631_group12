import os


def delete_files_in_directory(directory_path):
    if os.path.exists(directory_path):
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                    print("Deleted:", file_path)
            except Exception as e:
                print("Error deleting:", e)
        print("All files deleted successfully.")
    else:
        print("Directory does not exist.")


directory_path = "output"
directory2_path = "data"
delete_files_in_directory(directory_path)
delete_files_in_directory(directory2_path)
