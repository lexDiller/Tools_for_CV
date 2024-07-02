import os


def rename_files_in_folder(folder):
    files = os.listdir(folder)
    for i, file in enumerate(files):
        if os.path.isfile(os.path.join(folder, file)):
            file_ext = os.path.splitext(file)[1]
            new_name = f"frame_{i}{file_ext}"
            os.rename(os.path.join(folder, file), os.path.join(folder, new_name))


#  Delete files with exception .jpg
def delete_files_in_folder_except_jpg(folder):
    files = os.listdir(folder)
    for file in files:
        if os.path.isfile(os.path.join(folder, file)):
            if not file.endswith(".jpg"):
                os.remove(os.path.join(folder, file))


# Delete the space in the beginning and end of the strings
def remove_trailing_space_from_files(folder):
    files = os.listdir(folder)  # List of files
    for file in files:
        if os.path.isfile(os.path.join(folder, file)) and file.endswith(
                ".txt"):  # Check that this is .txt
            file_path = os.path.join(folder, file)
            with open(file_path, "r") as f:
                lines = f.readlines()

            with open(file_path, "w") as f:
                for line in lines:
                    line = line.rstrip()  # Delete space
                    f.write(line + "\n")
