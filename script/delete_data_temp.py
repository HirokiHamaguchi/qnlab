# delete data/temp/16/*/NTQN.npz

import glob
import os


def delete_temp_data():
    temp_data_path = os.path.join("data", "temp")
    precision = 16
    method = "NTQN"
    pattern = os.path.join(temp_data_path, str(precision), "*", f"{method}.npz")
    files = glob.glob(pattern)

    print(f"Found {len(files)} files to delete.")

    # show the files to be deleted
    print("Files to be deleted:")
    for file_path in files:
        print(file_path)

    confirm = input("Are you sure you want to delete these files? (y/n): ")
    if confirm.lower() != "y":
        print("Deletion cancelled.")
        return

    for file_path in files:
        try:
            print(f"Deleted: {file_path}")
            os.remove(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")


if __name__ == "__main__":
    delete_temp_data()
