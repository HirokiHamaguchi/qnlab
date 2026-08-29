import glob
import os


def delete_temp_data(method, precision):
    temp_data_path = os.path.join("data", "temp")
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
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        except OSError as e:
            print(f"Error deleting {file_path}: {e}")


if __name__ == "__main__":
    for method in ["NTRQN", "NTRQN-MS"]:
        for precision in [16, 32, 64, "noisy"]:
            delete_temp_data(method=method, precision=precision)
