import pip
from pathlib import Path

def find_fpzip_path():
    # Get the path to the installed fpzip package
    try:
        fpzip_path = next(Path(pip.__file__).parents[2].glob('*/fpzip'))
        return str(fpzip_path)
    except StopIteration:
        return None

if __name__ == "__main__":
    fpzip_path = find_fpzip_path()
    if fpzip_path:
        print("Path to fpzip:", fpzip_path)
    else:
        print("fpzip not found.")