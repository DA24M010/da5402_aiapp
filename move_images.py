import os
import random
import shutil

def move_images(src_folder, dst_folder, keep_count=5000):
    os.makedirs(dst_folder, exist_ok=True)

    all_files = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))]
    total_files = len(all_files)

    if total_files <= keep_count:
        print(f"Folder '{src_folder}' already has {total_files} or fewer images. Skipping.")
        return

    move_count = total_files - keep_count
    to_move = random.sample(all_files, move_count)

    for file_name in to_move:
        src_path = os.path.join(src_folder, file_name)
        dst_path = os.path.join(dst_folder, file_name)
        shutil.move(src_path, dst_path)

    print(f"Moved {move_count} images from '{src_folder}' to '{dst_folder}'.")

def main():
    base_src = "./data/raw_data"
    base_dst = "./temp"
    categories = ["positive", "negative"]

    for category in categories:
        src_folder = os.path.join(base_src, category)
        dst_folder = os.path.join(base_dst, category)
        move_images(src_folder, dst_folder, keep_count=5000)

if __name__ == "__main__":
    main()
