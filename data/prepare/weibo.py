import pickle
import os
import requests
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


class WeiboDataProcessor:
    """
    Modular, engineering-style processor for Weibo dataset.
    Outputs DataFrame with columns: ['post_id', 'image_name', 'text', 'label'].
    """

    def __init__(self, image_dir, tweets_dir):
        self.image_dir = image_dir
        self.tweets_dir = tweets_dir
        # 确保图片目录存在
        os.makedirs(self.image_dir, exist_ok=True)
        self.image_map = self._load_images()

    def _load_images(self):
        """
        Strict Mode: Parallelly open and convert every image to RGB.
        This ensures 100% data integrity but takes longer to initialize.
        """
        from PIL import Image, ImageFile
        # 允许加载截断图片（可选，如果你希望截断图也算坏图，请注释掉这行）
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        image_set = set()
        print(f"Deep scanning images in {self.image_dir} (This may take a while)...")

        if not os.path.exists(self.image_dir):
            return image_set

        files = os.listdir(self.image_dir)

        # 定义一个验证函数，专门给线程池用
        def verify_worker(filename):
            file_path = os.path.join(self.image_dir, filename)
            try:
                # 1. 基础大小检查
                if os.path.getsize(file_path) == 0:
                    os.remove(file_path)
                    return None

                # 2. 深度像素检查 (最耗时的一步)
                with Image.open(file_path) as img:
                    # 强制解压所有像素，检测坏块
                    img.convert('RGB')

                return filename  # 成功返回文件名
            except Exception as e:
                # print(f"Deleting corrupt image: {filename} ({e})")
                try:
                    os.remove(file_path)
                except:
                    pass
                return None

        # 使用多线程并发检查
        with ThreadPoolExecutor(max_workers=25) as executor:
            # 提交所有任务
            futures = {executor.submit(verify_worker, f): f for f in files}

            # 使用 tqdm 显示进度条
            for future in tqdm(as_completed(futures), total=len(files), desc="Deep Verifying"):
                result = future.result()
                if result:
                    image_set.add(result)

        print(f"Deep scan finished. Found {len(image_set)} valid images.")
        return image_set

    def _download_single_image(self, task):
        """
        Worker function for downloading a single image.
        task: (url, save_path, filename)
        """
        url, save_path, filename = task

        if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
            return filename, True  # 已经存在

        try:
            # 使用 requests 替代 wget，超时控制更灵活
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, stream=True, timeout=10)
            if response.status_code == 200:
                with open(save_path, 'wb') as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
                return filename, True
            else:
                return filename, False
        except Exception as e:
            # 下载失败删除空文件
            if os.path.exists(save_path):
                os.remove(save_path)
            return filename, False

    def _process_downloads(self, missing_images):
        """
        Execute parallel downloads.
        missing_images: list of (url, filename)
        """
        if not missing_images:
            return

        print(f"Starting parallel download for {len(missing_images)} images...")
        tasks = []
        for url, fname in missing_images:
            save_path = os.path.join(self.image_dir, fname)
            tasks.append((url, save_path, fname))

        # 开启多线程下载，max_workers 建议设置为 CPU 核心数 * 2 或 4
        with ThreadPoolExecutor(max_workers=30) as executor:
            # 提交任务
            future_to_file = {executor.submit(self._download_single_image, task): task for task in tasks}

            for future in tqdm(as_completed(future_to_file), total=len(tasks), desc="Downloading"):
                fname, success = future.result()
                if success:
                    self.image_map.add(fname)

        print("Download phase finished.")

    def _read_posts(self, flag):
        """
        Reads posts for a given flag.
        Step 1: Parse lines and identify MISSING images.
        Step 2: Download missing images in parallel.
        Step 3: Build final records.
        """
        pre_path = self.tweets_dir
        file_map = {
            'train_nonrumor': os.path.join(pre_path, "train_nonrumor.txt"),
            'train_rumor': os.path.join(pre_path, "train_rumor.txt"),
            'test_nonrumor': os.path.join(pre_path, "test_nonrumor.txt"),
            'test_rumor': os.path.join(pre_path, "test_rumor.txt")
        }

        keys = ['train_nonrumor', 'train_rumor'] if flag == 'train' else ['test_nonrumor', 'test_rumor']

        parsed_data = []  # 暂存解析结果 (post_id, img_urls_list, text, label)
        missing_downloads = set()  # 待下载列表 (url, filename)，使用 set 去重

        print(f"Parsing text files for {flag}...")
        for key in keys:
            label = 0 if 'nonrumor' in key else 1
            filepath = file_map[key]
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for i in range(0, len(lines), 3):
                post_id = lines[i].strip().split('|')[0]
                img_line = lines[i + 1].strip().split('|')
                text_line = lines[i + 2].strip()

                # 提取这一条微博的所有 potential images
                potential_imgs = []
                for img_url in img_line:
                    if img_url == 'null': continue

                    tmp_img_name = img_url.split('/')[-1]
                    # bec07d16jw1eaq6dokq3jj20c8094mxs.jpg is not available
                    if tmp_img_name == 'bec07d16jw1eaq6dokq3jj20c8094mxs.jpg':
                        continue
                    potential_imgs.append((img_url, tmp_img_name))

                    # 如果本地没有，加入待下载队列
                    if tmp_img_name not in self.image_map:
                        html_path = "https://image.baidu.com/search/down?url="
                        full_url = html_path + img_url
                        missing_downloads.add((full_url, tmp_img_name))

                parsed_data.append({
                    'post_id': post_id,
                    'potential_imgs': potential_imgs,
                    'text': text_line,
                    'label': label
                })

        # --- STEP 2: 执行并行下载 ---
        if missing_downloads:
            self._process_downloads(list(missing_downloads))

        # --- STEP 3: 构建最终数据 ---
        records = []
        print(f"Building final dataset for {flag}...")
        for item in parsed_data:
            final_img_name = None

            # 找到第一个真实存在的图片
            for _, img_name in item['potential_imgs']:
                if img_name in self.image_map:
                    final_img_name = img_name
                    break  # 只需要一张图

            if final_img_name:
                records.append((item['post_id'], final_img_name, item['text'], item['label']))

        return records

    def build_pkl(self, flag):
        """
        Builds pandas DataFrame
        """
        posts = self._read_posts(flag)
        data = {'post_id': [], 'image_name': [], 'text': [], 'label': []}
        for post_id, img_name, text_line, label in posts:
            data['post_id'].append(post_id)
            data['text'].append(text_line)
            data['label'].append(label)
            data['image_name'].append(img_name)

        print(f"Get {len(posts)} posts for {flag} set.")
        output_path = os.path.join(args.output_dir, f"{flag}_data.pkl")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved {flag} data to {output_path}")

        args_file = os.path.join(args.output_dir, f"{flag}_args.txt")
        with open(args_file, 'w') as f:
            f.write(str(vars(args)))


def main(args):
    processor = WeiboDataProcessor(
        image_dir=args.images_path,
        tweets_dir=args.tweets_dir,
    )
    for split in ['train', 'test']:
        processor.build_pkl(split)


def get_args():
    parser = argparse.ArgumentParser(description="Prepare Weibo 数据")
    parser.add_argument("--images_path", type=str, default="./data/raw/weibo/images",
                        help="图片目录")
    parser.add_argument("--tweets_dir", type=str, default="./data/raw/weibo/tweets",
                        help="微博文本文件目录")
    parser.add_argument("--output_dir", type=str, default="./data/processed/weibo/raw/",
                        help="输出目录，用于保存处理后的数据")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)


# python ./data/prepare/weibo.py

# python ./data/prepare/weibo.py --images_path data/raw/weibo/images --tweets_dir data/raw/weibo/tweets --output_dir data/processed/weibo/raw/
