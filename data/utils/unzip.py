import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

# 配置路径
SOURCE_ROOT = "./data/raw/"
DEST_ROOT = "./data/raw/"


# 查找所有 zip 文件
def find_zips(root_dir):
    zip_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".zip"):
                zip_files.append(os.path.join(root, file))
    return zip_files


def unzip_task(src_path):
    try:
        # 1. 计算目标目录
        # 并且去掉文件名，只保留目录结构
        dst_dir = os.path.dirname(src_path)  # 目标目录与源目录相同

        # 2. 如果目标目录不存在，创建它
        os.makedirs(dst_dir, exist_ok=True)

        # 3. 调用系统 unzip 命令
        # -q: 安静模式 (减少IO输出)
        # -o: 覆盖不询问
        # -d: 指定输出目录
        cmd = ["unzip", "-q", "-o", src_path, "-d", dst_dir]

        print(f"🚀 开始解压: {os.path.basename(src_path)} -> {dst_dir}")
        subprocess.run(cmd, check=True)
        print(f"✅ 完成: {os.path.basename(src_path)}")
        return True
    except Exception as e:
        print(f"❌ 失败: {src_path}, 错误: {e}")
        return False


if __name__ == "__main__":
    start_time = time.time()

    # 获取文件列表
    zips = find_zips(SOURCE_ROOT)
    print(f"找到 {len(zips)} 个压缩包，准备利用并行解压...")

    # 核心：使用线程池并发处理
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(unzip_task, zips))

    end_time = time.time()
    print(f"\n🎉 全部任务结束！耗时: {end_time - start_time:.2f} 秒")
    print(f"输出位置: {DEST_ROOT}")

# python data/utils/unzip.py