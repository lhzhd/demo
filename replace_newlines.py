import os
import sys


def replace_newlines_with_periods(folder_path):
    """
    将指定文件夹中所有txt文件中的换行符替换为句号。
    """
    if not os.path.isdir(folder_path):
        print(f"错误: 路径 '{folder_path}' 不是一个有效的文件夹。")
        return

    # 获取文件夹中所有 .txt 文件
    txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

    if not txt_files:
        print(f"在文件夹 '{folder_path}' 中未找到任何 .txt 文件。")
        return

    print(f"找到 {len(txt_files)} 个 txt 文件，开始处理...")

    for filename in txt_files:
        file_path = os.path.join(folder_path, filename)

        try:
            # 读取文件内容，使用 utf-8 编码，如果报错可以尝试 'gbk' 或其他编码
            # errors='ignore' 或 'replace' 可以防止因编码问题导致程序崩溃
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # 替换换行符
            # 这里同时处理常见的三种换行符情况，确保替换干净
            # 先处理 \r\n (Windows), 再处理 \r (旧Mac), 最后处理 \n (Unix/Linux/Mac)
            # 或者直接简单地将所有空白换行序列替换，但为了精确对应“换行符变句号”，
            # 我们通常直接替换 \n 即可覆盖大多数情况。
            # 为了稳妥，我们将 \r\n 和 \r 也视为换行进行替换。

            modified_content = content.replace('\r\n', '。').replace('\r', '。').replace('\n', '。')

            # 写回文件
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(modified_content)

            print(f"已处理: {filename}")

        except Exception as e:
            print(f"处理文件 {filename} 时出错: {e}")

    print("所有文件处理完毕。")


if __name__ == "__main__":
    # 使用方法 1: 直接在代码中修改下面的路径
    # target_folder = r"C:\Users\YourName\Documents\my_texts"
    # 或者
    # target_folder = "/home/user/documents/my_texts"

    # 使用方法 2: 通过命令行参数传入文件夹路径
    if len(sys.argv) > 1:
        target_folder = sys.argv[1]
    else:
        # 如果没有传入参数，默认使用当前脚本所在的文件夹
        target_folder = os.getcwd()
        print(f"未指定文件夹，默认使用当前目录: {target_folder}")
        response = input("确定要处理当前目录下的所有txt文件吗？(y/n): ")
        if response.lower() != 'y':
            sys.exit(0)

    replace_newlines_with_periods(target_folder)