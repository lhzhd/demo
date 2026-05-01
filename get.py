import os
import sys


def fix_word_list():
    # 1. 自动获取脚本所在的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, "word.txt")
    output_path = os.path.join(script_dir, "tcm_keywords_gen.py")

    print(f"🔍 正在查找文件: {input_path}")

    # 2. 检查文件是否存在
    if not os.path.exists(input_path):
        print(f"❌ 错误：在当前目录下找不到 'word.txt'！")
        print(f"   请确保 word.txt 和这个脚本在同一个文件夹里。")
        input("按回车键退出...")
        return

    try:
        # 3. 读取并清洗词汇 (增加 errors='ignore' 防止编码报错)
        words = []
        with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                word = line.strip()
                if word:  # 去除空行
                    words.append(word)

        if len(words) == 0:
            print("❌ 错误：word.txt 是空的，或者读取失败。")
            return

        print(f"✅ 成功读取 {len(words)} 个词汇。正在生成...")

        # 4. 格式化
        formatted_lines = []
        line_buffer = []

        for i, word in enumerate(words):
            formatted_word = f'"{word}",'
            line_buffer.append(formatted_word)

            # 每 5 个词换一行
            if (i + 1) % 5 == 0:
                formatted_lines.append("    " + " ".join(line_buffer))
                line_buffer = []

        # 处理剩余的词
        if line_buffer:
            formatted_lines.append("    " + " ".join(line_buffer))

        # 5. 拼接最终内容
        final_content = "tcm_keywords = [\n"
        final_content += "\n".join(formatted_lines)
        final_content += "\n]"

        # 6. 写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_content)

        print(f"🚀 成功！文件已生成：{output_path}")

        # 7. 自动打开文件夹 (Windows)
        try:
            os.startfile(os.path.dirname(output_path))
            print("ℹ️ 已自动打开文件夹，请查看 tcm_keywords_gen.py")
        except:
            print("ℹ️ 生成成功，但未自动打开文件夹，请在当前目录查找。")

    except Exception as e:
        print(f"❌ 发生未知错误: {e}")
        input("按回车键退出...")


if __name__ == "__main__":
    fix_word_list()