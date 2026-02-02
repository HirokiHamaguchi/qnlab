import glob
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 対象ファイル
files = glob.glob("[1-4]_*.tex")

print(files)

is_ok = input("Proceed with replacements? (y/n): ")
if is_ok.lower() != "y":
    print("Aborted.")
    exit()

for filepath in files:
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    original_content = content

    # \norm{...}を\lVert ... \rVertに置き換え
    # ネストされたブレースに対応
    def replace_norm(text):
        result = []
        i = 0
        while i < len(text):
            if text[i : i + 5] == "\\norm":
                # \norm{...}の中身を見つける
                if i + 5 < len(text) and text[i + 5] == "{":
                    # ネストされたブレースをカウント
                    depth = 0
                    j = i + 6
                    while j < len(text):
                        if text[j] == "{":
                            depth += 1
                        elif text[j] == "}":
                            if depth == 0:
                                # 終端を見つけた
                                inner = text[i + 6 : j]
                                result.append(f"\\lVert {inner} \\rVert")
                                i = j + 1
                                break
                            else:
                                depth -= 1
                        j += 1
                    else:
                        result.append(text[i])
                        i += 1
                else:
                    result.append(text[i])
                    i += 1
            else:
                result.append(text[i])
                i += 1
        return "".join(result)

    content = replace_norm(content)

    # \qty(...)を\left( ... \right)に置き換え（ネスト対応）
    def replace_qty(text):
        result = []
        i = 0
        while i < len(text):
            if text[i : i + 5] == "\\qty(":
                # \qty( ... ) の中身を見つける
                if i + 5 < len(text) and text[i + 5] == "(":
                    depth = 0
                    j = i + 6
                    while j < len(text):
                        if text[j] == "(":
                            depth += 1
                        elif text[j] == ")":
                            if depth == 0:
                                inner = text[i + 6 : j]
                                result.append(f"\\left({inner}\\right)")
                                i = j + 1
                                break
                            else:
                                depth -= 1
                        j += 1
                    else:
                        result.append(text[i])
                        i += 1
                else:
                    result.append(text[i])
                    i += 1
            else:
                result.append(text[i])
                i += 1
        return "".join(result)

    content = replace_qty(content)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Updated: {filepath}")
