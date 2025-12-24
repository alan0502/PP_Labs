#!/usr/bin/env python3
import sys

def compare_files(file1, file2):
    with open(file1, "rb") as f1, open(file2, "rb") as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()

    if lines1 == lines2:
        print(f"✅ {file1} 和 {file2} 完全相同")
        return

    print(f"⚠️ {file1} 和 {file2} 有差異：")
    max_len = max(len(lines1), len(lines2))
    for i in range(max_len):
        l1 = lines1[i].rstrip(b"\r\n") if i < len(lines1) else b"<NO LINE>"
        l2 = lines2[i].rstrip(b"\r\n") if i < len(lines2) else b"<NO LINE>"
        if l1 != l2:
            # 嘗試用 UTF-8 轉字串，不行就 fallback
            try:
                s1 = l1.decode("utf-8")
            except:
                s1 = str(l1)
            try:
                s2 = l2.decode("utf-8")
            except:
                s2 = str(l2)

            print(f"第 {i+1} 行不同：")
            print(f"  {file1}: {s1}")
            print(f"  {file2}: {s2}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"用法: {sys.argv[0]} file1.out file2.out")
        sys.exit(1)
    compare_files(sys.argv[1], sys.argv[2])
