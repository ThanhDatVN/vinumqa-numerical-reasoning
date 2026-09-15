# -*- coding: utf-8 -*-
"""Ghi URL repo GitHub của bạn vào notebook + tài liệu, rồi (tuỳ chọn) khởi tạo git.

Chạy MỘT LẦN sau khi tạo repo trống trên GitHub:

    python tools/set_repo.py https://github.com/ten-cua-ban/vinumqa-ladder --init
    git push -u origin main

Sau đó mở notebook trên Colab là chạy được ngay, không phải sửa dòng nào nữa:
cell cấu hình đã biết URL, tự clone code + dữ liệu về.

Chạy lại lúc nào cũng được (ví dụ khi đổi tên repo) — script thay URL cũ bằng URL mới.
"""
import argparse
import io
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = ("README.md", "HUONG_DAN_COLAB.md", "KE_HOACH_THU_NGHIEM.md",
        "CAC_BUOC_THUC_HIEN.md")

# https://github.com/owner/name(.git) | git@github.com:owner/name(.git) | owner/name
_URL_RE = re.compile(
    r"^(?:(?:https?://)?(?:[^@/]+@)?github\.com[:/])?"
    r"(?P<owner>[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?)/"
    r"(?P<name>[A-Za-z0-9._-]+?)(?:\.git)?/?$"
)


def parse_repo(arg):
    """'…/owner/name.git' → ('owner', 'name'). Báo lỗi rõ ràng nếu sai dạng."""
    m = _URL_RE.match(arg.strip())
    if not m:
        sys.exit(
            f"Không đọc được URL repo: {arg!r}\n"
            "Dùng một trong các dạng:\n"
            "    https://github.com/ten-cua-ban/vinumqa-ladder\n"
            "    git@github.com:ten-cua-ban/vinumqa-ladder.git\n"
            "    ten-cua-ban/vinumqa-ladder")
    return m.group("owner"), m.group("name")


def stamp_notebooks(owner, name):
    """Điền GITHUB_REPO vào mọi cell cấu hình + gắn huy hiệu Open in Colab."""
    url = f"https://github.com/{owner}/{name}"
    nb_dir = os.path.join(ROOT, "notebooks")
    n_cell = 0
    for fn in sorted(os.listdir(nb_dir)):
        if not fn.endswith(".ipynb"):
            continue
        path = os.path.join(nb_dir, fn)
        nb = json.load(io.open(path, encoding="utf-8"))
        hit = 0
        for cell in nb["cells"]:
            src = "".join(cell["source"])
            new = src
            if cell["cell_type"] == "code":
                new = re.sub(r'(?m)^GITHUB_REPO = ".*"$',
                             f'GITHUB_REPO = "{url}"', new)
            new = re.sub(r"(colab\.research\.google\.com/github/)[^/]+/[^/]+(/blob/)",
                         rf"\g<1>{owner}/{name}\g<2>", new)
            if new != src:
                cell["source"] = new.splitlines(keepends=True)
                hit += 1
        if hit:
            with io.open(path, "w", encoding="utf-8", newline="\n") as f:
                json.dump(nb, f, ensure_ascii=False, indent=1)
                f.write("\n")
            n_cell += hit
            print(f"  notebooks/{fn}: {hit} cell")
    return n_cell


def stamp_docs(owner, name):
    """Cập nhật mọi liên kết github/colab trong tài liệu."""
    n_file = 0
    for fn in DOCS:
        path = os.path.join(ROOT, fn)
        if not os.path.exists(path):
            continue
        src = io.open(path, encoding="utf-8").read()
        new = re.sub(r"(?<!@)(https://github\.com/)[A-Za-z0-9._-]+/[A-Za-z0-9._-]+?(\.git)?(?=[)\s\"'`]|$)",
                     rf"\g<1>{owner}/{name}\g<2>", src)
        new = re.sub(r"(colab\.research\.google\.com/github/)[^/]+/[^/]+(/blob/)",
                     rf"\g<1>{owner}/{name}\g<2>", new)
        if new != src:
            io.open(path, "w", encoding="utf-8", newline="\n").write(new)
            n_file += 1
            print(f"  {fn}")
    return n_file


def git(*args, check=True):
    r = subprocess.run(["git", "-C", ROOT, *args], capture_output=True, text=True)
    if check and r.returncode:
        sys.exit(f"git {' '.join(args)} lỗi:\n{r.stderr.strip()}")
    return r


def init_repo(remote_url):
    """git init + commit đầu + gắn remote. Không push — để bạn tự kiểm tra rồi push."""
    if os.path.isdir(os.path.join(ROOT, ".git")):
        print("[GIT] đã là repo — bỏ qua init")
    else:
        git("init", "-q")
        print("[GIT] đã init")
    git("add", "-A")
    if git("diff", "--cached", "--quiet", check=False).returncode == 0:
        print("[GIT] không có gì để commit")
    else:
        git("commit", "-q", "-m", "ViNumQA: lộ trình thí nghiệm 5 nấc trên Qwen3-8B")
        print("[GIT] đã commit")
    git("branch", "-M", "main")
    if git("remote", "get-url", "origin", check=False).returncode == 0:
        git("remote", "set-url", "origin", remote_url)
    else:
        git("remote", "add", "origin", remote_url)
    print(f"[GIT] origin → {remote_url}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repo", help="URL repo GitHub, ví dụ https://github.com/ten/vinumqa-ladder")
    ap.add_argument("--init", action="store_true",
                    help="git init + commit đầu tiên + gắn remote origin")
    a = ap.parse_args()

    owner, name = parse_repo(a.repo)
    url = f"https://github.com/{owner}/{name}"
    print(f"Repo: {url}\n")

    n_cell = stamp_notebooks(owner, name)
    n_doc = stamp_docs(owner, name)
    print(f"\nĐã điền URL vào {n_cell} cell notebook và {n_doc} tài liệu.")

    if a.init:
        remote = a.repo.strip() if a.repo.strip().startswith("git@") else url + ".git"
        init_repo(remote)
        print("\nCòn một bước cuối:\n    git push -u origin main")
    else:
        print("\nTiếp theo:\n"
              f"    git init && git add -A && git commit -m \"lần đầu\"\n"
              f"    git branch -M main\n"
              f"    git remote add origin {url}.git\n"
              f"    git push -u origin main\n"
              "(hoặc chạy lại script này kèm --init để làm hộ)")


if __name__ == "__main__":
    main()
