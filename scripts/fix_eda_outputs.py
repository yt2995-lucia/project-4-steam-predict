"""
fix_eda_outputs.py — 一次性修复 Group A / Group B EDA 输出的位置和列名问题

修复内容:
1. unsupervised_output.csv 里的 _x/_y 列名 bug
   - topic_0_x ... topic_9_x: 全是空的 NaN, 删掉
   - topic_0_y ... topic_9_y: 真正的 LDA topic 分布, 改名为 topic_0..topic_9
   - 保存到正确位置 data/processed/unsupervised_output.csv
2. 把 Group B 的 11 张 PNG 从 GroupB/ 复制到 figures/groupB/
   - fig6 / fig7 加 'B_' 前缀, 避免和 Group A 的 fig6 撞名
3. 创建 figures/groupA/ 目录占位 (Group A 脚本跑完会自动填进去)

运行方式 (在项目根目录):
    python scripts/fix_eda_outputs.py
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def fix_unsupervised_output() -> None:
    """修 unsupervised_output.csv 的 _x/_y 列名 bug。"""
    src = PROJECT_ROOT / "unsupervised_output.csv"
    dst = PROJECT_ROOT / "data" / "processed" / "unsupervised_output.csv"

    if not src.exists():
        print(f"[skip] {src} not found")
        return

    df = pd.read_csv(src)
    print(f"[load] {src}  shape={df.shape}")
    print(f"[load] columns: {list(df.columns)}")

    # 删掉空的 topic_*_x 列 (注意: 不要碰 umap_x, 那个是真正的坐标)
    x_cols = [c for c in df.columns if c.startswith("topic_") and c.endswith("_x")]
    if x_cols:
        all_empty = df[x_cols].isna().all().all()
        if not all_empty:
            print(f"[warn] topic_*_x columns are not all empty, will NOT drop them")
            print(df[x_cols].head())
            return
        df = df.drop(columns=x_cols)
        print(f"[drop] {len(x_cols)} empty topic_*_x columns")

    # 重命名 topic_*_y 列为 topic_* (注意: 不要碰 umap_y)
    rename_map = {
        c: c.removesuffix("_y")
        for c in df.columns
        if c.startswith("topic_") and c.endswith("_y")
    }
    if rename_map:
        df = df.rename(columns=rename_map)
        print(f"[rename] {len(rename_map)} topic_*_y columns -> topic_*")

    # 保存到 data/processed/
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst, index=False)
    print(f"[save] {dst}  shape={df.shape}")
    print(f"[save] columns: {list(df.columns)}")

    # 简单 sanity check: topic 列加起来约等于 1
    topic_cols = [c for c in df.columns if c.startswith("topic_")]
    if topic_cols:
        row_sums = df[topic_cols].sum(axis=1)
        print(
            f"[check] topic columns sum per row: "
            f"min={row_sums.min():.4f}, max={row_sums.max():.4f}, "
            f"median={row_sums.median():.4f}"
        )


def move_groupb_figures() -> None:
    """把 Group B 的 PNG 移到 figures/groupB/, 解决 fig6 撞名问题。"""
    src_dir = PROJECT_ROOT / "GroupB"
    dst_dir = PROJECT_ROOT / "figures" / "groupB"

    if not src_dir.exists():
        print(f"[skip] {src_dir} not found")
        return

    dst_dir.mkdir(parents=True, exist_ok=True)

    # fig6 / fig7 跟 Group A 撞编号, 加 B_ 前缀; fig8/9/10 是 Group B 独有的
    rename_map = {
        "fig6_success_rate_by_release_month.png": "figB6_success_rate_by_release_month.png",
        "fig7_language_vs_success.png": "figB7_language_vs_success.png",
    }

    pngs = sorted(src_dir.glob("*.png"))
    for src in pngs:
        new_name = rename_map.get(src.name, src.name)
        dst = dst_dir / new_name
        shutil.copy2(src, dst)
        print(f"[copy] {src.name} -> figures/groupB/{new_name}")

    print(f"[done] {len(pngs)} PNGs copied to {dst_dir}")


def ensure_groupa_figdir() -> None:
    d = PROJECT_ROOT / "figures" / "groupA"
    d.mkdir(parents=True, exist_ok=True)
    print(f"[ensure] {d} exists (Group A script will populate it)")


def fix_groupb_notebook_path() -> None:
    """把 GroupB.ipynb 里的硬编码绝对路径改成相对路径, 方便后续复现。"""
    nb_path = PROJECT_ROOT / "GroupB" / "GroupB.ipynb"
    if not nb_path.exists():
        print(f"[skip] {nb_path} not found")
        return

    text = nb_path.read_text(encoding="utf-8")
    # notebook 里 string 是 JSON-escape 的, 反斜杠加引号
    bad = '/Users/gaorunji/Desktop/project-4-steam-predict/data/interim/cleaned.csv'
    good = 'data/interim/cleaned.csv'

    if bad not in text:
        print(f"[skip] hardcoded path not found in notebook (already fixed?)")
        return

    new_text = text.replace(bad, good)
    n_replacements = text.count(bad)
    nb_path.write_text(new_text, encoding="utf-8")
    print(f"[fix]  replaced {n_replacements} hardcoded path(s) in {nb_path.name}")


def main() -> None:
    print("=" * 60)
    print("fix_eda_outputs.py")
    print("=" * 60)

    print("\n--- Step 1: fix unsupervised_output.csv ---")
    fix_unsupervised_output()

    print("\n--- Step 2: move Group B figures ---")
    move_groupb_figures()

    print("\n--- Step 3: ensure figures/groupA/ exists ---")
    ensure_groupa_figdir()

    print("\n--- Step 4: fix Group B notebook hardcoded path ---")
    fix_groupb_notebook_path()

    print("\nAll done. Next steps:")
    print("  1. Run Group A: python src/groupA_descriptive_eda.py")
    print("     (path inside groupA script has been fixed to data/interim/cleaned.csv)")
    print("  2. Verify data/processed/unsupervised_output.csv has clean column names")


if __name__ == "__main__":
    main()
