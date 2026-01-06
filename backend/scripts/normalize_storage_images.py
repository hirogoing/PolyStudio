"""
把 backend/storage/images 下的历史图片批量归一化到 sRGB，并移除 ICC profile。

背景：
- 聊天框用 <img> 显示图片：浏览器通常会做 ICC/广色域色彩管理
- Excalidraw 画板是 <canvas> 渲染：很多情况下工作在 sRGB，色彩管理更弱
因此同一张“带 Display-P3/ICC 的图片”会出现：聊天正常、画板偏色、但把原图拷贝/新标签页打开又正常。

这个脚本会把存量图片覆盖写回为 sRGB（像素值已转换），从根源减少差异。

用法：
  cd backend
  python scripts/normalize_storage_images.py
"""

from __future__ import annotations

import os
import sys
from io import BytesIO
from pathlib import Path


def main() -> int:
    try:
        from PIL import Image, ImageCms  # type: ignore
    except Exception as e:
        print("❌ 未安装 Pillow，无法归一化。请先安装：pip install -r requirements.txt")
        print(f"   详细错误: {e}")
        return 2

    backend_dir = Path(__file__).resolve().parents[1]
    images_dir = backend_dir / "storage" / "images"
    if not images_dir.exists():
        print(f"⚠️ 未找到目录：{images_dir}")
        return 0

    exts = {".png", ".jpg", ".jpeg", ".webp"}
    files = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    if not files:
        print("ℹ️ 没有需要处理的图片文件。")
        return 0

    ok = 0
    changed = 0
    failed = 0

    for p in sorted(files, key=lambda x: x.stat().st_mtime):
        try:
            raw = p.read_bytes()
            im = Image.open(BytesIO(raw))
            im.load()

            # ICC/广色域 -> sRGB
            icc = getattr(im, "info", {}).get("icc_profile")
            if icc:
                try:
                    src_profile = ImageCms.ImageCmsProfile(BytesIO(icc))
                    dst_profile = ImageCms.createProfile("sRGB")
                    output_mode = "RGBA" if (
                        im.mode in ("RGBA", "LA") or ("transparency" in getattr(im, "info", {}))
                    ) else "RGB"
                    im = ImageCms.profileToProfile(im, src_profile, dst_profile, outputMode=output_mode)
                except Exception:
                    # 转换失败也继续走后续流程
                    pass

            # 移除 ICC
            try:
                if getattr(im, "info", None) and "icc_profile" in im.info:
                    im.info.pop("icc_profile", None)
            except Exception:
                pass

            # 输出策略与后端生成一致：
            # - 不透明：统一写回为 JPEG（减少 png gamma/chrm 等导致的 <img> vs canvas 差异）
            # - 透明：写回为 PNG
            suffix = p.suffix.lower()
            out_path = p
            fmt = None
            save_kwargs: dict = {}

            has_alpha = im.mode in ("RGBA", "LA") or ("transparency" in getattr(im, "info", {}))
            is_transparent = False
            if has_alpha:
                try:
                    alpha = im.getchannel("A")
                    lo, hi = alpha.getextrema()
                    is_transparent = lo < 255
                except Exception:
                    is_transparent = True

            if not is_transparent:
                fmt = "JPEG"
                if im.mode != "RGB":
                    im = im.convert("RGB")
                out_path = p.with_suffix(".jpg")
                save_kwargs = {"quality": 95, "optimize": True, "progressive": True}
            else:
                fmt = "PNG"
                out_path = p.with_suffix(".png")
                if im.mode not in ("RGBA", "RGB"):
                    im = im.convert("RGBA")
                save_kwargs = {"optimize": True}

            # 原子覆盖写回
            tmp = out_path.with_suffix(out_path.suffix + ".tmp")
            im.save(tmp, format=fmt, **save_kwargs)
            # 替换目标文件
            os.replace(tmp, out_path)
            # 如果发生了后缀变更（png/webp -> jpg 或 webp/png/jpg 互转），删掉原文件
            if out_path != p and p.exists():
                p.unlink()
            ok += 1
            changed += 1
            print(f"✅ {out_path.name}")
        except Exception as e:
            failed += 1
            print(f"❌ {p.name}: {e}")

    print(f"\n完成：处理 {len(files)} 个，成功 {ok}，失败 {failed}，写回 {changed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())



