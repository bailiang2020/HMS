# -*- coding: utf-8 -*-
"""
单模态标签标注工具。

功能：
1) 从 pkl 数据集中抽样导出 Excel，方便人工标注 img_label/text_label。
2) 将标注后的 Excel 转换为 json（与 full_samples 格式一致）。

当前内置支持：weibo
- pkl 需包含字段：post_id, image_name, text, label

可扩展支持：通过 CLI 指定字段映射。
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import re
import shutil
import zipfile
from typing import Any, Dict, List, Optional
import xml.etree.ElementTree as ET

import pandas as pd


def _load_pkl(path: str) -> Dict[str, List[Any]]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"pkl must be a dict[str, list], got {type(data)}")
    return data


def _validate_fields(data: Dict[str, List[Any]], required: List[str]) -> None:
    for key in required:
        if key not in data or not isinstance(data[key], list):
            raise ValueError(f"Missing field in pkl: {key}")


def _build_samples_weibo(
    data: Dict[str, List[Any]],
    img_dir: Optional[str],
) -> List[Dict[str, Any]]:
    required = ["post_id", "image_name", "text", "label"]
    _validate_fields(data, required)
    n = len(data["post_id"])
    samples: List[Dict[str, Any]] = []
    for i in range(n):
        image_name = data["image_name"][i]
        image_abs_path = os.path.join(img_dir, str(image_name)) if img_dir else ""
        sample = {
            "id": data["post_id"][i],
            "text": data["text"][i],
            "image_path": image_name,
            "image_abs_path": image_abs_path,
            "label": data["label"][i],
            "img_label": None,
            "text_label": None,
        }
        samples.append(sample)
    return samples


def _build_samples_generic(
    data: Dict[str, List[Any]],
    img_dir: Optional[str],
    id_key: str,
    text_key: str,
    image_key: str,
    label_key: str,
) -> List[Dict[str, Any]]:
    required = [id_key, text_key, image_key, label_key]
    _validate_fields(data, required)
    n = len(data[id_key])
    samples: List[Dict[str, Any]] = []
    for i in range(n):
        image_name = data[image_key][i]
        image_abs_path = os.path.join(img_dir, str(image_name)) if img_dir else ""
        sample = {
            "id": data[id_key][i],
            "text": data[text_key][i],
            "image_path": image_name,
            "image_abs_path": image_abs_path,
            "label": data[label_key][i],
            "img_label": None,
            "text_label": None,
        }
        samples.append(sample)
    return samples


def _sample_items(items: List[Dict[str, Any]], size: int, seed: int) -> List[Dict[str, Any]]:
    if size <= 0:
        return []
    if len(items) <= size:
        return list(items)
    random.seed(seed)
    return random.sample(items, size)


def _assign_and_copy_images(
    items: List[Dict[str, Any]],
    image_output_dir: Optional[str],
) -> List[Dict[str, Any]]:
    if image_output_dir:
        os.makedirs(image_output_dir, exist_ok=True)

    updated: List[Dict[str, Any]] = []
    for idx, item in enumerate(items, start=1):
        new_item = dict(item)
        new_item["image_index"] = idx
        new_name = f"{idx}.jpg"
        new_item["image_name_for_label"] = new_name
        src_path = new_item.get("image_abs_path", "")
        if image_output_dir and src_path and os.path.exists(str(src_path)):
            dst_path = os.path.join(image_output_dir, new_name)
            shutil.copy(str(src_path), dst_path)
        updated.append(new_item)
    return updated


def _reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    tail_cols = ["text", "image_index", "label", "img_label", "text_label"]
    existing_tail = [c for c in tail_cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in existing_tail]
    ordered = other_cols + existing_tail
    return df[ordered]


def export_excel(
    dataset: str,
    train_pkl: str,
    test_pkl: str,
    output_dir: str,
    train_size: int = 100,
    test_size: int = 100,
    seed: int = 2024,
    img_dir: Optional[str] = None,
    train_image_output_dir: Optional[str] = None,
    test_image_output_dir: Optional[str] = None,
    id_key: str = "",
    text_key: str = "",
    image_key: str = "",
    label_key: str = "",
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    train_raw = _load_pkl(train_pkl)
    test_raw = _load_pkl(test_pkl)

    if dataset.lower() == "weibo":
        train_samples = _build_samples_weibo(train_raw, img_dir=img_dir)
        test_samples = _build_samples_weibo(test_raw, img_dir=img_dir)
    else:
        if not (id_key and text_key and image_key and label_key):
            raise ValueError("Generic dataset requires id/text/image/label keys")
        train_samples = _build_samples_generic(
            train_raw,
            img_dir=img_dir,
            id_key=id_key,
            text_key=text_key,
            image_key=image_key,
            label_key=label_key,
        )
        test_samples = _build_samples_generic(
            test_raw,
            img_dir=img_dir,
            id_key=id_key,
            text_key=text_key,
            image_key=image_key,
            label_key=label_key,
        )

    train_pick = _sample_items(train_samples, train_size, seed)
    test_pick = _sample_items(test_samples, test_size, seed + 1)

    train_pick = _assign_and_copy_images(train_pick, train_image_output_dir)
    test_pick = _assign_and_copy_images(test_pick, test_image_output_dir)

    train_df = pd.DataFrame(train_pick)
    test_df = pd.DataFrame(test_pick)

    train_df = _reorder_columns(train_df)
    test_df = _reorder_columns(test_df)

    train_path = os.path.join(output_dir, "train_annotation.xlsx")
    test_path = os.path.join(output_dir, "test_annotation.xlsx")

    train_df.to_excel(train_path, index=False)
    test_df.to_excel(test_path, index=False)

    print(f"Saved train excel to {train_path} ({len(train_pick)} samples)")
    print(f"Saved test excel to {test_path} ({len(test_pick)} samples)")


def _xlsx_col_to_idx(ref: str) -> int:
    letters = re.match(r"([A-Z]+)", ref)
    if letters is None:
        raise ValueError(f"Invalid cell reference: {ref}")
    idx = 0
    for ch in letters.group(1):
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return idx - 1


def _extract_si_text(si_elem: ET.Element, ns: Dict[str, str]) -> str:
    return "".join(node.text or "" for node in si_elem.findall(".//a:t", ns))


def _coerce_xlsx_value(value: Optional[str]) -> Any:
    if value in (None, ""):
        return None
    if re.fullmatch(r"-?\d+", value):
        try:
            return int(value)
        except Exception:
            return value
    if re.fullmatch(r"-?\d+\.\d+", value):
        try:
            return float(value)
        except Exception:
            return value
    return value


def _read_xlsx_rows_stdlib(xlsx_path: str) -> List[Dict[str, Any]]:
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(xlsx_path, "r") as zf:
        shared_strings: List[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            shared_root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            shared_strings = [_extract_si_text(si, ns) for si in shared_root.findall("a:si", ns)]

        sheet_root = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
        rows: List[Dict[int, Any]] = []
        max_idx = -1
        for row in sheet_root.findall(".//a:sheetData/a:row", ns):
            row_map: Dict[int, Any] = {}
            for cell in row.findall("a:c", ns):
                ref = cell.attrib.get("r", "")
                col_idx = _xlsx_col_to_idx(ref)
                cell_type = cell.attrib.get("t")
                value: Any = None
                if cell_type == "s":
                    v = cell.find("a:v", ns)
                    if v is not None and v.text is not None:
                        value = shared_strings[int(v.text)]
                elif cell_type == "inlineStr":
                    is_elem = cell.find("a:is", ns)
                    if is_elem is not None:
                        value = _extract_si_text(is_elem, ns)
                else:
                    v = cell.find("a:v", ns)
                    value = v.text if v is not None else None
                row_map[col_idx] = _coerce_xlsx_value(value)
                max_idx = max(max_idx, col_idx)
            rows.append(row_map)

    if not rows:
        return []

    header_row = rows[0]
    headers = {idx: header_row.get(idx) for idx in range(max_idx + 1) if header_row.get(idx) is not None}
    records: List[Dict[str, Any]] = []
    for row_map in rows[1:]:
        record: Dict[str, Any] = {}
        non_empty = False
        for idx, name in headers.items():
            key = str(name)
            value = row_map.get(idx, None)
            record[key] = value
            if value is not None:
                non_empty = True
        if non_empty:
            records.append(record)
    return records


def excel_to_json(excel_path: str, json_path: str) -> None:
    try:
        df = pd.read_excel(excel_path)
    except Exception as exc:
        print(f"pd.read_excel failed for {excel_path}, fallback to stdlib xlsx parser: {exc}")
        df = pd.DataFrame(_read_xlsx_rows_stdlib(excel_path))
    required_cols = ["id", "text", "image_path", "label", "img_label", "text_label"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column in excel: {col}")

    def _norm(v):
        if pd.isna(v):
            return None
        return v

    records: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        item = {
            "id": _norm(row.get("id")),
            "text": _norm(row.get("text")),
            "image_path": _norm(row.get("image_path")),
            "label": _norm(row.get("label")),
            "img_label": _norm(row.get("img_label")),
            "text_label": _norm(row.get("text_label")),
        }
        if item["img_label"] is None and item["text_label"] is None:
            continue
        records.append(item)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Saved json to {json_path} ({len(records)} samples)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Annotation tools for unimodal labels")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export", help="export sampled excel for annotation")
    export_parser.add_argument("--dataset", type=str, required=True, help="weibo or custom")
    export_parser.add_argument("--train_pkl", type=str, required=True)
    export_parser.add_argument("--test_pkl", type=str, required=True)
    export_parser.add_argument("--output_dir", type=str, required=True)
    export_parser.add_argument("--train_size", type=int, default=100)
    export_parser.add_argument("--test_size", type=int, default=100)
    export_parser.add_argument("--seed", type=int, default=2024)
    export_parser.add_argument("--img_dir", type=str, default="")
    export_parser.add_argument("--train_image_output_dir", type=str, default="")
    export_parser.add_argument("--test_image_output_dir", type=str, default="")
    export_parser.add_argument("--id_key", type=str, default="")
    export_parser.add_argument("--text_key", type=str, default="")
    export_parser.add_argument("--image_key", type=str, default="")
    export_parser.add_argument("--label_key", type=str, default="")

    convert_parser = subparsers.add_parser("to_json", help="convert annotated excel to json")
    convert_parser.add_argument("--excel_path", type=str, required=True)
    convert_parser.add_argument("--json_path", type=str, required=True)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "export":
        export_excel(
            dataset=args.dataset,
            train_pkl=args.train_pkl,
            test_pkl=args.test_pkl,
            output_dir=args.output_dir,
            train_size=args.train_size,
            test_size=args.test_size,
            seed=args.seed,
            img_dir=args.img_dir or None,
            train_image_output_dir=args.train_image_output_dir or None,
            test_image_output_dir=args.test_image_output_dir or None,
            id_key=args.id_key,
            text_key=args.text_key,
            image_key=args.image_key,
            label_key=args.label_key,
        )
    elif args.command == "to_json":
        excel_to_json(args.excel_path, args.json_path)


if __name__ == "__main__":
    main()

"""
======================== 使用说明（可直接复制） ========================
1) 导出 Weibo 标注 Excel（训练/测试各约 100 条）
python -m src.data.utils.annotation_tools export \
  --dataset weibo \
  --train_pkl data/processed/weibo/raw/train_data.pkl \
  --test_pkl data/processed/weibo/raw/test_data.pkl \
  --output_dir data/processed/weibo/annotation/ \
  --train_size 500 \
  --test_size 500 \
  --img_dir data/raw/weibo/images/ \
  --train_image_output_dir data/processed/weibo/annotation/images/train \
  --test_image_output_dir data/processed/weibo/annotation/images/test

2) 导出自定义数据集标注 Excel（字段手动映射）
python -m src.data.utils.annotation_tools export \
  --dataset custom \
  --train_pkl /path/to/train.pkl \
  --test_pkl /path/to/test.pkl \
  --output_dir /path/to/output \
  --train_size 100 \
  --test_size 100 \
  --id_key id \
  --text_key text \
  --image_key image_path \
  --label_key label \
  --img_dir /path/to/images \
  --train_image_output_dir /path/to/output/images/train \
  --test_image_output_dir /path/to/output/images/test

3) 将标注后的 Excel 转换为 json（仅保留单模态标签非空样本）
python -m data.utils.annotation_tools to_json \
  --excel_path data/processed/weibo/annotation/train_annotation_filled.xlsx \
  --json_path data/processed/weibo/annotation/train_annotation.json

python -m data.utils.annotation_tools to_json \
  --excel_path data/processed/weibo/annotation/test_annotation_filled.xlsx \
  --json_path data/processed/weibo/annotation/test_annotation.json  

"""
