# -*- coding: utf-8 -*-
"""
可扩展的 baseline 数据集构建框架。

设计目标：
1. 根据 baseline_version + dataset_version 自动路由到对应处理流程。
2. 处理逻辑与路由逻辑解耦，统一返回 payload。
3. 如需缓存请在调用侧自行处理。
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
try:
    import cv2
except ImportError:
    cv2 = None
import torch
import random
import copy
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

try:
    import albumentations as A
except ImportError:
    A = None


class ProcessorNotFoundError(LookupError):
    """当 baseline_version + dataset_version 找不到对应处理器时抛出。"""


class DatasetFormatError(ValueError):
    """当数据字段不满足当前 dataset_version 约束时抛出。"""


@dataclass(frozen=True)
class BuildConfig:
    """数据集构建配置。"""

    baseline_version: str
    dataset_version: str
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BuildContext:
    """传递给处理器的上下文。"""

    config: BuildConfig
    raw_data: Any = None


@dataclass
class BuildResult:
    """构建结果。"""

    baseline_version: str
    dataset_version: str
    metadata: Dict[str, Any]
    payload: Any


class DatasetProcessor(ABC):
    """具体处理流程接口。"""

    @abstractmethod
    def process(self, context: BuildContext) -> Any:
        """返回处理后的对象（结构由业务自行定义）。"""




class ProcessorRegistry:
    """
    baseline + dataset 版本路由注册表。

    支持三级匹配优先级：
    1) (baseline_version, dataset_version)
    2) (baseline_version, "*")
    3) ("*", dataset_version)
    """

    def __init__(self) -> None:
        self._processors: Dict[Tuple[str, str], DatasetProcessor] = {}

    def register(self, baseline_version: str, dataset_version: str, processor: DatasetProcessor) -> None:
        key = (baseline_version, dataset_version)
        if key in self._processors:
            raise ValueError(f"Processor already registered for {key}")
        self._processors[key] = processor

    def resolve(self, baseline_version: str, dataset_version: str) -> DatasetProcessor:
        candidates = [
            (baseline_version, dataset_version),
            (baseline_version, "*"),
            ("*", dataset_version),
        ]
        for key in candidates:
            processor = self._processors.get(key)
            if processor is not None:
                return processor
        raise ProcessorNotFoundError(
            f"No processor registered for baseline={baseline_version}, dataset={dataset_version}"
        )

    def decorator(self, baseline_version: str, dataset_version: str) -> Callable[[type], type]:
        """支持装饰器方式注册处理器类。"""

        def _wrap(cls: type) -> type:
            if not issubclass(cls, DatasetProcessor):
                raise TypeError("Registered class must inherit from DatasetProcessor")
            self.register(baseline_version, dataset_version, cls())
            return cls

        return _wrap


class BaselineDatasetBuilder:
    """统一入口：根据版本路由处理并返回 payload。"""

    def __init__(self, registry: Optional[ProcessorRegistry] = None) -> None:
        self.registry = registry or ProcessorRegistry()

    def build(self, config: BuildConfig, raw_data: Any = None) -> BuildResult:
        context = BuildContext(config=config, raw_data=raw_data)
        processor = self.registry.resolve(config.baseline_version, config.dataset_version)
        payload = processor.process(context)

        metadata = self._build_metadata(config)
        return BuildResult(
            baseline_version=config.baseline_version,
            dataset_version=config.dataset_version,
            metadata=metadata,
            payload=payload,
        )

    @staticmethod
    def _build_metadata(config: BuildConfig) -> Dict[str, Any]:
        return {
            "baseline_version": config.baseline_version,
            "dataset_version": config.dataset_version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "options": config.options,
        }


def _required_option(options: Dict[str, Any], key: str) -> Any:
    if key not in options or options[key] in (None, ""):
        raise DatasetFormatError(f"Missing required option: {key}")
    return options[key]


def _validate_equal_length(data: Dict[str, List[Any]], required_keys: List[str]) -> int:
    for k in required_keys:
        if k not in data or not isinstance(data[k], list):
            raise DatasetFormatError(f"DGM4 data missing list field: {k}")
    lengths = {k: len(data[k]) for k in required_keys}
    n = lengths[required_keys[0]]
    if any(v != n for v in lengths.values()):
        raise DatasetFormatError(f"DGM4 field lengths not equal: {lengths}")
    return n


def _build_dgm4_full_samples(raw_data: Dict[str, List[Any]], img_dir: str) -> List[Dict[str, Any]]:
    required = ["id", "image_path", "text", "label", "img_label", "text_label"]
    n = _validate_equal_length(raw_data, required)

    samples: List[Dict[str, Any]] = []
    for i in range(n):
        sample = {
            "id": raw_data["id"][i],
            "text": raw_data["text"][i],
            "image_path": raw_data["image_path"][i],
            "image_abs_path": os.path.join(img_dir, str(raw_data["image_path"][i])),
            "label": raw_data["label"][i],
            "img_label": raw_data["img_label"][i],
            "text_label": raw_data["text_label"][i],
        }
        samples.append(sample)
    return samples


def _build_weibo_full_samples(raw_data: Dict[str, List[Any]], img_dir: str) -> List[Dict[str, Any]]:
    required = ["post_id", "image_name", "text", "label"]
    n = _validate_equal_length(raw_data, required)

    samples: List[Dict[str, Any]] = []
    for i in range(n):
        image_name = raw_data["image_name"][i]
        sample = {
            "id": raw_data["post_id"][i],
            "text": raw_data["text"][i],
            "image_path": image_name,
            "image_abs_path": os.path.join(img_dir, str(image_name)),
            "label": raw_data["label"][i],
            "img_label": None,
            "text_label": None,
        }
        samples.append(sample)
    return samples


def _slice_subset_size(total_len: int, ratio: float) -> int:
    if ratio <= 0:
        return 0
    if ratio > 1:
        raise ValueError("unimodal_ratio must be in [0, 1]")
    return int(total_len * ratio)


def _build_unimodal_subsets(
    full_samples: List[Dict[str, Any]],
    ratio: float,
    annotated_samples: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    生成单模态子集。

    - 默认使用 full_samples 前 ratio 比例切分（label 置为 None）。
    - 若提供 annotated_samples，则优先使用其携带的 text_label/img_label 生成子集。
      该字段可由后续人工标注的 json 提供（格式与 full_samples 一致，但单模态标签非空）。
    """

    text_only: List[Dict[str, Any]] = []
    img_only: List[Dict[str, Any]] = []

    source = annotated_samples if annotated_samples is not None else full_samples[:_slice_subset_size(len(full_samples), ratio)]

    for item in source:
        text_item = dict(item)
        text_item["modality"] = "text_only"
        text_item["label"] = item.get("text_label")
        text_item["img_label"] = None
        # 纯文本样本：剔除视觉输入，避免后续链路误用
        text_item["image_path"] = ""
        text_item["image_abs_path"] = ""

        img_item = dict(item)
        img_item["modality"] = "img_only"
        img_item["label"] = item.get("img_label")
        img_item["text_label"] = None
        # 纯图像样本：剔除文本输入，避免后续链路误用
        img_item["text"] = ""

        if annotated_samples is None or text_item["label"] is not None:
            text_only.append(text_item)
        if annotated_samples is None or img_item["label"] is not None:
            img_only.append(img_item)

    return {
        "text_only": text_only,
        "img_only": img_only,
    }


# 全局默认实例（可直接使用）
DEFAULT_REGISTRY = ProcessorRegistry()
DEFAULT_BUILDER = BaselineDatasetBuilder(registry=DEFAULT_REGISTRY)


@DEFAULT_REGISTRY.decorator(baseline_version="*", dataset_version="dgm4")
class DGM4Processor(DatasetProcessor):
    """
    DGM4 v1 数据处理流程。

    必填 options:
    - img_dir: 图片根目录
    - data_path: DGM4 pkl 文件路径

    可选 options:
    - unimodal_ratio: [0,1]，从 full 中切出前 ratio 比例用于 img_only/text_only
    """

    def process(self, context: BuildContext) -> Any:
        options = context.config.options
        img_dir = _required_option(options, "img_dir")
        data_path = _required_option(options, "data_path")
        unimodal_ratio = float(options.get("unimodal_ratio", 0.0))

        with open(data_path, "rb") as f:
            raw_data = pickle.load(f)
        if not isinstance(raw_data, dict):
            raise DatasetFormatError("DGM4 pkl must be a dict[str, list]")

        full_samples = _build_dgm4_full_samples(raw_data, img_dir=img_dir)
        subsets = _build_unimodal_subsets(full_samples, ratio=unimodal_ratio)

        return {
            "dataset_version": "dgm4",
            "full": full_samples,
            "img_only_subset": subsets["img_only"],
            "text_only_subset": subsets["text_only"],
            "stats": {
                "full_size": len(full_samples),
                "subset_size": len(subsets["img_only"]),
                "unimodal_ratio": unimodal_ratio,
            },
        }


@DEFAULT_REGISTRY.decorator(baseline_version="*", dataset_version="weibo")
class WeiboProcessor(DatasetProcessor):
    """
    Weibo 数据处理流程。

    必填 options:
    - img_dir: 图片根目录
    - data_path: Weibo pkl 文件路径（post_id/image_name/text/label）

    可选 options:
    - unimodal_ratio: [0,1]，从 full 中切出前 ratio 比例用于 img_only/text_only
    - unimodal_annot_path: 手工标注单模态 json 路径（字段与 full_samples 一致，且单模态标签非空）
    """

    def process(self, context: BuildContext) -> Any:
        options = context.config.options
        img_dir = _required_option(options, "img_dir")
        data_path = _required_option(options, "data_path")
        unimodal_ratio = float(options.get("unimodal_ratio", 0.0))
        unimodal_annot_path = options.get("unimodal_annot_path")

        with open(data_path, "rb") as f:
            raw_data = pickle.load(f)
        if not isinstance(raw_data, dict):
            raise DatasetFormatError("Weibo pkl must be a dict[str, list]")

        full_samples = _build_weibo_full_samples(raw_data, img_dir=img_dir)

        annotated_samples: Optional[List[Dict[str, Any]]] = None
        if unimodal_annot_path:
            with open(unimodal_annot_path, "r", encoding="utf-8") as f:
                annotated_samples = json.load(f)
            if not isinstance(annotated_samples, list):
                raise DatasetFormatError("unimodal_annot_path must be a list of samples")
            full_by_id = {sample.get("id"): sample for sample in full_samples}
            merged_samples: List[Dict[str, Any]] = []
            for item in annotated_samples:
                if not isinstance(item, dict):
                    raise DatasetFormatError("each annotated sample must be a dict")
                base = dict(full_by_id.get(item.get("id"), {}))
                base.update(item)
                image_path = base.get("image_path") or ""
                if image_path and not base.get("image_abs_path"):
                    base["image_abs_path"] = os.path.join(img_dir, str(image_path))
                merged_samples.append(base)
            annotated_samples = merged_samples

        subsets = _build_unimodal_subsets(
            full_samples,
            ratio=unimodal_ratio,
            annotated_samples=annotated_samples,
        )

        return {
            "dataset_version": "weibo",
            "full": full_samples,
            "img_only_subset": subsets["img_only"],
            "text_only_subset": subsets["text_only"],
            "stats": {
                "full_size": len(full_samples),
                "subset_size": len(subsets["img_only"]),
                "unimodal_ratio": unimodal_ratio,
                "annotated_unimodal_size": 0 if annotated_samples is None else len(annotated_samples),
            },
        }


def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """注入控制台参数，供训练/预处理脚本复用。"""
    parser.add_argument("--baseline_version", type=str, required=True)
    parser.add_argument("--dataset_version", type=str, required=True)

    # DGM4 当前必须参数（与脚本中的用法保持一致）
    parser.add_argument("--img_dir", type=str, default="")
    parser.add_argument("--data_path", type=str, default="")

    # 子集拆分比例，逻辑参考 purification_sft_v1.py
    parser.add_argument("--unimodal_ratio", type=float, default=0.0)

    # Weibo 可选单模态标注 json
    parser.add_argument("--unimodal_annot_path", type=str, default="")
    return parser


def build_config_from_args(args: argparse.Namespace) -> BuildConfig:
    options = {
        "img_dir": getattr(args, "img_dir", ""),
        "data_path": getattr(args, "data_path", ""),
        "unimodal_ratio": getattr(args, "unimodal_ratio", 0.0),
        "unimodal_annot_path": getattr(args, "unimodal_annot_path", ""),
    }
    return BuildConfig(
        baseline_version=args.baseline_version,
        dataset_version=args.dataset_version,
        options=options,
    )


def build_from_args(args: argparse.Namespace, builder: Optional[BaselineDatasetBuilder] = None) -> BuildResult:
    worker = builder or DEFAULT_BUILDER
    config = build_config_from_args(args)
    return worker.build(config)


class MIMoEDataset(Dataset):
    """按 FakeNet 风格在 __getitem__ 中即时读取与处理图片。"""

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        image_size: int = 224,
        modality: str = "full",
        is_train: bool = True,
        dataset_version: str = "dgm4",
    ) -> None:
        self.samples = list(samples)
        self.image_size = image_size
        self.modality = modality
        self.is_train = is_train
        self.dataset_version = dataset_version
        if A is None:
            raise ImportError(
                "albumentations is required to use MIMoEDataset. "
                "Please install albumentations to enable this dataset."
            )
        self.resize_and_to_tensor = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

        self.transform_just_resize = A.Compose(
            [A.Resize(always_apply=True, height=image_size, width=image_size)]
        )
        self.transform = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ElasticTransform(p=0.5),
                A.OneOf(
                    [
                        A.RGBShift(always_apply=False, p=0.25),
                    ]
                ),
                A.OneOf(
                    [
                        A.GaussNoise(always_apply=False, p=0.2),
                        A.ISONoise(always_apply=False, p=0.2),
                    ]
                ),
                A.Resize(always_apply=True, height=image_size, width=image_size),
            ]
        )

        labels = [int(s.get("label", 0)) for s in self.samples] if self.samples else [0]
        fake_news_num = int(np.sum(labels))
        total = max(1, len(labels))
        self.thresh = (total - fake_news_num) / total
        self.pos_weight = torch.tensor((total - fake_news_num) / max(1, fake_news_num))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        if self.dataset_version == "weibo":
            return self._getitem_weibo(index)
        return self._getitem_default(index)
    
    def bgr2ycbcr(self,img, only_y=True):
        '''bgr version of rgb2ycbcr
        only_y: only return Y channel
        Input:
            uint8, [0, 255]
            float, [0, 1]
        '''
        in_img_type = img.dtype
        img.astype(np.float32)
        if in_img_type != np.uint8:
            img *= 255.
        # convert
        if only_y:
            rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
        else:
            rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                                [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
        if in_img_type == np.uint8:
            rlt = rlt.round()
        else:
            rlt /= 255.
        return rlt.astype(in_img_type)

    def channel_convert(self,in_c, tar_type, img_list):
        # conversion among BGR, gray and y
        if in_c == 3 and tar_type == 'gray':  # BGR to gray
            gray_list = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in img_list]
            return [np.expand_dims(img, axis=2) for img in gray_list]
        elif in_c == 3 and tar_type == 'y':  # BGR to y
            y_list = [bgr2ycbcr(img, only_y=True) for img in img_list]
            return [np.expand_dims(img, axis=2) for img in y_list]
        elif in_c == 1 and tar_type == 'RGB':  # gray/y to BGR
            return [cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) for img in img_list]
        else:
            return img_list


    def _getitem_default(self, index: int):
        item = self.samples[index]
        content = str(item.get("text", ""))
        label = int(item.get("label", 0))
        gt_path = str(item.get("image_abs_path", ""))

        if self.modality == "img_only":
            content = ""

        if self.modality == "text_only":
            img_gt = torch.zeros((3, self.image_size, self.image_size), dtype=torch.float32)
            return (content, img_gt, img_gt, label, 0), (gt_path)

        gt_size = self.image_size
        img_gt = cv2.imread(gt_path, cv2.IMREAD_COLOR) if cv2 is not None else None
        if img_gt is None:
            img_gt = Image.open(gt_path)
            img_gt = self.resize_and_to_tensor(img_gt).float()
            if img_gt.shape[0] == 1:
                img_gt = img_gt.expand(3, -1, -1)
            elif img_gt.shape[0] == 4:
                img_gt = img_gt[:3, :, :]
        else:
            img_gt = img_gt.astype(np.float32) / 255.0
            if img_gt.ndim == 2:
                img_gt = np.expand_dims(img_gt, axis=2)
            if img_gt.shape[2] > 3:
                img_gt = img_gt[:, :, :3]


            img_gt = self.channel_convert(img_gt.shape[2], "RGB", [img_gt])[0]
            img_gt = cv2.resize(
                np.copy(img_gt),
                (gt_size, gt_size),
                interpolation=cv2.INTER_LINEAR,
            )

            if img_gt.shape[2] == 3:
                img_gt = img_gt[:, :, [2, 1, 0]]

            img_gt = torch.from_numpy(
                np.ascontiguousarray(np.transpose(img_gt, (2, 0, 1)))
            ).float()

        return (content, img_gt, img_gt, label, 0), (gt_path)

    def _getitem_weibo(self, index: int):
        item = self.samples[index]
        content = str(item.get("text", ""))
        label = int(item.get("label", 0))
        gt_path = str(item.get("image_abs_path", ""))

        if self.modality == "img_only":
            content = ""

        if self.modality == "text_only":
            img_gt = torch.zeros((3, self.image_size, self.image_size), dtype=torch.float32)
            return (content, img_gt, img_gt, label, 0), ("")

        gt_size = self.image_size
        img_gt = cv2.imread(gt_path, cv2.IMREAD_COLOR)
        if img_gt is None:
            img_gt = Image.open(gt_path)
            img_gt = self.resize_and_to_tensor(img_gt).float()
            if img_gt.shape[0] == 1:
                img_gt = img_gt.expand(3, -1, -1)
            elif img_gt.shape[0] == 4:
                img_gt = img_gt[:3, :, :]
        H_origin, W_origin, _ = img_gt.shape
        if img_gt.ndim == 2:
            img_gt = np.expand_dims(img_gt, axis=2)
        # some images have 4 channels
        if img_gt.shape[2] > 3:
            img_gt = img_gt[:, :, :3]

        img_gt = self.channel_convert(
            img_gt.shape[2], "RGB", [img_gt]
        )[0]

        try:
            img_gt_augment = self.transform_just_resize(image=copy.deepcopy(img_gt))[
                "image"
            ]
        except TypeError:
            return self.__getitem__(0)
        
        img_gt = self.transform_just_resize(image=copy.deepcopy(img_gt))["image"]
        img_gt = img_gt.astype(np.float32) / 255.0
        img_gt_augment = img_gt_augment.astype(np.float32) / 255.0

        orig_height, orig_width, _ = img_gt.shape
        H, W, _ = img_gt.shape

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_gt.shape[2] == 3:
            img_gt = img_gt[:, :, [2, 1, 0]]
        if img_gt_augment.shape[2] == 3:
            img_gt_augment = img_gt_augment[:, :, [2, 1, 0]]

        img_gt = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_gt, (2, 0, 1)))
        ).float()
        img_gt_augment = torch.from_numpy(
            np.ascontiguousarray(np.transpose(img_gt_augment, (2, 0, 1)))
        ).float()

        return (content, img_gt, img_gt_augment, label, 0), (gt_path)


def get_mimoe_datasets(
    baseline_version: str,
    dataset_version: str,
    train_data_path: str,
    train_img_dir: str,
    val_data_path: str,
    val_img_dir: str,
    test_data_path: str,
    test_img_dir: str,
    unimodal_ratio: float = 0.0,
    image_size: int = 224,
    train_unimodal_annot_path: str = "",
    val_unimodal_annot_path: str = "",
    test_unimodal_annot_path: str = "",
) -> Dict[str, Dataset]:
    """一步返回 train/full+img_only+text_only + val/full + test/full。"""

    processor = DEFAULT_REGISTRY.resolve(baseline_version, dataset_version)
    dataset_version_lower = str(dataset_version).lower()
    eval_unimodal_ratio = 1.0 if dataset_version_lower == "dgm4" else 0.0

    def _valid_unimodal_samples(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [sample for sample in samples if sample.get("label") is not None]

    train_payload = processor.process(
        BuildContext(
            config=BuildConfig(
                baseline_version=baseline_version,
                dataset_version=dataset_version,
                options={
                    "img_dir": train_img_dir,
                    "data_path": train_data_path,
                    "unimodal_ratio": unimodal_ratio,
                    "unimodal_annot_path": train_unimodal_annot_path,
                },
            )
        )
    )

    val_payload = processor.process(
        BuildContext(
            config=BuildConfig(
                baseline_version=baseline_version,
                dataset_version=dataset_version,
                options={
                    "img_dir": val_img_dir,
                    "data_path": val_data_path,
                    "unimodal_ratio": eval_unimodal_ratio,
                    "unimodal_annot_path": val_unimodal_annot_path,
                },
            )
        )
    )

    test_payload = processor.process(
        BuildContext(
            config=BuildConfig(
                baseline_version=baseline_version,
                dataset_version=dataset_version,
                options={
                    "img_dir": test_img_dir,
                    "data_path": test_data_path,
                    "unimodal_ratio": eval_unimodal_ratio,
                    "unimodal_annot_path": test_unimodal_annot_path,
                },
            )
        )
    )

    datasets = {
        "train_full": MIMoEDataset(
            train_payload["full"],
            image_size=image_size,
            modality="full",
            is_train=True,
            dataset_version=dataset_version,
        ),
        "train_img_only": MIMoEDataset(
            _valid_unimodal_samples(train_payload["img_only_subset"]),
            image_size=image_size,
            modality="img_only",
            is_train=True,
            dataset_version=dataset_version,
        ),
        "train_text_only": MIMoEDataset(
            _valid_unimodal_samples(train_payload["text_only_subset"]),
            image_size=image_size,
            modality="text_only",
            is_train=True,
            dataset_version=dataset_version,
        ),
        "val_full": MIMoEDataset(
            val_payload["full"],
            image_size=image_size,
            modality="full",
            is_train=False,
            dataset_version=dataset_version,
        ),
        "val_img_only": MIMoEDataset(
            _valid_unimodal_samples(val_payload["img_only_subset"]),
            image_size=image_size,
            modality="img_only",
            is_train=False,
            dataset_version=dataset_version,
        ),
        "val_text_only": MIMoEDataset(
            _valid_unimodal_samples(val_payload["text_only_subset"]),
            image_size=image_size,
            modality="text_only",
            is_train=False,
            dataset_version=dataset_version,
        ),
        "test_full": MIMoEDataset(
            test_payload["full"],
            image_size=image_size,
            modality="full",
            is_train=False,
            dataset_version=dataset_version,
        ),
        "test_img_only": MIMoEDataset(
            _valid_unimodal_samples(test_payload["img_only_subset"]),
            image_size=image_size,
            modality="img_only",
            is_train=False,
            dataset_version=dataset_version,
        ),
        "test_text_only": MIMoEDataset(
            _valid_unimodal_samples(test_payload["text_only_subset"]),
            image_size=image_size,
            modality="text_only",
            is_train=False,
            dataset_version=dataset_version,
        ),
    }
    return datasets


def _build_standalone_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build baseline-aware datasets")
    return add_cli_args(parser)


if __name__ == "__main__":
    parser = _build_standalone_parser()
    cli_args = parser.parse_args()
    result = build_from_args(cli_args)
    print(
        f"Dataset built (baseline={result.baseline_version}, dataset={result.dataset_version})"
    )
