# -*- coding: utf-8 -*-
import os
import pickle
from torch.utils.data import Dataset
import PIL
import warnings

warnings.simplefilter("once", category=UserWarning)


class VLMDataset(Dataset):
    """
    Dataset class for Vision-Language Model (VLM) tasks.
    每次 __getitem__ 按路径重新加载图片，不使用 LMDB 或缓存。
    """

    def __init__(self, args):
        self.model_path = args.model_path
        self.trust_remote_code = args.trust_remote_code
        self.model_type = args.model_type
        self.prompt_version = args.prompt_version
        self.meta_in_item = args.meta_in_item
        self.meta_keys_for_item = args.meta_keys_for_item
        self.img_only = args.img_only
        self.text_only = args.text_only
        self._processor = None
        self._img_only_prompt = None
        self.data_path = args.data_path
        self.img_dir = args.img_dir

        with open(self.data_path, "rb") as f:
            self.raw_data = pickle.load(f)

        self.image_path = self.raw_data.get("image_path", self.raw_data.get("image_name", []))
        self.text = self.raw_data.get("text", [])
        self.labels = self.raw_data.get("label", None)

        if self.labels is None:
            raise ValueError("'label' not found in data.")
        if len(self.image_path) != len(self.text):
            raise ValueError("image_path and text should have same number of items")
        if len(self.labels) != len(self.image_path):
            raise ValueError("labels and image_path should have same number of items")

        self.data = dict(self.raw_data)

    def _build_prompt(self, text: str, version: str = None) -> str:
        from src.utils.utils import switch_question_version
        ver = self.prompt_version if version is None else version
        return switch_question_version(text, ver)

    def _ensure_img_only_prompt(self):
        if self._img_only_prompt is None:
            base_prompt = self._build_prompt("")
            # image 只能是占位符，不会被实际处理器使用，所以创建一个纯白图像
            image = PIL.Image.new("RGB", (224, 224), color=(255, 255, 255))
            msg = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                        },
                        {"type": "text", "text": base_prompt},
                    ],
                }
            ]
            self._ensure_processor()
            self._img_only_prompt = self._processor.apply_chat_template(
                [msg], tokenize=False, add_generation_prompt=True
            )[0]

    def _ensure_processor(self):
        """Lazily create the AutoProcessor once per process to avoid per-sample overhead."""
        if getattr(self, "_processor", None) is None:
            from transformers import AutoProcessor as ModelProcessor
            self._processor = ModelProcessor.from_pretrained(
                self.model_path, trust_remote_code=self.trust_remote_code
            )

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, index):
        if index < 0 or index >= len(self.image_path):
            raise IndexError(index)

        sig_text = self.text[index]
        sig_image_path = self.image_path[index]
        label = self.labels[index]

        full_img_path = os.path.join(self.img_dir, sig_image_path)
        image = PIL.Image.open(full_img_path).convert("RGB")

        image.verify()

        prompt = self._build_prompt(sig_text, version=self.prompt_version)
        message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        self._ensure_processor()
        message = self._processor.apply_chat_template([message], tokenize=False, add_generation_prompt=True)
        data_item = {
            "prompt": message[0],
            "multi_modal_data": {"image": image},
        }

        if self.text_only:
            data_item["multi_modal_data"]["image"] = None
            data_item["prompt"] = data_item["prompt"].replace(
                "<|vision_start|><|image_pad|><|vision_end|>", ""
            )
        elif self.img_only:
            self._ensure_img_only_prompt()
            data_item["prompt"] = self._img_only_prompt

        if self.text_only:
            v = self.data.get("text_label", None)
            if isinstance(v, (list, tuple)):
                label = v[index]
            elif v is not None:
                label = v
        elif self.img_only:
            v = self.data.get("img_label", None)
            if isinstance(v, (list, tuple)):
                label = v[index]
            elif v is not None:
                label = v

        try:
            label = int(label)
        except Exception:
            pass

        return data_item, label


if __name__ == "__main__":
    from src.utils.utils import get_parser

    parser = get_parser()
    args = parser.parse_args()
    dataset = VLMDataset(args)
    print(f"Dataset size: {len(dataset)}")
    for i in range(min(3, len(dataset))):
        item, label = dataset[i]
        print(f"Sample {i}: Label={label}, Prompt=\n{item['prompt']}\n, Image Type={type(item['multi_modal_data']['image'])}")
