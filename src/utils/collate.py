import torch


def _ppb_find_pattern_1d(seq_list, patterns):
    """在 Python list 中查找 pattern，返回 (idx, len) 或 (None, None)"""
    seq_len = len(seq_list)
    best_idx = None
    best_len = None
    for pat in patterns:
        pat_len = len(pat)
        if pat_len > seq_len:
            continue
        search_limit = seq_len - pat_len + 1 if best_idx is None else best_idx + 1
        for i in range(search_limit):
            if seq_list[i:i + pat_len] == pat:
                if best_idx is None or i < best_idx:
                    best_idx = i
                    best_len = pat_len
                break
    return best_idx, best_len


class CollateFn:
    def __init__(self, args, processor, cn=True):
        self.args = args
        self.processor = processor

        # -----------------------------
        # [PPB] 预计算 anchor patterns 为 IDs（只在 __init__ 做一次）
        # -----------------------------
        tokenizer = self.processor.tokenizer
        left_patterns = [
            "Text: ", "Text:", "\nText: ", "\nText:", "Text:\n", "\nText:\n",
            "文字: ", "文字:", "\n文字: ", "\n文字:", "文字:\n", "\n文字:\n",
        ]
        right_patterns = [
            "\n\nPlease reply with:",
            "\n\nPlease reply with: ",
            "\nPlease reply with:",
            "\nPlease reply with: ",
            "Please reply with: ",
            "Please reply with:",
            "\n\n请回复：",
            "\n\n请回复： ",
            "\n请回复：",
            "\n请回复： ",
            "请回复： ",
            "请回复："
        ]

        self._left_ids = [tokenizer(s, add_special_tokens=False)["input_ids"] for s in left_patterns]
        self._right_ids = [tokenizer(s, add_special_tokens=False)["input_ids"] for s in right_patterns]

        try:
            self._newline_id = tokenizer("\n", add_special_tokens=False)["input_ids"][-1]
        except Exception:
            self._newline_id = 198  # Fallback for Qwen

        if cn:
            # 中文的真和假
            self.true_id = self.processor.tokenizer("真", add_special_tokens=False).input_ids[0]
            self.false_id = self.processor.tokenizer("假", add_special_tokens=False).input_ids[0]
        else:
            # 英文的 true 和 false
            self.true_id = self.processor.tokenizer("Real", add_special_tokens=False).input_ids[0]
            self.false_id = self.processor.tokenizer("Fake", add_special_tokens=False).input_ids[0]

    def _ppb_build_masks(self, input_ids_cpu: torch.Tensor):
        """
        CPU 侧生成 image_mask 和 news_text_mask。
        - image_mask: visual tokens + vision start/end tags
        - news_text_mask: Text between 'Text:' and 'Please reply with:' (right-stripped)
        """
        # 获取 token IDs（优先级：args > tokenizer > fallback）
        image_token_id = getattr(self.args, "image_token_id", None)
        if image_token_id is None:
            image_token_id = getattr(getattr(self.processor, "tokenizer", None), "image_token_id", None)
        if image_token_id is None:
            image_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")

        vision_start_id = getattr(self.args, "vision_start_token_id", 151652)
        vision_end_id = getattr(self.args, "vision_end_id", 151653)

        # Image mask: visual tokens + vision tags
        image_mask = (input_ids_cpu == image_token_id) | \
                     (input_ids_cpu == vision_start_id) | \
                     (input_ids_cpu == vision_end_id)

        # News text mask: anchor 匹配
        news_text_mask = torch.zeros_like(input_ids_cpu, dtype=torch.bool)
        batch_size = input_ids_cpu.shape[0]

        for i in range(batch_size):
            seq = input_ids_cpu[i].tolist()
            l_idx, l_len = _ppb_find_pattern_1d(seq, self._left_ids)
            r_idx, r_len = _ppb_find_pattern_1d(seq, self._right_ids)

            if l_idx is None or r_idx is None:
                continue

            txt_start = l_idx + l_len
            txt_end = r_idx

            # Right-strip: 去除末尾换行符
            while txt_end > txt_start and seq[txt_end - 1] == self._newline_id:
                txt_end -= 1

            if txt_end > txt_start:
                news_text_mask[i, txt_start:txt_end] = True
                news_text_mask[i] &= ~image_mask[i]

        return image_mask, news_text_mask

    def __call__(self, batch):
        data, labels = zip(*batch)
        texts = [ex["prompt"] for ex in data]
        imgs = [ex["multi_modal_data"]["image"] for ex in data]
        if self.args.text_only:
            encodings = self.processor(
                text=texts,
                images=None,
                return_tensors="pt",
                padding=True,
            )
        else:
            encodings = self.processor(
                text=texts,
                images=imgs,
                return_tensors="pt",
                padding=True,
            )
        prompt_input_ids = encodings.input_ids  # [B, L1]

        # 构造单-token 的 label_ids [B, 1]
        label_ids = torch.tensor(
            [[self.true_id] if l == 0 else [self.false_id] for l in labels],
            dtype=torch.long
        )
        # 构造 labels: prompt 部分为 -100 忽略，label 部分为真实 id
        prompt_labels = torch.full_like(prompt_input_ids, -100)[:, 1:]  # [B, L1-1]
        label_ids_tensor = torch.cat([prompt_labels, label_ids], dim=1)  # [B, L1]

        # -----------------------------
        # [PPB] 生成 masks 并塞入 encodings
        # -----------------------------
        try:
            input_ids_cpu = encodings.input_ids.detach().cpu()
            img_mask, txt_mask = self._ppb_build_masks(input_ids_cpu)
            encodings["ppb_image_mask"] = img_mask
            encodings["ppb_news_text_mask"] = txt_mask
        except Exception:
            pass

        return encodings, labels, label_ids_tensor
