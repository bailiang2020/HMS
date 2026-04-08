# prompt_templates.py

TEMPLATES = {
    "weibo": """
作为一个顶尖社交平台新闻专家，请基于所提供的图片和文本，判定给定新闻是否为虚假新闻。
文字: {text}

选项

A. 假
B. 真

在做出回复前请一步步思考，并保证回复中的最后一句应为："ANSWER: $LETTER"，其中 $LETTER 为你的最终答案 (e.g., "ANSWER: A").
""", "weibo_sft": """作为一位顶尖新闻专家，请基于所提供的图片和文本，判定给定新闻是否为虚假新闻。
文字: {text}

请回复：真/假。
""", "DGM4": """As a top-tier news expert, please determine whether the given news is fake based on the provided image and text.
Text: {text}

Options

A. Fake
B. Real

Think step by step before answering, and ensure the last sentence of your response is: "ANSWER: $LETTER", where $LETTER is your final answer (e.g., "ANSWER: A").
""", "DGM4_sft": """As a top-tier news expert, please determine whether the given news is fake based on the provided image and text.
Text: {text}

Please reply with: Real/Fake.
""",
}
