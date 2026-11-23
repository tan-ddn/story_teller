"""Keep DeepSeek's generation prompt and supervised answer boundary aligned."""


def align_deepseek_template(tokenizer):
    render = tokenizer.apply_chat_template

    def aligned(messages, *args, **kwargs):
        if not messages or messages[-1].get("role") != "assistant":
            return render(messages, *args, **kwargs)
        if args:
            raise ValueError("positional template options are not supported")
        options = dict(kwargs)
        tokenize = options.pop("tokenize", True)
        return_dict = options.pop("return_dict", False)
        options.pop("add_generation_prompt", None)
        if return_dict:
            raise ValueError("return_dict is not supported for supervised rendering")
        prefix = render(messages[:-1], tokenize=False, add_generation_prompt=True, **options)
        full = render(messages, tokenize=False, add_generation_prompt=False, **options)
        marker = "<｜Assistant｜>"
        location = full.rfind(marker)
        start = location + len(marker)
        if location < 0 or not full[start:].startswith(messages[-1]["content"]):
            raise ValueError("unexpected DeepSeek assistant boundary")
        text = prefix + "</think>\n" + full[start:]
        return tokenizer.encode(text, add_special_tokens=False) if tokenize else text

    tokenizer.apply_chat_template = aligned
