"""
Model adaptors.

Supported models:
- Vicuna
- Koala
- OpenAssistant/oasst-sft-1-pythia-12b
- StabilityAI/stablelm-tuned-alpha-7b
- databricks/dolly-v2-12b
- THUDM/chatglm-6b
- project-baize/baize-lora-7B
- Alpaca/LLaMa
"""

model_adaptors = []


def register_model_adaptor(adaptor):
    model_adaptors.append(adaptor)


class ModelAdaptor:
    def __init__(self):
        pass

    def match(self, model_name: str):
        raise NotImplementedError()

    def load_model(self, model_name: str):
        raise NotImplementedError()

    def get_prompt(self):
        pass

    def compute_skip_echo_len(model_name: str, conv, prompt: str):
        raise NotImplementedError()


class VicunaAdaptor(ModelAdaptor):
    def match(self, model_name: str):
        return "vicuna" in model_name

    def load_model(self, model_name: str):
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, **kwargs
        )
        self.raise_warning_for_old_weights(model_path, model)
        return model, tokenizer

    def get_default_conv_template(self, conv):
        return 

    def compute_skip_echo_len(model_name: str, conv, prompt: str):
        raise NotImplementedError()

    def raise_warning_for_old_weights(model_path, model):
        if "vicuna" in model_path.lower():
            try:
                is_vicuna = isinstance(model, LlamaForCausalLM)
            except Exception:
                is_vicuna = isinstance(model, LLamaForCausalLM)
            if is_vicuna and model.model.vocab_size > 32000:
                warnings.warn(
                    "\nYou are probably using the old Vicuna-v0 model, "
                    "which will generate unexpected results with the "
                    "current fschat.\nYou can try one of the following methods:\n"
                    "1. Upgrade your weights to the new Vicuna-v1.1: https://github.com/lm-sys/FastChat#vicuna-weights.\n"
                    "2. Use the old conversation template by `python3 -m fastchat.serve.cli --model-path /path/to/vicuna-v0 --conv-template conv_one_shot`\n"
                    "3. Downgrade fschat to fschat==0.1.10 (Not recommonded).\n"
                )


class DollyV2Adaptor(ModelAdaptor):
    def match(self, model_name: str):
        return "dolly-v2" in model_name

    def load_model(self, model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, **kwargs
        )
        # 50277 means "### End"
        tokenizer.eos_token_id = 50277
        return model, tokenizer

    def get_prompt(self, conv):
        seps = [conv.sep, conv.sep2]
        ret = conv.system
        for i, (role, message) in enumerate(conv.messages):
            if message:
                ret += role + ":\n" + message + seps[i % 2]
                if i % 2 == 1:
                    ret += "\n\n"
            else:
                ret += role + ":\n"
        return ret

    def compute_skip_echo_len(model_name: str, conv, prompt: str):
        special_toks = ["### Instruction:", "### Response:", "### End"]
        skip_echo_len = len(prompt)
        for tok in special_toks:
            skip_echo_len -= prompt.count(tok) * len(tok)
