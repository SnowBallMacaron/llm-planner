import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
    BitsAndBytesConfig,
)

class StopOnSequence(StoppingCriteria):
    """
    Stop generation once any of the specified token sequences is produced.
    """
    def __init__(self, stop_sequences, tokenizer):
        self.stop_token_ids = [
            tokenizer(seq, add_special_tokens=False)["input_ids"]
            for seq in stop_sequences
        ]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for seq_ids in self.stop_token_ids:
            if len(input_ids[0]) >= len(seq_ids):
                if torch.equal(
                    input_ids[0, -len(seq_ids):].cpu(),
                    torch.tensor(seq_ids, dtype=torch.long)
                ):
                    return True
        return False

class LocalLLM:
    """
    A local Hugging Face LLM wrapper supporting full sampling hyperparameters,
    custom stop-sequences, quantization, and Qwen3 thinking mode with token-level filtering.
    """
    def __init__(
        self,
        model_name: str,
        torch_dtype: str = "auto",
        device_map: str = "auto",
        trust_remote_code: bool = True,
        # Quantization flags
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        bnb_4bit_quant_type: str = "nf4",
        bnb_4bit_compute_dtype: str = "auto",
        # Offloading
        offload_folder: str = None,
        # Qwen3 thinking
        enable_thinking: bool = False,
        thinking_open: str = "<think>",
        thinking_close: str = "</think>",
        **kwargs
    ):
        self.enable_thinking = enable_thinking
        self.thinking_open = thinking_open
        self.thinking_close = thinking_close
        # Prepare BitsAndBytes config if 4-bit quantization is requested
        quant_config = None
        if load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=(torch.float16 if bnb_4bit_compute_dtype in ("auto", "float16") else torch.bfloat16),
            )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            trust_remote_code=trust_remote_code
        )

        # Load model with optional quantization/offloading
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            load_in_8bit=load_in_8bit,
            quantization_config=quant_config,
            offload_folder=offload_folder,
            **kwargs
        )

    def generate(
        self,
        *,
        prompt: str = None,
        messages: list[dict] = None,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        num_return_sequences: int = 1,
        no_repeat_ngram_size: int = 0,
        length_penalty: float = 1.0,
        do_sample: bool = True,
        stop: list[str] = None,
    ) -> list[str]:
        # Build input text
        if messages is not None:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        elif prompt is not None:
            text = prompt
        else:
            raise ValueError("Either prompt or messages must be provided.")

        # Tokenize prompt
        inputs = self.tokenizer([text], return_tensors="pt", padding=True).to(self.model.device)
        input_ids = inputs["input_ids"][0]

        # Prepare stopping criteria
        stopping_criteria = None
        if stop:
            stopping_criteria = StoppingCriteriaList([StopOnSequence(stop, self.tokenizer)])

        # Generate tokens
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            # repetition_penalty=repetition_penalty,
            num_return_sequences=num_return_sequences,
            no_repeat_ngram_size=no_repeat_ngram_size,
            length_penalty=length_penalty,
            do_sample=do_sample,
            # stopping_criteria=stopping_criteria,
            use_cache=True,
        )

        results = []
        idx = 0
        # Process each generated sequence
        for seq in outputs:
            # Remove prompt tokens
            gen_ids = seq[len(input_ids):].tolist()
            if self.enable_thinking:
                # Identify token IDs for thinking markers
                open_id = self.tokenizer(self.thinking_open, add_special_tokens=False)["input_ids"][0]
                close_id = self.tokenizer(self.thinking_close, add_special_tokens=False)["input_ids"][0]
                # Find last occurrence of close marker
                idx = len(gen_ids) - gen_ids[::-1].index(close_id)
                # Slice to get only answer tokens
                answer_ids = gen_ids[idx:]
            else:
                answer_ids = gen_ids
            # Decode and append
            think_text = self.tokenizer.decode(gen_ids[:idx], skip_special_tokens=True).strip()
            answer_text = self.tokenizer.decode(answer_ids, skip_special_tokens=True).strip().split(":")[-1]
            results.append(answer_text)
            # print(f"Input: {text}")
            # print(f"Answer (think): {think_text} \n Answer (out): {answer_text}")
        return results
