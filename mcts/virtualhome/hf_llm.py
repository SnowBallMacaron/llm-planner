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
        # Convert each stop string into its token-ID sequence
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
    custom stop-sequences, and optional quantization via BitsAndBytes.
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
        **kwargs
    ):
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
            attn_implementation="flash_attention_2",
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
        # Build text input
        if messages is not None:
            # For chat-capable models, apply HF chat template
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        elif prompt is not None:
            text = prompt
        else:
            raise ValueError("Either prompt or messages must be provided.")

        # Tokenize
        inputs = self.tokenizer([text], return_tensors="pt", padding=True).to(self.model.device)

        # Prepare stopping criteria
        stopping_criteria = None
        if stop:
            stopping_criteria = StoppingCriteriaList([StopOnSequence(stop, self.tokenizer)])

        # Generate
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            num_return_sequences=num_return_sequences,
            no_repeat_ngram_size=no_repeat_ngram_size,
            length_penalty=length_penalty,
            do_sample=do_sample,
            stopping_criteria=stopping_criteria,
        )

        # Decode, stripping the prompt tokens
        results = []
        prompt_len = inputs["input_ids"].shape[1]
        for seq in outputs:
            gen_tokens = seq[prompt_len:]
            text_out = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
            results.append(text_out)

        return results
