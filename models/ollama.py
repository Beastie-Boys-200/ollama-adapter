from pydantic import BaseModel 
from typing import Any


class OllamaOptions(BaseModel):

    # --- load time options ---

    # Enables or disables NUMA (Non-Uniform Memory Access) 
    # to optimize memory management on multiprocessor systems.
    numa: bool | None = None
    # Sets the size of the context window for generating the next 
    # token (the size of the history the model works with).
    num_ctx: int | None = None
    # Batch size (number of examples processed simultaneously) 
    # during training or generation.
    num_batch: int | None = None
    # The number of graphics processing units (GPUs) used.
    num_gpu: int | None = None
    # The index of the primary GPU to be used for computation.
    main_gpu: int | None = None
    # Enables a low-VRAM mode, optimizing performance 
    # on systems with limited VRAM.
    low_vram: bool | None = None
    # Enables the use of 16-bit values (float16) to reduce 
    # the amount of memory consumed by key-value pairs.
    f16_kv: bool | None = None
    # Uses a memory-mapped file to optimize performance 
    # when handling large models loaded into memory.
    use_mmap: bool | None = None
    # Number of threads used for calculation
    num_thread: int | None = None


    # --- runtime options ---
    
    # Specifies the number of tokens to preserve when generating text 
    # (for example, to preserve a certain part of the context).
    num_keep: int | None = None
    # Sets the seed value for random number generation.
    seed: int | None = None
    # The maximum number of tokens for prediction. 
    # A value of -1 allows infinite generation.
    num_predict: int | None = None
    # Limits the number of tokens the model can select at each step. 
    # The higher the value, the greater the diversity of responses.
    top_k: int | None = None
    # Used for kernel sampling (probability sampling). 
    # Specifies the proportion of probability that the model will 
    # consider when generating samples.
    top_p: float | None = None
    # Tail free sampling Reduces the impact of unlikely tokens on output.
    tfs_z: float | None = None
    # Indicates a typical probability to balance diversity and text quality.
    typical_p: float | None = None
    # Specifies the number of tokens the model should consider to avoid 
    # repetitions. -1 means use the entire context.
    repeat_last_n: int | None = None
    # Controls the model's creativity. High values make responses more 
    # diverse, while low values make them more predictable.
    temperature: float | None = None
    # Sets the penalty level for repetition. The higher the value, 
    # the greater the penalty for repeated phrases.
    repeat_penalty: float | None = None
    # Controls the presence of new tokens in the output, 
    # increasing the likelihood of new words appearing.
    presence_penalty: float | None = None
    # Controls the frequency of token occurrence, reducing 
    # the likelihood of repetition of frequently occurring words.
    frequence_penalty: float | None = None
    # Enables the Mirostat algorithm for text perplexity control. 
    # 0 — disabled, 1 — enabled, 2 — Mirostat 2.0.
    mirostat: int | None = None
    # Controls the balance between coherence and diversity of text.
    mirostat_tau: float | None = None
    # Adjusts the Mirostat's response speed. 
    # A higher value makes the model more responsive.
    mirostat_tea: float | None = None


    @property
    def get_dict(self) -> dict[str, Any]: return { 
        i: getattr(self, i) for i in self.model_dump() if getattr(self, i) is not None 
    }
