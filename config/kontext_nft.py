import imp
import os

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))


def get_config(name):
    return globals()[name]()

def _get_config(base_model="kontext", n_gpus=1, gradient_step_per_epoch=1, reward_fn={}, name="", num_image_per_prompt=12, initial_bsz=3):
    config = base.get_config()
    config.logdir = os.getenv("LOGDIR", os.getenv("TRAIN_LOGDIR", config.logdir))
    config.num_epochs = int(os.getenv("NUM_EPOCHS", str(config.num_epochs)))
    config.eval_freq = int(os.getenv("EVAL_FREQ", "1"))

    config.base_model = base_model
    config.dataset = os.getenv("DATASET_ROOT", "/nvmedata/workspace2/users/wzt/dataset/GRPO")

    config.pretrained.model = os.getenv(
        "PRETRAINED_MODEL",
        "/nvmedata/workspace2/users/wzt/pretrained/FLUX.1-Kontext-dev",
    )
    config.sample.num_steps = int(os.getenv("SAMPLE_NUM_STEPS", "20"))
    config.sample.eval_num_steps = int(os.getenv("SAMPLE_EVAL_NUM_STEPS", "15"))
    config.sample.guidance_scale = float(os.getenv("SAMPLE_GUIDANCE_SCALE", "1.5"))
    config.resolution = int(os.getenv("TRAIN_RESOLUTION", "512"))
    config.save_freq = int(os.getenv("SAVE_FREQ", "10"))
    lora_path = os.getenv(
        "LORA_PATH",
        os.getenv(
            "TRAIN_LORA_PATH",
            "/nvmedata/workspace2/users/wzt/DiffSynth-Studio/checkpoint/step-30000.safetensors",
        ),
    )
    if str(lora_path).strip().lower() in {"", "none", "null", "false", "off", "0"}:
        lora_path = None
    config.train.lora_path = lora_path
    config.train.beta = 0.0001
    config.sample.noise_level = float(os.getenv("SAMPLE_NOISE_LEVEL", "0.7"))
    bsz = initial_bsz

    config.sample.num_image_per_prompt = num_image_per_prompt

    config.sample.ban_std_thres = 0.05
    config.sample.ban_mean_thres = 0.9
    config.sample.ban_prompt = False

    num_groups = int(os.getenv("NUM_GROUPS_PER_EPOCH", "5"))

    while True:
        if bsz < 1:
            assert False, "Cannot find a proper batch size."
        if (
            num_groups * config.sample.num_image_per_prompt % (n_gpus * bsz) == 0
            and bsz * n_gpus % config.sample.num_image_per_prompt == 0
        ):
            n_batch_per_epoch = num_groups * config.sample.num_image_per_prompt // (n_gpus * bsz)
            if n_batch_per_epoch % gradient_step_per_epoch == 0:
                config.sample.train_batch_size = bsz
                config.sample.num_batches_per_epoch = n_batch_per_epoch
                config.train.batch_size = config.sample.train_batch_size
                config.train.gradient_accumulation_steps = (
                    config.sample.num_batches_per_epoch // gradient_step_per_epoch
                )
                break
        bsz -= 1

    # special design, the test set has a total of 1018/2212/2048 for ocr/geneval/pickscore, to make gpu_num*bs*n as close as possible to it, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.
    config.sample.test_batch_size = bsz
    if n_gpus > 32:
        config.sample.test_batch_size = config.sample.test_batch_size // 2

    config.prompt_fn = "geneval"

    config.run_name = f"nft_{base_model}_{name}"
    config.save_dir = os.path.join(config.logdir, "nft", base_model, name)
    config.reward_fn = reward_fn

    config.decay_type = 1
    config.beta = 1.0
    config.train.adv_mode = "all"

    # config.sample.guidance_scale = 1.0
    config.sample.deterministic = os.getenv("SAMPLE_DETERMINISTIC", "1").strip().lower() in {"1", "true", "yes", "y"}
    config.sample.solver = os.getenv("SAMPLE_SOLVER", "dpm2").strip()
    return config

def kontext_mllm_reward():
    reward_fn = {
        "mllm_score_continue": 1.0,
    }
    n_gpus = int(os.getenv("NPROC_PER_NODE", "8"))
    num_image_per_prompt = int(
        os.getenv("QWEN_NUM_CANDIDATES", os.getenv("SINGLE_NUM_CANDIDATES", "8"))
    )
    config = _get_config(
        base_model="kontext",
        n_gpus=n_gpus,
        gradient_step_per_epoch=1,
        reward_fn=reward_fn,
        name="mllm_score_continue",
        num_image_per_prompt=num_image_per_prompt,
        initial_bsz=int(os.getenv("QWEN_INITIAL_BSZ", os.getenv("SINGLE_INITIAL_BSZ", "8" if n_gpus == 1 else "3"))),
    )
    return config

def kontext_mllm_reward_ban_prompt():
    reward_fn = {
        "mllm_score_continue": 1.0,
    }
    n_gpus = int(os.getenv("NPROC_PER_NODE", "8"))
    num_image_per_prompt = int(
        os.getenv("QWEN_NUM_CANDIDATES", os.getenv("SINGLE_NUM_CANDIDATES", "8"))
    )
    config = _get_config(
        base_model="kontext",
        n_gpus=n_gpus,
        gradient_step_per_epoch=1,
        reward_fn=reward_fn,
        name="mllm_score_continue_ban_prompt",
        num_image_per_prompt=num_image_per_prompt,
        initial_bsz=int(os.getenv("QWEN_INITIAL_BSZ", os.getenv("SINGLE_INITIAL_BSZ", "8" if n_gpus == 1 else "3"))),
    )
    config.sample.ban_prompt = True
    config.sample.ban_std_thres = 0.05
    return config


def kontext_relative_api_reward():
    reward_fn = {
        "mllm_relative_api_score": 1.0,
    }
    n_gpus = int(os.getenv("NPROC_PER_NODE", "8"))
    num_image_per_prompt = int(os.getenv("RELATIVE_NUM_CANDIDATES", "8"))
    config = _get_config(
        base_model="kontext",
        n_gpus=n_gpus,
        gradient_step_per_epoch=1,
        reward_fn=reward_fn,
        name="relative_api_score",
        num_image_per_prompt=num_image_per_prompt,
        initial_bsz=int(os.getenv("RELATIVE_INITIAL_BSZ", "8" if n_gpus == 1 else "3")),
    )
    return config


def kontext_single_api_reward():
    reward_fn = {
        "mllm_single_api_score": 1.0,
    }
    n_gpus = int(os.getenv("NPROC_PER_NODE", "8"))
    num_image_per_prompt = int(os.getenv("SINGLE_NUM_CANDIDATES", "8"))
    config = _get_config(
        base_model="kontext",
        n_gpus=n_gpus,
        gradient_step_per_epoch=1,
        reward_fn=reward_fn,
        name="single_api_score",
        num_image_per_prompt=num_image_per_prompt,
        initial_bsz=int(os.getenv("SINGLE_INITIAL_BSZ", "8" if n_gpus == 1 else "3")),
    )
    return config
