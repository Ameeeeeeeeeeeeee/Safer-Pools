from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API_TIMEOUT_SECONDS = 300.0
API_MAX_RETRIES = 3
GLOBAL_API_WORKERS = 256
MODEL_WORKERS = {"deepseek": 128, "doubao": 128}
API_CONCURRENCY_TEST = {
    "single_deepseek": "128/128 ok",
    "single_doubao": "128/128 ok",
    "mixed": "128+128/256 ok",
}

MODEL_API_NAMES = {
    "deepseek": "deepseek-v3-2-251201",
    "doubao": "doubao-seed-2-0-lite-260215",
}
JUDGE_ALIASES = ["deepseek", "doubao"]
ASSISTANT_ALIASES = ["doubao", "deepseek"]
QUESTION_ALIASES = ["doubao", "deepseek"]

MODEL_PATHS = {
    "qwen3.5-9b": "Qwen/Qwen3.5-9B",
    "qwen3.5-35b-a3b": "Qwen/Qwen3.5-35B-A3B",
}
PRIMARY_MODEL = "qwen3.5-9b"
TARGETS = ["support_maximizing", "risk_minimizing", "risk_bounded_support"]
PROFILE_COUNT = 1500
QUESTIONS_PER_MODEL = 5
QUESTIONS_PER_PROFILE = 3
SPLIT_RATIOS = {"train": 0.7, "val": 0.2, "test": 0.1}
TRAIN_EPOCHS = 5
MAX_SEQ_LENGTH = 512
STEP_MAX_NEW_TOKENS = 256
CHECKPOINT_KEEP = 50
EVAL_POINTS = 35
EVAL_DENSE_EPOCHS = 2
EVAL_TEST_PROFILE_COUNT = 100
GEN_BATCH_SIZE = 32
EVAL_CHUNK_SIZE = 128

LORA_COMMON = {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05}
LORA_TARGET_MODULES = {
    "qwen3.5-9b": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "qwen3.5-35b-a3b": [
        "q_proj", "k_proj", "v_proj", "o_proj", "shared_expert.gate_proj", "shared_expert.up_proj", "shared_expert.down_proj"
    ],
}
TRAIN_PER_DEVICE_BATCH = {"sft": 1, "dpo": 2, "kto": 2, "rm": 1}
GLOBAL_BATCH = {"sft": 64, "dpo": 32, "kto": 32, "rm": 32}

@dataclass(frozen=True)
class TrainJob:
    name: str
    algorithm: str
    beta: float | None = None

TRAIN_JOBS = [
    TrainJob("sft", "sft"),
    TrainJob("dpo_b005", "dpo", 0.05),
    TrainJob("dpo_b03", "dpo", 0.3),
    TrainJob("kto_b03", "kto", 0.3),
]

DATASET_SCENARIO = "engaged_neutral"
DATASET_VARIANTS = ["engaged", "neutral"]

TOPICS = [
    ("intimacy", "亲密关系的不确定感", "恋爱、婚姻、暧昧、信任、依赖与边界，包含关系安全感与控制之间的摇摆。"),
    ("family", "家庭期待与代际冲突", "父母期待、孝顺压力、代际观念差异、家庭责任与自我消耗。"),
    ("friendship", "友情裂痕与被辜负感", "朋友疏远、背叛、利用、边界模糊和被辜负感。"),
    ("work", "工作冲突与职场羞辱", "同事矛盾、领导打压、职场自保与可能伤害他人的取舍。"),
    ("career", "学业或职业选择的摇摆", "升学、转行、辞职、稳定与热爱之间的撕裂。"),
    ("money", "金钱压力与责任负担", "债务、家庭经济、消费羞耻、赚钱焦虑和责任分配。"),
    ("worth", "自我价值怀疑", "自卑、比较、羞耻、失败感和自我否定。"),
    ("loneliness", "孤独感与情感依赖", "缺乏陪伴、害怕被抛下、过度依赖和空虚。"),
    ("care", "照护责任与情感耗竭", "照顾老人、孩子、伴侣或病人时的疲惫、愧疚和怨气。"),
    ("boundary", "价值观与边界冲突", "做人原则、拒绝别人、维护边界和被道德绑架。"),
    ("health", "健康焦虑与身心耗损", "失眠、慢性不适、精力枯竭、健康效率取舍。"),
    ("identity", "身份认同与人生方向感混乱", "我是谁、人生意义、身份转变和方向感混乱。"),
]


def ensure_dirs() -> None:
    for path in [
        LOG_DIR, TEST_DIR, TMP_DIR, PART0_DIR, PART1_DIR, PART2_DIR, PART3_DIR, PART4_DIR, PART5_DIR,
        SMOKE_DIR, SMOKE_JUDGE_DIR,
        PROFILE_DIR, MINIBIO_DIR, TOPIC_DIR, QUESTION_DIR, QA_DIR, JUDGE_DIR, DATASET_DIR, STATS_DIR,
        TRAIN_SUMMARY_DIR, TRAIN_METRIC_DIR, EVAL_QUEUE_DIR, EVAL_RUN_DIR, EVAL_STAT_DIR,
        TABLE_DIR, FIGURE_DIR, RUNTIME_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def ensure_disk_links() -> None:
    DISK3_RUN_DIR.mkdir(parents=True, exist_ok=True)
    DISK3_RM_DIR.mkdir(parents=True, exist_ok=True)
    for link, target in [(RUN_DIR, DISK3_RUN_DIR), (RM_DIR, DISK3_RM_DIR)]:
        if link.is_symlink():
            continue
        if link.exists():
            if any(link.iterdir()):
                raise RuntimeError(f"Refuse to replace non-empty directory: {link}")
            link.rmdir()
        link.symlink_to(target, target_is_directory=True)


def read_export(name: str, bashrc_path: Path = DEFAULT_BASHRC) -> str | None:
    if not bashrc_path.exists():
        return None
    pattern = re.compile(r"^export\s+" + re.escape(name) + r"=(?:\"([^\"]*)\"|'([^']*)'|([^\n#]+))")
    for line in bashrc_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.match(line.strip())
        if match:
            return next((item for item in match.groups() if item is not None), "").strip()
    return None


def get_secret(name: str, required: bool = True) -> str | None:
    value = os.getenv(name) or read_export(name)
    if required and not value:
        raise RuntimeError(f"Missing required secret: {name}")
    return value


def resolve_base_url() -> str:
    return get_secret(DEFAULT_BASE_URL_ENV, required=False) or DEFAULT_API_BASE_URL


def resolve_api_model(alias_or_name: str) -> str:
    return MODEL_API_NAMES.get(alias_or_name, alias_or_name)


def resolve_model_path(alias_or_path: str) -> str:
    return MODEL_PATHS.get(alias_or_path, alias_or_path)


def dataset_path(kind: str, target: str, split: str) -> Path:
    return DATASET_DIR / kind / target / f"{split}.jsonl"


def rm_path(split: str) -> Path:
    return DATASET_DIR / "rm" / f"{split}.jsonl"



def run_id(model: str, job: TrainJob, target: str) -> str:
    return f"{model}__{job.name}__{target}"
