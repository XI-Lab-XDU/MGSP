# SPDX-License-Identifier: Apache-2.0
"""
GAPTR: Granularity-Aware Progressive Text Refinement
"""

import math
import os
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple
from enum import Enum
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.io import loadmat
import torch
import torch.nn.functional as F
import os.path as osp

from vllm import LLM, EngineArgs, SamplingParams
from vllm.utils import FlexibleArgumentParser


IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28
MAX_RATIO = 200

MODEL_PATH = "Qwen/Qwen2.5-VL-32B-Instruct"
OUTPUT_DIR = "./output"
OUTPUT_FILE = None

DATASET_ROOTS = {
    "PRW": "./data/PRW",
    "CUHK": "./data/CUHK-SYSU",
    "G2APS": "./data/G2APS",
}

GAPTR_CONFIG = {
    "lambda_param": 0.5,
    "threshold_global": 0.6,
    "threshold_local": 0.6,
    "max_iterations": 5,
}

POSITIVE_DIMENSIONS = {
    "global": {
        "count": ["one person", "two people", "three people", "multiple people"],
        "spatial": ["left", "right", "center", "top", "bottom"],
        "density": ["sparse", "crowded", "evenly distributed", "clustered"],
    },
    "local": {
        "top": ["shirt", "t-shirt", "jacket", "coat", "sweater", "hoodie", "vest"],
        "bottom": ["pants", "jeans", "shorts", "skirt", "trousers"],
        "footwear": ["shoes", "sneakers", "boots", "sandals"],
        "accessory": ["bag", "backpack", "handbag", "shoulder bag", "purse",
                 "hat", "cap", "glasses", "sunglasses", "scarf"],
        "color": ["white", "black", "red", "blue", "green", "gray", "grey", "brown",
                 "yellow", "orange", "pink", "purple", "navy", "dark", "light"],
        "hair": ["hair", "short hair", "long hair", "bald",
                 "black hair", "brown hair", "blonde"],
    },
}

NEGATIVE_KEYWORDS = {
    "global": [
        "wearing", "wears", "dressed",
        "shirt", "jacket", "pants", "dress", "coat", "skirt",
        "shoes", "boots", "sneakers",
        "bag", "backpack", "hat", "scarf",
        "red", "blue", "white", "black",
        "short hair", "long hair",
    ],
    "local": [
        "street", "road", "sidewalk", "building", "indoor", "outdoor",
        "parking lot", "entrance", "hallway", "store", "shop",
        "background scene", "environment", "setting",
        "city", "urban", "rural", "parking area",
        "wall", "floor", "ground", "sky",
    ],
}


class Granularity(Enum):
    GLOBAL = "global"
    LOCAL = "local"


@dataclass
class QualityScore:
    overall: float
    pos_score: float
    neg_score: float
    dim_hits: Dict[str, int]
    pollution: List[str]
    refinement_hints: List[str]


def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor


def smart_resize(
        height: int,
        width: int,
        factor: int = IMAGE_FACTOR,
        min_pixels: int = MIN_PIXELS,
        max_pixels: int = MAX_PIXELS
) -> Tuple[int, int]:
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(f"Aspect ratio must be smaller than {MAX_RATIO}")

    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)

    return h_bar, w_bar


def resize_image(image: Image.Image) -> Image.Image:
    width, height = image.size
    resized_h, resized_w = smart_resize(
        height, width,
        factor=IMAGE_FACTOR,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    return image.resize((resized_w, resized_h))


class TextScorer:
    def __init__(self):
        self.pos_dims = POSITIVE_DIMENSIONS
        self.neg_kws = NEGATIVE_KEYWORDS

    def compute_scores(
            self,
            text: str,
            granularity: Granularity
    ) -> Tuple[float, float, Dict[str, int], List[str]]:
        text_lower = text.lower()
        gran_key = granularity.value

        dimensions = self.pos_dims[gran_key]
        dim_hits = {}
        total_dim_score = 0.0

        for dim_name, keywords in dimensions.items():
            hit_count = sum(1 for kw in keywords if kw in text_lower)
            dim_hits[dim_name] = hit_count
            dim_score = 1.0 if hit_count > 0 else 0.0
            total_dim_score += dim_score

        pos_score = total_dim_score / len(dimensions) if dimensions else 0.0

        neg_keywords = self.neg_kws[gran_key]
        pollution = [kw for kw in neg_keywords if kw in text_lower]
        neg_score = 1.0 if pollution else 0.0

        return pos_score, neg_score, dim_hits, pollution

    def generate_hints(
            self,
            dim_hits: Dict[str, int],
            pollution: List[str],
            granularity: Granularity
    ) -> List[str]:
        hints = []
        is_global = granularity == Granularity.GLOBAL

        if is_global:
            if dim_hits.get("count", 0) == 0:
                hints.append("Add person count description (e.g., one person, multiple people)")
            if dim_hits.get("spatial", 0) == 0:
                hints.append("Add spatial position description (e.g., left, right, center)")
            if dim_hits.get("density", 0) == 0:
                hints.append("Add crowd density description (e.g., sparse, crowded)")
            if pollution:
                hints.append("Remove clothing/appearance details (focus on spatial layout only)")
        else:
            if dim_hits.get("top", 0) == 0 and dim_hits.get("bottom", 0) == 0:
                hints.append("Add clothing description (top and bottom garments)")
            if dim_hits.get("color", 0) == 0:
                hints.append("Add color information")
            if dim_hits.get("accessory", 0) == 0:
                hints.append("Add accessory information (e.g., bag, hat, glasses)")
            if dim_hits.get("footwear", 0) == 0:
                hints.append("Add footwear information")
            if pollution:
                hints.append("Remove background/scene descriptions (focus on person appearance only)")

        return hints if hints else ["Provide more detailed and accurate description"]


class QualityEvaluator:
    def __init__(self, lambda_param: float = None):
        self.lambda_param = lambda_param or GAPTR_CONFIG["lambda_param"]
        self.scorer = TextScorer()

    def evaluate(
            self,
            text: str,
            image: Image.Image,
            granularity: Granularity
    ) -> QualityScore:
        pos_score, neg_score, dim_hits, pollution = self.scorer.compute_scores(
            text, granularity
        )

        hints = self.scorer.generate_hints(dim_hits, pollution, granularity)

        overall_score = (
            self.lambda_param * pos_score +
            (1 - self.lambda_param) * (1 - neg_score)
        )

        return QualityScore(
            overall=overall_score,
            pos_score=pos_score,
            neg_score=neg_score,
            dim_hits=dim_hits,
            pollution=pollution,
            refinement_hints=hints,
        )

    def need_refinement(
            self,
            quality_score: QualityScore,
            granularity: Granularity
    ) -> bool:
        if granularity == Granularity.GLOBAL:
            threshold = GAPTR_CONFIG["threshold_global"]
        else:
            threshold = GAPTR_CONFIG["threshold_local"]
        return quality_score.overall < threshold


class PromptGenerator:
    @staticmethod
    def get_initial_prompt(granularity: Granularity) -> str:
        if granularity == Granularity.GLOBAL:
            return (
                "You are given an image containing single or multiple pedestrians.\n"
                "Generate a global description that summarizes only the spatial layout "
                "and distribution of the people in the image.\n"
                "Focus on:\n"
                "- Where the pedestrians are located "
                "(e.g., mostly in the center, left, top, or bottom)\n"
                "- How many people are visible "
                "(one person, two people, or multiple people)\n"
                "- How dense the crowd is "
                "(e.g., sparse, crowded, evenly distributed)\n"
                "- Any noticeable variations in size that suggest distance\n"
                "Do not describe clothing, actions, objects, or background details.\n"
                "Write 2 to 3 concise sentences. Use simple and direct language."
            )
        else:
            return (
                "You are given an image of a single person.\n"
                "Your task is to generate a detailed but objective and concise natural "
                "language description of the person's visible appearance.\n"
                "Only describe clearly visible attributes such as clothing, accessories, "
                "belongings, and footwear.\n"
                "Do not make assumptions about material, style, or context.\n"
                "Do not include any statements about image quality "
                "(such as 'too blurry' or 'not clear').\n"
                "Always try to identify the person's gender (man/woman) from the image "
                "if possible. Use 'the person' only if gender is truly ambiguous.\n"
                "If you can only identify one detail (e.g. 'a red bag'), "
                "output only that.\n"
                "Use simple and direct language.\n"
                "Example: The man is wearing a blue shirt, black pants, and white shoes, "
                "carrying a black backpack."
            )

    @staticmethod
    def generate_refinement_prompt(
            original_text: str,
            quality_score: QualityScore,
            granularity: Granularity,
            iteration_history: List[Dict] = None
    ) -> str:
        hints_text = "\n".join([f"- {hint}" for hint in quality_score.refinement_hints])

        history_context = ""
        if iteration_history:
            failed_hints = set()
            for hist in iteration_history:
                prev_hint_list = hist.get("refinement_hints", [])
                for h in prev_hint_list:
                    failed_hints.add(h)

            if failed_hints:
                failed_hints_text = "\n".join([f"- {hint}" for hint in failed_hints])
                history_context = (
                    f"\n\nNote: The following refinement attempts were made in previous "
                    f"iterations but did not fully resolve the issues:\n"
                    f"{failed_hints_text}\n"
                    f"Please try a different approach or rephrase the description "
                    f"from a different angle.\n"
                )

        if granularity == Granularity.GLOBAL:
            return (
                f'You previously generated this description:\n'
                f'"{original_text}"\n\n'
                f"This description was evaluated and found to have "
                f"the following issues:\n"
                f"{hints_text}\n"
                f"{history_context}"
                "Please regenerate a improved description that:\n"
                "1. Fixes the above issues\n"
                "2. Maintains the correct spatial and density information\n"
                "3. Is concise (2-3 sentences)\n\n"
                "Write only the new description, nothing else."
            )
        else:
            return (
                f'You previously generated this description:\n'
                f'"{original_text}"\n\n'
                f"This description was evaluated and found to have "
                f"the following issues:\n"
                f"{hints_text}\n"
                f"{history_context}"
                "Please regenerate a improved description that:\n"
                "1. Fixes the above issues\n"
                "2. Provides more complete appearance details "
                "(clothing, color, accessories)\n"
                "3. Is concise and objective\n\n"
                "Write only the new description, nothing else."
            )


class VLLMInference:
    def __init__(self, model_path: str):
        import vllm.transformers_utils.config as _cfg

        _orig_rope = _cfg.patch_rope_scaling_dict

        def _patch_rope_scaling_dict(rope_scaling):
            if isinstance(rope_scaling, dict):
                rope_scaling.pop("type", None)
            return _orig_rope(rope_scaling)

        _cfg.patch_rope_scaling_dict = _patch_rope_scaling_dict

        engine_args = EngineArgs(
            model=model_path,
            max_model_len=4096,
            max_num_seqs=1,
            tensor_parallel_size=2,
        )

        self.llm = LLM(**asdict(engine_args))

        self.sampling_params = SamplingParams(
            temperature=0.2,
            max_tokens=128,
        )

    def build_prompt(self, prompt_text: str) -> str:
        return (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
            f"{prompt_text}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def generate(self, image: Image.Image, prompt_text: str, granularity: Granularity) -> str:
        image = resize_image(image)

        inputs = {
            "prompt": self.build_prompt(prompt_text),
            "multi_modal_data": {
                "image": image
            },
        }

        outputs = self.llm.generate(
            inputs,
            sampling_params=self.sampling_params,
            use_tqdm=False
        )

        return outputs[0].outputs[0].text.strip()


class GAPTRPipeline:
    def __init__(
            self,
            model_path: str,
            lambda_param: float = None,
            max_iterations: int = None,
    ):
        self.inference = VLLMInference(model_path)
        self.evaluator = QualityEvaluator(lambda_param)
        self.max_iterations = max_iterations or GAPTR_CONFIG["max_iterations"]

    def process_single(
            self,
            image: Image.Image,
            granularity: Granularity,
            image_name: str = ""
    ) -> Dict:
        history = []
        current_text = ""
        current_prompt = PromptGenerator.get_initial_prompt(granularity)

        for iteration in range(self.max_iterations):
            generated_text = self.inference.generate(
                image, current_prompt, granularity
            )

            quality_score = self.evaluator.evaluate(
                generated_text, image, granularity
            )

            print(
                f"[{image_name}] Iter {iteration + 1}: "
                f"Q={quality_score.overall:.3f} "
                f"(pos={quality_score.pos_score:.3f}, "
                f"neg={quality_score.neg_score:.3f})"
            )
            print(f"         Text: {generated_text[:80]}...")

            if not self.evaluator.need_refinement(quality_score, granularity):
                current_text = generated_text
                print(f"         [Early stop: quality >= threshold]")
                break

            if iteration < self.max_iterations - 1:
                current_prompt = PromptGenerator.generate_refinement_prompt(
                    generated_text, quality_score, granularity,
                    iteration_history=history[:-1]
                )
                current_text = generated_text

        return {
            "image_name": image_name,
            "granularity": granularity.value,
            "final_text": current_text,
            "quality_score": history[-1]["quality_score"] if history else 0.0,
            "iterations": len(history),
            "history": history,
        }

    def process_dataset(
            self,
            images: List[Tuple[str, Image.Image]],
            granularity: Granularity,
            desc: str = "Processing"
    ) -> List[Dict]:
        results = []

        for img_name, image in tqdm(images, desc=desc, ncols=130):
            result = self.process_single(image, granularity, img_name)
            results.append(result)

        return results

    def process_dataset_by_path(
            self,
            image_paths: List[Tuple[str, str]],
            granularity: Granularity,
            desc: str = "Processing"
    ) -> List[Dict]:
        results = []

        for img_name, img_path in tqdm(image_paths, desc=desc, ncols=130):
            image = Image.open(img_path).convert("RGB")
            result = self.process_single(image, granularity, img_name)
            image.close()
            del image
            results.append(result)

        return results


class DatasetProcessor:
    def __init__(self, dataset_name: str, root: str = None):
        self.dataset_name = dataset_name
        if root:
            self.root = root
        elif dataset_name in DATASET_ROOTS:
            self.root = DATASET_ROOTS[dataset_name]
        else:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Available: {list(DATASET_ROOTS.keys())}"
            )

    def load_global_images(self) -> List[Tuple[str, str]]:
        image_paths = []

        if self.dataset_name == "PRW":
            frame_mat = loadmat(osp.join(self.root, "frame_train.mat"))
            img_indices = frame_mat["img_index_train"]
            img_names = [img[0][0] + ".jpg" for img in img_indices]
            img_dir = osp.join(self.root, "frames")

        elif self.dataset_name == "CUHK":
            pool_mat = loadmat(osp.join(self.root, "annotation", "pool.mat"))
            gallery_imgs = pool_mat["pool"].squeeze()
            gallery_imgs = set([str(a[0]) for a in gallery_imgs])
            all_imgs_mat = loadmat(osp.join(self.root, "annotation", "Images.mat"))
            all_imgs = all_imgs_mat["Img"].squeeze()
            all_imgs = set([str(a[0][0]) for a in all_imgs])
            training_imgs = sorted(list(all_imgs - gallery_imgs))
            img_names = training_imgs
            img_dir = osp.join(self.root, "Image", "SSM")

        elif self.dataset_name == "G2APS":
            pool_mat = loadmat(osp.join(self.root, "annotation", "pool.mat"))
            gallery_imgs = pool_mat["pool"].squeeze()
            gallery_imgs = set([str(a[0]) for a in gallery_imgs])
            all_imgs_mat = loadmat(osp.join(self.root, "annotation", "Images.mat"))
            all_imgs = all_imgs_mat["Img"].squeeze()
            all_imgs = set([str(a[0][0]) for a in all_imgs])
            training_imgs = sorted(list(all_imgs - gallery_imgs))
            img_names = training_imgs
            img_dir = osp.join(self.root, "Image", "SSM")

        else:
            raise ValueError(
                f"Unsupported dataset for global: {self.dataset_name}"
            )

        for img_name in tqdm(
                img_names,
                desc=f"Scanning {self.dataset_name} global images",
                ncols=130
        ):
            img_path = osp.join(img_dir, img_name)
            if osp.exists(img_path):
                image_paths.append((img_name, img_path))

        return image_paths

    def load_local_images(self) -> List[Tuple[str, str]]:
        image_paths = []
        crop_dir = osp.join(self.root, "crop_image")

        if not osp.exists(crop_dir):
            print(f"Warning: crop_image directory not found in {self.root}")
            return image_paths

        for img_name in tqdm(
                os.listdir(crop_dir),
                desc=f"Scanning {self.dataset_name} local images",
                ncols=130
        ):
            if img_name.endswith((".jpg", ".png", ".jpeg")):
                img_path = osp.join(crop_dir, img_name)
                image_paths.append((img_name, img_path))

        return image_paths

    def save_results(
            self,
            results: List[Dict],
            save_path: str,
            granularity: Granularity
    ):
        output = {}
        for result in results:
            img_name = result["image_name"].replace(".jpg", "")
            output[img_name] = result["final_text"]

        save_dir = osp.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        avg_quality = np.mean([r["quality_score"] for r in results])
        avg_quality_pre = np.mean([r["history"][0]["quality_score"] for r in results])
        avg_iterations = np.mean([r["iterations"] for r in results])

        history_path = save_path.replace(".json", "_history.json")
        history_data = {
            "statistics": {
                "average_quality_score": float(avg_quality),
                "average_quality_score_pre": float(avg_quality_pre),
                "quality_improvement": float(avg_quality - avg_quality_pre),
                "average_iterations": float(avg_iterations),
                "total_images": len(results),
            },
            "granularity": granularity.value,
            "results": results,
        }
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, ensure_ascii=False, indent=2)

        print(f"Results saved to {save_path}")
        print(f"History saved to {history_path}")
        print(f"\n=== Statistics ===")
        print(f"Average quality score: {avg_quality:.3f}")
        print(f"Average iterations: {avg_iterations:.2f}")


def main(args):
    if args.lambda_param is not None:
        GAPTR_CONFIG["lambda_param"] = args.lambda_param
    if args.max_iterations is not None:
        GAPTR_CONFIG["max_iterations"] = args.max_iterations
    if args.quality_threshold is not None:
        GAPTR_CONFIG["threshold_global"] = args.quality_threshold
        GAPTR_CONFIG["threshold_local"] = args.quality_threshold
    if args.quality_threshold_global is not None:
        GAPTR_CONFIG["threshold_global"] = args.quality_threshold_global
    if args.quality_threshold_local is not None:
        GAPTR_CONFIG["threshold_local"] = args.quality_threshold_local

    dataset_processor = DatasetProcessor(args.dataset, args.dataset_root)

    granularity = (
        Granularity.GLOBAL
        if args.granularity == "global"
        else Granularity.LOCAL
    )

    output_dir = args.output_dir or dataset_processor.root
    if args.output_file:
        save_path = osp.join(output_dir, args.output_file)
    else:
        save_path = osp.join(
            output_dir,
            f"{args.dataset}_{args.granularity}.json"
        )

    if granularity == Granularity.GLOBAL:
        print("Scanning global image paths...")
        image_paths = dataset_processor.load_global_images()
    else:
        print("Scanning local image paths...")
        image_paths = dataset_processor.load_local_images()

    print(f"Found {len(image_paths)} images")

    if args.max_images > 0:
        image_paths = image_paths[:args.max_images]
        print(f"Limited to {len(image_paths)} images for processing")

    pipeline = GAPTRPipeline(
        model_path=args.model_path,
        lambda_param=GAPTR_CONFIG["lambda_param"],
        max_iterations=GAPTR_CONFIG["max_iterations"],
    )

    results = pipeline.process_dataset_by_path(
        image_paths,
        granularity,
        desc=f"Processing {args.granularity} images"
    )

    dataset_processor.save_results(results, save_path, granularity)


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description='GAPTR: Granularity-Aware Progressive Text Refinement'
    )

    parser.add_argument(
        '--dataset', '-d',
        type=str,
        required=True,
        choices=['PRW', 'CUHK', 'G2APS'],
        help='Dataset name'
    )

    granularity_group = parser.add_mutually_exclusive_group(required=True)
    granularity_group.add_argument(
        '--global',
        dest='granularity',
        action='store_const',
        const='global',
        help='Global granularity (detection task)'
    )
    granularity_group.add_argument(
        '--local',
        dest='granularity',
        action='store_const',
        const='local',
        help='Local granularity (REID task)'
    )

    parser.add_argument(
        '--model-path',
        type=str,
        default=MODEL_PATH,
        help='Path to VLM model (default: MODEL_PATH in config)'
    )
    parser.add_argument(
        '--dataset-root',
        type=str,
        default=None,
        help='Dataset root path (override default)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=OUTPUT_DIR,
        help='Output directory (default: OUTPUT_DIR in config or dataset root)'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default=OUTPUT_FILE,
        help='Output file name (default: OUTPUT_FILE in config or auto-generated)'
    )
    parser.add_argument(
        '--lambda-param', '-l',
        type=float,
        default=None,
        help='Weight for alignment score in Q_g formula'
    )
    parser.add_argument(
        '--max-iterations', '-i',
        type=int,
        default=None,
        help='Maximum refinement iterations'
    )
    parser.add_argument(
        '--quality-threshold',
        type=float,
        default=None,
        help='Quality threshold (overrides global/local defaults)'
    )
    parser.add_argument(
        '--quality-threshold-global',
        type=float,
        default=None,
        help='Quality threshold for global granularity'
    )
    parser.add_argument(
        '--quality-threshold-local',
        type=float,
        default=None,
        help='Quality threshold for local granularity'
    )
    parser.add_argument(
        '--max-images',
        type=int,
        default=0,
        help='Maximum images to process (0 = all)'
    )
    parser.add_argument(
        '--model-suffix',
        type=str,
        default="qwen2_5vl",
        help='Suffix for output file name'
    )

    args = parser.parse_args()
    main(args)
