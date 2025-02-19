import torch
from transformers import AutoModelForCausalLM

from ..base import BaseModel
from ...dataset import DATASET_TYPE, DATASET_MODALITY
from ...smp import *

try:
    from namo.api.namo import NamoVL
except ImportError:
    print("Run: pip install namo to use Namo model evaluation.")


class Namo(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True
    SIZE_DICT = {
        (24, 896): "1B",  # (num_hidden_layers, hidden_size)
        (28, 1536): "2B",
        (36, 2048): "4B",
        (28, 3584): "8B",
        (48, 5120): "16B",
        (64, 5120): "34B",
    }

    def __init__(self, model_path="MonolithFoundation/Namo-500M", **kwargs):
        assert model_path is not None
        self.model_path = model_path
        self.device = (
            torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        )
        self.dtype = torch.bfloat16
        self.model = NamoVL(self.model_path, device=self.device)

        self.image_placeholder = "<image>"
        self.gen_kwargs = dict(
            max_new_tokens=1024,
            do_sample=False,
            top_p=None,
            top_k=None,
            temperature=None,
            repetition_penalty=None,
            eos_token_id=self.model.tokenizer.eos_token_id,
            pad_token_id=self.model.tokenizer.pad_token_id,
            use_cache=True,
        )
        self.use_cot = {
            "1B": {"MathVerse", "MathVision"},
            "2B": {"MMVet", "MMStar", "MathVerse", "MathVision"},
            "4B": {"MathVerse", "MathVision"},
            "8B": {"MMVet", "MMStar", "MMMU", "MathVista", "MathVerse", "MathVision"},
            "16B": {"MMVet", "MMStar", "MMMU", "MathVista", "MathVerse", "MathVision"},
            "34B": {"MMVet", "MMStar", "MMMU", "MathVista", "MathVerse", "MathVision"},
        }
        self.frame_selector = None

    def use_custom_prompt(self, dataset):
        if any(
            dataset.startswith(prefix)
            for prefix in ["MMVet", "MathVista", "MathVerse", "MathVision"]
        ):
            return True
        if DATASET_TYPE(dataset) == "Y/N" or DATASET_TYPE(dataset) == "MCQ":
            return True
        return False

    def build_yorn_prompt(self, line, dataset=None):
        prompt = line["question"]
        if listinstr(["HallusionBench"], dataset) and self.size == "34B":
            prompt += " Please answer yes or no."
        prompt += (
            "\nAnswer the question using a single word or phrase."
            if cn_string(prompt)
            else "\nAnswer the question using a single word or phrase."
        )
        return prompt

    def build_multi_choice_prompt(self, line, dataset=None, use_cot=False):
        prompt = line["question"]
        hint = line["hint"] if ("hint" in line and not pd.isna(line["hint"])) else None
        if hint is not None:
            prompt = hint + "\n" + prompt

        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        for key, item in options.items():
            prompt += f"\n{key}. {item}"

        if len(options):
            if use_cot:
                prompt += "\nProvide a step-by-step solution to the problem, and conclude with 'the answer is' followed by the final solution."
            else:
                prompt += (
                    "\n请直接回答选项字母。"
                    if cn_string(prompt)
                    else "\nAnswer with the option's letter from the given choices directly."
                )
        return prompt

    def build_mmvet_prompt(self, line, dataset=None, use_cot=False):
        prompt = line["question"]
        if use_cot:
            prompt += "\nProvide a step-by-step solution to the problem carefully."
        return prompt

    def build_math_prompt(self, line, dataset=None, use_cot=False):
        prompt = line["question"]
        if use_cot:
            prompt += "\nProvide a step-by-step solution to the problem, and conclude with 'the answer is' followed by the final solution."
        return prompt

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)

        if dataset == "MMVet":
            prompt = self.build_mmvet_prompt(line, dataset)
        elif any(
            dataset.startswith(prefix)
            for prefix in ("MathVista", "MathVerse", "MathVision")
        ):
            prompt = self.build_math_prompt(line, dataset)
        elif DATASET_TYPE(dataset) == "Y/N":
            prompt = self.build_yorn_prompt(line, dataset)
        elif DATASET_TYPE(dataset) == "MCQ":
            prompt = self.build_multi_choice_prompt(line, dataset)
        else:
            raise RuntimeError(f"Invalid dataset type: {DATASET_TYPE(dataset)}")

        message = [dict(type="image", value=s) for s in tgt_path] + [
            dict(type="text", value=prompt)
        ]

        # interleave dataset
        if dataset.startswith("MMMU_"):
            from ... import MMMUDataset

            message = MMMUDataset.split_MMMU(message)

        return message

    def generate_inner(self, message, dataset=None):
        def _extract_answer(text):
            answer_index = text.lower().find("the answer is")
            if answer_index != -1:
                answer_index += len("the answer is")
                answer = text[answer_index:].lstrip(":").strip()
            else:
                answer = text
            return answer

        # DynaMath
        if dataset == "DynaMath" and self.size == "34B":
            message[-1][
                "value"
            ] += "\nProvide a step-by-step solution to the problem, and conclude with 'the answer is' followed by the final solution."

        prompt, images = self.prepare_inputs(message, dataset)
        print(prompt, images)
        response = self.model.generate(
            prompt=prompt,
            images=images,
            stream=False,
            keep_history=False,
            verbose=False,
        )
        if "conclude with 'the answer is' followed by the final solution." in prompt:
            response = _extract_answer(response)
        print(f"Response ==> {response}")
        return response

    def prepare_inputs(self, message, dataset=None):
        # build query
        images = [x["value"] for x in message if x["type"] == "image"]
        texts = [x["value"] for x in message if x["type"] == "text"]
        if DATASET_MODALITY(dataset) == "VIDEO":  # video inputs
            chunks = [self.image_placeholder for x in message if x["type"] != "text"]
            chunks += [
                x["value"].strip()
                for x in message
                if x["type"] == "text" and x["value"] != ""
            ]
            query = "\n".join(chunks)
        elif len(images) == 0:  # text-only inputs
            query = "\n".join(texts)
        elif len(images) == 1 and len(texts) == 1:  # single-image inputs
            query = self.image_placeholder + "\n" + texts[0]
        else:  # interleaved inputs
            chunks = [
                x["value"].strip() if x["type"] == "text" else self.image_placeholder
                for x in message
            ]
            query = "\n".join(chunks)

        # preprocess inputs
        if DATASET_MODALITY(dataset) == "VIDEO":
            max_partition = 1
        elif any(
            dataset.startswith(prefix)
            for prefix in (
                "HallusionBench",
                "TextVQA",
                "ChartQA",
                "OCRBench",
                "InfoVQA",
                "DocVQA",
                "MTVQA",
            )
        ):
            max_partition = 12
        else:
            max_partition = 9

        return query, images
