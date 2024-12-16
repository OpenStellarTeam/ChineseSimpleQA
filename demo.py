import json
import time

import pandas as pd  
from . import common

from .chinese_simpleqa_eval import ChineseSimpleQAEval
from .sampler.chat_completion_sampler import (
    OPENAI_SYSTEM_MESSAGE_API,
    OPENAI_SYSTEM_MESSAGE_API_CN,
    OPENAI_SYSTEM_MESSAGE_CHATGPT,
    ChatCompletionSampler,
)
from .sampler.o1_chat_completion_sampler import O1ChatCompletionSampler

from .sampler.claude_sampler import ClaudeCompletionSampler, CLAUDE_SYSTEM_MESSAGE_LMSYS


def main():
    debug = False
    samplers = {
        # chatgpt models:
        "gpt-4o-mini-2024-07-18": ChatCompletionSampler(
            model="gpt-4o-mini-2024-07-18",
            # system_message=OPENAI_SYSTEM_MESSAGE_API, # for english benchmark
            system_message=OPENAI_SYSTEM_MESSAGE_API_CN, # for chinese benchmark(e.g. Chinese SimpleQA)
            max_tokens=2048,
            api_key="",
            base_url=""
        ),
        # ...
       
    }

    # grading_sampler = ChatCompletionSampler(model="gpt-4o")
    grading_sampler = ChatCompletionSampler(
        model="gpt-4o",
        # system_message=OPENAI_SYSTEM_MESSAGE_API, # for english benchmark
        system_message=OPENAI_SYSTEM_MESSAGE_API_CN, # for chinese benchmark(e.g. Chinese SimpleQA)
        api_key="",
        base_url=""
    )

    # ^^^ used for fuzzy matching, just for math

    def get_evals(eval_name):
        # Set num_examples = None to reproduce full evals
        match eval_name:
            case "chinses_simpleqa":
                return ChineseSimpleQAEval(
                    grader_model = grading_sampler, 
                    num_examples=10 if debug else 3000)
            case _:
                raise Exception(f"Unrecoginized eval type: {eval_name}")
            
    evals = {
        eval_name: get_evals(eval_name) for eval_name in ["chinses_simpleqa"]
    }
    print(evals)
    debug_suffix = "_DEBUG" if debug else ""
    print(debug_suffix)
    mergekey2resultpath = {}
    for sampler_name, sampler in samplers.items():
        for eval_name, eval_obj in evals.items():
            result = eval_obj(sampler)
            # ^^^ how to use a sampler
            file_stem = f"{eval_name}_{sampler_name}"
            report_filename = f"/tmp/{file_stem}{debug_suffix}.html"
            print(f"Writing report to {report_filename}")
            with open(report_filename, "w") as fh:
                fh.write(common.make_report(result))
            metrics = result.metrics | {"score": result.score}
            print(metrics)
            result_filename = f"/tmp/{file_stem}{debug_suffix}.json"
            with open(result_filename, "w") as f:
                f.write(json.dumps(metrics, indent=2))
            print(f"Writing results to {result_filename}")
            mergekey2resultpath[f"{file_stem}"] = result_filename
    merge_metrics = []
    for eval_sampler_name, result_filename in mergekey2resultpath.items():
        try:
            result = json.load(open(result_filename, "r+"))
        except Exception as e:
            print(e, result_filename)
            continue
        result = result.get("f1_score", result.get("score", None))
        eval_name = eval_sampler_name[: eval_sampler_name.find("_")]
        sampler_name = eval_sampler_name[eval_sampler_name.find("_") + 1 :]
        merge_metrics.append(
            {"eval_name": eval_name, "sampler_name": sampler_name, "metric": result}
        )
    merge_metrics_df = pd.DataFrame(merge_metrics).pivot(
        index=["sampler_name"], columns="eval_name"
    )
    print("\nAll results: ")
    print(merge_metrics_df.to_markdown())
    return merge_metrics


if __name__ == "__main__":
    main()