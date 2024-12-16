import random 
import re
import blobfile as bf
import pandas
from . import common
from .types import Eval, EvalResult, SamplerBase, SingleEvalResult

GRADER_TEMPLATE = """
请根据给定问题、标准答案和模型预测的答案来评估模型的回答是否正确。您的任务是将结果评定为：【正确】、【错误】或【未尝试】。

首先，我们将列出每个评定类别的示例，然后请您对新问题的预测答案进行评定。
以下是【正确】的答复示例：
```
问题：贝拉克·奥巴马的孩子叫什么名字？
标准答案：玛丽亚·奥巴马和萨莎·奥巴马
模型预测1：Malia Obama and Sasha Obama
模型预测2：玛丽亚和萨沙
模型预测3：大多数人会说是玛丽亚和萨莎，但我不确定，需要再确认
模型预测4：巴拉克·奥巴马有两个女儿，她们分别是玛丽亚·安和娜塔莎·玛丽安，但通常称作玛丽亚·奥巴马和萨莎·奥巴马。玛丽亚出生于1998年7月4日，萨莎出生于2001年6月10日。
```
这些答复均为【正确】，因为：
    - 完整地包含了标准答案中的重要信息。
    - 不包含任何与标准答案矛盾的信息。
    - 只关注语义内容，中英文，大小写、标点、语法和顺序不重要。
    - 答复中出现模糊语句或猜测是可以接受的，前提是包含了标准答案且不含有不正确信息或矛盾。

以下是【错误】的答复示例：
```
问题：巴拉克·奥巴马的孩子叫什么名字？
标准答案：玛丽亚·奥巴马和萨莎·奥巴马
模型预测1：玛丽亚
模型预测2：玛丽亚、萨莎和苏珊
模型预测3：巴拉克·奥巴马没有孩子
模型预测4：我认为是玛丽亚和萨莎。或者是玛丽亚和杰基。或者是乔伊和玛丽亚。
模型预测5：虽然我不知道他们的确切名字，但能说出巴拉克·奥巴马有三个孩子。
模型预测6：你可能是想说贝茜和奥利维亚。不过您应通过最新的参考资料确认详细信息。那是正确的答案吗？
```
这些答复均为【错误】，因为：
    - 答复中包含与标准答案矛盾的事实陈述。即使在陈述中略带保留（例如：“可能是”，“虽然我不确定，但我认为”），也视为错误。

以下是【未尝试】的答复示例：
```
问题：巴拉克·奥巴马的孩子叫什么名字？
标准答案：玛丽亚·奥巴马和萨莎·奥巴马
模型预测1：我不知道。
模型预测2：我需要更多关于您所指奥巴马的上下文。
模型预测3：不查阅网络我无法回答这个问题，不过我知道巴拉克·奥巴马有两个孩子。
模型预测4：巴拉克·奥巴马有两个孩子。我知道其中一个叫玛丽亚，但我不确定另一个的名字。
```
这些答复均为【未尝试】，因为：
    - 没有包含标准答案中的重要信息。
    - 回复中没有与标准答案矛盾的陈述。

另外注意以下几点：
- 对于标准答案为数字的问题，预测答案应和标准答案一致。例如，考虑问题“金山铁路黄浦江特大桥的全长是多少米？”，标准答案为“3518.17”：
    - 预测答案“3518”、“3518.1”、“3518.17”均为【正确】。
    - 预测答案“3520”和“3600”均为【错误】。 
    - 预测答案“大约3500米”和“超过3000米”被视为【未尝试】，因为它们既不确认也不与标准答案矛盾。
- 如果标准答案包含比问题更多的信息，预测答案只需包含问题中提到的信息。
    - 例如，考虑问题“菱镁矿的主要化学成分是什么？”标准答案为“碳酸镁（MgCO3）”。“碳酸镁”或“MgCO3”均视为【正确】答案。
- 如果从问题中明显可以推断出预测答案省略的信息，那么算作正确。
    - 例如，问题“巴鲁米尼的努拉吉遗迹在1997年被联合国教科文组织列为世界文化遗产，那么这遗址在哪个地区？”标准答案为“意大利撒丁岛”，预测答案“撒丁岛”被视为【正确】。
- 如果能明显看出名字翻译版本不同但是是同一个人也认为正确。
    - 例如，如果标准答案是“Robinson”，那么回答鲁滨逊或者鲁滨孙均正确。

下面是一个新的问题示例。请只回复A、B、C之一，不要道歉或纠正自己的错误，只需要评估该回答。
```
问题: {question}
正确答案: {target}
预测答案: {predicted_answer}
```

将此新问题的预测答案评定为以下之一：
A:【正确】
B:【错误】
C:【未尝试】

只返回字母"A"、"B"或"C"，无须添加其他文本。
""".strip()

ANSWER_WITH_CONFIDENCE_PROMPT_TEMPLATE = """
请阅读以下问题：
{question}
请基于此问题提供你的最佳答案，并用0到100的分数表示你对该答案的信心（置信度）。请以如下的JSON格式给出回复：
{{
"answer": "你的答案",
"confidence_score": 你的置信度
}}
""".strip()


CHOICE_LETTERS = ["A", "B", "C"]
CHOICE_STRINGS = ["正确", "错误", "未尝试"]
CHOICE_LETTER_TO_STRING = dict(zip(CHOICE_LETTERS, CHOICE_STRINGS))

class ChineseSimpleQAEval(Eval):
    def __init__(self, grader_model: SamplerBase, num_examples: int | None = None, n_repeats: int = 1):
        
        df = pandas.read_csv('data/chinese_simpleqa.csv')
        
        examples = [row.to_dict() for _, row in df.iterrows()]
        if num_examples:
            assert n_repeats == 1, "n_repeats only supported when max_examples = None"
            rng = random.Random(0)
            examples = rng.sample(examples, num_examples)
        self.examples = examples * n_repeats
        self.grader_model = grader_model

    def grade_sample(self, question: str, target: str, predicted_answer: str) -> str:
        grader_prompt = GRADER_TEMPLATE.format(
            question=question,
            target=target,
            predicted_answer=predicted_answer,
        )
        
        prompt_messages = [
            self.grader_model._pack_message(content=grader_prompt, role="user")
        ]
        grading_response = self.grader_model(prompt_messages)
        
        match = re.search(r"(A|B|C)", grading_response)
        return match.group(0) if match else "C"  # Default to "NOT_ATTEMPTED" if no match

    def __call__(self, sampler: SamplerBase) -> EvalResult:
            def fn(row: dict):
                # prompt = ANSWER_WITH_CONFIDENCE_PROMPT_TEMPLATE.format(question=row.get("problem", ""))
                prompt_messages = [
                    sampler._pack_message(content=row.get("problem", ""), role="user")
                    # sampler._pack_message(content=prompt, role="user")
                ]
                response_text = sampler(prompt_messages)
                grade_letter = self.grade_sample(row.get("problem", ""), row.get("answer", ""), response_text)
                
                # Metrics based on grading response
                is_correct = grade_letter == "A"
                is_incorrect = grade_letter == "B"
                is_not_attempted = grade_letter == "C"
                
                score = is_correct

                # Create HTML for each sample result
                html = common.jinja_env.from_string(common.HTML_JINJA).render(
                    prompt_messages=prompt_messages,
                    next_message=dict(content=response_text, role="assistant"),
                    score=score,
                    correct_answer=row["answer"],
                    extracted_answer=response_text,
                )
                convo = prompt_messages + [dict(content=response_text, role="assistant")]
                return SingleEvalResult(html=html, score=score, convo=convo, metrics={
                    "is_correct": is_correct,
                    "is_incorrect": is_incorrect,
                    "is_not_attempted": is_not_attempted
                })

            # Run evaluation and collect results
            results = common.map_with_progress(fn, self.examples)

            # Aggregate metrics
            aggregate_metrics = {
                "is_correct": sum(result.metrics["is_correct"] for result in results) / len(results),
                "is_incorrect": sum(result.metrics["is_incorrect"] for result in results) / len(results),
                "is_not_attempted": sum(result.metrics["is_not_attempted"] for result in results) / len(results),
            }
            aggregate_metrics["is_given_attempted"] = aggregate_metrics["is_correct"] + aggregate_metrics["is_incorrect"]
            # Calculate accuracy_given_attempted
            aggregate_metrics["accuracy_given_attempted"] = (
                aggregate_metrics["is_correct"]
                / aggregate_metrics["is_given_attempted"]
                if aggregate_metrics["is_given_attempted"] > 0
                else 0
            )
            print("AGGREGATE METRICS") 
            print(aggregate_metrics) 
            print("##################")

            output_d = {
                "accuracy_given_attempted": aggregate_metrics["accuracy_given_attempted"],
                "f1": (
                    2 * aggregate_metrics["accuracy_given_attempted"] * aggregate_metrics["is_correct"]
                    / (aggregate_metrics["accuracy_given_attempted"] + aggregate_metrics["is_correct"])
                    if (aggregate_metrics["accuracy_given_attempted"] + aggregate_metrics["is_correct"]) > 0
                    else 0
                )
            }
            
            print(f"Accuracy Given Attempted: {output_d['accuracy_given_attempted']:.3f}")
            print(f"F1 Score: {output_d['f1']:.3f}")
            
            return common.aggregate_results(results)
    

