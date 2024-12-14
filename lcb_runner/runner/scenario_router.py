from typing import Union, Dict
import json
from functools import partial
from rank_bm25 import BM25Okapi
from collections import defaultdict
from enum import Enum
import datetime
from datetime import datetime

from lcb_runner.utils.scenarios import Scenario
from lcb_runner.lm_styles import LanguageModel
from lcb_runner.evaluation import (
    codegen_metrics,
    test_output_metrics,
    code_execution_metrics,
)

from lcb_runner.prompts import (
    format_prompt_generation,
    format_prompt_test_output,
    format_prompt_execution,
    format_prompt_execution_cot,
    format_prompt_self_repair,
    format_prompt_generation_cot,
    format_prompt_generation_cot_retrieval,
)

from lcb_runner.utils.extraction_utils import (
    extract_code,
    extract_test_output_code,
    extract_execution_code,
)

from lcb_runner.benchmarks import (
    CodeGenerationProblem,
    TestOutputPredictionProblem,
    CodeExecutionProblem,
    load_code_generation_dataset,
    load_code_generation_dataset_not_fast,
    load_test_prediction_dataset,
    load_code_execution_dataset,
)

# BenchMarkType = list[CodeGenerationProblem | TestOutputPredictionProblem]
BenchMarkType = list[
    Union[CodeGenerationProblem, CodeExecutionProblem, TestOutputPredictionProblem]
]

class RetrievalSetting(Enum):
    EPISODIC = 1
    OPTIMAL = 2


def formulate_retrieval_base(
    queries: Dict[str, str], corpus: Dict[str, str], benchmark, model: str, num_docs=1, retrieval_setting=RetrievalSetting.EPISODIC
) -> dict[str, str]:
    '''
    Formulates a retrieval knowledge base. Episodic setting takes in 
    an unaltered eval_results json file, Optimal setting needs a json file
    with output list solutions replaced with the desired item for retrieval.
    '''
    result = dict()
    retrieved = defaultdict(list)

    tokenized_corpus = [doc.split(' ') for doc in corpus.values()]
    bm25 = BM25Okapi(tokenized_corpus)
    
    for problem in benchmark:
        if retrieval_setting == RetrievalSetting.EPISODIC:
            # corpus = [question['output_list'][0] for question in eval_results if question['graded_list'][0] and question['question_id'] != problem.question_id]
            # curr_eval = [item for item in eval_results if item['question_id'] == problem.question_id][0]
            tokenized_query = queries[problem.question_id].split(' ')
            similar_problem_texts = bm25.get_top_n(tokenized_query, list(corpus.values()), n=num_docs+1)

            final_text = ''
            retrieved_problems = []
            for i, text in enumerate(similar_problem_texts):
                # For each text, you get the current retrieved problem (inefficiently)
                retrieved_problems.extend([key for key in corpus if corpus[key] == text])
            assert len(retrieved_problems) == num_docs + 1
            
            for i, question_title in enumerate(retrieved_problems):
                if question_title.lower() == problem.question_title.lower():
                    similar_problem_texts.pop(i)
                    retrieved_problems.pop(i)
                    break
            
            similar_problem_texts = similar_problem_texts[:num_docs]
            retrieved_problems = retrieved_problems[:num_docs]

            retrieved[problem.question_id].extend(retrieved_problems)
            for i, text in enumerate(similar_problem_texts):
                final_text += f'Similar Problem Number {i+1}\n\n {text}'
            result[problem.question_id] = final_text
        
        # elif retrieval_setting == RetrievalSetting.OPTIMAL:
        #     curr_eval = [item for item in eval_results if item['question_id'] == problem.question_id][0]
        #     result[problem.question_id] = curr_eval['output_list'][0]
    timestamp_str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S_%f")
    with open(f'retrieval_matches_{model}_{timestamp_str}.json', 'w') as f:
        json.dump([result, retrieved], f)
    
    return result, retrieved

def build_prompt_benchmark(
    args,
) -> tuple[
    list[CodeExecutionProblem]
    | list[CodeGenerationProblem]
    | list[TestOutputPredictionProblem],
    callable,
]:
    scenario: Scenario = args.scenario

    if scenario == Scenario.codegeneration:
        not_fast: bool = args.not_fast
        if not_fast:
            benchmark = load_code_generation_dataset_not_fast(args.release_version)
        else:
            benchmark = load_code_generation_dataset(args.release_version)
        benchmark = sorted(benchmark, key=lambda x: x.question_id)
        if args.cot_code_generation:
            if args.query_json or args.corpus_json:
                if args.query_json and args.corpus_json:
                    with open(args.query_json, 'r') as f:
                        # Queries is dictionary: problem_id -> query_text
                        queries = json.load(f)
                    with open(args.corpus_json, 'r') as f:
                        corpus = json.load(f)
                    if args.retrieval_setting == 1: # Episodic
                        retrieval_base, id_to_retrieved = formulate_retrieval_base(queries, corpus, benchmark, args.model, num_docs=args.num_docs)
                    elif args.retrieval_setting == 2: # Optimal = directly providing the retrieved text
                        retrieval_base = corpus
                    # retrieval base: question_id -> retrieved_text
                    format_prompt = partial(format_prompt_generation_cot_retrieval, retrieval_base=retrieval_base)
                else:
                    raise RuntimeError("If specifying retrieval query/corpus, need to specify both --query_json and --corpus_json arguments.")
            else:
                format_prompt = format_prompt_generation_cot
        else:
            format_prompt = format_prompt_generation
    elif scenario == Scenario.testoutputprediction:
        benchmark = load_test_prediction_dataset(args.release_version)
        benchmark = sorted(benchmark, key=lambda x: (x.question_id, x.test_id))
        format_prompt = format_prompt_test_output
    elif scenario == Scenario.selfrepair:
        benchmark = load_code_generation_dataset(args.release_version)
        benchmark = sorted(benchmark, key=lambda x: x.question_id)
        format_prompt = format_prompt_self_repair
    elif scenario == Scenario.codeexecution:
        cot_code_execution: bool = args.cot_code_execution
        benchmark = load_code_execution_dataset(args.release_version)
        benchmark = sorted(benchmark, key=lambda x: int(x.id.split("_")[1]))
        if cot_code_execution:
            format_prompt = format_prompt_execution_cot
        else:
            format_prompt = format_prompt_execution
    else:
        raise ValueError(f"Scenario {scenario} not implemented")
    
    if args.selected_problems_path:
        with open(args.selected_problems_path, 'r') as f:
            selected_problems = json.load(f)
        final_benchmark = []
        for problem in benchmark:
            if problem.question_title in selected_problems:
                final_benchmark.append(problem)
    else:
        final_benchmark = benchmark

    return final_benchmark, format_prompt


def combine_results(
    scenario: Scenario,
    results: list[list[str]],
    model: LanguageModel,
    cot_code_execution: bool = False,
):
    if scenario == Scenario.codegeneration:
        combined_results = [
            (
                outputs_list,
                [extract_code(output, model.model_style) for output in outputs_list],
            )
            for outputs_list in results
        ]
    elif scenario == Scenario.testoutputprediction:
        combined_results = [
            (
                outputs_list,
                [
                    extract_test_output_code(output, model.model_style)
                    for output in outputs_list
                ],
            )
            for outputs_list in results
        ]
    elif scenario == Scenario.selfrepair:
        combined_results = [
            (
                [
                    output[0] if type(output) is list else output
                    for output in outputs_list
                ],
                [
                    (
                        extract_code(output[0], model.model_style)
                        if type(output) is list
                        else extract_code(output, model.model_style)
                    )
                    for output in outputs_list
                ],
            )
            for outputs_list in results
        ]
    elif scenario == Scenario.codeexecution:
        combined_results = [
            (
                outputs_list,
                [
                    extract_execution_code(
                        output, model.model_style, cot=cot_code_execution
                    )
                    for output in outputs_list
                ],
            )
            for outputs_list in results
        ]
    else:
        raise ValueError(f"Scenario {scenario} not implemented")

    return combined_results


def sort_and_extract_save_results(scenario: Scenario, save_results: list[dict]):
    if scenario == Scenario.codegeneration:
        save_results = sorted(save_results, key=lambda x: x["question_id"])
        combined_results = [
            (save_result_instance["output_list"], save_result_instance["code_list"])
            for save_result_instance in save_results
        ]

    elif scenario == Scenario.testoutputprediction:
        save_results = sorted(
            save_results, key=lambda x: (x["question_id"], x["test_id"])
        )
        combined_results = [
            (save_result_instance["output_list"], save_result_instance["pred_list"])
            for save_result_instance in save_results
        ]
    elif scenario == Scenario.selfrepair:
        save_results = sorted(save_results, key=lambda x: x["question_id"])
        combined_results = [
            (save_result_instance["output_list"], save_result_instance["code_list"])
            for save_result_instance in save_results
        ]
    elif scenario == Scenario.codeexecution:
        save_results = sorted(save_results, key=lambda x: int(x["id"].split("_")[1]))
        combined_results = [
            (save_result_instance["output_list"], save_result_instance["pred_list"])
            for save_result_instance in save_results
        ]

    else:
        raise ValueError(f"Scenario {scenario} not implemented")

    return save_results, combined_results


def get_metrics(
    scenario: Scenario,
    args,
    benchmark: list[
        CodeGenerationProblem | CodeExecutionProblem | TestOutputPredictionProblem
    ],
    combined_results,
):
    eval_samples = [instance.get_evaluation_sample() for instance in benchmark]
    generations = [extracted for _, extracted in combined_results]

    if scenario == Scenario.codegeneration or scenario == Scenario.selfrepair:
        metrics = codegen_metrics(
            eval_samples,
            generations,
            num_process_evaluate=args.num_process_evaluate,
            timeout=args.timeout,
        )

    elif args.scenario == Scenario.testoutputprediction:
        metrics = test_output_metrics(
            eval_samples,
            generations,
            k_list=[1, 5],
        )

    elif args.scenario == Scenario.codeexecution:
        metrics = code_execution_metrics(
            eval_samples,
            generations,
        )

    else:
        raise ValueError(f"Scenario {scenario} not implemented")

    print(metrics[0]["pass@1"])

    return metrics
