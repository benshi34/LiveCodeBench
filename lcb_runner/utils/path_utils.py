import pathlib
from datetime import datetime
from lcb_runner.lm_styles import LanguageModel, LMStyle
from lcb_runner.utils.scenarios import Scenario


def ensure_dir(path: str, is_file=True):
    if is_file:
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    else:
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    return


def get_cache_path(model_repr:str, args) -> str:
    scenario: Scenario = args.scenario
    n = args.n
    temperature = args.temperature
    path = f"cache/{model_repr}/{scenario}_{n}_{temperature}.json"
    ensure_dir(path)
    return path


def get_output_path(model_repr:str, args) -> str:
    timestamp_str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S_%f")
    scenario: Scenario = args.scenario
    n = args.n
    temperature = args.temperature
    cot_suffix = "_cot" if args.cot_code_execution or args.cot_code_generation else ""
    retrieval_suffix = "_retrieval" if args.query_json or args.corpus_json else ""
    if retrieval_suffix:
        num_docs_suffix = f"_p{args.num_docs}"
    else:
        num_docs_suffix = ""
    path = f"output/{model_repr}/{scenario}_{n}_{temperature}{cot_suffix}{retrieval_suffix}{num_docs_suffix}{args.corpus_name}_{timestamp_str}.json"
    ensure_dir(path)
    return path


def get_eval_all_output_path(model_repr:str, args) -> str:
    scenario: Scenario = args.scenario
    n = args.n
    temperature = args.temperature
    cot_suffix = "_cot" if args.cot_code_execution else ""
    path = f"output/{model_repr}/{scenario}_{n}_{temperature}{cot_suffix}_eval_all.json"
    return path
