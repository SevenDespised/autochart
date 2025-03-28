import os

from src.core.component_interface import IProcessor
from src.prompt_optimization.prompt_optimizer import PromptOptimizer
from src.pipe.storage import StageExecutionData
from utils.data_preprocess import mask_chart_types

BASE_DIR = ""
TEMPLATE_PATH = "llm_pipe/templates"
DATA_PATH = "data/visEval_dataset/visEval.json"
DB_DIR = "data/visEval_dataset/databases"
TMP_OUTPUT_DIR = "tmp_output"

class Processor(IProcessor):
    def __init__(self):
        """
        初始化 Processor 类
        """
        self.db_id = None
    
    @property
    def if_store_variable(self) -> bool:
        """是否在流水线中存储该组件的变量"""
        return False
    
    @property
    def if_post_process(self) -> bool:
        """是否启用后处理逻辑"""
        return True
    
    def generate_prompt(self, input_data, data: StageExecutionData):
        """
        处理列选择输出，生成pandas代码生成提示词
        """
        initial_input = data.get_initial_input()
        self.db_id = initial_input["db_id"]
        # 获取输入数据中的查询文本
        query = initial_input["nl_queries"][0]
        masked_query = mask_chart_types(query)
        # 获取表选择阶段的schema信息
        schema_info = data.get_cache("column_selection")["schema_info"]
        # 获取列选择阶段的字典
        cols = {k: v for k, v in input_data.items() if k != "chain_of_thought_reasoning"}
        # 优化提示词
        prompt_optimizer = PromptOptimizer(masked_query, 'en')
        prompt_optimizer.add_template(os.path.join(BASE_DIR, TEMPLATE_PATH, "pandas_generation.tpl"), 
                                    "QUESTION", 
                                    "optimized_prompt",
                                    DATABASE_SCHEMA = schema_info,
                                    #TABLE_PATH = os.path.join(BASE_DIR, DB_DIR, initial_input["db_id"]),
                                    HINT = "NONE HINT",
                                    COLS = cols,
                                    QUESTION = prompt_optimizer.prompt)
        
        return prompt_optimizer.optimized_prompt
    
    def post_process(self, output_data):
        """
        处理pandas代码生成输出，生成pandas代码
        """
        # 执行code_str,接收result中的dataframe
        code_str = output_data["pandas_code"]
        local_dict = {"table_dir": os.path.join(BASE_DIR, DB_DIR, self.db_id)}
        exec(code_str, globals(), local_dict)
        result = local_dict["result"]
        # 暂存中间文件
        result.to_csv(os.path.join(BASE_DIR, TMP_OUTPUT_DIR, "data.csv"), index=None, na_rep='nan')
        return {"df": result}
    
    def store_variable_in_pipeline(self) -> None:
        """
        向流水线暴露需要存储的变量，直接返回None
        """
        return None