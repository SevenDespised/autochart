import os
from prompt_optimization.prompt_optimizer import PromptOptimizer
from utils.data_preprocess import mask_chart_types, get_db_tables_path
from utils.schema_info_generation import get_csv_schema

BASE_DIR = ""
TEMPLATE_PATH = "llm_pipe/templates"
DATA_PATH = "data/visEval_dataset/visEval.json"
DB_DIR = "data/visEval_dataset/databases"

class Processor:
    def __init__(self):
        """
        初始化 Processor 类
        """
        pass
    def generate_prompt(self, input_data, stage_output):
        """
        处理表选择输出，生成列选择提示词
        """
        initial_input = stage_output["initial_input"]
        
        # 获取输入数据中的查询文本
        query = initial_input["nl_queries"][0]
        masked_query = mask_chart_types(query)
        # 获取上阶段选择的表的schema信息
        db_id = initial_input["db_id"]
        table_names = input_data["table_names"]
        db_path = os.path.join(BASE_DIR, DB_DIR)
        tables_path = get_db_tables_path(db_path, "db_tables.json", db_id)
        schema_info = get_csv_schema(tables_path, table_names)
        # 优化提示词
        prompt_optimizer = PromptOptimizer(masked_query, 'en')
        prompt_optimizer.add_template(os.path.join(BASE_DIR, TEMPLATE_PATH, "column_selection.tpl"), 
                                    "QUESTION", 
                                    "optimized_prompt",
                                    DATABASE_SCHEMA = schema_info,
                                    HINT = "NONE HINT",
                                    QUESTION = prompt_optimizer.prompt)
        
        return prompt_optimizer.optimized_prompt