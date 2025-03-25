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
        处理初始输入，生成表选择提示词
        """
        # 获取输入数据中的查询文本
        query = input_data["nl_queries"][0]
        masked_query = mask_chart_types(query)
        # 获取schema信息
        db_id = input_data["db_id"]
        db_path = os.path.join(BASE_DIR, DB_DIR)
        tables_path = get_db_tables_path(db_path, "db_tables.json", db_id)
        schema_info = get_csv_schema(tables_path)
        # 优化提示词
        prompt_optimizer = PromptOptimizer(masked_query, 'en')
        prompt_optimizer.add_template(os.path.join(BASE_DIR, TEMPLATE_PATH, "table_selection.tpl"), 
                                    "QUESTION", 
                                    "optimized_prompt",
                                    DATABASE_SCHEMA = schema_info,
                                    HINT = "NONE HINT",
                                    QUESTION = prompt_optimizer.prompt)
        
        return prompt_optimizer.optimized_prompt