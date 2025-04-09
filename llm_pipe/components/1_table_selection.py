import os

from ..src.core.component_interface import IProcessor
from ..src.prompt_optimization.prompt_optimizer import PromptOptimizer
from ..src.pipe.storage import StageExecutionData
from ..utils.data_preprocess import mask_chart_types, get_db_tables_path
from ..utils.schema_info_generation import get_csv_schema

BASE_DIR = ""
TEMPLATE_PATH = "llm_pipe/templates"
DATA_PATH = "data/visEval_dataset/visEval.json"
DB_DIR = "data/visEval_dataset/databases"

class Processor(IProcessor):
    def __init__(self):
        """
        初始化 Processor 类
        """
        self._prompt_optimizer = None
        self._store_variable = {}
        
    @property
    def if_store_variable(self) -> bool:
        """是否在流水线中存储该组件的变量"""
        return True
    
    @property
    def if_post_process(self) -> bool:
        """是否启用后处理逻辑"""
        return False

    def generate_prompt(self, input_data, data: StageExecutionData = None) -> str:
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
        # 获取hint信息
        describe = input_data["describe"]
        describe = describe if describe else "None Describe"
        ir_table = str(input_data["irrelevant_tables"])
        ir_table = ir_table if ir_table else "None Irrelevant Tables"
        hint = f"describe: {describe}\nirrelevant_tables: {ir_table}"
        self._store_variable["hint"] = hint
        # 优化提示词
        self._prompt_optimizer = PromptOptimizer(masked_query, 'en')
        self._prompt_optimizer.add_template(os.path.join(BASE_DIR, TEMPLATE_PATH, "1_table_selection.tpl"), 
                                    "QUESTION", 
                                    "optimized_prompt",
                                    DATABASE_SCHEMA = schema_info,
                                    HINT = hint,
                                    QUESTION = self._prompt_optimizer.prompt)
        
        return self._prompt_optimizer.optimized_prompt

    def post_process(self, output_data) -> dict:
        """
        对模型响应进行后处理，如不启用，直接返回原始数据
        """
        return output_data

    def store_variable_in_pipeline(self):
        """
        向流水线暴露需要存储的变量，直接返回None
        """
        return self._store_variable