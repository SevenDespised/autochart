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
        
    @property
    def if_store_variable(self) -> bool:
        """是否在流水线中存储该组件的变量"""
        return False
    
    @property
    def if_post_process(self) -> bool:
        """是否启用后处理逻辑"""
        return False

    def generate_prompt(self, input_data, data: StageExecutionData = None) -> str:
        """
        处理初始输入，生成表选择提示词
        """
        # 获取输入数据中的数据
        table_data = input_data["x_data"] + input_data["y_data"]
        table_data = [f"data {i}: " + str(item) for i, item in enumerate(table_data)]
        # 列表转换为单个字符串
        data_string = "\n".join(table_data)
        # 获取schema信息
        db_id = input_data["db_id"]
        db_path = os.path.join(BASE_DIR, DB_DIR)
        tables_path = get_db_tables_path(db_path, "db_tables.json", db_id)
        schema_info = get_csv_schema(tables_path)
        # 优化提示词
        self._prompt_optimizer = PromptOptimizer(data_string, 'en')
        self._prompt_optimizer.add_template(os.path.join(BASE_DIR, TEMPLATE_PATH, "E_chart_classification.tpl"), 
                                    "DATA", 
                                    "optimized_prompt",
                                    DATABASE_SCHEMA = schema_info,
                                    DATA = self._prompt_optimizer.prompt)
        
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
        return None