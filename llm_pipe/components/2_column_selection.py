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
        self._store_variable = {}

    @property
    def if_store_variable(self) -> bool:
        """是否在流水线中存储该组件的变量"""
        return True
    
    @property
    def if_post_process(self) -> bool:
        """是否启用后处理逻辑"""
        return False
    
    def generate_prompt(self, input_data, data: StageExecutionData):
        """
        处理表选择输出，生成列选择提示词
        """
        initial_input = data.get_initial_input()
        
        # 获取输入数据中的查询文本
        query = initial_input["nl_queries"][0]
        masked_query = mask_chart_types(query)
        # 获取上阶段选择的表的schema信息
        db_id = initial_input["db_id"]
        table_names = input_data["table_names"]
        db_path = os.path.join(BASE_DIR, DB_DIR)
        tables_path = get_db_tables_path(db_path, "db_tables.json", db_id)
        schema_info = get_csv_schema(tables_path, table_names)
        self._store_variable["schema_info"] = schema_info
        # 优化提示词
        prompt_optimizer = PromptOptimizer(masked_query, 'en')
        prompt_optimizer.add_template(os.path.join(BASE_DIR, TEMPLATE_PATH, "2_column_selection.tpl"), 
                                    "QUESTION", 
                                    "optimized_prompt",
                                    DATABASE_SCHEMA = schema_info,
                                    HINT = "NONE HINT",
                                    QUESTION = prompt_optimizer.prompt)
        
        return prompt_optimizer.optimized_prompt
    
    def store_variable_in_pipeline(self):
        """
        向流水线暴露需要存储的变量
        """
        return self._store_variable
    
    def post_process(self, output_data):
        """
        对模型响应进行后处理
        """
        return output_data