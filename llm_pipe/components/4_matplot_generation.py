import os
import pandas as pd
import matplotlib.pyplot as plt

from ..src.core.component_interface import IProcessor
from ..src.prompt_optimization.prompt_optimizer import PromptOptimizer
from ..src.pipe.storage import StageExecutionData
from ..utils.data_preprocess import mask_chart_types

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
        self.result = None
        
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
        result = input_data["df"]
        self.result = result
        data_string = result.to_csv(index=None, na_rep='nan')

        # 获取输入数据中的查询文本
        query = initial_input["nl_queries"][0]
        masked_query = mask_chart_types(query)

        # 优化提示词
        prompt_optimizer = PromptOptimizer(masked_query, 'en')
        prompt_optimizer.add_template(os.path.join(BASE_DIR, TEMPLATE_PATH, "4_matplot_generation.tpl"), 
                                    "QUESTION", 
                                    "optimized_prompt",
                                    DATAFRAME = data_string,
                                    COLUMNS = result.columns.tolist(),
                                    #CHART_TYPE = chart_types,
                                    #SAVE_PATH = os.path.join(BASE_DIR, TMP_OUTPUT_DIR, "data.csv"),
                                    QUESTION = prompt_optimizer.prompt)
        
        return prompt_optimizer.optimized_prompt
    
    def post_process(self, output_data):
        """
        处理生成代码输出，生成图表
        """
        # 执行生成的代码
        code_str = output_data["code"]
        local_dict = {"result": self.result, "save_dir": os.path.join(BASE_DIR, TMP_OUTPUT_DIR)}
        exec(code_str, globals(), local_dict)
        # 保存plt图片
        local_dict["plt"].savefig(os.path.join(BASE_DIR, TMP_OUTPUT_DIR, "output.png"))
        return {"image_path": os.path.join(BASE_DIR, TMP_OUTPUT_DIR, "output.png"),
                "code": code_str}
    
    def store_variable_in_pipeline(self) -> None:
        """
        向流水线暴露需要存储的变量，直接返回None
        """
        return None
