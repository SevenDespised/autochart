import os
from prompt_optimization.prompt_optimizer import PromptOptimizer
from utils.data_preprocess import mask_chart_types

BASE_DIR = ""
TEMPLATE_PATH = "llm_pipe/templates"
class Processor:
    def __init__(self):
        """
        初始化 Processor 类，接收系统信息作为参数
        """
        pass
    def generate_prompt(self, input_data, stage_output = None):
        """
        处理初始输入，生成表选择提示词
        """
        # 获取输入数据中的查询文本
        query = input_data["nl_queries"][input_data["query_idx"]]
        masked_query = mask_chart_types(query)
        prompt_optimizer = PromptOptimizer(masked_query, 'en')
        prompt_optimizer.add_template(os.path.join(BASE_DIR, TEMPLATE_PATH, "table_selection.txt"), 
                                    "QUESTION", 
                                    "optimized_prompt",
                                    DATABASE_SCHEMA = input_data["schema_info"],
                                    HINT = "NONE HINT",
                                    QUESTION = masked_query)
        
        return prompt_optimizer.optimized_prompt