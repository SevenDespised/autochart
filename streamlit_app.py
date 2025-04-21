import streamlit as st
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# 导入项目中的模块
from llm_pipe.src.pipe.pipeline import PipelineProcessor
from llm_pipe.utils.data_preprocess import extract_key_values, safe_json_serialize

# 配置路径
BASE_DIR = ""
CONFIG_DIR = "llm_pipe/config/st_config.json"  # 默认配置文件
DB_DIR = "upload_files"
RESULT_DIR = "tmp_output/results"

# 确保目录存在
os.makedirs(DB_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# 设置页面标题和布局
st.set_page_config(
    page_title="图表生成助手",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化会话状态
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'pipeline' not in st.session_state:
    # 加载配置
    with open(os.path.join(BASE_DIR, CONFIG_DIR), "r") as f:
        config = json.load(f)
    # 初始化pipeline
    st.session_state.pipeline = PipelineProcessor(config)
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = {}
# 初始化数据库相关的会话状态
if 'databases' not in st.session_state:
    st.session_state.databases = [d for d in os.listdir(DB_DIR) if os.path.isdir(os.path.join(DB_DIR, d))]
if 'selected_db' not in st.session_state:
    st.session_state.selected_db = None

# 侧边栏 - 配置区域
with st.sidebar:
    st.title("图表生成助手设置")
    
    # 配置文件选择
    st.subheader("配置文件")
    config_files = [f for f in os.listdir("llm_pipe/config") if f.endswith(".json")]
    selected_config = st.selectbox("选择配置文件", config_files, index=config_files.index(CONFIG_DIR.split("/")[-1]) if CONFIG_DIR.split("/")[-1] in config_files else 0)
    
    if st.button("重新加载配置"):
        config_path = os.path.join("llm_pipe/config", selected_config)
        with open(config_path, "r") as f:
            config = json.load(f)
        st.session_state.pipeline = PipelineProcessor(config)
        st.success(f"已重新加载配置: {selected_config}")
    
    # 数据库管理区域
    st.header("数据库管理")
    
    # 更新数据库列表
    st.session_state.databases = [d for d in os.listdir(DB_DIR) if os.path.isdir(os.path.join(DB_DIR, d))]
    
    # 1. 创建数据库
    with st.expander("1. 创建数据库"):
        new_db_name = st.text_input("数据库名称", key="new_db_input")
        if st.button("创建"):
            if new_db_name:
                db_path = os.path.join(DB_DIR, new_db_name)
                if os.path.exists(db_path):
                    st.error(f"数据库 '{new_db_name}' 已存在!")
                else:
                    os.makedirs(db_path)
                    # 更新数据库列表
                    st.session_state.databases.append(new_db_name)
                    st.success(f"数据库 '{new_db_name}' 创建成功!")
                    # 自动刷新页面
                    st.rerun()
            else:
                st.warning("请输入数据库名称")
    
    # 2. 选择数据库
    with st.expander("2. 选择数据库", expanded=True):
        if st.session_state.databases:
            # 使用下拉菜单选择数据库
            selected_db = st.selectbox(
                "选择数据库",
                ["请选择..."] + st.session_state.databases,
                index=0 if not st.session_state.selected_db else st.session_state.databases.index(st.session_state.selected_db) + 1
            )
            
            if selected_db != "请选择...":
                if selected_db != st.session_state.selected_db:
                    st.session_state.selected_db = selected_db
                    st.success(f"已选择数据库: {selected_db}")
        else:
            st.info("暂无数据库，请先创建数据库")
    
    # 3. 删除数据库
    with st.expander("3. 删除数据库"):
        if st.session_state.databases:
            db_to_delete = st.selectbox(
                "选择要删除的数据库",
                ["请选择..."] + st.session_state.databases,
                key="delete_db_select"
            )
            
            if st.button("删除数据库"):
                if db_to_delete != "请选择...":
                    import shutil
                    db_path = os.path.join(DB_DIR, db_to_delete)
                    # 确认删除
                    try:
                        shutil.rmtree(db_path)
                        # 如果删除的是当前选中的数据库，清空选择
                        if st.session_state.selected_db == db_to_delete:
                            st.session_state.selected_db = None
                        # 从上传文件记录中移除
                        if db_to_delete in st.session_state.uploaded_files:
                            st.session_state.uploaded_files.pop(db_to_delete)
                        # 更新数据库列表
                        st.session_state.databases.remove(db_to_delete)
                        st.success(f"数据库 '{db_to_delete}' 已删除!")
                        # 自动刷新页面
                        st.rerun()
                    except Exception as e:
                        st.error(f"删除失败: {str(e)}")
                else:
                    st.warning("请选择要删除的数据库")
        else:
            st.info("暂无数据库可删除")
    
    # 4. 文件管理（将"上传文件"改为"文件管理"）
    with st.expander("4. 文件管理"):
        if st.session_state.selected_db:
            # 上传文件部分
            st.subheader("上传文件")
            uploaded_file = st.file_uploader(
                f"上传CSV文件到数据库: {st.session_state.selected_db}", 
                type=['csv'], 
                key="db_file_uploader"
            )
            
            if uploaded_file is not None:
                # 保存上传的文件到选定的数据库文件夹
                db_file_path = os.path.join(DB_DIR, st.session_state.selected_db, uploaded_file.name)
                with open(db_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # 将文件添加到会话状态
                if st.session_state.selected_db not in st.session_state.uploaded_files:
                    st.session_state.uploaded_files[st.session_state.selected_db] = {}
                st.session_state.uploaded_files[st.session_state.selected_db][uploaded_file.name] = db_file_path
                # 刷新页面以显示新上传的文件
                st.rerun()
            
            # 文件列表和删除功能
            st.subheader("文件列表")
            
            # 确保数据库文件记录是最新的
            if st.session_state.selected_db not in st.session_state.uploaded_files:
                st.session_state.uploaded_files[st.session_state.selected_db] = {}
            
            # 检查文件夹中的实际文件
            db_path = os.path.join(DB_DIR, st.session_state.selected_db)
            files = [f for f in os.listdir(db_path) if os.path.isfile(os.path.join(db_path, f))]
            
            # 更新会话状态中的文件列表
            for filename in files:
                file_path = os.path.join(db_path, filename)
                if filename not in st.session_state.uploaded_files[st.session_state.selected_db]:
                    st.session_state.uploaded_files[st.session_state.selected_db][filename] = file_path
            
            # 显示文件列表并提供删除功能
            db_files = st.session_state.uploaded_files[st.session_state.selected_db]
            if db_files:
                for filename in list(db_files.keys()):  # 使用list创建副本，因为我们可能会在循环中修改字典
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.text(filename)
                    with col2:
                        # 为每个文件创建一个唯一的键
                        delete_key = f"delete_btn_{st.session_state.selected_db}_{filename}"
                        if st.button("x", key=delete_key):
                            try:
                                # 删除文件系统中的文件
                                file_path = db_files[filename]
                                os.remove(file_path)
                                
                                # 从会话状态中移除
                                st.session_state.uploaded_files[st.session_state.selected_db].pop(filename)
                                
                                st.success(f"文件 '{filename}' 已删除!")
                                # 刷新页面以更新文件列表
                                st.rerun()
                            except Exception as e:
                                st.error(f"删除失败: {str(e)}")
            else:
                st.info("暂无文件")
        else:
            st.warning("请先选择数据库")

# 主界面
st.title("图表生成助手")

# 显示聊天历史
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and "image_path" in message:
            # 显示图表
            st.image(message["image_path"])
            # 显示生成的代码
            if "code" in message:
                with st.expander("查看生成的代码"):
                    st.code(message["code"])
        st.markdown(message["content"])

# 用户输入
prompt = st.chat_input("请输入您的问题...")

if prompt:
    # 添加用户消息到历史记录
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 处理用户输入
    with st.chat_message("assistant"):
        with st.spinner("生成回复中..."):
            try:
                # 检查是否选择了数据库
                if not st.session_state.selected_db:
                    error_msg = "请先选择一个数据库"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                else:
                    # 构建输入数据
                    input_data = {
                        "nl_queries": prompt,
                        "db_name": st.session_state.selected_db  # 使用选定的数据库
                    }
                    
                    # 使用PipelineProcessor处理请求
                    result = st.session_state.pipeline.execute_pipeline(input_data)
                    
                    if result["success"]:
                        # 提取输出内容
                        output_data = result["output_data"]
                        
                        # 检查是否包含图表图像
                        image_path = None
                        code = None
                        
                        if "image_path" in output_data:
                            image_path = output_data["image_path"]
                            if "code" in output_data:
                                code = output_data["code"]
                            # 显示图表
                            st.image(image_path)
                            # 显示代码
                            if code:
                                with st.expander("查看生成的代码"):
                                    st.code(code)
                            
                            response_text = f"已为您生成图表。"
                        else:
                            # 显示基本响应
                            response_text = f"处理完成！"
                            if isinstance(output_data, dict) and "df" in output_data:
                                # 如果有数据框但没有图表，显示数据预览
                                st.dataframe(output_data["df"].head())
                                response_text += f"\n\n数据已准备就绪，共 {len(output_data['df'])} 行。"
                            elif isinstance(output_data, dict) and "columns_data" in output_data:
                                # 显示列数据
                                columns_data = output_data["columns_data"]
                                df = pd.DataFrame(columns_data)
                                st.dataframe(df.head())
                                response_text += f"\n\n数据已准备就绪，共 {len(df)} 行。"
                        
                        # 显示文本响应
                        st.markdown(response_text)
                        
                        # 保存助手回复到历史记录
                        assistant_message = {
                            "role": "assistant", 
                            "content": response_text
                        }
                        if image_path:
                            assistant_message["image_path"] = image_path
                        if code:
                            assistant_message["code"] = code
                        
                        st.session_state.messages.append(assistant_message)
                        
                        # 显示token消耗
                        st.caption(f"Token消耗: {result['tokens']}")
                    else:
                        error_msg = f"处理失败: {result.get('error', '未知错误')}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    
            except Exception as e:
                error_msg = f"出现错误: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# 底部信息
st.markdown("---")
st.caption("图表生成助手 - 基于本地大模型的数据可视化工具")