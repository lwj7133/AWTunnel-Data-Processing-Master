import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from scipy import interpolate
from pandas.api.types import is_numeric_dtype
import requests
import re

# 在文件顶部添加必要的导入
import streamlit as st

# 在主要内容之前添加以下代码
st.markdown(
    """
    <style>
    .right-sidebar {
        position: fixed;
        top: 0;
        right: 0;
        width: 250px;
        height: 100%;
        background-color: #f0f2f6;
        padding: 20px;
        overflow-y: auto;
        z-index: 1000;
    }
    .main-content {
        margin-right: 250px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 在侧边栏添加署名
st.sidebar.markdown("""
    <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #f6f8fa, #e9ecef); border-radius: 15px; box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);">
        <h4 style="color: #1a5f7a; margin: 0 0 5px 0; font-weight: bold; font-family: 'Arial', sans-serif;">🛫✨ Airfoil Wind Tunnel Data Processing Master ✨🛫</h4>
        <p style="color: #3498db; font-size: 0.9em; font-style: italic; margin: 0 0 5px 0;">Professional / Efficient / Precise</p>
        <hr style="border: 0; height: 1px; background-image: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(0, 0, 0, 0.75), rgba(0, 0, 0, 0)); margin: 0;">
        <p style="color: #34495e; font-size: 1.1em; margin: 10px 0; font-family: 'Microsoft YaHei', sans-serif;">👨‍💻 Developed By LuWeiJing</p>
        <p style="color: #2c3e50; font-size: 1em; margin: 5px 0;">🚀 Version: 1.0.0 | 📅 September 2024</p>
        <p style="color: #546e7a; font-size: 0.9em; margin: 10px 0 0 0;">
            <span style="margin-right: 5px;">💖 欢迎使用</span>
            <span style="margin-left: 5px;">|</span>
            <a href="https://github.com/your_username/your_repo/issues" target="_blank" style="color: #2196F3; text-decoration: none; font-weight: bold; padding: 3px 6px; border-radius: 4px;">
            <span style="margin-left: 5px;">💬 提意见</span>
            </a>
        </p>
    </div>
    """, unsafe_allow_html=True)

import streamlit as st
import requests

# 在侧边栏中添加聊天助手
st.sidebar.markdown("---")
with st.sidebar.expander("🤖 AI-流体力学专家 ", expanded=False):

    # 初始化聊天历史和上下文
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'chat_context' not in st.session_state:
        st.session_state.chat_context = []
    if 'last_uploaded_image' not in st.session_state:
        st.session_state.last_uploaded_image = None

    # API设置
    if 'api_key' not in st.session_state:
        st.session_state.api_key = "sk-1xOLoJ1NRluWwc5oC5Cc8f32E8D940C791AdEb8b656bD4C6"
    if 'api_base' not in st.session_state:
        st.session_state.api_base = "https://api.tu-zi.com"
    if 'model' not in st.session_state:
        st.session_state.model = "gpt-4o"
    
    api_key = st.text_input("输入API密钥", value="默认", type="password")
    api_base = st.text_input("输入API基础URL", value="默认")
    model = st.text_input("输入模型名称", value="默认")
    
    # 使用实际的默认值，而不是"默认"字符串
    api_key_to_use = st.session_state.api_key if api_key == "默认" else api_key
    api_base_to_use = st.session_state.api_base if api_base == "默认" else api_base
    
    model_to_use = st.session_state.model if model == "默认" else model
    
    if api_key != "默认":
        st.session_state.api_key = api_key
    if api_base != "默认":
        st.session_state.api_base = api_base
    if model != "默认":
        st.session_state.model = model

    def latex_to_streamlit(text):
        """将文本中的LaTeX公式转换为Streamlit支持的格式"""
        # 行内公式
        text = re.sub(r'\\\((.*?)\\\)', r'$\1$', text)
        # 行间公式
        text = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', text)
        return text

    def render_message(message):
        """渲染消息，处理LaTeX公式"""
        # 分割文本和公式
        parts = re.split(r'(\$\$.*?\$\$|\$.*?\$)', message)
        for part in parts:
            if part.startswith('$') and part.endswith('$'):
                # 这是一个公式，使用 st.latex 渲染
                st.latex(part.strip('$'))
            else:
                # 这是普通文本，使用 st.markdown 渲染
                st.markdown(part)

    # 显示聊天历史
    for message in st.session_state.chat_history:
        if isinstance(message, tuple) and message[0] == "image":
            st.image(message[1], caption="上传图片", use_column_width=True)
        elif message.startswith("AI:"):
            st.markdown("AI:")
            render_message(latex_to_streamlit(message[3:]))  # 去掉"AI:"前缀
        else:
            st.text(message)

    # 定义API调用函数
    def simplify_context(context, max_messages=7):
        if len(context) <= max_messages:
            return context
        
        # 保留系统消息（如果存在）
        simplified = [msg for msg in context if msg["role"] == "system"]
        
        # 添加最近的消息，确保用户和助手的消息交替出现
        recent_messages = context[-max_messages:]
        for i, msg in enumerate(recent_messages):
            if i == 0 and msg["role"] == "assistant":
                simplified.append({"role": "user", "content": "继续我们的对话。"})
            simplified.append(msg)
        
        return simplified

    def call_api(context):
        headers = {
            "Authorization": f"Bearer {api_key_to_use}",
            "Content-Type": "application/json"
        }
        
        # 添加专家设定的提示词
        system_message = "你是一位流体力学和飞行器设计方面的专家，请用专业且易懂的方式回答问题，并举一些简单贴合的例子来说明，对提问多鼓励赞赏，可以多使用一些emoji，并严格遵这些规则"
        
        # 简化上下文，现在默认保留8条消息
        simplified_context = simplify_context(context)
        
        # 确保系统消息在最前面
        if simplified_context[0]["role"] != "system":
            simplified_context.insert(0, {"role": "system", "content": system_message})
        
        data = {
            "model": model_to_use,
            "messages": simplified_context,
            "max_tokens": 1000
        }
        
        try:
            url = f"{api_base_to_use}/v1/chat/completions"
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            return f"API请求错误: {str(e)}"
        except ValueError as e:
            return f"JSON解析错误: {str(e)}\n响应内容: {response.text}"
        except KeyError as e:
            return f"响应格式错误: {str(e)}\n响应内容: {response.json()}"

    # 创建输入框
    user_input = st.text_input("在这里输入你的问题:")

    # 添加文件上传器
    uploaded_file = st.file_uploader("上传图片", type=["png", "jpg", "jpeg"])

    # 创建两列布局
    col1, col2 = st.columns(2)

    # 在第一列放置发送按钮
    with col1:
        if st.button("发送", key="send_button"):
            if api_key_to_use and (user_input or uploaded_file):
                # 将用户输入添加到聊天历史和上下文
                if user_input:
                    st.session_state.chat_history.append(f"你: {user_input}")
                    st.session_state.chat_context.append({"role": "user", "content": user_input})
                
                image_url = None
                if uploaded_file:
                    # 将图片转换为base64编码
                    import base64
                    image_base64 = base64.b64encode(uploaded_file.getvalue()).decode('utf-8')
                    image_url = f"data:image/jpeg;base64,{image_base64}"
                    st.session_state.chat_history.append(("image", uploaded_file))
                    st.session_state.chat_context.append({"role": "user", "content": [
                        {"type": "text", "text": user_input if user_input else "请分析这张图片"},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]})
                
                # 调用API
                with st.spinner('AI在思考中...'):
                    # 调用API获取AI响应
                    ai_response = call_api(st.session_state.chat_context)
                
                # 将AI回答添加到聊天历史和上下文
                st.session_state.chat_history.append(f"AI: {ai_response}")
                st.session_state.chat_context.append({"role": "assistant", "content": ai_response})
                
                # 渲染AI的响应
                st.markdown("AI:")
                render_message(latex_to_streamlit(ai_response))
                
                # 清空输入框和上传的文件
                st.session_state.user_input = ""
                st.session_state.last_uploaded_image = None
                
                # 重新加载页面以显示新消息
                st.rerun()
            else:
                st.warning("请输入API密钥和问题或上传图片。")

    # 在第二列放置清空聊天按钮
    with col2:
        if st.button("清空聊天", key="clear_chat_button"):
            st.session_state.chat_history = []
            st.session_state.chat_context = []
            st.session_state.last_uploaded_image = None
            st.rerun()

# 在侧边栏中添加新功能
st.sidebar.markdown("---")

# 使用 expander 创建可折叠的数入部分
with st.sidebar.expander("📈 绘制Cl-α曲线 "):
    # 创建一个空的DataFrame来存储用户输入的数据
    lift_data = pd.DataFrame(columns=['攻角α', '升力系数Cl'])

    # 允许用户输入最多15组数据
    for i in range(15):
        col1, col2 = st.columns(2)
        with col1:
            angle = st.number_input(f"攻角 {i+1}", key=f"angle_{i}", format="%.2f")
        with col2:
            cl = st.number_input(f"升力系数 {i+1}", key=f"cl_{i}", format="%.4f")
        
        # 将非零数据添加到DataFrame中
        if angle != 0.0 or cl != 0.0:
            new_data = pd.DataFrame({'攻角': [angle], '升力系数': [cl]})
            lift_data = pd.concat([lift_data, new_data], ignore_index=True)
 
    # 添加绘制曲线的按钮（保持在expander内部）
    if st.button("绘制升力系数曲线"):
        if not lift_data.empty and is_numeric_dtype(lift_data['攻角']) and is_numeric_dtype(lift_data['升力系数']):
            # 对据进行排序
            lift_data = lift_data.sort_values('攻角')
            
            # 创建图形
            fig, ax = plt.subplots(figsize=(10, 6))
        
            # 绘制原始数据点
            ax.scatter(lift_data['攻角'], lift_data['升力系数'], color='blue', s=30, zorder=5)
        
            # 进行样条插值
            if len(lift_data) > 2:  # 至少需要3个点才能进行三次样条插值
                x_smooth = np.linspace(lift_data['攻角'].min(), lift_data['攻角'].max(), 200)
                spl = interpolate.make_interp_spline(lift_data['攻角'], lift_data['升力系数'], k=3)
                y_smooth = spl(x_smooth)
            
                # 绘制插线
                ax.plot(x_smooth, y_smooth, 'r-', label="插值曲线")
            else:
                # 如果点数不足，则只连接这些点
                ax.plot(lift_data['攻角'], lift_data['升力系数'], 'r-', label="连接线")
        
            # 设置坐标轴
            ax.set_xlabel('攻角 (度)')
            ax.set_ylabel('升力系数')
        
            # 添加水平线表示Cl=0
            ax.axhline(y=0, color='k', linestyle='--', linewidth=0.7)
        
            # 添加图例
            ax.legend()
        
            # 添加网格
            ax.grid(True, linestyle=':', alpha=0.7)
        
            # 设置标题
            ax.set_title('翼型升力系数曲线Cl-α')
        
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
        
            # 在侧边栏中显示图
            st.pyplot(fig)
        
            # 添加下载图片的功能
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            st.download_button(
                label="下载Cl-α曲线图",
                data=buffer,
                file_name="Cl-α曲线图.png",
                mime="image/png"
            )
        else:
            st.warning("请至少输入一组有效的攻角和升力系数数据。")
    
    # 在侧边栏中添加新功能
    st.sidebar.markdown("---")
    
# 使用 expander 创建可折叠的数据输入部分
with st.sidebar.expander("📈 绘制不同α下的Cl-Re曲线"):
    # 创建一个空的DataFrame来存储用户输入的数据
    cl_re_data = pd.DataFrame(columns=['攻角', '雷诺数', '升力系数'])

    # 允许用户输入最多5个攻角
    num_angles = st.number_input("输入攻角数量", min_value=1, max_value=5, value=1, key="num_angles_cl_re")

    for i in range(num_angles):
        st.markdown(f"### 攻角 {i+1}")
        angle = st.number_input(f"攻角值 (度)", key=f"angle_cl_re_{i}", format="%.2f")
        
        # 为每个攻角允许输入最多8组数据
        for j in range(8):
            col1, col2 = st.columns(2)
            with col1:
                re = st.number_input(f"雷数 {j+1}", key=f"re_cl_re_{i}_{j}", format="%.2e")
            with col2:
                cl = st.number_input(f"升力系数 {j+1}", key=f"cl_cl_re_{i}_{j}", format="%.4f")
            
            # 将非零数据添加到DataFrame中
            if re != 0.0 or cl != 0.0:
                new_data = pd.DataFrame({'攻角': [angle], '雷诺数': [re], '升力系数': [cl]})
                cl_re_data = pd.concat([cl_re_data, new_data], ignore_index=True)

    # 添加绘制曲线的按钮
    if st.button("绘制升力系数-雷诺数曲线", key="plot_cl_re_curve"):
        if not cl_re_data.empty:
            # 创建图形
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # 为每个攻角绘制曲线
            for angle in cl_re_data['攻角'].unique():
                angle_data = cl_re_data[cl_re_data['攻角'] == angle]
                
                # 过滤掉无效的数据点
                angle_data = angle_data[(angle_data['雷诺数'] > 0) & (angle_data['升力系数'].notna())]
                
                if not angle_data.empty:
                    # 对数据进行排序
                    angle_data = angle_data.sort_values('雷诺数')
                    
                    # 进行样条插值
                    if len(angle_data) > 2:  # 至少需要3个点才能进行三次样条插值
                        x_smooth = np.logspace(np.log10(angle_data['雷诺数'].min()), 
                                               np.log10(angle_data['雷诺数'].max()), 200)
                        try:
                            spl = interpolate.make_interp_spline(np.log10(angle_data['雷诺数']), angle_data['升力系数'], k=min(3, len(angle_data) - 1))
                            y_smooth = spl(np.log10(x_smooth))
                        
                            # 绘制插值曲线
                            ax.plot(x_smooth, y_smooth, '-', label=f"α = {angle}°")
                        except ValueError:
                            # 如果插值失败，只绘制原始数据点
                            ax.plot(angle_data['雷诺数'], angle_data['升力系数'], '-', label=f"α = {angle}°")
                    else:
                        # 如果点数不足，则只连接这些点
                        ax.plot(angle_data['雷诺数'], angle_data['升力系数'], '-', label=f"α = {angle}°")
                    
                    # 绘制原始数据点
                    ax.scatter(angle_data['雷诺数'], angle_data['升力系数'], s=30, zorder=5)
            
            # 设置X轴为对数坐标
            ax.set_xscale('log')
            
            # 设置坐标轴标签
            ax.set_xlabel('雷诺数 Re')
            ax.set_ylabel('升力系数 Cl')
            
            # 自动调整y轴范围，确保包含所有数据点
            y_min = cl_re_data['升力系数'].min()
            y_max = cl_re_data['升力系数'].max()
            if np.isfinite(y_min) and np.isfinite(y_max):
                y_range = y_max - y_min
                ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
            
            # 添加图例
            ax.legend()
            
            # 添加网格（对数坐标下的网格）
            ax.grid(True, which="both", ls="-", alpha=0.2)
            
            # 设置标题
            ax.set_title('不同攻角下的升力系数-雷诺数曲线')
            
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 在侧边栏中显示图形
            st.pyplot(fig)
            
            # 添加下载图片的功能
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            st.download_button(
                label="下载升力系数-雷诺数曲线图",
                data=buffer,
                file_name="不同α下的Cl-Re曲线图.png",
                mime="image/png",
                key="download_cl_re_curve"
            )
        else:
            st.warning("请至少输入一组有效的攻角、雷诺数和升力系数数据。")

# 萨瑟兰公式计算动力粘度
def sutherland_viscosity(T):
    T0 = 273.15  # 参考温度(K)
    mu0 = 1.716e-5  # 参考粘度(kg/m·s)
    S = 110.4  # 萨瑟兰常数(K)
    return mu0 * (T/T0)**(3/2) * (T0 + S) / (T + S)

# 标题
st.markdown("<h2 style='text-align: center;'>🛫✨翼型风洞实验数据处理大师✨🛫</h2>", unsafe_allow_html=True)

# 初始条件输入
st.subheader("🛠️初始条件")
col1, col2 = st.columns(2)
with col1:
    v_inf = st.number_input("来流速度 (m/s):", min_value=0.0, value=30.0, format="%.2f", help="输入风洞中的设定的风速")
    p_atm = st.number_input("大气压力 (Pa):", min_value=0.0, value=101325.0, format="%.1f")
    temp = st.number_input("环境温度 (K):", min_value=0.0, value=288.15, format="%.2f")
with col2:
    chord = st.number_input("翼型弦长 (m):", min_value=0.0, value=0.2, format="%.3f")
    g = st.number_input("重力加速度 (m/s²):", min_value=0.0, value=9.8, format="%.1f")
    angle_of_attack = st.number_input("攻角（度）:", min_value=-90.0, max_value=90.0, value=0.0)

# 计算空气密度
R = 287  # 空气的气体常数，单位：J/(kg·K)
rho = p_atm / (R * temp)

# 计算动力粘度
mu = sutherland_viscosity(temp)

# 计算运动粘度
nu = mu / rho

# 计算雷诺数
Re = v_inf * chord / nu

# 计算声速
gamma = 1.4  # 空气的比热比
a = np.sqrt(gamma * R * temp)

# 计算马赫数
Mach = v_inf / a
# 显示计算结果
st.subheader("💻计算得到")
col1, col2 = st.columns(2)
with col1:
    st.write(f"空气密度ρ: {rho:.3f} kg/m³")
    st.write(f"动力粘度μ: {mu:.3e} kg/(m·s)")
    st.write(f"运动粘度ν: {nu:.3e} m²/s")
with col2:
    st.write(f"雷诺数Re: {Re:.2e}")
    st.write(f"声速a: {a:.2f} m/s")
    st.write(f"马赫数Ma: {Mach:.4f}")

# 创建一个表格用于输入水位高度差数据
st.subheader("📏水位高度差数据输入")
st.write("请输入水位高度差数据（单位：厘米），共33个数据点（翼型驻点一个，上表面16个，下表面16个:")

# 创建多个 DataFrame 作为输入表格
columns1 = ['前缘点'] + [f'上{i}' for i in range(1, 9)]
columns2 = [f'上{i}' for i in range(9, 17)]
columns3 = [f'下{i}' for i in range(1, 9)]
columns4 = [f'下{i}' for i in range(9, 17)]

input_df1 = pd.DataFrame([[0.0] * len(columns1)], columns=columns1)
input_df2 = pd.DataFrame([[0.0] * len(columns2)], columns=columns2)
input_df3 = pd.DataFrame([[0.0] * len(columns3)], columns=columns3)
input_df4 = pd.DataFrame([[0.0] * len(columns4)], columns=columns4)

# 使用 st.data_editor 创建可编辑的表格，并设置列宽
edited_df1 = st.data_editor(
    input_df1,
    column_config={col: st.column_config.NumberColumn(label=col, width="small") for col in columns1},
    hide_index=True,
    num_rows="fixed"
)

edited_df2 = st.data_editor(
    input_df2,
    column_config={col: st.column_config.NumberColumn(label=col, width="small") for col in columns2},
    hide_index=True,
    num_rows="fixed"
)

edited_df3 = st.data_editor(
    input_df3,
    column_config={col: st.column_config.NumberColumn(label=col, width="small") for col in columns3},
    hide_index=True,
    num_rows="fixed"
)

edited_df4 = st.data_editor(
    input_df4,
    column_config={col: st.column_config.NumberColumn(label=col, width="small") for col in columns4},
    hide_index=True,
    num_rows="fixed"
)

# 定义水的密度
rho_water = 1000  # kg/m³

# 添加x坐标数据（单位：米）
x_coords = [0] + [x/1000 for x in [12.5,24,35.5,47,58.5,70,81.5,93,104.5,116,127.5,139,150.5,162,173.5,185]]

# 计算法向力系数的函数
def calculate_cn(cp_upper, cp_lower, x_coords, chord):
    if len(cp_upper) != len(cp_lower) or len(cp_upper) != len(x_coords) - 1:
        raise ValueError("压力系数和坐标数据长度不匹配")
    
    cn = 0
    # 理驻点到第一个测量点的区间
    delta_xi = (x_coords[1] - x_coords[0]) / chord
    f_0 = 0  # 驻点的压力差为0
    f_1 = cp_lower[0] - cp_upper[0]
    cn += 0.5 * (f_0 + f_1) * delta_xi
    
    # 处理剩余的区间
    for i in range(len(cp_upper) - 1):
        delta_xi = (x_coords[i+2] - x_coords[i+1]) / chord  # 注意这里使用i+2和i+1
        f_i = cp_lower[i] - cp_upper[i]
        f_i_plus_1 = cp_lower[i+1] - cp_upper[i+1]
        cn += 0.5 * (f_i + f_i_plus_1) * delta_xi
    
    return cn

# 定义NACA 0012翼型的导数函数
def naca0012_derivative(x, t=0.12):
    return 5 * t * (0.14845 / np.sqrt(x) - 0.1260 - 0.7032 * x + 0.8529 * x**2 - 0.4060 * x**3)

# 计算翼型表面斜率
def calculate_surface_slopes(x_coords, chord, t=0.12):
    slopes_upper = []
    slopes_lower = []
    for x in x_coords[1:]:  # 跳过x=0的点
        x_normalized = x / chord
        slope = naca0012_derivative(x_normalized, t)
        slopes_upper.append(slope)
        slopes_lower.append(-slope)
    return slopes_upper, slopes_lower

# 计算轴向力系数Ca
def calculate_ca(cp_upper, cp_lower, x_coords, chord):
    if len(cp_upper) != len(cp_lower) or len(cp_upper) != len(x_coords) - 1:
        raise ValueError("压力系数和坐标数据长度不匹配")
    
    slopes_upper, slopes_lower = calculate_surface_slopes(x_coords, chord)
    
    ca = 0
    # 处理驻点到第一个测量点的区间
    delta_xi = (x_coords[1] - x_coords[0]) / chord
    f_0 = 0  # 驻点的献为0
    f_1 = cp_upper[0] * slopes_upper[0] - cp_lower[0] * slopes_lower[0]
    ca += 0.5 * (f_0 + f_1) * delta_xi
    
    # 处理剩余的区间
    for i in range(len(cp_upper) - 1):
        delta_xi = (x_coords[i+2] - x_coords[i+1]) / chord
        f_i = cp_upper[i] * slopes_upper[i] - cp_lower[i] * slopes_lower[i]
        f_i_plus_1 = cp_upper[i+1] * slopes_upper[i+1] - cp_lower[i+1] * slopes_lower[i+1]
        ca += 0.5 * (f_i + f_i_plus_1) * delta_xi
    
    return ca

# 在"开始算"按钮的处理逻辑中添加以下代码
if st.button("⚡开始计算⚡"):
    try:
        # 合并所有输入数据
        all_data = pd.concat([edited_df1, edited_df2, edited_df3, edited_df4], axis=1)
        
        # 获取水位高度差数据并转换为米
        delta_h_list = [float(h) / 100 for h in all_data.values.flatten()]
        
        # 检查数据点数量
        if len(delta_h_list) != 33:
            st.error("请确保输入33个数据点。")
        else:
            # 计算压强
            pressure = [p_atm + rho_water * g * h for h in delta_h_list]
            
            # 计算压力系数
            q_inf = 0.5 * rho * v_inf**2
            cp = [(p - p_atm) / q_inf for p in pressure]
            
            # 提取上表面和下表的压系数
            cp_upper = cp[1:17]  # 上表面压力系数
            cp_lower = cp[17:]   # 下表面压力系数
            
            # 计算法向力系数
            cn = calculate_cn(cp_upper, cp_lower, x_coords, chord)
            
            # 计算轴向力系数
            ca = calculate_ca(cp_upper, cp_lower, x_coords, chord)
            
            # 显示结果
            st.header("📊计算结果")
            
            # 压强结果
            st.subheader("📌翼型表面静压（单位：Pa）：")
            pressure_df = pd.DataFrame([pressure], columns=all_data.columns)
            
            # 使用 st.dataframe 分多行显示压强结果
            st.dataframe(pressure_df.iloc[:, :9], height=80)
            st.dataframe(pressure_df.iloc[:, 9:17], height=80)
            st.dataframe(pressure_df.iloc[:, 17:25], height=80)
            st.dataframe(pressure_df.iloc[:, 25:], height=80)
            
            # 压力系数结果
            st.subheader("📌翼型表面压力系数：")
            cp_df = pd.DataFrame([cp], columns=all_data.columns)
            
            # 使用 st.dataframe 分多行显示压力系数结果
            st.dataframe(cp_df.iloc[:, :9], height=80)
            st.dataframe(cp_df.iloc[:, 9:17], height=80)
            st.dataframe(cp_df.iloc[:, 17:25], height=80)
            st.dataframe(cp_df.iloc[:, 25:], height=80)
             
            # 计算升力系数和阻力系数
            alpha_rad = np.radians(angle_of_attack)  # 将攻角转换为弧度
            cl = cn * np.cos(alpha_rad) - ca * np.sin(alpha_rad)
            cd = cn * np.sin(alpha_rad) + ca * np.cos(alpha_rad)
                   
            # 计算V/V∞
            v_ratio = np.sqrt(1 - np.array(cp[:17]))

            # 修改export_data列表，添加V/V∞数据
            export_data = [
                ["初始条件", "", "", "", "", "", "", "", "", "", "", "", "", "", ""],
                ["来流速度 (m/s)", v_inf, "", "", "", "", "", "", "", "", "", "", "", "", ""],
                ["大气压力 (Pa)", p_atm, "", "", "", "", "", "", "", "", "", "", "", "", ""],
                ["环境温度 (K)", temp, "", "", "", "", "", "", "", "", "", "", "", "", ""],
                ["翼型弦长 (m)", chord, "", "", "", "", "", "", "", "", "", "", "", "", ""],
                ["重力加速度 (m/s²)", g, "", "", "", "", "", "", "", "", "", "", "", "", ""],
                ["攻角 (度)", angle_of_attack, "", "", "", "", "", "", "", "", "", "", "", "", ""],
                ["", "", "", "", "", "", "", "", "", "", "", "", "", "", ""],
                ["测压点", "前缘点"] + [f"上{i}" for i in range(1, 17)] + [f"下{i}" for i in range(1, 17)],
                ["水位高度差 (cm)"] + delta_h_list,
                ["翼型表面静压 (Pa)"] + pressure,
                ["翼型表面压力系数"] + cp,
                ["V/V∞"] + list(v_ratio) + [""] * 16,  # 添加V/V∞数据，只有上表面和前缘点有数据
                ["", "", "", "", "", "", "", "", "", "", "", "", "", "", ""],
                ["计算得到", "", "", "", "", "", "", "", "", "", "", "", "", "", ""],
                ["空气密度ρ (kg/m³)", rho, "", "", "", "", "", "", "", "", "", "", "", "", ""],
                ["动力粘度μ (kg/(m·s))", mu, "", "", "", "", "", "", "", "", "", "", "", "", ""],
                ["运动粘度ν (m²/s)", nu, "", "", "", "", "", "", "", "", "", "", "", "", ""],
                ["雷诺数Re", Re, "", "", "", "", "", "", "", "", "", "", "", "", ""],
                ["声速a (m/s)", a, "", "", "", "", "", "", "", "", "", "", "", "", ""],
                ["马赫数Ma", Mach, "", "", "", "", "", "", "", "", "", "", "", "", ""],
                ["", "", "", "", "", "", "", "", "", "", "", "", "", "", ""],
                ["力系数", "", "", "", "", "", "", "", "", "", "", "", "", "", ""],
                ["法向力系数 Cn", cn, "", "", "", "", "", "", "", "", "", "", "", "", ""],
                ["轴向力系数 Ca", ca, "", "", "", "", "", "", "", "", "", "", "", "", ""],
                ["升力系数 Cl", cl, "", "", "", "", "", "", "", "", "", "", "", "", ""],
                ["阻力系数 Cd", cd, "", "", "", "", "", "", "", "", "", "", "", "", ""]
            ]

            # 创建DataFrame
            export_df = pd.DataFrame(export_data)

            # 创建CSV文件，不包含索引和列名
            csv = export_df.to_csv(index=False, header=False)
            
            # 创建文件名，包含来流速度和攻角信息
            file_name = f"NACA0012风洞实验数据-{v_inf:.1f}风速-{angle_of_attack:.1f}°攻角.csv"

            # 添加下载数据为CSV的按钮
            st.download_button(
                label="📥下载数据为CSV",
                data=csv,
                file_name=file_name,
                mime="text/csv",
            )

            # 绘制Cp-x曲线和压力系数分布矢量图
            st.subheader("📌Cp-x曲线和压力系数分布矢量图")

            # 准备数据
            x_normalized = [0] + [x / chord for x in x_coords] + [1]
            cp_upper = [0] + [cp[0]] + cp[1:17] + [0]  # 包含(0,0)、前缘点和(1,0)
            cp_lower = [0] + cp[17:] + [0]  # 包含(0,0)和(1,0)，但不包括前缘点

            # 确保x和y的长度匹配
            assert len(x_normalized) == len(cp_upper), f"上表面数长度不匹配: x={len(x_normalized)}, y={len(cp_upper)}"
            assert len(x_normalized) - 1 == len(cp_lower), f"下表面数据长度不匹配: x={len(x_normalized)}, y={len(cp_lower)}"

            # 创建图形
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), gridspec_kw={'height_ratios': [2, 3]})

            # 绘制NACA0012翼型轮廓
            def naca0012(x, t=0.12):
                return 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)

            x_airfoil = np.linspace(0, 1, 100)
            y_airfoil = naca0012(x_airfoil)

            ax1.plot(x_airfoil, y_airfoil, 'k-', linewidth=2)
            ax1.plot(x_airfoil, -y_airfoil, 'k-', linewidth=2)

            # 添加对称轴
            ax1.axhline(y=0, color='r', linestyle='--', linewidth=1)

            # 计算并绘制原始数据点
            x_data = [0] + x_normalized[1:-1]  # 包括(0,0)和所有中间点，不包括尾缘点
            y_data_upper = [naca0012(x) for x in x_data]
            y_data_lower = [-y for y in y_data_upper]

            # 绘制压力系数分布矢量
            scale = 0.1  # 调整此值以改变矢量长度
            for i in range(len(x_data)):
                if i == 0:  # 前缘点
                    normal_upper = np.array([0, 1])
                    # 不为前缘点绘制下表面矢量
                else:
                    # 计算翼型表面的切线
                    if i < len(x_data) - 1:
                        dx = x_data[i+1] - x_data[i-1]
                        dy_upper = y_data_upper[i+1] - y_data_upper[i-1]
                        dy_lower = y_data_lower[i+1] - y_data_lower[i-1]
                    else:
                        dx = x_data[i] - x_data[i-1]
                        dy_upper = y_data_upper[i] - y_data_upper[i-1]
                        dy_lower = y_data_lower[i] - y_data_lower[i-1]
                    
                    tangent_upper = np.array([dx, dy_upper])
                    tangent_lower = np.array([dx, dy_lower])
                    
                    # 计算法向量
                    normal_upper = np.array([-tangent_upper[1], tangent_upper[0]])
                    normal_lower = np.array([tangent_lower[1], -tangent_lower[0]])
                    
                    # 归一化法向量
                    normal_upper = normal_upper / np.linalg.norm(normal_upper)
                    normal_lower = normal_lower / np.linalg.norm(normal_lower)
                
                # 上表面
                cp_vector = -normal_upper * cp_upper[i] * scale
                ax1.arrow(x_data[i], y_data_upper[i], cp_vector[0], cp_vector[1], 
                          head_width=0.01, head_length=0.02, fc='b', ec='b')
                
                # 下表面 (不包括前缘点)
                if i > 0:
                    cp_vector = normal_lower * cp_lower[i-1] * scale  # 注意这里使用i-1
                    ax1.arrow(x_data[i], y_data_lower[i], cp_vector[0], cp_vector[1], 
                              head_width=0.01, head_length=0.02, fc='r', ec='r')

            ax1.axis('equal')
            ax1.set_xlim(-0.1, 1.1)
            ax1.set_ylim(-0.3, 0.3)
            ax1.set_xlabel('x/c')
            ax1.set_ylabel('y/c')
            ax1.set_title('NACA 0012 压力系数分布矢量图')
            ax1.grid(True, linestyle=':', alpha=0.7)

            # 分段插值
            def piecewise_interpolation_upper(x, y):
                x_smooth = np.linspace(0, 1, 200)
                
                # 直线连接从(0,0)到前缘点
                y_smooth_1 = np.interp(x_smooth[x_smooth <= x[1]], [x[0], x[1]], [y[0], y[1]])
                
                # 样条插值从前缘点到尾缘
                f_spline = interpolate.make_interp_spline(x[1:], y[1:], k=3)
                y_smooth_2 = f_spline(x_smooth[x_smooth > x[1]])
                
                return np.concatenate((y_smooth_1, y_smooth_2))

            def piecewise_interpolation_lower(x, y):
                x_smooth = np.linspace(0, 1, 200)
                
                # 样条插值从(0,0)到尾缘
                f_spline = interpolate.make_interp_spline(x, y, k=3)
                y_smooth = f_spline(x_smooth)
                
                return y_smooth

            # 应用分段插值
            x_smooth = np.linspace(0, 1, 200)
            cp_upper_smooth = piecewise_interpolation_upper(x_normalized, cp_upper)
            cp_lower_smooth = piecewise_interpolation_lower(x_normalized[1:], cp_lower)  # 注意这里去掉了第一个点

            # 绘制光滑曲线
            ax2.plot(x_smooth, cp_upper_smooth, 'b-', label="上翼面")
            ax2.plot(x_smooth, cp_lower_smooth, 'r-', label="下翼面")

            # 绘制原始数据点
            ax2.scatter(x_normalized[1:-1], cp_upper[1:-1], color='blue', s=30, zorder=5)
            ax2.scatter(x_normalized[2:-1], cp_lower[1:-1], color='red', s=30, zorder=5)

            # 绘制从原点到前缘点的直线
            ax2.plot([x_normalized[0], x_normalized[1]], [cp_upper[0], cp_upper[1]], 'b-', linewidth=2)
            ax2.plot([x_normalized[0], x_normalized[1]], [cp_upper[0], cp_upper[1]], color='purple', linewidth=2)

            # 设置坐标轴
            ax2.set_xlabel('x/c')
            ax2.set_ylabel('Cp')
            ax2.invert_yaxis()  # 反转y轴

            # 设置y轴范围,确保0在中间
            y_max = max(max(cp_upper_smooth), max(cp_lower_smooth))
            y_min = min(min(cp_upper_smooth), min(cp_lower_smooth))
            y_range = max(abs(y_max), abs(y_min))
            ax2.set_ylim(y_range, -y_range)

            # 添加水平线表示Cp=0
            ax2.axhline(y=0, color='k', linestyle='--', linewidth=0.5)

            # 添加图例
            ax2.legend()

            # 添加网格
            ax2.grid(True, linestyle=':', alpha=0.7)

            # 设置标题
            fig.suptitle(f'NACA 0012 翼型和压力系数分布 (α={angle_of_attack}°, Re={Re:.2e}, V∞={v_inf:.2f} m/s)', fontsize=16)

            # 调整子图之间的间距
            plt.tight_layout()

            # 在Streamlit中显示图形
            st.pyplot(fig)

            # 添加下载图片的功能
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            st.download_button(
                label="📥下载Cp-x曲线图",
                data=buffer,
                file_name=f"NACA0012_Cp-x_curve_{v_inf:.1f}ms_{angle_of_attack:.1f}deg.png",
                mime="image/png"
            )

            # 创建DataFrame
            columns = ['前缘'] + [f'上{i}' for i in range(1, 17)]
            v_ratio_df = pd.DataFrame([v_ratio], columns=columns)

            # 显示结果
            st.subheader("📌V/V∞ 计算结果:")
            st.dataframe(v_ratio_df)

            # 准备x坐标数据
            x_coords = [0] + [x/1000 for x in [12.5,24,35.5,47,58.5,70,81.5,93,104.5,116,127.5,139,150.5,162,173.5,185]]
            x_normalized = [x / chord for x in x_coords]

            # 创建图形
            fig, ax = plt.subplots(figsize=(10, 6))

            # 进行样条插值
            x_smooth = np.linspace(0, x_normalized[-1], 200)
            spl = interpolate.make_interp_spline(x_normalized, v_ratio, k=3)
            v_ratio_smooth = spl(x_smooth)

            # 绘制插值曲线
            ax.plot(x_smooth, v_ratio_smooth, 'b-', label="插值曲线")

            # 绘制原始数据点
            ax.scatter(x_normalized, v_ratio, color='blue', s=30, zorder=5, label="原始数据点")
            
            # 绘制从原点到前缘点的直线
            ax.plot([0, x_normalized[0]], [0, v_ratio[0]], 'b-', linewidth=2)

           
            # 设置坐标轴
            ax.set_xlabel('x/c')
            ax.set_ylabel('V/V∞')

            # 设置y轴从0开始，并设置刻度间隔为0.2
            y_max = max(v_ratio) * 1.1  # 给最大值留一些余量
            ax.set_ylim(0, y_max)
            ax.yaxis.set_ticks(np.arange(0, y_max, 0.2))

            # 添加水平线表示V/V∞=1
            ax.axhline(y=1, color='red', linestyle='--', linewidth=0.5)

            # 添加图例
            ax.legend()

            # 添加网格
            ax.grid(True, linestyle=':', alpha=0.7)

            # 设置标题
            ax.set_title(f'NACA 0012 V/V∞ 分布 (α={angle_of_attack}°, Re={Re:.2e}, V∞={v_inf:.2f} m/s)')

            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False

            # 在Streamlit中显示图形
            st.pyplot(fig)

            # 添加下载图片的功能
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            st.download_button(
                label="📥下载V/V∞-x曲线图",
                data=buffer,
                file_name=f"NACA0012_V_Vinf-x_curve_{v_inf:.1f}ms_{angle_of_attack:.1f}deg.png",
                mime="image/png"
            )

            # 在V/V∞-x曲线下方添加力系数显示和CSV下载按钮

            # 显示力系数结果
            st.subheader("📌力系数：")
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"法向力系数 Cn = {cn:.4f}")
                st.write(f"轴向力系数 Ca = {ca:.4f}")

            with col2:
                st.write(f"升力系数 Cl = {cl:.4f}")
                st.write(f"阻力系数 Cd = {cd:.4f}")

            # 创建CSV文件，不包含索引和列名
            csv = export_df.to_csv(index=False, header=False)

            # 创建文件名，包含来流速度和攻角信息
            file_name = f"NACA0012风洞实验数据-{v_inf:.1f}风速-{angle_of_attack:.1f}°攻角.csv"

            # 添加下载数据为CSV的按钮（使用唯一的key）
            st.download_button(
                label="📥下载数据为CSV",
                data=csv,
                file_name=file_name,
                mime="text/csv",
                key="download_csv_button"  # 添加唯一的key
            )

    except ValueError as e:
        st.error(f"计算错误: {str(e)}")
# 在所有主要内容之后关闭div
st.markdown('</div>', unsafe_allow_html=True)