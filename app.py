#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重症患者肠内营养三级风险分层护理干预管理系统 V1.0
Critical Care Enteral Nutrition Risk Stratification Nursing Intervention Management System

作者: 护理数据科学团队
功能: 三级风险分层、护理干预推荐、护理计划管理、效果追踪
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os

# Get base directory for model files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# 设置页面
st.set_page_config(
    page_title="重症患者肠内营养风险分层系统 V1.0",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义样式
st.markdown("""
<style>
    .main-header {
        font-size: 32px;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 20px;
    }
    .risk-low { 
        background: linear-gradient(135deg, #2ecc71, #27ae60);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .risk-moderate {
        background: linear-gradient(135deg, #f39c12, #e67e22);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .risk-high {
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .intervention-box {
        background-color: #f8f9fa;
        border-left: 4px solid #3498db;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# 护理干预知识库
# ============================================

INTERVENTION_KNOWLEDGE_BASE = {
    "high": {
        "color": "🔴",
        "risk_level": "高风险",
        "monitoring": [
            "每小时监测胃残余量(GRV)",
            "每日评估急性胃肠损伤(AGI)分级",
            "持续监测腹内压(IAP)",
            "每4小时评估腹部症状",
            "每班评估误吸风险"
        ],
        "prevention": [
            "床头抬高30-45°预防误吸",
            "使用促胃肠动力药物(如甲氧氯普胺、红霉素)",
            "实施口腔护理Q4H",
            "监测电解质平衡(钾、镁、磷)",
            "预防应激性溃疡(质子泵抑制剂)"
        ],
        "nutrition": [
            "营养液调整:考虑免疫增强型配方",
            "起始剂量:10-20 mL/h，谨慎递增",
            "蛋白质目标:1.5-2.0 g/kg/d",
            "热量目标:20-25 kcal/kg/d(急性期)",
            "考虑补充谷氨酰胺(0.3-0.5 g/kg/d)"
        ],
        "complications": [
            "监测腹泻发生率及严重程度",
            "预防喂养管堵塞(温水冲洗Q4H)",
            "监测再喂养综合征",
            "评估高血糖风险并调整胰岛素",
            "监测肝功能及血脂代谢"
        ],
        "communication": [
            "立即通知主治医师",
            "多学科团队(MDT)会诊",
            "每班交接重点评估内容",
            "家属沟通营养支持方案"
        ]
    },
    "moderate": {
        "color": "🟡",
        "risk_level": "中风险",
        "monitoring": [
            "每4小时监测胃残余量(GRV)",
            "隔日评估急性胃肠损伤(AGI)分级",
            "每日监测腹内压(IAP)",
            "每班评估腹部症状"
        ],
        "prevention": [
            "床头抬高30-45°",
            "按需使用促胃肠动力药物",
            "实施口腔护理Q6H",
            "定期评估误吸风险"
        ],
        "nutrition": [
            "标准喂养方案:20-30 mL/h起始",
            "蛋白质目标:1.2-1.5 g/kg/d",
            "热量目标:25 kcal/kg/d",
            "根据耐受性逐步增量",
            "评估是否需要高蛋白配方"
        ],
        "complications": [
            "每日监测腹泻情况",
            "预防喂养管堵塞(温水冲洗Q6H)",
            "每周评估营养指标"
        ],
        "communication": [
            "常规医护沟通",
            "记录护理评估结果"
        ]
    },
    "low": {
        "color": "🟢",
        "risk_level": "低风险",
        "monitoring": [
            "每8小时监测胃残余量(GRV)",
            "每3日评估AGI分级",
            "每班常规腹部评估"
        ],
        "prevention": [
            "床头抬高30°",
            "常规口腔护理Q8H",
            "保持喂养管路通畅"
        ],
        "nutrition": [
            "标准喂养方案:30-50 mL/h起始",
            "蛋白质目标:1.0-1.2 g/kg/d",
            "热量目标:25-30 kcal/kg/d",
            "常规增量方案"
        ],
        "complications": [
            "常规观察并发症征象",
            "喂养管护理(冲洗Q8H)"
        ],
        "communication": [
            "常规护理记录"
        ]
    }
}

# ============================================
# 加载模型
# ============================================

@st.cache_resource
def load_model():
    """加载训练好的模型和标准化器"""
    try:
        model = pickle.load(open(os.path.join(BASE_DIR, 'best_model.pkl'), 'rb'))
        scaler = pickle.load(open(os.path.join(BASE_DIR, 'scaler.pkl'), 'rb'))
        return model, scaler
    except:
        return None, None

# ============================================
# 会话状态初始化
# ============================================

def init_session_state():
    """初始化会话状态"""
    if 'patient_history' not in st.session_state:
        st.session_state.patient_history = []
    if 'intervention_records' not in st.session_state:
        st.session_state.intervention_records = []
    if 'current_patient' not in st.session_state:
        st.session_state.current_patient = {}
    if 'current_prediction' not in st.session_state:
        st.session_state.current_prediction = None

def get_risk_key(risk_level):
    """将数字风险等级转换为字符串键"""
    mapping = {0: "low", 1: "moderate", 2: "high"}
    return mapping.get(risk_level, "low")

def get_risk_display(risk_level):
    """获取风险等级显示信息"""
    mapping = {
        0: ("🟢 低风险", "risk-low", "#27ae60"),
        1: ("🟡 中风险", "risk-moderate", "#e67e22"),
        2: ("🔴 高风险", "risk-high", "#c0392b")
    }
    return mapping.get(risk_level, ("未知", "", "#95a5a6"))

# ============================================
# 预测函数
# ============================================

def predict_risk(features, model, scaler):
    """预测风险等级"""
    feature_array = np.array(features).reshape(1, -1)
    feature_scaled = scaler.transform(feature_array)
    prediction = model.predict(feature_array)[0]
    probabilities = model.predict_proba(feature_array)[0]
    return prediction, probabilities

# ============================================
# 主界面
# ============================================

def main():
    init_session_state()
    
    # 侧边栏导航
    st.sidebar.title("📋 导航菜单")
    page = st.sidebar.radio(
        "选择功能模块:",
        ["🏠 系统首页", "📝 患者评估", "⚠️ 风险预测", "📋 护理干预推荐", 
         "📊 风险趋势追踪", "📈 特征解释", "📄 护理报告", "❓ 使用帮助"]
    )
    
    # 加载模型
    model, scaler = load_model()
    
    if page == "🏠 系统首页":
        show_homepage()
    elif page == "📝 患者评估":
        show_patient_assessment()
    elif page == "⚠️ 风险预测":
        show_risk_prediction(model, scaler)
    elif page == "📋 护理干预推荐":
        show_intervention_recommendation()
    elif page == "📊 风险趋势追踪":
        show_trend_tracking()
    elif page == "📈 特征解释":
        show_feature_explanation()
    elif page == "📄 护理报告":
        show_nursing_report()
    elif page == "❓ 使用帮助":
        show_help()

def show_homepage():
    """系统首页"""
    st.markdown('<div class="main-header">🏥 重症患者肠内营养三级风险分层护理干预管理系统 V1.0</div>', 
                unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🤖 智能风险分层</h3>
            <p>基于8种机器学习模型</p>
            <p>Macro-AUC > 0.99</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 精准护理推荐</h3>
            <p>三级风险对应护理方案</p>
            <p>个性化干预措施</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 全程追踪管理</h3>
            <p>多时间点风险评估</p>
            <p>动态趋势分析</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 系统功能概览
    st.subheader("📋 系统功能模块")
    
    features = {
        "📝 患者评估": "录入患者基本信息、临床指标和营养相关参数",
        "⚠️ 风险预测": "基于AI模型自动预测肠内营养风险等级（低/中/高）",
        "📋 护理干预推荐": "根据风险等级自动生成个性化护理措施清单",
        "📊 风险趋势追踪": "记录多次评估结果，展示风险变化趋势",
        "📈 特征解释": "展示特征重要性分析和SHAP解释",
        "📄 护理报告": "生成完整的PDF护理评估报告"
    }
    
    for name, desc in features.items():
        st.markdown(f"**{name}**: {desc}")
    
    # 快速统计
    st.markdown("---")
    st.subheader("📊 系统数据概览")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📁 训练样本", "900例")
    with col2:
        st.metric("🔢 评估特征", "35项")
    with col3:
        st.metric("🤖 ML模型", "8种")
    with col4:
        st.metric("🎯 最优AUC", "0.9989")

def show_patient_assessment():
    """患者评估页面"""
    st.header("📝 患者信息录入")
    
    with st.form("patient_form"):
        st.subheader("基本信息")
        col1, col2, col3 = st.columns(3)
        with col1:
            patient_id = st.text_input("患者ID", value=f"P{datetime.now().strftime('%Y%m%d%H%M')}")
            age = st.number_input("年龄 (岁)", 18, 90, 60)
        with col2:
            gender = st.selectbox("性别", ["男", "女"])
            bmi = st.number_input("BMI (kg/m²)", 15.0, 45.0, 24.0)
        with col3:
            primary_diagnosis = st.selectbox("主要诊断", 
                ["呼吸衰竭", "脓毒症", "多发伤", "神经外科", "心源性休克", "其他"])
            comorbidity_count = st.number_input("合并症数量", 0, 6, 2)
        
        st.subheader("🩺 临床评分")
        col1, col2, col3 = st.columns(3)
        with col1:
            apache_ii = st.number_input("APACHE II评分", 0, 40, 15)
            sofa = st.number_input("SOFA评分", 0, 24, 6)
        with col2:
            gcs = st.number_input("GCS评分", 3, 15, 12)
            agi_grade = st.selectbox("AGI分级", [0, 1, 2, 3, 4], index=1)
        with col3:
            nrs2002 = st.number_input("NRS-2002评分", 0, 7, 3)
            mnutric = st.number_input("mNUTRIC评分", 0.0, 10.0, 4.0)
        
        st.subheader("🧪 实验室指标")
        col1, col2, col3 = st.columns(3)
        with col1:
            albumin = st.number_input("白蛋白 (g/L)", 20.0, 50.0, 32.0)
            prealbumin = st.number_input("前白蛋白 (mg/L)", 100.0, 400.0, 200.0)
        with col2:
            hemoglobin = st.number_input("血红蛋白 (g/L)", 70.0, 170.0, 100.0)
            lymphocyte = st.number_input("淋巴细胞计数 (×10⁹/L)", 0.3, 5.0, 1.2)
        with col3:
            crp = st.number_input("CRP (mg/L)", 0.0, 500.0, 50.0)
            blood_glucose = st.number_input("血糖 (mmol/L)", 3.0, 25.0, 8.0)
        
        st.subheader("🔍 营养与胃肠评估")
        col1, col2, col3 = st.columns(3)
        with col1:
            sga_grade = st.selectbox("SGA分级", [1, 2, 3], index=1)
            pa_albumin_ratio = st.number_input("PA/Alb比值", 10.0, 200.0, 65.0)
        with col2:
            bowel_sounds = st.selectbox("肠鸣音", ["正常", "减弱", "消失"], index=0)
            grv = st.number_input("胃残余量 (mL)", 0.0, 500.0, 50.0)
        with col3:
            iap = st.number_input("腹内压 (mmHg)", 0.0, 25.0, 8.0)
            abdominal_distension = st.selectbox("腹胀程度", ["无", "轻度", "中度", "重度"], index=0)
        
        st.subheader("💊 治疗与护理")
        col1, col2, col3 = st.columns(3)
        with col1:
            mechanical_ventilation = st.checkbox("机械通气", value=False)
            vasoactive_drugs = st.checkbox("血管活性药物", value=False)
        with col2:
            bowel_movement = st.selectbox("排便情况", ["正常", "便秘", "腹泻"], index=0)
            vomiting_reflux = st.checkbox("呕吐/反流", value=False)
        with col3:
            braden_score = st.number_input("Braden评分", 6.0, 23.0, 16.0)
            aspiration_risk = st.checkbox("误吸风险", value=False)
        
        st.subheader("🍼 营养支持")
        col1, col2, col3 = st.columns(3)
        with col1:
            feeding_route = st.selectbox("喂养途径", ["鼻胃管", "鼻空肠管", "PEG"], index=0)
        with col2:
            en_start_time = st.number_input("EN启动时间 (h)", 0.0, 72.0, 12.0)
        with col3:
            formula_type = st.selectbox("配方类型", ["标准配方", "高蛋白", "免疫增强", "纤维增强"], index=0)
        
        submitted = st.form_submit_button("💾 保存患者信息", type="primary")
        
        if submitted:
            # 转换分类变量
            gender_map = {"男": 1, "女": 0}
            diagnosis_map = {"呼吸衰竭": 1, "脓毒症": 2, "多发伤": 3, "神经外科": 4, "心源性休克": 5, "其他": 6}
            bowel_sounds_map = {"正常": 0, "减弱": 1, "消失": 2}
            abd_dist_map = {"无": 0, "轻度": 1, "中度": 2, "重度": 3}
            bowel_move_map = {"正常": 0, "便秘": 1, "腹泻": 2}
            feeding_route_map = {"鼻胃管": 1, "鼻空肠管": 2, "PEG": 3}
            formula_map = {"标准配方": 1, "高蛋白": 2, "免疫增强": 3, "纤维增强": 4}
            
            patient_data = {
                "patient_id": patient_id,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Age": age,
                "Gender": gender_map[gender],
                "BMI": bmi,
                "Primary_Diagnosis": diagnosis_map[primary_diagnosis],
                "Comorbidity_Count": comorbidity_count,
                "PreICU_Hospital_Days": 3,
                "APACHE_II": apache_ii,
                "SOFA": sofa,
                "GCS": gcs,
                "AGI_Grade": agi_grade,
                "NRS2002_Score": nrs2002,
                "mNUTRIC_Score": mnutric,
                "Albumin": albumin,
                "Prealbumin": prealbumin,
                "Hemoglobin": hemoglobin,
                "Lymphocyte_Count": lymphocyte,
                "PA_Albumin_Ratio": pa_albumin_ratio,
                "SGA_Grade": sga_grade,
                "Bowel_Sounds": bowel_sounds_map[bowel_sounds],
                "GRV": grv,
                "IAP": iap,
                "Abdominal_Distension": abd_dist_map[abdominal_distension],
                "Bowel_Movement": bowel_move_map[bowel_movement],
                "Vomiting_Reflux": int(vomiting_reflux),
                "CRP": crp,
                "Blood_Glucose": blood_glucose,
                "Mechanical_Ventilation": int(mechanical_ventilation),
                "Vasoactive_Drugs": int(vasoactive_drugs),
                "Feeding_Route": feeding_route_map[feeding_route],
                "EN_Start_Time": en_start_time,
                "Formula_Type": formula_map[formula_type],
                "Braden_Score": braden_score,
                "Aspiration_Risk": int(aspiration_risk),
                "Catheter_Infection_Risk": 0,
                "Fall_Risk": 1 if braden_score < 12 else 0
            }
            
            st.session_state.current_patient = patient_data
            st.success(f"✅ 患者 {patient_id} 信息已保存! 请前往风险预测页面进行评估。")

def show_risk_prediction(model, scaler):
    """风险预测页面"""
    st.header("⚠️ 肠内营养风险预测")
    
    if not st.session_state.current_patient:
        st.warning("⚠️ 请先前往'患者评估'页面录入患者信息")
        return
    
    patient = st.session_state.current_patient
    
    # 显示患者信息摘要
    with st.expander("📋 查看患者信息摘要", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**患者ID**: {patient['patient_id']}")
            st.write(f"**年龄**: {patient['Age']}岁")
            st.write(f"**性别**: {'男' if patient['Gender']==1 else '女'}")
        with col2:
            st.write(f"**APACHE II**: {patient['APACHE_II']}")
            st.write(f"**SOFA**: {patient['SOFA']}")
            st.write(f"**GCS**: {patient['GCS']}")
        with col3:
            st.write(f"**白蛋白**: {patient['Albumin']} g/L")
            st.write(f"**AGI分级**: {patient['AGI_Grade']}")
            st.write(f"**IAP**: {patient['IAP']} mmHg")
    
    # 预测按钮
    if st.button("🤖 开始风险预测", type="primary"):
        if model is None or scaler is None:
            st.error("❌ 模型加载失败，请检查模型文件是否存在")
            return
        
        # 准备特征
        feature_order = ['Age', 'Gender', 'BMI', 'Primary_Diagnosis', 'Comorbidity_Count',
                        'PreICU_Hospital_Days', 'APACHE_II', 'SOFA', 'GCS', 'AGI_Grade',
                        'NRS2002_Score', 'mNUTRIC_Score', 'Albumin', 'Prealbumin',
                        'Hemoglobin', 'Lymphocyte_Count', 'PA_Albumin_Ratio', 'SGA_Grade',
                        'Bowel_Sounds', 'GRV', 'IAP', 'Abdominal_Distension',
                        'Bowel_Movement', 'Vomiting_Reflux', 'CRP', 'Blood_Glucose',
                        'Mechanical_Ventilation', 'Vasoactive_Drugs', 'Feeding_Route',
                        'EN_Start_Time', 'Formula_Type', 'Braden_Score',
                        'Aspiration_Risk', 'Catheter_Infection_Risk', 'Fall_Risk']
        
        features = [patient[f] for f in feature_order]
        prediction, probabilities = predict_risk(features, model, scaler)
        
        st.session_state.current_prediction = {
            "risk_level": int(prediction),
            "probabilities": probabilities.tolist(),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 显示预测结果
        display_name, css_class, color = get_risk_display(int(prediction))
        
        st.markdown("---")
        st.subheader("🎯 预测结果")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(f'<div class="{css_class}">{display_name}</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("**各类别预测概率:**")
            prob_df = pd.DataFrame({
                "风险等级": ["🟢 低风险", "🟡 中风险", "🔴 高风险"],
                "概率": [f"{p:.1%}" for p in probabilities]
            })
            st.dataframe(prob_df, hide_index=True, use_container_width=True)
        
        # 记录历史
        history_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "risk_level": int(prediction),
            "probabilities": probabilities.tolist(),
            "patient_id": patient["patient_id"]
        }
        st.session_state.patient_history.append(history_entry)
        
        # 自动显示护理干预推荐
        st.markdown("---")
        show_intervention_for_risk(int(prediction))

def show_intervention_for_risk(risk_level):
    """显示特定风险等级的护理干预推荐"""
    risk_key = get_risk_key(risk_level)
    intervention = INTERVENTION_KNOWLEDGE_BASE[risk_key]
    
    st.subheader(f"📋 护理干预推荐方案 - {intervention['color']} {intervention['risk_level']}")
    
    tabs = st.tabs(["🔍 监测要点", "🛡️ 预防措施", "🍼 营养方案", "⚡ 并发症预防", "💬 沟通协作"])
    
    with tabs[0]:
        st.markdown("<div class='intervention-box'>", unsafe_allow_html=True)
        for item in intervention["monitoring"]:
            st.markdown(f"- ✅ {item}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tabs[1]:
        st.markdown("<div class='intervention-box'>", unsafe_allow_html=True)
        for item in intervention["prevention"]:
            st.markdown(f"- ✅ {item}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tabs[2]:
        st.markdown("<div class='intervention-box'>", unsafe_allow_html=True)
        for item in intervention["nutrition"]:
            st.markdown(f"- ✅ {item}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tabs[3]:
        st.markdown("<div class='intervention-box'>", unsafe_allow_html=True)
        for item in intervention["complications"]:
            st.markdown(f"- ✅ {item}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with tabs[4]:
        st.markdown("<div class='intervention-box'>", unsafe_allow_html=True)
        for item in intervention["communication"]:
            st.markdown(f"- ✅ {item}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # 护理措施执行记录
    st.markdown("---")
    st.subheader("📝 护理措施执行记录")
    
    with st.form("intervention_record"):
        col1, col2 = st.columns(2)
        with col1:
            measure = st.text_input("护理措施内容")
            executor = st.text_input("执行人")
        with col2:
            status = st.selectbox("执行状态", ["已完成", "进行中", "待执行", "暂停"])
            notes = st.text_area("备注说明")
        
        if st.form_submit_button("➕ 添加记录"):
            record = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "measure": measure,
                "executor": executor,
                "status": status,
                "notes": notes,
                "patient_id": st.session_state.current_patient.get("patient_id", "")
            }
            st.session_state.intervention_records.append(record)
            st.success("✅ 护理记录已添加!")
    
    # 显示历史记录
    if st.session_state.intervention_records:
        st.subheader("📋 护理记录历史")
        records_df = pd.DataFrame(st.session_state.intervention_records)
        st.dataframe(records_df, use_container_width=True)

def show_intervention_recommendation():
    """护理干预推荐页面"""
    st.header("📋 护理干预知识库")
    
    st.info("💡 本页面展示三个风险等级对应的标准护理干预方案，供临床护理人员参考使用。")
    
    for risk_key in ["low", "moderate", "high"]:
        intervention = INTERVENTION_KNOWLEDGE_BASE[risk_key]
        with st.expander(f"{intervention['color']} {intervention['risk_level']}护理方案", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**🔍 监测要点**")
                for item in intervention["monitoring"]:
                    st.markdown(f"- {item}")
                st.markdown("**🛡️ 预防措施**")
                for item in intervention["prevention"]:
                    st.markdown(f"- {item}")
            with col2:
                st.markdown("**🍼 营养方案**")
                for item in intervention["nutrition"]:
                    st.markdown(f"- {item}")
                st.markdown("**⚡ 并发症预防**")
                for item in intervention["complications"]:
                    st.markdown(f"- {item}")

def show_trend_tracking():
    """风险趋势追踪页面"""
    st.header("📊 风险趋势追踪")
    
    if not st.session_state.patient_history:
        st.warning("⚠️ 暂无评估历史数据。请先在'风险预测'页面进行评估。")
        return
    
    history_df = pd.DataFrame(st.session_state.patient_history)
    
    # 趋势图
    st.subheader("📈 风险等级变化趋势")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = range(len(history_df))
    colors = ['#2ecc71' if r == 0 else '#f39c12' if r == 1 else '#e74c3c' for r in history_df['risk_level']]
    
    ax.plot(x, history_df['risk_level'], 'o-', color='#34495e', linewidth=2, markersize=8)
    ax.scatter(x, history_df['risk_level'], c=colors, s=150, zorder=5, edgecolors='white', linewidth=2)
    
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['低风险', '中风险', '高风险'])
    ax.set_xlabel('评估次数', fontsize=12)
    ax.set_ylabel('风险等级', fontsize=12)
    ax.set_title('患者肠内营养风险等级变化趋势', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.3, 2.3)
    
    for i, (xi, yi) in enumerate(zip(x, history_df['risk_level'])):
        ax.annotate(history_df['timestamp'].iloc[i], (xi, yi), 
                   textcoords="offset points", xytext=(0, 15), ha='center', fontsize=8)
    
    st.pyplot(fig)
    
    # 概率变化
    st.subheader("📊 各类别概率变化")
    probs = np.array(history_df['probabilities'].tolist())
    
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(x, probs[:, 0], 'o-', color='#2ecc71', label='低风险', linewidth=2)
    ax2.plot(x, probs[:, 1], 's-', color='#f39c12', label='中风险', linewidth=2)
    ax2.plot(x, probs[:, 2], '^-', color='#e74c3c', label='高风险', linewidth=2)
    ax2.set_xlabel('评估次数', fontsize=12)
    ax2.set_ylabel('预测概率', fontsize=12)
    ax2.set_title('各风险等级预测概率变化', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 1)
    
    st.pyplot(fig2)
    
    # 历史记录表
    st.subheader("📋 评估历史记录")
    display_df = history_df.copy()
    display_df['risk_level'] = display_df['risk_level'].map({0: '🟢 低风险', 1: '🟡 中风险', 2: '🔴 高风险'})
    st.dataframe(display_df, use_container_width=True)

def show_feature_explanation():
    """特征解释页面"""
    st.header("📈 模型特征解释")
    
    # 特征重要性
    st.subheader("🔍 特征重要性分析 (Random Forest)")
    
    try:
        model = pickle.load(open(os.path.join(BASE_DIR, 'best_model.pkl'), 'rb'))
        feature_names = ['Age', 'Gender', 'BMI', 'Primary_Diagnosis', 'Comorbidity_Count',
                        'PreICU_Hospital_Days', 'APACHE_II', 'SOFA', 'GCS', 'AGI_Grade',
                        'NRS2002_Score', 'mNUTRIC_Score', 'Albumin', 'Prealbumin',
                        'Hemoglobin', 'Lymphocyte_Count', 'PA_Albumin_Ratio', 'SGA_Grade',
                        'Bowel_Sounds', 'GRV', 'IAP', 'Abdominal_Distension',
                        'Bowel_Movement', 'Vomiting_Reflux', 'CRP', 'Blood_Glucose',
                        'Mechanical_Ventilation', 'Vasoactive_Drugs', 'Feeding_Route',
                        'EN_Start_Time', 'Formula_Type', 'Braden_Score',
                        'Aspiration_Risk', 'Catheter_Infection_Risk', 'Fall_Risk']
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        top_features = importance_df.head(15)
        colors = plt.cm.RdYlBu(np.linspace(0.2, 0.8, len(top_features)))
        ax.barh(top_features['Feature'][::-1], top_features['Importance'][::-1], color=colors[::-1])
        ax.set_xlabel('Feature Importance', fontsize=12)
        ax.set_title('Top 15 Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        
        # 显示完整表
        st.subheader("📊 完整特征重要性表")
        st.dataframe(importance_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"模型加载失败: {e}")

def show_nursing_report():
    """护理报告页面"""
    st.header("📄 护理评估报告")
    
    if not st.session_state.current_patient:
        st.warning("⚠️ 请先录入患者信息并进行风险预测")
        return
    
    patient = st.session_state.current_patient
    prediction = st.session_state.current_prediction
    
    # 报告预览
    st.subheader("📋 报告预览")
    
    report_content = f"""
    ============================================================
        重症患者肠内营养风险评估护理报告
    ============================================================
    
    报告生成时间: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}
    
    【基本信息】
    患者ID: {patient.get('patient_id', 'N/A')}
    年龄: {patient.get('Age', 'N/A')} 岁
    性别: {'男' if patient.get('Gender', 0) == 1 else '女'}
    BMI: {patient.get('BMI', 'N/A')} kg/m²
    主要诊断: {patient.get('Primary_Diagnosis', 'N/A')}
    
    【临床评分】
    APACHE II: {patient.get('APACHE_II', 'N/A')}
    SOFA: {patient.get('SOFA', 'N/A')}
    GCS: {patient.get('GCS', 'N/A')}
    AGI分级: {patient.get('AGI_Grade', 'N/A')}
    NRS-2002: {patient.get('NRS2002_Score', 'N/A')}
    mNUTRIC: {patient.get('mNUTRIC_Score', 'N/A')}
    
    【营养指标】
    白蛋白: {patient.get('Albumin', 'N/A')} g/L
    前白蛋白: {patient.get('Prealbumin', 'N/A')} mg/L
    血红蛋白: {patient.get('Hemoglobin', 'N/A')} g/L
    
    【风险评估结果】
    预测风险等级: {'低风险' if prediction and prediction['risk_level']==0 else '中风险' if prediction and prediction['risk_level']==1 else '高风险' if prediction else '未评估'}
    预测概率: {prediction and ' | '.join([f'{class_names[i]}: {p:.1%}' for i, p in enumerate(prediction['probabilities'])]) or 'N/A'}
    
    【护理建议】
    请根据风险等级参照护理干预知识库执行相应护理措施。
    
    ============================================================
    报告人: _____________    审核人: _____________
    ============================================================
    """
    
    st.text_area("报告内容", report_content, height=500)
    
    # 导出按钮
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="📥 下载报告 (TXT)",
            data=report_content,
            file_name=f"nursing_report_{patient.get('patient_id', 'unknown')}_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain"
        )

def show_help():
    """使用帮助页面"""
    st.header("❓ 使用帮助")
    
    help_content = """
    ### 📖 系统使用指南
    
    #### 1. 患者评估
    - 点击左侧菜单"📝 患者评估"
    - 完整录入患者的基本信息、临床评分、实验室指标等
    - 点击"保存患者信息"按钮
    
    #### 2. 风险预测
    - 录入患者信息后，点击"⚠️ 风险预测"
    - 确认患者信息无误后，点击"开始风险预测"
    - 系统将自动显示风险等级和各类别概率
    
    #### 3. 护理干预推荐
    - 预测完成后，系统会自动显示对应的护理干预方案
    - 也可通过左侧菜单"📋 护理干预推荐"查看完整知识库
    - 可以记录护理措施的执行情况
    
    #### 4. 风险趋势追踪
    - 对同一患者进行多次评估后，可查看风险变化趋势
    - 系统将显示风险等级和概率的变化图表
    
    #### 5. 护理报告
    - 在"📄 护理报告"页面可生成完整的护理评估报告
    - 支持导出为文本文件
    
    ### ⚠️ 注意事项
    1. 本系统仅供临床辅助决策参考，不能替代医护人员的临床判断
    2. 风险预测结果应结合患者实际情况综合评估
    3. 护理干预措施应根据医嘱和科室规范执行
    4. 系统使用数据应遵循医疗机构信息安全和患者隐私保护规定
    """
    
    st.markdown(help_content)

# ============================================
# 运行主程序
# ============================================

if __name__ == "__main__":
    main()
