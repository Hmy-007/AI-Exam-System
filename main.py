# main.py
import streamlit as st
import dashscope
import json
import os
import re
import sqlite3
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time

# ==================== 配置 ====================
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-f5ff6c0e2d7d44d195e1c5e2ad8dec50")
dashscope.api_key = DASHSCOPE_API_KEY

# 创建数据文件夹
for folder in ["data/papers", "data/results"]:
    if not os.path.exists(folder):
        os.makedirs(folder)


# ==================== 数据库操作辅助函数 ====================
def get_db_connection():
    """获取数据库连接，设置超时避免锁定"""
    return sqlite3.connect('data/education.db', timeout=10)


def execute_with_retry(query, params=None, max_retries=3):
    """带重试机制的数据库执行函数"""
    for i in range(max_retries):
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            conn.commit()
            result = cursor.fetchall() if cursor.description else None
            conn.close()
            return result
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e) and i < max_retries - 1:
                time.sleep(0.5)
                continue
            else:
                raise e
        finally:
            try:
                conn.close()
            except:
                pass
    return None


# ==================== 数据库初始化 ====================
def init_database():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            created_at TEXT DEFAULT (datetime('now', 'localtime'))
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS papers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            knowledge_points TEXT,
            difficulty TEXT,
            questions TEXT,
            created_at TEXT DEFAULT (datetime('now', 'localtime'))
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS exam_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            paper_id INTEGER,
            paper_title TEXT,
            score INTEGER,
            max_score INTEGER,
            percentage REAL,
            answers TEXT,
            created_at TEXT DEFAULT (datetime('now', 'localtime'))
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS wrong_questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            paper_title TEXT,
            question_content TEXT,
            student_answer TEXT,
            correct_answer TEXT,
            knowledge_point TEXT,
            created_at TEXT DEFAULT (datetime('now', 'localtime'))
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS learning_advice (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            advice_content TEXT,
            generated_at TEXT DEFAULT (datetime('now', 'localtime'))
        )
    ''')

    conn.commit()
    conn.close()


init_database()


# ==================== 本地算法 ====================
def calculate_text_similarity(text1, text2):
    if not text1 or not text2:
        return 0.0
    try:
        vectorizer = TfidfVectorizer().fit_transform([text1, text2])
        similarity = cosine_similarity(vectorizer[0:1], vectorizer[1:2])[0][0]
        return round(similarity, 4)
    except:
        return 0.0


def local_evaluate_essay(student_answer, reference_answer, max_score=5):
    similarity = calculate_text_similarity(student_answer, reference_answer)
    keywords = re.findall(r'[\u4e00-\u9fa5]{2,}', reference_answer)
    matched = sum(1 for kw in keywords if kw in student_answer)
    keyword_score = matched / len(keywords) if keywords else 0
    combined = similarity * 0.6 + keyword_score * 0.4
    earned = round(combined * max_score, 1)
    earned = min(max(earned, 0), max_score)
    return earned, similarity


# ==================== AI智能批改简答题 ====================
def ai_evaluate_essay(question_content, student_answer, reference_answer, max_score=10):
    prompt = f"""你是一位严格的教师，请批改学生的简答题答案。

【题目】{question_content}
【参考答案】{reference_answer}
【学生答案】{student_answer}
【满分】{max_score}分

请按以下JSON格式输出（只输出JSON，不要有其他内容）：
{{
    "score": 得分（整数或小数，不超过满分），
    "comment": "评语（50字以内，指出优点和不足）",
    "suggestions": "改进建议（50字以内）",
    "key_points_matched": ["学生答对的关键点1", "关键点2"],
    "key_points_missing": ["学生遗漏的关键点1", "关键点2"]
}}
"""
    try:
        response = dashscope.Generation.call(
            model='qwen-turbo',
            messages=[{"role": "user", "content": prompt}],
            result_format='message',
            max_tokens=800,
            temperature=0.3
        )
        if response.status_code == 200:
            content = response.output.choices[0].message.content
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                result = json.loads(json_match.group())
                result['score'] = min(max(result.get('score', 0), 0), max_score)
                return result
            else:
                return {
                    "score": max_score * 0.5,
                    "comment": "批改解析失败",
                    "suggestions": "请重试",
                    "key_points_matched": [],
                    "key_points_missing": []
                }
        else:
            return {
                "score": max_score * 0.5,
                "comment": f"API错误: {response.message}",
                "suggestions": "",
                "key_points_matched": [],
                "key_points_missing": []
            }
    except Exception as e:
        return {
            "score": max_score * 0.5,
            "comment": f"调用失败: {str(e)}",
            "suggestions": "",
            "key_points_matched": [],
            "key_points_missing": []
        }


# ==================== AI学习建议生成 ====================
def generate_learning_advice(student_name, wrong_questions, knowledge_scores):
    if wrong_questions:
        wrong_summary = "\n".join([
            f"- 知识点「{w[4]}」：{w[1][:100]}"
            for w in wrong_questions[:10]
        ])
    else:
        wrong_summary = "暂无错题，表现优秀！"

    if knowledge_scores:
        kp_summary = "\n".join([
            f"- {kp}: {data['earned']}/{data['total']}分 ({data['earned'] / data['total'] * 100:.0f}%)"
            for kp, data in knowledge_scores.items()
        ])
    else:
        kp_summary = "暂无考试记录"

    prompt = f"""你是一位AI学习导师，请根据学生的学习数据生成个性化学习建议。

【学生姓名】{student_name}
【最近错题记录】
{wrong_summary}

【知识点掌握情况】
{kp_summary}

请按以下JSON格式输出个性化学习建议（只输出JSON）：
{{
    "overall_assessment": "整体评估（100字以内，总结学生当前学习状态）",
    "weak_areas": ["薄弱知识点1", "薄弱知识点2", "薄弱知识点3"],
    "strong_areas": ["优势知识点1", "优势知识点2"],
    "action_plan": "具体学习行动计划（150字以内，包含每天的学习安排）",
    "recommended_resources": "推荐的学习资源或练习方向（50字以内）",
    "encouragement": "鼓励的话（30字以内）"
}}
"""
    try:
        response = dashscope.Generation.call(
            model='qwen-turbo',
            messages=[{"role": "user", "content": prompt}],
            result_format='message',
            max_tokens=1200,
            temperature=0.7
        )
        if response.status_code == 200:
            content = response.output.choices[0].message.content
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                advice = json.loads(json_match.group())
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute("SELECT id FROM students WHERE name = ?", (student_name,))
                student_row = cursor.fetchone()
                if student_row:
                    cursor.execute('''
                        INSERT INTO learning_advice (student_id, advice_content)
                        VALUES (?, ?)
                    ''', (student_row[0], json.dumps(advice, ensure_ascii=False)))
                    conn.commit()
                conn.close()
                return advice
            else:
                return {"error": "解析失败", "raw": content}
        else:
            return {"error": f"API错误: {response.message}"}
    except Exception as e:
        return {"error": f"调用失败: {str(e)}"}


# ==================== AI调用（试卷生成）====================
def call_qwen(prompt, max_tokens=1500):
    try:
        response = dashscope.Generation.call(
            model='qwen-turbo',
            messages=[{"role": "user", "content": prompt}],
            result_format='message',
            max_tokens=max_tokens,
            temperature=0.7
        )
        if response.status_code == 200:
            return response.output.choices[0].message.content
        else:
            return f"API错误: {response.message}"
    except Exception as e:
        return f"调用失败: {str(e)}"


def generate_paper(paper_name, knowledge_points, num_questions, difficulty):
    num_essay = max(1, int(num_questions * 0.3))
    num_choice = num_questions - num_essay

    prompt = f"""你是一位经验丰富的教师，请根据以下要求生成一套混合题型的试卷。

【试卷名称】{paper_name}
【知识点】{knowledge_points}
【题目总数】{num_questions}道
  - 选择题：{num_choice}道（每题2分）
  - 简答题：{num_essay}道（每题8分）
【难度等级】{difficulty}

请严格按照以下JSON格式输出，只输出JSON：

{{
    "title": "{paper_name}",
    "knowledge_points": "{knowledge_points}",
    "difficulty": "{difficulty}",
    "total_score": {num_choice * 2 + num_essay * 8},
    "questions": [
        {{
            "id": 1,
            "type": "choice",
            "content": "选择题题目内容",
            "options": ["A. 选项1", "B. 选项2", "C. 选项3", "D. 选项4"],
            "answer": "A",
            "score": 2,
            "knowledge_point": "所属知识点"
        }},
        {{
            "id": 2,
            "type": "essay",
            "content": "简答题题目内容（要求学生详细回答）",
            "answer": "详细的参考答案（150字左右）",
            "score": 8,
            "knowledge_point": "所属知识点",
            "grading_points": ["评分点1", "评分点2", "评分点3"]
        }}
    ]
}}

要求：
1. 选择题和简答题混合出现，先出选择题再出简答题
2. 简答题的答案要详细，包含评分要点
3. 题目内容与【知识点】密切相关
4. 难度要符合【难度等级】要求
"""
    response = call_qwen(prompt, max_tokens=4000)

    try:
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            paper = json.loads(json_match.group())
            return paper
        else:
            return {"error": "解析失败", "raw_response": response}
    except Exception as e:
        return {"error": f"JSON解析失败: {e}", "raw_response": response}


def extract_choice_letter(text):
    if not text:
        return ""
    text = str(text).strip().upper()
    # 跳过空选项
    if text == "(请选择)" or text == "":
        return ""
    match = re.match(r'^([A-D])[\.\s]?', text)
    if match:
        return match.group(1)
    # 如果直接是字母
    if text in ['A', 'B', 'C', 'D']:
        return text
    return text


# ==================== 答题评分 ====================
def evaluate_answers(paper, answers, student_name="匿名", use_ai_essay=True):
    questions = paper.get("questions", [])
    total_score = 0
    max_score = sum(q.get("score", 1) for q in questions)
    results = []
    paper_title = paper.get("title", "试卷")

    conn = get_db_connection()

    try:
        cursor = conn.cursor()

        cursor.execute("SELECT id FROM students WHERE name = ?", (student_name,))
        student_row = cursor.fetchone()
        if student_row:
            student_id = student_row[0]
        else:
            cursor.execute("INSERT INTO students (name) VALUES (?)", (student_name,))
            student_id = cursor.lastrowid
            conn.commit()

        for i, q in enumerate(questions):
            student_answer = answers[i] if i < len(answers) else ""
            correct = q.get("answer", "")
            score = q.get("score", 1)
            q_type = q.get("type", "choice")
            knowledge_point = q.get("knowledge_point", "未分类")
            question_content = q.get("content", "")

            if q_type == "choice":
                student_choice = extract_choice_letter(student_answer)
                correct_choice = extract_choice_letter(correct)
                # 如果学生没有选择（空选项），判断为错误
                if student_choice == "":
                    is_correct = False
                    earned = 0
                    feedback = f"✗ 未作答，正确答案: {correct}"
                    similarity = 0.0
                else:
                    is_correct = (student_choice == correct_choice)
                    earned = score if is_correct else 0
                    feedback = "✓ 正确" if is_correct else f"✗ 错误，正确答案: {correct}"
                    similarity = 1.0 if is_correct else 0.0
                ai_comment = ""
            else:
                if use_ai_essay and len(student_answer.strip()) > 5:
                    ai_result = ai_evaluate_essay(question_content, student_answer, correct, score)
                    earned = ai_result.get('score', 0)
                    is_correct = (earned >= score * 0.6)
                    feedback = f"得分: {earned}/{score}\n评语: {ai_result.get('comment', '')}"
                    similarity = earned / score if score > 0 else 0
                    ai_comment = json.dumps(ai_result, ensure_ascii=False)
                else:
                    earned, similarity = local_evaluate_essay(student_answer, correct, score)
                    is_correct = (earned == score)
                    feedback = f"得分: {earned}/{score} (相似度: {similarity:.2f})"
                    ai_comment = ""

            total_score += earned
            results.append({
                "id": i + 1,
                "content": question_content,
                "student_answer": student_answer,
                "correct_answer": correct,
                "earned": earned,
                "max_score": score,
                "knowledge_point": knowledge_point,
                "is_correct": is_correct,
                "similarity": similarity,
                "feedback": feedback,
                "ai_comment": ai_comment,
                "type": q_type
            })

            if not is_correct and student_answer != "" and student_answer != "(请选择)":
                execute_with_retry('''
                    INSERT INTO wrong_questions (student_id, paper_title, question_content, student_answer, correct_answer, knowledge_point)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (student_id, paper_title, question_content, student_answer, correct, knowledge_point))

        percentage = round(total_score / max_score * 100, 1) if max_score > 0 else 0
        cursor.execute('''
            INSERT INTO exam_records (student_id, paper_title, score, max_score, percentage, answers)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (student_id, paper_title, total_score, max_score, percentage, json.dumps(results, ensure_ascii=False)))

        conn.commit()

    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

    return {
        "total_score": total_score,
        "max_score": max_score,
        "percentage": percentage,
        "results": results,
        "wrong_count": sum(1 for r in results if not r["is_correct"])
    }


def generate_knowledge_radar(results):
    knowledge_scores = {}
    for r in results:
        kp = r.get("knowledge_point", "未分类")
        if kp not in knowledge_scores:
            knowledge_scores[kp] = {"total": 0, "earned": 0}
        knowledge_scores[kp]["total"] += r.get("max_score", 1)
        knowledge_scores[kp]["earned"] += r.get("earned", 0)

    if not knowledge_scores:
        return None, knowledge_scores

    categories = list(knowledge_scores.keys())
    scores = [round(knowledge_scores[k]["earned"] / knowledge_scores[k]["total"] * 100, 1) for k in categories]

    fig = go.Figure(data=go.Scatterpolar(
        r=scores,
        theta=categories,
        fill='toself',
        marker=dict(color='#667eea', size=8),
        line=dict(color='#667eea', width=2)
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        title="📊 知识点掌握情况雷达图",
        height=450
    )
    return fig, knowledge_scores


def get_performance_trend(student_name):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT er.created_at, er.percentage, er.paper_title
        FROM exam_records er
        JOIN students s ON er.student_id = s.id
        WHERE s.name = ?
        ORDER BY er.id
    ''', (student_name,))

    records = cursor.fetchall()
    conn.close()

    if not records:
        return None

    dates = []
    scores = []
    titles = []
    for r in records:
        date_str = r[0][5:10] if len(r[0]) >= 10 else r[0][:10]
        dates.append(date_str)
        scores.append(r[1])
        titles.append(r[2][:15] + "..." if len(r[2]) > 15 else r[2])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=scores,
        mode='lines+markers+text',
        line=dict(color='#667eea', width=2),
        marker=dict(size=10, color='#764ba2'),
        text=titles,
        textposition='top center',
        name='得分率'
    ))
    fig.update_layout(
        title="📈 历次考试成绩趋势",
        xaxis_title="考试日期 (月-日)",
        yaxis_title="得分率 (%)",
        yaxis_range=[0, 100],
        height=450,
        hovermode='closest'
    )
    return fig


def list_papers():
    if not os.path.exists("data/papers"):
        return []
    return [f for f in os.listdir("data/papers") if f.endswith(".json")]


def load_paper(filename):
    with open(f"data/papers/{filename}", "r", encoding="utf-8") as f:
        return json.load(f)


def save_paper(filename, paper):
    with open(f"data/papers/{filename}", "w", encoding="utf-8") as f:
        json.dump(paper, f, ensure_ascii=False, indent=2)


def rename_paper(old_filename, new_name):
    old_path = f"data/papers/{old_filename}"
    timestamp = old_filename.split('_')[-1].replace('.json', '')
    safe_name = re.sub(r'[\\/*?:"<>|]', '_', new_name)
    new_filename = f"{safe_name}_{timestamp}.json"
    new_path = f"data/papers/{new_filename}"

    os.rename(old_path, new_path)

    paper = load_paper(new_filename)
    paper["title"] = new_name
    save_paper(new_filename, paper)

    return new_filename


# ==================== Streamlit 界面 ====================
st.set_page_config(page_title="智能试卷生成与评估系统", page_icon="📚", layout="wide")

st.markdown("""
<style>
    .stButton > button { background-color: #4CAF50; color: white; font-size: 16px; border-radius: 8px; }
    .advice-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin: 10px 0;
    }
    .weak-area { background-color: #ffebee; padding: 10px; border-radius: 10px; margin: 5px 0; }
    .strong-area { background-color: #e8f5e9; padding: 10px; border-radius: 10px; margin: 5px 0; }
    .paper-card {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        background-color: #fafafa;
    }
    .question-choice { border-left: 4px solid #4CAF50; padding-left: 10px; margin: 10px 0; }
    .question-essay { border-left: 4px solid #2196F3; padding-left: 10px; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

st.sidebar.title("📚 智能试卷系统")
student_name = st.sidebar.text_input("👤 学生姓名", value="张三")
st.sidebar.markdown("---")

use_ai_essay = st.sidebar.checkbox("🤖 启用AI智能批改简答题", value=True)
st.sidebar.caption("启用后使用大模型批改简答题，评分更智能但稍慢")

role = st.sidebar.radio("👥 身份", ["👨‍🏫 教师端", "👨‍🎓 学生端"])

if role == "👨‍🏫 教师端":
    mode = st.sidebar.radio("功能", ["📝 生成试卷", "📂 试卷管理"])
else:
    mode = st.sidebar.radio("功能", ["✏️ 在线答题", "📊 成绩报告", "📚 错题本", "🤖 AI学习建议"])

st.sidebar.caption("v2.0 | 混合题型 | AI批改 | AI学习建议")

if 'current_paper' not in st.session_state:
    st.session_state.current_paper = None
if 'last_result' not in st.session_state:
    st.session_state.last_result = None


# ==================== 教师端：生成试卷 ====================
if mode == "📝 生成试卷":
    st.title("📝 智能试卷生成")

    col1, col2 = st.columns(2)
    with col1:
        paper_name = st.text_input("📌 试卷名称", value=f"Python测试卷_{datetime.now().strftime('%m%d')}")
        knowledge = st.text_area("📖 知识点（用逗号分隔）", "Python基础, 列表, 字典, 函数", height=80)
        num_q = st.slider("📊 题目总数", 3, 10, 5)

        essay_ratio = st.slider("📝 简答题比例", 0.0, 1.0, 0.3, 0.1,
                                help="简答题数量占总题数的比例，0表示全选择题，1表示全简答题")
        num_essay = int(num_q * essay_ratio)
        num_choice = num_q - num_essay
        st.caption(f"📋 将生成：选择题 {num_choice} 道 + 简答题 {num_essay} 道")

    with col2:
        difficulty = st.select_slider("⭐ 难度", ["简单", "中等", "困难"])

        with st.expander("📖 题型说明"):
            st.markdown("""
            **选择题** (每题2分)
            - 从A/B/C/D中选择正确答案
            - 自动评分

            **简答题** (每题8分)
            - 需要学生文字作答
            - AI智能批改评分
            - 提供详细评语和建议
            """)

        st.info(f"将生成「{paper_name}」\n{num_q}道{difficulty}题")

    if st.button("🚀 生成试卷", type="primary"):
        if not paper_name.strip():
            st.warning("请输入试卷名称")
        elif not knowledge.strip():
            st.warning("请输入知识点")
        else:
            with st.spinner("AI生成中..."):
                paper = generate_paper(paper_name, knowledge, num_q, difficulty)
                if "error" in paper:
                    st.error(f"失败: {paper.get('error')}")
                    if "raw_response" in paper:
                        with st.expander("查看原始响应"):
                            st.code(paper["raw_response"])
                else:
                    questions = paper.get("questions", [])
                    choice_count = sum(1 for q in questions if q.get("type") == "choice")
                    essay_count = sum(1 for q in questions if q.get("type") == "essay")

                    safe_name = re.sub(r'[\\/*?:"<>|]', '_', paper_name)
                    filename = f"data/papers/{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    with open(filename, "w", encoding="utf-8") as f:
                        json.dump(paper, f, ensure_ascii=False, indent=2)

                    st.success(f"✅ 试卷「{paper_name}」生成成功！")
                    st.info(f"📊 题型统计：选择题 {choice_count} 道，简答题 {essay_count} 道，总分 {paper.get('total_score', '计算中')} 分")
                    st.session_state.current_paper = paper
                    st.balloons()

    if st.session_state.current_paper:
        paper = st.session_state.current_paper
        st.subheader(f"📄 {paper.get('title', '试卷预览')}")

        for i, q in enumerate(paper.get("questions", []), 1):
            q_type = q.get("type", "choice")
            type_icon = "🔘" if q_type == "choice" else "📝"
            type_text = "选择题" if q_type == "choice" else "简答题"

            with st.expander(f"{type_icon} 第{i}题 [{type_text}] {q.get('content', '')[:60]}..."):
                st.markdown(f"**题目：** {q.get('content')}")
                if q_type == "choice":
                    st.markdown(f"**选项：** {', '.join(q.get('options', []))}")
                st.markdown(f"**答案：** {q.get('answer')}")
                st.markdown(f"**分值：** {q.get('score', 1)}分")
                st.markdown(f"**知识点：** {q.get('knowledge_point', '未标注')}")
                if q_type == "essay" and q.get("grading_points"):
                    st.markdown(f"**评分要点：** {', '.join(q.get('grading_points', []))}")


# ==================== 教师端：试卷管理 ====================
elif mode == "📂 试卷管理":
    st.title("📂 试卷管理")

    papers = list_papers()

    if not papers:
        st.info("📭 暂无试卷，请先在「生成试卷」中创建试卷")
    else:
        search_term = st.text_input("🔍 搜索试卷", placeholder="输入试卷名称关键词...")

        filtered_papers = papers
        if search_term:
            filtered_papers = [p for p in papers if search_term.lower() in p.lower()]

        st.caption(f"📊 共 {len(filtered_papers)} 份试卷")

        for idx, p in enumerate(filtered_papers):
            display_name = p.replace('.json', '')

            with st.container():
                st.markdown(f'<div class="paper-card">', unsafe_allow_html=True)

                col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 1])

                with col1:
                    st.markdown(f"**📄 {display_name}**")

                with col2:
                    if st.button("👁️ 查看", key=f"view_{idx}"):
                        st.session_state[f"viewing_{p}"] = not st.session_state.get(f"viewing_{p}", False)

                with col3:
                    if st.button("✏️ 重命名", key=f"rename_btn_{idx}"):
                        st.session_state[f"renaming_{p}"] = not st.session_state.get(f"renaming_{p}", False)
                        st.session_state[f"viewing_{p}"] = False

                with col4:
                    if st.button("📋 复制", key=f"copy_{idx}"):
                        paper = load_paper(p)
                        new_name = f"{display_name}_副本"
                        safe_name = re.sub(r'[\\/*?:"<>|]', '_', new_name)
                        new_filename = f"{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                        save_paper(new_filename, paper)
                        st.success(f"已复制为「{new_name}」")
                        st.rerun()

                with col5:
                    if st.button("🗑️ 删除", key=f"del_{idx}"):
                        os.remove(f"data/papers/{p}")
                        st.success(f"已删除 {display_name}")
                        st.session_state.pop(f"viewing_{p}", None)
                        st.session_state.pop(f"renaming_{p}", None)
                        st.rerun()

                if st.session_state.get(f"renaming_{p}", False):
                    st.markdown("---")
                    new_name = st.text_input("请输入新名称", value=display_name.split('_')[0], key=f"new_name_{idx}")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("✅ 确认重命名", key=f"confirm_rename_{idx}"):
                            if new_name.strip():
                                try:
                                    new_filename = rename_paper(p, new_name.strip())
                                    st.success(f"✅ 已重命名为「{new_name}」")
                                    st.session_state[f"renaming_{p}"] = False
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"重命名失败: {e}")
                            else:
                                st.warning("请输入有效名称")
                    with col2:
                        if st.button("❌ 取消", key=f"cancel_rename_{idx}"):
                            st.session_state[f"renaming_{p}"] = False
                            st.rerun()
                    st.markdown("---")

                if st.session_state.get(f"viewing_{p}", False):
                    paper = load_paper(p)

                    with st.expander(f"📋 {paper.get('title', display_name)} - 试卷详情", expanded=True):
                        questions = paper.get("questions", [])
                        choice_count = sum(1 for q in questions if q.get("type") == "choice")
                        essay_count = sum(1 for q in questions if q.get("type") == "essay")
                        total_score = paper.get("total_score", sum(q.get("score", 1) for q in questions))

                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("📝 试卷名称", paper.get("title", "未知"))
                        col2.metric("⭐ 难度", paper.get("difficulty", "中等"))
                        col3.metric("📊 总分", total_score)
                        col4.metric("📋 题型", f"{choice_count}选择 + {essay_count}简答")

                        st.markdown("---")
                        st.subheader("📖 知识点分布")
                        st.info(paper.get("knowledge_points", ""))

                        st.subheader("📋 题目列表")
                        for i, q in enumerate(questions, 1):
                            q_type = q.get("type", "choice")
                            if q_type == "choice":
                                st.markdown(f'<div class="question-choice">', unsafe_allow_html=True)
                                st.markdown(f"**{i}. [{q_type.upper()}] {q.get('content')}**  `({q.get('score', 1)}分)`")
                                st.markdown(f"   **选项：** {', '.join(q.get('options', []))}")
                                st.markdown(f"   **答案：** `{q.get('answer')}`")
                                st.markdown(f"   **知识点：** {q.get('knowledge_point', '未标注')}")
                                st.markdown(f'</div>', unsafe_allow_html=True)
                            else:
                                st.markdown(f'<div class="question-essay">', unsafe_allow_html=True)
                                st.markdown(f"**{i}. [{q_type.upper()}] {q.get('content')}**  `({q.get('score', 1)}分)`")
                                st.markdown(f"   **参考答案：** {q.get('answer', '无')[:200]}...")
                                st.markdown(f"   **知识点：** {q.get('knowledge_point', '未标注')}")
                                if q.get("grading_points"):
                                    st.markdown(f"   **评分要点：** {', '.join(q.get('grading_points', []))}")
                                st.markdown(f'</div>', unsafe_allow_html=True)
                            st.markdown("---")

                        st.download_button(
                            label="📥 导出试卷JSON",
                            data=json.dumps(paper, ensure_ascii=False, indent=2),
                            file_name=f"{paper.get('title', 'paper')}.json",
                            mime="application/json"
                        )

                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)


# ==================== 学生端：在线答题 ====================
elif mode == "✏️ 在线答题":
    st.title("✏️ 在线答题")
    papers = list_papers()
    if not papers:
        st.info("暂无试卷，请联系教师生成试卷")
    else:
        paper_names = [p.replace('.json', '') for p in papers]
        selected_name = st.selectbox("选择试卷", paper_names)
        selected_file = selected_name + ".json"
        paper = load_paper(selected_file)
        questions = paper.get("questions", [])

        st.markdown(f"### 📋 {paper.get('title', selected_name)}")

        choice_count = sum(1 for q in questions if q.get("type") == "choice")
        essay_count = sum(1 for q in questions if q.get("type") == "essay")
        total_score = sum(q.get("score", 1) for q in questions)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("难度", paper.get("difficulty", "中等"))
        col2.metric("总分", total_score)
        col3.metric("选择题", f"{choice_count}道")
        col4.metric("简答题", f"{essay_count}道")

        if use_ai_essay:
            st.info("🤖 AI智能批改已启用，简答题将获得详细评语")

        st.markdown("---")

        answers = []
        with st.form("answer_form"):
            for i, q in enumerate(questions, 1):
                q_type = q.get("type", "choice")
                type_icon = "🔘" if q_type == "choice" else "📝"
                st.markdown(f"{type_icon} **{i}. {q.get('content')}** `({q.get('score', 1)}分)`")

                if q_type == "choice":
                    # 获取原始选项列表
                    original_options = q.get('options', [])
                    # 在选项列表最前面添加一个空选项作为默认值
                    options_with_empty = ["(请选择)"] + original_options
                    ans = st.radio(
                        "",
                        options_with_empty,
                        key=f"q{i}",
                        label_visibility="collapsed",
                        horizontal=False,
                        index=0  # 默认选中第一个（请选择）
                    )
                    answers.append(ans)
                else:
                    st.caption("💡 提示：简答题将使用AI智能批改，请尽可能详细作答")
                    ans = st.text_area("答案", key=f"q{i}", height=100, label_visibility="collapsed",
                                       placeholder="请输入你的答案...")
                    answers.append(ans)
                st.divider()

            if st.form_submit_button("✅ 提交答案", type="primary"):
                with st.spinner("AI智能评分中..."):
                    try:
                        result = evaluate_answers(paper, answers, student_name, use_ai_essay)
                        st.session_state.last_result = result

                        st.balloons()
                        st.success(f"📊 得分：{result['total_score']}/{result['max_score']} ({result['percentage']}%)")
                        st.progress(result['percentage'] / 100)

                        fig, _ = generate_knowledge_radar(result['results'])
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)

                        st.subheader("答题详情")
                        for r in result['results']:
                            if r['is_correct']:
                                st.success(f"第{r['id']}题：{r['earned']}/{r['max_score']}分 ✓")
                            else:
                                st.error(f"第{r['id']}题：{r['earned']}/{r['max_score']}分 ✗")
                                with st.expander("查看详情"):
                                    st.markdown(f"**你的答案：** {r['student_answer']}")
                                    st.markdown(f"**正确答案：** {r['correct_answer']}")
                                    if r.get('ai_comment'):
                                        try:
                                            ai_comment = json.loads(r['ai_comment'])
                                            st.markdown(f"**🤖 AI评语：** {ai_comment.get('comment', '')}")
                                            st.markdown(f"**💡 改进建议：** {ai_comment.get('suggestions', '')}")
                                            if ai_comment.get('key_points_matched'):
                                                st.markdown(f"**✅ 答对要点：** {', '.join(ai_comment['key_points_matched'])}")
                                            if ai_comment.get('key_points_missing'):
                                                st.markdown(f"**❌ 遗漏要点：** {', '.join(ai_comment['key_points_missing'])}")
                                        except:
                                            st.markdown(f"**反馈：** {r['feedback']}")
                                    else:
                                        st.markdown(f"**反馈：** {r['feedback']}")
                    except sqlite3.OperationalError as e:
                        st.error(f"数据库错误：{e}")
                        st.info("请稍后重试，如果问题持续存在，请重启应用")
                    except Exception as e:
                        st.error(f"评分失败：{e}")


# ==================== 学生端：成绩报告 ====================
elif mode == "📊 成绩报告":
    st.title("📊 成绩报告")

    trend_fig = get_performance_trend(student_name)
    if trend_fig:
        st.plotly_chart(trend_fig, use_container_width=True)
    else:
        st.info("暂无成绩数据，请先完成答题")

    if st.session_state.get('last_result'):
        r = st.session_state.last_result
        col1, col2, col3 = st.columns(3)
        col1.metric("最新总分", f"{r['total_score']}/{r['max_score']}")
        col2.metric("最新得分率", f"{r['percentage']}%")
        col3.metric("错题数", r['wrong_count'])


# ==================== 学生端：错题本 ====================
elif mode == "📚 错题本":
    st.title("📚 错题本")

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT wq.paper_title, wq.question_content, wq.student_answer, wq.correct_answer, wq.knowledge_point, wq.created_at
        FROM wrong_questions wq
        JOIN students s ON wq.student_id = s.id
        WHERE s.name = ?
        ORDER BY wq.created_at DESC
    ''', (student_name,))
    wrongs = cursor.fetchall()
    conn.close()

    if not wrongs:
        st.info("🎉 暂无错题，继续加油！")
    else:
        st.info(f"📌 共 {len(wrongs)} 道错题")
        for w in wrongs:
            with st.expander(f"📄 {w[0]} - {w[5][:16]}"):
                st.markdown(f"**题目：** {w[1]}")
                st.markdown(f"**你的答案：** {w[2]}")
                st.markdown(f"**正确答案：** {w[3]}")
                st.markdown(f"**知识点：** {w[4]}")


# ==================== AI学习建议 ====================
elif mode == "🤖 AI学习建议":
    st.title("🤖 AI个性化学习建议")
    st.markdown("基于你的学习数据，AI将为你生成专属学习计划")

    if st.button("🔄 生成/刷新学习建议", type="primary"):
        st.session_state.advice_generated = False

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM students WHERE name = ?", (student_name,))
    student_row = cursor.fetchone()

    wrong_questions = []
    knowledge_scores = {}

    if student_row:
        student_id = student_row[0]
        cursor.execute('''
            SELECT paper_title, question_content, student_answer, correct_answer, knowledge_point, created_at
            FROM wrong_questions
            WHERE student_id = ?
            ORDER BY created_at DESC
            LIMIT 20
        ''', (student_id,))
        wrong_questions = cursor.fetchall()

        cursor.execute('''
            SELECT answers, percentage
            FROM exam_records
            WHERE student_id = ?
            ORDER BY created_at DESC
            LIMIT 1
        ''', (student_id,))
        last_exam = cursor.fetchone()
        if last_exam and last_exam[0]:
            try:
                results = json.loads(last_exam[0])
                for r in results:
                    kp = r.get("knowledge_point", "未分类")
                    if kp not in knowledge_scores:
                        knowledge_scores[kp] = {"total": 0, "earned": 0}
                    knowledge_scores[kp]["total"] += r.get("max_score", 1)
                    knowledge_scores[kp]["earned"] += r.get("earned", 0)
            except:
                pass
    conn.close()

    if not st.session_state.get('advice_generated', False) and (wrong_questions or knowledge_scores):
        with st.spinner("🤖 AI正在分析你的学习数据，生成个性化建议..."):
            advice = generate_learning_advice(student_name, wrong_questions, knowledge_scores)
            st.session_state.learning_advice = advice
            st.session_state.advice_generated = True

    advice = st.session_state.get('learning_advice', {})

    if not wrong_questions and not knowledge_scores:
        st.info("📝 请先完成至少一次答题，AI将根据你的学习数据生成个性化建议")
    elif advice and "error" not in advice:
        st.markdown("---")

        st.markdown("### 🎯 整体评估")
        st.info(advice.get('overall_assessment', '暂无评估'))

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🔴 薄弱知识点")
            weak = advice.get('weak_areas', [])
            if weak:
                for w in weak:
                    st.markdown(f'<div class="weak-area">⚠️ {w}</div>', unsafe_allow_html=True)
            else:
                st.success("🎉 暂无明显薄弱点！")

        with col2:
            st.markdown("### 🟢 优势知识点")
            strong = advice.get('strong_areas', [])
            if strong:
                for s in strong:
                    st.markdown(f'<div class="strong-area">✅ {s}</div>', unsafe_allow_html=True)
            else:
                st.caption("继续努力，发现你的优势！")

        st.markdown("---")
        st.markdown("### 📋 个性化学习行动计划")
        st.markdown(f'<div class="advice-box">📖 {advice.get("action_plan", "暂无")}</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📚 推荐资源")
            st.write(advice.get('recommended_resources', '暂无推荐'))
        with col2:
            st.markdown("### 💪 鼓励")
            st.success(f"✨ {advice.get('encouragement', '加油！')}")

        st.markdown("---")
        st.markdown("### 📊 学习数据概览")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📝 错题总数", len(wrong_questions))
        with col2:
            if knowledge_scores:
                avg_score = sum(v['earned'] / v['total'] for v in knowledge_scores.values()) / len(knowledge_scores) * 100
                st.metric("📈 最近平均掌握度", f"{avg_score:.1f}%")
    elif advice and "error" in advice:
        st.error(f"生成建议失败：{advice['error']}")
        st.info("提示：请确保网络正常，API Key有效")