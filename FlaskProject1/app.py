from encodings import undefined

import numpy as np
from pyecharts.charts import Line, Radar
from pyecharts import options as opts
from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import chardet
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.font_manager as fm
from auth import login_manager, login_required, current_user
from auth import init_app
from io import BytesIO  # 添加这一行
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from flask_login import login_required


# 创建Flask应用
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'  # 用于会话签名
app.config['JSON_AS_ASCII'] = False  # 确保JSON响应使用UTF-8编码

# 初始化认证模块
init_app(app)

# 初始化登录管理器
login_manager.init_app(app)
login_manager.login_view = 'auth.login'

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei"]


def read_cleaned_data():
    """读取清洗后的数据，自动检测文件编码"""
    try:
        # 检测文件编码
        with open('score_cleaned.csv', 'rb') as f:
            result = chardet.detect(f.read())
            encoding = result['encoding']
            print(f"检测到的文件编码为: {encoding}")

        # 使用正确的编码格式加载清洗后的数据
        df = pd.read_csv('score_cleaned.csv', encoding=encoding)
        print(f"成功加载数据，共 {len(df)} 条记录")
        return df
    except Exception as e:
        print(f"读取数据失败: {e}")
        # 返回一个空的DataFrame，避免应用崩溃
        return pd.DataFrame()



def generate_score_distribution_chart(df):
    """生成总分分布直方图"""
    if df.empty:
        return ""

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['总分'], kde=True, bins=20, ax=ax)
    ax.set_title('总分分布直方图')
    ax.set_xlabel('总分')
    ax.set_ylabel('频数')

    # 保存图表为图片
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    img_data = base64.b64encode(output.getvalue()).decode('utf-8')
    plt.close(fig)
    return img_data


def generate_school_scores_chart(df, min_score=300):
    """生成各学校平均分数对比条形图（可设置最低分数筛选）

    Args:
        df: 数据框
        min_score: 只显示平均分高于此值的学校，默认300
    """
    if df.empty:
        return ""

    # 计算各学校平均分并筛选
    school_avg = df.groupby('学校')['总分'].mean().reset_index()
    school_avg = school_avg[school_avg['总分'] >= min_score].sort_values('总分', ascending=False)

    # 如果没有符合条件的数据
    if school_avg.empty:
        return ""

    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 8))

    # 根据学校数量动态调整图表高度
    n_schools = len(school_avg)
    fig.set_size_inches(12, max(6, n_schools * 0.5))  # 每个学校0.5英寸高度

    # 绘制条形图
    sns.barplot(
        x='总分',
        y='学校',
        data=school_avg,
        ax=ax,
        palette='viridis',
        hue='学校',
        legend=False,
        dodge=False
    )

    # 添加图表元素
    ax.set_title(f'各学校平均分数对比（≥{min_score}分）', fontsize=14)
    ax.set_xlabel('平均总分', fontsize=12)
    ax.set_ylabel('')

    # 添加数据标签
    for i, v in enumerate(school_avg['总分']):
        ax.text(v + 1, i, f'{v:.1f}', va='center', fontsize=10)

    # 调整布局
    plt.tight_layout()

    # 转换为图片
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    img_data = base64.b64encode(output.getvalue()).decode('utf-8')
    plt.close(fig)

    return img_data


def generate_province_distribution_chart(df):
    """生成省份分布饼图"""
    if df.empty:
        return ""

    province_counts = df['省份'].value_counts().reset_index()
    province_counts.columns = ['省份', '数量']

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.pie(province_counts['数量'], labels=province_counts['省份'], autopct='%1.1f%%',
           startangle=90, shadow=True)
    ax.set_title('各省份学校数量分布')
    ax.axis('equal')  # 使饼图为正圆形

    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    img_data = base64.b64encode(output.getvalue()).decode('utf-8')
    plt.close(fig)
    return img_data


def generate_correlation_chart(df):
    """生成各科成绩相关性矩阵图"""
    if df.empty:
        return ""

    # 选择分数列
    score_columns = ['总分', '政治', '英语', '数学', '408专业基础']
    score_data = df[score_columns].dropna()

    if score_data.empty:
        return ""

    fig, ax = plt.subplots(figsize=(10, 8))
    corr = score_data.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', square=True, ax=ax)
    ax.set_title('各科成绩相关性矩阵')

    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    img_data = base64.b64encode(output.getvalue()).decode('utf-8')
    plt.close(fig)
    return img_data


def generate_subject_radar_chart(df):
    """生成各科分数雷达图"""
    if df.empty:
        return ""

    # 计算各科平均分
    avg_scores = {
        '政治': df['政治'].mean(),
        '英语': df['英语'].mean(),
        '数学': df['数学'].mean(),
        '408专业基础': df['408专业基础'].mean()
    }

    # 打印调试信息
    print("雷达图数据:", avg_scores)

    # 检查是否有缺失值
    if any(pd.isna(score) for score in avg_scores.values()):
        print("警告: 某些科目存在缺失值")
        for subject in avg_scores:
            if pd.isna(avg_scores[subject]):
                avg_scores[subject] = 0

    # 准备雷达图数据
    subjects = list(avg_scores.keys())
    values = list(avg_scores.values())

    # 创建雷达图
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)

    # 计算角度
    angles = np.linspace(0, 2 * np.pi, len(subjects), endpoint=False).tolist()

    # 闭合雷达图
    values += values[:1]
    angles += angles[:1]

    # 绘制雷达图
    ax.plot(angles, values, 'o-', linewidth=2, color='#165DFF')  # 设置线条颜色
    ax.fill(angles, values, alpha=0.25, color='#165DFF')  # 设置填充颜色

    # 设置坐标轴
    ax.set_thetagrids(np.degrees(angles[:-1]), subjects, fontsize=12)  # 设置标签字体大小
    ax.set_ylim(0, 100)
    ax.set_title('各科分数平均分布雷达图', fontsize=16, pad=20)  # 设置标题字体大小和间距

    # 添加网格线和背景
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_facecolor('#f8f9fa')  # 设置背景颜色

    # 转换为Base64
    output = BytesIO()
    FigureCanvas(fig).print_png(output)
    img_data = base64.b64encode(output.getvalue()).decode('utf-8')
    plt.close(fig)

    return img_data


def generate_score_trend_chart(df):
    """生成录取分数线趋势图"""
    if df.empty:
        return ""

    # 检查数据中是否有年份列
    if '年份' not in df.columns:
        print("数据中缺少'年份'列，无法生成趋势图")
        return ""

    # 按年份计算平均分
    yearly_avg = df.groupby('年份')['总分'].mean().reset_index()

    # 确保年份按升序排列
    yearly_avg = yearly_avg.sort_values('年份')

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(yearly_avg['年份'], yearly_avg['总分'], marker='o', linestyle='-', color='#165DFF')

    # 添加数据标签
    for x, y in zip(yearly_avg['年份'], yearly_avg['总分']):
        ax.annotate(f'{y:.1f}', (x, y), textcoords='offset points',
                    xytext=(0, 10), ha='center', fontsize=10)

    ax.set_title('历年录取分数线趋势')
    ax.set_xlabel('年份')
    ax.set_ylabel('平均分')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xticks(yearly_avg['年份'])  # 设置x轴刻度为实际年份

    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    img_data = base64.b64encode(output.getvalue()).decode('utf-8')
    plt.close(fig)

    return img_data


def generate_application_heatmap(df):
    """生成报考热度地图"""
    if df.empty:
        return ""

    # 计算各省份的报考人数（假设每个记录代表一个考生）
    province_counts = df['省份'].value_counts().reset_index()
    province_counts.columns = ['省份', '报考人数']

    # 准备地图数据
    map_data = []
    for _, row in province_counts.iterrows():
        map_data.append({
            'name': row['省份'],
            'value': row['报考人数']
        })

    # 创建图表
    from pyecharts import options as opts
    from pyecharts.charts import Map

    c = (
        Map()
        .add("报考人数", map_data, "china")
        .set_global_opts(
            title_opts=opts.TitleOpts(title="全国各省份报考热度地图"),
            visualmap_opts=opts.VisualMapOpts(max_=province_counts['报考人数'].max()),
        )
    )

    # 渲染图表为HTML
    c.render("templates/application_heatmap.html")

    # 读取HTML内容并返回
    with open("templates/application_heatmap.html", "r", encoding="utf-8") as f:
        html_content = f.read()

    # 将HTML内容转换为Base64编码
    encoded_html = base64.b64encode(html_content.encode("utf-8")).decode("utf-8")

    return encoded_html


# 新增：按省份筛选学校的函数
def generate_school_by_province_chart(df):
    if df.empty:
        return ""

    # 创建交叉表
    school_province = pd.crosstab(df['学校'], df['省份'])

    # 准备绘图数据
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(school_province, annot=True, cmap='YlGnBu', fmt='d', ax=ax)
    ax.set_title('各省份学校分布热力图')
    ax.set_xlabel('省份')
    ax.set_ylabel('学校')

    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    img_data = base64.b64encode(output.getvalue()).decode('utf-8')
    plt.close(fig)
    return img_data

@app.route('/')
@login_required
def index():
    """主路由，展示所有图表和数据"""
    df = read_cleaned_data()

    # 生成图表
    score_distribution = generate_score_distribution_chart(df)

    # 生成图表（添加min_score参数）
    school_scores = generate_school_scores_chart(df, min_score=300)  # 只显示300分以上的学校

    province_distribution = generate_province_distribution_chart(df)
    correlation_chart = generate_correlation_chart(df)
    subject_radar = generate_subject_radar_chart(df)
    score_trend = generate_score_trend_chart(df)

    # 新增：生成报考热度地图
    application_heatmap = generate_application_heatmap(df)

    # 准备表格数据
    table_data = df.head(20).to_dict('records')

    # 准备省份数据
    province_counts = df['省份'].value_counts().reset_index()
    province_counts.columns = ['省份', '数量']
    province_data = province_counts.to_dict('records')

    return render_template('index.html',
                           score_distribution=score_distribution,
                           school_scores=school_scores,
                           province_distribution=province_distribution,
                           correlation_chart=correlation_chart,
                           subject_radar=subject_radar,
                           score_trend=score_trend,
                           application_heatmap=application_heatmap,  # 添加报考热度地图数据
                           table_data=table_data,
                           current_user=current_user,
                           province_data=province_data,
                           df=df)

# 新增院校推荐路由

# 增强院校推荐功能
@app.route('/school_recommendation', methods=['GET', 'POST'])
@login_required
def school_recommendation():
    df = read_cleaned_data()

    if request.method == 'POST':
        try:
            user_scores = {
                '总分': int(request.form.get('total_score', 350)),
                '政治': int(request.form.get('politics_score', 60)),
                '英语': int(request.form.get('english_score', 60)),
                '数学': int(request.form.get('math_score', 100)),
                '408专业基础': int(request.form.get('major_score', 100))
            }

            user_attributes = {
                '省份': request.form.get('province', '不限'),
                '院校层次': request.form.get('school_level', '不限')
            }

            recommended_schools = recommend_schools(df, user_scores, user_attributes)

            if not recommended_schools:
                flash('没有找到符合条件的院校，请放宽筛选条件', 'warning')
                return redirect(url_for('school_recommendation'))

            radar_chart = generate_comparison_radar(df, recommended_schools, user_scores)

            return render_template('school_recommendation_result.html',
                                   recommended_schools=recommended_schools,
                                   radar_chart=radar_chart,
                                   user_scores=user_scores)

        except Exception as e:
            flash(f'发生错误: {str(e)}', 'error')
            print(f"院校推荐错误: {str(e)}")
            return redirect(url_for('school_recommendation'))

    # GET请求处理
    provinces = ['不限'] + sorted(df['省份'].dropna().unique().tolist())
    school_levels = ['不限'] + sorted(df['院校层次'].dropna().unique().tolist())

    return render_template('school_recommendation_form.html',
                           provinces=provinces,
                           school_levels=school_levels)
# except Exception as e:
#         print("发生错误:", str(e))  # 输出到控制台
#         import traceback
#         traceback.print_exc()  # 打印完整堆栈
#         flash(f'发生错误: {str(e)}', 'error')
#         return redirect(url_for('school_recommendation'))

# 增强推荐算法
def recommend_schools(df, user_scores, user_attributes):
    try:
        score_columns = ['总分', '政治', '英语', '数学', '408专业基础']
        weights = np.array([0.4, 0.1, 0.1, 0.2, 0.2])

        # 计算相似度
        weighted_scores = df[score_columns].fillna(0) * weights
        user_array = np.array([user_scores[col] for col in score_columns]) * weights
        similarities = cosine_similarity([user_array], weighted_scores)[0]
        df['similarity'] = similarities

        # 属性筛选
        filtered_df = df.copy()
        for attr, value in user_attributes.items():
            if value != '不限' and attr in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[attr] == value]

        # 返回结构化数据
        top_schools = filtered_df.nlargest(5, 'similarity')
        return [{
            'name': row['学校'],
            'province': row['省份'],
            'level': row.get('院校层次', '未知'),
            'match': int(row['similarity'] * 100),
            'avg_score': int(row['总分']),
            'scores': {  # 确保包含这个字段
                '政治': int(row['政治']),
                '英语': int(row['英语']),
                '数学': int(row['数学']),
                '专业': int(row.get('408专业基础', 0))
            }
        } for _, row in top_schools.iterrows()]

    except Exception as e:
        print(f"推荐算法错误: {str(e)}")
        return []


# 生成对比雷达图
def generate_comparison_radar(df, recommended_schools, user_scores):
    """生成用户分数与推荐院校对比雷达图"""
    subjects = ['政治', '英语', '数学', '408专业基础']

    # 准备数据 - 确保处理可能缺失的键
    data = []
    for school in recommended_schools[:3]:  # 只显示前3个学校的对比
        if 'scores' not in school:
            print(f"警告: 学校数据缺少scores字段 - {school}")
            continue

        data.append({
            'value': [school['scores'].get(sub, 0) for sub in subjects],
            'name': school.get('name', '未知学校')
        })

    # 添加用户数据
    data.append({
        'value': [user_scores.get(sub, 0) for sub in subjects],
        'name': '我的分数'
    })

    # 创建雷达图
    radar = (
        Radar()
        .add_schema(
            schema=[
                opts.RadarIndicatorItem(name=sub, max_=100)
                for sub in subjects
            ],
            splitarea_opt=opts.SplitAreaOpts(is_show=True),
            textstyle_opts=opts.TextStyleOpts(color="#333")
        )
        .add(
            series_name="分数对比",
            data=data,
            linestyle_opts=opts.LineStyleOpts(width=2),
            areastyle_opts=opts.AreaStyleOpts(opacity=0.1)
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(title="我的分数与推荐院校对比"),
            legend_opts=opts.LegendOpts(selected_mode="single")
        )
    )

    return radar.render_embed()




def generate_trend_chart(trend_data):
    """生成带预测区间的趋势图表"""
    if not trend_data:
        return ""

    # 分离实际数据和预测数据
    actual_data = [d for d in trend_data['trend_data'] if not d.get('is_prediction')]
    predict_data = [d for d in trend_data['trend_data'] if d.get('is_prediction')]

    # 准备数据
    all_years = [str(d['year']) for d in trend_data['trend_data']]
    all_scores = [d['avg_score'] for d in trend_data['trend_data']]

    # 创建折线图
    line = (
        Line(init_opts=opts.InitOpts(width="100%", height="500px"))
        .add_xaxis(all_years)
        .add_yaxis(
            series_name="录取均分",
            y_axis=all_scores,
            is_smooth=True,
            linestyle_opts=opts.LineStyleOpts(width=3),
            label_opts=opts.LabelOpts(is_show=False),
            markpoint_opts=opts.MarkPointOpts(
                data=[
                    opts.MarkPointItem(type_="max", name="最高"),
                    opts.MarkPointItem(type_="min", name="最低")
                ]
            ),
            markline_opts=opts.MarkLineOpts(
                data=[opts.MarkLineItem(type_="average", name="平均值")]
            )
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title=f"{trend_data['school']} {trend_data['major']}专业录取趋势",
                subtitle="蓝色:历史数据 | 红色:预测数据"
            ),
            tooltip_opts=opts.TooltipOpts(
                trigger="axis",
                formatter="年份: {b0}<br/>平均分: {c0}分"
            ),
            visualmap_opts=opts.VisualMapOpts(
                dimension=0,
                is_piecewise=True,
                pieces=[
                    {"lte": len(actual_data) - 1, "color": "#5470C6", "label": "历史数据"},
                    {"gt": len(actual_data) - 1, "color": "#EE6666", "label": "预测数据"}
                ],
                pos_top="50",
                pos_right="10"
            ),
            datazoom_opts=[opts.DataZoomOpts()]
        )
    )

    # 添加预测区间背景色
    if predict_data:
        line.set_series_opts(
            markarea_opts=opts.MarkAreaOpts(
                data=[[
                    {"xAxis": all_years[len(actual_data)]},
                    {"xAxis": all_years[-1]}
                ]],
                itemstyle_opts=opts.ItemStyleOpts(color="rgba(238, 102, 102, 0.1)"),
                label_opts=opts.LabelOpts(
                    position="inside",
                    formatter="预测区间",
                    color="#EE6666"
                )
            )
        )

    return line.render_embed()


def generate_enrollment_chart():
    try:
        # 确保文件存在
        import os
        if not os.path.exists('student2.csv'):
            print("错误: students2.csv文件不存在")
            return ""

        # 读取文件
        df = pd.read_csv('student2.csv')

        # 检查必要列
        required_columns = ['year', 'enrollment']
        if not all(col in df.columns for col in required_columns):
            print(f"错误: CSV文件缺少必要的列，需要{required_columns}")
            return ""

        # 准备数据
        years = df['year'].astype(str).tolist()  # 确保年份是字符串
        enrollments = df['enrollment'].tolist()

        # 创建折线图
        line = (
            Line(init_opts=opts.InitOpts(width="100%", height="400px"))
            .add_xaxis(years)
            .add_yaxis(
                series_name="",
                y_axis=enrollments,
                is_smooth=True,
                linestyle_opts=opts.LineStyleOpts(width=3),
                label_opts=opts.LabelOpts(is_show=True),
                markpoint_opts=opts.MarkPointOpts(
                    data=[
                        opts.MarkPointItem(type_="max", name="最高"),
                        opts.MarkPointItem(type_="min", name="最低")
                    ]
                )
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(
                    title="2023-2025年研究生报考招生人数趋势",
                    pos_left="center"
                ),
                tooltip_opts=opts.TooltipOpts(trigger="axis"),
                xaxis_opts=opts.AxisOpts(
                    name="年份",
                    name_location="middle",
                    name_gap=30
                ),
                yaxis_opts=opts.AxisOpts(
                    name="人数",
                    name_location="middle",
                    name_gap=30
                ),
                datazoom_opts=[opts.DataZoomOpts()]
            )
        )

        return line.render_embed()

    except Exception as e:
        print(f"生成报考招生人数折线图出错: {str(e)}")
        return ""




# 趋势预测功能
# 趋势预测功能
# 趋势预测路由
@app.route('/trend_prediction', methods=['GET', 'POST'])
@login_required
def trend_prediction():
    df = read_cleaned_data()

    enrollment_chart = generate_enrollment_chart()
    if not enrollment_chart:
        print("生成招生人数图表失败")  # 调试输出
    if request.method == 'POST':
        try:
            target_school = request.form.get('school', '').strip()
            if not target_school:
                flash('请选择学校', 'error')
                return redirect(url_for('trend_prediction'))

            target_major = request.form.get('major', '计算机科学与技术').strip()

            trend_data = predict_trend(df, target_school, target_major)
            if not trend_data or 'trend_data' not in trend_data:
                flash('未找到该学校的录取数据', 'error')
                return redirect(url_for('trend_prediction'))

            # 确保数据格式正确
            if not all('year' in d and 'avg_score' in d for d in trend_data['trend_data']):
                flash('数据格式错误', 'error')
                return redirect(url_for('trend_prediction'))

            chart_html = generate_trend_chart(trend_data)

            return render_template('trend_prediction_result.html',
                                   chart_html=chart_html,
                                   school=target_school,
                                   major=target_major,
                                   trend_data=trend_data,
                                   growth_rate=trend_data.get('growth_rate', 0))

        except Exception as e:
            flash(f'预测出错: {str(e)}', 'error')
            print(f"趋势预测错误: {str(e)}")
            return redirect(url_for('trend_prediction'))

    # GET请求处理
    schools = sorted(df['学校'].dropna().unique().tolist())
    return render_template('trend_prediction_form.html',
                           schools=schools,enrollment_chart=enrollment_chart)

#青研博客功能
#青研博客路由
@app.route('/qingyan-blog')
def qingyan_blog():
    return render_template('qingyan_blog.html')


@app.route('/submit-contribution', methods=['POST'])
def submit_contribution():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        date = request.form.get('date')
        title = request.form.get('title')
        message = request.form.get('message')

        # 这里添加处理投稿的逻辑，比如存入数据库
        # ...

        return redirect(url_for('qingyan_blog'))  # 提交后重定向回博客页面

# 在创建app后添加
def is_undefined_or_false(value):
    return value is undefined or not value

app.jinja_env.tests['undefined_or_false'] = is_undefined_or_false


def predict_trend(df, school, major):
    """改进的趋势预测函数"""
    try:
        # 筛选数据
        school_df = df[(df['学校'] == school)].copy()
        if school_df.empty:
            return None

        # 计算年度统计77+
        yearly_stats = school_df.groupby('年份')['总分'].agg(
            ['mean', 'count']
        ).reset_index()
        yearly_stats.columns = ['year', 'avg_score', 'count']

        # 生成预测
        predictions = []
        if len(yearly_stats) >= 2:
            last_avg = yearly_stats.iloc[-1]['avg_score']
            growth_rate = 0.03  # 默认增长率

            # 计算实际增长率（如果数据充足）
            if len(yearly_stats) >= 3:
                prev_avg = yearly_stats.iloc[-2]['avg_score']
                growth_rate = max(0, (last_avg - prev_avg) / prev_avg)

            # 生成未来3年预测
            for i in range(1, 4):
                predictions.append({
                    'year': yearly_stats.iloc[-1]['year'] + i,
                    'avg_score': round(last_avg * (1 + growth_rate * i), 1),
                    'is_prediction': True
                })

        # 合并历史数据和预测数据
        history_data = yearly_stats.to_dict('records')
        for d in history_data:
            d['is_prediction'] = False

        return {
            'school': school,
            'major': major,
            'trend_data': history_data + predictions,  # 列表直接相加
            'has_prediction': bool(predictions),
            'growth_rate': round(growth_rate * 100, 1) if predictions else None
        }

    except Exception as e:
        print(f"预测出错: {e}")
        return None




if __name__ == '__main__':
    app.run(debug=True)

