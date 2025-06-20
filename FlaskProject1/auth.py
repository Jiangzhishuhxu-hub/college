from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Email, EqualTo

# 创建认证蓝图
auth_bp = Blueprint('auth', __name__)
login_manager = LoginManager()
login_manager.login_view = 'auth.login'  # 确保这里使用 'auth.login'

# 模拟用户数据库 - 在实际应用中应使用数据库
users = {
    '450@qq.com': {'password': '450', 'name': '江直树旭'},
    '750@163.com': {'password': '750', 'name': '李四'}
}


# 用户类
class User(UserMixin):
    def __init__(self, id, name):
        self.id = id
        self.name = name


# 用户加载回调函数
@login_manager.user_loader
def load_user(user_id):
    # 从用户数据库中加载用户
    if user_id in users:
        return User(user_id, users[user_id]['name'])
    return None


# 登录表单
class LoginForm(FlaskForm):
    email = StringField('邮箱', validators=[DataRequired(), Email()])
    password = PasswordField('密码', validators=[DataRequired()])
    submit = SubmitField('登录')


# 注册表单
class RegisterForm(FlaskForm):
    email = StringField('邮箱', validators=[DataRequired(), Email()])
    name = StringField('姓名', validators=[DataRequired()])
    password = PasswordField('密码', validators=[DataRequired()])
    confirm_password = PasswordField('确认密码', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('注册')


# 登录路由
@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        email = form.email.data
        password = form.password.data

        if email in users and users[email]['password'] == password:
            user = User(email, users[email]['name'])
            login_user(user)
            flash('登录成功', 'success')
            return redirect(url_for('index'))
        else:
            flash('登录失败，请检查邮箱和密码', 'error')

    return render_template('login.html', form=form)


# 注册路由
@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        email = form.email.data
        name = form.name.data
        password = form.password.data

        if email in users:
            flash('邮箱已注册', 'error')
            return redirect(url_for('auth.register'))

        # 注册新用户
        users[email] = {'password': password, 'name': name}
        flash('注册成功，请登录', 'success')
        return redirect(url_for('auth.login'))

    return render_template('register.html', form=form)


# 登出路由
@auth_bp.route('/logout')
@login_required
def logout():
    logout_user()
    flash('已退出登录', 'info')
    return redirect(url_for('index'))


# 初始化认证模块
def init_app(app):
    app.register_blueprint(auth_bp)
    login_manager.init_app(app)


def login_required():
    return None


def current_user():
    return None