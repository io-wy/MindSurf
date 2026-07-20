"""Build and audit the frozen 192-item stage-correct pretraining MCQ suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from python_starter.core.data_contract import (  # noqa: E402
    iter_jsonl,
    sha256_file,
    write_json_atomic,
)

CATEGORIES = (
    "zh_fact",
    "math",
    "code",
    "english",
    "long_context",
    "calibration_reasoning",
)


def _normalized(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split())


def _item(
    *,
    item_id: str,
    category: str,
    prompt: str,
    correct: str,
    distractors: tuple[str, str, str],
    answer: int,
    source: str,
) -> dict[str, Any]:
    values = [correct, *distractors]
    choices = values[1 : answer + 1] + [values[0]] + values[answer + 1 :]
    return {
        "id": item_id,
        "category": category,
        "prompt": prompt,
        "choices": [f" {choice}" for choice in choices],
        "answer": answer,
        "source": source,
    }


def _authored_items(
    category: str,
    rows: list[tuple[str, str, str, tuple[str, str, str]]],
) -> list[dict[str, Any]]:
    return [
        _item(
            item_id=f"{category}_{item_id}",
            category=category,
            prompt=prompt,
            correct=correct,
            distractors=distractors,
            answer=index % 4,
            source="maintainer-authored-v2",
        )
        for index, (item_id, prompt, correct, distractors) in enumerate(rows)
    ]


def build_items() -> list[dict[str, Any]]:
    """Create six balanced 32-item domains with no assistant-instruction scoring."""
    facts = [
        ("china_capital", "问题：中国的首都是哪座城市？\n答案：", "北京", ("上海", "广州", "南京")),
        ("water_formula", "问题：水的化学式是什么？\n答案：", "H2O", ("CO2", "NaCl", "O2")),
        ("sun_type", "问题：太阳属于哪类天体？\n答案：", "恒星", ("行星", "卫星", "彗星")),
        ("week_days", "问题：通常一周有几天？\n答案：", "7天", ("5天", "10天", "12天")),
        (
            "earth_satellite",
            "问题：地球的天然卫星叫什么？\n答案：",
            "月球",
            ("火星", "太阳", "金星"),
        ),
        ("oxygen_symbol", "问题：氧元素的化学符号是什么？\n答案：", "O", ("H", "C", "N")),
        (
            "largest_ocean",
            "问题：世界面积最大的海洋是哪一个？\n答案：",
            "太平洋",
            ("大西洋", "印度洋", "北冰洋"),
        ),
        (
            "plants_gas",
            "问题：绿色植物光合作用通常吸收哪种气体？\n答案：",
            "二氧化碳",
            ("氧气", "氮气", "氢气"),
        ),
        (
            "red_planet",
            "问题：常被称为红色星球的是哪颗行星？\n答案：",
            "火星",
            ("木星", "水星", "海王星"),
        ),
        (
            "boiling",
            "问题：标准大气压下水的沸点约是多少摄氏度？\n答案：",
            "100",
            ("0", "50", "212"),
        ),
        ("author_luxun", "问题：《呐喊》的作者是谁？\n答案：", "鲁迅", ("老舍", "巴金", "茅盾")),
        ("great_wall", "问题：长城主要位于哪个国家？\n答案：", "中国", ("印度", "埃及", "意大利")),
        ("cpu", "问题：CPU 通常指什么？\n答案：", "中央处理器", ("图形处理器", "内存", "硬盘")),
        (
            "ram",
            "问题：计算机断电后通常会丢失内容的是哪种存储？\n答案：",
            "内存",
            ("固态硬盘", "光盘", "磁带"),
        ),
        (
            "dna",
            "问题：DNA 的中文常用名称是什么？\n答案：",
            "脱氧核糖核酸",
            ("核糖核酸", "蛋白质", "葡萄糖"),
        ),
        ("seasons", "问题：北半球通常一年有几个季节？\n答案：", "四个", ("两个", "三个", "六个")),
        (
            "gravity",
            "问题：物体下落与哪种基本现象直接相关？\n答案：",
            "重力",
            ("浮力", "静电", "折射"),
        ),
        ("meter", "问题：国际单位制中长度的基本单位是什么？\n答案：", "米", ("升", "克", "秒")),
        ("mammal", "问题：下面哪种动物属于哺乳动物？\n答案：", "海豚", ("鲨鱼", "章鱼", "企鹅")),
        (
            "volcano",
            "问题：岩浆到达地表后通常称为什么？\n答案：",
            "熔岩",
            ("冰川", "沉积岩", "地下水"),
        ),
        (
            "photosynthesis",
            "问题：光合作用主要发生在植物细胞的哪个结构？\n答案：",
            "叶绿体",
            ("细胞核", "核糖体", "液泡"),
        ),
        (
            "blood_pump",
            "问题：人体负责泵送血液的器官是什么？\n答案：",
            "心脏",
            ("肺", "肝脏", "胃"),
        ),
        ("binary", "问题：二进制只使用哪两个数字？\n答案：", "0和1", ("1和2", "0和2", "2和3")),
        (
            "http",
            "问题：HTTP 主要用于哪类通信？\n答案：",
            "网页资源传输",
            ("卫星导航", "电力输送", "音频编码"),
        ),
        (
            "compiler",
            "问题：编译器的主要作用是什么？\n答案：",
            "翻译程序代码",
            ("管理文件夹", "绘制图片", "压缩视频"),
        ),
        (
            "database",
            "问题：关系数据库通常以什么结构组织数据？\n答案：",
            "表",
            ("音轨", "像素", "帧"),
        ),
        ("yangtze", "问题：中国最长的河流是哪一条？\n答案：", "长江", ("黄河", "珠江", "黑龙江")),
        (
            "everest",
            "问题：世界海拔最高的山峰是哪一座？\n答案：",
            "珠穆朗玛峰",
            ("泰山", "富士山", "乞力马扎罗山"),
        ),
        ("pi", "问题：圆周率通常用哪个希腊字母表示？\n答案：", "π", ("α", "β", "λ")),
        ("triangle", "问题：平面三角形内角和是多少度？\n答案：", "180", ("90", "270", "360")),
        ("ice_state", "问题：冰属于水的哪种物态？\n答案：", "固态", ("液态", "气态", "等离子态")),
        (
            "sound_medium",
            "问题：声音不能在哪种环境中传播？\n答案：",
            "真空",
            ("空气", "水", "钢铁"),
        ),
    ]
    code = [
        (
            "py_def",
            "问题：Python 定义函数使用哪个关键字？\n答案：",
            "def",
            ("class", "import", "yield"),
        ),
        (
            "py_loop",
            "问题：Python 遍历序列常用哪个关键字？\n答案：",
            "for",
            ("try", "raise", "lambda"),
        ),
        (
            "py_append",
            "问题：Python 列表末尾添加元素常用哪个方法？\n答案：",
            "append",
            ("split", "strip", "join"),
        ),
        (
            "json_object",
            "问题：JSON 对象使用哪种括号？\n答案：",
            "花括号",
            ("方括号", "圆括号", "尖括号"),
        ),
        (
            "git_commit",
            "问题：Git 创建提交使用哪个子命令？\n答案：",
            "commit",
            ("clone", "status", "branch"),
        ),
        (
            "git_diff",
            "问题：Git 查看未提交差异常用哪个子命令？\n答案：",
            "diff",
            ("push", "tag", "init"),
        ),
        ("bool_true", "问题：Python 布尔真值如何书写？\n答案：", "True", ("true", "TRUE", "yes")),
        (
            "length",
            "问题：Python 获取序列长度常用哪个函数？\n答案：",
            "len",
            ("size", "length", "count"),
        ),
        (
            "dict_key",
            "问题：Python 字典通过什么访问对应值？\n答案：",
            "键",
            ("行号", "像素", "端口"),
        ),
        (
            "exception",
            "问题：Python 捕获异常使用哪个关键字？\n答案：",
            "except",
            ("break", "yield", "global"),
        ),
        (
            "sql_select",
            "问题：SQL 查询数据通常使用哪个关键字？\n答案：",
            "SELECT",
            ("DROP", "COMMIT", "GRANT"),
        ),
        ("html_link", "问题：HTML 创建超链接常用哪个标签？\n答案：", "a", ("div", "span", "table")),
        (
            "css_color",
            "问题：CSS 设置文字颜色常用哪个属性？\n答案：",
            "color",
            ("margin", "display", "height"),
        ),
        (
            "http_get",
            "问题：HTTP 中读取资源通常使用哪个方法？\n答案：",
            "GET",
            ("DELETE", "PATCH", "CONNECT"),
        ),
        (
            "status_404",
            "问题：HTTP 404 通常表示什么？\n答案：",
            "资源未找到",
            ("请求成功", "永久重定向", "服务正常"),
        ),
        (
            "shell_pwd",
            "问题：类 Unix shell 中显示当前目录常用什么命令？\n答案：",
            "pwd",
            ("mkdir", "touch", "grep"),
        ),
        (
            "shell_ls",
            "问题：类 Unix shell 中列出目录内容常用什么命令？\n答案：",
            "ls",
            ("cd", "rm", "cat"),
        ),
        (
            "docker_build",
            "问题：Docker 根据 Dockerfile 构建镜像常用哪个命令？\n答案：",
            "docker build",
            ("docker ps", "docker logs", "docker stop"),
        ),
        (
            "yaml",
            "问题：YAML 通常使用什么表达层级？\n答案：",
            "缩进",
            ("二进制位", "固定列宽", "像素"),
        ),
        (
            "api",
            "问题：REST API 的资源通常由什么标识？\n答案：",
            "URL",
            ("显卡编号", "字体", "时钟频率"),
        ),
        (
            "complexity",
            "问题：二分查找的典型时间复杂度是什么？\n答案：",
            "O(log n)",
            ("O(n²)", "O(2ⁿ)", "O(n!)"),
        ),
        (
            "stack",
            "问题：栈结构通常遵循什么顺序？\n答案：",
            "后进先出",
            ("先进先出", "随机进出", "按字母排序"),
        ),
        (
            "queue",
            "问题：普通队列通常遵循什么顺序？\n答案：",
            "先进先出",
            ("后进先出", "随机访问", "按哈希排序"),
        ),
        (
            "recursion",
            "问题：递归函数必须具备什么以避免无限调用？\n答案：",
            "终止条件",
            ("更多线程", "网络端口", "图形界面"),
        ),
        (
            "test_assert",
            "问题：单元测试中的断言主要验证什么？\n答案：",
            "可观察行为",
            ("文件颜色", "编辑器主题", "鼠标位置"),
        ),
        (
            "seed",
            "问题：训练时固定随机种子主要为了什么？\n答案：",
            "提高可复现性",
            ("增加显存", "改变许可证", "压缩文件"),
        ),
        (
            "checkpoint",
            "问题：训练 checkpoint 通常应包含什么以精确恢复？\n答案：",
            "模型与优化器状态",
            ("只有日志标题", "只有文件名", "只有 README"),
        ),
        (
            "hash",
            "问题：SHA-256 常用于验证什么？\n答案：",
            "内容身份",
            ("屏幕亮度", "GPU 温度单位", "键盘布局"),
        ),
        (
            "transaction",
            "问题：数据库事务的原子性表示什么？\n答案：",
            "全部成功或全部回滚",
            ("查询必然更快", "数据永不删除", "只允许一个用户"),
        ),
        (
            "process",
            "问题：操作系统中的 PID 标识什么？\n答案：",
            "进程",
            ("文件扩展名", "网络协议", "显卡型号"),
        ),
        (
            "env",
            "问题：环境变量通常用于提供什么？\n答案：",
            "进程配置",
            ("模型参数梯度", "图片像素", "硬盘扇区"),
        ),
        (
            "lock",
            "问题：并发写共享状态时使用锁主要防止什么？\n答案：",
            "竞态条件",
            ("语法高亮", "网络压缩", "字体缺失"),
        ),
    ]
    english = [
        (
            "water_freeze",
            "Question: What does liquid water become when it freezes?\nAnswer:",
            "ice",
            ("steam", "sand", "fire"),
        ),
        (
            "hot_opposite",
            "Question: What is the opposite of hot?\nAnswer:",
            "cold",
            ("warm", "large", "fast"),
        ),
        (
            "child_plural",
            "Question: What is the plural of child?\nAnswer:",
            "children",
            ("childs", "childes", "child"),
        ),
        (
            "grass_color",
            "Question: What color is healthy grass usually?\nAnswer:",
            "green",
            ("blue", "black", "purple"),
        ),
        (
            "thirsty",
            "Question: Complete: I drink water because I am ___.\nAnswer:",
            "thirsty",
            ("square", "loud", "wooden"),
        ),
        (
            "past_go",
            "Question: What is the past tense of go?\nAnswer:",
            "went",
            ("goed", "gone", "going"),
        ),
        (
            "book_reader",
            "Question: In 'Maya reads the book', who reads?\nAnswer:",
            "Maya",
            ("the book", "the desk", "the room"),
        ),
        (
            "sun_rises",
            "Question: The sun appears to rise in which direction?\nAnswer:",
            "east",
            ("west", "north", "down"),
        ),
        (
            "bird",
            "Question: Which animal normally has feathers?\nAnswer:",
            "bird",
            ("fish", "snake", "whale"),
        ),
        (
            "oxygen",
            "Question: Humans need which gas for respiration?\nAnswer:",
            "oxygen",
            ("helium", "neon", "argon"),
        ),
        (
            "triangle",
            "Question: How many sides does a triangle have?\nAnswer:",
            "three",
            ("two", "four", "five"),
        ),
        (
            "month",
            "Question: Which month comes after March?\nAnswer:",
            "April",
            ("January", "February", "December"),
        ),
        (
            "synonym_fast",
            "Question: Which word is closest in meaning to fast?\nAnswer:",
            "quick",
            ("slow", "silent", "heavy"),
        ),
        (
            "antonym_early",
            "Question: Which word is the opposite of early?\nAnswer:",
            "late",
            ("soon", "first", "ready"),
        ),
        ("article", "Question: Choose the article: ___ apple.\nAnswer:", "an", ("a", "thee", "am")),
        ("verb", "Question: Which word is a verb?\nAnswer:", "run", ("blue", "table", "quiet")),
        (
            "noun",
            "Question: Which word is a noun?\nAnswer:",
            "river",
            ("quickly", "under", "bright"),
        ),
        (
            "past_walk",
            "Question: What is the regular past tense of walk?\nAnswer:",
            "walked",
            ("walks", "walking", "walker"),
        ),
        (
            "comparative",
            "Question: What is the comparative form of small?\nAnswer:",
            "smaller",
            ("smallest", "smallly", "more smallest"),
        ),
        (
            "planet",
            "Question: Earth is a ___.\nAnswer:",
            "planet",
            ("star", "galaxy", "comet tail"),
        ),
        (
            "photosynthesis",
            "Question: Plants use sunlight in which process?\nAnswer:",
            "photosynthesis",
            ("evaporation", "erosion", "combustion"),
        ),
        (
            "capital_france",
            "Question: What is the capital of France?\nAnswer:",
            "Paris",
            ("Rome", "Madrid", "Berlin"),
        ),
        (
            "ocean",
            "Question: Which is an ocean?\nAnswer:",
            "Pacific",
            ("Sahara", "Alps", "Amazon city"),
        ),
        (
            "metal",
            "Question: Which material conducts electricity well?\nAnswer:",
            "copper",
            ("rubber", "dry wood", "glass"),
        ),
        (
            "computer_memory",
            "Question: RAM is a type of computer ___.\nAnswer:",
            "memory",
            ("keyboard", "network cable", "screen color"),
        ),
        (
            "browser",
            "Question: A web browser is used to view ___.\nAnswer:",
            "web pages",
            ("engine oil", "paper mail", "radio waves only"),
        ),
        (
            "database",
            "Question: A database stores organized ___.\nAnswer:",
            "data",
            ("weather", "electricity", "paint"),
        ),
        (
            "accurate",
            "Question: Which word means correct and precise?\nAnswer:",
            "accurate",
            ("vague", "random", "broken"),
        ),
        (
            "uncertain",
            "Question: If evidence is missing, a careful answer should express ___.\nAnswer:",
            "uncertainty",
            ("absolute certainty", "a password", "a threat"),
        ),
        (
            "because",
            "Question: Which word commonly introduces a reason?\nAnswer:",
            "because",
            ("although", "where", "unless"),
        ),
        (
            "if_then",
            "Question: In logic, 'if A, then B' states a ___.\nAnswer:",
            "conditional",
            ("color", "measurement unit", "file"),
        ),
        (
            "summary",
            "Question: A summary should contain the main ___.\nAnswer:",
            "ideas",
            ("passwords", "pixels", "timestamps only"),
        ),
    ]
    reasoning = [
        (
            "unknown",
            "问题：证据不足时最合适的做法是什么？\n答案：",
            "说明不确定并查证",
            ("编造答案", "忽略问题", "宣称绝对正确"),
        ),
        (
            "holdout",
            "问题：测试集的主要用途是什么？\n答案：",
            "最终独立评估",
            ("更新模型参数", "存放密码", "替代训练集"),
        ),
        (
            "single_variable",
            "问题：受控实验一次改变一个主要变量是为了什么？\n答案：",
            "便于归因",
            ("增加文件数量", "隐藏失败", "跳过评测"),
        ),
        (
            "loss_only",
            "问题：只看训练 loss 选择模型有什么风险？\n答案：",
            "不能代表完整能力",
            ("显存自动增加", "数据自动合法", "测试必然通过"),
        ),
        (
            "shared_gpu",
            "问题：共享 GPU 上启动任务前应该先做什么？\n答案：",
            "检查现有占用和总显存预算",
            ("终止他人进程", "删除日志", "关闭监控"),
        ),
        (
            "oom",
            "问题：避免 GPU OOM 最直接需要约束什么？\n答案：",
            "所有任务峰值显存总和",
            ("用户名长度", "提交信息", "文件扩展名"),
        ),
        (
            "checkpoint",
            "问题：验证精确恢复应比较什么？\n答案：",
            "步数、游标、优化器和随机状态",
            ("只有文件大小", "只有进程名", "只有时间"),
        ),
        (
            "hash",
            "问题：数据哈希变化通常说明什么？\n答案：",
            "数据身份发生变化",
            ("模型必然更好", "GPU 必然空闲", "许可证自动通过"),
        ),
        (
            "bootstrap",
            "问题：配对 bootstrap 比较同一题集候选时保留了什么？\n答案：",
            "逐题配对关系",
            ("GPU 温度", "文件权限", "训练进程 PID"),
        ),
        (
            "mcnemar",
            "问题：McNemar 检验适合比较什么？\n答案：",
            "同一批样本上的两种分类结果",
            ("两个磁盘容量", "两种字体", "两个端口"),
        ),
        (
            "seed",
            "问题：第二个独立 seed 的主要作用是什么？\n答案：",
            "检查结论稳定性",
            ("增加词表", "改变许可证", "压缩 checkpoint"),
        ),
        (
            "leakage",
            "问题：训练集包含测试题会导致什么？\n答案：",
            "评测污染",
            ("显存降低", "网络更快", "代码更短"),
        ),
        (
            "baseline",
            "问题：比较候选前为什么要冻结 baseline？\n答案：",
            "建立可归因参照",
            ("减少数据哈希", "删除历史", "绕过测试"),
        ),
        (
            "trigger",
            "问题：pilot 未达到预设触发线时应该怎么办？\n答案：",
            "停止该方向",
            ("无限扩训", "修改结果", "删除阈值"),
        ),
        (
            "identity",
            "问题：完整 run identity 不应缺少什么？\n答案：",
            "代码、数据、配置与 seed",
            ("编辑器主题", "鼠标型号", "窗口位置"),
        ),
        (
            "atomic",
            "问题：checkpoint 原子写入防止什么？\n答案：",
            "中断留下半写文件",
            ("模型过拟合", "网络延迟", "tokenizer 变大"),
        ),
        (
            "license",
            "问题：能力门通过但数据许可未澄清时可以公开权重吗？\n答案：",
            "不可以",
            ("可以且无需说明", "只要 loss 低就可以", "自动可以"),
        ),
        (
            "domain",
            "问题：同域 loss 更低能否直接证明通用能力更强？\n答案：",
            "不能",
            ("一定能", "与数据无关", "只由显存决定"),
        ),
        (
            "tokens",
            "问题：公平比较训练预算时优先固定什么？\n答案：",
            "seen tokens",
            ("日志行数", "文件名长度", "终端宽度"),
        ),
        (
            "resume",
            "问题：恢复训练后数据游标不连续会造成什么？\n答案：",
            "样本重复或跳过",
            ("许可证变化", "GPU 数量增加", "模型层数减少"),
        ),
        (
            "registry",
            "问题：候选注册表的主要作用是什么？\n答案：",
            "追踪模型身份和证据",
            ("替代训练数据", "保存用户密码", "控制显示器"),
        ),
        (
            "fail_closed",
            "问题：缺少关键评测证据时门禁应该怎样？\n答案：",
            "拒绝通过",
            ("默认通过", "随机决定", "只看文件名"),
        ),
        (
            "disk",
            "问题：训练前检查磁盘空间是为了防止什么？\n答案：",
            "checkpoint 写入失败",
            ("注意力退化", "词表重复", "HTTP 重定向"),
        ),
        (
            "monitor",
            "问题：长时间无训练进展应该触发什么？\n答案：",
            "告警和诊断",
            ("公开发布", "删除数据", "提高阈值"),
        ),
        (
            "retry",
            "问题：任务重试要避免覆盖已有正式 run，需要什么性质？\n答案：",
            "幂等性",
            ("随机性", "匿名性", "模糊性"),
        ),
        (
            "parent",
            "问题：continuation 实验必须记录什么？\n答案：",
            "父 checkpoint 身份",
            ("桌面背景", "SSH 密码", "浏览器历史"),
        ),
        (
            "mixture",
            "问题：混合数据实验必须新增什么？\n答案：",
            "混合比例和新 manifest",
            ("旧结果覆盖", "相同 run ID", "隐藏采样策略"),
        ),
        (
            "capacity",
            "问题：并行 GPU 任务安全启动的充分依据是什么？\n答案：",
            "实时占用加预留峰值低于容量",
            ("GPU 利用率为零一次", "没有终端窗口", "进程名不同"),
        ),
        (
            "statistical",
            "问题：小题集上相差一两题时最稳妥的结论是什么？\n答案：",
            "差异可能不稳定",
            ("候选必然更强", "立即发布", "忽略置信区间"),
        ),
        (
            "repetition",
            "问题：生成文本大量重复通常属于什么信号？\n答案：",
            "退化信号",
            ("许可通过信号", "磁盘健康信号", "数据身份"),
        ),
        (
            "clean_rebuild",
            "问题：干净环境重建主要验证什么？\n答案：",
            "依赖与资产可恢复",
            ("训练 loss 必然降低", "模型参数自动增加", "网络永不失败"),
        ),
        (
            "provenance",
            "问题：模型血缘需要能追溯到什么？\n答案：",
            "数据、源码、配置和 checkpoint",
            ("屏幕分辨率", "键盘语言", "登录头像"),
        ),
    ]
    items = _authored_items("zh_fact", facts)
    items.extend(_authored_items("code", code))
    items.extend(_authored_items("english", english))
    items.extend(_authored_items("calibration_reasoning", reasoning))

    for index in range(32):
        left = 7 + index
        right = 3 + (index % 9)
        correct = left * right
        items.append(
            _item(
                item_id=f"math_multiply_{index:02d}",
                category="math",
                prompt=f"问题：{left} × {right} 等于多少？\n答案：",
                correct=str(correct),
                distractors=(str(correct + right), str(correct - left), str(left + right)),
                answer=index % 4,
                source="deterministic-arithmetic-v2",
            )
        )
    names = ("甲", "乙", "丙", "丁")
    weekdays = ("周一", "周二", "周三", "周四")
    for index in range(32):
        rotation = index % 4
        ordered_days = weekdays[rotation:] + weekdays[:rotation]
        target = index % 4
        facts_text = "，".join(
            f"样品{names[position]}在{ordered_days[position]}入库" for position in range(4)
        )
        correct_day = ordered_days[target]
        other_days = [day for day in weekdays if day != correct_day]
        items.append(
            _item(
                item_id=f"long_context_recall_{index:02d}",
                category="long_context",
                prompt=(
                    f"阅读批次 {100 + index} 的记录：{facts_text}。"
                    f"\n问题：样品{names[target]}何时入库？\n答案："
                ),
                correct=correct_day,
                distractors=(other_days[0], other_days[1], other_days[2]),
                answer=index % 4,
                source="deterministic-context-v2",
            )
        )
    return sorted(
        items, key=lambda item: (CATEGORIES.index(str(item["category"])), str(item["id"]))
    )


def validate_items(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate size, uniqueness, schema, and per-domain answer balance."""
    if len(items) != 192:
        raise ValueError(f"expected 192 benchmark items, got {len(items)}")
    ids = [str(item["id"]) for item in items]
    prompts = [_normalized(str(item["prompt"])) for item in items]
    if len(set(ids)) != len(ids) or len(set(prompts)) != len(prompts):
        raise ValueError("benchmark IDs and normalized prompts must be unique")
    category_counts = Counter(str(item["category"]) for item in items)
    if category_counts != Counter(dict.fromkeys(CATEGORIES, 32)):
        raise ValueError(f"unexpected category counts: {category_counts}")
    answer_counts: dict[str, Counter[int]] = {}
    for item in items:
        if len(item["choices"]) != 4 or item["answer"] not in range(4):
            raise ValueError(f"invalid choices for {item['id']}")
        answer_counts.setdefault(str(item["category"]), Counter())[int(item["answer"])] += 1
    for category, counts in answer_counts.items():
        if counts != Counter({0: 8, 1: 8, 2: 8, 3: 8}):
            raise ValueError(f"unbalanced answers for {category}: {counts}")
    return {
        "items": len(items),
        "categories": dict(sorted(category_counts.items())),
        "answer_positions": {
            category: {str(key): value for key, value in sorted(counts.items())}
            for category, counts in sorted(answer_counts.items())
        },
    }


def contamination_audit(items: list[dict[str, Any]], training_paths: list[Path]) -> dict[str, Any]:
    """Check exact normalized benchmark-prompt collisions in audited training views."""
    if not training_paths:
        return {
            "method": "exact SHA-256 over NFKC whitespace-normalized prompt and training text",
            "training_rows_scanned": 0,
            "collisions": [],
            "passed": False,
            "status": "not_run",
        }
    prompt_hashes = {
        hashlib.sha256(_normalized(str(item["prompt"])).encode()).hexdigest(): str(item["id"])
        for item in items
    }
    collisions: list[dict[str, Any]] = []
    rows_scanned = 0
    for path in training_paths:
        for line_number, row in iter_jsonl(path):
            text = row.get("text")
            if not isinstance(text, str):
                continue
            rows_scanned += 1
            digest = hashlib.sha256(_normalized(text).encode()).hexdigest()
            if digest in prompt_hashes:
                collisions.append(
                    {
                        "benchmark_id": prompt_hashes[digest],
                        "training_path": str(path),
                        "line_number": line_number,
                    }
                )
    return {
        "method": "exact SHA-256 over NFKC whitespace-normalized prompt and training text",
        "training_rows_scanned": rows_scanned,
        "collisions": collisions,
        "passed": not collisions,
        "status": "passed" if not collisions else "failed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("configs/evaluation/pretrain_mcq_benchmark_v2.jsonl"),
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=Path("configs/evaluation/pretrain_mcq_benchmark_v2.audit.json"),
    )
    parser.add_argument("--training-view", type=Path, action="append", default=[])
    parser.add_argument(
        "--human-reviewed",
        action="store_true",
        help="Record that answer correctness, ambiguity, and category assignment were reviewed",
    )
    args = parser.parse_args()

    items = build_items()
    validation = validate_items(items)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
    audit = {
        "schema_version": 1,
        "benchmark_sha256": sha256_file(args.output),
        "validation": validation,
        "contamination": contamination_audit(items, args.training_view),
        "human_review": {
            "status": "completed" if args.human_reviewed else "maintainer_review_required",
            "sample_size_minimum": 32,
            "scope": "answer correctness, ambiguity, and domain assignment",
        },
    }
    write_json_atomic(args.audit_output, audit)
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
