#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.api import VolcClient
from common.config import GLOBAL_API_WORKERS, PROFILE_DIR, ensure_dirs
from common.io import read_json, write_json
from common.prompts import PROFILE_SYSTEM, PROFILE_USER

REQUIRED = ["name", "age", "gender", "residence", "occupation", "education", "long_term_goals", "style", "personality", "preferences", "skills", "beliefs", "behaviors", "health", "relationships", "other_facts"]
LATIN = re.compile(r"[A-Za-z]")
HAN = re.compile(r"[\u4e00-\u9fff]")
BAD_SCRIPT = re.compile(r"[\u3040-\u30ff\uac00-\ud7af\u0400-\u04ff\u0370-\u03ff\u0600-\u06ff\u0e00-\u0e7f]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=1)
    return parser.parse_args()


def profile_id(index: int) -> str:
    return f"{index:04d}"


def strings(payload: Any, key: str | None = None):
    if isinstance(payload, dict):
        for child_key, value in payload.items():
            yield from strings(value, str(child_key))
    elif isinstance(payload, list):
        for value in payload:
            yield from strings(value, key)
    elif isinstance(payload, str) and key != "profile_id":
        yield payload


def persona(row: dict[str, Any]) -> dict[str, Any]:
    item = row.get("persona", row)
    if not isinstance(item, dict):
        raise RuntimeError("profile persona is not object")
    return item


def list_len(item: Any) -> int:
    return len(item) if isinstance(item, list) else 0


def check_profile(row: dict[str, Any]) -> None:
    item = persona(row)
    missing = [key for key in REQUIRED if key not in item]
    if missing:
        raise RuntimeError(f"profile missing keys: {missing}")
    name = str(item["name"])
    if not HAN.search(name) or LATIN.search(name):
        raise RuntimeError(f"bad Chinese name: {name}")
    if not isinstance(item.get("occupation"), dict):
        raise RuntimeError("occupation is not object")
    if list_len(item.get("long_term_goals")) < 4:
        raise RuntimeError("too few long_term_goals")
    if list_len(item.get("skills")) < 8 or list_len(item.get("beliefs")) < 16:
        raise RuntimeError("profile skills/beliefs too short")
    if list_len(item.get("relationships")) < 4 or list_len(item.get("other_facts")) < 18:
        raise RuntimeError("profile relationships/other_facts too short")
    personality = item.get("personality")
    preferences = item.get("preferences")
    behaviors = item.get("behaviors")
    if not isinstance(personality, dict) or list_len(personality.get("traits")) < 8:
        raise RuntimeError("personality traits too short")
    if not isinstance(preferences, dict) or list_len(preferences.get("likes")) < 14 or list_len(preferences.get("dislikes")) < 8:
        raise RuntimeError("preferences too short")
    if not isinstance(behaviors, dict) or not isinstance(behaviors.get("routines"), dict):
        raise RuntimeError("behaviors/routines missing")
    for value in strings(row):
        if value in {"TinyPerson"}:
            continue
        if BAD_SCRIPT.search(value) or ("?" * 2) in value:
            raise RuntimeError(f"third-language or garbled value: {value}")
        if not HAN.search(value) and len(value.strip()) > 0:
            raise RuntimeError(f"value has no Chinese characters: {value}")


PROFILE_CONTEXTS = [
    {"prompt": "29岁左右，女，四川成都，软件工程师，未婚，与父母催婚和职业发展压力有关。", "gender": "女", "region": "成都", "job": ["软件工程师", "工程师"]},
    {"prompt": "42岁左右，女，河南郑州，初中语文教师，已婚有一个上初中的孩子，照顾老人和教学压力并存。", "gender": "女", "region": "郑州", "job": ["教师", "语文"]},
    {"prompt": "36岁左右，男，广东深圳，外卖站点骑手或小组长，离异，有孩子抚养和收入波动压力。", "gender": "男", "region": "深圳", "job": ["外卖", "骑手", "站点"]},
    {"prompt": "34岁左右，女，北京，互联网大厂产品经理，未婚但有稳定伴侣，面对裁员传闻、房租上涨、父母催回老家和职业倦怠。", "gender": "女", "region": "北京", "job": ["产品经理", "互联网", "大厂"]},
    {"prompt": "31岁左右，女，浙江杭州，三甲医院护士，未婚，夜班、医患沟通和亲密关系压力明显。", "gender": "女", "region": "杭州", "job": ["护士", "医院"]},
    {"prompt": "45岁左右，男，江苏苏州，制造业车间主管，已婚二孩，有房贷和父母健康压力。", "gender": "男", "region": "苏州", "job": ["制造", "车间", "主管"]},
    {"prompt": "27岁左右，女，湖北武汉，电商运营或小店主，租房，收入不稳定，和闺蜜/合伙人有矛盾。", "gender": "女", "region": "武汉", "job": ["电商", "小店", "运营"]},
    {"prompt": "39岁左右，女，山东青岛，会计或财务主管，已婚，夹在公司合规和老板要求之间。", "gender": "女", "region": "青岛", "job": ["会计", "财务"]},
    {"prompt": "24岁左右，男，湖南长沙，考研二战或待业青年，和父母同住，社交退缩但仍有目标。", "gender": "男", "region": "长沙", "job": ["考研", "待业"]},
    {"prompt": "47岁左右，男，上海，证券营业部理财顾问或客户经理，已婚一孩，客户亏损投诉、业绩排名和中年职业安全感压力叠在一起。", "gender": "男", "region": "上海", "job": ["证券", "理财", "客户经理"]},
    {"prompt": "52岁左右，女，辽宁沈阳，社区工作者或街道办工作人员，照顾孙辈和基层事务压力并存。", "gender": "女", "region": "沈阳", "job": ["社区", "街道"]},
    {"prompt": "33岁左右，男，陕西西安，室内设计师或装修项目经理，客户沟通和现金流压力明显。", "gender": "男", "region": "西安", "job": ["设计", "装修"]},
    {"prompt": "30岁左右，女，云南昆明，旅游民宿运营者，淡旺季收入差异大，和家人经营理念不同。", "gender": "女", "region": "昆明", "job": ["民宿", "旅游"]},
    {"prompt": "48岁左右，男，重庆，出租车或网约车司机，慢性病、家庭责任和行业变化压力并存。", "gender": "男", "region": "重庆", "job": ["出租车", "网约车", "司机"]},
    {"prompt": "22岁左右，女，黑龙江哈尔滨，专升本应届生或实习行政助理，刚离开校园，面对就业、租房和异地恋的不确定感。", "gender": "女", "region": "哈尔滨", "job": ["专升本", "实习", "行政"]},
    {"prompt": "30岁左右，男，广东广州，连锁餐饮门店店长，单身，夹在总部巡店、员工流失、顾客差评和自己想开小店的计划之间。", "gender": "男", "region": "广州", "job": ["餐饮", "店长"]},
    {"prompt": "26岁左右，男，福建厦门，跨境电商客服或运营助理，合租生活，英语沟通压力、绩效考核和职业转型焦虑并存。", "gender": "男", "region": "厦门", "job": ["跨境电商", "客服", "运营"]},
    {"prompt": "28岁左右，女，广西南宁，公立幼儿园老师，未婚，家长期待、编制考试和与伴侣城市选择的拉扯明显。", "gender": "女", "region": "南宁", "job": ["幼儿园", "老师", "教师"]},
    {"prompt": "34岁左右，男，内蒙古呼和浩特，奶制品企业质检员或班组技术员，已婚，有孩子入托、夜班轮班和本地人情往来压力。", "gender": "男", "region": "呼和浩特", "job": ["质检", "技术员", "奶制品"]},
    {"prompt": "37岁左右，女，江西南昌，银行网点客户经理，已婚无孩，背负业绩指标、亲戚理财咨询和备孕压力。", "gender": "女", "region": "南昌", "job": ["银行", "客户经理"]},
    {"prompt": "27岁左右，女，广东深圳，芯片公司测试工程师，刚从外包转正，担心技术积累不够，也被高房价、通勤和同龄人比较压着。", "gender": "女", "region": "深圳", "job": ["芯片", "测试工程师", "工程师"]},
    {"prompt": "41岁左右，男，山西太原，煤机或能源设备销售，长期出差，夹在客户回款、公司指标和家庭陪伴不足之间。", "gender": "男", "region": "太原", "job": ["能源", "设备", "销售"]},
    {"prompt": "55岁左右，女，甘肃兰州，国企后勤或食堂管理员，临近退休，担心养老金、子女买房和老伴身体检查结果。", "gender": "女", "region": "兰州", "job": ["国企", "后勤", "食堂"]},
    {"prompt": "46岁左右，男，贵州贵阳，物业项目经理或小区工程主管，已婚，面对业主投诉、维修预算和青春期孩子沟通困难。", "gender": "男", "region": "贵阳", "job": ["物业", "项目经理", "工程主管"]},
    {"prompt": "32岁左右，女，海南海口，免税店导购或旅游零售主管，收入随旺季波动，和外地男友、父母养老安排都有分歧。", "gender": "女", "region": "海口", "job": ["免税店", "导购", "零售"]},
    {"prompt": "43岁左右，女，北京，三甲医院药剂科主管药师，已婚有一孩，科室考核、老人看病资源、孩子升学和夫妻分工都让她长期紧绷。", "gender": "女", "region": "北京", "job": ["药剂科", "药师", "医院"]},
    {"prompt": "59岁左右，男，安徽芜湖，退休前的公交司机或调度员，身体小毛病增多，既想照顾孙辈又不想完全失去自己的生活节奏。", "gender": "男", "region": "芜湖", "job": ["公交", "司机", "调度"]},
    {"prompt": "23岁左右，男，河北保定，汽车零部件厂一线工人或学徒，住员工宿舍，想攒钱学技术但常被班组关系和加班打乱计划。", "gender": "男", "region": "保定", "job": ["零部件", "工人", "学徒"]},
    {"prompt": "38岁左右，女，江苏南通，家纺外贸跟单或小厂合伙人，二孩家庭，订单减少、婆媳边界和现金流压力交织。", "gender": "女", "region": "南通", "job": ["家纺", "外贸", "跟单", "合伙"]},
    {"prompt": "50岁左右，男，浙江义乌，小商品批发档口经营者，夫妻共同经营，面对直播电商冲击、库存积压和亲戚赊账难题。", "gender": "男", "region": "义乌", "job": ["批发", "档口", "小商品"]},
    {"prompt": "31岁左右，女，上海，独立设计工作室合伙人，租住老小区，客户审美反复、合伙分账、社保缴纳和是否继续创业让她摇摆。", "gender": "女", "region": "上海", "job": ["设计", "工作室", "合伙"]},
    {"prompt": "44岁左右，女，广东佛山，陶瓷或家电厂人事主管，已婚，有裁员沟通、员工情绪和自己职业天花板的压力。", "gender": "女", "region": "佛山", "job": ["人事", "陶瓷", "家电"]},
    {"prompt": "61岁左右，女，新疆乌鲁木齐，退休社区医生或卫生服务站返聘人员，关注慢病管理，也为独居母亲和远方子女牵挂。", "gender": "女", "region": "乌鲁木齐", "job": ["医生", "卫生服务站", "返聘"]},
    {"prompt": "35岁左右，男，宁夏银川，新能源电站运维工程师，常驻郊外项目，收入稳定但社交圈窄，和妻子关于是否回市区生活争执。", "gender": "男", "region": "银川", "job": ["新能源", "运维", "工程师"]},
    {"prompt": "39岁左右，男，广东广州，城中村物流网点负责人，已婚二孩，面对快递价格战、场地租金、骑手管理和孩子教育支出的压力。", "gender": "男", "region": "广州", "job": ["物流", "网点", "快递"]},
    {"prompt": "29岁左右，女，吉林长春，汽车主机厂采购专员，未婚，供应商关系、部门内卷和父母希望她考编的压力并存。", "gender": "女", "region": "长春", "job": ["汽车", "采购"]},
    {"prompt": "40岁左右，男，河南洛阳，文旅景区运营或讲解管理人员，旺季忙淡季焦虑，既想守住本地文化工作又担心收入上限。", "gender": "男", "region": "洛阳", "job": ["文旅", "景区", "运营", "讲解"]},
    {"prompt": "53岁左右，女，山东临沂，批发市场会计兼家庭账本管理者，长期照顾公婆，最近因为儿子创业借钱和丈夫意见不合。", "gender": "女", "region": "临沂", "job": ["批发市场", "会计"]},
    {"prompt": "57岁左右，男，北京，建筑央企项目安全负责人，常年跑工地，临近退休仍担心事故责任、年轻人不服管和老家父母照护。", "gender": "男", "region": "北京", "job": ["建筑", "安全", "项目"]},
    {"prompt": "25岁左右，男，青海西宁，基层医院检验科技师，刚入职不久，面对编外身份、科室排班和是否继续考研的纠结。", "gender": "男", "region": "西宁", "job": ["医院", "检验", "技师"]},
    {"prompt": "24岁左右，女，上海，高校实验室科研助理或硕士延期学生，住学校宿舍，论文数据、导师沟通、同门比较和毕业去向都压在一起。", "gender": "女", "region": "上海", "job": ["科研助理", "硕士", "实验室"]},
    {"prompt": "52岁左右，女，广东深圳，家政公司培训督导，从保洁阿姨一路做到管理岗，面对客户投诉、员工流动和自己身体透支。", "gender": "女", "region": "深圳", "job": ["家政", "培训", "督导"]},
]


def profile_context(index: int) -> dict[str, Any]:
    return PROFILE_CONTEXTS[(index - 1) % len(PROFILE_CONTEXTS)]


def check_context(row: dict[str, Any], context: dict[str, Any]) -> None:
    item = persona(row)
    occupation = item.get("occupation", {})
    job_text = " ".join(str(value) for value in occupation.values()) if isinstance(occupation, dict) else str(occupation)
    if context["gender"] not in str(item.get("gender", "")):
        raise RuntimeError(f"profile gender misses context: {context['gender']}")
    if context["region"] not in str(item.get("residence", "")):
        raise RuntimeError(f"profile region misses context: {context['region']}")
    if not any(word in job_text for word in context["job"]):
        raise RuntimeError(f"profile job misses context: {context['job']}")

def build_one(index: int, client: VolcClient, force: bool) -> dict:
    pid = profile_id(index)
    path = PROFILE_DIR / f"{pid}.json"
    if path.exists() and not force:
        row = read_json(path)
        check_profile(row)
        return {"profile_id": pid, "cached": True}
    last_error: Exception | None = None
    for attempt in range(1, 6):
        context = profile_context(index)
        user_prompt = PROFILE_USER + f"\n\n本条 profile 的硬约束：{context['prompt']}\n这些硬约束必须直接体现在 gender、residence 和 occupation 字段里；不要改成互联网产品经理、项目经理或其他默认职业，除非硬约束本来如此。\nprofile_index={index}，attempt={attempt}。请保持中文为主体，整体使用中文，可以保留英文专业词汇，不要第三种语言或乱码问号。所有数组元素都必须是带双引号的 JSON string。"
        try:
            payload, _ = client.chat_json([
                {"role": "system", "content": PROFILE_SYSTEM},
                {"role": "user", "content": user_prompt},
            ], model="deepseek", max_tokens=6144, tag=f"profile:{pid}:try{attempt}")
            if not isinstance(payload, dict):
                raise RuntimeError("profile payload is not object")
            check_profile(payload)
            check_context(payload, context)
        except Exception as exc:
            last_error = exc
            continue
        payload["profile_id"] = pid
        write_json(path, payload)
        return {"profile_id": pid, "cached": False}
    raise RuntimeError(f"profile validation failed after retries: {last_error}")


def main() -> None:
    args = parse_args()
    ensure_dirs()
    client = VolcClient("part1_profile", default_model="deepseek")
    indexes = list(range(args.start_index, args.count + 1))
    workers = min(args.workers or GLOBAL_API_WORKERS, len(indexes)) or 1
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(build_one, index, client, args.force) for index in indexes]
        for future in concurrent.futures.as_completed(futures):
            future.result()
            done += 1
            if done % 50 == 0 or done == len(indexes):
                print(f"profiles {done}/{len(indexes)}", flush=True)


if __name__ == "__main__":
    main()
