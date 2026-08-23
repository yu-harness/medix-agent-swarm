#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成多轮指代评测集 multiturn_50.jsonl。"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "eval" / "data" / "multiturn_50.jsonl"

# id, turn1, turn2, dependency, category, ok_criteria
RAW = [
    ("mt_hc_01", "我有高血压，平时要注意什么？", "那饮食方面呢？", "高血压", "health_consult", "第2轮回答须明确围绕高血压饮食，而非泛泛谈饮食"),
    ("mt_hc_02", "糖尿病患者血糖控制不好怎么办？", "运动可以怎么安排？", "糖尿病/血糖控制", "health_consult", "第2轮须结合糖尿病谈运动，不能答成无关健身"),
    ("mt_hc_03", "孩子经常感冒，怎么提高抵抗力？", "饮食上有什么建议？", "儿童反复感冒/提高抵抗力", "health_consult", "回答须针对儿童抵抗力/反复感冒的饮食建议"),
    ("mt_hc_04", "孕期可以喝咖啡吗？", "那茶呢？", "孕期饮品/咖啡语境", "health_consult", "须在孕期安全语境下谈茶，不能忽略怀孕"),
    ("mt_hc_05", "失眠怎么办，晚上很难入睡", "有没有不吃药的办法？", "失眠", "health_consult", "第2轮须针对失眠的非药物方法"),
    ("mt_hc_06", "痛风发作了脚很疼", "最近能吃海鲜吗？", "痛风", "health_consult", "须结合痛风谈海鲜限制，不能只谈营养"),
    ("mt_hc_07", "胃溃疡在吃药治疗", "饮食禁忌有哪些？", "胃溃疡", "health_consult", "饮食禁忌须对应胃溃疡"),
    ("mt_hc_08", "过敏性鼻炎一到春天就犯", "有什么日常护理建议？", "过敏性鼻炎", "health_consult", "护理建议须针对过敏性鼻炎"),
    ("mt_hc_09", "血脂偏高体检出来的", "要不要先吃药？", "高血脂", "health_consult", "用药建议须对应高血脂语境"),
    ("mt_hc_10", "便秘好几年了", "喝酸奶有用吗？", "慢性便秘", "health_consult", "须在便秘语境评价酸奶"),
    ("mt_hc_11", "甲减在吃优甲乐", "能吃海鲜补碘吗？", "甲减/优甲乐", "health_consult", "须结合甲减与补碘风险回答"),
    ("mt_hc_12", "颈椎病脖子僵硬", "办公室怎么坐姿更好？", "颈椎病", "health_consult", "坐姿建议须针对颈椎病"),
    ("mt_hc_13", "脂肪肝中度", "喝酒要完全戒吗？", "脂肪肝", "health_consult", "须针对脂肪肝谈酒精"),
    ("mt_sd_01", "最近反复头痛，太阳穴胀痛", "会不会是高血压引起的？", "头痛/太阳穴胀痛", "symptom_diagnosis", "须关联头痛主诉讨论与高血压关系"),
    ("mt_sd_02", "咳嗽两周了还有点低热", "要不要去做胸部CT？", "咳嗽伴低热", "symptom_diagnosis", "检查建议须针对咳嗽低热，不能答无关检查"),
    ("mt_sd_03", "胸口闷，活动后更明显", "和心脏病有关系吗？", "胸闷/活动后加重", "symptom_diagnosis", "须围绕胸闷鉴别心脏相关，不能跑题"),
    ("mt_sd_04", "腹泻三天，一天四五次", "需要补液盐吗？", "急性腹泻", "symptom_diagnosis", "补液建议须针对腹泻脱水风险"),
    ("mt_sd_05", "嗓子疼吞咽困难两天", "是不是化脓性扁桃体炎？", "咽痛吞咽困难", "symptom_diagnosis", "须结合当前咽痛症状谈扁桃体炎可能"),
    ("mt_sd_06", "膝盖走路就疼，上下楼更明显", "要拍片子吗？", "膝关节痛", "symptom_diagnosis", "影像建议须针对膝痛"),
    ("mt_sd_07", "皮肤起红疹还痒", "会不会是药物过敏？", "皮疹瘙痒", "symptom_diagnosis", "须关联皮疹讨论药物过敏可能"),
    ("mt_sd_08", "心慌手抖出汗", "要查甲状腺吗？", "心慌手抖出汗", "symptom_diagnosis", "检查建议须回指上述症状簇"),
    ("mt_sd_09", "夜尿多口渴", "像不像糖尿病？", "夜尿多口渴", "symptom_diagnosis", "须用当前症状讨论糖尿病可能"),
    ("mt_sd_10", "脚踝扭了一下肿了", "要冷敷还是热敷？", "踝关节扭伤肿胀", "symptom_diagnosis", "冷热敷建议须针对急性扭伤"),
    ("mt_sd_11", "耳鸣右边更明显一个月了", "和颈椎有关吗？", "右耳耳鸣", "symptom_diagnosis", "须围绕耳鸣谈与颈椎关联"),
    ("mt_sd_12", "反酸烧心晚上躺下更重", "要吃奥美拉唑吗？", "反酸烧心/疑似反流", "symptom_diagnosis", "用药讨论须对应反流症状"),
    ("mt_sd_13", "月经推迟十天还恶心", "要先测孕吗？", "月经推迟+恶心", "symptom_diagnosis", "须结合推迟与恶心谈测孕"),
    ("mt_dk_01", "什么是心房颤动？", "它常见并发症有哪些？", "心房颤动", "disease_knowledge", "并发症须明确属于房颤，不能列无关病"),
    ("mt_dk_02", "哮喘和慢性支气管炎有什么区别？", "那治疗原则上呢？", "哮喘 vs 慢支对比", "disease_knowledge", "治疗原则须延续二者鉴别语境"),
    ("mt_dk_03", "乙肝表面抗原阳性是什么意思？", "会传染给家人吗？", "HBsAg阳性/乙肝", "disease_knowledge", "传染性回答须针对乙肝抗原阳性"),
    ("mt_dk_04", "类风湿关节炎是怎么回事？", "和骨关节炎一样吗？", "类风湿关节炎", "disease_knowledge", "对比须以类风湿为参照"),
    ("mt_dk_05", "解释一下胰岛素抵抗", "和2型糖尿病什么关系？", "胰岛素抵抗", "disease_knowledge", "须连接胰岛素抵抗与2型糖尿病"),
    ("mt_dk_06", "幽门螺杆菌是什么？", "一定要根除治疗吗？", "幽门螺杆菌", "disease_knowledge", "根除讨论须针对Hp"),
    ("mt_dk_07", "什么是慢性肾病分期？", "三期大概意味着什么？", "慢性肾病分期", "disease_knowledge", "须解释CKD三期含义"),
    ("mt_dk_08", "带状疱疹的病因是什么？", "会留下神经痛吗？", "带状疱疹", "disease_knowledge", "后遗神经痛须对应带状疱疹"),
    ("mt_dk_09", "介绍一下COPD", "长期吸氧指征是什么？", "COPD", "disease_knowledge", "吸氧指征须针对COPD"),
    ("mt_dk_10", "什么是缺铁性贫血？", "常见实验室指标有哪些？", "缺铁性贫血", "disease_knowledge", "指标须对应缺铁性贫血"),
    ("mt_dk_11", "甲状腺结节怎么分类？", "恶性风险怎么评估？", "甲状腺结节", "disease_knowledge", "风险评估须针对甲状腺结节"),
    ("mt_dk_12", "解释一下心肌梗死", "溶栓和支架怎么选？", "心肌梗死", "disease_knowledge", "治疗选择须在心梗语境"),
    ("mt_gr_01", "请根据指南说明成人高血压诊断标准", "家庭血压监测阈值呢？", "高血压诊断/监测指南", "guideline_retrieval", "须给出家庭血压相关指南阈值，承接高血压"),
    ("mt_gr_02", "糖尿病诊疗指南里血糖控制目标是什么？", "老年患者目标有不同吗？", "糖尿病血糖控制目标", "guideline_retrieval", "须在糖尿病指南语境谈老年目标"),
    ("mt_gr_03", "请检索哮喘急性发作处理要点", "什么情况需要住院？", "哮喘急性发作", "guideline_retrieval", "住院指征须对应哮喘急性发作"),
    ("mt_gr_04", "指南中慢性心衰的基础用药有哪些？", "合并房颤时要注意什么？", "慢性心衰用药", "guideline_retrieval", "须在心衰基础上谈合并房颤"),
    ("mt_gr_05", "请说明社区获得性肺炎的诊断要点", "抗菌药物初始经验怎么选？", "社区获得性肺炎", "guideline_retrieval", "抗菌选择须针对CAP"),
    ("mt_gr_06", "指南对脑卒中二级预防有哪些建议？", "抗血小板药怎么用？", "脑卒中二级预防", "guideline_retrieval", "抗血小板须在卒中二级预防语境"),
    ("mt_gr_07", "请给出慢性肾脏病血压控制目标", "蛋白尿明显时目标一样吗？", "CKD血压目标", "guideline_retrieval", "须针对CKD/蛋白尿谈血压目标"),
    ("mt_gr_08", "检索胃食管反流病的生活方式建议", "抑酸疗程一般多久？", "GERD指南", "guideline_retrieval", "疗程须对应GERD抑酸治疗"),
    ("mt_gr_09", "请说明成人发热门诊分诊的危险征象", "儿童标准一样吗？", "发热危险征象", "guideline_retrieval", "须区分或说明儿童与成人差异，承接发热分诊"),
    ("mt_gr_10", "指南里骨质疏松诊断的骨密度标准？", "药物治疗适应证呢？", "骨质疏松", "guideline_retrieval", "药物适应证须承接骨质疏松诊断"),
    ("mt_gr_11", "请根据指南说明血脂异常危险分层", "他汀强度怎么选？", "血脂异常危险分层", "guideline_retrieval", "他汀选择须基于前述分层"),
    ("mt_mx_01", "我妈有高血压和糖尿病", "她最近头晕，优先查什么？", "母亲：高血压+糖尿病+头晕", "health_consult", "检查建议须结合其母慢病与头晕"),
]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    assert len(RAW) == 50, len(RAW)
    with OUT.open("w", encoding="utf-8") as f:
        for id_, t1, t2, dep, cat, ok in RAW:
            row = {
                "id": id_,
                "turn1": t1,
                "turn2": t2,
                "dependency": dep,
                "category": cat,
                "ok_criteria": ok,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("wrote", OUT, "n=", len(RAW))


if __name__ == "__main__":
    main()
