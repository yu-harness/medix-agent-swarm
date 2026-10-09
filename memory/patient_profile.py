"""
患者档案（PatientProfile）：跨会话的确定性医疗事实（facts）

对应设计：长期记忆 · L3 关键事实层
- 生命周期：月/年级（独立于短期记忆的 1 小时 TTL）
- 内容：过敏史 / 当前用药 / 既往病史（结构化，带来源与置信度）
- 来源：对话提取（规则初筛 → LLM 结构化提取 → 规则校验）
- 白名单：allergy / current_medication 无条件强制注入——
  因为"过敏史与本次问题是否相关"无法提前证明，
  被相关性筛选误杀的代价（用药事故）远大于多带几十 token 的代价。
- 其他类型：按与当前问题的关键词相关性筛选注入

设计原则（医疗正确性底线）：
1. 宁可漏，不可编：value 必须能在源消息原文中找到证据，否则标 unconfirmed
2. 同事实更新而非追加：fact_key = type:value，重复表述只刷新时间
3. 白名单事实无论 confirmed 与否都强制注入（"过敏史不详"本身也要带）
"""
import asyncio
import json
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from loguru import logger

# ===== 事实类型 =====
FACT_TYPES = {
    "allergy": "过敏史",
    "medication": "当前用药",
    "condition": "既往病史",
}
# 白名单：无法提前排除相关性的高危事实 → 无条件强制注入
FORCE_INJECT_TYPES = {"allergy", "medication"}

# 规则初筛关键词（命中才进入 LLM 提取队列，省调用）
_PREFILTER_KEYWORDS = {
    "allergy": ("过敏", "阿司匹林", "青霉素", "磺胺"),
    "medication": ("在吃", "正在吃", "服用", "口服", "吃药", "用药", "剂量"),
    "condition": ("查出", "确诊", "患有", "病史", "得过", "检查出来", "诊断"),
}

# LLM 提取失败时的规则降级（只做最可靠的正则；宁可漏，不可错）
_RULE_FALLBACK = {
    # 只匹配明确"对X过敏"结构；"我青霉素过敏"这类无"对"的漏掉，交给 LLM 主路径
    "allergy": re.compile(r"对([\u4e00-\u9fa5A-Za-z0-9]{1,10}?)过敏"),
    "medication": re.compile(
        r"(?:在|正在)?(?:吃|服用|口服)([\u4e00-\u9fa5A-Za-z0-9]{2,12})"
    ),
    # "确诊/查出/患有 + 病名"；贪婪匹配停在非汉字（逗号/句号）前
    "condition": re.compile(
        r"(?:确诊|查出|患有)(?:了|有)?([\u4e00-\u9fa5A-Za-z0-9]{2,10})"
    ),
}

# 规则提取值的后处理：按常见剂量/标点截断，避免把"每天一次"等带进 value
_TRIM_SEPS = ("每天", "每日", "一天", "一次", "一周", "，", "。", "；", ",", ";", " ")  # noqa: E501

_EXTRACT_PROMPT = """你是医疗信息抽取器。从下面的用户消息中抽取【用户明确表述】的确定性医疗事实。
只抽取三种类型：
- allergy（过敏史）：用户明确说过敏什么
- medication（当前用药）：用户明确说正在吃什么药
- condition（既往病史）：用户明确说确诊/查出/患有什么病

要求：
1. value 必须是用户原话中出现过的词或短语，禁止概括、禁止添加原文没有的信息
2. 不确定就不抽取
3. 只输出 JSON 数组，不要任何其他文字或解释
4. 输出格式：[{"type": "allergy|medication|condition", "value": "...", "evidence": "原文中的证据片段"}]

用户消息：
{content}"""


class PatientProfile:
    """患者档案：跨会话确定性事实的存储、提取与注入。"""

    def __init__(
        self,
        base_dir: Optional[str] = None,
        llm_client: Optional[Any] = None,
    ):
        # 默认存到项目内 memory/patient_profiles（基于模块位置，不依赖 cwd）
        if base_dir is None:
            base_dir = str(Path(__file__).resolve().parent / "patient_profiles")
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.llm_client = llm_client
        # 并发安全：Skill 线程池 / 多请求并发读写同一档案文件
        self._lock = threading.Lock()

    # ===== 存储 =====

    def _profile_path(self, patient_id: str) -> Path:
        return self.base_dir / f"{patient_id}.json"

    def load(self, patient_id: str) -> Dict[str, Any]:
        """读取患者档案（facts dict，key = "type:value"）。"""
        if not patient_id:
            return {}
        path = self._profile_path(patient_id)
        try:
            if path.exists():
                with self._lock:
                    return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Load patient profile failed ({patient_id}): {e}")
        return {}

    def _save(self, patient_id: str, facts: Dict[str, Any]) -> None:
        if not patient_id:
            return
        path = self._profile_path(patient_id)
        tmp = path.with_suffix(".json.tmp")
        try:
            with self._lock:
                tmp.write_text(
                    json.dumps(facts, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                tmp.replace(path)
        except Exception as e:
            logger.warning(f"Save patient profile failed ({patient_id}): {e}")

    def get_facts(self, patient_id: str) -> List[Dict[str, Any]]:
        """返回该患者全部事实（按更新时间倒序）。"""
        facts = self.load(patient_id)
        items = list(facts.values())
        items.sort(key=lambda f: f.get("updated_at") or "", reverse=True)
        return items

    # ===== 提取 =====

    def _prefilter(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """规则初筛：只留命中关键词的 user 消息，减少 LLM 调用。"""
        candidates = []
        for m in messages or []:
            if m.get("role") != "user":
                continue
            content = str(m.get("content") or "")
            if len(content) > 500:  # 过长消息多为系统注入，跳过
                continue
            hit = any(kw in content for kws in _PREFILTER_KEYWORDS.values() for kw in kws)
            if hit:
                candidates.append(m)
        return candidates

    @staticmethod
    def _validate_fact(fact: Dict[str, Any], content: str) -> bool:
        """规则校验：value 必须能在源消息原文中找到证据。"""
        value = str(fact.get("value") or "").strip()
        evidence = str(fact.get("evidence") or "").strip()
        if not value:
            return False
        # 证据必须来自原文（evidence 应包含 value，且都来自 content）
        if evidence and evidence in content and value in evidence:
            return True
        return value in content

    async def _llm_extract(self, contents: List[str]) -> List[Dict[str, Any]]:
        """LLM 结构化提取。失败返回空列表（调用方走规则降级）。"""
        if not self.llm_client:
            return []
        try:
            result = await asyncio.wait_for(
                self.llm_client.chat(
                    [
                        {"role": "system", "content": _EXTRACT_PROMPT.split("用户消息：")[0]},
                        {"role": "user", "content": "用户消息：\n" + "\n---\n".join(contents)},
                    ],
                    temperature=0.0,
                ),
                timeout=2.0,
            )
            text = (result or "").strip()
            # 提取 JSON 数组（容忍前后有文字包裹）
            start, end = text.find("["), text.rfind("]")
            if start == -1 or end == -1:
                return []
            data = json.loads(text[start:end + 1])
            facts = []
            for item in data if isinstance(data, list) else []:
                ftype = str(item.get("type") or "").strip().lower()
                if ftype in FACT_TYPES:
                    facts.append({
                        "type": ftype,
                        "value": str(item.get("value") or "").strip(),
                        "evidence": str(item.get("evidence") or "").strip(),
                    })
            return facts
        except Exception as e:
            logger.debug(f"LLM fact extraction failed (fallback to rules): {e}")
            return []

    @staticmethod
    def _trim_value(value: str) -> str:
        """按常见剂量/标点截断规则提取值。"""
        for sep in _TRIM_SEPS:
            idx = value.find(sep)
            if idx != -1:
                value = value[:idx]
        return value.strip()

    @classmethod
    def _rule_fallback_extract(cls, content: str) -> List[Dict[str, Any]]:
        """规则降级提取：只做最可靠的正则，标 unconfirmed。"""
        facts = []
        for ftype, pattern in _RULE_FALLBACK.items():
            for m in pattern.finditer(content):
                value = cls._trim_value(m.group(1) or "")
                if len(value) >= 2:
                    facts.append({
                        "type": ftype,
                        "value": value,
                        "evidence": m.group(0),
                    })
        return facts

    async def extract_and_update(
        self,
        patient_id: str,
        messages: List[Dict[str, Any]],
    ) -> int:
        """从对话消息提取事实并合并入库（add/update 语义）。返回新增条数。

        - 规则初筛 → LLM 提取 → 规则校验 → 合并
        - LLM 失败自动降级为规则提取（unconfirmed）
        """
        candidates = self._prefilter(messages)
        if not candidates:
            return 0

        contents = [str(m.get("content") or "") for m in candidates]
        llm_facts = await self._llm_extract(contents)

        # 校验：每一条 value 必须能在其来源消息中找到证据
        all_content = "\n".join(contents)
        validated = []
        for f in llm_facts:
            if self._validate_fact(f, all_content):
                f["status"] = "confirmed"
                validated.append(f)
            else:
                logger.debug(
                    f"Fact rejected (no evidence in source): "
                    f"{f.get('type')}={f.get('value')}"
                )

        # LLM 未产出任何通过校验的事实 → 规则降级（unconfirmed）
        if not validated:
            for f in self._rule_fallback_extract(all_content):
                f["status"] = "unconfirmed"
                validated.append(f)

        if not validated:
            return 0

        facts = self.load(patient_id)
        now = datetime.now().isoformat(timespec="seconds")
        added = 0
        for f in validated:
            ftype, value = f["type"], str(f["value"]).strip()
            if not value:
                continue
            key = f"{ftype}:{value.lower()}"
            existing = facts.get(key)
            if existing and existing.get("status") == "confirmed":
                # 已确认事实不因规则降级而覆盖；仅刷新时间与证据
                existing["updated_at"] = now
                existing["evidence"] = f.get("evidence") or existing.get("evidence")
                continue
            facts[key] = {
                "type": ftype,
                "value": value,
                "status": f.get("status", "unconfirmed"),
                "source": "conversation_extract",
                "evidence": f.get("evidence") or "",
                "updated_at": now,
            }
            added += 1

        if added or True:
            self._save(patient_id, facts)
        return added

    # ===== 注入 =====

    @staticmethod
    def _is_related(fact: Dict[str, Any], question: str) -> bool:
        """相关性筛选：value/类型标签命中当前问题关键词。"""
        if not question:
            return False
        q = question.lower()
        value = str(fact.get("value") or "").lower()
        return bool(value) and (value in q or q in value)

    def build_injection_block(self, patient_id: str, question: str = "") -> str:
        """构建注入上下文前的【患者档案】区块字符串。

        - 白名单类型（过敏史/当前用药）：无条件强制注入
        - 其他类型（既往病史）：按与当前问题的相关性筛选
        - 未确认事实带"（待确认）"标记，不静默冒充已确认
        """
        facts = self.get_facts(patient_id)
        if not facts:
            return ""

        forced = [f for f in facts if f["type"] in FORCE_INJECT_TYPES]
        related = [
            f for f in facts
            if f["type"] not in FORCE_INJECT_TYPES and self._is_related(f, question)
        ]
        selected = forced + related
        if not selected:
            return ""

        labels = FACT_TYPES
        parts = []
        for f in selected:
            text = f"{labels.get(f['type'], f['type'])}：{f['value']}"
            if f.get("status") != "confirmed":
                text += "（待确认）"
            parts.append(text)
        return "【患者档案】" + " | ".join(parts)
