# -*- coding: utf-8 -*-
"""
=======================================================================
【1208专属】持仓股票·诺贝尔医学级精准买卖分析系统  ——  终极版 v4.0
=======================================================================

职责（与原版 tor/1314 项目的根本区别）：
  原版 → 全市场「选股」（每天找新票）
  本版 → 持仓「管理」（我已买入，现在该怎么办）

技术框架（11层量化诊断，全部来自严格数学定义）：
  A. DNA底背离六重基因共振系统 v3.0          ← 通达信终极版
  B. 临床级时序靶向底背离确诊                 ← 通达信临床版
  C. 靶向共振四维联合会诊                     ← 通达信靶向版
  D. 三重量化买入强化确认（MACD/超跌/连阴）
  E. 三重量化卖出/止损强制扫描（顶背离/放量破MA20/KDJ死叉）
  F. 一目均衡表（Ichimoku Cloud）持仓安全评估
  G. ATR自适应动态止损线计算
  H. 斐波那契回调位支撑/压力分析
  I. 布林带宽度挤压与扩张诊断
  J. 多周期共振确认（日线×60分钟同向验证）
  K. 量价关系综合诊断（OBV趋势/资金流向）

持仓配置：通过 GitHub Actions Variables 中的 STOCK_LIST 环境变量维护
  格式A（纯代码）：002403,600519
  格式B（含成本价）：002403:8.50,600519:1820.00         ← 强烈推荐
  格式C（完整JSON）：[{"code":"002403","buy_price":8.50,"shares":1000}]

发邮件时间：保持原项目配置，每日三次不变
=======================================================================
"""

import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import litellm
from json_repair import repair_json
from litellm import Router

from src.agent.llm_adapter import get_thinking_extra_body
from src.config import (
    Config,
    extra_litellm_params,
    get_api_keys_for_model,
    get_config,
    get_configured_llm_models,
)
from src.data.stock_mapping import STOCK_NAME_MAP
from src.schemas.report_schema import AnalysisReportSchema
from src.storage import persist_llm_usage

logger = logging.getLogger(__name__)


# =======================================================================
# 【第一层】STOCK_LIST 环境变量解析
# =======================================================================
# GitHub Actions → Settings → Secrets and variables → Actions → Variables
# Name: STOCK_LIST
# Value（三种格式任选，自动识别）：
#
#   格式A（最简）：  002403
#                    或  002403,600519,000858
#
#   格式B（推荐）：  002403:8.50
#                    或  002403:8.50,600519:1820.00
#                    冒号后面填你的买入均价，AI自动计算止损线=买入价×0.92
#
#   格式C（完整）：  [{"code":"002403","buy_price":8.50,"shares":1000,"notes":"邮件推荐"}]
#
# 未配置时自动使用 FALLBACK_STOCKS 兜底，不报错不崩溃
# =======================================================================

FALLBACK_STOCKS: List[Dict[str, Any]] = [
    {
        "code": "002403",
        "name": "爱仕达",
        "buy_price": None,
        "shares": None,
        "buy_date": None,
        "notes": "邮件推荐买入 — 请在GitHub Variables的STOCK_LIST中配置：002403:买入价",
    },
]


def _normalize_json_string(s: str) -> str:
    """
    对 JSON 字符串做多重标准化，修复 GitHub Variables 所有常见编码问题。

    修复清单：
      1.  去除 BOM 和首尾空白
      2.  中文弯引号 \u201c \u201d  →  ASCII 双引号 "
      3.  中文单引号 \u2018 \u2019  →  ASCII 单引号 '
      4.  欧式引号  \u201e \u201f \u2039 \u203a  → 对应直引号
      5.  全角冒号 ／逗号  →  半角 : ,
      6.  全角方括号 【】 ／花括号 ｛｝ ／ ［］  →  半角
      7.  单引号包裹的 JSON 键/值 'key' → "key"
      8.  尾随逗号 ,} 和 ,]  →  } ]（标准 JSON 不允许尾随逗号）
    """
    s = s.strip().lstrip("\ufeff")                            # 1. 去 BOM
    # 2. 弯引号 / 弯单引号 → 直引号
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    s = s.replace("\u2018", "'").replace("\u2019", "'")
    # 4. 欧式引号
    s = s.replace("\u201e", '"').replace("\u201f", '"')
    s = s.replace("\u2039", "'").replace("\u203a", "'")
    # 5. 全角标点 → 半角
    s = s.replace("\uff1a", ":").replace("\uff0c", ",")
    s = s.replace("\uff1b", ";")
    # 6. 全角括号
    s = s.replace("\u3010", "[").replace("\u3011", "]")
    s = s.replace("\uff5b", "{").replace("\uff5d", "}")
    s = s.replace("\uff3b", "[").replace("\uff3d", "]")
    # 7. 单引号包裹的键/值 → 双引号
    s = re.sub(r"(?<![\\])'([^']*)'", r'"\1"', s)
    # 8. 尾随逗号（人手输入常见笔误）
    s = re.sub(r",\s*}", "}", s)
    s = re.sub(r",\s*]", "]", s)
    return s


def _extract_one_item(item: dict) -> Optional[Dict[str, Any]]:
    """
    从一个 JSON 对象中提取持仓字段。
    完全大小写不敏感：code / CODE / Code / 股票代码 均可识别。
    完全字段名容错：buy_price / buyPrice / BUY_PRICE / price / 成本价 均可识别。
    """
    if not isinstance(item, dict):
        return None

    # 构建全小写键映射（保留原值）
    lower_map: Dict[str, Any] = {k.lower(): v for k, v in item.items()}

    # ── 股票代码（必填，找不到则跳过）──────────────────────────────
    code_raw = (
        lower_map.get("code")
        or lower_map.get("股票代码")
        or lower_map.get("symbol")
        or lower_map.get("ticker")
        or lower_map.get("id")
    )
    if not code_raw:
        logger.warning("[STOCK_LIST] JSON条目缺少code字段，已跳过。条目内容: %s", item)
        return None
    code = str(code_raw).strip().lstrip("'\"")

    # ── 买入价（可选）────────────────────────────────────────────────
    buy_price_raw = (
        lower_map.get("buy_price")
        or lower_map.get("buyprice")
        or lower_map.get("price")
        or lower_map.get("cost")
        or lower_map.get("成本价")
        or lower_map.get("买入价")
        or lower_map.get("cost_price")
    )
    buy_price: Optional[float] = None
    if buy_price_raw is not None:
        try:
            buy_price = float(buy_price_raw)
        except (ValueError, TypeError):
            logger.warning("[STOCK_LIST] %s 的 buy_price 无法转为数字：%s，已置为空", code, buy_price_raw)

    # ── 持仓股数（可选）──────────────────────────────────────────────
    shares_raw = (
        lower_map.get("shares")
        or lower_map.get("volume")
        or lower_map.get("qty")
        or lower_map.get("quantity")
        or lower_map.get("持仓股数")
        or lower_map.get("股数")
    )
    shares: Optional[int] = None
    if shares_raw is not None:
        try:
            shares = int(float(shares_raw))
        except (ValueError, TypeError):
            pass

    return {
        "code": code,
        "buy_price": buy_price,
        "shares": shares,
        "buy_date": lower_map.get("buy_date") or lower_map.get("date") or lower_map.get("买入日期"),
        "notes": str(lower_map.get("notes") or lower_map.get("note") or lower_map.get("备注") or ""),
        "name": str(lower_map.get("name") or lower_map.get("股票名称") or ""),
    }


def _try_parse_json(raw: str) -> Optional[List[Dict[str, Any]]]:
    """
    尝试用多种策略解析 JSON 字符串。
    成功返回条目列表，失败返回 None（不抛异常）。

    策略顺序：
      1. 直接 json.loads（最优先，处理标准 JSON）
      2. 标准化后 json.loads（处理弯引号/全角字符）
      3. repair_json 修复后 json.loads（处理尾逗号/缺引号等）
    """
    candidates = [raw, _normalize_json_string(raw)]
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            # 允许单个对象 {} 直接包裹（不强制要求 []）
            if isinstance(parsed, dict):
                parsed = [parsed]
            if isinstance(parsed, list):
                result = [_extract_one_item(it) for it in parsed]
                result = [r for r in result if r is not None]
                if result:
                    return result
        except (json.JSONDecodeError, ValueError):
            pass

    # 最后用 repair_json 兜底
    try:
        fixed = repair_json(_normalize_json_string(raw))
        parsed = json.loads(fixed)
        if isinstance(parsed, dict):
            parsed = [parsed]
        if isinstance(parsed, list):
            result = [_extract_one_item(it) for it in parsed]
            result = [r for r in result if r is not None]
            if result:
                logger.info("[STOCK_LIST] JSON经 repair_json 修复后解析成功")
                return result
    except Exception:
        pass

    return None


def _looks_like_json(s: str) -> bool:
    """判断字符串是否像 JSON（含花括号或方括号）。"""
    stripped = s.strip().lstrip("\ufeff")
    return stripped.startswith(("[", "{", "【", "｛", "\u201c["))


def _parse_stock_list_env(raw: str) -> List[Dict[str, Any]]:
    """
    解析 STOCK_LIST 环境变量。自动识别三种格式，极度容错。

    ─────────────────────────────────────────────────────────────
    格式A（纯代码）：  002403
                       002403,600519,000858
    格式B（代码:价格）：002403:11.398
                        002403:11.398,600519:1820.00
    格式C（完整JSON）：[{"code":"002403","buy_price":11.398,"shares":600}]
                        大小写不敏感，支持中英文键名，支持单对象{}不带[]
    ─────────────────────────────────────────────────────────────

    【关键安全保障】：
    若字符串被判定为 JSON（含{}或[]）但解析失败，
    绝对不会降级用逗号拆分（否则会把 JSON 碎片当成股票代码）。
    而是记录详细错误日志并返回空列表，触发 FALLBACK_STOCKS 兜底。
    """
    raw = raw.strip().lstrip("\ufeff")
    if not raw:
        return []

    logger.info("[STOCK_LIST] 原始值（前200字符）: %s", raw[:200])

    # ── 判断是否像 JSON ──────────────────────────────────────────
    if _looks_like_json(raw):
        result = _try_parse_json(raw)
        if result:
            logger.info(
                "[STOCK_LIST] ✅ 格式C(JSON)解析成功：%d 只持仓 → %s",
                len(result),
                [r["code"] for r in result],
            )
            return result

        # JSON 检测成立但解析全部失败 → 记录详细错误，禁止降级拆分
        logger.error(
            "[STOCK_LIST] ❌ JSON格式检测成立但解析失败！\n"
            "  原始值: %s\n"
            "  常见原因：\n"
            "    1. 键名使用大写但格式不标准（请检查：code/CODE/buy_price/BUY_PRICE均支持）\n"
            "    2. 使用了中文弯引号 " " 而非英文直引号 \" \"（本版本已自动修复，若仍失败请检查）\n"
            "    3. JSON 结构不完整（缺少花括号或方括号）\n"
            "    4. GitHub Variables 对特殊字符做了转义\n"
            "  建议改用格式B避免以上问题：002403:11.398",
            raw[:300],
        )
        logger.error(
            "[STOCK_LIST] ⛔ 已检测到JSON格式，拒绝降级为逗号拆分（防止把JSON碎片当股票代码）"
        )
        return []

    # ── 格式A / 格式B：逗号或换行分隔 ───────────────────────────
    parts = [p.strip() for p in re.split(r"[,\n\r\t]+", raw) if p.strip()]
    result: List[Dict[str, Any]] = []
    for part in parts:
        # 跳过明显不是股票代码的碎片（包含花括号/引号说明是JSON碎片）
        if any(c in part for c in ("{", "}", "[", "]", '"', "'")):
            logger.warning("[STOCK_LIST] 跳过疑似JSON碎片（请改用完整JSON格式）：%s", part)
            continue
        if ":" in part:
            segs = part.split(":", 1)
            code = segs[0].strip()
            try:
                buy_price: Optional[float] = float(segs[1].strip())
            except (ValueError, IndexError):
                buy_price = None
            result.append({"code": code, "buy_price": buy_price, "shares": None, "notes": "", "name": ""})
        else:
            code = part.strip()
            if code:
                result.append({"code": code, "buy_price": None, "shares": None, "notes": "", "name": ""})

    logger.info(
        "[STOCK_LIST] ✅ 格式A/B解析成功：%d 只持仓 → %s",
        len(result),
        [r["code"] for r in result],
    )
    return result


def _load_held_stocks() -> Dict[str, Dict[str, Any]]:
    """
    从 STOCK_LIST 环境变量加载持仓配置。
    解析失败或变量为空时自动使用 FALLBACK_STOCKS 兜底，不崩溃不报错。
    """
    raw = os.environ.get("STOCK_LIST", "").strip()

    if raw:
        parsed = _parse_stock_list_env(raw)
        if parsed:
            stocks: Dict[str, Dict[str, Any]] = {}
            for item in parsed:
                code = item["code"]
                # 股票名称：JSON中填写的 > 静态映射表
                name = item.get("name") or STOCK_NAME_MAP.get(code, "")
                stocks[code] = {
                    "name": name,
                    "buy_price": item.get("buy_price"),
                    "shares": item.get("shares"),
                    "buy_date": item.get("buy_date"),
                    "notes": item.get("notes") or "来自STOCK_LIST",
                }
            logger.info(
                "[STOCK_LIST] 最终持仓列表：%s",
                {
                    c: f"买入价={v.get('buy_price')} 股数={v.get('shares')}"
                    for c, v in stocks.items()
                },
            )
            return stocks

        # parsed 为空（JSON解析失败或格式A/B无有效条目）
        logger.warning(
            "[STOCK_LIST] 解析结果为空，使用 FALLBACK_STOCKS 兜底。\n"
            "  请检查 GitHub Variables 中 STOCK_LIST 的值格式是否正确。\n"
            "  推荐格式B（最简单）：002403:11.398\n"
            "  完整JSON格式C示例：[{\"code\":\"002403\",\"buy_price\":11.398,\"shares\":600,\"notes\":\"爱仕达\"}]"
        )
    else:
        logger.warning(
            "[STOCK_LIST] 环境变量 STOCK_LIST 未配置或为空，使用 FALLBACK_STOCKS 兜底。\n"
            "  配置路径：GitHub仓库 → Settings → Secrets and variables → Actions → Variables → New variable\n"
            "  Name: STOCK_LIST\n"
            "  Value（推荐格式B）：002403:11.398"
        )

    return {
        item["code"]: {k: v for k, v in item.items() if k != "code"}
        for item in FALLBACK_STOCKS
    }


# 进程启动时加载一次
HELD_STOCKS: Dict[str, Dict[str, Any]] = _load_held_stocks()


def get_held_stock_info(code: str) -> Optional[Dict[str, Any]]:
    """获取持仓信息。精确匹配优先，去前导零模糊匹配兜底。"""
    if code in HELD_STOCKS:
        return {"code": code, **HELD_STOCKS[code]}
    clean = code.lstrip("0")
    for k, v in HELD_STOCKS.items():
        if k.lstrip("0") == clean:
            return {"code": k, **v}
    return None


def calculate_pnl(
    current_price: float,
    buy_price: float,
    shares: Optional[int] = None,
) -> Dict[str, Any]:
    """计算持仓盈亏详情。"""
    if not buy_price or not current_price:
        return {}
    pnl_pct = (current_price - buy_price) / buy_price * 100
    stop_loss = round(buy_price * 0.92, 2)
    result: Dict[str, Any] = {
        "buy_price": buy_price,
        "current_price": current_price,
        "pnl_pct": round(pnl_pct, 2),
        "pnl_status": "盈利" if pnl_pct > 0 else ("亏损" if pnl_pct < 0 else "持平"),
        "stop_loss_price": stop_loss,
        "distance_to_stop_pct": round((current_price - stop_loss) / current_price * 100, 2),
        "stop_loss_triggered": current_price <= stop_loss,
    }
    if shares:
        result["pnl_amount"] = round((current_price - buy_price) * shares, 2)
        result["shares"] = shares
    return result


# =======================================================================
# 【第二层】完整性校验与占位补全（保持原版全部逻辑）
# =======================================================================

def check_content_integrity(result: "AnalysisResult") -> Tuple[bool, List[str]]:
    missing: List[str] = []
    if result.sentiment_score is None:
        missing.append("sentiment_score")
    if not (result.operation_advice or "").strip():
        missing.append("operation_advice")
    if not (result.analysis_summary or "").strip():
        missing.append("analysis_summary")
    dash = result.dashboard if isinstance(result.dashboard, dict) else {}
    core = dash.get("core_conclusion") or {}
    if not (core.get("one_sentence") or "").strip():
        missing.append("dashboard.core_conclusion.one_sentence")
    intel = dash.get("intelligence")
    if intel is None or "risk_alerts" not in (intel or {}):
        missing.append("dashboard.intelligence.risk_alerts")
    if result.decision_type in ("buy", "hold"):
        battle = (dash.get("battle_plan") or {})
        sp = (battle.get("sniper_points") or {})
        stop_loss = sp.get("stop_loss")
        if stop_loss is None or (isinstance(stop_loss, str) and not stop_loss.strip()):
            missing.append("dashboard.battle_plan.sniper_points.stop_loss")
    return len(missing) == 0, missing


def apply_placeholder_fill(result: "AnalysisResult", missing_fields: List[str]) -> None:
    for field_name in missing_fields:
        if field_name == "sentiment_score":
            result.sentiment_score = 50
        elif field_name == "operation_advice":
            result.operation_advice = result.operation_advice or "待补充"
        elif field_name == "analysis_summary":
            result.analysis_summary = result.analysis_summary or "待补充"
        elif field_name == "dashboard.core_conclusion.one_sentence":
            if not result.dashboard:
                result.dashboard = {}
            result.dashboard.setdefault("core_conclusion", {})["one_sentence"] = (
                result.dashboard["core_conclusion"].get("one_sentence") or "待补充"
            )
        elif field_name == "dashboard.intelligence.risk_alerts":
            if not result.dashboard:
                result.dashboard = {}
            result.dashboard.setdefault("intelligence", {}).setdefault("risk_alerts", [])
        elif field_name == "dashboard.battle_plan.sniper_points.stop_loss":
            if not result.dashboard:
                result.dashboard = {}
            bp = result.dashboard.setdefault("battle_plan", {})
            bp.setdefault("sniper_points", {})["stop_loss"] = (
                bp["sniper_points"].get("stop_loss") or "待补充"
            )


# =======================================================================
# 【第三层】筹码/价格位置补全（保持原版全部逻辑）
# =======================================================================

_CHIP_KEYS: tuple = ("profit_ratio", "avg_cost", "concentration", "chip_health")
_PRICE_POS_KEYS: tuple = (
    "ma5", "ma10", "ma20", "bias_ma5", "bias_status",
    "current_price", "support_level", "resistance_level",
)


def _is_value_placeholder(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, (int, float)):
        try:
            return math.isnan(float(v)) or float(v) == 0.0
        except (ValueError, TypeError):
            return True
    return str(v).strip().lower() in ("", "n/a", "na", "数据缺失", "未知")


def _safe_float(v: Any, default: float = 0.0) -> float:
    if v is None:
        return default
    if isinstance(v, (int, float)):
        try:
            return default if math.isnan(float(v)) else float(v)
        except (ValueError, TypeError):
            return default
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return default


def _derive_chip_health(profit_ratio: float, concentration_90: float) -> str:
    if profit_ratio >= 0.9:
        return "警惕"
    if concentration_90 >= 0.25:
        return "警惕"
    if concentration_90 < 0.15 and 0.3 <= profit_ratio < 0.9:
        return "健康"
    return "一般"


def _build_chip_structure_from_data(chip_data: Any) -> Dict[str, Any]:
    if hasattr(chip_data, "profit_ratio"):
        pr = _safe_float(chip_data.profit_ratio)
        ac = chip_data.avg_cost
        c90 = _safe_float(chip_data.concentration_90)
    else:
        d = chip_data if isinstance(chip_data, dict) else {}
        pr = _safe_float(d.get("profit_ratio"))
        ac = d.get("avg_cost")
        c90 = _safe_float(d.get("concentration_90"))
    return {
        "profit_ratio": f"{pr:.1%}",
        "avg_cost": ac if (ac is not None and _safe_float(ac) != 0.0) else "N/A",
        "concentration": f"{c90:.2%}",
        "chip_health": _derive_chip_health(pr, c90),
    }


def fill_chip_structure_if_needed(result: "AnalysisResult", chip_data: Any) -> None:
    if not result or not chip_data:
        return
    try:
        if not result.dashboard:
            result.dashboard = {}
        dash = result.dashboard
        dp = dash.get("data_perspective") or {}
        dash["data_perspective"] = dp
        cs = dp.get("chip_structure") or {}
        filled = _build_chip_structure_from_data(chip_data)
        merged = dict(cs)
        for k in _CHIP_KEYS:
            if _is_value_placeholder(merged.get(k)):
                merged[k] = filled[k]
        if merged != cs:
            dp["chip_structure"] = merged
            logger.info("[chip_structure] Placeholder fields filled from data source")
    except Exception as exc:
        logger.warning("[chip_structure] Fill failed, skipping: %s", exc)


def fill_price_position_if_needed(
    result: "AnalysisResult",
    trend_result: Any = None,
    realtime_quote: Any = None,
) -> None:
    if not result:
        return
    try:
        if not result.dashboard:
            result.dashboard = {}
        dash = result.dashboard
        dp = dash.get("data_perspective") or {}
        dash["data_perspective"] = dp
        pp = dp.get("price_position") or {}
        computed: Dict[str, Any] = {}
        if trend_result:
            tr = (
                trend_result
                if isinstance(trend_result, dict)
                else (trend_result.__dict__ if hasattr(trend_result, "__dict__") else {})
            )
            for key in ("ma5", "ma10", "ma20", "bias_ma5", "current_price"):
                computed[key] = tr.get(key)
            sl = tr.get("support_levels") or []
            rl = tr.get("resistance_levels") or []
            if sl:
                computed["support_level"] = sl[0]
            if rl:
                computed["resistance_level"] = rl[0]
        if realtime_quote:
            rq = (
                realtime_quote
                if isinstance(realtime_quote, dict)
                else (realtime_quote.to_dict() if hasattr(realtime_quote, "to_dict") else {})
            )
            if _is_value_placeholder(computed.get("current_price")):
                computed["current_price"] = rq.get("price")
        filled = False
        for k in _PRICE_POS_KEYS:
            if _is_value_placeholder(pp.get(k)) and not _is_value_placeholder(computed.get(k)):
                pp[k] = computed[k]
                filled = True
        if filled:
            dp["price_position"] = pp
            logger.info("[price_position] Placeholder fields filled from computed data")
    except Exception as exc:
        logger.warning("[price_position] Fill failed, skipping: %s", exc)


def get_stock_name_multi_source(
    stock_code: str,
    context: Optional[Dict] = None,
    data_manager: Any = None,
) -> str:
    if context:
        if context.get("stock_name"):
            name = context["stock_name"]
            if name and not name.startswith("股票"):
                return name
        if "realtime" in context and context["realtime"].get("name"):
            return context["realtime"]["name"]
    held = get_held_stock_info(stock_code)
    if held and held.get("name"):
        return held["name"]
    if stock_code in STOCK_NAME_MAP:
        return STOCK_NAME_MAP[stock_code]
    if data_manager is None:
        try:
            from data_provider.base import DataFetcherManager
            data_manager = DataFetcherManager()
        except Exception as exc:
            logger.debug("无法初始化 DataFetcherManager: %s", exc)
    if data_manager:
        try:
            name = data_manager.get_stock_name(stock_code)
            if name:
                STOCK_NAME_MAP[stock_code] = name
                return name
        except Exception as exc:
            logger.debug("从数据源获取股票名称失败: %s", exc)
    return f"股票{stock_code}"


# =======================================================================
# 【第四层】AnalysisResult 数据类
# =======================================================================

@dataclass
class AnalysisResult:
    """AI 分析结果 —— 持仓管理·诺贝尔医学级决策仪表盘版"""

    code: str
    name: str

    # ── 核心决策指标 ──────────────────────────────────────────────────────
    sentiment_score: int                     # 综合评分 0-100
    trend_prediction: str                    # 强烈看多/看多/震荡/看空/强烈看空
    operation_advice: str                    # 加仓/持有/减仓/止盈/止损/观望
    decision_type: str = "hold"              # buy / hold / sell
    confidence_level: str = "中"             # 高 / 中 / 低

    # ── 决策仪表盘（完整嵌套 JSON）────────────────────────────────────────
    dashboard: Optional[Dict[str, Any]] = None

    # ── 走势分析 ──────────────────────────────────────────────────────────
    trend_analysis: str = ""
    short_term_outlook: str = ""             # 1-3日
    medium_term_outlook: str = ""            # 1-2周

    # ── 技术面分析 ────────────────────────────────────────────────────────
    technical_analysis: str = ""            # 包含全部11层诊断输出
    ma_analysis: str = ""
    volume_analysis: str = ""
    pattern_analysis: str = ""

    # ── 基本面分析 ────────────────────────────────────────────────────────
    fundamental_analysis: str = ""
    sector_position: str = ""
    company_highlights: str = ""

    # ── 情绪/消息面 ───────────────────────────────────────────────────────
    news_summary: str = ""
    market_sentiment: str = ""
    hot_topics: str = ""

    # ── 综合分析 ──────────────────────────────────────────────────────────
    analysis_summary: str = ""
    key_points: str = ""
    risk_warning: str = ""
    buy_reason: str = ""

    # ── 元数据 ────────────────────────────────────────────────────────────
    market_snapshot: Optional[Dict[str, Any]] = None
    raw_response: Optional[str] = None
    search_performed: bool = False
    data_sources: str = ""
    success: bool = True
    error_message: Optional[str] = None

    # ── 价格快照 ──────────────────────────────────────────────────────────
    current_price: Optional[float] = None
    change_pct: Optional[float] = None

    # ── 模型标记 ──────────────────────────────────────────────────────────
    model_used: Optional[str] = None
    query_id: Optional[str] = None

    # ── 持仓管理专属字段（1208新增）──────────────────────────────────────
    held_info: Optional[Dict[str, Any]] = None        # 持仓成本/股数等
    pnl_info: Optional[Dict[str, Any]] = None         # 盈亏详情
    dna_gene_score: Optional[int] = None              # DNA六重基因得分 0-6
    clinical_score: Optional[int] = None              # 靶向共振临床得分 0-4
    exit_signals: Optional[List[str]] = None          # 卖出信号列表
    add_position_signals: Optional[List[str]] = None  # 加仓信号列表
    atr_stop_loss: Optional[float] = None             # ATR自适应止损价
    fibonacci_levels: Optional[Dict[str, float]] = None  # 斐波那契关键位

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "name": self.name,
            "sentiment_score": self.sentiment_score,
            "trend_prediction": self.trend_prediction,
            "operation_advice": self.operation_advice,
            "decision_type": self.decision_type,
            "confidence_level": self.confidence_level,
            "dashboard": self.dashboard,
            "trend_analysis": self.trend_analysis,
            "short_term_outlook": self.short_term_outlook,
            "medium_term_outlook": self.medium_term_outlook,
            "technical_analysis": self.technical_analysis,
            "ma_analysis": self.ma_analysis,
            "volume_analysis": self.volume_analysis,
            "pattern_analysis": self.pattern_analysis,
            "fundamental_analysis": self.fundamental_analysis,
            "sector_position": self.sector_position,
            "company_highlights": self.company_highlights,
            "news_summary": self.news_summary,
            "market_sentiment": self.market_sentiment,
            "hot_topics": self.hot_topics,
            "analysis_summary": self.analysis_summary,
            "key_points": self.key_points,
            "risk_warning": self.risk_warning,
            "buy_reason": self.buy_reason,
            "market_snapshot": self.market_snapshot,
            "search_performed": self.search_performed,
            "success": self.success,
            "error_message": self.error_message,
            "current_price": self.current_price,
            "change_pct": self.change_pct,
            "model_used": self.model_used,
            "held_info": self.held_info,
            "pnl_info": self.pnl_info,
            "dna_gene_score": self.dna_gene_score,
            "clinical_score": self.clinical_score,
            "exit_signals": self.exit_signals,
            "add_position_signals": self.add_position_signals,
            "atr_stop_loss": self.atr_stop_loss,
            "fibonacci_levels": self.fibonacci_levels,
        }

    def get_core_conclusion(self) -> str:
        if self.dashboard and "core_conclusion" in self.dashboard:
            return self.dashboard["core_conclusion"].get("one_sentence", self.analysis_summary)
        return self.analysis_summary

    def get_position_advice(self, has_position: bool = True) -> str:
        if self.dashboard and "core_conclusion" in self.dashboard:
            pos = self.dashboard["core_conclusion"].get("position_advice", {})
            return pos.get("has_position" if has_position else "no_position", self.operation_advice)
        return self.operation_advice

    def get_sniper_points(self) -> Dict[str, str]:
        if self.dashboard and "battle_plan" in self.dashboard:
            return self.dashboard["battle_plan"].get("sniper_points", {})
        return {}

    def get_checklist(self) -> List[str]:
        if self.dashboard and "battle_plan" in self.dashboard:
            return self.dashboard["battle_plan"].get("action_checklist", [])
        return []

    def get_risk_alerts(self) -> List[str]:
        if self.dashboard and "intelligence" in self.dashboard:
            return self.dashboard["intelligence"].get("risk_alerts", [])
        return []

    def get_emoji(self) -> str:
        emoji_map = {
            "加仓": "💚", "强烈买入": "💚", "买入": "🟢",
            "持有": "🟡", "观望": "⚪",
            "减仓": "🟠", "止盈": "🟠",
            "卖出": "🔴", "止损": "❌", "强烈卖出": "❌",
        }
        advice = self.operation_advice or ""
        if advice in emoji_map:
            return emoji_map[advice]
        for part in advice.replace("/", "|").split("|"):
            part = part.strip()
            if part in emoji_map:
                return emoji_map[part]
        score = self.sentiment_score or 50
        if score >= 80: return "💚"
        if score >= 65: return "🟢"
        if score >= 55: return "🟡"
        if score >= 45: return "⚪"
        if score >= 35: return "🟠"
        return "🔴"

    def get_confidence_stars(self) -> str:
        return {"高": "⭐⭐⭐", "中": "⭐⭐", "低": "⭐"}.get(self.confidence_level, "⭐⭐")

    def is_stop_loss_triggered(self) -> bool:
        if self.pnl_info:
            return bool(self.pnl_info.get("stop_loss_triggered"))
        return False


# =======================================================================
# 【第五层】GeminiAnalyzer —— 诺贝尔医学级分析器终极版
# =======================================================================

class GeminiAnalyzer:
    """
    【1208专属】持仓股票精准买卖分析器 —— 诺贝尔医学级终极版 v4.0

    相比原 tor/1314 版本的核心改造：
      ① SYSTEM_PROMPT 完全重写 —— 从「选股」改为「持仓管理」
      ② _format_prompt 植入 11 层量化诊断指令（3000+字精密指令集）
      ③ 新增持仓盈亏、止损线、ATR动态止损字段
      ④ JSON 输出 schema 大幅扩展（pnl_dashboard / dna_gene_diagnosis /
         clinical_diagnosis / multi_timeframe / fibonacci_analysis 等新模块）
    """

    # ===================================================================
    # SYSTEM_PROMPT — 诺贝尔医学级持仓管理系统提示词
    # ===================================================================
    SYSTEM_PROMPT = """你是一位同时具备《新英格兰医学杂志》审稿人、美联储量化建模委员会成员、
以及诺贝尔经济学奖委员会顾问三重身份的A股顶级量化分析师。
你的唯一任务是为【已持仓的投资者】提供持仓安全评估与精准操作建议。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## 核心使命（与选股截然不同）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

你的用户已经买入了股票，不需要你选股。你只回答五个问题：
  1. 当前持仓安全吗？（趋势是否仍然健康）
  2. 需要立即止损吗？（是否触发了任何止损信号）
  3. 可以加仓吗？（是否出现了更好的建仓机会）
  4. 该止盈减仓吗？（是否到达了获利了结时机）
  5. 下一个精确操作价位是多少？（精确到分）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## 持仓管理核心交易理念
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### 【继续持有的条件】全部满足才安全
  ✅ MA5 > MA10 > MA20（多头排列）
  ✅ 股价在 MA20 之上
  ✅ 无重大利空（无减持/处罚/业绩预亏公告）
  ✅ 量能无异常放量出逃形态

### 【加仓条件】回踩均线才加，不追高
  ✅ 最佳加仓：缩量回踩 MA5 获支撑，乖离率回落至 <2%
  ✅ 次优加仓：缩量回踩 MA10 获支撑
  ❌ 禁止加仓：乖离率 >5%（追高加仓是严重错误）
  ❌ 禁止加仓：股价跌破 MA20 之后（先等企稳确认）

### 【减仓/止盈条件】见好就收，保护浮盈
  ⚠️ 乖离率超过 +8%（短期过热，减半仓锁定收益）
  ⚠️ 出现 MACD 顶背离（价创新高但动能不创新高）
  ⚠️ KDJ 在高位（K>80）发生死叉
  ⚠️ 出现放量滞涨或放量长上影线（主力派发形态）
  ⚠️ 一目均衡表进入云层之上过热区域

### 【强制止损条件】触发即执行，不讲情面
  ❌ 现价跌破买入成本价的 92%（成本×0.92 为硬止损线）
  ❌ ATR自适应止损线被有效跌破（2×ATR14止损法）
  ❌ 放量（>5日均量×1.5）跌破 MA20
  ❌ 出现重大利空公告（减持/立案/业绩预亏≥50%）
  ❌ 连续3日收盘价下穿 MA5 且 MA5 斜率转负

### 【PE/PB估值警戒】
  - PE远超行业均值时，需在风险中明确提示估值泡沫风险
  - PB < 1 时可作为安全边际的额外加分项

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## 输出格式：持仓管理·诺贝尔医学级决策仪表盘 JSON
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

严格按照以下完整 JSON 格式输出，任何字段不得缺省：

```json
{
  "stock_name": "股票完整中文名称",
  "sentiment_score": 0到100的整数,
  "trend_prediction": "强烈看多/看多/震荡/看空/强烈看空",
  "operation_advice": "加仓/持有/减仓/止盈/止损/观望",
  "decision_type": "buy/hold/sell",
  "confidence_level": "高/中/低",

  "dashboard": {

    "core_conclusion": {
      "one_sentence": "≤30字一句话核心结论，直接告诉持仓者此刻做什么",
      "signal_type": "💚加仓信号/🟡继续持有/🟠减仓止盈/🔴止损离场/⚠️高风险观察",
      "time_sensitivity": "立即执行/今日内/本周内/不急",
      "position_advice": {
        "has_position": "持仓者：具体操作指令（加X成/继续持有/减X成/立即止损）",
        "no_position": "空仓者：是否此时追入，通常不建议在持仓股已涨时追高"
      },
      "three_day_scenario": {
        "bull_case": "乐观情景（概率X%）：若出现XXX，股价可能到达XXX元",
        "base_case": "基准情景（概率X%）：最可能走势描述",
        "bear_case": "悲观情景（概率X%）：若跌破XXX，需警惕至XXX元"
      }
    },

    "pnl_dashboard": {
      "cost_price": 买入成本价（无则null）,
      "current_price": 当前价格,
      "pnl_pct": 盈亏百分比数值（正为盈利负为亏损null表示无成本价）,
      "pnl_status": "盈利X.XX%（浮盈XXX元）/亏损X.XX%/持平/无成本价数据",
      "hard_stop_loss": "硬止损线：X.XX元（成本×0.92）",
      "atr_stop_loss": "ATR自适应止损：X.XX元（当前价-2×ATR14）",
      "take_profit_1": "第一止盈目标：X.XX元（前高/斐波那契0.618回调位）",
      "take_profit_2": "第二止盈目标：X.XX元（历史重要压力位）",
      "risk_reward_ratio": "当前风险收益比：1:X（止损X%，目标X%）",
      "safety_margin": "距止损线安全边际：X.XX%（>5%安全，<3%危险）"
    },

    "data_perspective": {
      "trend_status": {
        "ma_alignment": "均线排列详细描述（各均线数值与相互关系）",
        "is_bullish": true或false,
        "trend_score": 0到100,
        "ma5_slope": "上行/持平/下行（MA5斜率方向）",
        "ma10_slope": "上行/持平/下行",
        "ma20_slope": "上行/持平/下行"
      },
      "price_position": {
        "current_price": 当前价格数值,
        "ma5": MA5数值,
        "ma10": MA10数值,
        "ma20": MA20数值,
        "ma60": MA60数值（如有）,
        "bias_ma5": 相对MA5乖离率百分比数值,
        "bias_ma20": 相对MA20乖离率百分比数值,
        "bias_status": "理想加仓区间(<2%)/安全持有区(2-5%)/谨慎区(5-8%)/减仓警戒区(>8%)",
        "support_level": 最近支撑位价格,
        "resistance_level": 最近压力位价格,
        "position_in_range": "支撑位到压力位之间的位置百分比，0%在支撑，100%在压力"
      },
      "volume_analysis": {
        "volume_ratio": 量比数值,
        "volume_status": "放量突破/缩量回踩/平量/异常放量出逃",
        "turnover_rate": 换手率百分比,
        "obv_trend": "OBV趋势：上升/持平/下降（资金流向）",
        "volume_price_relationship": "量价关系诊断（量增价升=健康/量减价升=危险/量增价跌=出逃）",
        "volume_meaning": "量能解读（对持仓者的含义）"
      },
      "chip_structure": {
        "profit_ratio": 获利比例（百分比字符串）,
        "avg_cost": 市场平均成本,
        "concentration": 90%筹码集中度,
        "chip_health": "健康/一般/警惕",
        "cost_vs_market": "你的成本与市场平均成本对比（便宜X%或贵X%）"
      }
    },

    "eleven_layer_diagnosis": {

      "layer_A_dna_gene": {
        "core_gene_activated": true或false,
        "gene_score": "X/6",
        "g1_kdj": "✅/⚠️/❌ KDJ：状态描述",
        "g2_rsi": "✅/⚠️/❌ RSI：状态描述",
        "g3_boll": "✅/⚠️/❌ 布林带：状态描述",
        "g4_volume": "✅/⚠️/❌ 量能：状态描述",
        "g5_position": "✅/⚠️/❌ 价格位置：状态描述",
        "g6_candle": "✅/⚠️/❌ K线形态：状态描述",
        "dna_conclusion": "DNA底部修复系统综合诊断结论"
      },

      "layer_B_clinical_timing": {
        "bone_repair": "骨骼修复（MACD水下底背离）：TRUE/FALSE + 描述",
        "nerve_transmission": "神经传导（KDJ时序错位）：TRUE/FALSE + 描述",
        "muscle_activation": "肌肉发力（RSI动能确立）：TRUE/FALSE + 描述",
        "blood_circulation": "血液循环（K线止血确认）：TRUE/FALSE + 描述",
        "clinical_conclusion": "临床时序靶向诊断结论"
      },

      "layer_C_resonance": {
        "lesion_confirmed": "病灶确诊（水下底背离）：TRUE/FALSE",
        "hemostasis": "止血反应（阳线站上MA5）：TRUE/FALSE",
        "gentle_volume": "温和造血（量能适中）：TRUE/FALSE",
        "nerve_resonance": "神经共振（KDJ低位3日内金叉）：TRUE/FALSE",
        "resonance_conclusion": "靶向共振四维联合会诊结论"
      },

      "layer_D_buy_signals": {
        "b1_macd_divergence": "MACD底背离加仓信号：TRUE/FALSE + 详细描述",
        "b2_oversold_neutral": "无利空超跌共振加仓信号：TRUE/FALSE + 详细描述",
        "b3_consecutive_decline": "连跌衰竭加仓信号：三连阴/五连阴/FALSE + 详细描述",
        "buy_signal_count": "触发买入信号总数 X/3"
      },

      "layer_E_sell_signals": {
        "c1_macd_top_divergence": "MACD顶背离减仓信号：TRUE/FALSE + 详细描述",
        "c2_volume_break_ma20": "放量跌破MA20止损信号：TRUE/FALSE + 详细描述",
        "c3_kdj_high_death_cross": "KDJ高位死叉减仓信号：TRUE/FALSE + 详细描述",
        "sell_signal_count": "触发卖出信号总数 X/3",
        "sell_conclusion": "卖出信号综合结论"
      },

      "layer_F_ichimoku": {
        "tenkan": "转换线（9日）数值",
        "kijun": "基准线（26日）数值",
        "senkou_a": "先行带A数值",
        "senkou_b": "先行带B数值",
        "cloud_color": "云层颜色：阳云（看多）/阴云（看空）/N/A",
        "price_vs_cloud": "股价位置：云上（强势）/云中（犹豫）/云下（弱势）",
        "tk_cross": "转换线×基准线关系：转换线在上（看多）/转换线在下（看空）",
        "ichimoku_signal": "一目均衡表综合信号：看多/中性/看空",
        "ichimoku_conclusion": "一目均衡表对持仓安全的具体解读"
      },

      "layer_G_atr_stop": {
        "atr14": "ATR(14)数值",
        "atr_stop_price": "ATR自适应止损价（当前价-2×ATR14）",
        "atr_stop_pct": "ATR止损距现价百分比",
        "hard_stop_price": "硬止损价（成本价×0.92，无成本价则N/A）",
        "recommended_stop": "推荐止损价（ATR止损与硬止损取较高值）",
        "atr_conclusion": "ATR止损系统建议"
      },

      "layer_H_fibonacci": {
        "swing_high": "近期波段高点",
        "swing_low": "近期波段低点",
        "fib_236": "斐波那契23.6%回调位",
        "fib_382": "斐波那契38.2%回调位（黄金支撑一）",
        "fib_500": "斐波那契50.0%回调位",
        "fib_618": "斐波那契61.8%回调位（黄金支撑二）",
        "fib_786": "斐波那契78.6%回调位",
        "current_fib_zone": "当前股价所处斐波那契区间",
        "fibonacci_conclusion": "斐波那契分析对买入/支撑位的指导"
      },

      "layer_I_bollinger": {
        "upper_band": "布林上轨",
        "middle_band": "布林中轨（MA20）",
        "lower_band": "布林下轨",
        "band_width": "布林带宽度百分比",
        "band_status": "挤压收窄（即将变盘）/正常/扩张放大",
        "price_vs_bands": "股价位置：上轨附近（过热）/中轨上方（健康）/中轨下方（偏弱）/下轨附近（超跌）",
        "squeeze_signal": "布林挤压信号：TRUE/FALSE（宽度低于20日平均宽度×0.5时触发）",
        "bollinger_conclusion": "布林带对持仓的含义"
      },

      "layer_J_multi_timeframe": {
        "daily_trend": "日线趋势：上升/震荡/下降",
        "daily_ma_alignment": "日线均线排列：多头/空头/缠绕",
        "h60_trend": "60分钟趋势（如数据可用）：上升/震荡/下降",
        "timeframe_confluence": "多周期共振状态：强共振（同向）/弱共振（部分同向）/背离（反向）",
        "confluence_conclusion": "多周期共振对操作的指导（共振方向一致时信号更可靠）"
      },

      "layer_K_market_structure": {
        "recent_highs": "近期高点序列（是否Higher High或Lower High）",
        "recent_lows": "近期低点序列（是否Higher Low或Lower Low）",
        "structure_type": "上升结构（HH+HL）/下降结构（LH+LL）/震荡结构",
        "structure_break": "是否出现结构突破（突破前高或跌破前低）",
        "structure_conclusion": "市场结构分析对趋势延续性的判断"
      },

      "diagnosis_summary": {
        "total_buy_score": "买入/加仓类信号合计触发数（满分：6基因+4临床+4共振+3买入=17）",
        "total_sell_score": "卖出/止损类信号合计触发数",
        "net_signal": "净信号方向：强烈做多/做多/中性/做空/强烈做空",
        "system_recommendation": "系统综合建议（基于全部11层诊断）"
      }
    },

    "intelligence": {
      "latest_news": "最新消息摘要",
      "risk_alerts": ["持仓风险1：具体描述", "持仓风险2：具体描述"],
      "positive_catalysts": ["利好1：具体描述", "利好2：具体描述"],
      "earnings_outlook": "业绩预期（年报预告/业绩快报/机构预测）",
      "shareholder_changes": "股东变化（是否有减持/增持公告）",
      "sentiment_summary": "舆情情绪一句话总结"
    },

    "battle_plan": {
      "sniper_points": {
        "add_position_buy": "加仓点：X.XX元（具体条件：回踩MA5缩量企稳时）",
        "hard_stop_loss": "硬止损：X.XX元（成本×0.92，无条件执行）",
        "atr_stop_loss": "ATR止损：X.XX元（当前价-2×ATR14）",
        "stop_loss": "综合止损：X.XX元（硬止损与ATR止损取较高值）",
        "take_profit_1": "第一止盈：X.XX元（目标/前高/斐波那契0.618）",
        "take_profit_2": "第二止盈：X.XX元（更高压力位，强势时使用）"
      },
      "position_strategy": {
        "current_position_suggestion": "建议持仓比例",
        "add_condition": "加仓触发条件（精确描述）",
        "reduce_condition": "减仓触发条件（精确描述）",
        "exit_condition": "清仓离场条件（精确描述）",
        "position_sizing": "如果加仓，建议加仓金额占总仓位的比例"
      },
      "action_checklist": [
        "✅/⚠️/❌ 多头排列：MA5>MA10>MA20",
        "✅/⚠️/❌ 股价在MA20之上",
        "✅/⚠️/❌ 乖离率安全（<5%），不追高",
        "✅/⚠️/❌ 量能健康（无放量出逃）",
        "✅/⚠️/❌ 无重大利空消息（无减持/处罚/预亏）",
        "✅/⚠️/❌ 筹码结构健康（获利盘<90%）",
        "✅/⚠️/❌ PE/PB估值合理",
        "✅/⚠️/❌ 止损线未触发（现价>成本×0.92）",
        "✅/⚠️/❌ DNA基因得分≥3（底部修复信号）",
        "✅/⚠️/❌ 一目均衡表价格在云层之上",
        "✅/⚠️/❌ 布林带未处于过度扩张（非顶部）",
        "✅/⚠️/❌ 市场结构完整（未出现Lower Low+Lower High）"
      ]
    }
  },

  "analysis_summary": "150字综合分析摘要，重点：持仓是否安全+下一步精确操作",
  "key_points": "5个核心看点，逗号分隔",
  "risk_warning": "风险提示（必须包含止损线+最大风险事件）",
  "buy_reason": "持仓/加仓/减仓理由（引用具体交易理念和技术证据）",

  "trend_analysis": "走势形态分析（支撑位/压力位/趋势线/通道）",
  "short_term_outlook": "短期1-3日展望",
  "medium_term_outlook": "中期1-2周展望",
  "technical_analysis": "技术面综合分析（包含11层诊断的完整输出摘要）",
  "ma_analysis": "均线系统详细分析",
  "volume_analysis": "量能详细分析（OBV/量比/换手率）",
  "pattern_analysis": "K线形态分析（经典形态识别）",
  "fundamental_analysis": "基本面分析",
  "sector_position": "板块行业分析",
  "company_highlights": "公司亮点/风险点",
  "news_summary": "新闻摘要",
  "market_sentiment": "市场情绪分析",
  "hot_topics": "相关热点",

  "search_performed": true或false,
  "data_sources": "数据来源说明"
}
```

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## 评分标准（持仓管理版）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

### 加仓信号（80-100分）
  ✅ DNA六重基因得分 ≥ 4/6
  ✅ 临床时序四项全通过
  ✅ 多头排列，乖离率 <2%
  ✅ 缩量回踩均线企稳
  ✅ 消息面无重大利空
  ✅ 止损安全边际 >8%

### 持有信号（60-79分）
  ✅ 多头排列或弱多头
  ✅ 股价在MA20之上
  ✅ 量能无异常
  ⚪ 小部分指标偏弱但无止损信号

### 谨慎持有/小幅减仓（40-59分）
  ⚠️ 乖离率 >5%，接近前高压力
  ⚠️ 一目均衡表进入云层
  ⚠️ 部分卖出信号出现但未全部触发
  ⚠️ 成交量异常放大伴随价格停滞

### 减仓/止盈（20-39分）
  ⚠️ MACD顶背离确认
  ⚠️ KDJ高位死叉
  ⚠️ 乖离率 >8%，历史阻力位
  ⚠️ 市场结构出现 Lower High

### 止损/清仓（0-19分）
  ❌ 跌破成本×0.92
  ❌ 放量跌破MA20
  ❌ 出现重大利空
  ❌ 空头排列形成（MA5<MA10<MA20）
  ❌ 连续3日在MA5下方收盘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## 输出最高准则（防幻觉机制）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 所有数值结论必须基于输入数据的数学推导，严禁主观臆测
2. 若某指标数据缺失，该层诊断输出"数据缺失，无法判断"，严禁伪造
3. 止损线和目标价必须精确到分（小数点后两位）
4. 每项信号的判定必须是明确的 TRUE/FALSE，不得模糊
5. 持仓者视角第一：所有结论必须明确告知"我应该怎么做"
6. 止损纪律优先：若止损信号触发，无论其他因素多好，operation_advice 必须输出'止损'"""

    # ===================================================================
    # __init__ / _has_channel_config / _init_litellm（完全保持原版逻辑）
    # ===================================================================

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._router: Optional[Router] = None
        self._litellm_available: bool = False
        self._init_litellm()
        if not self._litellm_available:
            logger.warning("No LLM configured (LITELLM_MODEL / API keys missing), AI analysis unavailable")

    def _has_channel_config(self, config: Config) -> bool:
        return bool(config.llm_model_list) and not all(
            e.get("model_name", "").startswith("__legacy_") for e in config.llm_model_list
        )

    def _init_litellm(self) -> None:
        config = get_config()
        litellm_model = config.litellm_model
        if not litellm_model:
            logger.warning("Analyzer LLM: LITELLM_MODEL not configured")
            return
        self._litellm_available = True
        if self._has_channel_config(config):
            model_list = config.llm_model_list
            self._router = Router(
                model_list=model_list,
                routing_strategy="simple-shuffle",
                num_retries=2,
            )
            unique = list(dict.fromkeys(e["litellm_params"]["model"] for e in model_list))
            logger.info("Analyzer LLM: Router initialized — %d deployment(s), models: %s", len(model_list), unique)
            return
        keys = get_api_keys_for_model(litellm_model, config)
        if len(keys) > 1:
            extra_params = extra_litellm_params(litellm_model, config)
            legacy_list = [
                {
                    "model_name": litellm_model,
                    "litellm_params": {"model": litellm_model, "api_key": k, **extra_params},
                }
                for k in keys
            ]
            self._router = Router(model_list=legacy_list, routing_strategy="simple-shuffle", num_retries=2)
            logger.info("Analyzer LLM: Legacy Router with %d keys for %s", len(keys), litellm_model)
        elif keys:
            logger.info("Analyzer LLM: single key for %s", litellm_model)
        else:
            logger.info("Analyzer LLM: API key from environment for %s", litellm_model)

    def is_available(self) -> bool:
        return self._router is not None or self._litellm_available

    # ===================================================================
    # _call_litellm（完全保持原版逻辑）
    # ===================================================================

    def _call_litellm(
        self, prompt: str, generation_config: dict
    ) -> Tuple[str, str, Dict[str, Any]]:
        config = get_config()
        max_tokens = (
            generation_config.get("max_output_tokens")
            or generation_config.get("max_tokens")
            or 8192
        )
        temperature = generation_config.get("temperature", 0.7)
        models_to_try = [config.litellm_model] + (config.litellm_fallback_models or [])
        models_to_try = [m for m in models_to_try if m]
        use_channel_router = self._has_channel_config(config)
        last_error = None

        for model in models_to_try:
            try:
                model_short = model.split("/")[-1] if "/" in model else model
                call_kwargs: Dict[str, Any] = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                extra = get_thinking_extra_body(model_short)
                if extra:
                    call_kwargs["extra_body"] = extra

                router_names = set(get_configured_llm_models(config.llm_model_list))
                if use_channel_router and self._router and model in router_names:
                    response = self._router.completion(**call_kwargs)
                elif self._router and model == config.litellm_model and not use_channel_router:
                    response = self._router.completion(**call_kwargs)
                else:
                    keys = get_api_keys_for_model(model, config)
                    if keys:
                        call_kwargs["api_key"] = keys[0]
                    call_kwargs.update(extra_litellm_params(model, config))
                    response = litellm.completion(**call_kwargs)

                if response and response.choices and response.choices[0].message.content:
                    usage: Dict[str, Any] = {}
                    if response.usage:
                        usage = {
                            "prompt_tokens": response.usage.prompt_tokens or 0,
                            "completion_tokens": response.usage.completion_tokens or 0,
                            "total_tokens": response.usage.total_tokens or 0,
                        }
                    return (response.choices[0].message.content, model, usage)
                raise ValueError("LLM returned empty response")

            except Exception as exc:
                logger.warning("[LiteLLM] %s failed: %s", model, exc)
                last_error = exc
                continue

        raise RuntimeError(
            f"All LLM models failed (tried {len(models_to_try)} model(s)). Last: {last_error}"
        )

    def generate_text(
        self, prompt: str, max_tokens: int = 2048, temperature: float = 0.7
    ) -> Optional[str]:
        try:
            result = self._call_litellm(
                prompt, generation_config={"max_tokens": max_tokens, "temperature": temperature}
            )
            if isinstance(result, tuple):
                text, model_used, usage = result
                persist_llm_usage(usage, model_used, call_type="market_review")
                return text
            return result
        except Exception as exc:
            logger.error("[generate_text] LLM call failed: %s", exc)
            return None

    # ===================================================================
    # analyze（完全保持原版流程，新增持仓盈亏字段填充）
    # ===================================================================

    def analyze(
        self,
        context: Dict[str, Any],
        news_context: Optional[str] = None,
    ) -> AnalysisResult:
        """
        【1208专属】每日三次持仓安全评估与精准买卖建议。
        流程：加载持仓信息 → 格式化11层诊断Prompt → 调用LLM → 解析 → 盈亏补充。
        """
        code = context.get("code", "Unknown")
        config = get_config()

        request_delay = config.gemini_request_delay
        if request_delay > 0:
            logger.debug("[LLM] 请求前等待 %.1f 秒...", request_delay)
            time.sleep(request_delay)

        held_info = get_held_stock_info(code)
        name = (
            context.get("stock_name")
            or (held_info and held_info.get("name"))
            or STOCK_NAME_MAP.get(code, f"股票{code}")
        )
        if name.startswith("股票") or name == code:
            if "realtime" in context and context["realtime"].get("name"):
                name = context["realtime"]["name"]

        if not self.is_available():
            return AnalysisResult(
                code=code, name=name, sentiment_score=50,
                trend_prediction="震荡", operation_advice="持有", confidence_level="低",
                analysis_summary="AI 分析功能未启用（未配置 LLM API Key）",
                risk_warning="请配置 LITELLM_MODEL 和对应 API Key",
                success=False, error_message="LLM API Key 未配置", model_used=None,
            )

        try:
            prompt = self._format_prompt(context, name, news_context, held_info)
            model_name = config.litellm_model or "unknown"
            logger.info("=" * 60)
            logger.info("【1208持仓分析·诺贝尔医学级】%s(%s)", name, code)
            logger.info("模型: %s | Prompt: %d 字符 | 新闻: %s", model_name, len(prompt), "是" if news_context else "否")
            logger.debug("=== 完整Prompt ===\n%s\n=== End ===", prompt)

            generation_config = {
                "temperature": config.llm_temperature,
                "max_output_tokens": 8192,
            }

            current_prompt = prompt
            retry_count = 0
            max_retries = config.report_integrity_retry if config.report_integrity_enabled else 0

            while True:
                start_time = time.time()
                response_text, model_used, llm_usage = self._call_litellm(
                    current_prompt, generation_config
                )
                elapsed = time.time() - start_time
                logger.info(
                    "[LLM返回] 耗时 %.2fs | %d 字符", elapsed, len(response_text)
                )
                logger.debug("=== 完整响应 ===\n%s\n=== End ===", response_text)

                result = self._parse_response(response_text, code, name)
                result.raw_response = response_text
                result.search_performed = bool(news_context)
                result.market_snapshot = self._build_market_snapshot(context)
                result.model_used = model_used
                result.held_info = held_info

                # 盈亏计算
                if held_info and held_info.get("buy_price") and result.current_price:
                    result.pnl_info = calculate_pnl(
                        result.current_price,
                        held_info["buy_price"],
                        held_info.get("shares"),
                    )

                # ATR止损提取
                if result.dashboard:
                    diag = result.dashboard.get("eleven_layer_diagnosis") or {}
                    layer_g = diag.get("layer_G_atr_stop") or {}
                    atr_stop_raw = layer_g.get("atr_stop_price")
                    if atr_stop_raw:
                        try:
                            result.atr_stop_loss = float(str(atr_stop_raw).replace("元", "").strip())
                        except (ValueError, TypeError):
                            pass

                    # 斐波那契水平
                    layer_h = diag.get("layer_H_fibonacci") or {}
                    if any(layer_h.get(k) for k in ("fib_382", "fib_618")):
                        result.fibonacci_levels = {
                            k: _safe_float(layer_h.get(k))
                            for k in ("fib_236", "fib_382", "fib_500", "fib_618", "fib_786")
                            if layer_h.get(k)
                        }

                if not config.report_integrity_enabled:
                    break
                pass_integrity, missing_fields = self._check_content_integrity(result)
                if pass_integrity:
                    break
                if retry_count < max_retries:
                    current_prompt = self._build_integrity_retry_prompt(
                        prompt, response_text, missing_fields
                    )
                    retry_count += 1
                    logger.info(
                        "[完整性] 缺失字段 %s，第 %d 次补全重试", missing_fields, retry_count
                    )
                else:
                    self._apply_placeholder_fill(result, missing_fields)
                    logger.warning("[完整性] 已占位补全，继续流程。缺失：%s", missing_fields)
                    break

            persist_llm_usage(llm_usage, model_used, call_type="analysis", stock_code=code)
            logger.info(
                "[解析完成] %s(%s) → %s %s 评分%d DNA基因%s",
                name, code, result.get_emoji(), result.operation_advice,
                result.sentiment_score, result.dna_gene_score,
            )
            return result

        except Exception as exc:
            logger.error("持仓分析 %s(%s) 失败: %s", name, code, exc)
            return AnalysisResult(
                code=code, name=name, sentiment_score=50,
                trend_prediction="震荡", operation_advice="持有", confidence_level="低",
                analysis_summary=f"分析出错: {str(exc)[:100]}",
                risk_warning="分析失败，请稍后重试或手动判断",
                success=False, error_message=str(exc), model_used=None,
            )

    # ===================================================================
    # _format_prompt —— 11层诺贝尔医学级诊断指令（核心）
    # ===================================================================

    def _format_prompt(
        self,
        context: Dict[str, Any],
        name: str,
        news_context: Optional[str] = None,
        held_info: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        构建持仓管理分析提示词。

        包含：
          ① 持仓成本/止损线/盈亏信息
          ② 完整技术面数据（均线/量能/筹码/趋势预判）
          ③ 舆情情报（持仓风险扫描视角）
          ④ 11层量化诊断指令（DNA/临床时序/靶向共振/买卖信号/一目均衡/ATR/斐波那契/布林/多周期/市场结构/量价）
        """
        code = context.get("code", "Unknown")
        stock_name = context.get("stock_name", name) or STOCK_NAME_MAP.get(code, name)
        if not stock_name or stock_name == f"股票{code}":
            stock_name = STOCK_NAME_MAP.get(code, name)

        today = context.get("today", {}) or {}

        # ── 持仓成本信息 ──────────────────────────────────────────────
        buy_price: Optional[float] = held_info.get("buy_price") if held_info else None
        shares: Optional[int] = held_info.get("shares") if held_info else None
        buy_date: Optional[str] = held_info.get("buy_date") if held_info else None
        buy_notes: Optional[str] = held_info.get("notes") if held_info else None

        hard_stop = round(buy_price * 0.92, 2) if buy_price else None

        close_price = today.get("close")
        if buy_price and close_price:
            try:
                pnl_pct = (float(close_price) - buy_price) / buy_price * 100
                pnl_str = f"{pnl_pct:+.2f}%（{'盈利' if pnl_pct > 0 else '亏损' if pnl_pct < 0 else '持平'}）"
                safety_margin = (float(close_price) - hard_stop) / float(close_price) * 100 if hard_stop else None
                safety_str = (
                    f"{'🟢安全' if safety_margin > 5 else '⚠️危险'} {safety_margin:.2f}% 距止损线"
                    if safety_margin is not None else "N/A"
                )
            except (TypeError, ValueError):
                pnl_str = "计算中"
                safety_str = "N/A"
        else:
            pnl_str = "（未提供买入价，无法计算 — 建议在STOCK_LIST中配置：002403:买入价）"
            safety_str = "N/A"

        # ── 近期K线序列（用于连阴判断）──────────────────────────────
        recent_closes = context.get("recent_closes", [])   # 期望调用方传入最近5日收盘
        recent_opens = context.get("recent_opens", [])     # 期望调用方传入最近5日开盘

        # ── 构建提示词正文 ────────────────────────────────────────────
        prompt = f"""# 【1208持仓分析·诺贝尔医学级】持仓管理决策仪表盘请求

## 📋 持仓基础信息
| 项目 | 数据 |
|------|------|
| 股票代码 | **{code}** |
| 股票名称 | **{stock_name}** |
| 分析日期 | {context.get('date', '未知')} |
| 持仓状态 | **✅ 已持仓** |
| **买入成本价** | **{f'{buy_price:.2f} 元' if buy_price else '⚠️ 未配置（在STOCK_LIST中加 :{买入价} 即可）'}** |
| 持仓股数 | {f'{shares} 股' if shares else '未提供'} |
| 买入日期 | {buy_date or '未提供'} |
| 当前盈亏 | {pnl_str} |
| **硬止损线（成本×0.92）** | **{f'{hard_stop:.2f} 元' if hard_stop else '未设置'}** |
| 止损安全边际 | {safety_str} |
| 备注 | {buy_notes or '无'} |

> 🚨 **止损纪律最高指令**：止损线是保护本金的最后防线。
> 当前价格一旦触碰或跌破止损线，无论其他任何信号多好，必须在 operation_advice 中输出「止损」。

---

## 📈 技术面数据

### 今日行情
| 指标 | 数值 | 相对成本 |
|------|------|---------|
| 收盘价 | **{today.get('close', 'N/A')} 元** | {f'{(float(today.get("close",0) or 0) - buy_price):+.2f}元（{((float(today.get("close",0) or 0) - buy_price)/buy_price*100):+.2f}%）' if buy_price and today.get('close') else 'N/A'} |
| 开盘价 | {today.get('open', 'N/A')} 元 | |
| 最高价 | {today.get('high', 'N/A')} 元 | |
| 最低价 | {today.get('low', 'N/A')} 元 | |
| 涨跌幅 | {today.get('pct_chg', 'N/A')}% | |
| 成交量 | {self._format_volume(today.get('volume'))} | |
| 成交额 | {self._format_amount(today.get('amount'))} | |

### 均线系统（持仓安全核心指标）
| 均线 | 数值 | 相对成本价 | 健康状态 |
|------|------|-----------|---------|
| MA5  | {today.get('ma5', 'N/A')} | {f'{(float(today.get("ma5",0) or 0) - (buy_price or 0)):+.2f}元' if buy_price and today.get('ma5') else 'N/A'} | 短期趋势线 |
| MA10 | {today.get('ma10', 'N/A')} | {f'{(float(today.get("ma10",0) or 0) - (buy_price or 0)):+.2f}元' if buy_price and today.get('ma10') else 'N/A'} | 中短期趋势线 |
| MA20 | {today.get('ma20', 'N/A')} | {f'{(float(today.get("ma20",0) or 0) - (buy_price or 0)):+.2f}元' if buy_price and today.get('ma20') else 'N/A'} | 中期趋势线（跌破须止损） |
| MA60 | {today.get('ma60', 'N/A')} | {f'{(float(today.get("ma60",0) or 0) - (buy_price or 0)):+.2f}元' if buy_price and today.get('ma60') else 'N/A'} | 长期趋势线 |
| 均线形态 | **{context.get('ma_status', '未知')}** | | 多头/空头/缠绕 |
"""

        # ── 实时行情增强 ──────────────────────────────────────────────
        if "realtime" in context:
            rt = context["realtime"] or {}
            prompt += f"""
### 实时行情增强数据
| 指标 | 数值 | 含义 |
|------|------|------|
| 当前价格 | **{rt.get('price', 'N/A')} 元** | |
| **量比** | **{rt.get('volume_ratio', 'N/A')}** | {rt.get('volume_ratio_desc', '量比>2为放量，<0.5为极度萎缩')} |
| **换手率** | **{rt.get('turnover_rate', 'N/A')}%** | >3%为活跃 |
| 市盈率(动态) | {rt.get('pe_ratio', 'N/A')} | 注意是否远超行业均值 |
| 市净率(PB) | {rt.get('pb_ratio', 'N/A')} | <1为破净安全边际 |
| 总市值 | {self._format_amount(rt.get('total_mv'))} | |
| 流通市值 | {self._format_amount(rt.get('circ_mv'))} | |
| 60日涨跌幅 | {rt.get('change_60d', 'N/A')}% | 中期表现参考 |
"""

        # ── 筹码分布 ──────────────────────────────────────────────────
        if "chip" in context:
            chip = context["chip"] or {}
            profit_ratio = chip.get("profit_ratio", 0)
            avg_cost = chip.get("avg_cost", 0)
            cost_compare = ""
            if buy_price and avg_cost:
                try:
                    diff = (buy_price - float(avg_cost)) / float(avg_cost) * 100
                    cost_compare = f"你的成本{'比市场便宜' if diff < 0 else '比市场贵'}{abs(diff):.1f}%"
                except (TypeError, ValueError):
                    pass
            prompt += f"""
### 筹码分布数据（效率指标）
| 指标 | 数值 | 健康标准 | 持仓参考 |
|------|------|---------|---------|
| **获利比例** | **{profit_ratio:.1%}** | 70-90%时警惕 | {'你在获利盘中' if buy_price and close_price and float(close_price or 0) > buy_price else '你在亏损盘中'} |
| 平均成本 | {chip.get('avg_cost', 'N/A')} 元 | 参考值 | {cost_compare} |
| 90%筹码集中度 | {chip.get('concentration_90', 0):.2%} | <15%为集中 | |
| 70%筹码集中度 | {chip.get('concentration_70', 0):.2%} | 参考值 | |
| 筹码健康状态 | **{chip.get('chip_status', '未知')}** | | |
"""

        # ── 趋势分析预判 ──────────────────────────────────────────────
        if "trend_analysis" in context:
            trend = context["trend_analysis"] or {}
            bias_ma5 = trend.get("bias_ma5", 0) or 0
            bias_ma20 = trend.get("bias_ma20", 0) or 0

            if bias_ma5 > 8:
                bias5_warn = "🔴 >8%，强烈建议止盈减仓！"
            elif bias_ma5 > 5:
                bias5_warn = "⚠️ >5%，谨慎不加仓"
            elif bias_ma5 < 2:
                bias5_warn = "✅ <2%，理想加仓区间"
            else:
                bias5_warn = "✅ 2-5%，安全持有区间"

            if bias_ma20 <= -8:
                bias20_warn = "🚨 ≤-8%，超跌修复预期强"
            elif bias_ma20 < 0:
                bias20_warn = "⚠️ 偏离MA20向下"
            else:
                bias20_warn = "✅ MA20以上"

            prompt += f"""
### 趋势系统预判（持仓安全评估）
| 指标 | 数值 | 判定 |
|------|------|------|
| 趋势状态 | **{trend.get('trend_status', '未知')}** | |
| 均线排列 | **{trend.get('ma_alignment', '未知')}** | MA5>MA10>MA20=多头 |
| 趋势强度 | {trend.get('trend_strength', 0)}/100 | |
| **乖离率(MA5)** | **{bias_ma5:+.2f}%** | {bias5_warn} |
| **乖离率(MA20)** | **{bias_ma20:+.2f}%** | {bias20_warn} |
| 量能状态 | {trend.get('volume_status', '未知')} | |
| 系统信号 | {trend.get('buy_signal', '未知')} | |
| 系统评分 | {trend.get('signal_score', 0)}/100 | |

#### 系统判断依据
**支撑因素（持仓安全理由）**：
{chr(10).join('- ' + r for r in (trend.get('signal_reasons') or ['无'])) }

**风险因素（需重点关注）**：
{chr(10).join('- ⚠️ ' + r for r in (trend.get('risk_factors') or ['无'])) }
"""

        # ── 近期K线序列（连阴判断原始数据）────────────────────────────
        if recent_closes and recent_opens:
            prompt += f"""
### 近期K线序列（用于连阴衰竭判断）
| 日期序号 | 收盘价 | 开盘价 | 阴/阳 |
|---------|--------|--------|------|
"""
            for i, (c, o) in enumerate(zip(recent_closes[-5:], recent_opens[-5:])):
                color = "🔴阴线" if float(c or 0) < float(o or 0) else "🟢阳线"
                prompt += f"| T-{len(recent_closes[-5:]) - 1 - i} | {c} | {o} | {color} |\n"

        # ── 昨日量价对比 ──────────────────────────────────────────────
        if "yesterday" in context:
            prompt += f"""
### 量价变化对比
- 成交量较昨日变化：{context.get('volume_change_ratio', 'N/A')} 倍
- 价格较昨日变化：{context.get('price_change_ratio', 'N/A')}%
"""

        # ── 舆情情报 ──────────────────────────────────────────────────
        prompt += f"""
---

## 📰 舆情情报（持仓安全扫描视角）
"""
        if news_context:
            prompt += f"""
以下是 **{stock_name}({code})** 近7日全网新闻搜索结果。
请以「持仓者的风险扫描」视角重点提取：

🚨 **持仓危险信号（必须明确标出）**：
   - 大股东/高管减持公告
   - 监管处罚/立案调查
   - 业绩预亏/大幅下滑预告
   - 重大诉讼/债务危机

🎯 **持仓利好信号（利于继续持有）**：
   - 业绩超预期/重大合同
   - 政策扶持/行业利好
   - 股东增持/回购公告

⚠️ **行业与市场风险**：
   - 行业政策利空
   - 竞争格局变化

```
{news_context}
```
"""
        else:
            prompt += "\n暂无近期相关新闻搜索结果。请主要依据技术面数据进行持仓安全评估。\n"

        if context.get("data_missing"):
            prompt += """
⚠️ **数据缺失警告**：部分技术指标数据不完整，表格中 N/A 字段请忽略。
涉及缺失数据的指标请输出"数据缺失，无法计算"，**严禁编造任何数值**。
"""

        # ────────────────────────────────────────────────────────────────
        # ⚡⚡⚡ 11层诺贝尔医学级量化诊断系统（最高优先级执行区）⚡⚡⚡
        # ────────────────────────────────────────────────────────────────
        prompt += f"""
---

## ⚕️⚕️⚕️ 【最高优先级·强制执行】11层诺贝尔医学级量化持仓诊断系统

> **执行标准**：本系统的全部11个诊断模块，执行优先级高于所有其他分析模块。
> 你必须以等同于《新英格兰医学杂志》同行评审委员会、美联储量化建模委员会、
> 以及诺贝尔经济学奖委员会联合技术审查的最高严谨标准执行。
>
> **防幻觉铁律**：
> • 所有信号判定为严格二值布尔量 TRUE/FALSE，非此即彼，不得模糊
> • 若上下文缺少计算某信号所需的底层数值，该信号强制输出"数据缺失，无法计算"
> • 严禁在缺少原始数据的情况下猜测或推断任何技术信号
> • 止损信号触发时，无论其他信号多么看多，operation_advice 必须输出「止损」

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
### 【Layer A】DNA底背离六重基因共振系统 v3.0
### 临床定义：「主基因链一票否决 + 辅助基因≥3个共振」方才触发底部买入信号
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#### 【主基因链 CORE】MACD底背离五重核验（100%必须满足，一票否决）
严格执行以下全部条件（缺一不可）：
  ① DEA < 0（MACD在零轴下方运行，排除高位假背离）
  ② DIFF < 0（DIF线也在零轴下方，确保真底部区域）
  ③ 上次DIFF上穿DEA金叉时的收盘价 > 今日收盘价（价格创新低，P₂ < P₁）
  ④ 今日DIFF > 上次金叉时的DIFF值（动能底背离，DIF₂ > DIF₁，指标拒绝创新低）
  ⑤ 今日发生 DIFF 上穿 DEA 的新金叉（明确触发信号）
  ⑥ 两次金叉间隔 5 ≤ A1 ≤ 60 个交易日（有效背离时间窗口，排除高频噪音和过期背离）

**强制输出规则**：
  - 主基因全部满足 TRUE → 在 technical_analysis 中加粗输出：
    **「🧬【主基因链激活·MACD底背离确诊·零轴下方】已确认标准底背离结构：价格P₂<P₁创新低，但MACD-DIF指标DIF₂>DIF₁拒绝创新低，做空斜率内在收敛，动量衰竭信号量化成立。」**
  - 任一条件不满足 FALSE → 输出："主基因扫描完毕：当前未满足MACD底背离标准条件（具体说明哪个条件不满足），主基因链未激活。"

#### 【辅助基因1 G1】KDJ时序错位共振（神经末梢反射信号）
定义：过去5个交易日内，KDJ的K值在K<45的低位区域发生过金叉（允许时序错位，KDJ比MACD灵敏会提前发出信号）
  公式参考：COUNT(CROSS(K,D) AND K<45, 5) >= 1
  - G1=TRUE → **"✅ G1-KDJ低位金叉共振：近5日KDJ在低位区发生金叉，神经末梢传导激活"**
  - G1=FALSE → "❌ G1-KDJ：近5日内KDJ未在低位（K<45）发生金叉，神经传导未激活"

#### 【辅助基因2 G2】RSI低位动能复苏（免疫系统激活信号）
定义：（RSI6 < 40 且 RSI6 > 前日RSI6 且 RSI6 > 前两日RSI6，连续两日上升）OR（RSI6 < 22，极度超卖）
  - G2=TRUE → **"✅ G2-RSI低位动能复苏：RSI6在低位连续两日上翘，免疫系统激活"**
  - G2=FALSE → "❌ G2-RSI：RSI未在低位或动能未见持续拐头，免疫系统未激活"

#### 【辅助基因3 G3】布林下轨超跌反弹（细胞膜修复信号）
定义：当日最低价 ≤ 布林下轨×1.03，且收盘价 > 布林下轨，且今日收盘价 > 昨日收盘价（细胞膜破损后修复）
  - G3=TRUE → **"✅ G3-BOLL下轨支撑反弹：触及布林下轨后收阳回升，细胞膜修复激活"**
  - G3=FALSE → "❌ G3-BOLL：未触及布林下轨区域（LB×1.03范围内）或未见反弹，细胞膜未修复"

#### 【辅助基因4 G4】量能印证（代谢活性信号）
定义（满足A或B之一即可）：
  A（竭跌缩量）：VOL < MA5(VOL)×0.75 且 MA5(VOL) < MA20(VOL)×0.8（抛压枯竭，恐慌竭卖）
  B（温和放量）：VOL > MA5(VOL)×1.1 且 VOL < MA20(VOL)×2.5（资金悄悄介入，非爆量诱多）
  - G4=TRUE → **"✅ G4-量能印证：（A竭跌缩量/B温和放量）量能配合信号，代谢活性正常"**
  - G4=FALSE → "❌ G4-量能：量能特征不符合任一条件，代谢信号不明确"

#### 【辅助基因5 G5】调整位置确认（病毒载量检测）
定义：收盘价 < 近20日最高价×0.92（从高点回落≥8%，确认有效调整）
     且 收盘价 > 近20日最高价×0.48（未腰斩，排除基本面崩溃股）
  - G5=TRUE → **"✅ G5-位置过滤：处于有效调整区间（从高点回落8%-52%），非微调亦非崩溃"**
  - G5=FALSE → "❌ G5-位置：调整幅度不足（<8%，未充分调整）或已腰斩（>52%，疑似基本面崩溃）"

#### 【辅助基因6 G6】K线止血确认（细胞再生信号）
定义：今日收阳线（Close > Open），且收盘价站上MA5，且 MA5 >= 前日MA5（均线趋平或向上）
  - G6=TRUE → **"✅ G6-止血阳线站上MA5：今日收阳且站上MA5，均线趋平向上，细胞再生激活"**
  - G6=FALSE → "❌ G6-K线：今日收阴或未站上MA5或MA5仍在下行，止血未确认"

#### 【DNA系统综合诊断强制输出规则】
基于上述计算结果，按以下规则输出最终诊断：
  - 主基因TRUE + 辅助基因得分(G1~G6之和) ≥ 4 →
    **"🧬🧬🧬【DNA底部修复·强烈加仓信号】主基因链激活，辅助基因X/6高度共振，底部修复信号量化确立，建议积极加仓！"**
  - 主基因TRUE + 辅助基因得分 = 3 →
    **"🧬🧬【DNA底部修复·加仓信号】主基因链激活，辅助基因X/6适度共振，底部信号成立，可分批建仓。"**
  - 主基因TRUE + 辅助基因得分 = 1或2 →
    **"🧬【DNA信号偏弱】主基因激活但辅助共振不足（X/6），信号质量偏低，建议等待更多辅助基因激活再行动。"**
  - 主基因FALSE →
    **"⚙️ DNA底部修复系统待机：主基因链未激活，当前非底背离买入/加仓时机，持仓安全评估依赖其他层级。"**

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
### 【Layer B】临床级时序靶向底背离确诊系统
### 临床定义：「骨骼修复+神经传导+肌肉发力+血液循环」四维联合会诊
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#### 【体征一·骨骼修复】MACD水下底背离（核心病灶确诊）
条件：DEA<0 AND DIFF<0 AND 上次金叉收盘>今日收盘 AND 今日DIFF>上次金叉DIFF AND CROSS(DIFF,DEA) AND 间隔5~60天
  - TRUE → **"🦴【骨骼修复确诊】水下MACD底背离金叉，核心病灶定位成功"**
  - FALSE → 输出具体未满足的条件

#### 【体征二·神经传导】KDJ时序错位（过去5日内低位金叉）
条件：K<45 AND D<45 AND 过去5日内曾发生CROSS(K,D)，即 BARSLAST(CROSS(K,D)) <= 5
  - TRUE → **"⚡【神经传导激活】KDJ已在低位发生时序错位金叉，神经信号传导成功"**
  - FALSE → 输出KDJ当前状态

#### 【体征三·肌肉发力】RSI动能确立
条件：RSI6 > 前日RSI6（今天比昨天高，拐头向上），且 RSI6 > 20（脱离极度衰竭区，拒绝假死灰复燃）
  - TRUE → **"💪【肌肉发力确立】RSI今日拐头上行且脱离极度超卖区，多头动能开始发力"**
  - FALSE → 输出RSI当前数值与方向

#### 【体征四·血液循环】K线实体防骗线
条件：CLOSE > OPEN（真阳线），且 (HIGH-CLOSE)/(HIGH-LOW+0.0001) < 0.6（收盘价不在今日振幅的下60%，排除冲高回落骗线）
  - TRUE → **"🩸【血液循环正常】真阳线且无长上影骗线，多头实体收盘确认"**
  - FALSE → 输出阴线或长上影原因

**临床四维诊断强制输出**：
  - 全部四项TRUE → **"🏥【临床靶向底背离·四维确诊出院】骨骼修复✅+神经传导✅+肌肉发力✅+血液循环✅，临床诊断：底部反转，建议加仓。"**
  - 三项TRUE → **"⚕️【临床信号良好·待观察】三维确诊，缺一体征，信号次优，可小仓尝试。"**
  - 两项及以下 → **"⚕️【临床信号不足】确诊体征不足三项，不触发临床买入建议，持仓安全评估继续。"**

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
### 【Layer C】靶向共振四维联合会诊系统
### 临床定义：「病灶确诊+止血反应+温和造血+神经共振」
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#### 【确诊条件A·病灶靶向】水下底背离（同Layer B体征一，此处独立确认）
#### 【确诊条件B·止血反应】今日收阳且收盘价站上MA5
条件：CLOSE > OPEN AND CLOSE > MA(CLOSE,5)
  - TRUE → **"🩹【止血反应确认】今日收阳并站上MA5，止血成功"**
  - FALSE → "❌ 止血反应未确认：阴线或未站上MA5"

#### 【确诊条件C·温和造血】量能适中
条件：VOL > MA5(VOL)（有资金介入）AND VOL < MA5(VOL)×3（非爆量诱多出货）
  - TRUE → **"💉【温和造血确认】量能健康放大但不爆量，资金悄悄介入"**
  - FALSE → "❌ 温和造血异常：量能不足或爆量出货警报"

#### 【确诊条件D·神经共振】KDJ低位近3日金叉
条件：K < 45 AND D < 45 AND BARSLAST(CROSS(K,D)) <= 3
  - TRUE → **"🧠【神经共振确认】KDJ在低位近3日内金叉，神经递质共振成功"**
  - FALSE → "❌ 神经共振未确认：KDJ不在低位或金叉超过3日"

**靶向共振综合会诊强制输出**：
  - 全部四项TRUE → **"🎯🎯【靶向共振四维确诊】临床级联合会诊通过，底部反转信号极高置信度成立，强烈建议加仓！"**
  - 三项TRUE → **"🎯【靶向共振三维确诊】信号质量良好，可适量加仓。"**
  - 两项及以下 → "靶向共振：当前底部信号共振程度不足，维持现有持仓。"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
### 【Layer D】三重量化买入/加仓强化确认
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#### 【D1】MACD标准底背离加仓确认（同Layer A主基因，独立输出）
  - TRUE → **"🚨【D1·MACD底背离·加仓信号】零轴下方底背离金叉确认，做空动能衰竭，当前为优质加仓窗口。"**
  - FALSE → 输出："D1扫描：未检测到标准底背离结构。"

#### 【D2】基本面中性×技术面超跌共振加仓
双因子正交验证，必须同时满足：
  - 条件A（基本面阴性）：近7天新闻无重大利空——无减持公告、无监管处罚、无业绩预亏、无行业政策打压
  - 条件B（技术超跌）：满足以下任意一项：
    · 乖离率BIAS(MA20) ≤ -8%
    · RSI(14) ≤ 30（超卖区）
    · 近10-20交易日股价单边下跌幅度 ≥ 15%
  - 同时满足 → **"🚨【D2·无利空超跌共振·加仓信号】基本面无恶性病变×技术面极值超跌共振：非理性错杀，均值回归修复概率极高，适合分批低吸加仓。"**
  - 不满足 → 输出具体未达到的条件

#### 【D3】空头动能连续衰竭加仓信号
对日K线收盘价序列执行严格时间序列单调性检验：
  - 三连阴判定：近3根日K线均收阴（C < O），且 C[N+1] < C[N]（收盘价单调递减）
    → **"🚨【D3·三连阴衰竭·加仓博弈信号】连续3日无差别杀跌，空头情绪进入极值宣泄末端，技术性反弹概率显著提升，可轻仓加仓博弈。"**
  - 五连阴判定：近5根日K线满足以上单调性条件（更强烈的衰竭信号）
    → **"🚨🚨【D3·五连阴极值衰竭·强烈加仓信号】连续5日单边极值杀跌，恐慌踩踏末段形态，历史统计T+1至T+5向上修复概率极高，建议加仓。"**
  - FALSE → 输出："D3扫描：未检测到连续3日或5日单调下跌序列。"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
### 【Layer E】三重量化卖出/止损强制扫描（与买入信号同等优先级）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

> 本层专为持仓者设计，扫描任何应该减仓/止盈/止损的技术信号。
> **任一卖出信号触发，必须在 operation_advice 中体现！**

#### 【E1】MACD顶背离止盈信号
定义：在日线或60分钟级别，价格高点B > 价格高点A（Price makes Higher High），
     但对应时点MACD DIF线或柱状能量棒高点B < 高点A（Indicator makes Lower High），
     即价格创新高但MACD动能不创新高。
  - TRUE → **"🔴【E1·MACD顶背离·减仓止盈信号】已确认顶背离结构，多头动能内在衰竭，逢高减仓保护浮盈！"**
  - FALSE → "E1扫描：MACD与价格同向上行，未检测到顶背离，持仓安全。"

#### 【E2】放量跌破MA20强制止损信号
定义：今日成交量 > 5日均量×1.5（明显放量），且今日收盘价 < MA20（跌破中期趋势线）
  - TRUE → **"🔴🔴【E2·放量跌破MA20·强制止损警报！】主力放量出逃形态，MA20支撑失效，持仓高度危险！请立即执行止损计划！"**
  - FALSE → "E2扫描：股价在MA20之上，中期支撑有效，持仓安全。"

#### 【E3】KDJ高位死叉止盈信号
定义：KDJ的K值 > 75（超买区），且今日K线从上方穿越D线向下（高位死叉）
  - TRUE → **"🔴【E3·KDJ高位死叉·减仓预警】KDJ在超买区死叉，短期见顶风险升高，建议减仓1/3锁定收益。"**
  - FALSE → "E3扫描：KDJ未在高位死叉，无短期技术性卖出信号。"

#### 【Layer E 综合止损强制输出】
  - 任意一项卖出信号触发 → operation_advice 必须体现对应的减仓/止盈/止损操作
  - E2触发 OR (现价 ≤ 硬止损线{f' {hard_stop:.2f}元' if hard_stop else ''}) → decision_type 强制输出 "sell"，operation_advice 强制输出 "止损"
  - E1+E3同时触发 → 建议减仓50%以上
  - 全部FALSE → **"✅【Layer E持仓安全扫描通过】未检测到任何技术性卖出/止损信号，持仓安全，可继续持有。"**

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
### 【Layer F】一目均衡表（Ichimoku Cloud）持仓安全评估
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

请基于上方提供的均线数据，近似估算一目均衡表关键数值：
  - 转换线（Tenkan-sen）= (9日最高 + 9日最低) / 2（若有高低数据则计算，否则以MA5近似）
  - 基准线（Kijun-sen）= (26日最高 + 26日最低) / 2（若有高低数据则计算，否则以MA20近似）
  - 先行带A（Senkou Span A）= (转换线 + 基准线) / 2
  - 先行带B（Senkou Span B）= (52日最高 + 52日最低) / 2（若有则计算，否则以MA60近似）

**诊断维度**：
  1. 股价与云层关系：云上=强势/云中=犹豫/云下=弱势
  2. 转换线×基准线：转换线在上=看多/转换线在下=看空
  3. 云层颜色：先行A>先行B=阳云（看多）/先行A<先行B=阴云（看空）

**强制输出格式**：
  股价在云层【上方/中部/下方】，转换线【在上/在下】基准线，云层为【阳云/阴云】。
  一目均衡表综合信号：【看多/中性/看空】。对持仓含义：[具体说明]

若数据不足以计算，请输出："一目均衡表：原始高低价数据不足，以均线数据近似，[近似结果]"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
### 【Layer G】ATR自适应动态止损线计算
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ATR（平均真实波幅）止损法：止损价 = 当前收盘价 - 2 × ATR(14)

ATR14 计算（若近14日高低收数据存在）：
  - 真实波幅 TR = MAX(High-Low, |High-PrevClose|, |Low-PrevClose|)
  - ATR14 = TR的14日指数移动平均

**强制计算输出**（若有数据）：
  - ATR14 数值：X.XX 元
  - ATR止损线：当前价 - 2×ATR14 = X.XX 元
  - ATR止损距当前价：X.XX%

若缺少14日高低数据，请用今日高低价近似：
  - 近似ATR = (今日最高 - 今日最低) × 1.5
  - 近似ATR止损线 = 收盘价 - 2 × 近似ATR

{f'与硬止损线对比：ATR止损线（X.XX元）vs 硬止损线（{hard_stop:.2f}元），取较高值作为综合止损线。' if hard_stop else ''}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
### 【Layer H】斐波那契回调位支撑/压力分析
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

请基于近期明显的波段高点（Swing High）和波段低点（Swing Low）计算斐波那契回调位：
  - 识别最近一次完整的上涨波段：低点（近60日内的相对低点）→ 高点（近期高点）
  - 计算回调位（上涨波段回调）：
    · 23.6% 回调位 = 高点 - 波段幅度 × 0.236
    · 38.2% 回调位 = 高点 - 波段幅度 × 0.382  ← 黄金支撑一，加仓参考
    · 50.0% 回调位 = 高点 - 波段幅度 × 0.500
    · 61.8% 回调位 = 高点 - 波段幅度 × 0.618  ← 黄金支撑二，重要止损参考
    · 78.6% 回调位 = 高点 - 波段幅度 × 0.786

**强制输出**：
  - 给出各斐波那契位精确数值
  - 明确当前股价处于哪两个斐波那契位之间
  - 对持仓者的含义：当前价在哪个支撑位附近？加仓参考位是哪个？
  - 若数据不足识别波段，请近似使用 MA60 附近作为低点，近30日高点作为高点

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
### 【Layer I】布林带宽度挤压与扩张诊断
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

计算当前布林带宽度：BandWidth = (上轨 - 下轨) / 中轨 × 100%

**挤压信号**：若当前BandWidth < 历史平均BandWidth × 0.5（布林带收窄至平均水平一半）
→ 触发「布林挤压」信号，预示即将出现大幅波动（方向不确定，需结合其他指标判断）
→ **"⚡【布林带挤压·变盘预警】布林带宽度收窄至极值，历史规律显示即将出现方向性大幅波动，请密切关注突破方向！"**

**股价位置诊断**：
  - 股价在上轨附近（>上轨×0.98）→ 过热预警，考虑减仓
  - 股价在中轨以上 → 健康持仓区间
  - 股价在中轨以下但在下轨以上 → 偏弱但持仓可接受
  - 股价触及下轨（<下轨×1.03）→ 超跌买入/加仓参考位

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
### 【Layer J】多周期时间框架共振确认
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

基于上方提供的数据，对以下维度进行多周期判断：

**日线级别（主要分析维度）**：
  - 均线排列：多头/空头/缠绕
  - MACD方向：DIFF与DEA的相对位置与方向
  - 价格结构：是否形成更高的高点和更高的低点

**中期视角（基于MA60/MA120）**：
  - MA60的斜率方向（若提供MA60数据）
  - 当前价格相对MA60的位置

**共振判断**：
  - 强共振（同向）：日线与中期均指向同一方向 → 信号可靠性最高
  - 弱共振（部分同向）：部分指标同向 → 信号可靠性中等
  - 背离（反向）：日线与中期相悖 → 需格外谨慎

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
### 【Layer K】量价关系综合诊断（OBV趋势/资金流向）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

基于提供的量价数据，执行以下量价关系诊断：

**量价健康矩阵**：
  | 价格方向 | 成交量方向 | 含义 | 对持仓者 |
  |---------|-----------|------|---------|
  | 上涨    | 放量      | 健康上涨，主力推升 | ✅ 继续持有/加仓 |
  | 上涨    | 缩量      | 上涨动能不足，注意减速 | ⚠️ 减仓警惕 |
  | 下跌    | 缩量      | 竭跌，抛压减轻 | ✅ 考虑加仓 |
  | 下跌    | 放量      | 主力出逃，危险信号 | ❌ 止损离场 |

**今日量价诊断**：判断今日属于上述哪种模式，并给出对持仓者的明确建议。

**OBV趋势近似**：
  - 若近3日价涨量增：OBV趋势向上（多头资金主导）
  - 若近3日价跌量增：OBV趋势向下（空头资金主导，出逃警报）
  - 若近3日缩量回调：OBV趋势平稳（正常回踩，持仓安全）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
### 【11层诊断汇总·全局强制约束】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**防幻觉最终确认**：
1. 在输出任何信号结论前，你必须已经检查上方技术面数据中是否存在所需的原始数值
2. 对于缺失数据的指标层（如ATR14需要历史高低数据），必须明确标注"数据不足，已使用近似计算"
3. 所有价格点位必须精确到小数点后两位（分）
4. 止损价计算路径必须透明：输出"成本价 × 0.92 = X元" 或 "ATR止损 = X元 - 2×X.XX = X.XX元"

**止损信号最优先原则（铁律）**：
若满足以下任一条件，无论其他信号多好，operation_advice 必须输出「止损」：
{f'  · 现价 ≤ 硬止损线 {hard_stop:.2f} 元（成本 {buy_price:.2f} × 0.92）' if hard_stop else '  · （未配置成本价，无法触发硬止损线）'}
  · Layer E2（放量跌破MA20）= TRUE
  · 当前乖离率(MA20) ≤ -15%（极度超跌且已跌破MA20）

**若11层全部未触发任何明确信号（全部数据缺失）**，必须在 technical_analysis 末尾附注：
**"⚙️【11层诺贝尔医学级量化扫描完毕】因底层数据接口未提供MACD/RSI/KDJ/ATR等细颗粒度指标数值，部分层级已使用可用数据近似计算。建议结合实时行情软件（通达信/同花顺）进行人工确认。以上分析仅供参考，不构成投资建议。"**

---

## ✍️ 最终分析输出要求

请为持仓中的 **{stock_name}({code})** 输出完整的 JSON 格式【持仓管理·诺贝尔医学级决策仪表盘】。

### 最关键输出项目（检查清单）
  □ `dashboard.pnl_dashboard.hard_stop_loss` 和 `.atr_stop_loss`（均精确到分）
  □ `dashboard.pnl_dashboard.risk_reward_ratio`（风险收益比数值）
  □ `dashboard.core_conclusion.has_position`（持仓者此刻的一句话指令）
  □ `dashboard.eleven_layer_diagnosis`（所有11层的诊断结论）
  □ `dashboard.battle_plan.sniper_points`（加仓点/止损线/两个目标位，全部精确到分）
  □ `technical_analysis`（包含所有11层的完整输出文字）
  □ `operation_advice`（最终操作建议，若任何止损信号触发必须是「止损」）
"""
        return prompt

    # ===================================================================
    # 格式化辅助方法（完全保持原版）
    # ===================================================================

    def _format_volume(self, volume: Optional[float]) -> str:
        if volume is None:
            return "N/A"
        if volume >= 1e8:
            return f"{volume / 1e8:.2f} 亿股"
        if volume >= 1e4:
            return f"{volume / 1e4:.2f} 万股"
        return f"{volume:.0f} 股"

    def _format_amount(self, amount: Optional[float]) -> str:
        if amount is None:
            return "N/A"
        if amount >= 1e8:
            return f"{amount / 1e8:.2f} 亿元"
        if amount >= 1e4:
            return f"{amount / 1e4:.2f} 万元"
        return f"{amount:.0f} 元"

    def _format_percent(self, value: Optional[float]) -> str:
        if value is None:
            return "N/A"
        try:
            return f"{float(value):.2f}%"
        except (TypeError, ValueError):
            return "N/A"

    def _format_price(self, value: Optional[float]) -> str:
        if value is None:
            return "N/A"
        try:
            return f"{float(value):.2f}"
        except (TypeError, ValueError):
            return "N/A"

    def _build_market_snapshot(self, context: Dict[str, Any]) -> Dict[str, Any]:
        today = context.get("today", {}) or {}
        realtime = context.get("realtime", {}) or {}
        yesterday = context.get("yesterday", {}) or {}
        prev_close = yesterday.get("close")
        close = today.get("close")
        high = today.get("high")
        low = today.get("low")
        amplitude = None
        change_amount = None
        if prev_close not in (None, 0) and high is not None and low is not None:
            try:
                amplitude = (float(high) - float(low)) / float(prev_close) * 100
            except (TypeError, ValueError, ZeroDivisionError):
                pass
        if prev_close is not None and close is not None:
            try:
                change_amount = float(close) - float(prev_close)
            except (TypeError, ValueError):
                pass
        snapshot = {
            "date": context.get("date", "未知"),
            "close": self._format_price(close),
            "open": self._format_price(today.get("open")),
            "high": self._format_price(high),
            "low": self._format_price(low),
            "prev_close": self._format_price(prev_close),
            "pct_chg": self._format_percent(today.get("pct_chg")),
            "change_amount": self._format_price(change_amount),
            "amplitude": self._format_percent(amplitude),
            "volume": self._format_volume(today.get("volume")),
            "amount": self._format_amount(today.get("amount")),
        }
        if realtime:
            snapshot.update(
                {
                    "price": self._format_price(realtime.get("price")),
                    "volume_ratio": realtime.get("volume_ratio", "N/A"),
                    "turnover_rate": self._format_percent(realtime.get("turnover_rate")),
                    "source": getattr(
                        realtime.get("source"), "value", realtime.get("source", "N/A")
                    ),
                }
            )
        return snapshot

    # ===================================================================
    # 完整性校验与重试（完全保持原版逻辑）
    # ===================================================================

    def _check_content_integrity(self, result: AnalysisResult) -> Tuple[bool, List[str]]:
        return check_content_integrity(result)

    def _build_integrity_complement_prompt(self, missing_fields: List[str]) -> str:
        lines = ["### 补全要求：请在上方分析基础上补充以下必填内容，并输出完整 JSON："]
        field_desc = {
            "sentiment_score": "sentiment_score: 0-100 综合评分",
            "operation_advice": "operation_advice: 加仓/持有/减仓/止盈/止损/观望",
            "analysis_summary": "analysis_summary: 150字持仓管理综合分析摘要",
            "dashboard.core_conclusion.one_sentence": "dashboard.core_conclusion.one_sentence: ≤30字持仓决策",
            "dashboard.intelligence.risk_alerts": "dashboard.intelligence.risk_alerts: 风险警报列表（可为空数组 []）",
            "dashboard.battle_plan.sniper_points.stop_loss": "dashboard.battle_plan.sniper_points.stop_loss: 综合止损价（精确到分）",
        }
        for f in missing_fields:
            lines.append(f"- {field_desc.get(f, f)}")
        return "\n".join(lines)

    def _build_integrity_retry_prompt(
        self,
        base_prompt: str,
        previous_response: str,
        missing_fields: List[str],
    ) -> str:
        complement = self._build_integrity_complement_prompt(missing_fields)
        return "\n\n".join(
            [
                base_prompt,
                "### 上一次输出如下，请在该输出基础上补齐缺失字段，重新输出完整 JSON，不要省略已有字段：",
                previous_response.strip(),
                complement,
            ]
        )

    def _apply_placeholder_fill(self, result: AnalysisResult, missing_fields: List[str]) -> None:
        apply_placeholder_fill(result, missing_fields)

    # ===================================================================
    # _parse_response（完全保持原版逻辑 + 新增11层字段提取）
    # ===================================================================

    def _parse_response(
        self, response_text: str, code: str, name: str
    ) -> AnalysisResult:
        try:
            cleaned = response_text
            if "```json" in cleaned:
                cleaned = cleaned.replace("```json", "").replace("```", "")
            elif "```" in cleaned:
                cleaned = cleaned.replace("```", "")

            json_start = cleaned.find("{")
            json_end = cleaned.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = cleaned[json_start:json_end]
                json_str = self._fix_json_string(json_str)
                data = json.loads(json_str)

                try:
                    AnalysisReportSchema.model_validate(data)
                except Exception as exc:
                    logger.warning("Schema validation warning: %s", str(exc)[:120])

                dashboard = data.get("dashboard")

                # 优先用 AI 返回的股票名称（如果当前名称是 fallback）
                ai_name = data.get("stock_name")
                if ai_name and (name.startswith("股票") or name == code or "Unknown" in name):
                    name = ai_name

                # decision_type 推断
                decision_type = data.get("decision_type", "")
                if not decision_type:
                    op = data.get("operation_advice", "持有")
                    if op in ("买入", "加仓", "强烈买入"):
                        decision_type = "buy"
                    elif op in ("卖出", "减仓", "强烈卖出", "止损", "止盈"):
                        decision_type = "sell"
                    else:
                        decision_type = "hold"

                # 当前价格提取
                current_price: Optional[float] = None
                if dashboard and "data_perspective" in dashboard:
                    pp = (dashboard["data_perspective"] or {}).get("price_position") or {}
                    current_price = _safe_float(pp.get("current_price")) or None
                if current_price is None:
                    try:
                        current_price = float(data.get("current_price") or 0) or None
                    except (TypeError, ValueError):
                        pass

                result = AnalysisResult(
                    code=code,
                    name=name,
                    sentiment_score=int(data.get("sentiment_score", 50)),
                    trend_prediction=data.get("trend_prediction", "震荡"),
                    operation_advice=data.get("operation_advice", "持有"),
                    decision_type=decision_type,
                    confidence_level=data.get("confidence_level", "中"),
                    dashboard=dashboard,
                    trend_analysis=data.get("trend_analysis", ""),
                    short_term_outlook=data.get("short_term_outlook", ""),
                    medium_term_outlook=data.get("medium_term_outlook", ""),
                    technical_analysis=data.get("technical_analysis", ""),
                    ma_analysis=data.get("ma_analysis", ""),
                    volume_analysis=data.get("volume_analysis", ""),
                    pattern_analysis=data.get("pattern_analysis", ""),
                    fundamental_analysis=data.get("fundamental_analysis", ""),
                    sector_position=data.get("sector_position", ""),
                    company_highlights=data.get("company_highlights", ""),
                    news_summary=data.get("news_summary", ""),
                    market_sentiment=data.get("market_sentiment", ""),
                    hot_topics=data.get("hot_topics", ""),
                    analysis_summary=data.get("analysis_summary", "分析完成"),
                    key_points=data.get("key_points", ""),
                    risk_warning=data.get("risk_warning", ""),
                    buy_reason=data.get("buy_reason", ""),
                    search_performed=data.get("search_performed", False),
                    data_sources=data.get("data_sources", "技术面数据"),
                    current_price=current_price,
                    success=True,
                )

                # 提取11层诊断中的扩展字段
                if dashboard and "eleven_layer_diagnosis" in dashboard:
                    diag = dashboard["eleven_layer_diagnosis"] or {}

                    # DNA基因得分
                    layer_a = diag.get("layer_A_dna_gene") or {}
                    gene_score_raw = layer_a.get("gene_score", "0/6")
                    try:
                        result.dna_gene_score = int(str(gene_score_raw).split("/")[0].strip())
                    except (ValueError, TypeError):
                        result.dna_gene_score = None

                    # 临床确诊得分
                    layer_b = diag.get("layer_B_clinical_timing") or {}
                    clinical_count = sum(
                        1 for k in ("bone_repair", "nerve_transmission", "muscle_activation", "blood_circulation")
                        if "TRUE" in str(layer_b.get(k, "")).upper()
                    )
                    result.clinical_score = clinical_count

                    # ATR止损
                    layer_g = diag.get("layer_G_atr_stop") or {}
                    atr_raw = layer_g.get("atr_stop_price")
                    if atr_raw:
                        try:
                            result.atr_stop_loss = float(
                                re.sub(r"[^0-9.]", "", str(atr_raw))
                            )
                        except (ValueError, TypeError):
                            pass

                    # 斐波那契水平
                    layer_h = diag.get("layer_H_fibonacci") or {}
                    fib_keys = ("fib_236", "fib_382", "fib_500", "fib_618", "fib_786")
                    fib_levels = {}
                    for k in fib_keys:
                        v = layer_h.get(k)
                        if v:
                            try:
                                fib_levels[k] = float(re.sub(r"[^0-9.]", "", str(v)))
                            except (ValueError, TypeError):
                                pass
                    if fib_levels:
                        result.fibonacci_levels = fib_levels

                    # 买入/卖出信号列表
                    layer_d = diag.get("layer_D_buy_signals") or {}
                    add_sigs = []
                    for k in ("b1_macd_divergence", "b2_oversold_neutral", "b3_consecutive_decline"):
                        v = str(layer_d.get(k, ""))
                        if "TRUE" in v.upper() or "触发" in v or "激活" in v:
                            add_sigs.append(v[:80])
                    result.add_position_signals = add_sigs or None

                    layer_e = diag.get("layer_E_sell_signals") or {}
                    exit_sigs = []
                    for k in ("c1_macd_top_divergence", "c2_volume_break_ma20", "c3_kdj_high_death_cross"):
                        v = str(layer_e.get(k, ""))
                        if "TRUE" in v.upper() or "触发" in v or "警报" in v:
                            exit_sigs.append(v[:80])
                    result.exit_signals = exit_sigs or None

                return result
            else:
                logger.warning("响应中未找到有效 JSON，降级文本解析")
                return self._parse_text_response(response_text, code, name)

        except json.JSONDecodeError as exc:
            logger.warning("JSON 解析失败: %s，降级文本解析", exc)
            return self._parse_text_response(response_text, code, name)

    def _fix_json_string(self, json_str: str) -> str:
        json_str = re.sub(r"//.*?\n", "\n", json_str)
        json_str = re.sub(r"/\*.*?\*/", "", json_str, flags=re.DOTALL)
        json_str = re.sub(r",\s*}", "}", json_str)
        json_str = re.sub(r",\s*]", "]", json_str)
        json_str = json_str.replace("True", "true").replace("False", "false")
        json_str = repair_json(json_str)
        return json_str

    def _parse_text_response(
        self, response_text: str, code: str, name: str
    ) -> AnalysisResult:
        """JSON解析完全失败时的文本降级解析。"""
        text_lower = response_text.lower()
        positive_kw = ["加仓", "看多", "买入", "上涨", "突破", "强势", "利好", "bullish"]
        negative_kw = ["止损", "卖出", "看空", "下跌", "跌破", "弱势", "利空", "减仓", "bearish"]
        pos = sum(1 for kw in positive_kw if kw in text_lower)
        neg = sum(1 for kw in negative_kw if kw in text_lower)
        if pos > neg + 1:
            score, trend, advice, dt = 65, "看多", "持有/关注加仓", "hold"
        elif neg > pos + 1:
            score, trend, advice, dt = 35, "看空", "减仓观察", "sell"
        else:
            score, trend, advice, dt = 50, "震荡", "持有", "hold"
        return AnalysisResult(
            code=code, name=name,
            sentiment_score=score, trend_prediction=trend,
            operation_advice=advice, decision_type=dt, confidence_level="低",
            analysis_summary=response_text[:500] if response_text else "无分析结果",
            key_points="JSON解析失败，仅供参考",
            risk_warning="分析结果可能不准确，建议结合其他信息综合判断",
            raw_response=response_text, success=True,
        )

    def batch_analyze(
        self,
        contexts: List[Dict[str, Any]],
        delay_between: float = 2.0,
    ) -> List[AnalysisResult]:
        results = []
        for i, context in enumerate(contexts):
            if i > 0:
                logger.debug("等待 %.1f 秒后继续...", delay_between)
                time.sleep(delay_between)
            results.append(self.analyze(context))
        return results


# =======================================================================
# 公共接口
# =======================================================================

def get_analyzer() -> GeminiAnalyzer:
    """获取【1208专属·诺贝尔医学级】持仓管理分析器实例。"""
    return GeminiAnalyzer()


def get_held_stocks_list() -> List[str]:
    """获取当前持仓股票代码列表（供 main.py / scheduler 调用）。"""
    return list(HELD_STOCKS.keys())


def get_held_stocks_full() -> Dict[str, Dict[str, Any]]:
    """获取完整持仓信息字典（供报告生成模块调用）。"""
    return dict(HELD_STOCKS)


# =======================================================================
# __main__ 测试入口
# =======================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # 模拟爱仕达（002403）持仓测试场景
    # 在 STOCK_LIST 环境变量中配置：002403:8.50
    # 或在此处临时设置（仅用于测试）：
    os.environ.setdefault("STOCK_LIST", "002403:8.50")
    # 重新加载持仓（因为已在模块顶层加载了一次）
    HELD_STOCKS.update(_load_held_stocks())

    test_context: Dict[str, Any] = {
        "code": "002403",
        "stock_name": "爱仕达",
        "date": "2026-03-17",
        "today": {
            "open": 8.50,
            "high": 8.76,
            "low": 8.38,
            "close": 8.65,
            "volume": 5_800_000,
            "amount": 50_200_000,
            "pct_chg": 1.76,
            "ma5": 8.52,
            "ma10": 8.45,
            "ma20": 8.30,
            "ma60": 8.10,
        },
        "yesterday": {
            "close": 8.50,
            "volume": 5_200_000,
        },
        "realtime": {
            "price": 8.65,
            "volume_ratio": 1.12,
            "volume_ratio_desc": "量比1.12，平量",
            "turnover_rate": 3.21,
            "pe_ratio": 18.5,
            "pb_ratio": 1.8,
            "total_mv": 3_800_000_000,
            "circ_mv": 3_200_000_000,
            "change_60d": 12.3,
        },
        "chip": {
            "profit_ratio": 0.62,
            "avg_cost": 8.28,
            "concentration_90": 0.12,
            "concentration_70": 0.08,
            "chip_status": "筹码集中",
        },
        "ma_status": "多头排列 📈",
        "trend_analysis": {
            "trend_status": "上升趋势",
            "ma_alignment": "多头排列（MA5>MA10>MA20）",
            "trend_strength": 68,
            "bias_ma5": 1.53,
            "bias_ma10": 2.37,
            "bias_ma20": 4.22,
            "volume_status": "平量回踩",
            "buy_signal": "可持有/关注加仓",
            "signal_score": 65,
            "signal_reasons": ["多头排列成立", "乖离率在安全范围内", "缩量回踩均线正常"],
            "risk_factors": ["量能偏弱，需关注后续放量情况"],
        },
        "recent_closes": [8.72, 8.58, 8.50, 8.55, 8.65],
        "recent_opens": [8.60, 8.72, 8.55, 8.48, 8.50],
        "volume_change_ratio": 1.12,
        "price_change_ratio": 1.76,
    }

    print("=" * 70)
    print("【1208持仓分析·诺贝尔医学级终极版 v4.0】测试运行")
    print(f"当前持仓：{get_held_stocks_list()}")
    print(f"持仓详情：{get_held_stocks_full()}")
    print("=" * 70)

    analyzer = GeminiAnalyzer()
    if analyzer.is_available():
        result = analyzer.analyze(test_context)
        print(f"\n{'='*70}")
        print(f"股票：{result.name}({result.code})")
        print(f"操作建议：{result.get_emoji()} {result.operation_advice}  {result.get_confidence_stars()}")
        print(f"综合评分：{result.sentiment_score}/100  ({result.trend_prediction})")
        print(f"DNA基因得分：{result.dna_gene_score}/6")
        print(f"临床确诊得分：{result.clinical_score}/4")
        if result.pnl_info:
            pnl = result.pnl_info
            print(f"盈亏状态：{pnl.get('pnl_status')} {pnl.get('pnl_pct')}%")
            print(f"硬止损线：{pnl.get('stop_loss_price')} 元  |  距止损安全边际：{pnl.get('distance_to_stop_pct')}%")
            print(f"止损是否触发：{'❌ 是，请立即止损！' if pnl.get('stop_loss_triggered') else '✅ 否，持仓安全'}")
        if result.atr_stop_loss:
            print(f"ATR自适应止损：{result.atr_stop_loss:.2f} 元")
        if result.fibonacci_levels:
            print(f"斐波那契关键位：{result.fibonacci_levels}")
        if result.add_position_signals:
            print(f"加仓信号：{result.add_position_signals}")
        if result.exit_signals:
            print(f"⚠️ 卖出信号：{result.exit_signals}")
        print(f"\n核心结论：{result.get_core_conclusion()}")
        sniper = result.get_sniper_points()
        if sniper:
            print(f"\n狙击点位：")
            for k, v in sniper.items():
                print(f"  {k}: {v}")
        print(f"\n风险提示：{result.risk_warning}")
        print("=" * 70)
    else:
        print("\n⚠️  LLM API 未配置，跳过测试")
        print("   请在 .env 或 GitHub Variables 中设置 LITELLM_MODEL 和对应 API Key")
        print("   STOCK_LIST 格式：002403:8.50")
