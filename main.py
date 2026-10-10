"""Campux 稿件审核助手 —— AstrBot 插件

审核群内发送 `//bot <稿件id>` 时：
  1. 依据群号找到该群绑定的稿件规则；
  2. 从本群早前收到的 Campux 审核通知中取出该稿件的正文与元信息；
  3. （可选）用稿件内容联网检索相关信息；
  4. 调用大模型，按规则判定是否违规，输出【通过/拒绝】建议、原因与发布影响。

Campux 审核通知样例（群内消息）:
    📮 中山中学校园墙 新稿件
    编号：#6928
    投稿人：匿名（QQ 257xxxxx0）
    来源：网页投稿
    图片：0 张

    今天初一二班中午在三楼排煮粉的那几个男生想干嘛 别人不用吃了吗 插个队还那么理直气壮

    通过：#通过 6928
    拒绝：#拒绝 <理由> 6928
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register

# ---------------------------------------------------------------------------
# 解析 Campux 审核通知
# ---------------------------------------------------------------------------

# 编号：#6928 / 编号: 6928 / 编号：6928
_RE_ID = re.compile(r"编\s*号\s*[:：]\s*#?\s*(\d{1,12})")
_RE_WALL = re.compile(r"^\s*(?:📮\s*)?(.+?)\s*新稿件\s*$", re.M)
_RE_SUBMITTER = re.compile(r"投稿人\s*[:：]\s*(.+)")
_RE_SOURCE = re.compile(r"来源\s*[:：]\s*(.+)")
_RE_IMAGES = re.compile(r"图片\s*[:：]\s*(\d+)\s*张")
# 通过/拒绝说明行（用于从正文中剔除）
_RE_FOOTER = re.compile(r"^\s*(通过|拒绝)\s*[:：]")
_RE_HASH_ID = re.compile(r"#\s*(\d{1,12})")


@dataclass
class CampuxPost:
    post_id: str
    wall: str = ""
    submitter: str = ""
    source: str = ""
    image_count: int = 0
    content: str = ""
    raw: str = ""
    ts: float = field(default_factory=time.time)


def parse_campux_notification(text: str) -> CampuxPost | None:
    """尝试把一段群消息解析为 Campux 稿件审核通知，失败返回 None。"""
    if not text:
        return None
    # 必须有「编号：#xxx」，这是最可靠的锚点
    m_id = _RE_ID.search(text)
    if not m_id:
        return None
    # 且应带有新稿件 / 投稿人 之类特征，避免误抓普通聊天
    if ("新稿件" not in text) and ("投稿人" not in text):
        return None

    post = CampuxPost(post_id=m_id.group(1))
    if (m := _RE_WALL.search(text)):
        post.wall = m.group(1).strip()
    if (m := _RE_SUBMITTER.search(text)):
        post.submitter = m.group(1).strip()
    if (m := _RE_SOURCE.search(text)):
        post.source = m.group(1).strip()
    if (m := _RE_IMAGES.search(text)):
        try:
            post.image_count = int(m.group(1))
        except ValueError:
            post.image_count = 0

    # 正文 = 去掉头部元信息行与底部「通过/拒绝」说明行后的剩余内容
    body_lines: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if _RE_WALL.match(s) or _RE_ID.search(s):
            continue
        if _RE_SUBMITTER.search(s) or _RE_SOURCE.search(s) or _RE_IMAGES.search(s):
            continue
        if _RE_FOOTER.match(s):          # 通过：#通过 6928
            continue
        if "新稿件" in s:
            continue
        body_lines.append(s)
    post.content = "\n".join(body_lines).strip()
    post.raw = text
    return post


# ---------------------------------------------------------------------------
# 稿件内容缓存（按 群号 -> 稿件id）
# ---------------------------------------------------------------------------

class PostCache:
    """被动缓存群内出现过的 Campux 稿件通知。

    AstrBot 插件通常拿不到任意群的历史消息，因此这里被动监听群消息，
    把解析成功的审核通知按 (群号, 稿件id) 存下来，供 //bot <id> 查询。

    为免长期占用内存，做四重约束：
      1) TTL：超过 ttl 秒的条目自动失效（默认 72h / 3 天）；
      2) 单群条数上限 max_per_group（默认 50）；
      3) 群数量上限 max_groups（默认 50），超出淘汰最久未活动的群；
      4) 正文长度上限 max_content_len（默认 2000 字），超长截断。
      另有 cleanup() 可被定时调用，主动清掉过期项。
    """

    def __init__(
        self,
        max_per_group: int = 50,
        ttl: float = 24 * 3600,
        max_groups: int = 50,
        max_content_len: int = 2000,
    ):
        self._data: dict[str, dict[str, CampuxPost]] = {}
        self._group_seen: dict[str, float] = {}   # 群号 -> 最近活动时间
        self._max = max_per_group
        self._ttl = ttl
        self._max_groups = max_groups
        self._max_content_len = max_content_len

    def put(self, group_id: str, post: CampuxPost) -> None:
        g = str(group_id)
        # 正文过长则截断，避免囤积大文本
        if post.content and len(post.content) > self._max_content_len:
            post.content = post.content[: self._max_content_len] + "…"
        bucket = self._data.setdefault(g, {})
        bucket[post.post_id] = post
        self._group_seen[g] = post.ts
        # 单群超限：淘汰最旧的
        while len(bucket) > self._max:
            oldest = min(bucket.items(), key=lambda kv: kv[1].ts)[0]
            bucket.pop(oldest, None)
        # 群数超限：淘汰最久未活动的群
        if len(self._data) > self._max_groups:
            stalest = min(self._group_seen.items(), key=lambda kv: kv[1])[0]
            self._data.pop(stalest, None)
            self._group_seen.pop(stalest, None)

    def get(self, group_id: str, post_id: str) -> CampuxPost | None:
        post = self._data.get(str(group_id), {}).get(str(post_id))
        if post is None:
            return None
        # 命中时顺手判过期
        if self._ttl > 0 and (time.time() - post.ts) > self._ttl:
            self._data.get(str(group_id), {}).pop(str(post_id), None)
            return None
        return post

    def cleanup(self) -> int:
        """清理过期条目与空群，返回删除的条目数。"""
        now = time.time()
        removed = 0
        for g in list(self._data.keys()):
            bucket = self._data[g]
            for pid in list(bucket.keys()):
                if self._ttl > 0 and (now - bucket[pid].ts) > self._ttl:
                    bucket.pop(pid, None)
                    removed += 1
            if not bucket:
                self._data.pop(g, None)
                self._group_seen.pop(g, None)
        return removed

    def stats(self) -> dict:
        return {
            "groups": len(self._data),
            "posts": sum(len(b) for b in self._data.values()),
        }


def _extract_plain(event: AstrMessageEvent) -> str:
    """尽量从事件里取出纯文本（含被 strip 掉的换行尽量还原）。"""
    try:
        return event.message_str or ""
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# 插件主体
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 联网检索（请求细节固定在插件内，配置页只需填 API Key）
# ---------------------------------------------------------------------------

# AnySearch 搜索接口（固定）
ANYSEARCH_URL = "https://api.anysearch.com/v1/search"
# API Key 申请地址（用于配置页提示）
ANYSEARCH_KEY_URL = "https://anysearch.com"


def _search_via_anysearch_sync(query: str, top_k: int, api_key: str) -> str:
    """调用 AnySearch 检索，返回拼接文本；失败返回空串。

    请求格式固定在本插件内，用户只需在配置页填 API Key。
    """
    import json as _json
    import urllib.request as _url

    api_key = str(api_key or "").strip()
    if not api_key:
        logger.warning("[campux_review] 未配置搜索 API Key，跳过联网检索")
        return ""

    body = {"query": query, "max_results": int(top_k)}
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    req = _url.Request(
        ANYSEARCH_URL, data=_json.dumps(body).encode("utf-8"),
        headers=headers, method="POST",
    )
    try:
        with _url.urlopen(req, timeout=20) as resp:
            data = _json.loads(resp.read().decode("utf-8", "replace"))
    except Exception as e:
        logger.warning(f"[campux_review] 联网检索失败: {e}")
        return ""

    results = _extract_search_results(data)
    if not results:
        return ""
    lines = []
    for i, r in enumerate(results[:top_k], 1):
        title = str(r.get("title") or r.get("name") or "").strip()
        body_txt = str(
            r.get("snippet") or r.get("body") or r.get("description")
            or r.get("content") or r.get("summary") or ""
        ).strip()
        href = str(r.get("url") or r.get("link") or r.get("href") or "").strip()
        lines.append(f"{i}. {title}\n   {body_txt}\n   {href}")
    return "\n".join(lines)


def _extract_search_results(data) -> list[dict]:
    """从任意搜索接口返回中，尽量找出结果列表（兼容多种字段名）。"""
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if not isinstance(data, dict):
        return []
    # 常见包裹字段
    for key in ("results", "items", "data", "web", "organic", "hits", "documents"):
        v = data.get(key)
        if isinstance(v, list):
            return [x for x in v if isinstance(x, dict)]
        if isinstance(v, dict):  # 再钻一层，如 data.results
            for k2 in ("results", "items", "list", "web", "organic"):
                v2 = v.get(k2)
                if isinstance(v2, list):
                    return [x for x in v2 if isinstance(x, dict)]
    return []


async def _do_web_search(query: str, top_k: int, api_key: str = "") -> str:
    """联网检索。用 AnySearch 接口（请求细节固定在插件内，只需 Key）。"""
    if not query.strip():
        return ""
    return await asyncio.to_thread(_search_via_anysearch_sync, query, top_k, api_key)


def _build_prompt(post: CampuxPost, rules: str, search_text: str, school_info: dict | None = None) -> str:
    school_info = school_info or {}
    parts = [
        "你是校园墙（Campux）的稿件审核助手。请严格依据下面给出的《稿件规则》，"
        "判断这条稿件是否应当通过。",
        "",
        "【审核对象背景】（用于判断稿件中提到的时间/地点/人物/事件/热梗是否可核实的语境）",
    ]
    school_name = str(school_info.get("school_name", "") or "").strip()
    region = str(school_info.get("region", "") or "").strip()
    wall_name = str(school_info.get("wall_name", "") or "").strip() or (post.wall or "")
    if school_name:
        parts.append(f"学校：{school_name}")
    if region:
        parts.append(f"所在地区：{region}")
    if wall_name:
        parts.append(f"校园墙名称：{wall_name}")
    if not (school_name or region or wall_name):
        parts.append("（未配置，仅依据稿件本身判断）")

    parts += [
        "",
        "【稿件规则】",
        rules.strip() or "（该群未配置具体规则，请依据通用校园墙审核常识判断：真实姓名/班级曝光、人身攻击、"
        "广告、违法不良信息、隐私泄露等应拒绝。）",
        "",
        "【稿件信息】",
        f"稿件编号：{post.post_id}",
    ]
    if post.wall:
        parts.append(f"所属校园墙：{post.wall}")
    if post.submitter:
        parts.append(f"投稿人：{post.submitter}")
    if post.source:
        parts.append(f"来源：{post.source}")
    parts.append(f"图片数量：{post.image_count} 张")
    parts.append("")
    parts.append("【稿件正文】")
    parts.append(post.content or "（空）")

    if search_text:
        parts += [
            "",
            "【联网检索到的相关信息】（仅为辅助，用于核实稿件中的地名/事件/热梗；"
            "若与稿件无关请忽略，不要据此臆测）",
            search_text,
        ]

    parts += [
        "",
        "请按以下格式输出（务必简洁，不要输出多余内容）：",
        "结论：通过 / 拒绝",
        "原因：<依据具体规则逐条说明>",
        "发布影响：<该稿件若发布可能带来的影响，1~3 句>",
        "若信息不足以判断，请说明还需要哪些信息。",
    ]
    return "\n".join(parts)


@register(
    "astrbot_plugin_campux_review",
    "haohaoxuedili",
    "配合 Campux 的稿件审核助手：//bot <稿件id> 触发联网核查 + 规则判定",
    "1.0.0",
)
class CampuxReviewPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig | None = None):
        super().__init__(context)
        self.config = config or {}
        self.cache = PostCache(
            max_per_group=int(self.config.get("cache_max_per_group", 50) or 50),
            ttl=float(self.config.get("cache_ttl_hours", 72) or 72) * 3600,
            max_groups=int(self.config.get("cache_max_groups", 50) or 50),
        )
        self._cleanup_task: asyncio.Task | None = None

    async def initialize(self):
        logger.info("[campux_review] Campux 稿件审核助手已加载")
        # 后台定期清理过期缓存，避免长期占用内存
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def terminate(self):
        """插件卸载时停止清理任务并清空缓存。"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except (asyncio.CancelledError, Exception):
                pass
            self._cleanup_task = None
        self.cache = PostCache()
        logger.info("[campux_review] 插件已卸载，缓存已清空")

    async def _cleanup_loop(self):
        """每小时清理一次过期稿件缓存。"""
        try:
            while True:
                await asyncio.sleep(3600)
                removed = self.cache.cleanup()
                if removed:
                    logger.info(
                        f"[campux_review] 定期清理过期稿件缓存 {removed} 条，"
                        f"当前 {self.cache.stats()}"
                    )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("[campux_review] 缓存清理任务异常退出")

    # -- 被动监听：缓存群内 Campux 审核通知 -------------------------------
    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE)
    async def on_group_message(self, event: AstrMessageEvent):
        text = _extract_plain(event)
        post = parse_campux_notification(text)
        if not post:
            return
        group_id = event.get_group_id() or ""
        if not group_id:
            return
        self.cache.put(group_id, post)
        logger.info(
            f"[campux_review] 已缓存稿件 #{post.post_id}（群 {group_id}），"
            f"正文 {len(post.content)} 字"
        )

    # -- 帮助 ---------------------------------------------------------------
    @filter.command("campux审核")
    async def help_cmd(self, event: AstrMessageEvent):
        prefix = self.config.get("trigger_prefix", "//bot")
        yield event.plain_result(
            "Campux 稿件审核助手\n"
            f"用法：在审核群发送 {prefix} <稿件id>，例如 {prefix} 5555\n"
            "插件会自动读取该稿件的通知内容、联网检索，并按本群绑定的规则给出通过/拒绝建议。"
        )

    # -- 审核触发 -----------------------------------------------------------
    # 用普通群消息监听手动匹配前缀（与缓存监听同一机制，兼容性最好，
    # 不依赖 AstrBot 的指令解析层，避免 // 前缀被截走而失效）。
    @filter.event_message_type(filter.EventMessageType.GROUP_MESSAGE)
    async def review(self, event: AstrMessageEvent):
        _raw = _extract_plain(event)
        logger.info(f"[campux_review][debug] review 收到原文: {_raw!r}")
        post_id = self._match_id(_raw)
        if not post_id:
            logger.info("[campux_review][debug] _match_id 未匹配，忽略本条")
            return
        group_id = event.get_group_id() or ""
        # 命中后阻止后续默认 LLM 回复，避免又被当普通聊天回一句「12喵~」
        try:
            event.stop_event()
        except Exception:
            pass

        logger.info(f"[campux_review] 收到审核指令 #{post_id}（群 {group_id}）")

        entry = self._find_rule_entry(group_id)
        if entry is None:
            return event.plain_result(
                f"❌ 当前群（{group_id or '私聊'}）未在插件配置中绑定稿件规则，"
                "请先在 WebUI → 插件配置 中添加「审核 QQ 群号 ↔ 稿件规则」对应项。"
            )
        rules = str(entry.get("rules", "") or "")
        school_info = self._school_info_for(entry)

        post = self.cache.get(group_id, post_id)
        if post is None:
            return event.plain_result(
                f"❌ 未找到稿件 #{post_id} 的通知内容。\n"
                "请确认：1) 该稿件通知已在本群出现过；2) 插件已在通知之后加载。\n"
                "（插件只能记住自己加载后收到的通知；若刚装好插件，请让稿件重新通知一次。）"
            )

        # 先主动发一条进度提示（不占用最终返回值）
        try:
            await event.send(
                event.plain_result(
                    f"🔍 正在审核稿件 #{post_id}（{post.wall or '未知墙'}）…"
                )
            )
        except Exception:
            logger.exception("[campux_review] 发送进度提示失败（继续审核）")

        # 联网检索
        search_text = ""
        if self.config.get("enable_web_search", True):
            top_k = int(self.config.get("max_search_results", 5) or 5)
            query = (post.content or "").strip()[:120]
            if query:
                logger.info(f"[campux_review] 开始联网检索：{query[:40]}")
                try:
                    search_text = await asyncio.wait_for(
                        _do_web_search(
                            query, top_k, str(self.config.get("search_api_key", "") or "")
                        ),
                        timeout=25,
                    )
                except asyncio.TimeoutError:
                    logger.warning("[campux_review] 联网检索超时（25s），跳过检索继续审核")
                    search_text = ""
                logger.info(
                    f"[campux_review] 检索完成，结果长度 {len(search_text)}"
                )

        # 调模型
        prompt = _build_prompt(post, rules, search_text, school_info)
        try:
            provider_id = (self.config.get("llm_provider_id") or "").strip()
            if not provider_id:
                provider_id = self._default_provider_from_config()
            if not provider_id:
                logger.info("[campux_review] 未配置审核模型，尝试获取当前会话模型…")
                try:
                    provider_id = await asyncio.wait_for(
                        self.context.get_current_chat_provider_id(
                            umo=event.unified_msg_origin
                        ),
                        timeout=10,
                    )
                except asyncio.TimeoutError:
                    logger.warning("[campux_review] 获取当前会话模型超时（10s）")
                    return event.plain_result(
                        "❌ 审核失败：获取当前会话模型超时。\n"
                        "请在插件配置的「审核用大模型」里直接选择一个模型后重试。"
                    )
            if not provider_id:
                return event.plain_result(
                    "❌ 未找到可用的对话模型。\n"
                    "请在插件配置的「审核用大模型」中指定一个模型。"
                )
            logger.info(f"[campux_review] 开始调用模型 provider={provider_id!r}")
            resp = await asyncio.wait_for(
                self.context.llm_generate(
                    chat_provider_id=provider_id,
                    prompt=prompt,
                ),
                timeout=90,
            )
            answer = (resp.completion_text or "").strip()
            logger.info(f"[campux_review] 模型返回，长度 {len(answer)}")
        except asyncio.TimeoutError:
            logger.warning("[campux_review] 调用大模型超时（90s）")
            return event.plain_result(
                "❌ 审核失败：调用大模型超时（90 秒）。\n"
                "通常是中转 API 不稳定或模型过慢，请稍后重试或更换模型。"
            )
        except Exception as e:
            logger.exception("[campux_review] 调用大模型失败")
            return event.plain_result(f"❌ 审核失败：调用大模型出错：{e}")

        if not answer:
            answer = "（模型未返回内容）"

        header = f"📋 稿件 #{post.post_id} 审核建议" + (f" · {post.wall}" if post.wall else "")
        footer_bits = []
        if search_text:
            footer_bits.append("已联网核查")
        footer_bits.append("审核结果仅供参考，最终由人工确认")

        text = f"{header}\n\n{answer}\n\n———\n" + "｜".join(footer_bits)

        # OneBot 会 strip 前后空白，用零宽空格保护格式
        return event.plain_result("\u200b" + text + "\u200b")
    # -- 工具方法 -----------------------------------------------------------
    def _default_provider_from_config(self) -> str:
        """从 AstrBot 全局配置里读取默认对话模型（不发起网络请求，不会卡）。"""
        try:
            ctx = self.context
            cfg = getattr(ctx, "_config", None) or getattr(ctx, "config", None)
            if isinstance(cfg, dict):
                pid = cfg.get("provider_id") or cfg.get("provider_settings", {}).get(
                    "default_provider_id"
                )
                if pid:
                    return str(pid).strip()
        except Exception:
            pass
        return ""

    def _match_id(self, text: str) -> str | None:
        """按配置的触发前缀解析稿件 id。

        兼容场景：AstrBot / OneBot 可能在消息进入插件前剥掉开头的指令前缀
        （例如 `//bot 12` 变成 `bot 12` 或 `/bot 12`），因此这里对前缀做
        「去斜杠」「忽略大小写」「忽略前后空白」的宽松匹配。
        """
        prefix = str(self.config.get("trigger_prefix", "//bot") or "//bot").strip()
        if not prefix or not text:
            return None

        raw = text.strip()
        # 候选前缀：原前缀 + 去掉全部前导斜杠的版本 + 去掉一个斜杠的版本
        pref_forms = {prefix}
        pref_forms.add(prefix.lstrip("/"))
        if prefix.startswith("/"):
            pref_forms.add(prefix[1:])

        # 候选文本：原文 + 去掉全部/一个前导斜杠的版本
        text_forms = {raw}
        if raw.startswith("/"):
            text_forms.add(raw.lstrip("/"))
            text_forms.add(raw[1:])

        for p in pref_forms:
            if not p:
                continue
            for t in text_forms:
                m = re.match(
                    r"^\s*" + re.escape(p) + r"\s+(\d{1,12})\s*$",
                    t,
                    re.IGNORECASE,
                )
                if m:
                    return m.group(1)

        # 兜底：文本中以「前缀（忽略斜杠/大小写）+ 空白 + 数字」结尾即命中
        m = re.search(
            r"(?:^|\s|[\u4e00-\u9fff])" + re.escape(prefix.lstrip("/"))
            + r"\s+(\d{1,12})\s*$",
            raw,
            re.IGNORECASE,
        )
        return m.group(1) if m else None

    def _find_rule_entry(self, group_id: str) -> dict | None:
        """按群号找规则条目；未命中返回 None。"""
        items = self.config.get("review_rules") or []
        # 兼容 template_list 保存格式（list[dict]）
        if isinstance(items, dict):  # 极少数情况被存成 dict
            items = list(items.values())
        for item in items:
            if not isinstance(item, dict):
                continue
            gid = str(item.get("group_id", "")).strip()
            if gid and gid == str(group_id):
                return item
        return None

    def _school_info_for(self, entry: dict | None) -> dict:
        """把规则条目里的学校背景信息整理成 prompt 用的 dict。"""
        entry = entry or {}
        return {
            "school_name": str(entry.get("school_name", "") or "").strip(),
            "region": str(entry.get("region", "") or "").strip(),
            "wall_name": str(entry.get("wall_name", "") or "").strip(),
        }
