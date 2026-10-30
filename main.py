import random
import time
from collections import defaultdict, deque
from pathlib import Path

from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.event.filter import CustomFilter
from astrbot.api.message_components import At, AtAll, Reply
from astrbot.api.star import Context, Star


CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "config"
    / "astrbot_plugin_ambient_reply_config.json"
)

DEFAULT_CONFIG = {
    "enabled": True,
    "enabled_group_ids": [],
    "ignored_sender_ids": [],
    "reply_probability": 0.33,
    "high_frequency_probability": 0.8,
    "high_frequency_duration_seconds": 90,
    "min_group_interval_seconds": 75,
    "min_user_interval_seconds": 180,
    "max_replies_per_10_minutes": 3,
    "min_text_chars": 2,
}

_config_cache: dict = dict(DEFAULT_CONFIG)
_config_mtime_ns: int | None = None
_last_group_reply: dict[str, float] = defaultdict(float)
_last_user_reply: dict[tuple[str, str], float] = defaultdict(float)
_recent_replies: dict[str, deque[float]] = defaultdict(deque)
_high_frequency_until: dict[str, float] = defaultdict(float)


def _load_config() -> dict:
    global _config_cache, _config_mtime_ns

    try:
        mtime_ns = CONFIG_PATH.stat().st_mtime_ns
    except OSError:
        return _config_cache

    if mtime_ns == _config_mtime_ns:
        return _config_cache

    import json

    try:
        loaded = json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return _config_cache

    merged = dict(DEFAULT_CONFIG)
    if isinstance(loaded, dict):
        merged.update(loaded)
    _config_cache = merged
    _config_mtime_ns = mtime_ns
    return _config_cache


def _group_matches(event: AstrMessageEvent, settings: dict) -> bool:
    configured = {
        str(value).strip()
        for value in settings.get("enabled_group_ids", [])
        if str(value).strip()
    }
    if not configured:
        return True
    group_id = str(event.get_group_id() or "")
    return group_id in configured or event.unified_msg_origin in configured


def _has_at_or_self_reply(event: AstrMessageEvent) -> bool:
    self_id = str(event.get_self_id())
    for component in event.get_messages():
        if isinstance(component, At) and str(component.qq) == self_id:
            return True
        if isinstance(component, Reply) and str(component.sender_id) == self_id:
            return True
    return False


def _has_at_all(event: AstrMessageEvent) -> bool:
    return any(isinstance(component, AtAll) for component in event.get_messages())


def _is_ignored_sender(event: AstrMessageEvent, settings: dict) -> bool:
    sender_id = str(event.get_sender_id())
    ignored = {
        str(value).strip()
        for value in settings.get("ignored_sender_ids", [])
        if str(value).strip()
    }
    return sender_id in ignored


class AmbientReplyFilter(CustomFilter):
    """Wake the bot for selected ordinary group messages."""

    def filter(self, event: AstrMessageEvent, cfg) -> bool:
        settings = _load_config()
        if not bool(settings.get("enabled", True)):
            return False
        if event.get_extra("_keyword_wake", False):
            return False
        if event.is_private_chat() or not _group_matches(event, settings):
            return False
        if (
            str(event.get_sender_id()) == str(event.get_self_id())
            or _is_ignored_sender(event, settings)
        ):
            return False
        if _has_at_or_self_reply(event) or _has_at_all(event):
            return False

        text = event.message_str.strip()
        min_chars = max(0, int(settings.get("min_text_chars", 2)))
        if text.startswith(("/", "!", "#sl")):
            return False

        now = time.monotonic()
        group_id = str(event.get_group_id())
        sender_id = str(event.get_sender_id())
        user_key = (group_id, sender_id)
        high_frequency = now < _high_frequency_until[group_id]

        recent = _recent_replies[group_id]
        if len(text) < min_chars:
            return False

        if not high_frequency:
            if now - _last_group_reply[group_id] < float(
                settings.get("min_group_interval_seconds", 75)
            ):
                return False
            if now - _last_user_reply[user_key] < float(
                settings.get("min_user_interval_seconds", 180)
            ):
                return False

            while recent and now - recent[0] >= 600:
                recent.popleft()
            if len(recent) >= max(
                0,
                int(settings.get("max_replies_per_10_minutes", 3)),
            ):
                return False

        probability_key = (
            "high_frequency_probability"
            if high_frequency
            else "reply_probability"
        )
        probability = min(
            1.0,
            max(0.0, float(settings.get(probability_key, 0.33))),
        )
        if random.random() >= probability:
            return False

        _last_group_reply[group_id] = now
        _last_user_reply[user_key] = now
        recent.append(now)
        _high_frequency_until[group_id] = now + max(
            0.0,
            float(settings.get("high_frequency_duration_seconds", 90)),
        )
        event.set_extra("_ambient_reply", True)
        event.set_extra("_ambient_high_frequency", high_frequency)
        event.is_at_or_wake_command = True
        event.is_wake = True
        return True


class AmbientReplyPlugin(Star):
    """Let the bot join ordinary group chat at a restrained rate."""

    def __init__(self, context: Context) -> None:
        super().__init__(context)

    @filter.event_message_type(
        filter.EventMessageType.GROUP_MESSAGE,
        priority=10000,
    )
    async def block_ignored_bot_senders(self, event: AstrMessageEvent) -> None:
        settings = _load_config()
        if not _is_ignored_sender(event, settings):
            return
        self.logger.debug(
            "Ignored bot sender %s in %s",
            event.get_sender_id(),
            event.unified_msg_origin,
        )
        event.stop_event()

    @filter.custom_filter(AmbientReplyFilter)
    async def mark_ambient_reply(self, event: AstrMessageEvent) -> None:
        event.set_extra("_ambient_reply", True)
        self.logger.info(
            "Ambient reply selected for %s sender=%s",
            event.unified_msg_origin,
            event.get_sender_id(),
        )

    @filter.on_llm_request()
    async def add_ambient_prompt(
        self,
        event: AstrMessageEvent,
        *args,
    ) -> None:
        req = next(
            (
                candidate
                for candidate in args
                if hasattr(candidate, "system_prompt")
            ),
            None,
        )
        if req is None:
            self.logger.warning(
                "Ambient LLM request hook did not receive a ProviderRequest: %s",
                [type(candidate).__name__ for candidate in args],
            )
            return
        if not event.get_extra("_ambient_reply", False):
            return
        req.system_prompt += (
            "\n\n[群聊自然接话规则]\n"
            "这条消息没有直接呼唤你。你现在是在群聊中自主决定插话。"
            "如果消息里 @ 了其他人，要明确理解它是在对那个人说话，不是在叫你；"
            "先看清谁在对谁说话，再决定这套内容是否适合你自然接一句，"
            "不要把别人之间的对话误当成对你的指令。"
            "只回应与你确实相关、能自然接上的内容；通常只说一句，简短口语化。"
            "不要复述消息，不要解释自己为什么说话，"
            "不要突然把话题转向自己，也不要主动列建议。"
        )
