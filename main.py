"""
Production-ready Discord bot with AI chat (Google Gemini), moderation,
anti-spam, profanity filtering, logging, and SQLite persistence.

Single-file implementation. Requires a .env file with:
    DISCORD_TOKEN=
    GEMINI_API_KEY=
    OWNER_ID=
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import re
import sys
import time
import traceback
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple

import aiosqlite
import discord
from discord import app_commands
from discord.ext import commands, tasks
from dotenv import load_dotenv

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover
    genai = None
    genai_types = None


# --------------------------------------------------------------------------- #
#                                CONFIGURATION                                #
# --------------------------------------------------------------------------- #

load_dotenv()


class Colors:
    """ANSI color codes for terminal startup logging."""

    RESET = "\033[0m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    BOLD = "\033[1m"


def _colored_log(message: str, color: str = Colors.CYAN) -> None:
    print(f"{color}{Colors.BOLD}[STARTUP]{Colors.RESET} {color}{message}{Colors.RESET}")


class Config:
    """Central configuration loaded from environment variables."""

    DISCORD_TOKEN: str = os.getenv("DISCORD_TOKEN", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    OWNER_ID: int = int(os.getenv("OWNER_ID", "0") or "0")

    DB_PATH: str = os.getenv("DB_PATH", "bot_data.db")
    DB_BACKUP_DIR: str = os.getenv("DB_BACKUP_DIR", "backups")

    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    AI_MAX_HISTORY: int = 20
    AI_MAX_RETRIES: int = 3
    AI_RETRY_BASE_DELAY: float = 1.5
    AI_MAX_OUTPUT_TOKENS: int = 1024
    DISCORD_MSG_LIMIT: int = 2000

    SPAM_MESSAGE_WINDOW_SECONDS: int = 8
    SPAM_MESSAGE_THRESHOLD: int = 6
    SPAM_DUPLICATE_THRESHOLD: int = 3
    SPAM_MENTION_THRESHOLD: int = 5
    SPAM_CAPS_RATIO: float = 0.7
    SPAM_CAPS_MIN_LEN: int = 10
    SPAM_EMOJI_THRESHOLD: int = 8
    SPAM_ATTACHMENT_DUPLICATE_THRESHOLD: int = 3

    WARNING_TIMEOUT_THRESHOLD: int = 3
    WARNING_TIMEOUT_DURATION_MINUTES: int = 10

    INVITE_REGEX: re.Pattern = re.compile(
        r"(discord\.gg|discordapp\.com/invite|discord\.com/invite)/\S+", re.IGNORECASE
    )

    DEFAULT_PROFANITY_LIST: Tuple[str, ...] = (
        "badword1",
        "badword2",
        "badword3",
    )

    STATUS_ROTATION_SECONDS: int = 30
    CLEANUP_INTERVAL_MINUTES: int = 60
    BACKUP_INTERVAL_HOURS: int = 6

    LOG_CHANNEL_SETTING_KEY: str = "log_channel_id"
    AI_CHANNEL_SETTING_KEY: str = "ai_channel_id"


def validate_config() -> None:
    """Validate required configuration is present, exit if not."""
    missing = []
    if not Config.DISCORD_TOKEN:
        missing.append("DISCORD_TOKEN")
    if not Config.GEMINI_API_KEY:
        missing.append("GEMINI_API_KEY")
    if not Config.OWNER_ID:
        missing.append("OWNER_ID")
    if missing:
        _colored_log(f"Missing required environment variables: {', '.join(missing)}", Colors.RED)
        sys.exit(1)


# --------------------------------------------------------------------------- #
#                                   LOGGING                                   #
# --------------------------------------------------------------------------- #

logger = logging.getLogger("bot")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"))
logger.addHandler(_handler)


# --------------------------------------------------------------------------- #
#                               DATABASE MANAGER                              #
# --------------------------------------------------------------------------- #

class DatabaseManager:
    """Handles all SQLite persistence via aiosqlite."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._conn: Optional[aiosqlite.Connection] = None
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        """Open the database connection and create tables if needed."""
        self._conn = await aiosqlite.connect(self.path)
        self._conn.row_factory = aiosqlite.Row
        await self._create_tables()
        logger.info("Database connected at %s", self.path)

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()

    async def _create_tables(self) -> None:
        assert self._conn is not None
        await self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS warnings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                guild_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                moderator_id INTEGER NOT NULL,
                reason TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS conversation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS guild_settings (
                guild_id INTEGER NOT NULL,
                key TEXT NOT NULL,
                value TEXT,
                PRIMARY KEY (guild_id, key)
            );

            CREATE TABLE IF NOT EXISTS user_settings (
                user_id INTEGER NOT NULL,
                key TEXT NOT NULL,
                value TEXT,
                PRIMARY KEY (user_id, key)
            );

            CREATE TABLE IF NOT EXISTS ai_memory (
                user_id INTEGER PRIMARY KEY,
                summary TEXT,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS statistics (
                guild_id INTEGER NOT NULL,
                key TEXT NOT NULL,
                value INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (guild_id, key)
            );

            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                guild_id INTEGER,
                event_type TEXT NOT NULL,
                details TEXT,
                created_at TEXT NOT NULL
            );
            """
        )
        await self._conn.commit()

    # ---------------------------- warnings ---------------------------- #

    async def add_warning(self, guild_id: int, user_id: int, moderator_id: int, reason: str) -> int:
        assert self._conn is not None
        async with self._lock:
            cursor = await self._conn.execute(
                "INSERT INTO warnings (guild_id, user_id, moderator_id, reason, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (guild_id, user_id, moderator_id, reason, datetime.now(timezone.utc).isoformat()),
            )
            await self._conn.commit()
            return cursor.lastrowid

    async def get_warnings(self, guild_id: int, user_id: int) -> List[aiosqlite.Row]:
        assert self._conn is not None
        cursor = await self._conn.execute(
            "SELECT * FROM warnings WHERE guild_id = ? AND user_id = ? ORDER BY created_at DESC",
            (guild_id, user_id),
        )
        return await cursor.fetchall()

    async def clear_warnings(self, guild_id: int, user_id: int) -> None:
        assert self._conn is not None
        async with self._lock:
            await self._conn.execute(
                "DELETE FROM warnings WHERE guild_id = ? AND user_id = ?", (guild_id, user_id)
            )
            await self._conn.commit()

    # ------------------------ conversation history ---------------------- #

    async def add_message(self, user_id: int, role: str, content: str) -> None:
        assert self._conn is not None
        async with self._lock:
            await self._conn.execute(
                "INSERT INTO conversation_history (user_id, role, content, created_at) "
                "VALUES (?, ?, ?, ?)",
                (user_id, role, content, datetime.now(timezone.utc).isoformat()),
            )
            await self._conn.commit()

    async def get_history(self, user_id: int, limit: int) -> List[aiosqlite.Row]:
        assert self._conn is not None
        cursor = await self._conn.execute(
            "SELECT * FROM conversation_history WHERE user_id = ? ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        )
        rows = await cursor.fetchall()
        return list(reversed(rows))

    async def clear_history(self, user_id: int) -> None:
        assert self._conn is not None
        async with self._lock:
            await self._conn.execute("DELETE FROM conversation_history WHERE user_id = ?", (user_id,))
            await self._conn.commit()

    async def trim_history(self, user_id: int, keep: int) -> None:
        assert self._conn is not None
        async with self._lock:
            await self._conn.execute(
                "DELETE FROM conversation_history WHERE user_id = ? AND id NOT IN ("
                "SELECT id FROM conversation_history WHERE user_id = ? "
                "ORDER BY id DESC LIMIT ?)",
                (user_id, user_id, keep),
            )
            await self._conn.commit()

    # --------------------------- guild settings -------------------------- #

    async def set_guild_setting(self, guild_id: int, key: str, value: str) -> None:
        assert self._conn is not None
        async with self._lock:
            await self._conn.execute(
                "INSERT INTO guild_settings (guild_id, key, value) VALUES (?, ?, ?) "
                "ON CONFLICT(guild_id, key) DO UPDATE SET value = excluded.value",
                (guild_id, key, value),
            )
            await self._conn.commit()

    async def get_guild_setting(self, guild_id: int, key: str) -> Optional[str]:
        assert self._conn is not None
        cursor = await self._conn.execute(
            "SELECT value FROM guild_settings WHERE guild_id = ? AND key = ?", (guild_id, key)
        )
        row = await cursor.fetchone()
        return row["value"] if row else None

    # ---------------------------- user settings --------------------------- #

    async def set_user_setting(self, user_id: int, key: str, value: str) -> None:
        assert self._conn is not None
        async with self._lock:
            await self._conn.execute(
                "INSERT INTO user_settings (user_id, key, value) VALUES (?, ?, ?) "
                "ON CONFLICT(user_id, key) DO UPDATE SET value = excluded.value",
                (user_id, key, value),
            )
            await self._conn.commit()

    async def get_user_setting(self, user_id: int, key: str) -> Optional[str]:
        assert self._conn is not None
        cursor = await self._conn.execute(
            "SELECT value FROM user_settings WHERE user_id = ? AND key = ?", (user_id, key)
        )
        row = await cursor.fetchone()
        return row["value"] if row else None

    # ------------------------------ ai memory ------------------------------ #

    async def set_ai_memory(self, user_id: int, summary: str) -> None:
        assert self._conn is not None
        async with self._lock:
            await self._conn.execute(
                "INSERT INTO ai_memory (user_id, summary, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(user_id) DO UPDATE SET summary = excluded.summary, "
                "updated_at = excluded.updated_at",
                (user_id, summary, datetime.now(timezone.utc).isoformat()),
            )
            await self._conn.commit()

    async def get_ai_memory(self, user_id: int) -> Optional[str]:
        assert self._conn is not None
        cursor = await self._conn.execute(
            "SELECT summary FROM ai_memory WHERE user_id = ?", (user_id,)
        )
        row = await cursor.fetchone()
        return row["summary"] if row else None

    # ------------------------------ statistics ------------------------------ #

    async def increment_stat(self, guild_id: int, key: str, amount: int = 1) -> None:
        assert self._conn is not None
        async with self._lock:
            await self._conn.execute(
                "INSERT INTO statistics (guild_id, key, value) VALUES (?, ?, ?) "
                "ON CONFLICT(guild_id, key) DO UPDATE SET value = value + excluded.value",
                (guild_id, key, amount),
            )
            await self._conn.commit()

    async def get_stat(self, guild_id: int, key: str) -> int:
        assert self._conn is not None
        cursor = await self._conn.execute(
            "SELECT value FROM statistics WHERE guild_id = ? AND key = ?", (guild_id, key)
        )
        row = await cursor.fetchone()
        return row["value"] if row else 0

    # -------------------------------- logs -------------------------------- #

    async def add_log(self, guild_id: Optional[int], event_type: str, details: str) -> None:
        assert self._conn is not None
        async with self._lock:
            await self._conn.execute(
                "INSERT INTO logs (guild_id, event_type, details, created_at) VALUES (?, ?, ?, ?)",
                (guild_id, event_type, details, datetime.now(timezone.utc).isoformat()),
            )
            await self._conn.commit()

    async def backup(self, backup_dir: str) -> str:
        """Create a timestamped backup copy of the database."""
        assert self._conn is not None
        os.makedirs(backup_dir, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(backup_dir, f"backup_{timestamp}.db")
        backup_conn = await aiosqlite.connect(backup_path)
        await self._conn.backup(backup_conn)
        await backup_conn.close()
        return backup_path


# --------------------------------------------------------------------------- #
#                                  AI MANAGER                                  #
# --------------------------------------------------------------------------- #

class AIManager:
    """Manages Gemini AI interactions and per-user conversational memory."""

    SYSTEM_INSTRUCTION = (
        "You are a helpful, friendly Discord bot assistant. Keep responses natural, "
        "conversational, and concise unless the user asks for detail. You have memory "
        "of the recent conversation with this user."
    )

    def __init__(self, db: DatabaseManager, api_key: str, model: str) -> None:
        self.db = db
        self.model = model
        self._client = None
        if genai is not None and api_key:
            self._client = genai.Client(api_key=api_key)
        self._locks: Dict[int, asyncio.Lock] = defaultdict(asyncio.Lock)

    def is_ready(self) -> bool:
        return self._client is not None

    async def generate_response(self, user_id: int, prompt: str) -> str:
        """Generate an AI response for a user, maintaining conversation history."""
        if not self.is_ready():
            return "The AI system is not configured correctly. Please contact the bot owner."

        async with self._locks[user_id]:
            await self.db.add_message(user_id, "user", prompt)
            history_rows = await self.db.get_history(user_id, Config.AI_MAX_HISTORY)

            contents: List[Any] = []
            for row in history_rows:
                role = "user" if row["role"] == "user" else "model"
                contents.append(
                    genai_types.Content(
                        role=role, parts=[genai_types.Part.from_text(text=row["content"])]
                    )
                )

            response_text = await self._call_with_retry(contents)
            await self.db.add_message(user_id, "assistant", response_text)
            await self.db.trim_history(user_id, Config.AI_MAX_HISTORY * 2)
            return response_text

    async def _call_with_retry(self, contents: List[Any]) -> str:
        last_error: Optional[Exception] = None
        for attempt in range(Config.AI_MAX_RETRIES):
            try:
                return await asyncio.wait_for(
                    asyncio.to_thread(self._generate_sync, contents), timeout=45
                )
            except asyncio.TimeoutError as exc:
                last_error = exc
                logger.warning("Gemini call timed out (attempt %d)", attempt + 1)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning("Gemini call failed (attempt %d): %s", attempt + 1, exc)
            await asyncio.sleep(Config.AI_RETRY_BASE_DELAY * (attempt + 1))
        logger.error("Gemini call failed after retries: %s", last_error)
        return "Sorry, I'm having trouble reaching the AI service right now. Please try again shortly."

    def _generate_sync(self, contents: List[Any]) -> str:
        assert self._client is not None
        response = self._client.models.generate_content(
            model=self.model,
            contents=contents,
            config=genai_types.GenerateContentConfig(
                system_instruction=self.SYSTEM_INSTRUCTION,
                max_output_tokens=Config.AI_MAX_OUTPUT_TOKENS,
                temperature=0.9,
            ),
        )
        text = getattr(response, "text", None)
        if not text:
            return "I couldn't generate a response for that. Could you rephrase?"
        return text.strip()

    @staticmethod
    def split_message(text: str, limit: int = Config.DISCORD_MSG_LIMIT) -> List[str]:
        """Split a long message into Discord-safe chunks, preferring line breaks."""
        if len(text) <= limit:
            return [text]

        chunks: List[str] = []
        remaining = text
        while len(remaining) > limit:
            split_at = remaining.rfind("\n", 0, limit)
            if split_at == -1:
                split_at = remaining.rfind(" ", 0, limit)
            if split_at == -1:
                split_at = limit
            chunks.append(remaining[:split_at])
            remaining = remaining[split_at:].lstrip()
        if remaining:
            chunks.append(remaining)
        return chunks


# --------------------------------------------------------------------------- #
#                                LOGGING MANAGER                              #
# --------------------------------------------------------------------------- #

class LoggingManager:
    """Builds and dispatches embed-based logs to a guild's configured log channel."""

    def __init__(self, bot: "ModBot", db: DatabaseManager) -> None:
        self.bot = bot
        self.db = db

    async def _get_log_channel(self, guild: discord.Guild) -> Optional[discord.TextChannel]:
        channel_id = await self.db.get_guild_setting(guild.id, Config.LOG_CHANNEL_SETTING_KEY)
        if not channel_id:
            return None
        channel = guild.get_channel(int(channel_id))
        return channel if isinstance(channel, discord.TextChannel) else None

    async def log(
        self,
        guild: Optional[discord.Guild],
        title: str,
        description: str,
        color: discord.Color = discord.Color.blurple(),
        fields: Optional[List[Tuple[str, str, bool]]] = None,
    ) -> None:
        """Send a log embed and persist it to the database."""
        if guild is not None:
            await self.db.add_log(guild.id, title, description)
            channel = await self._get_log_channel(guild)
            if channel is not None:
                embed = discord.Embed(
                    title=title,
                    description=description,
                    color=color,
                    timestamp=datetime.now(timezone.utc),
                )
                if fields:
                    for name, value, inline in fields:
                        embed.add_field(name=name, value=value, inline=inline)
                try:
                    await channel.send(embed=embed)
                except discord.Forbidden:
                    logger.warning("Missing permission to send logs in guild %s", guild.id)
                except discord.HTTPException as exc:
                    logger.warning("Failed to send log embed: %s", exc)
        else:
            await self.db.add_log(None, title, description)
        logger.info("[%s] %s", title, description)


# --------------------------------------------------------------------------- #
#                               WARNING MANAGER                               #
# --------------------------------------------------------------------------- #

class WarningManager:
    """Handles warning issuance, retrieval, and escalation logic."""

    def __init__(self, db: DatabaseManager, logging_manager: LoggingManager) -> None:
        self.db = db
        self.logging_manager = logging_manager

    async def warn_user(
        self,
        guild: discord.Guild,
        member: discord.Member,
        moderator: discord.abc.User,
        reason: str,
    ) -> int:
        """Issue a warning, returns the total warning count for that user."""
        await self.db.add_warning(guild.id, member.id, moderator.id, reason)
        warnings = await self.db.get_warnings(guild.id, member.id)
        count = len(warnings)

        await self.logging_manager.log(
            guild,
            "Member Warned",
            f"{member.mention} was warned by {moderator.mention}",
            discord.Color.yellow(),
            fields=[("Reason", reason, False), ("Total Warnings", str(count), True)],
        )

        if count >= Config.WARNING_TIMEOUT_THRESHOLD:
            try:
                duration = timedelta(minutes=Config.WARNING_TIMEOUT_DURATION_MINUTES)
                await member.timeout(duration, reason=f"Reached {count} warnings")
                await self.logging_manager.log(
                    guild,
                    "Auto-Timeout",
                    f"{member.mention} was automatically timed out after reaching "
                    f"{count} warnings.",
                    discord.Color.orange(),
                )
            except (discord.Forbidden, discord.HTTPException) as exc:
                logger.warning("Failed to auto-timeout member %s: %s", member.id, exc)
        return count


# --------------------------------------------------------------------------- #
#                                SPAM DETECTOR                                #
# --------------------------------------------------------------------------- #

@dataclass
class UserActivity:
    """Tracks recent message activity for a single user."""

    timestamps: Deque[float] = field(default_factory=deque)
    last_contents: Deque[str] = field(default_factory=lambda: deque(maxlen=5))
    last_attachment_hashes: Deque[str] = field(default_factory=lambda: deque(maxlen=5))
    warn_count_session: int = 0


class SpamDetector:
    """Detects and handles various forms of spam and abuse in messages."""

    def __init__(
        self,
        db: DatabaseManager,
        logging_manager: LoggingManager,
        warning_manager: WarningManager,
    ) -> None:
        self.db = db
        self.logging_manager = logging_manager
        self.warning_manager = warning_manager
        self._activity: Dict[int, UserActivity] = defaultdict(UserActivity)

    @staticmethod
    def _caps_ratio(text: str) -> float:
        letters = [c for c in text if c.isalpha()]
        if not letters:
            return 0.0
        caps = sum(1 for c in letters if c.isupper())
        return caps / len(letters)

    @staticmethod
    def _emoji_count(text: str) -> int:
        custom = len(re.findall(r"<a?:\w+:\d+>", text))
        unicode_emoji = len(re.findall(
            r"[\U0001F300-\U0001FAFF\U00002600-\U000027BF]", text
        ))
        return custom + unicode_emoji

    async def evaluate(self, message: discord.Message) -> Optional[str]:
        """
        Evaluate a message for spam. Returns a violation reason string if
        the message should be treated as spam, otherwise None.
        """
        if message.guild is None or message.author.bot:
            return None

        activity = self._activity[message.author.id]
        now = time.monotonic()
        activity.timestamps.append(now)
        while activity.timestamps and now - activity.timestamps[0] > Config.SPAM_MESSAGE_WINDOW_SECONDS:
            activity.timestamps.popleft()

        if len(activity.timestamps) >= Config.SPAM_MESSAGE_THRESHOLD:
            return "Sending messages too quickly (flood)"

        content = message.content.strip()
        if content:
            duplicate_count = sum(1 for c in activity.last_contents if c == content)
            activity.last_contents.append(content)
            if duplicate_count + 1 >= Config.SPAM_DUPLICATE_THRESHOLD:
                return "Repeated identical messages"

            if len(message.mentions) >= Config.SPAM_MENTION_THRESHOLD:
                return "Mass mentions"

            if Config.INVITE_REGEX.search(content):
                return "Posted an unauthorized invite link"

            if len(content) >= Config.SPAM_CAPS_MIN_LEN and self._caps_ratio(content) >= Config.SPAM_CAPS_RATIO:
                return "Excessive capital letters"

            if self._emoji_count(content) >= Config.SPAM_EMOJI_THRESHOLD:
                return "Excessive repeated emojis"

        if message.attachments:
            for attachment in message.attachments:
                key = f"{attachment.filename}:{attachment.size}"
                dup = sum(1 for h in activity.last_attachment_hashes if h == key)
                activity.last_attachment_hashes.append(key)
                if dup + 1 >= Config.SPAM_ATTACHMENT_DUPLICATE_THRESHOLD:
                    return "Repeated duplicate attachments"

        return None

    async def handle_violation(self, message: discord.Message, reason: str) -> None:
        """Delete the offending message and apply escalating anti-spam actions."""
        guild = message.guild
        assert guild is not None
        member = message.author
        assert isinstance(member, discord.Member)

        try:
            await message.delete()
        except (discord.Forbidden, discord.NotFound, discord.HTTPException) as exc:
            logger.warning("Could not delete spam message: %s", exc)

        activity = self._activity[member.id]
        activity.warn_count_session += 1

        await self.logging_manager.log(
            guild,
            "Spam Detected",
            f"Message from {member.mention} removed.",
            discord.Color.red(),
            fields=[("Reason", reason, False), ("Session Violations", str(activity.warn_count_session), True)],
        )
        await self.db.increment_stat(guild.id, "spam_deleted")

        try:
            await member.send(
                f"Your message in **{guild.name}** was removed for: {reason}. "
                "Repeated violations may result in a timeout."
            )
        except (discord.Forbidden, discord.HTTPException):
            pass

        if activity.warn_count_session >= 3:
            try:
                await member.timeout(
                    timedelta(minutes=Config.WARNING_TIMEOUT_DURATION_MINUTES),
                    reason="Repeated spam violations",
                )
                await self.logging_manager.log(
                    guild,
                    "Anti-Spam Timeout",
                    f"{member.mention} was timed out for repeated spam violations.",
                    discord.Color.orange(),
                )
                activity.warn_count_session = 0
            except (discord.Forbidden, discord.HTTPException) as exc:
                logger.warning("Failed to timeout spammer %s: %s", member.id, exc)
        else:
            await self.warning_manager.warn_user(guild, member, guild.me, f"Anti-spam: {reason}")


# --------------------------------------------------------------------------- #
#                             PROFANITY FILTER                                #
# --------------------------------------------------------------------------- #

class ProfanityFilter:
    """Simple case-insensitive regex-based profanity blacklist filter."""

    def __init__(self, blacklist: Tuple[str, ...]) -> None:
        self._pattern = self._compile(blacklist)

    @staticmethod
    def _compile(words: Tuple[str, ...]) -> re.Pattern:
        escaped = [rf"\b{re.escape(w)}\b" for w in words if w]
        if not escaped:
            return re.compile(r"(?!x)x")  # never matches
        return re.compile("|".join(escaped), re.IGNORECASE)

    def contains_profanity(self, text: str) -> bool:
        return bool(self._pattern.search(text))


# --------------------------------------------------------------------------- #
#                             MODERATION MANAGER                              #
# --------------------------------------------------------------------------- #

class ModerationManager:
    """Encapsulates moderation actions with permission checks and logging."""

    def __init__(self, logging_manager: LoggingManager, db: DatabaseManager) -> None:
        self.logging_manager = logging_manager
        self.db = db

    @staticmethod
    def check_hierarchy(moderator: discord.Member, target: discord.Member) -> bool:
        """Return True if moderator's top role outranks target's top role."""
        if moderator.guild.owner_id == moderator.id:
            return True
        return moderator.top_role > target.top_role

    async def kick(self, interaction: discord.Interaction, member: discord.Member, reason: str) -> str:
        moderator = interaction.user
        assert isinstance(moderator, discord.Member)
        if not self.check_hierarchy(moderator, member):
            return "You cannot kick a member with an equal or higher role than you."
        try:
            await member.kick(reason=f"{reason} (by {moderator})")
        except discord.Forbidden:
            return "I don't have permission to kick that member."
        except discord.HTTPException as exc:
            return f"Failed to kick member: {exc}"

        await self.db.increment_stat(interaction.guild_id, "kicks")
        await self.logging_manager.log(
            interaction.guild,
            "Member Kicked",
            f"{member.mention} was kicked by {moderator.mention}",
            discord.Color.orange(),
            fields=[("Reason", reason, False)],
        )
        return f"{member.mention} has been kicked. Reason: {reason}"

    async def ban(
        self, interaction: discord.Interaction, member: discord.abc.User, reason: str, delete_days: int = 0
    ) -> str:
        moderator = interaction.user
        assert isinstance(moderator, discord.Member)
        if isinstance(member, discord.Member) and not self.check_hierarchy(moderator, member):
            return "You cannot ban a member with an equal or higher role than you."
        try:
            await interaction.guild.ban(
                member, reason=f"{reason} (by {moderator})", delete_message_seconds=delete_days * 86400
            )
        except discord.Forbidden:
            return "I don't have permission to ban that member."
        except discord.HTTPException as exc:
            return f"Failed to ban member: {exc}"

        await self.db.increment_stat(interaction.guild_id, "bans")
        await self.logging_manager.log(
            interaction.guild,
            "Member Banned",
            f"{member.mention} was banned by {moderator.mention}",
            discord.Color.red(),
            fields=[("Reason", reason, False)],
        )
        return f"{member.mention} has been banned. Reason: {reason}"

    async def timeout(
        self, interaction: discord.Interaction, member: discord.Member, minutes: int, reason: str
    ) -> str:
        moderator = interaction.user
        assert isinstance(moderator, discord.Member)
        if not self.check_hierarchy(moderator, member):
            return "You cannot timeout a member with an equal or higher role than you."
        try:
            await member.timeout(timedelta(minutes=minutes), reason=f"{reason} (by {moderator})")
        except discord.Forbidden:
            return "I don't have permission to timeout that member."
        except discord.HTTPException as exc:
            return f"Failed to timeout member: {exc}"

        await self.db.increment_stat(interaction.guild_id, "timeouts")
        await self.logging_manager.log(
            interaction.guild,
            "Member Timed Out",
            f"{member.mention} was timed out for {minutes} minute(s) by {moderator.mention}",
            discord.Color.orange(),
            fields=[("Reason", reason, False)],
        )
        return f"{member.mention} has been timed out for {minutes} minute(s). Reason: {reason}"


# --------------------------------------------------------------------------- #
#                                    BOT CLASS                                #
# --------------------------------------------------------------------------- #

class ModBot(commands.Bot):
    """Main bot class tying together all managers and event handlers."""

    STATUS_MESSAGES: List[Tuple[discord.ActivityType, str]] = [
        (discord.ActivityType.watching, "over the server"),
        (discord.ActivityType.listening, "/help for commands"),
        (discord.ActivityType.playing, "with Gemini AI"),
    ]

    def __init__(self) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        intents.guilds = True

        super().__init__(command_prefix="!", intents=intents, help_command=None)

        self.start_time = datetime.now(timezone.utc)
        self.db = DatabaseManager(Config.DB_PATH)
        self.ai_manager = AIManager(self.db, Config.GEMINI_API_KEY, Config.GEMINI_MODEL)
        self.logging_manager = LoggingManager(self, self.db)
        self.warning_manager = WarningManager(self.db, self.logging_manager)
        self.spam_detector = SpamDetector(self.db, self.logging_manager, self.warning_manager)
        self.moderation_manager = ModerationManager(self.logging_manager, self.db)
        self.profanity_filter = ProfanityFilter(Config.DEFAULT_PROFANITY_LIST)
        self._status_index = 0
        self._recent_deleted_cache: Deque[Tuple[int, str]] = deque(maxlen=200)

    async def setup_hook(self) -> None:
        """Called once before the bot connects; initializes DB and syncs commands."""
        await self.db.connect()
        register_commands(self)
        try:
            synced = await self.tree.sync()
            _colored_log(f"Synced {len(synced)} slash command(s).", Colors.GREEN)
        except discord.HTTPException as exc:
            logger.error("Failed to sync commands: %s", exc)

        self.status_rotation_task.start()
        self.cleanup_task.start()
        self.backup_task.start()

    async def close(self) -> None:
        await self.db.close()
        await super().close()

    async def get_ai_channel_id(self, guild_id: int) -> Optional[int]:
        value = await self.db.get_guild_setting(guild_id, Config.AI_CHANNEL_SETTING_KEY)
        return int(value) if value else None

    # ------------------------------ background tasks ------------------------------ #

    @tasks.loop(seconds=Config.STATUS_ROTATION_SECONDS)
    async def status_rotation_task(self) -> None:
        activity_type, text = self.STATUS_MESSAGES[self._status_index % len(self.STATUS_MESSAGES)]
        self._status_index += 1
        await self.change_presence(activity=discord.Activity(type=activity_type, name=text))

    @status_rotation_task.before_loop
    async def before_status_rotation(self) -> None:
        await self.wait_until_ready()

    @tasks.loop(minutes=Config.CLEANUP_INTERVAL_MINUTES)
    async def cleanup_task(self) -> None:
        logger.info("Running periodic cleanup task.")
        self._recent_deleted_cache.clear()

    @cleanup_task.before_loop
    async def before_cleanup(self) -> None:
        await self.wait_until_ready()

    @tasks.loop(hours=Config.BACKUP_INTERVAL_HOURS)
    async def backup_task(self) -> None:
        try:
            path = await self.db.backup(Config.DB_BACKUP_DIR)
            logger.info("Database backed up to %s", path)
        except Exception as exc:  # noqa: BLE001
            logger.error("Database backup failed: %s", exc)

    @backup_task.before_loop
    async def before_backup(self) -> None:
        await self.wait_until_ready()

    # ---------------------------------- events ---------------------------------- #

    async def on_ready(self) -> None:
        _colored_log(f"Logged in as {self.user} (ID: {self.user.id})", Colors.GREEN)
        _colored_log(f"Connected to {len(self.guilds)} guild(s).", Colors.GREEN)
        _colored_log("Bot is fully operational.", Colors.MAGENTA)
        await self.logging_manager.log(None, "Bot Startup", f"{self.user} has started successfully.")

    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot:
            return

        if message.guild is not None:
            violation = await self.spam_detector.evaluate(message)
            if violation:
                await self.spam_detector.handle_violation(message, violation)
                return

            if self.profanity_filter.contains_profanity(message.content):
                try:
                    await message.delete()
                except (discord.Forbidden, discord.NotFound, discord.HTTPException):
                    pass
                await self.warning_manager.warn_user(
                    message.guild, message.author, self.user, "Used prohibited language"
                )
                await self.logging_manager.log(
                    message.guild,
                    "Profanity Filtered",
                    f"Message from {message.author.mention} removed for prohibited language.",
                    discord.Color.dark_red(),
                )
                return

        await self._maybe_handle_ai(message)
        await self.process_commands(message)

    async def _maybe_handle_ai(self, message: discord.Message) -> None:
        """Route a message to the AI manager if it qualifies (mention or AI channel)."""
        is_mentioned = self.user in message.mentions if self.user else False
        is_ai_channel = False
        if message.guild is not None:
            ai_channel_id = await self.get_ai_channel_id(message.guild.id)
            is_ai_channel = ai_channel_id is not None and message.channel.id == ai_channel_id

        if not (is_mentioned or is_ai_channel):
            return

        prompt = message.content
        if self.user is not None:
            prompt = re.sub(rf"<@!?{self.user.id}>", "", prompt).strip()
        if not prompt:
            return

        async with message.channel.typing():
            try:
                response = await self.ai_manager.generate_response(message.author.id, prompt)
            except Exception as exc:  # noqa: BLE001
                logger.error("AI generation failed: %s", exc)
                response = "Something went wrong while generating a response."

        for chunk in AIManager.split_message(response):
            await message.reply(chunk, mention_author=False)

    async def on_member_join(self, member: discord.Member) -> None:
        await self.db.increment_stat(member.guild.id, "member_joins")
        await self.logging_manager.log(
            member.guild,
            "Member Joined",
            f"{member.mention} joined the server.",
            discord.Color.green(),
        )

    async def on_member_remove(self, member: discord.Member) -> None:
        await self.db.increment_stat(member.guild.id, "member_leaves")
        await self.logging_manager.log(
            member.guild,
            "Member Left",
            f"{member.mention} left the server.",
            discord.Color.dark_grey(),
        )

    async def on_message_delete(self, message: discord.Message) -> None:
        if message.author.bot or message.guild is None:
            return
        self._recent_deleted_cache.append((message.author.id, message.content))
        content = message.content or "*[no text content]*"
        if len(content) > 512:
            content = content[:512] + "..."
        await self.logging_manager.log(
            message.guild,
            "Message Deleted",
            f"A message by {message.author.mention} in {message.channel.mention} was deleted.",
            discord.Color.dark_orange(),
            fields=[("Content", content, False)],
        )

    async def on_message_edit(self, before: discord.Message, after: discord.Message) -> None:
        if before.author.bot or before.guild is None or before.content == after.content:
            return
        before_content = (before.content or "*[empty]*")[:512]
        after_content = (after.content or "*[empty]*")[:512]
        await self.logging_manager.log(
            before.guild,
            "Message Edited",
            f"A message by {before.author.mention} in {before.channel.mention} was edited.",
            discord.Color.blue(),
            fields=[("Before", before_content, False), ("After", after_content, False)],
        )

    async def on_guild_join(self, guild: discord.Guild) -> None:
        await self.logging_manager.log(guild, "Guild Joined", f"Bot added to guild: {guild.name} ({guild.id})")

    async def on_guild_remove(self, guild: discord.Guild) -> None:
        logger.info("Removed from guild: %s (%s)", guild.name, guild.id)

    async def on_command_error(self, ctx: commands.Context, error: commands.CommandError) -> None:
        if isinstance(error, commands.CommandNotFound):
            return
        logger.error("Command error in %s: %s", ctx.command, error)
        await self.logging_manager.log(
            ctx.guild, "Command Error", f"Error in command `{ctx.command}`: {error}", discord.Color.red()
        )
        try:
            await ctx.send(f"An error occurred: {error}")
        except discord.HTTPException:
            pass

    async def on_app_command_error(
        self, interaction: discord.Interaction, error: app_commands.AppCommandError
    ) -> None:
        message = "An unexpected error occurred while running this command."

        if isinstance(error, app_commands.MissingPermissions):
            message = "You don't have permission to use this command."
        elif isinstance(error, app_commands.BotMissingPermissions):
            message = "I don't have the required permissions to do that."
        elif isinstance(error, app_commands.CommandOnCooldown):
            message = f"This command is on cooldown. Try again in {error.retry_after:.1f}s."
        elif isinstance(error, discord.Forbidden):
            message = "I lack the necessary permissions to complete that action."
        elif isinstance(error, discord.NotFound):
            message = "The target of this command could not be found."
        elif isinstance(error, discord.HTTPException):
            message = "A Discord API error occurred. Please try again."
        else:
            logger.error("Unhandled app command error: %s\n%s", error, traceback.format_exc())

        await self.logging_manager.log(
            interaction.guild,
            "App Command Error",
            f"Command `{interaction.command.name if interaction.command else 'unknown'}` "
            f"failed: {error}",
            discord.Color.red(),
        )

        try:
            if interaction.response.is_done():
                await interaction.followup.send(message, ephemeral=True)
            else:
                await interaction.response.send_message(message, ephemeral=True)
        except discord.HTTPException:
            pass


# --------------------------------------------------------------------------- #
#                              PERMISSION HELPERS                             #
# --------------------------------------------------------------------------- #

def is_owner_or_permission(permission_name: str):
    """Check decorator allowing either the bot owner or a required permission."""

    async def predicate(interaction: discord.Interaction) -> bool:
        if interaction.user.id == Config.OWNER_ID:
            return True
        if isinstance(interaction.user, discord.Member):
            return getattr(interaction.user.guild_permissions, permission_name, False)
        return False

    return app_commands.check(predicate)


# --------------------------------------------------------------------------- #
#                              SLASH COMMAND SETUP                            #
# --------------------------------------------------------------------------- #

def register_commands(bot: ModBot) -> None:
    """Registers all application (slash) commands onto the bot's command tree."""

    tree = bot.tree

    # ------------------------------- AI ------------------------------- #

    @tree.command(name="ask", description="Ask the AI assistant a question.")
    @app_commands.describe(prompt="What would you like to ask?")
    async def ask(interaction: discord.Interaction, prompt: str) -> None:
        await interaction.response.defer(thinking=True)
        try:
            response = await bot.ai_manager.generate_response(interaction.user.id, prompt)
        except Exception as exc:  # noqa: BLE001
            logger.error("AI /ask failed: %s", exc)
            response = "Something went wrong while generating a response."
        chunks = AIManager.split_message(response)
        await interaction.followup.send(chunks[0])
        for chunk in chunks[1:]:
            await interaction.followup.send(chunk)

    @tree.command(name="setaichannel", description="Set the channel where the AI responds automatically.")
    @app_commands.describe(channel="The channel to designate as the AI channel")
    @is_owner_or_permission("manage_guild")
    async def set_ai_channel(interaction: discord.Interaction, channel: discord.TextChannel) -> None:
        await bot.db.set_guild_setting(interaction.guild_id, Config.AI_CHANNEL_SETTING_KEY, str(channel.id))
        await interaction.response.send_message(f"AI channel set to {channel.mention}.", ephemeral=True)

    @tree.command(name="setlogchannel", description="Set the channel where moderation logs are sent.")
    @app_commands.describe(channel="The channel to designate for logs")
    @is_owner_or_permission("manage_guild")
    async def set_log_channel(interaction: discord.Interaction, channel: discord.TextChannel) -> None:
        await bot.db.set_guild_setting(interaction.guild_id, Config.LOG_CHANNEL_SETTING_KEY, str(channel.id))
        await interaction.response.send_message(f"Log channel set to {channel.mention}.", ephemeral=True)

    @tree.command(name="clearmemory", description="Clear your AI conversation memory.")
    async def clear_memory(interaction: discord.Interaction) -> None:
        await bot.db.clear_history(interaction.user.id)
        await interaction.response.send_message("Your AI conversation history has been cleared.", ephemeral=True)

    # --------------------------- moderation --------------------------- #

    @tree.command(name="kick", description="Kick a member from the server.")
    @app_commands.describe(member="The member to kick", reason="Reason for the kick")
    @app_commands.checks.has_permissions(kick_members=True)
    @app_commands.checks.bot_has_permissions(kick_members=True)
    async def kick_cmd(interaction: discord.Interaction, member: discord.Member, reason: str = "No reason provided") -> None:
        await interaction.response.defer(ephemeral=True)
        result = await bot.moderation_manager.kick(interaction, member, reason)
        await interaction.followup.send(result)

    @tree.command(name="ban", description="Ban a member from the server.")
    @app_commands.describe(member="The member to ban", reason="Reason for the ban", delete_days="Days of messages to delete (0-7)")
    @app_commands.checks.has_permissions(ban_members=True)
    @app_commands.checks.bot_has_permissions(ban_members=True)
    async def ban_cmd(
        interaction: discord.Interaction,
        member: discord.User,
        reason: str = "No reason provided",
        delete_days: app_commands.Range[int, 0, 7] = 0,
    ) -> None:
        await interaction.response.defer(ephemeral=True)
        result = await bot.moderation_manager.ban(interaction, member, reason, delete_days)
        await interaction.followup.send(result)

    @tree.command(name="timeout", description="Timeout a member for a number of minutes.")
    @app_commands.describe(member="The member to timeout", minutes="Duration in minutes", reason="Reason")
    @app_commands.checks.has_permissions(moderate_members=True)
    @app_commands.checks.bot_has_permissions(moderate_members=True)
    async def timeout_cmd(
        interaction: discord.Interaction,
        member: discord.Member,
        minutes: app_commands.Range[int, 1, 40320],
        reason: str = "No reason provided",
    ) -> None:
        await interaction.response.defer(ephemeral=True)
        result = await bot.moderation_manager.timeout(interaction, member, minutes, reason)
        await interaction.followup.send(result)

    @tree.command(name="warn", description="Issue a warning to a member.")
    @app_commands.describe(member="The member to warn", reason="Reason for the warning")
    @app_commands.checks.has_permissions(moderate_members=True)
    async def warn_cmd(interaction: discord.Interaction, member: discord.Member, reason: str) -> None:
        await interaction.response.defer(ephemeral=True)
        count = await bot.warning_manager.warn_user(interaction.guild, member, interaction.user, reason)
        await interaction.followup.send(f"{member.mention} has been warned. Total warnings: {count}")

    @tree.command(name="warnings", description="View a member's warnings.")
    @app_commands.describe(member="The member to check")
    @app_commands.checks.has_permissions(moderate_members=True)
    async def warnings_cmd(interaction: discord.Interaction, member: discord.Member) -> None:
        rows = await bot.db.get_warnings(interaction.guild_id, member.id)
        if not rows:
            await interaction.response.send_message(f"{member.mention} has no warnings.", ephemeral=True)
            return
        embed = discord.Embed(
            title=f"Warnings for {member.display_name}",
            color=discord.Color.yellow(),
            timestamp=datetime.now(timezone.utc),
        )
        for row in rows[:25]:
            moderator = interaction.guild.get_member(row["moderator_id"])
            mod_name = moderator.display_name if moderator else str(row["moderator_id"])
            embed.add_field(
                name=f"#{row['id']} - {row['created_at'][:19]}",
                value=f"By {mod_name}: {row['reason']}",
                inline=False,
            )
        await interaction.response.send_message(embed=embed, ephemeral=True)

    @tree.command(name="clear", description="Clear all warnings for a member.")
    @app_commands.describe(member="The member whose warnings should be cleared")
    @app_commands.checks.has_permissions(administrator=True)
    async def clear_cmd(interaction: discord.Interaction, member: discord.Member) -> None:
        await bot.db.clear_warnings(interaction.guild_id, member.id)
        await interaction.response.send_message(f"Cleared all warnings for {member.mention}.", ephemeral=True)

    @tree.command(name="purge", description="Bulk delete messages in this channel.")
    @app_commands.describe(amount="Number of messages to delete (1-100)")
    @app_commands.checks.has_permissions(manage_messages=True)
    @app_commands.checks.bot_has_permissions(manage_messages=True)
    async def purge_cmd(interaction: discord.Interaction, amount: app_commands.Range[int, 1, 100]) -> None:
        await interaction.response.defer(ephemeral=True)
        if not isinstance(interaction.channel, discord.TextChannel):
            await interaction.followup.send("This command can only be used in text channels.")
            return
        deleted = await interaction.channel.purge(limit=amount)
        await bot.db.increment_stat(interaction.guild_id, "messages_purged", len(deleted))
        await bot.logging_manager.log(
            interaction.guild,
            "Messages Purged",
            f"{interaction.user.mention} purged {len(deleted)} message(s) in {interaction.channel.mention}.",
            discord.Color.orange(),
        )
        await interaction.followup.send(f"Deleted {len(deleted)} message(s).")

    @tree.command(name="slowmode", description="Set slowmode delay for this channel.")
    @app_commands.describe(seconds="Delay in seconds (0 to disable)")
    @app_commands.checks.has_permissions(manage_channels=True)
    @app_commands.checks.bot_has_permissions(manage_channels=True)
    async def slowmode_cmd(interaction: discord.Interaction, seconds: app_commands.Range[int, 0, 21600]) -> None:
        if not isinstance(interaction.channel, discord.TextChannel):
            await interaction.response.send_message("This command can only be used in text channels.", ephemeral=True)
            return
        await interaction.channel.edit(slowmode_delay=seconds)
        await bot.logging_manager.log(
            interaction.guild,
            "Slowmode Updated",
            f"{interaction.user.mention} set slowmode to {seconds}s in {interaction.channel.mention}.",
            discord.Color.blue(),
        )
        await interaction.response.send_message(f"Slowmode set to {seconds} second(s).", ephemeral=True)

    @tree.command(name="lock", description="Lock the current channel, preventing @everyone from sending messages.")
    @app_commands.checks.has_permissions(manage_channels=True)
    @app_commands.checks.bot_has_permissions(manage_channels=True)
    async def lock_cmd(interaction: discord.Interaction) -> None:
        if not isinstance(interaction.channel, discord.TextChannel):
            await interaction.response.send_message("This command can only be used in text channels.", ephemeral=True)
            return
        overwrite = interaction.channel.overwrites_for(interaction.guild.default_role)
        overwrite.send_messages = False
        await interaction.channel.set_permissions(interaction.guild.default_role, overwrite=overwrite)
        await bot.logging_manager.log(
            interaction.guild, "Channel Locked", f"{interaction.channel.mention} was locked by {interaction.user.mention}.", discord.Color.red()
        )
        await interaction.response.send_message(f"{interaction.channel.mention} has been locked.")

    @tree.command(name="unlock", description="Unlock the current channel, allowing @everyone to send messages again.")
    @app_commands.checks.has_permissions(manage_channels=True)
    @app_commands.checks.bot_has_permissions(manage_channels=True)
    async def unlock_cmd(interaction: discord.Interaction) -> None:
        if not isinstance(interaction.channel, discord.TextChannel):
            await interaction.response.send_message("This command can only be used in text channels.", ephemeral=True)
            return
        overwrite = interaction.channel.overwrites_for(interaction.guild.default_role)
        overwrite.send_messages = None
        await interaction.channel.set_permissions(interaction.guild.default_role, overwrite=overwrite)
        await bot.logging_manager.log(
            interaction.guild, "Channel Unlocked", f"{interaction.channel.mention} was unlocked by {interaction.user.mention}.", discord.Color.green()
        )
        await interaction.response.send_message(f"{interaction.channel.mention} has been unlocked.")

    # ---------------------------- utility ---------------------------- #

    @tree.command(name="userinfo", description="Show information about a member.")
    @app_commands.describe(member="The member to inspect (defaults to you)")
    async def userinfo_cmd(interaction: discord.Interaction, member: Optional[discord.Member] = None) -> None:
        target = member or interaction.user
        embed = discord.Embed(title=f"User Info: {target.display_name}", color=discord.Color.blurple())
        embed.set_thumbnail(url=target.display_avatar.url)
        embed.add_field(name="Username", value=str(target), inline=True)
        embed.add_field(name="ID", value=str(target.id), inline=True)
        embed.add_field(name="Bot", value=str(target.bot), inline=True)
        if isinstance(target, discord.Member):
            embed.add_field(
                name="Joined Server",
                value=discord.utils.format_dt(target.joined_at) if target.joined_at else "Unknown",
                inline=True,
            )
            embed.add_field(name="Account Created", value=discord.utils.format_dt(target.created_at), inline=True)
            roles = [r.mention for r in target.roles if r.name != "@everyone"]
            embed.add_field(name="Roles", value=", ".join(roles) if roles else "None", inline=False)
        await interaction.response.send_message(embed=embed)

    @tree.command(name="serverinfo", description="Show information about this server.")
    async def serverinfo_cmd(interaction: discord.Interaction) -> None:
        guild = interaction.guild
        if guild is None:
            await interaction.response.send_message("This command must be used in a server.", ephemeral=True)
            return
        embed = discord.Embed(title=f"Server Info: {guild.name}", color=discord.Color.blurple())
        if guild.icon:
            embed.set_thumbnail(url=guild.icon.url)
        embed.add_field(name="Owner", value=str(guild.owner), inline=True)
        embed.add_field(name="Members", value=str(guild.member_count), inline=True)
        embed.add_field(name="Created", value=discord.utils.format_dt(guild.created_at), inline=True)
        embed.add_field(name="Text Channels", value=str(len(guild.text_channels)), inline=True)
        embed.add_field(name="Voice Channels", value=str(len(guild.voice_channels)), inline=True)
        embed.add_field(name="Roles", value=str(len(guild.roles)), inline=True)
        embed.add_field(name="Boost Level", value=str(guild.premium_tier), inline=True)
        await interaction.response.send_message(embed=embed)

    @tree.command(name="ping", description="Check the bot's latency.")
    async def ping_cmd(interaction: discord.Interaction) -> None:
        latency_ms = round(bot.latency * 1000)
        await interaction.response.send_message(f"Pong! Latency: {latency_ms}ms")

    @tree.command(name="help", description="Show a list of available commands.")
    async def help_cmd(interaction: discord.Interaction) -> None:
        embed = discord.Embed(
            title="Bot Commands",
            description="Here is a list of all available commands.",
            color=discord.Color.blurple(),
        )
        embed.add_field(
            name="AI",
            value="`/ask` `/clearmemory` `/setaichannel`",
            inline=False,
        )
        embed.add_field(
            name="Moderation",
            value=(
                "`/kick` `/ban` `/timeout` `/warn` `/warnings` `/clear` "
                "`/purge` `/slowmode` `/lock` `/unlock`"
            ),
            inline=False,
        )
        embed.add_field(
            name="Utility",
            value="`/userinfo` `/serverinfo` `/ping` `/help` `/setlogchannel` `/stats`",
            inline=False,
        )
        await interaction.response.send_message(embed=embed)

    @tree.command(name="stats", description="Show bot and server statistics.")
    async def stats_cmd(interaction: discord.Interaction) -> None:
        guild_id = interaction.guild_id or 0
        uptime = datetime.now(timezone.utc) - bot.start_time
        kicks = await bot.db.get_stat(guild_id, "kicks")
        bans = await bot.db.get_stat(guild_id, "bans")
        timeouts = await bot.db.get_stat(guild_id, "timeouts")
        spam_deleted = await bot.db.get_stat(guild_id, "spam_deleted")
        joins = await bot.db.get_stat(guild_id, "member_joins")
        leaves = await bot.db.get_stat(guild_id, "member_leaves")

        embed = discord.Embed(title="Bot Statistics", color=discord.Color.blurple())
        embed.add_field(name="Uptime", value=str(uptime).split(".")[0], inline=True)
        embed.add_field(name="Guilds", value=str(len(bot.guilds)), inline=True)
        embed.add_field(name="Latency", value=f"{round(bot.latency * 1000)}ms", inline=True)
        embed.add_field(name="Kicks", value=str(kicks), inline=True)
        embed.add_field(name="Bans", value=str(bans), inline=True)
        embed.add_field(name="Timeouts", value=str(timeouts), inline=True)
        embed.add_field(name="Spam Removed", value=str(spam_deleted), inline=True)
        embed.add_field(name="Joins", value=str(joins), inline=True)
        embed.add_field(name="Leaves", value=str(leaves), inline=True)
        await interaction.response.send_message(embed=embed)


# --------------------------------------------------------------------------- #
#                                    ENTRYPOINT                                #
# --------------------------------------------------------------------------- #

def main() -> None:
    """Application entrypoint: validates config and starts the bot."""
    validate_config()

    _colored_log("Loading environment configuration...", Colors.CYAN)
    _colored_log(f"Database path: {Config.DB_PATH}", Colors.CYAN)
    _colored_log(f"Gemini model: {Config.GEMINI_MODEL}", Colors.CYAN)

    if genai is None:
        _colored_log("google-genai package not installed. AI features will be disabled.", Colors.YELLOW)

    bot = ModBot()

    try:
        _colored_log("Starting bot...", Colors.GREEN)
        bot.run(Config.DISCORD_TOKEN, log_handler=None)
    except discord.LoginFailure:
        _colored_log("Invalid Discord token provided.", Colors.RED)
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001
        _colored_log(f"Fatal error while running the bot: {exc}", Colors.RED)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
