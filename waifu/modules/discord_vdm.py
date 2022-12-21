'''
This module generates dialogue data from Discord dumps. Specifically, it:

- Looks for a DHT database in `/data/discord/archive.dht` to parse
- Builds a list of senders who meet certain criteria (enough messages sent,
  messages long enough), then
- Attempts to find uninterruped conversations between them and another person in
  public channels.

Since a DHT database necessarily contains personal information, this module must
be manually enabled and populated with your own data.
'''
import logging
import os
import re
import sqlite3
import typing as t

from waifu.modules import BaseModule
from waifu.utils.dataset import get_data_path

# Matches user mentions, channel links, emotes and maybe other stuff.
SPECIAL_TOKENS_REGEX = re.compile(r"<[@:#].+?>")

logger = logging.getLogger(__name__)


class DiscordVDM(BaseModule):
    '''A Vanilla Dialogue Module powered by Discord dumps.'''

    def generator(self) -> t.Generator[str, None, None]:
        root_data_path = get_data_path("discord")
        db_path = os.path.join(root_data_path, "archive.dht")
        db = sqlite3.connect(db_path)
        db.row_factory = sqlite3.Row
        cursor = db.cursor()

        sender_ids = _get_filtered_sender_ids(cursor)
        for sender_id in sender_ids:
            last_message_id = None
            while (episode_contents := _build_episode_turns(
                    db, sender_id,
                    start_after_message_id=last_message_id)) is not None:
                turns, last_message_id = episode_contents

                # Discard short episodes.
                if len(turns) < 8:
                    logger.debug("Found short %s-turn episode, discarding.",
                                 len(turns))
                    continue

                # Discard conversations with overly short messages.
                lengths = [len(x) for x in turns]
                avg = sum(lengths) / len(lengths)
                if avg < 64:
                    logger.debug(
                        "Found conversation where average message length was %s, discarding.",
                        avg)
                    continue

                yield "\n".join(turns)


#
# Private helpers.
#
def _clean_string(string: str) -> str:
    '''Removes user mentions, channel links and so on.'''
    return re.sub(SPECIAL_TOKENS_REGEX, "", string).strip()


def _looks_like_ooc(raw_string: str) -> bool:
    '''Tries to figure out whether a message looks like it's out of character.'''
    string = raw_string.strip()

    if string[0] == "(" and string[-1] == ")":
        return True

    if "OOC:" in string:
        return True

    return False


def _get_filtered_sender_ids(cursor: sqlite3.Cursor) -> list[int]:
    '''Gets a list of sender_ids that meet the filtering criteria.'''
    res = cursor.execute('''
        SELECT
            sender_id
        FROM (
            SELECT
                "sender_id",
                AVG(LENGTH("text")) AS average_message_length,
                COUNT("sender_id") AS messages_sent
            FROM
                "messages"
            GROUP BY
                "sender_id"
            ORDER BY
                "average_message_length" DESC
        )
        WHERE
            "messages_sent" > 8 AND "average_message_length" >= 32;
        ''').fetchall()

    return [x[0] for x in res]


def _build_episode_turns(
        db: sqlite3.Connection,
        sender_id: int,
        start_after_message_id: int | None = None
) -> tuple[list[str], int] | None:
    logger.debug("Building episode for sender_id %s, starting after message %s",
                 sender_id, start_after_message_id)

    # Fetch the first message for the episode.
    if start_after_message_id:
        query = """
            SELECT
                message_id, channel_id
            FROM
                messages
            WHERE
                sender_id = :sender_id AND message_id > :message_id;
            """
    else:
        query = """
            SELECT
                message_id, channel_id
            FROM
                messages
            WHERE
                sender_id = :sender_id LIMIT 1;
            """

    cursor = db.cursor()
    res = cursor.execute(query, {
        "sender_id": sender_id,
        "message_id": start_after_message_id,
    }).fetchone()

    if res is None:
        logger.debug("No more suitable first messages found.")
        return None

    message_id, channel_id = res["message_id"], res["channel_id"]
    logger.debug("Found suitable first message %s by %s.", message_id,
                 sender_id)

    # From there, fetch that specific channel's log from that point on.
    query = """
        SELECT
            *
        FROM
            messages
        WHERE
            channel_id = :channel_id
            AND
            message_id >= :message_id
        ;
    """
    res = cursor.execute(query, {
        "channel_id": channel_id,
        "message_id": message_id,
    })

    person_a_id = sender_id
    person_b_id = None
    last_message_id = -1
    turns: list[str] = []

    while (row := res.fetchone()) is not None:
        last_message_id = row["message_id"]

        # Save who `sender_id` is talking to.
        if person_b_id is None and row["sender_id"] != person_a_id:
            person_b_id = row["sender_id"]

        # Somebody else came into the conversation. Stop episode here.
        if person_b_id and row["sender_id"] not in (person_a_id, person_b_id):
            logger.debug(
                "%s barged into a conversation between %s and %s, assuming end of episode.",
                row["sender_id"],
                person_a_id,
                person_b_id,
            )
            break

        cleaned_text = _clean_string(row["text"])
        if not cleaned_text:
            # Message was empty after cleaning it up, skip.
            continue

        if _looks_like_ooc(cleaned_text):
            # Self-explanatory.
            continue

        # Get username.
        # TODO(11b): Anonymize.
        username_query = "SELECT name FROM users WHERE id = :user_id"
        username = db.cursor().execute(username_query, {
            "user_id": row["sender_id"]
        }).fetchone()["name"]

        # Build up the string and add it to the episode.
        turn_string = f"{username}: {cleaned_text}"
        turns.append(turn_string)

    if len(turns) == 0:
        logger.debug(
            "Empty episode, assuming no more conversations from this sender.")
        return None

    return turns, last_message_id
