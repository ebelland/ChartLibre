"""Rewriting the SELECT list of a series' SQL.

A series is a SQL query, and several operations change what it selects
without touching the rest: Clustering and Outliers add a ``"color"``
projection so the scatter renderer colours the points from a column. The
work is text surgery on the SELECT list - split it on top-level commas,
find a projection by its alias, replace it or append one - and has to
respect quotes, brackets and parentheses, which a plain regex cannot.
Nothing here touches a database.
"""
from __future__ import annotations

import re

from app.logs.logger import applogger


def sql_insert_select_expression(sql_query: str, expression: str) -> str:
    """Insert or replace a SELECT expression by alias.

    Used by clustering to make the selected series expose:

        "ClusterId" AS "color"

    The function is intentionally idempotent. If the SQL already projects a
    column/expression as color, replace that projection instead of appending a
    second color column.

    Supported color aliases:
        AS color
        AS "color"
        AS [color]
        AS `color`

    This avoids duplicate DataFrame columns where df["color"] returns a
    DataFrame instead of a Series.
    """
    sql = str(sql_query).strip().rstrip(";")
    match = top_level_from(sql)
    if match is None:
        applogger.error(
            "Selected series SQL must contain a FROM clause.",
            raise_error=True,
        )
        return sql

    select_part = sql[: match.start()].rstrip()
    from_part = sql[match.start():].lstrip()

    alias = select_alias_from_expression(expression)
    if not alias:
        return f"{select_part}, {expression} {from_part}"

    rewritten_select = replace_select_projection_by_alias(
        select_part,
        alias,
        expression,
    )
    if rewritten_select is not None:
        return f"{rewritten_select} {from_part}"

    return f"{select_part}, {expression} {from_part}"


def top_level_match(sql: str, pattern: str) -> re.Match[str] | None:
    """The first match of *pattern* in the outer query: not inside parentheses or quotes.

    A keyword inside a subquery or a CTE body ("WITH source AS (SELECT ...
    WHERE ...)") belongs to that inner query; reading it as the outer one's
    is how a Hide filter came to be appended as "FROM source AND ...".
    """
    depth = 0
    quote = ""
    scanner = re.compile(rf"""[()'"`\[\]]|{pattern}""", flags=re.IGNORECASE)
    for match in scanner.finditer(sql):
        token = match.group(0)
        if quote:
            if token == quote:
                quote = ""
            continue
        if token in ("'", '"', "`"):
            quote = token
        elif token == "[":
            quote = "]"
        elif token == "(":
            depth += 1
        elif token == ")":
            depth = max(0, depth - 1)
        elif depth == 0:
            return match
    return None


def top_level_from(sql: str) -> re.Match[str] | None:
    """The FROM of the outer query: not one inside a subquery or a string.

    A series whose points are a correlated subquery ("SELECT a.minute AS x,
    (SELECT COUNT(*) FROM events b WHERE ...) AS y FROM events a") has a
    FROM inside its select list; splitting there put the new expression in
    the subquery.
    """
    return top_level_match(sql, r"\bFROM\b")


def select_alias_from_expression(expression: str) -> str:
    """Extract the alias name from an expression ending with AS alias."""
    match = re.search(
        r'\bAS\s+(?:"([^"]+)"|\[([^\]]+)\]|`([^`]+)`|([A-Za-z_][A-Za-z0-9_]*))\s*$',
        str(expression).strip(),
        flags=re.IGNORECASE,
    )
    if match is None:
        return ""
    for group in match.groups():
        if group:
            return str(group)
    return ""


def replace_select_projection_by_alias(
    select_part: str,
    alias: str,
    replacement_expression: str,
) -> str | None:
    """Replace the first top-level SELECT projection whose alias matches.

    This avoids regex-only parsing of comma-separated SELECT lists, so common
    expressions containing commas, functions, quoted names, or CAST(...) are
    handled without producing duplicate aliases.
    """
    text = str(select_part).rstrip()
    match = re.match(r"(?is)^\s*SELECT\s+", text)
    if match is None:
        return None

    prefix = text[: match.end()]
    body = text[match.end():]
    projections = split_top_level_select_items(body)
    changed = False

    for index, projection in enumerate(projections):
        existing_alias = projection_alias(projection)
        if existing_alias.lower() == str(alias).lower():
            projections[index] = replacement_expression
            changed = True
            break

    if not changed:
        return None

    return prefix + ", ".join(projections)


def split_top_level_select_items(select_body: str) -> list[str]:
    """Split SELECT body on top-level commas only."""
    items: list[str] = []
    start = 0
    depth = 0
    quote: str | None = None
    bracket_quote = False
    text = str(select_body)
    index = 0

    while index < len(text):
        char = text[index]

        if quote is not None:
            if bracket_quote:
                if char == "]":
                    quote = None
                    bracket_quote = False
            elif char == quote:
                if index + 1 < len(text) and text[index + 1] == quote:
                    index += 1
                else:
                    quote = None
            index += 1
            continue

        if char in {'"', "'", "`"}:
            quote = char
            bracket_quote = False
            index += 1
            continue

        if char == "[":
            quote = "]"
            bracket_quote = True
            index += 1
            continue

        if char == "(":
            depth += 1
        elif char == ")" and depth > 0:
            depth -= 1
        elif char == "," and depth == 0:
            items.append(text[start:index].strip())
            start = index + 1

        index += 1

    tail = text[start:].strip()
    if tail:
        items.append(tail)
    return items


def projection_alias(projection: str) -> str:
    """Return the explicit AS alias for one SELECT projection, if present."""
    match = re.search(
        r'\bAS\s+(?:"([^"]+)"|\[([^\]]+)\]|`([^`]+)`|([A-Za-z_][A-Za-z0-9_]*))\s*$',
        str(projection).strip(),
        flags=re.IGNORECASE,
    )
    if match is None:
        return ""
    for group in match.groups():
        if group:
            return str(group)
    return ""



def has_projection_alias(sql_query: str, alias: str) -> bool:
    """True when the SELECT list of *sql_query* already projects something AS *alias*."""
    sql = str(sql_query).strip().rstrip(";")
    from_match = re.search(r"\bFROM\b", sql, flags=re.IGNORECASE)
    select_part = sql[: from_match.start()] if from_match else sql
    match = re.match(r"(?is)^\s*SELECT\s+", select_part)
    if match is None:
        return False
    return any(
        projection_alias(projection).lower() == str(alias).lower()
        for projection in split_top_level_select_items(select_part[match.end():])
    )


def projection_source(sql_query: str, alias: str) -> str:
    """The expression *sql_query* projects AS *alias*, unquoted when it is a plain column.

    ``SELECT species AS x FROM penguins`` -> "species" for alias "x": what a
    report names a column by, where the series only knows its role alias.
    Empty when the alias is not projected at the top level.
    """
    sql = str(sql_query).strip().rstrip(";")
    from_match = top_level_from(sql)
    select_part = sql[: from_match.start()] if from_match else sql
    match = re.match(r"(?is)^\s*SELECT\s+(?:DISTINCT\s+)?", select_part)
    if match is None:
        return ""
    for projection in split_top_level_select_items(select_part[match.end():]):
        if projection_alias(projection).lower() != str(alias).lower():
            continue
        expression = re.sub(r"(?is)\s+AS\s+\S+\s*$", "", projection).strip()
        plain = re.fullmatch(r'"([^"]+)"|\[([^\]]+)\]|`([^`]+)`|([A-Za-z_][A-Za-z0-9_.]*)', expression)
        if plain is not None:
            return next(group for group in plain.groups() if group)
        return expression
    return ""
