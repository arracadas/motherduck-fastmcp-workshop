"""Your MCP server for San Francisco 311 data.

FastMCP Gateway runs this file. Its entrypoint setting is `server.py:mcp`.
"""

from collections import Counter, defaultdict
from statistics import median

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from sf311.data import BY_ID, LOCAL_NAMES, NEIGHBORHOODS, REQUESTS, SERVICES, closest

mcp = FastMCP("sf-311")

MAX_ROWS = 50
STATUSES = ("Open", "Closed")
COUNT_GROUPS = ("neighborhood", "service", "agency", "source", "weekday", "status")
MEDIAN_GROUPS = ("neighborhood", "agency")
MIN_CLOSED_FOR_MEDIAN = 10

GROUP_KEYS = {
    "neighborhood": lambda r: r.neighborhood,
    "service": lambda r: r.service,
    "agency": lambda r: r.agency,
    "source": lambda r: r.source,
    "weekday": lambda r: r.opened.strftime("%A"),
    "status": lambda r: r.status,
}


def _check_name(value: str | None, kind: str, choices: list[str]) -> None:
    if value is not None and value not in choices:
        suggestions = closest(value, choices)
        hint = (
            f" Did you mean: {', '.join(suggestions)}?"
            if suggestions
            else f" Call dataset_info() for the valid {kind} names."
        )
        raise ToolError(f'Unknown {kind} "{value}".{hint}')


def _check_group_by(group_by: str | None, allowed: tuple[str, ...]) -> None:
    if group_by is not None and group_by not in allowed:
        raise ToolError(f'Unknown group_by "{group_by}". Use one of: {", ".join(allowed)}.')


def _matching(service: str | None, neighborhood: str | None, status: str | None) -> list:
    return [
        r
        for r in REQUESTS
        if (service is None or r.service == service)
        and (neighborhood is None or r.neighborhood == neighborhood)
        and (status is None or r.status == status)
    ]


@mcp.tool(annotations={"readOnlyHint": True})
def dataset_info() -> dict:
    """Describe the 311 dataset this server answers questions about.

    Call this first to learn the date range and the exact service and
    neighborhood names that other tools accept.
    """
    return {
        "description": "San Francisco 311 service requests filed in one week.",
        "first_opened": min(request.opened for request in REQUESTS).isoformat(),
        "last_opened": max(request.opened for request in REQUESTS).isoformat(),
        "requests": len(REQUESTS),
        "services": SERVICES,
        "neighborhoods": NEIGHBORHOODS,
        "local_neighborhood_names": len(LOCAL_NAMES),
    }


@mcp.tool(annotations={"readOnlyHint": True})
def search_requests(
    service: str | None = None,
    neighborhood: str | None = None,
    status: str | None = None,
    limit: int = 20,
) -> list[dict]:
    """List individual 311 requests matching the filters, newest first.

    Each row is a dict with every field of sf311.data.Request (request_id,
    opened, closed, status, service, subtype, agency, neighborhood,
    local_name, address, source). All filters are optional and combine with
    AND; `status` is "Open" or "Closed"; `limit` is 1 to 50 (default 20).

    Use this to show real cases or to find request ids for get_request.
    When the question asks "how many" or "which group is biggest", call
    count_requests instead of paging through rows.
    """
    _check_name(service, "service", SERVICES)
    _check_name(neighborhood, "neighborhood", NEIGHBORHOODS)
    if status is not None and status not in STATUSES:
        raise ToolError(f'Unknown status "{status}". Use "Open" or "Closed".')
    if not 1 <= limit <= MAX_ROWS:
        raise ToolError(f"limit must be between 1 and {MAX_ROWS}, got {limit}.")
    rows = _matching(service, neighborhood, status)
    rows.sort(key=lambda r: (r.opened, r.request_id), reverse=True)
    return [r.as_dict() for r in rows[:limit]]


@mcp.tool(annotations={"readOnlyHint": True})
def count_requests(
    service: str | None = None,
    neighborhood: str | None = None,
    group_by: str | None = None,
) -> int | dict[str, int]:
    """Count 311 requests matching the filters.

    Without `group_by`, returns a single number. With `group_by` (one of
    neighborhood, service, agency, source, weekday, status), returns a dict
    of group to count, largest first, at most 50 groups. `service` and
    `neighborhood` use the exact names from dataset_info.

    Use this for "how many" questions. For individual rows, use
    search_requests; for close times, use median_close_hours.
    """
    _check_name(service, "service", SERVICES)
    _check_name(neighborhood, "neighborhood", NEIGHBORHOODS)
    _check_group_by(group_by, COUNT_GROUPS)
    rows = _matching(service, neighborhood, None)
    if group_by is None:
        return len(rows)
    key = GROUP_KEYS[group_by]
    counts = Counter(key(r) for r in rows if key(r))
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return dict(ranked[:MAX_ROWS])


@mcp.tool(annotations={"readOnlyHint": True})
def median_close_hours(
    service: str | None = None,
    group_by: str | None = None,
) -> float | None | dict[str, float]:
    """Median hours from opened to closed for matching requests, rounded to 0.1.

    Open requests (not yet closed) are excluded; returns None when nothing is
    closed. With `group_by` ("neighborhood" or "agency"), returns a dict of
    group to median hours, fastest first, leaving out groups with fewer than
    10 closed requests, at most 50 groups.

    Use this for "how fast does X get fixed" questions. For counts, use
    count_requests; it also shows which groups have enough closed requests.
    """
    _check_name(service, "service", SERVICES)
    _check_group_by(group_by, MEDIAN_GROUPS)
    rows = [r for r in _matching(service, None, None) if r.hours_to_close is not None]
    if group_by is None:
        return round(median(r.hours_to_close for r in rows), 1) if rows else None
    key = GROUP_KEYS[group_by]
    hours: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if key(r):
            hours[key(r)].append(r.hours_to_close)
    medians = {group: round(median(values), 1) for group, values in hours.items() if len(values) >= MIN_CLOSED_FOR_MEDIAN}
    return dict(sorted(medians.items(), key=lambda item: (item[1], item[0]))[:MAX_ROWS])


@mcp.tool(annotations={"readOnlyHint": True})
def get_request(request_id: str) -> dict:
    """Return one 311 request by its id as a dict of all its fields.

    Use search_requests first to find ids that exist. An unknown id raises
    an error listing no suggestions, because ids are opaque.
    """
    request = BY_ID.get(request_id)
    if request is None:
        raise ToolError(f'No request with id "{request_id}". Find real ids with search_requests.')
    return request.as_dict()


if __name__ == "__main__":
    mcp.run()
