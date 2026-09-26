"""Pub/Sub-backed :class:`EventBus` for progress, audit and alert fan-out.

One publisher, many consumers: a UI streaming progress, a BigQuery sink for
cost analytics, an alerting subscription on ``agent-alerts``. Ordering keys
keep one run's events in order when a consumer needs that.
"""

from __future__ import annotations

import json
import logging
from typing import Any

log = logging.getLogger("lra.gcp.pubsub")


class PubSubEventBus:
    def __init__(self, *, project: str, client: Any | None = None, ordered: bool = True) -> None:
        from google.cloud import pubsub_v1  # lazy import

        if client is None:
            settings = pubsub_v1.types.PublisherOptions(enable_message_ordering=ordered)
            client = pubsub_v1.PublisherClient(publisher_options=settings)
        self.client = client
        self.project = project
        self.ordered = ordered

    def publish(self, topic: str, payload: dict[str, Any], attributes: dict[str, str] | None = None) -> str:
        topic_path = self.client.topic_path(self.project, topic)
        data = json.dumps(payload, default=str).encode("utf-8")
        attrs = dict(attributes or {})
        kwargs: dict[str, Any] = {}
        if self.ordered and "run_id" in attrs:
            kwargs["ordering_key"] = attrs["run_id"]
        future = self.client.publish(topic_path, data, **attrs, **kwargs)
        return future.result(timeout=10)
