"""Reusable notification-sink conformance checks."""

from __future__ import annotations

from math import isfinite

from celiums_rezero.knowledge.finalization import NotificationSink
from celiums_rezero.knowledge.schemas import NotificationReceipt, PreparedNotification


def check_notification_sink(
    sink: NotificationSink,
    command: PreparedNotification,
    *,
    timeout_seconds: float,
) -> NotificationReceipt:
    """Smoke-check one adapter instance for replay and exact receipt binding.

    Provider-side durability and hard timeout behavior require separate integration
    conformance across adapter/process restart.
    """
    if not isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("notification conformance timeout must be positive")
    if sink.sink_id != command.sink_id:
        raise ValueError("notification sink ID does not match the conformance command")
    first = sink.deliver(command, timeout_seconds=timeout_seconds)
    second = sink.deliver(command, timeout_seconds=timeout_seconds)
    if first != second:
        raise ValueError("notification sink replay did not return the original receipt")
    if not isinstance(first, NotificationReceipt):
        raise TypeError("notification sink returned an untyped receipt")
    if (
        first.tenant != command.tenant
        or first.job_id != command.job_id
        or first.notification_id != command.notification_id
        or first.sink_id != command.sink_id
        or first.command_digest != command.command_digest
    ):
        raise ValueError("notification sink receipt does not match its command")
    return first
