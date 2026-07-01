"""No-op telemetry shim for the vendored Mem0 baseline.

The upstream package reports anonymous usage through PostHog. For this benchmark
repository we disable telemetry entirely so evaluation runs do not contact third-
party analytics services.
"""


class AnonymousTelemetry:
    def __init__(self, vector_store=None):
        self.user_id = None

    def capture_event(self, event_name, properties=None, user_email=None):
        return None

    def close(self):
        return None


client_telemetry = AnonymousTelemetry()


def capture_event(event_name, memory_instance, additional_data=None):
    return None


def capture_client_event(event_name, instance, additional_data=None):
    return None