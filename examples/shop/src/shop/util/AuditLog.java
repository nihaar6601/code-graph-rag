package shop.util;

import java.util.ArrayList;
import java.util.List;

/** Per-instance activity history; nothing is persisted or sent externally. */
public class AuditLog {
    private final List<String> events = new ArrayList<>();

    public void record(String event) {
        if (event != null && !event.isEmpty()) {
            events.add(event);
        }
    }

    public int count() {
        return events.size();
    }

    public String lastEvent() {
        if (events.isEmpty()) {
            return "NO_EVENTS";
        }
        return events.get(events.size() - 1);
    }
}
