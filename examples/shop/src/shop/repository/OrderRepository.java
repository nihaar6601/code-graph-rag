package shop.repository;

import java.util.HashMap;
import java.util.Map;

public class OrderRepository {
    private final Map<String, String> statuses = new HashMap<>();
    private final Map<String, String> details = new HashMap<>();
    private int nextId = 1;

    public String save(String sku, int quantity, int paidCents) {
        String orderId = "ORD-" + nextId++;
        details.put(orderId, sku + ":" + quantity + ":" + paidCents);
        statuses.put(orderId, "PLACED");
        return orderId;
    }

    public String status(String orderId) {
        return statuses.getOrDefault(orderId, "NOT_FOUND");
    }

    public boolean updateStatus(String orderId, String newStatus) {
        if (!statuses.containsKey(orderId)) {
            return false;
        }
        statuses.put(orderId, newStatus);
        return true;
    }
}
