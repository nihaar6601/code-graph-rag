package shop.service;

import java.util.HashMap;
import java.util.Map;

public class ShippingService {
    private final Map<String, String> trackingCodes = new HashMap<>();

    public int quote(String zone) {
        if ("LOCAL".equals(zone)) {
            return 500;
        }
        return 1500;
    }

    public String dispatch(String orderId, String zone) {
        String code = "SHIP-" + zone + "-" + orderId;
        trackingCodes.put(orderId, code);
        return code;
    }

    public String tracking(String orderId) {
        return trackingCodes.getOrDefault(orderId, "NOT_DISPATCHED");
    }
}
