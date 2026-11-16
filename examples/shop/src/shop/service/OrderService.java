package shop.service;

import shop.repository.OrderRepository;
import shop.util.AuditLog;
import shop.util.PriceCalculator;

/** Coordinates checkout and cancellation for a single in-memory shop session. */
public class OrderService {
    private final InventoryService inventory = new InventoryService();
    private final PaymentService payments = new PaymentService();
    private final ShippingService shipping = new ShippingService();
    private final OrderRepository orders = new OrderRepository();
    private final AuditLog audit = new AuditLog();

    public String checkout(String sku, int quantity, int unitPriceCents, String coupon, String zone) {
        if (unitPriceCents < 0 || !inventory.reserve(sku, quantity)) {
            return "REJECTED";
        }
        int merchandiseCents = PriceCalculator.total(unitPriceCents, quantity, coupon);
        int deliveryCents = shipping.quote(zone);
        int chargeCents = merchandiseCents + deliveryCents;
        if (!payments.charge(chargeCents)) {
            inventory.release(sku, quantity);
            return "PAYMENT_FAILED";
        }
        String orderId = orders.save(sku, quantity, chargeCents);
        shipping.dispatch(orderId, zone);
        audit.record("PLACED " + orderId);
        return orderId;
    }

    public boolean cancel(String orderId, String sku, int quantity, int paidCents) {
        if (!"PLACED".equals(orders.status(orderId))) {
            return false;
        }
        if (!payments.refund(paidCents)) {
            return false;
        }
        inventory.release(sku, quantity);
        orders.updateStatus(orderId, "CANCELLED");
        audit.record("CANCELLED " + orderId);
        return true;
    }

    public String status(String orderId) {
        return orders.status(orderId);
    }
}
