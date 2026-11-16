package shop.controller;

import shop.service.OrderService;

/** Thin entry points; this fixture has no HTTP framework. */
public class OrderController {
    private final OrderService orders = new OrderService();

    public String placeOrder(String sku, int quantity, int unitPriceCents, String coupon, String zone) {
        return orders.checkout(sku, quantity, unitPriceCents, coupon, zone);
    }

    public boolean cancelOrder(String orderId, String sku, int quantity, int paidCents) {
        return orders.cancel(orderId, sku, quantity, paidCents);
    }

    public String orderStatus(String orderId) {
        return orders.status(orderId);
    }
}
