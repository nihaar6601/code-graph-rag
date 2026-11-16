package shop.util;

/** Integer cents and integer rounding keep the pricing example deterministic. */
public class PriceCalculator {
    public static int subtotal(int unitPriceCents, int quantity) {
        if (unitPriceCents < 0 || quantity <= 0) {
            return 0;
        }
        return unitPriceCents * quantity;
    }

    public static int discount(int subtotalCents, String coupon) {
        if ("SAVE10".equals(coupon)) {
            return subtotalCents / 10;
        }
        return 0;
    }

    public static int total(int unitPriceCents, int quantity, String coupon) {
        int baseCents = PriceCalculator.subtotal(unitPriceCents, quantity);
        return baseCents - PriceCalculator.discount(baseCents, coupon);
    }
}
