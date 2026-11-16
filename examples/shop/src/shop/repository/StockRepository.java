package shop.repository;

import java.util.HashMap;
import java.util.Map;

/** Each previously unseen SKU starts with twenty demonstration units. */
public class StockRepository {
    private final Map<String, Integer> quantities = new HashMap<>();

    public int quantity(String sku) {
        return quantities.getOrDefault(sku, 20);
    }

    public boolean reduce(String sku, int units) {
        int current = this.quantity(sku);
        if (units <= 0 || units > current) {
            return false;
        }
        quantities.put(sku, current - units);
        return true;
    }

    public void restore(String sku, int units) {
        if (units > 0) {
            int current = this.quantity(sku);
            quantities.put(sku, current + units);
        }
    }
}
