package shop.service;

import shop.repository.StockRepository;

public class InventoryService {
    private final StockRepository stock = new StockRepository();

    public boolean reserve(String sku, int quantity) {
        if (quantity <= 0 || stock.quantity(sku) < quantity) {
            return false;
        }
        return stock.reduce(sku, quantity);
    }

    public void release(String sku, int quantity) {
        if (quantity > 0) {
            stock.restore(sku, quantity);
        }
    }

    public int available(String sku) {
        return stock.quantity(sku);
    }
}
