package shop.service;

/** Simulated aggregate ledger; no actual payment provider is contacted. */
public class PaymentService {
    private int chargedCents = 0;

    public boolean charge(int amountCents) {
        if (amountCents <= 0 || amountCents > 100000) {
            return false;
        }
        chargedCents += amountCents;
        return true;
    }

    public boolean refund(int amountCents) {
        if (amountCents <= 0 || amountCents > chargedCents) {
            return false;
        }
        chargedCents -= amountCents;
        return true;
    }

    public int totalCharged() {
        return chargedCents;
    }
}
