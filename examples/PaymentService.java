class PaymentService {
    PaymentService() {}

    void processPayment() {
        validatePayment();
        chargeCard();
        System.out.println("Payment processed");
    }

    void validatePayment() {}

    void chargeCard() {}
}
