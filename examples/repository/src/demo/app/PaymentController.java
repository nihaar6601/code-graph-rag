package demo.app;

import demo.service.PaymentService;
import demo.util.Audit;

public class PaymentController {
    private PaymentService service = new PaymentService();

    public void process(PaymentService parameter) {
        service.processPayment();
        PaymentService local = new PaymentService();
        local.processPayment();
        parameter.processPayment();
        Audit.record();
        System.out.println("Requested");
    }
}
