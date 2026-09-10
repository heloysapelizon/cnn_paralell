#include "network.hpp"

LeNetParams lenet_init(unsigned seed) {
    LeNetParams p;
    p.conv1 = conv_init(1, 6, 5, 5, seed + 1);
    p.conv2 = conv_init(6, 16, 5, 5, seed + 2);
    p.fc1 = dense_init(16 * 4 * 4, 120, seed + 3);
    p.fc2 = dense_init(120, 84, seed + 4);
    p.fc3 = dense_init(84, 10, seed + 5);
    return p;
}

float lenet_forward_backward(const LeNetParams &p, const Vecf &image, int label,
                              LeNetGrad &grad, int &predicted) {
    /* ---------------- forward ---------------- */
    int h1, w1;
    Vecf c1_out; ConvCache c1_cache;
    conv_forward(p.conv1, image, 28, 28, c1_out, h1, w1, c1_cache);

    Vecf r1_out;
    relu_forward(c1_out, r1_out);

    int ph1, pw1;
    Vecf pool1_out; PoolCache pool1_cache;
    maxpool2x2_forward(r1_out, p.conv1.outC, h1, w1, pool1_out, ph1, pw1, pool1_cache);

    int h2, w2;
    Vecf c2_out; ConvCache c2_cache;
    conv_forward(p.conv2, pool1_out, ph1, pw1, c2_out, h2, w2, c2_cache);

    Vecf r2_out;
    relu_forward(c2_out, r2_out);

    int ph2, pw2;
    Vecf pool2_out; PoolCache pool2_cache;
    maxpool2x2_forward(r2_out, p.conv2.outC, h2, w2, pool2_out, ph2, pw2, pool2_cache);

    /* pool2_out ja esta "flat" (vector contiguo) -> entra direto na FC1 */
    Vecf fc1_out; DenseCache fc1_cache;
    dense_forward(p.fc1, pool2_out, fc1_out, fc1_cache);

    Vecf r3_out;
    relu_forward(fc1_out, r3_out);

    Vecf fc2_out; DenseCache fc2_cache;
    dense_forward(p.fc2, r3_out, fc2_out, fc2_cache);

    Vecf r4_out;
    relu_forward(fc2_out, r4_out);

    Vecf fc3_out; DenseCache fc3_cache;
    dense_forward(p.fc3, r4_out, fc3_out, fc3_cache);

    predicted = argmax(fc3_out);

    Vecf grad_logits;
    float loss = softmax_cross_entropy(fc3_out, label, grad_logits);

    /* ---------------- backward ---------------- */
    Vecf grad_r4;
    dense_backward(p.fc3, fc3_cache, grad_logits, grad_r4, grad.fc3);

    Vecf grad_fc2out;
    relu_backward(fc2_out, grad_r4, grad_fc2out);

    Vecf grad_r3;
    dense_backward(p.fc2, fc2_cache, grad_fc2out, grad_r3, grad.fc2);

    Vecf grad_fc1out;
    relu_backward(fc1_out, grad_r3, grad_fc1out);

    Vecf grad_pool2;
    dense_backward(p.fc1, fc1_cache, grad_fc1out, grad_pool2, grad.fc1);

    Vecf grad_r2;
    maxpool2x2_backward(pool2_cache, grad_pool2, p.conv2.outC, h2, w2, ph2, pw2, grad_r2);

    Vecf grad_c2out;
    relu_backward(c2_out, grad_r2, grad_c2out);

    Vecf grad_pool1;
    conv_backward(p.conv2, c2_cache, grad_c2out, h2, w2, grad_pool1, grad.conv2);

    Vecf grad_r1;
    maxpool2x2_backward(pool1_cache, grad_pool1, p.conv1.outC, h1, w1, ph1, pw1, grad_r1);

    Vecf grad_c1out;
    relu_backward(c1_out, grad_r1, grad_c1out);

    Vecf grad_image; /* descartado -- nao propagamos alem da entrada */
    conv_backward(p.conv1, c1_cache, grad_c1out, h1, w1, grad_image, grad.conv1);

    return loss;
}

int lenet_predict(const LeNetParams &p, const Vecf &image) {
    int h1, w1;
    Vecf c1_out; ConvCache c1_cache;
    conv_forward(p.conv1, image, 28, 28, c1_out, h1, w1, c1_cache);
    Vecf r1_out; relu_forward(c1_out, r1_out);
    int ph1, pw1; Vecf pool1_out; PoolCache pool1_cache;
    maxpool2x2_forward(r1_out, p.conv1.outC, h1, w1, pool1_out, ph1, pw1, pool1_cache);

    int h2, w2;
    Vecf c2_out; ConvCache c2_cache;
    conv_forward(p.conv2, pool1_out, ph1, pw1, c2_out, h2, w2, c2_cache);
    Vecf r2_out; relu_forward(c2_out, r2_out);
    int ph2, pw2; Vecf pool2_out; PoolCache pool2_cache;
    maxpool2x2_forward(r2_out, p.conv2.outC, h2, w2, pool2_out, ph2, pw2, pool2_cache);

    Vecf fc1_out; DenseCache fc1_cache;
    dense_forward(p.fc1, pool2_out, fc1_out, fc1_cache);
    Vecf r3_out; relu_forward(fc1_out, r3_out);

    Vecf fc2_out; DenseCache fc2_cache;
    dense_forward(p.fc2, r3_out, fc2_out, fc2_cache);
    Vecf r4_out; relu_forward(fc2_out, r4_out);

    Vecf fc3_out; DenseCache fc3_cache;
    dense_forward(p.fc3, r4_out, fc3_out, fc3_cache);

    return argmax(fc3_out);
}

static void sgd_vec(Vecf &w, const Vecf &g, float lr) {
    for (size_t i = 0; i < w.size(); i++) w[i] -= lr * g[i];
}

void lenet_sgd_update(LeNetParams &p, const LeNetGrad &grad, float lr) {
    sgd_vec(p.conv1.W, grad.conv1.dW, lr);
    sgd_vec(p.conv1.b, grad.conv1.db, lr);
    sgd_vec(p.conv2.W, grad.conv2.dW, lr);
    sgd_vec(p.conv2.b, grad.conv2.db, lr);
    sgd_vec(p.fc1.W, grad.fc1.dW, lr);
    sgd_vec(p.fc1.b, grad.fc1.db, lr);
    sgd_vec(p.fc2.W, grad.fc2.dW, lr);
    sgd_vec(p.fc2.b, grad.fc2.db, lr);
    sgd_vec(p.fc3.W, grad.fc3.dW, lr);
    sgd_vec(p.fc3.b, grad.fc3.db, lr);
}
