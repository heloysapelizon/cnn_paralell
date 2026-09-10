#!/usr/bin/env python3
"""
Checagem de gradiente por diferencas finitas para a mesma arquitetura
LeNet-like usada em network.cpp/layers.cpp:

    Conv(1->6,5x5) -> ReLU -> MaxPool2x2 -> Conv(6->16,5x5) -> ReLU ->
    MaxPool2x2 -> FC(256->120) -> ReLU -> FC(120->84) -> ReLU ->
    FC(84->10) -> Softmax+CrossEntropy

Reimplementacao independente em NumPy: valida a matematica do
forward/backward (a mesma que o C++ implementa a mao), nao o binario
C++ em si.
"""
import numpy as np

rng = np.random.default_rng(42)


def conv_init(in_c, out_c, kh, kw, seed):
    r = np.random.default_rng(seed)
    fan_in = in_c * kh * kw
    std = np.sqrt(2.0 / fan_in)
    W = r.normal(0.0, std, size=(out_c, in_c, kh, kw)).astype(np.float64)
    b = np.zeros(out_c)
    return {"W": W, "b": b}


def dense_init(in_f, out_f, seed):
    r = np.random.default_rng(seed)
    std = np.sqrt(2.0 / in_f)
    W = r.normal(0.0, std, size=(out_f, in_f)).astype(np.float64)
    b = np.zeros(out_f)
    return {"W": W, "b": b}


def conv_forward(p, x):
    """x: [inC,H,W] -> y: [outC,outH,outW]"""
    in_c, h, w = x.shape
    out_c, _, kh, kw = p["W"].shape
    out_h, out_w = h - kh + 1, w - kw + 1
    y = np.empty((out_c, out_h, out_w))
    for oc in range(out_c):
        acc = np.full((out_h, out_w), p["b"][oc])
        for ic in range(in_c):
            for i in range(kh):
                for j in range(kw):
                    acc += p["W"][oc, ic, i, j] * x[ic, i:i + out_h, j:j + out_w]
        y[oc] = acc
    return y


def conv_backward(p, x, grad_out, dW, db):
    in_c, h, w = x.shape
    out_c, _, kh, kw = p["W"].shape
    out_h, out_w = grad_out.shape[1], grad_out.shape[2]
    grad_in = np.zeros_like(x)
    for oc in range(out_c):
        db[oc] += grad_out[oc].sum()
        for ic in range(in_c):
            for i in range(kh):
                for j in range(kw):
                    patch = x[ic, i:i + out_h, j:j + out_w]
                    dW[oc, ic, i, j] += np.sum(grad_out[oc] * patch)
                    grad_in[ic, i:i + out_h, j:j + out_w] += grad_out[oc] * p["W"][oc, ic, i, j]
    return grad_in


def relu_forward(x):
    return np.maximum(x, 0.0)


def relu_backward(x, grad_out):
    return grad_out * (x > 0.0)


def maxpool2x2_forward(x):
    c, h, w = x.shape
    out_h, out_w = h // 2, w // 2
    y = np.empty((c, out_h, out_w))
    argmax = np.empty((c, out_h, out_w), dtype=np.int64)
    for ch in range(c):
        for oh in range(out_h):
            for ow in range(out_w):
                window = x[ch, oh * 2:oh * 2 + 2, ow * 2:ow * 2 + 2].reshape(-1)
                k = int(np.argmax(window))
                y[ch, oh, ow] = window[k]
                argmax[ch, oh, ow] = k
    return y, argmax


def maxpool2x2_backward(argmax, grad_out, shape):
    c, h, w = shape
    out_h, out_w = grad_out.shape[1], grad_out.shape[2]
    grad_in = np.zeros(shape)
    for ch in range(c):
        for oh in range(out_h):
            for ow in range(out_w):
                k = argmax[ch, oh, ow]
                dh, dw = k // 2, k % 2
                grad_in[ch, oh * 2 + dh, ow * 2 + dw] += grad_out[ch, oh, ow]
    return grad_in


def dense_forward(p, x):
    return p["W"] @ x + p["b"]


def dense_backward(p, x, grad_out, dW, db):
    dW += np.outer(grad_out, x)
    db += grad_out
    return p["W"].T @ grad_out


def softmax_cross_entropy(logits, label):
    m = np.max(logits)
    exps = np.exp(logits - m)
    s = exps.sum()
    probs = exps / s
    grad = probs.copy()
    grad[label] -= 1.0
    loss = -np.log(max(exps[label] / s, 1e-12))
    return loss, grad


def lenet_init(seed):
    return {
        "conv1": conv_init(1, 6, 5, 5, seed + 1),
        "conv2": conv_init(6, 16, 5, 5, seed + 2),
        "fc1": dense_init(16 * 4 * 4, 120, seed + 3),
        "fc2": dense_init(120, 84, seed + 4),
        "fc3": dense_init(84, 10, seed + 5),
    }


def zero_grad(params):
    return {name: {"W": np.zeros_like(p["W"]), "b": np.zeros_like(p["b"])}
            for name, p in params.items()}


def forward_backward(params, image, label, grad):
    x0 = image.reshape(1, 28, 28)

    c1 = conv_forward(params["conv1"], x0)
    r1 = relu_forward(c1)
    p1, am1 = maxpool2x2_forward(r1)

    c2 = conv_forward(params["conv2"], p1)
    r2 = relu_forward(c2)
    p2, am2 = maxpool2x2_forward(r2)

    flat = p2.reshape(-1)
    fc1_out = dense_forward(params["fc1"], flat)
    r3 = relu_forward(fc1_out)

    fc2_out = dense_forward(params["fc2"], r3)
    r4 = relu_forward(fc2_out)

    fc3_out = dense_forward(params["fc3"], r4)
    predicted = int(np.argmax(fc3_out))

    loss, grad_logits = softmax_cross_entropy(fc3_out, label)

    grad_r4 = dense_backward(params["fc3"], r4, grad_logits, grad["fc3"]["W"], grad["fc3"]["b"])
    grad_fc2out = relu_backward(fc2_out, grad_r4)
    grad_r3 = dense_backward(params["fc2"], r3, grad_fc2out, grad["fc2"]["W"], grad["fc2"]["b"])
    grad_fc1out = relu_backward(fc1_out, grad_r3)
    grad_flat = dense_backward(params["fc1"], flat, grad_fc1out, grad["fc1"]["W"], grad["fc1"]["b"])

    grad_p2 = grad_flat.reshape(p2.shape)
    grad_r2 = maxpool2x2_backward(am2, grad_p2, r2.shape)
    grad_c2out = relu_backward(c2, grad_r2)
    grad_p1 = conv_backward(params["conv2"], p1, grad_c2out, grad["conv2"]["W"], grad["conv2"]["b"])

    grad_r1 = maxpool2x2_backward(am1, grad_p1, r1.shape)
    grad_c1out = relu_backward(c1, grad_r1)
    conv_backward(params["conv1"], x0, grad_c1out, grad["conv1"]["W"], grad["conv1"]["b"])

    return loss, predicted


def run_loss(params, image, label):
    grad = zero_grad(params)
    loss, _ = forward_backward(params, image, label, grad)
    return loss


def numeric_grad(params, layer, field, idx, image, label, eps):
    w = params[layer][field]
    orig = w[idx]
    w[idx] = orig + eps
    lp = run_loss(params, image, label)
    w[idx] = orig - eps
    lm = run_loss(params, image, label)
    w[idx] = orig
    return (lp - lm) / (2.0 * eps)


def check_layer(name, field, params, analytic, image, label, n_samples, eps=1e-4, tol=0.05):
    w = params[name][field]
    failures = 0
    flat_size = w.size
    for _ in range(n_samples):
        flat_idx = rng.integers(0, flat_size)
        idx = np.unravel_index(flat_idx, w.shape)
        num = numeric_grad(params, name, field, idx, image, label, eps)
        ana = analytic[name][field][idx]
        diff = abs(num - ana)
        denom = max(1e-4, abs(num) + abs(ana))
        rel = diff / denom
        ok = rel < tol
        print(f"  {name}.{field:<3} idx={str(idx):<18} analitico={ana: .6f} "
              f"numerico={num: .6f} rel_err={rel:.4f}  {'OK' if ok else 'FALHOU'}")
        if not ok:
            failures += 1
    return failures


def main():
    params = lenet_init(123)
    image = rng.uniform(0.0, 1.0, size=28 * 28)
    label = 3

    grad = zero_grad(params)
    loss, predicted = forward_backward(params, image, label, grad)
    print(f"loss inicial = {loss:.4f}, predicted = {predicted}\n")

    plan = [
        ("conv1", "W", 5), ("conv1", "b", 3),
        ("conv2", "W", 5), ("conv2", "b", 3),
        ("fc1", "W", 5), ("fc1", "b", 3),
        ("fc2", "W", 5), ("fc2", "b", 3),
        ("fc3", "W", 5), ("fc3", "b", 3),
    ]

    failures = 0
    total = 0
    for layer, field, n in plan:
        failures += check_layer(layer, field, params, grad, image, label, n)
        total += n

    status = "GRADIENTE OK" if failures == 0 else "GRADIENTE COM PROBLEMAS"
    print(f"\n{status} ({failures} falha(s) em {total} checagens)")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
