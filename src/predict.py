def sample(model, dataset, device, size, prime, top_k=5):
    model.eval()
    model.to(device)
    chars = [ch for ch in prime]
    previous_hidden_states = model.init_hidden(1)
    for ch in prime:
        char, previous_hidden_states = model.predict(
            ch, dataset, device, previous_hidden_states, top_k=top_k
        )
    chars.append(char)
    for _ in range(size):
        char, previous_hidden_states = model.predict(
            chars[-1], dataset, device, previous_hidden_states, top_k=top_k
        )
        chars.append(char)
    return "".join(chars)
