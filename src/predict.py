def sample(model, dataset, device, size, prime):
    model.to(device)
    model.eval()
    chars = [ch for ch in prime]
    previous_hidden_states = model.init_hidden(1)
    for ch in prime:
        char, previous_hidden_states = model.predict(
            ch, dataset, device, previous_hidden_states
        )
    chars.append(char)
    for ii in range(size):
        char, previous_hidden_states = model.predict(
            chars[-1], dataset, device, previous_hidden_states
        )
        chars.append(char)
    return "".join(chars)

